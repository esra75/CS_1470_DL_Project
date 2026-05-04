"""
model/mlp_clock.py

MouseClockMLP: a fully-connected clock for HVG-filtered input

Designed as a drop-in comparison to MouseClockCNN. Matching features:
  - Input contract: (x, model_id, sex_id) --> (disease_pred, age_pred)
  - Model embedding: same Embedding(n_models, embed_dim)
  - Sex embedding: optional Embedding(2, sex_embed_dim), same pattern as CNN
  - Output heads: same disease sigmoid + age regression
  - Loss: symmetric MAE for both WT and AD + auxiliary age MSE
  - Interface: same count_parameters(), disease_only(), embedding_distance_matrix()

Different from CNN:
  - Encoder: 3 dense BN --> ReLU --> Dropout blocks instead of conv blocks
  - No spatial inductive bias — treats genes as a set, not a sequence
  - Better suited to HVG-reduced input where a simple baseline is desirable
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBlock(nn.Module):
    """Linear -->  BN --> ReLU --> Dropout"""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.4):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class MouseClockMLP(nn.Module):
    """
    3-layer MLP encoder + model embedding + sex embedding + dual output heads

    Parameters
    ----------
    n_genes : number of HVG input features (e.g. 2000)
    n_models : embedding table size (len(DATASET_TO_ID) = 4)
    embed_dim : per-model embedding size (32)
    sex_embed_dim : per-sex embedding size (8)
    hidden_dims : encoder layer widths [512, 256, 128]
    dropouts : dropout per layer [0.5, 0.4, 0.3]
    fc_hidden : fusion MLP width (128)
    fc_dropout : fusion dropout (0.3)
    lambda_age : auxiliary age loss weight (0.2)
    max_age : age normalisation constant in months (18.0)
    """

    def __init__(
        self,
        n_genes: int = 2000,
        n_models: int = 4,
        embed_dim: int = 32,
        sex_embed_dim: int = 8,
        hidden_dims: list = None,
        dropouts: list = None,
        fc_hidden: int = 128,
        fc_dropout: float = 0.3,
        lambda_age: float = 0.2,
        max_age: float = 18.0,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        if dropouts is None:
            dropouts = [0.5, 0.4, 0.3]

        assert len(dropouts) == len(hidden_dims), \
            "dropouts must have same length as hidden_dims"

        self.lambda_age = lambda_age
        self.max_age = max_age
        self.sex_embed_dim = sex_embed_dim

        # MLP encoder
        layers = []
        in_dim = n_genes
        for out_dim, drop in zip(hidden_dims, dropouts):
            layers.append(MLPBlock(in_dim, out_dim, dropout=drop))
            in_dim = out_dim
        self.encoder = nn.Sequential(*layers)
        enc_out_dim = hidden_dims[-1] # 128

        # Model embedding
        self.model_embed = nn.Embedding(n_models, embed_dim)
        nn.init.normal_(self.model_embed.weight, std=0.01)

        # Sex embedding
        # 2 sexes (SEX_TO_ID: male=0, female=1)
        self.sex_embed = nn.Embedding(2, sex_embed_dim)
        nn.init.normal_(self.sex_embed.weight, std=0.01)

        # Fusion MLP 
        fusion_in = enc_out_dim + embed_dim + sex_embed_dim # 128 + 32 + 8 = 168
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, fc_hidden, bias=False),
            nn.BatchNorm1d(fc_hidden),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
        )

        # Output heads
        self.disease_head = nn.Linear(fc_hidden, 1)
        self.age_head = nn.Linear(fc_hidden, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # Forward

    def forward(
        self,
        x: torch.Tensor,
        model_id: torch.Tensor,
        sex_id: torch.Tensor = None,
    ):
        """
        x : (B, n_genes)  — note: NO channel dim, unlike CNN
        model_id : (B,) long
        sex_id : (B,) long  — SEX_TO_ID: male=0, female=1
        If None, a zero tensor is used for backward compatibility

        Returns
        -------
        disease_pred : (B,) in [0, 1]
        age_pred : (B,) in months
        """
        feat = self.encoder(x) # (B, 128)
        emb = self.model_embed(model_id) # (B, 32)

        if sex_id is None:
            sex_id = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        s_emb = self.sex_embed(sex_id) # (B, sex_embed_dim)

        fused = self.fusion(torch.cat([feat, emb, s_emb], dim=1))

        disease_pred = torch.sigmoid(self.disease_head(fused)).squeeze(-1)
        age_pred = self.age_head(fused).squeeze(-1)
        return disease_pred, age_pred

    # SHAP-compatible wrapper

    def disease_only(
        self,
        x: torch.Tensor,
        model_id: torch.Tensor,
        sex_id: torch.Tensor = None,
    ):
        dp, _ = self.forward(x, model_id, sex_id)
        return dp.unsqueeze(-1)

    # Loss
    def compute_loss(
        self,
        disease_pred: torch.Tensor,
        age_pred: torch.Tensor,
        disease_true: torch.Tensor,
        age_true: torch.Tensor,
        genotype_norm: torch.Tensor = None,
        delta: float = 0.0,
    ):
        """
        Combined loss with symmetric MAE for both WT and AD samples

        Previous version used Hinge-MAE for AD (only penalising under-prediction),
        which created a free zone for over-predictions. This version uses
        symmetric MAE for AD as well, matching the fixed CNN logic.

        Parameters
        ----------
        disease_pred : (B,) model predictions in [0, 1]
        age_pred : (B,) predicted age in months
        disease_true : (B,) ground-truth disease score in [0, 1]
        age_true : (B,) ground-truth age in months
        genotype_norm : (B,) long tensor — 0 = WT, 1 = AD
        delta : retained for API compatibility; unused

        Returns
        -------
        total_loss, disease_loss, age_loss
        """
        if genotype_norm is None:
            d_loss = F.mse_loss(disease_pred, disease_true)
        else:
            wt_mask = (genotype_norm == 0)
            ad_mask = (genotype_norm == 1)

            d_loss = torch.tensor(0.0, device=disease_pred.device)

            # WT: standard MAE
            if wt_mask.any():
                d_loss = d_loss + F.l1_loss(
                    disease_pred[wt_mask], disease_true[wt_mask]
                )

            # AD: symmetric MAE
            if ad_mask.any():
                mae_ad = torch.abs(disease_pred[ad_mask] - disease_true[ad_mask])
                d_loss = d_loss + mae_ad.mean()

            # Average equally across present groups
            n_groups = wt_mask.any().float() + ad_mask.any().float()
            d_loss = d_loss / n_groups.clamp(min=1.0)

        a_loss = F.mse_loss(age_pred / self.max_age, age_true / self.max_age)
        total = d_loss + self.lambda_age * a_loss
        return total, d_loss, a_loss

    # Post-training analysis

    def embedding_distance_matrix(self, id_to_name: dict) -> dict:
        W = self.model_embed.weight.detach()
        W_norm = F.normalize(W, dim=1)
        cos = (W_norm @ W_norm.T).cpu().numpy()
        names = [id_to_name.get(i, str(i)) for i in range(len(W))]
        return {"names": names, "cosine_similarity": cos}

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


#Sanity check
if __name__ == "__main__":
    import sys
    from data.dataset import build_loaders, DATASET_TO_ID

    h5 = sys.argv[1] if len(sys.argv) > 1 else "data/hvg_combined_expression.h5"
    meta = sys.argv[2] if len(sys.argv) > 2 else "data/hvg_combined_metadata.csv"

    train_loader, val_loader, test_loader, ds = build_loaders(h5, meta, batch_size=16)
    n_genes = ds.get_gene_count()

    model = MouseClockMLP(n_genes=n_genes, n_models=len(DATASET_TO_ID))
    total, trainable = model.count_parameters()
    print(f"\nMLP parameters: {total:,} total  |  {trainable:,} trainable")

    # NOTE: MLP expects (B, n_genes), not (B, 1, n_genes) like the CNN
    x, disease, age, model_id, sex_id = next(iter(train_loader))
    x_flat = x.squeeze(1)

    model.eval()
    with torch.no_grad():
        d_pred, a_pred = model(x_flat, model_id, sex_id)
    print(f"Forward pass: x={x_flat.shape}  disease={d_pred.shape}  age={a_pred.shape}")
    print(f"  disease range: [{d_pred.min():.3f}, {d_pred.max():.3f}]")

    model.train()
    d_pred, a_pred = model(x_flat, model_id, sex_id)
    total_loss, d_loss, a_loss = model.compute_loss(
        d_pred,
        a_pred,
        disease,
        age,
        genotype_norm=None, # replace with genotype tensor if available in loader
    )
    print(
        f"Loss (random): disease={d_loss.item():.4f}  "
        f"age={a_loss.item():.4f}  total={total_loss.item():.4f}"
    )