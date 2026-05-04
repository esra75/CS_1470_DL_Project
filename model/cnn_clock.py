"""
model/cnn_clock.py

2 models in one file:
  1. MouseClockCNN  — 1D CNN with learnable model embedding and dual output heads
  2. ElasticNetClock — sklearn elastic net baseline for honest comparison

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# 1. Elastic net baseline
class ElasticNetClock:
    """
    Sklearn-backed elastic net clock - predicts disease_score from gene expression.

    Usage
    -----
    clf = ElasticNetClock(alpha=0.01, l1_ratio=0.5)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_val)
    """

    def __init__(self, alpha: float = 0.01, l1_ratio: float = 0.5, max_iter: int = 2000):
        from sklearn.linear_model import ElasticNet
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("enet",   ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, random_state=42)),
        ])

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        coefs = self.model.named_steps["enet"].coef_
        n_nonzero = (coefs != 0).sum()
        print(f" ElasticNet: {n_nonzero} non-zero features / {len(coefs)}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.clip(self.model.predict(X), 0.0, 1.0)

    def score_mse(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        return float(np.mean((preds - y) ** 2))

    def get_top_genes(self, gene_names: np.ndarray, n: int = 50):
        """Return top-n genes by absolute elastic net coefficient."""
        coefs = self.model.named_steps["enet"].coef_
        idx = np.argsort(np.abs(coefs))[::-1][:n]
        return list(zip(gene_names[idx], coefs[idx]))


# 2. CNN building blocks

class ConvBlock(nn.Module):
    """
    Pre-activation residual block: BN -> ReLU -> Conv -> BN -> ReLU -> Conv
    + skip connection with optional projection

    Pre-activation (He et al. 2016) because gradients flow through the skip
    path without passing through BN/ReLU — more stable with small n where
    batch statistics are noisy
    """

    def __init__(self, in_ch: int, out_ch: int,
                 kernel: int = 7, stride: int = 1,
                 pool_size: int = 0):
        super().__init__()
        pad = kernel // 2

        self.bn1   = nn.BatchNorm1d(in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=pad, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, stride=1, padding=pad, bias=False)

        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

        self.pool = nn.MaxPool1d(pool_size) if pool_size > 0 else None

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = out + self.shortcut(x)
        if self.pool is not None:
            out = self.pool(out)
        return out


# 3. Main CNN model

class MouseClockCNN(nn.Module):
    """
    Parameters
    ----------
    n_genes : number of input genes (default 55449)
    n_models : slots in embedding table; set to len(DATASET_TO_ID) = 4
    embed_dim : per-model embedding dimension (32)
    cnn_channels : filter counts per block [32, 64, 128, 256]
    fc_hidden : fusion MLP width (128)
    dropout : MLP dropout rate (0.3)
    lambda_age : auxiliary age loss weight (0.2)
    max_age : age normalization constant in months (18.0)
    """

    def __init__(self, 
                 n_genes: int = 55449,
                 n_models: int = 4,
                 embed_dim: int = 32,
                 sex_embed_dim: int = 8,
                 cnn_channels: list = None,
                 fc_hidden: int = 128,
                 dropout: float = 0.3,
                 lambda_age: float = 0.2,
                 max_age: float = 18.0):
        super().__init__()

        if cnn_channels is None:
            cnn_channels = [32, 64, 128, 256]

        self.lambda_age = lambda_age
        self.max_age = max_age
        self.sex_embed_dim = sex_embed_dim

        # CNN encoder
        # Spatial progression (approximate):
        #   Gene input -> stride8+pool4 -> ~1733 -> pool4 -> ~433 -> pool4 -> ~108 -> 108
        self.encoder = nn.Sequential(
            ConvBlock(1, cnn_channels[0], kernel=15, stride=8, pool_size=4),
            ConvBlock(cnn_channels[0], cnn_channels[1], kernel=7, stride=1, pool_size=4),
            ConvBlock(cnn_channels[1], cnn_channels[2], kernel=5, stride=1, pool_size=4),
            ConvBlock(cnn_channels[2], cnn_channels[3], kernel=3, stride=1, pool_size=0),
            nn.AdaptiveAvgPool1d(1),
        )
        cnn_out_dim = cnn_channels[-1] # 256

        # Model embedding
        self.model_embed = nn.Embedding(n_models, embed_dim)
        nn.init.normal_(self.model_embed.weight, std=0.01)

        # Sex embedding
        # 2 sexes (SEX_TO_ID: male=0, female=1); small embedding captures
        # the sex-specific bias without dominating the disease signal
        self.sex_embed = nn.Embedding(2, sex_embed_dim)
        nn.init.normal_(self.sex_embed.weight, std=0.01)

        # Fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(cnn_out_dim + embed_dim + sex_embed_dim, fc_hidden),
            nn.BatchNorm1d(fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Output heads
        self.disease_head = nn.Linear(fc_hidden, 1)
        self.age_head = nn.Linear(fc_hidden, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, model_id: torch.Tensor,
                sex_id: torch.Tensor = None):
        """
        x : (B, 1, n_genes)
        model_id : (B,) long
        sex_id : (B,) long  — SEX_TO_ID: male=0, female=1


        Returns
        -------
        disease_pred : (B,) in [0, 1]
        age_pred : (B,) in months
        """
        feat = self.encoder(x).squeeze(-1)
        emb = self.model_embed(model_id)

        if sex_id is None: sex_id = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        s_emb = self.sex_embed(sex_id)

        fused = self.fusion(torch.cat([feat, emb, s_emb], dim=1))

        disease_pred = torch.sigmoid(self.disease_head(fused)).squeeze(-1)
        age_pred = self.age_head(fused).squeeze(-1)
        return disease_pred, age_pred

    def disease_only(self, x: torch.Tensor, model_id: torch.Tensor, sex_id: torch.Tensor = None):
        dp, _ = self.forward(x, model_id, sex_id)
        return dp.unsqueeze(-1)

    def compute_loss(self,
                     disease_pred: torch.Tensor,
                     age_pred: torch.Tensor,
                     disease_true: torch.Tensor,
                     age_true: torch.Tensor,
                     genotype_norm: torch.Tensor = None,
                     delta: float = 0.0):
        """
        Combined loss with DeepQA-style Hinge-MAE for diseased samples!

        Motivation (Qi et al. 2025, DeepQA):
        Standard MSE penalises both under- and over-prediction equally.
        For WT samples (disease_score=0), this is correct — the model should
        predict exactly 0. For AD samples, biological age is *at least*
        chronological age: the model should never predict a lower disease score
        than the true label, but predicting higher is acceptable (the animal
        may be more progressed than the linear score implies).

        Hinge-MAE for AD samples:
        loss = max(0, disease_true - disease_pred - delta)
        This only penalizes under-prediction. delta is a tolerance margin
        (default 0.0 = no tolerance; try 0.05 to allow slight under-prediction).

        For WT samples, standard MAE is used (disease_score=0, no ambiguity).
        If genotype_norm is not provided (e.g. during a quick test), falls back
        to plain MSE for backward compatibility.

        Parameters
        ----------
        disease_pred : (B,) model predictions in [0, 1]
        age_pred : (B,)  predicted age in months
        disease_true : (B,) ground-truth disease score in [0, 1]
        age_true : (B,) ground-truth age in months
        genotype_norm : (B,) long tensor — 0 = WT, 1 = AD  (from dataset)
        delta : hinge margin (default 0.0)

        Returns
        -------
        total_loss, disease_loss, age_loss
        """
        if genotype_norm is None:
            # Fallback: plain MSE (used during __main__ sanity check)
            d_loss = F.mse_loss(disease_pred, disease_true)
        else:
            wt_mask = (genotype_norm == 0) # WT samples
            ad_mask = (genotype_norm == 1) # AD samples

            d_loss = torch.tensor(0.0, device=disease_pred.device)

            # WT: standard MAE (ground truth is exactly 0)
            if wt_mask.any():
                d_loss = d_loss + F.l1_loss(
                    disease_pred[wt_mask], disease_true[wt_mask])

            # AD: Hinge-MAE — only penalise under-prediction
            if ad_mask.any():
                hinge = F.relu(disease_true[ad_mask] - disease_pred[ad_mask] - delta)
                d_loss = d_loss + hinge.mean()

            # Average equally across both groups
            n_groups = wt_mask.any().float() + ad_mask.any().float()
            d_loss = d_loss / n_groups.clamp(min=1.0)

        a_loss = F.mse_loss(age_pred / self.max_age, age_true / self.max_age)
        total  = d_loss + self.lambda_age * a_loss
        return total, d_loss, a_loss

    def embedding_distance_matrix(self, id_to_name: dict) -> dict:
        """Pairwise cosine similarity between model embeddings post-training."""
        W = self.model_embed.weight.detach()
        W_norm = F.normalize(W, dim=1)
        cos_sim = (W_norm @ W_norm.T).cpu().numpy()
        names = [id_to_name.get(i, str(i)) for i in range(len(W))]
        return {"names": names, "cosine_similarity": cos_sim}

    def count_parameters(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# Sanity check
if __name__ == "__main__":
    import sys, h5py, pandas as pd
    from data.dataset import build_loaders, MouseClockDataset, DATASET_TO_ID

    h5   = sys.argv[1] if len(sys.argv) > 1 else "data/combined_expression.h5"
    meta = sys.argv[2] if len(sys.argv) > 2 else "data/combined_metadata.csv"

    train_loader, val_loader, test_loader, ds = build_loaders(h5, meta, batch_size=16)
    n_genes = ds.get_gene_count()

    # CNN
    cnn = MouseClockCNN(n_genes=n_genes, n_models=len(DATASET_TO_ID))
    total, trainable = cnn.count_parameters()
    print(f"\nCNN parameters: {total:,} total  |  {trainable:,} trainable")

    x, disease, age, model_id, sex_id = next(iter(train_loader))
    cnn.eval()
    with torch.no_grad():
        d_pred, a_pred = cnn(x, model_id)
    print(f"Forward pass:  x={x.shape}  disease={d_pred.shape}  age={a_pred.shape}")
    print(f"  disease range: [{d_pred.min():.3f}, {d_pred.max():.3f}]")

    cnn.train()
    d_pred, a_pred = cnn(x, model_id)
    total_loss, d_loss, a_loss = cnn.compute_loss(d_pred, a_pred, disease, age)
    print(f"Loss (random weights): disease={d_loss.item():.4f}  age={a_loss.item():.4f}  total={total_loss.item():.4f}")

    # Elastic net baseline 
    print(f"\n--- Elastic net baseline ---")
    with h5py.File(h5, "r") as f:
        X_all      = f["X"][:]
        gene_names = np.array(f["gene_names"]).astype(str)
    meta_df = pd.read_csv(meta)

    full_ds = MouseClockDataset(h5, meta, noise_std=0.0)
    train_idx, val_idx, _ = full_ds.get_splits()

    X_tr = X_all[train_idx];  y_tr = meta_df["disease_score"].values[train_idx]
    X_va = X_all[val_idx];    y_va = meta_df["disease_score"].values[val_idx]

    enet = ElasticNetClock(alpha=0.01, l1_ratio=0.5)
    enet.fit(X_tr, y_tr)
    print(f"  train MSE: {enet.score_mse(X_tr, y_tr):.4f}")
    print(f"  val   MSE: {enet.score_mse(X_va, y_va):.4f}")
    print("Top 10 elastic net genes:")
    for gene, coef in enet.get_top_genes(gene_names, n=10):
        print(f"  {gene:20s}  {coef:+.4f}")

    # Embedding cosine similarity (random init)
    id_to_name = {v: k for k, v in DATASET_TO_ID.items()}
    emb = cnn.embedding_distance_matrix(id_to_name)
    print(f"\nEmbedding cosine similarity (random init):")
    print(f"  Models: {emb['names']}")
    print(f"  {emb['cosine_similarity'].round(3)}")
