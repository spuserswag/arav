"""
Bitcoin Up/Down Vol-Transformer Classifier (v2)
===============================================

Architecture upgrades over v1 (see v1/transformer.py for the old model):
  - Time-based data pipeline (via dataloader.py v2). Horizon / windows / segments
    are expressed in seconds, not tick counts.
  - Causal 1D conv stem before the transformer, to capture short-range tick
    dynamics cheaply.
  - Dual-stream input projection: price features and volatility features are
    projected separately and summed into the model dim (so the optimizer can
    balance their magnitudes independently).
  - Per-layer FiLM conditioning: each transformer layer is modulated by (γ, β)
    vectors predicted from a pooled volatility summary of the current window,
    giving the model an explicit "current vol regime" knob.
  - Attention pooling with a learned query (instead of last-token readout),
    so transient signals anywhere in the window can drive the prediction.
  - Segment-aware sequence construction: windows never straddle a websocket
    dropout (see dataloader.valid_sequence_ends).

Infra upgrades:
  - Versioned outputs/vN/, scaler + config.json + metrics saved per run.
  - Train loader shuffles; eval loaders don't. Test sequences use stride to
    produce (mostly) independent samples for honest metrics.
  - float32 throughout; pin_memory when on GPU.
  - cudnn determinism when a seed is passed.
  - Threshold sweep over full [0.01, 0.99].
  - Config dump is complete: model + data + training knobs.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import time
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from dataloader import (
    LoadedData,
    load_clean_r_style,
    valid_sequence_ends,
    build_sequences,
)


# ---------------------------------------------------------------------------
# Output versioning
# ---------------------------------------------------------------------------

def get_next_version_dir(base: str = "outputs") -> Path:
    base_path = Path(base)
    base_path.mkdir(exist_ok=True)
    existing = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("v")]
    versions = []
    for d in existing:
        try:
            versions.append(int(d.name[1:]))
        except ValueError:
            pass
    next_v = max(versions, default=0) + 1
    out_dir = base_path / f"v{next_v}"
    out_dir.mkdir()
    print(f"Output directory: {out_dir}")
    return out_dir


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al.)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Causal 1D conv stem
# ---------------------------------------------------------------------------

class CausalConv1d(nn.Module):
    """1D convolution with left-padding so output at time t only depends on t'≤t."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, T)
        x = nn.functional.pad(x, (self.pad, 0))
        return self.conv(x)


class ConvStem(nn.Module):
    """Two dilated causal convs with a residual; runs over the time dim."""

    def __init__(self, d_model: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.c1 = CausalConv1d(d_model, d_model, kernel_size, dilation=1)
        self.c2 = CausalConv1d(d_model, d_model, kernel_size, dilation=2)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, D)
        h = x.transpose(1, 2)  # (B, D, T)
        h = self.act(self.c1(h))
        h = self.drop(h)
        h = self.c2(h)
        h = h.transpose(1, 2)  # (B, T, D)
        return self.norm(x + h)


# ---------------------------------------------------------------------------
# Volatility summary → per-layer FiLM parameters
# ---------------------------------------------------------------------------

class VolFiLM(nn.Module):
    """
    Produce (γ, β) per transformer layer from a pooled volatility summary.

    Input to forward:  x_vol of shape (B, T, V) where V is len(vol_feature_cols)
    Output:            gammas, betas each shape (num_layers, B, 1, d_model)
    """

    def __init__(
        self,
        vol_dim: int,
        d_model: int,
        num_layers: int,
        hidden: int = 64,
        dropout: float = 0.1,
        pool: str = "mean",
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.pool = pool
        self.mlp = nn.Sequential(
            nn.Linear(vol_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_layers * d_model * 2),
        )
        # Zero-init the last layer so FiLM starts as identity (γ=1, β=0).
        last = self.mlp[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, x_vol: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pool over the time dim. Mean is a defensible default; could try attn later.
        if self.pool == "last":
            summary = x_vol[:, -1, :]
        else:
            summary = x_vol.mean(dim=1)
        params = self.mlp(summary)  # (B, num_layers * d_model * 2)
        B = params.size(0)
        params = params.view(B, self.num_layers, 2, self.d_model)
        # Identity at init: γ = 1 + Δγ, β = Δβ.
        gamma = 1.0 + params[:, :, 0, :]  # (B, L, D)
        beta = params[:, :, 1, :]          # (B, L, D)
        # Reshape to (L, B, 1, D) for per-layer indexing and broadcasting over T.
        gamma = gamma.permute(1, 0, 2).unsqueeze(2).contiguous()
        beta = beta.permute(1, 0, 2).unsqueeze(2).contiguous()
        return gamma, beta


# ---------------------------------------------------------------------------
# Transformer layer with FiLM + optional diagonal attention bias
# ---------------------------------------------------------------------------

class FiLMTransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        attn_diagonal_bias: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn_diagonal_bias = attn_diagonal_bias
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

    def _diag_bias(self, T: int, device: torch.device) -> Optional[torch.Tensor]:
        if self.attn_diagonal_bias <= 0:
            return None
        i = torch.arange(T, device=device, dtype=torch.float32)
        j = torch.arange(T, device=device, dtype=torch.float32)
        dist = (i.unsqueeze(1) - j.unsqueeze(0)).abs()
        return -self.attn_diagonal_bias * dist

    def forward(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor,  # (B, 1, D)
        beta: torch.Tensor,   # (B, 1, D)
        return_attn: bool = False,
    ):
        # Pre-norm, then FiLM, then self-attention.
        x_norm = self.norm1(x)
        x_cond = gamma * x_norm + beta
        attn_out, attn_w = self.self_attn(
            x_cond, x_cond, x_cond,
            attn_mask=self._diag_bias(x.size(1), x.device),
            need_weights=return_attn,
            average_attn_weights=False,
        )
        x = x + self.dropout1(attn_out)
        x = x + self.dropout2(self.ff(self.norm2(x)))
        if return_attn:
            return x, attn_w
        return x


# ---------------------------------------------------------------------------
# Attention pooling (learned query)
# ---------------------------------------------------------------------------

class AttentionPool(nn.Module):
    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        q = self.query.expand(B, -1, -1)            # (B, 1, D)
        out, _ = self.attn(q, x, x)                 # (B, 1, D)
        return self.norm(out.squeeze(1))            # (B, D)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class VolTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        vol_indices: List[int],
        d_model: int = 96,
        nhead: int = 6,
        num_layers: int = 3,
        dim_feedforward: int = 384,
        dropout: float = 0.2,
        pos_encoding: str = "sinusoidal",
        attn_diagonal_bias: float = 0.0,
        conv_kernel: int = 3,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.vol_indices = list(vol_indices)
        self.price_indices = [i for i in range(input_dim) if i not in set(self.vol_indices)]

        # Dual-stream projections (sum to d_model so each stream has full bandwidth).
        self.price_proj = nn.Linear(len(self.price_indices), d_model)
        self.vol_proj = nn.Linear(len(self.vol_indices), d_model)

        if pos_encoding == "sinusoidal":
            self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        elif pos_encoding == "learnable":
            self.pos_enc = LearnablePositionalEncoding(d_model, dropout=dropout)
        else:
            raise ValueError(f"pos_encoding must be 'sinusoidal' or 'learnable', got {pos_encoding!r}")

        self.conv_stem = ConvStem(d_model, kernel_size=conv_kernel, dropout=dropout)
        self.film = VolFiLM(
            vol_dim=len(self.vol_indices),
            d_model=d_model,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.layers = nn.ModuleList(
            [
                FiLMTransformerLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    attn_diagonal_bias=attn_diagonal_bias,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.pool = AttentionPool(d_model, nhead=min(nhead, 4), dropout=dropout)
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def _split(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_price = x[..., self.price_indices]
        x_vol = x[..., self.vol_indices]
        return x_price, x_vol

    def _encode(self, x: torch.Tensor, return_attn: bool = False):
        """Run the model up to the pooled representation (just before cls_head)."""
        x_price, x_vol = self._split(x)
        h = self.price_proj(x_price) + self.vol_proj(x_vol)
        h = self.pos_enc(h)
        h = self.conv_stem(h)

        gammas, betas = self.film(x_vol)  # (L, B, 1, D)
        last_attn = None
        for i, layer in enumerate(self.layers):
            if return_attn and i == len(self.layers) - 1:
                h, last_attn = layer(h, gammas[i], betas[i], return_attn=True)
            else:
                h = layer(h, gammas[i], betas[i], return_attn=False)

        h = self.final_norm(h)
        pooled = self.pool(h)
        return pooled, last_attn

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return (B, d_model) pooled embeddings — handy for t-SNE / probes."""
        pooled, _ = self._encode(x, return_attn=False)
        return pooled

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        pooled, last_attn = self._encode(x, return_attn=return_attn)
        logits = self.cls_head(pooled).squeeze(-1)
        if return_attn:
            return logits, last_attn
        return logits


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def save_tsne_plot(
    model: "VolTransformer",
    loader: DataLoader,
    device: torch.device,
    out_path: Path,
    n_samples: int = 1500,
    perplexity: int = 30,
    title: str = "t-SNE of Vol-Transformer pooled embeddings",
) -> None:
    """
    Project the pooled (pre-head) embeddings into 2D with t-SNE, save a side-
    by-side scatter coloured by true label and by model confidence.

    Skips silently if scikit-learn is missing or the loader yields no samples.
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("  Skipping t-SNE: sklearn.manifold.TSNE unavailable.")
        return

    model.eval()
    feats_list, y_list, logits_list = [], [], []
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb_d = xb.to(device)
            pooled = model.encode(xb_d)
            logits = model.cls_head(pooled).squeeze(-1)
            feats_list.append(pooled.cpu().numpy())
            logits_list.append(logits.cpu().numpy())
            y_list.append(yb.numpy())
            total += xb.size(0)
            if total >= n_samples:
                break

    if total == 0:
        print("  Skipping t-SNE: no samples in loader.")
        return

    feats = np.concatenate(feats_list)[:n_samples]
    y = np.concatenate(y_list)[:n_samples]
    logits = np.concatenate(logits_list)[:n_samples]
    probs = 1.0 / (1.0 + np.exp(-logits))

    # t-SNE is O(n²) in memory; keep perplexity valid for tiny samples too.
    eff_perp = max(5, min(perplexity, (len(feats) - 1) // 3))
    print(f"  Running t-SNE on {len(feats)} samples (perplexity={eff_perp}) …")
    tsne = TSNE(n_components=2, perplexity=eff_perp, random_state=42, init="pca")
    X2 = tsne.fit_transform(feats)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    ax = axes[0]
    sc = ax.scatter(X2[:, 0], X2[:, 1], c=y, cmap="coolwarm",
                    alpha=0.7, s=18, linewidths=0)
    plt.colorbar(sc, ax=ax, label="True label (0=down, 1=up)")
    ax.set_title("Coloured by True Label")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2"); ax.grid(True, alpha=0.2)

    ax = axes[1]
    sc2 = ax.scatter(X2[:, 0], X2[:, 1], c=probs, cmap="RdYlGn",
                     alpha=0.7, s=18, linewidths=0, vmin=0, vmax=1)
    plt.colorbar(sc2, ax=ax, label="Model confidence P(up)")
    ax.set_title("Coloured by Model Confidence")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2"); ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_path, format="png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved t-SNE plot → {out_path}")


def save_attention_heatmap(attn: torch.Tensor, out_path: Path, sample_idx: int = 0, head: int = 0) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    a = attn.detach().cpu().numpy()
    if a.ndim != 4:
        raise ValueError(f"Expected (B,H,T,T), got {a.shape}")
    plt.figure(figsize=(7, 6))
    sns.heatmap(a[sample_idx, head], cmap="viridis")
    plt.title("Self-attention (last layer)")
    plt.xlabel("Key timestep"); plt.ylabel("Query timestep")
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()


def save_training_curves(history: dict, out_dir: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    best_ep = history.get("best_epoch_auc")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Training History", fontsize=14, fontweight="bold")
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    ax.plot(epochs, history["val_loss"], label="Val Loss", linewidth=2, linestyle="--")
    if best_ep: ax.axvline(best_ep, color="red", linestyle=":", alpha=0.6, label=f"Best ({best_ep})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("BCE Loss"); ax.set_title("Loss"); ax.legend(); ax.grid(True, alpha=0.3)
    ax = axes[1]
    ax.plot(epochs, history["train_auc"], label="Train AUC", linewidth=2)
    ax.plot(epochs, history["val_auc"], label="Val AUC", linewidth=2, linestyle="--")
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="Random")
    if best_ep: ax.axvline(best_ep, color="red", linestyle=":", alpha=0.6, label=f"Best ({best_ep})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("ROC-AUC"); ax.set_title("AUC"); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = out_dir / "training_curves.png"
    plt.savefig(p, format="png", bbox_inches="tight", dpi=150); plt.close()
    print(f"  Saved training curves → {p}")


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def compute_pos_weight(y: np.ndarray, device: torch.device) -> torch.Tensor:
    n_pos = float(y.sum()); n_neg = float(len(y) - n_pos)
    ratio = n_neg / max(n_pos, 1)
    print(f"  Class balance — pos: {int(n_pos)}, neg: {int(n_neg)}, pos_weight: {ratio:.3f}")
    return torch.tensor([ratio], dtype=torch.float, device=device)


def find_best_threshold(logits: np.ndarray, targets: np.ndarray) -> float:
    """Sweep [0.01, 0.99] and pick threshold with best macro F1."""
    probs = 1.0 / (1.0 + np.exp(-logits))
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.01, 1.0, 0.01):
        preds = (probs >= t).astype(int)
        f1 = f1_score(targets, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def collect_logits(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits, all_t = [], []
    with torch.no_grad():
        for xb, yb in loader:
            all_logits.append(model(xb.to(device)).cpu().numpy())
            all_t.append(yb.numpy())
    return np.concatenate(all_logits), np.concatenate(all_t)


def eval_auc_from(logits: np.ndarray, targets: np.ndarray) -> float:
    try:
        return roc_auc_score(targets, 1.0 / (1.0 + np.exp(-logits)))
    except ValueError:
        return 0.5


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_transformer(
    csv_file: str = "data/part2/advanced_orderflow_ws.csv",
    # Loader knobs (time units)
    step_seconds: float = 0.5,
    horizon_seconds: float = 15.0,
    backward_seconds: float = 120.0,
    gap_seconds: float = 2.0,
    deadband_bps: float = 0.5,
    vol_fast_seconds: float = 5.0,
    vol_med_seconds: float = 30.0,
    vol_of_vol_seconds: float = 30.0,
    # Sequence shape
    seq_len: int = 30,
    test_stride: Optional[int] = None,   # None → horizon_seconds/step_seconds (independent windows)
    # Training
    n_epochs: int = 50,
    batch_size: int = 128,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    patience: int = 8,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    outputs_base: str = "outputs",
    seed: Optional[int] = None,
    # Model
    d_model: int = 96,
    nhead: int = 6,
    num_layers: int = 3,
    dim_feedforward: int = 384,
    dropout: float = 0.2,
    pos_encoding: str = "sinusoidal",
    attn_diagonal_bias: float = 0.0,
    conv_kernel: int = 3,
) -> Path:

    if seed is None:
        seed = int(time.time() * 1e6) % (2**31)
        print(f"Using random seed: {seed}")
    else:
        print(f"Using fixed seed: {seed}")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    out_dir = get_next_version_dir(outputs_base)
    checkpoint_path = out_dir / "transformer_best.pt"
    scaler_path = out_dir / "scaler.pkl"

    # 1. Load & clean (time-based, segment-aware).
    loaded: LoadedData = load_clean_r_style(
        csv_file,
        step_seconds=step_seconds,
        horizon_seconds=horizon_seconds,
        backward_seconds=backward_seconds,
        gap_seconds=gap_seconds,
        deadband_bps=deadband_bps,
        vol_fast_seconds=vol_fast_seconds,
        vol_med_seconds=vol_med_seconds,
        vol_of_vol_seconds=vol_of_vol_seconds,
    )
    print(f"Loaded {len(loaded.df)} rows, {len(loaded.feature_cols)} features, "
          f"{len(set(loaded.segment_ids))} segments.")
    print(f"  price features : {[c for c in loaded.feature_cols if c not in loaded.vol_feature_cols]}")
    print(f"  vol features   : {loaded.vol_feature_cols}")
    if loaded.X.size == 0:
        print("No usable rows. Aborting."); return

    # 2. Temporal split (on rows, not sequences — sequences are rebuilt per split).
    n = len(loaded.X)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_val - n_test

    X_train = loaded.X[:n_train]; y_train = loaded.y[:n_train]; seg_train = loaded.segment_ids[:n_train]
    X_val   = loaded.X[n_train:n_train + n_val]; y_val = loaded.y[n_train:n_train + n_val]; seg_val = loaded.segment_ids[n_train:n_train + n_val]
    X_test  = loaded.X[n_train + n_val:]; y_test = loaded.y[n_train + n_val:]; seg_test = loaded.segment_ids[n_train + n_val:]
    print(f"Split — train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    # 3. Scaler fit on TRAIN only (standard for time series).
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    with open(scaler_path, "wb") as f:
        pickle.dump({"scaler": scaler, "feature_cols": loaded.feature_cols,
                     "vol_feature_cols": loaded.vol_feature_cols}, f)
    print(f"  Saved scaler → {scaler_path}")

    # 4. Build segment-aware sequences. Train uses stride 1; test uses horizon stride
    #    so independent windows don't over-count correlated predictions in metrics.
    if test_stride is None:
        test_stride = max(1, int(round(horizon_seconds / step_seconds)))
    Xtr, ytr = build_sequences(X_train_s, y_train, seg_train, seq_len=seq_len, stride=1)
    Xvl, yvl = build_sequences(X_val_s, y_val, seg_val, seq_len=seq_len, stride=1)
    Xte, yte = build_sequences(X_test_s, y_test, seg_test, seq_len=seq_len, stride=test_stride)
    print(f"Sequences — train: {Xtr.shape}, val: {Xvl.shape}, test (stride={test_stride}): {Xte.shape}")

    if Xtr.shape[0] == 0 or Xvl.shape[0] == 0 or Xte.shape[0] == 0:
        print("Empty sequence set after segment filter; try a shorter seq_len or a larger gap_seconds.")
        return

    # 5. Loaders.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    pin = device.type == "cuda"

    def to_tensor(x): return torch.from_numpy(x)
    train_ds = TensorDataset(to_tensor(Xtr), to_tensor(ytr))
    val_ds = TensorDataset(to_tensor(Xvl), to_tensor(yvl))
    test_ds = TensorDataset(to_tensor(Xte), to_tensor(yte))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=pin, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=pin)

    # 6. Model / loss / optimizer.
    model = VolTransformer(
        input_dim=loaded.X.shape[1],
        vol_indices=loaded.vol_feature_indices,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        pos_encoding=pos_encoding,
        attn_diagonal_bias=attn_diagonal_bias,
        conv_kernel=conv_kernel,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    pos_weight = compute_pos_weight(ytr, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    # 7. Training — checkpoint by val AUC.
    print("\n=== Vol-Transformer — Bitcoin Up/Down ===")
    history = {"train_loss": [], "val_loss": [], "train_auc": [], "val_auc": [], "best_epoch_auc": None}
    best_val_auc = 0.0
    epochs_no_improve = 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=pin)
            yb = yb.to(device, non_blocking=pin)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(train_loader.dataset)

        # Val: collect logits once, reuse for loss and AUC.
        val_logits, val_targets = collect_logits(model, val_loader, device)
        # Numerically stable pos-weighted BCE from logits+targets:
        #   target=1: w * log(1 + exp(-logit)) = w * logaddexp(0, -logit)
        #   target=0:      log(1 + exp( logit)) =     logaddexp(0,  logit)
        w = float(pos_weight.item())
        pos_mask = val_targets > 0.5
        per_sample = np.where(
            pos_mask,
            w * np.logaddexp(0.0, -val_logits),
            np.logaddexp(0.0, val_logits),
        )
        val_loss = float(per_sample.mean())
        val_auc = eval_auc_from(val_logits, val_targets)
        train_logits, train_targets = collect_logits(model, train_loader, device)
        train_auc = eval_auc_from(train_logits, train_targets)

        scheduler.step(val_auc)
        cur_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)
        history["train_auc"].append(train_auc)
        history["val_auc"].append(val_auc)

        print(f"Epoch {epoch:3d}/{n_epochs} | loss {epoch_loss:.4f}/{val_loss:.4f} | "
              f"AUC {train_auc:.4f}/{val_auc:.4f} | lr {cur_lr:.2e}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            epochs_no_improve = 0
            history["best_epoch_auc"] = epoch
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✓ Checkpoint saved (val AUC={val_auc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}."); break

    save_training_curves(history, out_dir)

    # 8. Final test evaluation (best checkpoint, threshold tuned on val).
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"\nLoaded best checkpoint (epoch {history['best_epoch_auc']}).")
    except Exception as e:
        print(f"Could not load checkpoint: {e}")

    test_logits, test_targets = collect_logits(model, test_loader, device)
    val_logits, val_targets = collect_logits(model, val_loader, device)
    best_threshold = find_best_threshold(val_logits, val_targets)
    print(f"Best threshold (val): {best_threshold:.2f}")

    y_prob = 1.0 / (1.0 + np.exp(-test_logits))
    y_pred = (y_prob >= best_threshold).astype(int)
    acc = accuracy_score(test_targets, y_pred)
    try:
        auc = roc_auc_score(test_targets, y_prob)
    except ValueError:
        auc = float("nan")

    print(f"\nTest Accuracy : {acc:.4f}")
    print(f"Test ROC-AUC  : {auc:.4f}")
    print("\nClassification Report:\n", classification_report(test_targets, y_pred, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(test_targets, y_pred))

    # 9. Persist everything useful for inference + reproduction.
    config = {
        "csv_file": csv_file,
        "data": loaded.config,
        "seq_len": seq_len,
        "test_stride": test_stride,
        "val_ratio": val_ratio, "test_ratio": test_ratio,
        "feature_cols": loaded.feature_cols,
        "vol_feature_cols": loaded.vol_feature_cols,
        "model": {
            "d_model": d_model, "nhead": nhead, "num_layers": num_layers,
            "dim_feedforward": dim_feedforward, "dropout": dropout,
            "pos_encoding": pos_encoding, "attn_diagonal_bias": attn_diagonal_bias,
            "conv_kernel": conv_kernel,
        },
        "training": {
            "n_epochs": n_epochs, "batch_size": batch_size, "lr": lr,
            "weight_decay": weight_decay, "patience": patience,
            "seed": seed,
        },
        "results": {
            "best_epoch_auc": history["best_epoch_auc"],
            "best_val_auc": best_val_auc,
            "test_accuracy": float(acc),
            "test_auc": float(auc) if not np.isnan(auc) else None,
            "threshold": best_threshold,
            "n_params": n_params,
        },
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    with open(out_dir / "metrics_summary.txt", "w") as f:
        f.write(f"Version dir      : {out_dir}\n")
        f.write(f"CSV              : {csv_file}\n")
        f.write(f"Horizon (s)      : {horizon_seconds}\n")
        f.write(f"Step (s)         : {step_seconds}\n")
        f.write(f"Seq len (steps)  : {seq_len}  (~ {seq_len * step_seconds:.1f}s context)\n")
        f.write(f"Seed             : {seed}\n")
        f.write(f"Best epoch (AUC) : {history['best_epoch_auc']}\n")
        f.write(f"Best val AUC     : {best_val_auc:.4f}\n")
        f.write(f"Threshold        : {best_threshold:.2f}\n")
        f.write(f"Test Accuracy    : {acc:.4f}\n")
        f.write(f"Test AUC         : {auc:.4f}\n\n")
        f.write(classification_report(test_targets, y_pred, zero_division=0))
    print(f"  Saved metrics summary → {out_dir / 'metrics_summary.txt'}")

    # 10. Self-attention heatmap from a val sample.
    try:
        xb0, _ = next(iter(val_loader))
        _, attn = model(xb0.to(device), return_attn=True)
        if attn is not None:
            save_attention_heatmap(attn, out_dir / "attention_heatmap.png")
            np.save(out_dir / "attention_weights.npy", attn.detach().cpu().numpy())
            print(f"  Saved attention heatmap → {out_dir / 'attention_heatmap.png'}")
    except Exception as e:
        print(f"  Could not save attention heatmap: {e}")

    # 11. t-SNE of pooled embeddings on the val set.
    try:
        save_tsne_plot(
            model=model,
            loader=val_loader,
            device=device,
            out_path=out_dir / "tsne.png",
            n_samples=1500,
            perplexity=30,
            title=f"t-SNE of pooled embeddings  ({out_dir.name}, val set)",
        )
    except Exception as e:
        print(f"  Could not save t-SNE plot: {e}")

    print(f"\nAll outputs saved to: {out_dir}/")
    return out_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bitcoin Up/Down Vol-Transformer")
    p.add_argument("--csv", type=str, default="data/part2/advanced_orderflow_ws.csv")
    # Loader
    p.add_argument("--step-seconds", type=float, default=0.5)
    p.add_argument("--horizon-seconds", type=float, default=15.0)
    p.add_argument("--backward-seconds", type=float, default=120.0)
    p.add_argument("--gap-seconds", type=float, default=2.0)
    p.add_argument("--deadband-bps", type=float, default=0.5)
    p.add_argument("--vol-fast-seconds", type=float, default=5.0)
    p.add_argument("--vol-med-seconds", type=float, default=30.0)
    p.add_argument("--vol-of-vol-seconds", type=float, default=30.0)
    # Sequence
    p.add_argument("--seq-len", type=int, default=30, help="window length in grid steps")
    p.add_argument("--test-stride", type=int, default=None,
                   help="stride for test windows; default = horizon/step (independent).")
    # Training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--outputs-base", type=str, default="outputs")
    p.add_argument("--seed", type=int, default=None)
    # Model
    p.add_argument("--d-model", type=int, default=96)
    p.add_argument("--nhead", type=int, default=6)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--dim-feedforward", type=int, default=384)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--pos-encoding", type=str, default="sinusoidal",
                   choices=["sinusoidal", "learnable"])
    p.add_argument("--attn-diagonal-bias", type=float, default=0.0)
    p.add_argument("--conv-kernel", type=int, default=3)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    train_transformer(
        csv_file=a.csv,
        step_seconds=a.step_seconds, horizon_seconds=a.horizon_seconds,
        backward_seconds=a.backward_seconds, gap_seconds=a.gap_seconds,
        deadband_bps=a.deadband_bps,
        vol_fast_seconds=a.vol_fast_seconds, vol_med_seconds=a.vol_med_seconds,
        vol_of_vol_seconds=a.vol_of_vol_seconds,
        seq_len=a.seq_len, test_stride=a.test_stride,
        n_epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
        weight_decay=a.weight_decay, patience=a.patience,
        val_ratio=a.val_ratio, test_ratio=a.test_ratio,
        outputs_base=a.outputs_base, seed=a.seed,
        d_model=a.d_model, nhead=a.nhead, num_layers=a.num_layers,
        dim_feedforward=a.dim_feedforward, dropout=a.dropout,
        pos_encoding=a.pos_encoding, attn_diagonal_bias=a.attn_diagonal_bias,
        conv_kernel=a.conv_kernel,
    )
