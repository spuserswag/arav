"""
Bitcoin Up/Down Transformer Classifier
======================================
Improvements in this version:
  - Versioned output folder: outputs/vN/ auto-increments each run
  - Checkpoint saved by val AUC (not val loss) — tracks what we actually care about
  - Training curves (loss + AUC per epoch) saved as PNG
  - Scaler saved alongside checkpoint for inference reuse
  - Threshold tuning on val set, applied to test
  - Gradient clipping, pos_weight, positional encoding, pre-norm all retained
"""

import argparse
import json
import os
import pickle
import random
import time
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
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

from dataloader import load_clean_r_style


# ---------------------------------------------------------------------------
# Output versioning
# ---------------------------------------------------------------------------

def get_next_version_dir(base: str = "outputs") -> Path:
    """Auto-increments outputs/v1, v2, v3 ... on each run."""
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
# Positional Encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al.)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(500.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional embeddings (trained with the model)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
        pos_encoding: str = "sinusoidal",
        attn_diagonal_bias: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        if pos_encoding == "sinusoidal":
            self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        elif pos_encoding == "positional":
            self.pos_enc = LearnablePositionalEncoding(d_model, dropout=dropout)
        else:
            raise ValueError(f"pos_encoding must be 'sinusoidal' or 'positional', got {pos_encoding!r}")
        self.encoder = TransformerEncoderWithAttn(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            attn_diagonal_bias=attn_diagonal_bias,
        )
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        h = self.input_proj(x)
        h = self.pos_enc(h)
        if return_attn:
            h, attn = self.encoder(h, return_attn=True)
        else:
            h = self.encoder(h)
            attn = None
        last = h[:, -1, :]
        logits = self.cls_head(last).squeeze(-1)
        if return_attn:
            return logits, attn
        return logits


class TransformerEncoderLayerWithAttn(nn.Module):
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

    def _get_attn_mask(self, T: int, device: torch.device) -> Optional[torch.Tensor]:
        """Bias favoring diagonal: -lambda * |i-j|. Encourages local attention."""
        if self.attn_diagonal_bias <= 0:
            return None
        i = torch.arange(T, device=device, dtype=torch.float32)
        j = torch.arange(T, device=device, dtype=torch.float32)
        dist = (i.unsqueeze(1) - j.unsqueeze(0)).abs()
        return (-self.attn_diagonal_bias * dist).to(device)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        x_norm = self.norm1(x)
        T = x.size(1)
        attn_mask = self._get_attn_mask(T, x.device)
        attn_out, attn_w = self.self_attn(
            x_norm,
            x_norm,
            x_norm,
            attn_mask=attn_mask,
            need_weights=return_attn,
            average_attn_weights=False,
        )
        x = x + self.dropout1(attn_out)
        x = x + self.dropout2(self.ff(self.norm2(x)))
        if return_attn:
            return x, attn_w  # (B, H, T, T)
        return x


class TransformerEncoderWithAttn(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        attn_diagonal_bias: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayerWithAttn(
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

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        last_attn = None
        for layer in self.layers:
            if return_attn:
                x, last_attn = layer(x, return_attn=True)
            else:
                x = layer(x, return_attn=False)
        x = self.final_norm(x)
        if return_attn:
            return x, last_attn
        return x


def save_attention_heatmap(
    attn: torch.Tensor,
    out_path: Path,
    sample_idx: int = 0,
    head: int = 0,
    title: str = "Self-attention heatmap (last layer)",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    attn_np = attn.detach().cpu().numpy()
    if attn_np.ndim != 4:
        raise ValueError(f"Expected attn with shape (B,H,T,T), got {attn_np.shape}")
    A = attn_np[sample_idx, head]
    plt.figure(figsize=(7, 6))
    sns.heatmap(A, cmap="viridis")
    plt.title(title)
    plt.xlabel("Key timestep")
    plt.ylabel("Query timestep")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def make_sequence_dataset(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(X) <= seq_len:
        raise ValueError("Not enough samples to create sequences.")
    X_seq, y_seq = [], []
    for t in range(seq_len - 1, len(X)):
        X_seq.append(X[t - seq_len + 1 : t + 1])
        y_seq.append(y[t])
    return np.stack(X_seq), np.array(y_seq)


def compute_pos_weight(y: np.ndarray, device: torch.device) -> torch.Tensor:
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    ratio = n_neg / max(n_pos, 1)
    print(f"  Class balance — pos: {int(n_pos)}, neg: {int(n_neg)}, pos_weight: {ratio:.3f}")
    return torch.tensor([ratio], dtype=torch.float, device=device)


def find_best_threshold(logits: np.ndarray, targets: np.ndarray) -> float:
    """Sweep [0.3, 0.7] and pick threshold with best macro F1."""
    probs = 1 / (1 + np.exp(-logits))
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.3, 0.71, 0.02):
        preds = (probs >= t).astype(int)
        f1 = f1_score(targets, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


def eval_auc(model, loader, device) -> float:
    """Return ROC-AUC of model on a DataLoader. Returns 0.5 if undefined."""
    model.eval()
    logits_list, targets_list = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits_list.append(model(xb.to(device)).cpu().numpy())
            targets_list.append(yb.numpy())
    logits  = np.concatenate(logits_list)
    targets = np.concatenate(targets_list)
    probs   = 1 / (1 + np.exp(-logits))
    try:
        return roc_auc_score(targets, probs)
    except ValueError:
        return 0.5


# ---------------------------------------------------------------------------
# Training curve plot
# ---------------------------------------------------------------------------

def save_training_curves(history: dict, out_dir: Path) -> None:
    """Save loss and AUC curves side-by-side as a PNG."""
    epochs = range(1, len(history["train_loss"]) + 1)
    best_ep = history.get("best_epoch_auc", None)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Training History", fontsize=14, fontweight="bold")

    # Loss
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    ax.plot(epochs, history["val_loss"],   label="Val Loss",   linewidth=2, linestyle="--")
    if best_ep:
        ax.axvline(best_ep, color="red", linestyle=":", alpha=0.6, label=f"Best epoch ({best_ep})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE Loss")
    ax.set_title("Loss over Epochs")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # AUC
    ax = axes[1]
    ax.plot(epochs, history["train_auc"], label="Train AUC", linewidth=2)
    ax.plot(epochs, history["val_auc"],   label="Val AUC",   linewidth=2, linestyle="--")
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="Random baseline")
    if best_ep:
        ax.axvline(best_ep, color="red", linestyle=":", alpha=0.6, label=f"Best epoch ({best_ep})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("AUC over Epochs")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    png_path = out_dir / "training_curves.png"
    plt.savefig(png_path, format="png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved training curves → {png_path}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_transformer(
    csv_file: str = "advanced_orderflow_data.csv",
    horizon: int = 5,
    seq_len: int = 40,
    n_epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-4,
    patience: int = 8,
    val_ratio: float = 0.15,
    outputs_base: str = "outputs",
    seed: Optional[int] = None,
    encoding: str = "sinusoidal",
    attn_diagonal_bias: float = 0.0,
) -> None:

    if seed is None:
        seed = int(time.time() * 1e6) % (2**31)
        print(f"Using random seed: {seed}")
    else:
        print(f"Using fixed seed: {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    out_dir         = get_next_version_dir(outputs_base)
    checkpoint_path = out_dir / "transformer_best.pt"
    scaler_path     = out_dir / "scaler.pkl"

    # 1. Load and clean data (mirror R clean.R logic)
    loaded = load_clean_r_style(csv_file)
    df, X, y, feature_cols = loaded.df, loaded.X, loaded.y, loaded.feature_cols
    print(f"Loaded {len(df)} rows with {len(feature_cols)} features.")
    if len(df) == 0 or X.size == 0:
        print("No usable rows.")
        return

    # 2. Temporal split
    n       = len(X)
    n_test  = int(n * 0.15)
    n_val   = int(n * val_ratio)
    n_train = n - n_val - n_test

    X_train_raw = X[:n_train];           y_train_raw = y[:n_train]
    X_val_raw   = X[n_train:n_train+n_val]; y_val_raw = y[n_train:n_train+n_val]
    X_test_raw  = X[n_train+n_val:];     y_test_raw  = y[n_train+n_val:]

    print(f"Split — train: {len(X_train_raw)}, val: {len(X_val_raw)}, test: {len(X_test_raw)}")

    # 3. Scale (fit only on train)
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_raw)
    X_val_s   = scaler.transform(X_val_raw)
    X_test_s  = scaler.transform(X_test_raw)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Saved scaler → {scaler_path}")

    # 4. Sequences
    try:
        X_train_seq, y_train_seq = make_sequence_dataset(X_train_s, y_train_raw, seq_len)
        X_val_seq,   y_val_seq   = make_sequence_dataset(X_val_s,   y_val_raw,   seq_len)
        X_test_seq,  y_test_seq  = make_sequence_dataset(X_test_s,  y_test_raw,  seq_len)
    except ValueError as e:
        print(f"Skipping: {e}"); return

    print(f"Sequences — train: {X_train_seq.shape}, val: {X_val_seq.shape}, test: {X_test_seq.shape}")
    print(f"Positional encoding: {encoding}")
    if attn_diagonal_bias > 0:
        print(f"Attention diagonal bias: {attn_diagonal_bias} (encourages local/diagonal attention)")

    # 5. DataLoaders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    def to_loader(Xs, ys):
        ds = TensorDataset(torch.from_numpy(Xs).float(), torch.from_numpy(ys).float())
        return DataLoader(ds, batch_size=batch_size, shuffle=False)

    train_loader = to_loader(X_train_seq, y_train_seq)
    val_loader   = to_loader(X_val_seq,   y_val_seq)
    test_loader  = to_loader(X_test_seq,  y_test_seq)

    # 6. Model / loss / optimizer
    model = TimeSeriesTransformer(
        input_dim=X_train_s.shape[1],
        pos_encoding=encoding,
        attn_diagonal_bias=attn_diagonal_bias,
    ).to(device)
    pos_weight = compute_pos_weight(y_train_seq, device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3   # mode=max: we track AUC
    )

    # 7. Training — checkpoint by val AUC
    print("\n=== Transformer — Bitcoin Up/Down ===")
    history = {
        "train_loss": [], "val_loss": [],
        "train_auc":  [], "val_auc":  [],
        "best_epoch_auc": 1,
    }
    best_val_auc      = 0.0
    epochs_no_improve = 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        train_auc = eval_auc(model, train_loader, device)
        val_auc   = eval_auc(model, val_loader,   device)

        scheduler.step(val_auc)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)
        history["train_auc"].append(train_auc)
        history["val_auc"].append(val_auc)

        print(
            f"Epoch {epoch:3d}/{n_epochs} | "
            f"loss {epoch_loss:.4f}/{val_loss:.4f} | "
            f"AUC {train_auc:.4f}/{val_auc:.4f} | "
            f"lr {current_lr:.2e}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            epochs_no_improve = 0
            history["best_epoch_auc"] = epoch
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✓ Checkpoint saved (val AUC={val_auc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    # 8. Save training curves
    save_training_curves(history, out_dir)

    # 9. Final test evaluation
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"\nLoaded best checkpoint (epoch {history['best_epoch_auc']}).")
    except Exception as e:
        print(f"Could not load checkpoint: {e}")

    model.eval()
    all_logits, all_targets = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            all_logits.append(model(xb.to(device)).cpu().numpy())
            all_targets.append(yb.numpy())

    y_logits  = np.concatenate(all_logits)
    y_targets = np.concatenate(all_targets)

    val_logits_list, val_targets_list = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            val_logits_list.append(model(xb.to(device)).cpu().numpy())
            val_targets_list.append(yb.numpy())

    best_threshold = find_best_threshold(
        np.concatenate(val_logits_list),
        np.concatenate(val_targets_list),
    )
    print(f"\nBest threshold (val): {best_threshold:.2f}")

    y_prob = 1 / (1 + np.exp(-y_logits))
    y_pred = (y_prob >= best_threshold).astype(int)

    acc = accuracy_score(y_targets, y_pred)
    try:
        auc = roc_auc_score(y_targets, y_prob)
    except ValueError:
        auc = float("nan")

    print(f"\nTest Accuracy : {acc:.4f}")
    print(f"Test ROC-AUC  : {auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_targets, y_pred, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_targets, y_pred))

    # Save metrics summary
    summary_path = out_dir / "metrics_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Version dir      : {out_dir}\n")
        f.write(f"CSV              : {csv_file}\n")
        f.write(f"Horizon          : {horizon}\n")
        f.write(f"Seq len          : {seq_len}\n")
        f.write(f"Pos encoding     : {encoding}\n")
        f.write(f"Attn diag bias   : {attn_diagonal_bias}\n")
        f.write(f"Seed             : {seed}\n")
        f.write(f"Best epoch (AUC) : {history['best_epoch_auc']}\n")
        f.write(f"Best val AUC     : {best_val_auc:.4f}\n")
        f.write(f"Threshold        : {best_threshold:.2f}\n")
        f.write(f"Test Accuracy    : {acc:.4f}\n")
        f.write(f"Test AUC         : {auc:.4f}\n\n")
        f.write(classification_report(y_targets, y_pred, zero_division=0))
    print(f"  Saved metrics summary → {summary_path}")

    # Save config for inference (live_inference, tsne can read attn_diagonal_bias)
    config_path = out_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(
            {
                "seq_len": seq_len,
                "encoding": encoding,
                "attn_diagonal_bias": attn_diagonal_bias,
                "seed": seed,
            },
            f,
            indent=2,
        )
    print(f"  Saved config → {config_path}")
    print(f"\nAll outputs saved to: {out_dir}/")

    # 10. Save a self-attention heatmap from the best model
    try:
        xb0, yb0 = next(iter(val_loader))
        xb0 = xb0.to(device)
        _, attn = model(xb0, return_attn=True)
        if attn is not None:
            heatmap_path = out_dir / "attention_heatmap.png"
            npy_path = out_dir / "attention_weights.npy"
            save_attention_heatmap(attn, heatmap_path, sample_idx=0, head=0)
            np.save(npy_path, attn.detach().cpu().numpy())
            print(f"  Saved attention heatmap → {heatmap_path}")
            print(f"  Saved attention weights → {npy_path}")
    except Exception as e:
        print(f"  Could not save attention heatmap: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bitcoin Up/Down Transformer")
    parser.add_argument("--csv",          type=str,   default="advanced_orderflow_data.csv")
    parser.add_argument("--horizon",      type=int,   default=5)
    parser.add_argument("--seq-len",      type=int,   default=40)
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--batch-size",   type=int,   default=64)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--patience",     type=int,   default=8)
    parser.add_argument("--val-ratio",    type=float, default=0.15)
    parser.add_argument("--outputs-base", type=str,   default="outputs")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="N",
        help="Random seed for reproducibility. Omit for a random seed each run.",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="sinusoidal",
        choices=["sinusoidal", "positional"],
        help="Positional encoding: 'sinusoidal' (fixed) or 'positional' (learnable)",
    )
    parser.add_argument(
        "--attn-diagonal-bias",
        type=float,
        default=0.0,
        metavar="LAMBDA",
        help="Attention bias favoring diagonal: -lambda*|i-j|. Use 0.1-0.5 to reduce collapse. 0=off.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_transformer(
        csv_file=args.csv,
        horizon=args.horizon,
        seq_len=args.seq_len,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        val_ratio=args.val_ratio,
        outputs_base=args.outputs_base,
        seed=args.seed,
        encoding=args.encoding,
        attn_diagonal_bias=args.attn_diagonal_bias,
    )
