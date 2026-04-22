"""
tSNE visualisation of transformer encoder representations.

Saves a PNG into the latest (or specified) outputs/vN/ folder.
Usage:
    python3 tsne_plot.py                        # auto-detect latest run
    python3 tsne_plot.py --version 3            # use outputs/v3/
    python3 tsne_plot.py --outputs-base runs    # custom base folder
"""

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from transformer import TimeSeriesTransformer, make_sequence_dataset
from initial import load_and_prepare_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_latest_version(base: str = "outputs") -> Path:
    base_path = Path(base)
    dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("v")]
    if not dirs:
        raise FileNotFoundError(f"No versioned run folders found in '{base}'.")
    versions = []
    for d in dirs:
        try:
            versions.append((int(d.name[1:]), d))
        except ValueError:
            pass
    versions.sort(key=lambda x: x[0])
    return versions[-1][1]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def plot_tsne(
    outputs_base: str = "outputs",
    version: int = None,
    csv_file: str = "advanced_orderflow_data.csv",
    seq_len: int = 40,
    horizon: int = 5,
    n_samples: int = 1500,
    perplexity: int = 30,
) -> None:

    # Locate version folder
    if version is not None:
        run_dir = Path(outputs_base) / f"v{version}"
    else:
        run_dir = find_latest_version(outputs_base)

    print(f"Loading from: {run_dir}")

    checkpoint_path = run_dir / "transformer_best.pt"
    scaler_path     = run_dir / "scaler.pkl"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint at {checkpoint_path}")

    # Load scaler if available (use it the same way training did)
    scaler = None
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        print("  Loaded scaler from run folder.")
    else:
        print("  Warning: no scaler.pkl found — using raw (unscaled) features.")

    # Load data
    df, X, y, feat_cols = load_and_prepare_data(csv_file=csv_file, horizon=horizon)
    print(f"Loaded {len(df)} rows, {len(feat_cols)} features.")

    # Apply same scaler used during training
    if scaler is not None:
        X = scaler.transform(X)

    # Build sequences
    X_seq, y_seq = make_sequence_dataset(X, y, seq_len)

    # Subsample (tSNE is O(n²))
    if len(X_seq) > n_samples:
        idx = np.random.default_rng(42).choice(len(X_seq), n_samples, replace=False)
        idx.sort()
        X_seq = X_seq[idx]
        y_seq = y_seq[idx]

    print(f"Running tSNE on {len(X_seq)} samples …")

    X_torch = torch.tensor(X_seq, dtype=torch.float)

    # Load model (read config for encoding and attn_diagonal_bias)
    import json
    d_in = X_seq.shape[-1]
    config = {}
    config_path = run_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    model = TimeSeriesTransformer(
        input_dim=d_in,
        pos_encoding=config.get("encoding", "sinusoidal"),
        attn_diagonal_bias=config.get("attn_diagonal_bias", 0.0),
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    # Extract representations from encoder
    with torch.no_grad():
        h = model.input_proj(X_torch)
        h = model.pos_enc(h)
        h = model.encoder(h)
        feats = h[:, -1, :]           # (N, d_model) — last timestep

    feats_np = feats.cpu().numpy()

    # tSNE
    tsne   = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    X_2d   = tsne.fit_transform(feats_np)

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"t-SNE of Transformer Encoder Representations  ({run_dir.name})",
        fontsize=13, fontweight="bold",
    )

    # Left: colour by true label
    ax = axes[0]
    sc = ax.scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=y_seq, cmap="coolwarm", alpha=0.65, s=18, linewidths=0,
    )
    plt.colorbar(sc, ax=ax, label="True label (0=down, 1=up)")
    ax.set_title("Coloured by True Label")
    ax.set_xlabel("tSNE-1");  ax.set_ylabel("tSNE-2")
    ax.grid(True, alpha=0.2)

    # Right: colour by model confidence (sigmoid of logit)
    with torch.no_grad():
        logits = model.cls_head(feats).squeeze(-1).cpu().numpy()
    probs = 1 / (1 + np.exp(-logits))

    ax = axes[1]
    sc2 = ax.scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=probs, cmap="RdYlGn", alpha=0.65, s=18, linewidths=0,
        vmin=0, vmax=1,
    )
    plt.colorbar(sc2, ax=ax, label="Model confidence (P(up))")
    ax.set_title("Coloured by Model Confidence")
    ax.set_xlabel("tSNE-1");  ax.set_ylabel("tSNE-2")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()

    png_path = run_dir / "tsne_plot.png"
    plt.savefig(png_path, format="png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved tSNE plot → {png_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="tSNE of transformer encoder reps.")
    parser.add_argument("--outputs-base", type=str,  default="outputs")
    parser.add_argument("--version",      type=int,  default=None,   help="Run version (default: latest)")
    parser.add_argument("--csv",          type=str,  default="advanced_orderflow_data.csv")
    parser.add_argument("--seq-len",      type=int,  default=40)
    parser.add_argument("--horizon",      type=int,  default=5)
    parser.add_argument("--n-samples",    type=int,  default=1500)
    parser.add_argument("--perplexity",   type=int,  default=30)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_tsne(
        outputs_base=args.outputs_base,
        version=args.version,
        csv_file=args.csv,
        seq_len=args.seq_len,
        horizon=args.horizon,
        n_samples=args.n_samples,
        perplexity=args.perplexity,
    )
