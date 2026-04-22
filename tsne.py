"""
Standalone t-SNE visualisation of Vol-Transformer pooled embeddings.

Rebuilds the val/test windows exactly like training did, loads the best
checkpoint, runs the model up to the pooled representation, and saves a
tsne.png into the run folder. Useful when you want to regenerate the t-SNE
with different knobs (perplexity, n_samples, val vs test) without retraining.

Usage:
    python tsne.py                           # latest outputs/vN/, val set
    python tsne.py --version 7               # use outputs/v7/
    python tsne.py --split test              # plot test windows instead
    python tsne.py --n-samples 3000 --perplexity 50
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from dataloader import load_clean_r_style, build_sequences
from transformer import VolTransformer, save_tsne_plot


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


def rebuild_split_loader(
    cfg: dict,
    split: str,
    scaler_dict: dict,
    batch_size: int,
):
    """Recreate val or test DataLoader using the exact config of the saved run."""
    data_cfg = cfg["data"]
    loaded = load_clean_r_style(
        cfg["csv_file"],
        step_seconds=data_cfg["step_seconds"],
        horizon_seconds=data_cfg["horizon_seconds"],
        backward_seconds=data_cfg["backward_seconds"],
        gap_seconds=data_cfg["gap_seconds"],
        deadband_bps=data_cfg["deadband_bps"],
        vol_fast_seconds=data_cfg.get("vol_fast_seconds", 5.0),
        vol_med_seconds=data_cfg.get("vol_med_seconds", 30.0),
        vol_of_vol_seconds=data_cfg.get("vol_of_vol_seconds", 30.0),
    )

    n = len(loaded.X)
    n_test = int(n * cfg["test_ratio"])
    n_val = int(n * cfg["val_ratio"])
    n_train = n - n_val - n_test

    if split == "val":
        sl = slice(n_train, n_train + n_val)
        stride = 1
    elif split == "test":
        sl = slice(n_train + n_val, None)
        stride = cfg.get("test_stride") or max(
            1, int(round(data_cfg["horizon_seconds"] / data_cfg["step_seconds"]))
        )
    else:
        raise ValueError(f"split must be 'val' or 'test', got {split!r}")

    X = loaded.X[sl]; y = loaded.y[sl]; seg = loaded.segment_ids[sl]
    X = scaler_dict["scaler"].transform(X)
    Xs, ys = build_sequences(X, y, seg, seq_len=cfg["seq_len"], stride=stride)
    ds = TensorDataset(torch.from_numpy(Xs), torch.from_numpy(ys))
    return DataLoader(ds, batch_size=batch_size, shuffle=False), loaded


def main() -> None:
    p = argparse.ArgumentParser(description="t-SNE of Vol-Transformer pooled embeddings")
    p.add_argument("--outputs-base", type=str, default="outputs")
    p.add_argument("--version", type=int, default=None, help="Run version (default: latest)")
    p.add_argument("--split", type=str, default="val", choices=["val", "test"])
    p.add_argument("--n-samples", type=int, default=1500)
    p.add_argument("--perplexity", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args()

    run_dir = (Path(args.outputs_base) / f"v{args.version}"
               if args.version is not None else find_latest_version(args.outputs_base))
    print(f"Loading from: {run_dir}")

    ckpt = run_dir / "transformer_best.pt"
    scaler_pkl = run_dir / "scaler.pkl"
    config_path = run_dir / "config.json"
    for p_ in (ckpt, scaler_pkl, config_path):
        if not p_.exists():
            raise FileNotFoundError(f"Missing: {p_}")

    with open(config_path) as f:
        cfg = json.load(f)
    with open(scaler_pkl, "rb") as f:
        scaler_dict = pickle.load(f)

    loader, loaded = rebuild_split_loader(cfg, args.split, scaler_dict, args.batch_size)

    # Rebuild model from config.
    vol_indices = [loaded.feature_cols.index(c) for c in cfg["vol_feature_cols"]]
    mcfg = cfg["model"]
    model = VolTransformer(
        input_dim=len(loaded.feature_cols),
        vol_indices=vol_indices,
        d_model=mcfg["d_model"], nhead=mcfg["nhead"], num_layers=mcfg["num_layers"],
        dim_feedforward=mcfg["dim_feedforward"], dropout=mcfg["dropout"],
        pos_encoding=mcfg["pos_encoding"],
        attn_diagonal_bias=mcfg.get("attn_diagonal_bias", 0.0),
        conv_kernel=mcfg.get("conv_kernel", 3),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()

    out_path = run_dir / f"tsne_{args.split}.png"
    save_tsne_plot(
        model=model,
        loader=loader,
        device=device,
        out_path=out_path,
        n_samples=args.n_samples,
        perplexity=args.perplexity,
        title=f"t-SNE of pooled embeddings  ({run_dir.name}, {args.split} set)",
    )


if __name__ == "__main__":
    main()
