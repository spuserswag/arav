"""
Hyperparameter sweep for the Vol-Transformer.

Runs train_transformer() across a set of configurations (grid or random),
captures each run's stdout to a log file, reads the resulting config.json
to harvest metrics, and prints a leaderboard sorted by best validation AUC.

Also writes outputs/sweep_<timestamp>.csv with one row per run.

Presets (choose with --preset):
  regularize   — small fixed model, sweep dropout × weight_decay × attn_diag_bias
                 (targeted at the overfitting we saw in v10)
  capacity     — fixed regularization, sweep d_model × num_layers × seq_len
  horizon      — sweep horizon_seconds × seq_len (does the signal get cleaner?)
  combined     — smaller cross-product of the three above
  random       — random search; --n-trials controls how many configs to draw
  minimal      — 3-config smoke test (use this first to verify the sweep runs)

Each config can be replicated across seeds (--seeds-per-config) — report the
mean val AUC across seeds, so you're not chasing seed noise.

Typical usage:
    python sweep.py --preset minimal
    python sweep.py --preset regularize
    python sweep.py --preset random --n-trials 20 --seeds-per-config 2
    python sweep.py --preset regularize --epochs-per-run 20 --patience-per-run 4
"""

from __future__ import annotations

import argparse
import csv
import io
import itertools
import json
import random
import sys
import time
import traceback
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

# Import lazily so `--dry-run` works even without torch installed.
def _import_train():
    from transformer import train_transformer  # noqa: E402
    return train_transformer


# ---------------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------------

# Fixed-across-all-presets: data knobs and a short-ish training budget.
_FIXED_DEFAULTS: Dict[str, Any] = {
    "csv_file": "data/part2/advanced_orderflow_ws.csv",
    "step_seconds": 0.5,
    "horizon_seconds": 15.0,
    "backward_seconds": 120.0,
    "gap_seconds": 2.0,
    "deadband_bps": 0.5,
    "vol_fast_seconds": 5.0,
    "vol_med_seconds": 30.0,
    "vol_of_vol_seconds": 30.0,
    "seq_len": 30,
    "batch_size": 128,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "d_model": 96,
    "nhead": 6,
    "num_layers": 3,
    "dim_feedforward": 384,
    "dropout": 0.2,
    "pos_encoding": "sinusoidal",
    "attn_diagonal_bias": 0.0,
    "conv_kernel": 3,
}


def _grid(axes: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Cartesian product of named lists → list of overrides."""
    keys = list(axes.keys())
    out: List[Dict[str, Any]] = []
    for combo in itertools.product(*[axes[k] for k in keys]):
        out.append(dict(zip(keys, combo)))
    return out


def _preset_minimal() -> List[Dict[str, Any]]:
    # 3 tiny fast configs to smoke-test the pipeline end-to-end.
    return [
        {"d_model": 48, "num_layers": 1, "dim_feedforward": 128, "dropout": 0.3,
         "nhead": 4, "weight_decay": 1e-3},
        {"d_model": 64, "num_layers": 2, "dim_feedforward": 256, "dropout": 0.3,
         "nhead": 4, "weight_decay": 1e-3},
        {"d_model": 96, "num_layers": 2, "dim_feedforward": 384, "dropout": 0.3,
         "nhead": 6, "weight_decay": 1e-3, "attn_diagonal_bias": 0.1},
    ]


def _preset_regularize() -> List[Dict[str, Any]]:
    # Small model (fight overfit). Sweep regularization strength.
    base = {
        "d_model": 64, "num_layers": 2, "dim_feedforward": 192,
        "nhead": 4, "conv_kernel": 3,
    }
    axes = {
        "dropout":            [0.2, 0.35, 0.5],
        "weight_decay":       [1e-4, 1e-3, 3e-3],
        "attn_diagonal_bias": [0.0, 0.1, 0.2],
    }
    return [{**base, **o} for o in _grid(axes)]


def _preset_capacity() -> List[Dict[str, Any]]:
    # Fix moderate regularization, sweep capacity + context length.
    base = {"dropout": 0.3, "weight_decay": 1e-3, "attn_diagonal_bias": 0.1}
    axes = {
        "d_model":    [48, 64, 96],
        "num_layers": [1, 2, 3],
        "seq_len":    [20, 30, 60],
    }
    configs = []
    for o in _grid(axes):
        # nhead must divide d_model; pick the largest valid head count up to 6.
        nh = 6 if o["d_model"] % 6 == 0 else (4 if o["d_model"] % 4 == 0 else 2)
        o["nhead"] = nh
        o["dim_feedforward"] = o["d_model"] * 4
        configs.append({**base, **o})
    return configs


def _preset_horizon() -> List[Dict[str, Any]]:
    # Does longer horizon reveal cleaner signal?
    base = {"d_model": 64, "num_layers": 2, "nhead": 4, "dim_feedforward": 192,
            "dropout": 0.3, "weight_decay": 1e-3, "attn_diagonal_bias": 0.1}
    axes = {
        "horizon_seconds": [15.0, 30.0, 60.0],
        "seq_len":         [30, 60],
    }
    return [{**base, **o} for o in _grid(axes)]


def _preset_combined() -> List[Dict[str, Any]]:
    # A small, curated cross-cut of the three above (~18 runs).
    base = {"nhead": 4, "conv_kernel": 3}
    picks = [
        # (d_model, num_layers, dim_ff, dropout, wd, adb, horizon, seq_len)
        (48, 1, 192, 0.35, 1e-3, 0.1, 15.0, 30),
        (48, 2, 192, 0.35, 1e-3, 0.1, 15.0, 30),
        (64, 2, 256, 0.30, 1e-3, 0.1, 15.0, 30),
        (64, 2, 256, 0.50, 3e-3, 0.2, 15.0, 30),
        (64, 2, 256, 0.30, 1e-3, 0.0, 15.0, 30),
        (96, 2, 384, 0.30, 1e-3, 0.1, 15.0, 30),
        (96, 3, 384, 0.20, 1e-4, 0.0, 15.0, 30),   # ≈ v10 baseline for comparison
        (64, 2, 256, 0.30, 1e-3, 0.1, 30.0, 30),
        (64, 2, 256, 0.30, 1e-3, 0.1, 30.0, 60),
        (64, 2, 256, 0.30, 1e-3, 0.1, 60.0, 60),
        (64, 3, 256, 0.40, 3e-3, 0.2, 15.0, 60),
        (96, 2, 384, 0.40, 3e-3, 0.2, 15.0, 60),
    ]
    out = []
    for d_model, L, dff, drop, wd, adb, h_sec, sl in picks:
        nh = 6 if d_model % 6 == 0 else (4 if d_model % 4 == 0 else 2)
        out.append({
            **base,
            "d_model": d_model, "num_layers": L, "dim_feedforward": dff,
            "dropout": drop, "weight_decay": wd, "attn_diagonal_bias": adb,
            "horizon_seconds": h_sec, "seq_len": sl, "nhead": nh,
        })
    return out


def _sample_random(rng: random.Random) -> Dict[str, Any]:
    d_model = rng.choice([48, 64, 96, 128])
    nh = 6 if d_model % 6 == 0 else (4 if d_model % 4 == 0 else 2)
    return {
        "d_model": d_model, "nhead": nh,
        "num_layers": rng.choice([1, 2, 3]),
        "dim_feedforward": d_model * rng.choice([2, 4]),
        "dropout": rng.choice([0.2, 0.3, 0.4, 0.5]),
        "weight_decay": 10 ** rng.uniform(-4, -2.5),
        "attn_diagonal_bias": rng.choice([0.0, 0.05, 0.1, 0.2]),
        "conv_kernel": rng.choice([3, 5]),
        "seq_len": rng.choice([20, 30, 40, 60]),
        "horizon_seconds": rng.choice([15.0, 30.0, 60.0]),
        "lr": 10 ** rng.uniform(-4, -3),
    }


def _preset_random(n_trials: int, rng: random.Random) -> List[Dict[str, Any]]:
    return [_sample_random(rng) for _ in range(n_trials)]


# ---------------------------------------------------------------------------
# Running one trial
# ---------------------------------------------------------------------------

def _make_run_config(
    override: Dict[str, Any],
    seed: int,
    epochs: int,
    patience: int,
    outputs_base: str,
) -> Dict[str, Any]:
    """Merge fixed defaults + override + per-run knobs into kwargs for train_transformer."""
    cfg = {**_FIXED_DEFAULTS, **override}
    cfg["seed"] = seed
    cfg["n_epochs"] = epochs
    cfg["patience"] = patience
    cfg["outputs_base"] = outputs_base
    return cfg


def _run_one(
    override: Dict[str, Any],
    seed: int,
    epochs: int,
    patience: int,
    outputs_base: str,
    log_dir: Path,
    trial_tag: str,
) -> Dict[str, Any]:
    """Run a single configuration; return a result dict (even on failure)."""
    train_transformer = _import_train()
    cfg = _make_run_config(override, seed, epochs, patience, outputs_base)

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{trial_tag}.log"

    t0 = time.time()
    result: Dict[str, Any] = {"trial": trial_tag, "seed": seed, "override": override}
    try:
        with open(log_path, "w") as f, redirect_stdout(f):
            run_dir = train_transformer(**cfg)
        elapsed = time.time() - t0
        # Harvest config.json from the produced run folder.
        with open(Path(run_dir) / "config.json") as f:
            run_cfg = json.load(f)
        results = run_cfg.get("results", {})
        result.update({
            "status": "ok",
            "run_dir": str(run_dir),
            "elapsed_s": round(elapsed, 1),
            "best_val_auc":  results.get("best_val_auc"),
            "test_auc":      results.get("test_auc"),
            "test_accuracy": results.get("test_accuracy"),
            "threshold":     results.get("threshold"),
            "best_epoch":    results.get("best_epoch_auc"),
            "n_params":      results.get("n_params"),
        })
    except Exception as e:
        result.update({
            "status": "error",
            "error": f"{type(e).__name__}: {e}",
            "elapsed_s": round(time.time() - t0, 1),
            "log_path": str(log_path),
        })
        # Also dump the traceback to the log.
        with open(log_path, "a") as f:
            f.write("\n" + traceback.format_exc())
    return result


# ---------------------------------------------------------------------------
# Leaderboard / CSV
# ---------------------------------------------------------------------------

def _flatten(result: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested `override` dict into top-level columns for CSV."""
    row = {k: v for k, v in result.items() if k != "override"}
    for k, v in result.get("override", {}).items():
        row[f"ov_{k}"] = v
    return row


def _aggregate_by_config(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group by override dict; report mean ± std val AUC across seeds."""
    groups: Dict[str, Dict[str, Any]] = {}
    for r in results:
        if r.get("status") != "ok":
            continue
        key = json.dumps(r["override"], sort_keys=True, default=str)
        g = groups.setdefault(key, {"override": r["override"], "seeds": [], "val_aucs": [],
                                    "test_aucs": [], "n_params": r.get("n_params")})
        g["seeds"].append(r["seed"])
        if r.get("best_val_auc") is not None:
            g["val_aucs"].append(r["best_val_auc"])
        if r.get("test_auc") is not None:
            g["test_aucs"].append(r["test_auc"])

    agg: List[Dict[str, Any]] = []
    for g in groups.values():
        v = np.asarray(g["val_aucs"], dtype=float) if g["val_aucs"] else np.array([])
        t = np.asarray(g["test_aucs"], dtype=float) if g["test_aucs"] else np.array([])
        agg.append({
            **g["override"],
            "n_seeds": len(g["seeds"]),
            "val_auc_mean":  float(v.mean()) if v.size else float("nan"),
            "val_auc_std":   float(v.std())  if v.size > 1 else 0.0,
            "test_auc_mean": float(t.mean()) if t.size else float("nan"),
            "test_auc_std":  float(t.std())  if t.size > 1 else 0.0,
            "n_params":      g["n_params"],
        })
    agg.sort(key=lambda r: (r["val_auc_mean"] is None, -(r["val_auc_mean"] or -1)))
    return agg


def _print_leaderboard(agg: List[Dict[str, Any]], top: int = 10) -> None:
    if not agg:
        print("No successful runs to rank.")
        return
    # Build a tidy table. Pick a stable column set that's easy to scan.
    cols_head = ["rank", "val_auc", "test_auc", "n_seeds", "n_params"]
    cols_ov = ["d_model", "num_layers", "nhead", "dim_feedforward", "dropout",
               "weight_decay", "attn_diagonal_bias", "seq_len", "horizon_seconds",
               "lr", "conv_kernel"]
    # Restrict override columns to those present in at least one config.
    cols_ov = [c for c in cols_ov if any(c in r for r in agg)]

    headers = cols_head + cols_ov

    def _fmt(v: Any) -> str:
        if v is None:
            return "-"
        if isinstance(v, float):
            if abs(v) < 1e-3:
                return f"{v:.2e}"
            return f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}"
        return str(v)

    rows = []
    for i, r in enumerate(agg[:top], 1):
        row = {
            "rank": i,
            "val_auc": f"{r['val_auc_mean']:.4f}" + (
                f"±{r['val_auc_std']:.3f}" if r["n_seeds"] > 1 else ""
            ),
            "test_auc": f"{r['test_auc_mean']:.4f}" + (
                f"±{r['test_auc_std']:.3f}" if r["n_seeds"] > 1 else ""
            ),
            "n_seeds": r["n_seeds"],
            "n_params": r.get("n_params") or "-",
        }
        for c in cols_ov:
            row[c] = _fmt(r.get(c))
        rows.append(row)

    widths = {h: max(len(h), max((len(str(r.get(h, ""))) for r in rows), default=0)) for h in headers}
    print("\n" + " | ".join(h.ljust(widths[h]) for h in headers))
    print("-+-".join("-" * widths[h] for h in headers))
    for r in rows:
        print(" | ".join(str(r.get(h, "")).ljust(widths[h]) for h in headers))


def _save_csv(results: List[Dict[str, Any]], path: Path) -> None:
    rows = [_flatten(r) for r in results]
    all_keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in all_keys:
                all_keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nSaved per-run results → {path}")


def _save_leaderboard_csv(agg: List[Dict[str, Any]], path: Path) -> None:
    if not agg:
        return
    keys: List[str] = []
    for r in agg:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in agg:
            w.writerow(r)
    print(f"Saved leaderboard  → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_configs(preset: str, n_trials: int, rng: random.Random) -> List[Dict[str, Any]]:
    if preset == "minimal":
        return _preset_minimal()
    if preset == "regularize":
        return _preset_regularize()
    if preset == "capacity":
        return _preset_capacity()
    if preset == "horizon":
        return _preset_horizon()
    if preset == "combined":
        return _preset_combined()
    if preset == "random":
        return _preset_random(n_trials, rng)
    raise ValueError(f"Unknown preset: {preset}")


def main() -> None:
    p = argparse.ArgumentParser(description="Hyperparameter sweep for Vol-Transformer.")
    p.add_argument("--preset", type=str, default="regularize",
                   choices=["minimal", "regularize", "capacity", "horizon", "combined", "random"])
    p.add_argument("--n-trials", type=int, default=20,
                   help="Only used for --preset random.")
    p.add_argument("--seeds-per-config", type=int, default=1)
    p.add_argument("--seed-base", type=int, default=42)
    p.add_argument("--epochs-per-run", type=int, default=25)
    p.add_argument("--patience-per-run", type=int, default=5)
    p.add_argument("--outputs-base", type=str, default="outputs")
    p.add_argument("--csv", type=str, default=None,
                   help="Override CSV path for every run.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the list of configs and exit.")
    p.add_argument("--top", type=int, default=10,
                   help="How many rows of the leaderboard to print.")
    args = p.parse_args()

    # Build configs.
    rng = random.Random(args.seed_base)
    overrides = _build_configs(args.preset, args.n_trials, rng)
    if args.csv:
        for o in overrides:
            o["csv_file"] = args.csv
    seeds = [args.seed_base + i for i in range(args.seeds_per_config)]
    total_runs = len(overrides) * len(seeds)

    print(f"Preset: {args.preset}")
    print(f"Configs: {len(overrides)}  ×  seeds: {len(seeds)}  =  {total_runs} runs")
    print(f"Budget per run: epochs={args.epochs_per_run}, patience={args.patience_per_run}")

    if args.dry_run:
        for i, o in enumerate(overrides):
            print(f"  [{i:02d}] {o}")
        return

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.outputs_base) / f"sweep_{timestamp}_logs"
    results: List[Dict[str, Any]] = []

    for i, o in enumerate(overrides):
        for s in seeds:
            tag = f"t{i:03d}_s{s}"
            run_idx = len(results) + 1
            # Compact one-liner of the override so the header is human-scannable.
            short = ", ".join(f"{k}={v}" for k, v in sorted(o.items())
                              if k not in ("csv_file",))
            print(f"\n[{run_idx}/{total_runs}] seed={s}  {short}")
            r = _run_one(
                o, seed=s,
                epochs=args.epochs_per_run, patience=args.patience_per_run,
                outputs_base=args.outputs_base, log_dir=log_dir, trial_tag=tag,
            )
            if r["status"] == "ok":
                print(f"   → val AUC {r['best_val_auc']:.4f}  "
                      f"test AUC {r.get('test_auc'):.4f}  "
                      f"(epoch {r.get('best_epoch')}, {r['elapsed_s']}s, "
                      f"{r.get('n_params'):,} params)  "
                      f"run={Path(r['run_dir']).name}")
            else:
                print(f"   × FAILED: {r['error']}  (log: {r['log_path']})")
            results.append(r)

    # Aggregate + report.
    agg = _aggregate_by_config(results)
    _print_leaderboard(agg, top=args.top)

    out_csv = Path(args.outputs_base) / f"sweep_{timestamp}.csv"
    out_lb = Path(args.outputs_base) / f"sweep_{timestamp}_leaderboard.csv"
    _save_csv(results, out_csv)
    _save_leaderboard_csv(agg, out_lb)
    print(f"Per-run logs       → {log_dir}/")


if __name__ == "__main__":
    main()
