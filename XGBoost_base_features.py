"""
XGBoost baseline (v2) — uses the same time-based loader as the transformer so
the comparison is apples-to-apples.

Key differences from v1 (see v1/XGBoost_base_features.py):
  - Horizon is in **seconds** (matches transformer's --horizon-seconds).
  - Label is built from `future_return` via dataloader.load_clean_r_style,
    including the dead-band filter for ties and the segment-aware forward shift.
  - Features are the full feature_cols from the shared loader (price + multi-
    scale returns + vol + obi_momentum + tod sin/cos).
  - Train/test split is temporal on the resampled grid.
"""

import argparse
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataloader import load_clean_r_style


def main() -> None:
    p = argparse.ArgumentParser(description="XGBoost baseline — shared loader")
    p.add_argument("--csv", type=str, default="data/part2/advanced_orderflow_ws.csv")
    p.add_argument("--step-seconds", type=float, default=0.5)
    p.add_argument("--horizon-seconds", type=float, default=15.0)
    p.add_argument("--backward-seconds", type=float, default=120.0)
    p.add_argument("--gap-seconds", type=float, default=2.0)
    p.add_argument("--deadband-bps", type=float, default=0.5)
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--outputs-base", type=str, default="outputs")
    args = p.parse_args()

    out = load_clean_r_style(
        args.csv,
        step_seconds=args.step_seconds,
        horizon_seconds=args.horizon_seconds,
        backward_seconds=args.backward_seconds,
        gap_seconds=args.gap_seconds,
        deadband_bps=args.deadband_bps,
    )
    print(f"Loaded {len(out.df)} rows, {len(out.feature_cols)} features, "
          f"{len(set(out.segment_ids))} segments.")
    print(f"Horizon {args.horizon_seconds}s, step {args.step_seconds}s, "
          f"deadband {args.deadband_bps} bp.")

    X = out.X
    y = out.y
    n = len(y)
    split = int(n * args.train_ratio)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    print(f"Train {X_train.shape}, Test {X_test.shape}, pos rate train={y_train.mean():.3f} test={y_test.mean():.3f}")

    clf = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    prob = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, prob)
    print(f"XGBoost test AUC: {auc:.4f}")

    # Feature importance plot (gain).
    booster = clf.get_booster()
    scores = booster.get_score(importance_type="gain")
    gain = np.array([scores.get(f"f{i}", 0.0) for i in range(len(out.feature_cols))], dtype=float)
    order = np.argsort(gain)[::-1]

    plt.figure(figsize=(9, max(4, int(len(out.feature_cols) * 0.45))))
    plt.barh([out.feature_cols[i] for i in order][::-1], gain[order][::-1])
    plt.title(
        f"XGBoost Feature Importance (gain)  |  AUC={auc:.4f}  "
        f"|  horizon={args.horizon_seconds}s, step={args.step_seconds}s"
    )
    plt.xlabel("Total gain")
    plt.tight_layout()

    out_dir = Path(args.outputs_base); out_dir.mkdir(exist_ok=True)
    png = out_dir / "xgboost_base_features_plus_time_importance_gain.png"
    plt.savefig(png, dpi=200); plt.close()
    print(f"Saved feature importance plot → {png}")


if __name__ == "__main__":
    main()
