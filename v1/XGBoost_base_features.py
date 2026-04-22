import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_base_features(csv_file: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file)

    # Normalize teacher column names → internal names
    rename_map = {}
    if "Mid Price" in df.columns:
        rename_map["Mid Price"] = "mid_price"
    if "Micro Price" in df.columns:
        rename_map["Micro Price"] = "micro_price"
    if "fractional price" in df.columns:
        rename_map["fractional price"] = "fractional_price"
    df = df.rename(columns=rename_map)

    required = [
        "mid_price",
        "micro_price",
        "fractional_price",
        "spread",
        "obi_1",
        "obi_10",
        "obi_diff",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Add time features if Time exists
    
    
    if "Time" in df.columns:
        # Convert mm:ss.s → seconds
        time_str = "00:" + df["Time"].astype(str).str.strip()
        df["time_seconds"] = pd.to_timedelta(time_str, errors="coerce").dt.total_seconds()

        if df["time_seconds"].isna().any():
            bad = df.loc[df["time_seconds"].isna(), "Time"].head(5).tolist()
            raise ValueError(f"Bad Time values: {bad}")

        time_features = ["time_seconds"]
    else:
        time_features = []
    feature_cols = required + time_features
    return df[feature_cols].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="XGBoost using base CSV columns + time")
    parser.add_argument("--csv", type=str, default="advanced_orderflow_data.csv")
    parser.add_argument(
        "--horizon-steps",
        type=int,
        default=6,
        help="How many rows ahead to define the label (6 ≈ 30s at 5s polling).",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    args = parser.parse_args()

    X_df = load_base_features(args.csv)
    mid = X_df["mid_price"].to_numpy()

    # Label: 1 if price goes up after horizon_steps, else 0
    future_mid = np.roll(mid, -args.horizon_steps)
    y = (future_mid > mid).astype(int)

    # Drop last horizon rows (their labels are invalid)
    if args.horizon_steps > 0:
        X_df = X_df.iloc[: -args.horizon_steps].reset_index(drop=True)
        y = y[: -args.horizon_steps]

    X = X_df.to_numpy(dtype=float)

    n = len(y)
    split = int(n * args.train_ratio)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

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
    print(f"XGBoost (base features + time) AUC: {auc:.4f}")

    # Feature importance plot (gain)
    feature_names = list(X_df.columns)
    booster = clf.get_booster()
    score = booster.get_score(importance_type="gain")
    gain = np.array([score.get(f"f{i}", 0.0) for i in range(len(feature_names))], dtype=float)
    order = np.argsort(gain)[::-1]

    plt.figure(figsize=(9, max(4, int(len(feature_names) * 0.6))))
    plt.barh([feature_names[i] for i in order][::-1], gain[order][::-1])
    plt.title("XGBoost Feature Importance (gain) — base CSV features + time")
    plt.xlabel("Total gain")
    plt.tight_layout()

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "xgboost_base_features_plus_time_importance_gain.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved feature importance plot → {out_path}")


if __name__ == "__main__":
    main()