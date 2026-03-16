import time
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

import ccxt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


CSV_FILE = "advanced_orderflow_data.csv"


def collect_orderflow_data(
    symbol: str = "BTC/USDT",
    poll_interval: int = 5,
    n_iterations: int = 15000,
    csv_file: str = CSV_FILE,
) -> None:
    """
    Collect order book snapshots from BinanceUS using ccxt and write
    microstructure features to a CSV file.

    Parameters
    ----------
    symbol : str
        Trading pair symbol, e.g. 'BTC/USDT'.
    poll_interval : int
        Seconds between API calls.
    n_iterations : int
        Number of snapshots to collect.
    csv_file : str
        Output CSV file path.
    """
    exchange = ccxt.binanceus()

    print(
        f"Starting advanced data collection for {symbol}. "
        f"Polling every {poll_interval}s for {n_iterations} iterations."
    )

    for i in range(1, n_iterations + 1):
        try:
            ob = exchange.fetch_order_book(symbol)

            bids_df = pd.DataFrame(ob["bids"], columns=["price", "amount"])
            asks_df = pd.DataFrame(ob["asks"], columns=["price", "amount"])

            if bids_df.empty or asks_df.empty:
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    "Empty order book, skipping."
                )
                time.sleep(poll_interval)
                continue

            best_bid = bids_df["price"].iloc[0]
            best_ask = asks_df["price"].iloc[0]
            bid_vol_1 = bids_df["amount"].iloc[0]

            ask_vol_1 = asks_df["amount"].iloc[0]

            mid_price = (best_bid + best_ask) / 2
            spread = best_ask - best_bid
            fractional_price = mid_price - int(mid_price)

            micro_price = (best_bid * ask_vol_1 + best_ask * bid_vol_1) / (
                bid_vol_1 + ask_vol_1
            )

            obi_1 = (bid_vol_1 - ask_vol_1) / (bid_vol_1 + ask_vol_1)

            top_bids_10 = bids_df.head(10)
            top_asks_10 = asks_df.head(10)
            bid_vol_10 = top_bids_10["amount"].sum()
            ask_vol_10 = top_asks_10["amount"].sum()
            depth_10 = bid_vol_10 + ask_vol_10
            obi_10 = (bid_vol_10 - ask_vol_10) / (bid_vol_10 + ask_vol_10)

            obi_diff = obi_10 - obi_1
            depth_imbalance_10 = (bid_vol_10 - ask_vol_10) / depth_10 if depth_10 != 0 else 0.0

            current_time = datetime.now()

            row_data = pd.DataFrame(
                [
                    {
                        "timestamp": current_time,
                        "mid_price": mid_price,
                        "micro_price": micro_price,
                        "fractional_price": fractional_price,
                        "spread": spread,
                        "obi_1": obi_1,
                        "obi_10": obi_10,
                        "obi_diff": obi_diff,
                        "bid_vol_10": bid_vol_10,
                        "ask_vol_10": ask_vol_10,
                        "depth_10": depth_10,
                        "depth_imbalance_10": depth_imbalance_10,
                    }
                ]
            )

            header = not pd.io.common.file_exists(csv_file)
            row_data.to_csv(csv_file, mode="a", index=False, header=header)

            print(
                f"[{current_time.strftime('%H:%M:%S')}] "
                f"Iter {i}/{n_iterations} | "
                f"Mid: {mid_price:.2f} | OBI(10): {obi_10:+.2f} | Spread: {spread:.2f}"
            )

        except Exception as e:
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"API error ({type(e).__name__}: {e}). Skipping iteration."
            )

        time.sleep(poll_interval)

    print("Data collection complete.")


def load_and_prepare_data(csv_file: str = CSV_FILE, horizon: int = 5) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    """
    Load collected data, construct features and target.

    Parameters
    ----------
    csv_file : str
        Path to CSV with collected microstructure features.
    horizon : int
        Number of steps ahead to define the prediction target.

    Returns
    -------
    df : pd.DataFrame
        Data with features and targets.
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Binary classification target (up vs. non-up).
    feature_cols : list of str
        Names of feature columns used in X.
    """
    df = pd.read_csv(csv_file, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["future_mid"] = df["mid_price"].shift(-horizon)
    df["future_ret"] = (df["future_mid"] - df["mid_price"]) / df["mid_price"]
    df["y"] = (df["future_ret"] > 0).astype(int)
    df = df.dropna(subset=["future_mid", "future_ret", "y"]).reset_index(drop=True)

    # Start from the full intended feature list but keep only those
    # that actually exist in the current CSV (for backward compatibility
    # with files collected before new features were added).
    base_features_all = [
        "mid_price",
        "micro_price",
        "fractional_price",
        "spread",
        "obi_1",
        "obi_10",
        "obi_diff",
        "bid_vol_10",
        "ask_vol_10",
        "depth_10",
        "depth_imbalance_10",
    ]
    base_features = [c for c in base_features_all if c in df.columns]

    # Returns and realized-volatility style measures.
    # Use relatively short windows so the code still works
    # with modest amounts of data.
    df["mid_ret"] = df["mid_price"].pct_change()
    df["mid_ret_abs"] = df["mid_ret"].abs()
    df["mid_ret_roll10_std"] = df["mid_ret"].rolling(window=10).std()
    df["mid_ret_roll20_std"] = df["mid_ret"].rolling(window=20).std()
    df["mid_ret_roll10_sum"] = df["mid_ret"].rolling(window=10).sum()

    for col in base_features:
        df[f"{col}_lag1"] = df[col].shift(1)
        df[f"{col}_roll5_mean"] = df[col].rolling(window=5).mean()
        df[f"{col}_roll5_std"] = df[col].rolling(window=5).std()

    df = df.dropna().reset_index(drop=True)

    feature_cols = [c for c in df.columns if any(c.startswith(b) for b in base_features)]
    feature_cols += [
        c for c in df.columns if any(c.startswith(b + "_lag1") for b in base_features)
    ]
    feature_cols += [
        c for c in df.columns if any(c.startswith(b + "_roll5") for b in base_features)
    ]
    feature_cols = sorted(list(set(feature_cols)))

    X = df[feature_cols].values
    y = df["y"].values

    return df, X, y, feature_cols


def train_test_split_time_series(
    X: np.ndarray, y: np.ndarray, train_ratio: float = 0.7
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple time-based train/test split.
    """
    n = len(y)
    split_idx = int(n * train_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    return X_train, X_test, y_train, y_test


def run_logistic_regression(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """
    Fit and evaluate an L2-regularized logistic regression model.
    """
    if X_train.size == 0 or X_test.size == 0:
        print("Skipping logistic regression: no samples available.")
        return
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        n_jobs=-1,
    )

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    print("=== Logistic Regression (L2) ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))
    print("\nClassification report:\n", classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))


def run_random_forest(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """
    Fit and evaluate a Random Forest classifier.
    """
    if X_train.size == 0 or X_test.size == 0:
        print("Skipping random forest: no samples available.")
        return

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=10,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("=== Random Forest ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))
    print("\nClassification report:\n", classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))


def time_series_cross_validation(
    X_train: np.ndarray, y_train: np.ndarray, n_splits: int = 5
) -> None:
    """
    Time-series cross-validation for logistic regression as an example.
    """
    # If there is only one class in the training data, CV is not meaningful.
    unique_classes = np.unique(y_train)
    if unique_classes.size < 2:
        print(
            "Skipping time-series CV: training data contains only one class "
            f"{unique_classes}."
        )
        return

    tscv = TimeSeriesSplit(n_splits=n_splits)
    scaler = StandardScaler()
    cv_acc: List[float] = []

    print("=== Time Series Cross-Validation (Logistic Regression) ===")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        # Some folds may still have a single class due to small sample size.
        if np.unique(y_tr).size < 2 or np.unique(y_val).size < 2:
            print(f"Fold {fold}: skipped (only one class in this fold).")
            continue

        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        model = LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, n_jobs=-1
        )
        model.fit(X_tr_s, y_tr)
        y_val_pred = model.predict(X_val_s)
        acc = accuracy_score(y_val, y_val_pred)
        cv_acc.append(acc)
        print(f"Fold {fold} accuracy: {acc:.3f}")

    print("Mean CV accuracy:", float(np.mean(cv_acc)))


def main() -> None:
    """
    Entry point: assumes CSV data has already been collected, then
    builds features, trains models, and reports performance.

    If you want to collect fresh data first, call collect_orderflow_data()
    before running the modeling steps.
    """
    # Uncomment this line if you want to (re)collect data from the exchange:
    # collect_orderflow_data(symbol="BTC/USDT", poll_interval=5, n_iterations=15000)

    df, X, y, feature_cols = load_and_prepare_data(CSV_FILE, horizon=5)
    print(f"Loaded {len(df)} rows with {len(feature_cols)} features.")

    if len(df) == 0 or X.size == 0:
        print(
            "No usable rows after feature/target construction. "
            "You may need to collect more data first."
        )
        return

    X_train, X_test, y_train, y_test = train_test_split_time_series(
        X, y, train_ratio=0.7
    )
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

    if X_train.size == 0 or X_test.size == 0:
        print(
            "Train/test split resulted in empty sets. "
            "Collect more data before modeling."
        )
        return

    time_series_cross_validation(X_train, y_train, n_splits=5)
    run_logistic_regression(X_train, X_test, y_train, y_test)
    run_random_forest(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()

