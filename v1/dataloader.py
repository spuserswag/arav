"""
Shared data loader used across training, evaluation, and visualization.

Goal: keep feature engineering, feature ordering, and label definition identical
everywhere, so saved scalers/checkpoints remain compatible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LoadedData:
    df: pd.DataFrame
    X: np.ndarray
    y: np.ndarray
    feature_cols: List[str]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    if "Mid Price" in df.columns:
        rename_map["Mid Price"] = "mid_price"
    if "Micro Price" in df.columns:
        rename_map["Micro Price"] = "micro_price"
    if "fractional price" in df.columns:
        rename_map["fractional price"] = "fractional_price"
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def load_clean_r_style(csv_file: str) -> LoadedData:
    """
    Mirror the R cleaning logic (see `transformer.py` history / `stuff/clean.R`).

    Returns a model-ready dataframe, numeric feature matrix X, binary label y,
    and the feature column names in the exact order used to build X.
    """
    df = pd.read_csv(csv_file)
    df = _normalize_columns(df)

    if "Time" not in df.columns:
        raise ValueError("Expected a 'Time' column in the CSV.")

    # Parse Time: supports both MM:SS.s (e.g. "36:50.6") and HH:MM:SS (e.g. "03:49:45").
    time_str = df["Time"].astype(str)
    time_split = time_str.str.split(":", expand=True)
    is_hms = time_str.str.count(":") == 2  # HH:MM:SS has 2 colons
    mmss_secs = (
        pd.to_numeric(time_split[0], errors="coerce").fillna(0) * 60
        + pd.to_numeric(time_split[1], errors="coerce").fillna(0)
    )
    if time_split.shape[1] >= 3:
        hms_secs = (
            pd.to_numeric(time_split[0], errors="coerce").fillna(0) * 3600
            + pd.to_numeric(time_split[1], errors="coerce").fillna(0) * 60
            + pd.to_numeric(time_split[2], errors="coerce").fillna(0)
        )
        seconds_in_hour = np.where(is_hms, hms_secs, mmss_secs)
    else:
        seconds_in_hour = mmss_secs.values
    time_step = pd.Series(seconds_in_hour).diff()
    time_step = np.where((~np.isnan(time_step)) & (time_step < 0), time_step + 3600, time_step)
    elapsed_seconds = pd.Series(time_step).fillna(0).cumsum()
    df["Elapsed_Seconds"] = elapsed_seconds

    # Backward / forward windows (match the R script’s indexing).
    df["time_passed_backward"] = df["Elapsed_Seconds"] - df["Elapsed_Seconds"].shift(24)
    df["time_passed_forward"] = df["Elapsed_Seconds"].shift(-6) - df["Elapsed_Seconds"]

    # Volatility over last 24 mid_price values if within ~120s.
    roll_std = df["mid_price"].rolling(window=24, min_periods=24).std()
    df["volatility_120s"] = np.where(df["time_passed_backward"] < 135, roll_std, np.nan)

    # OBI momentum: obi_10 - lag 24 if within ~120s.
    obi_10_lag = df["obi_10"].shift(24)
    df["obi_momentum"] = np.where(df["time_passed_backward"] < 135, df["obi_10"] - obi_10_lag, np.nan)

    # Future price / return and target class.
    future_price = np.where(df["time_passed_forward"] < 35, df["mid_price"].shift(-6), np.nan)
    df["future_price"] = future_price
    df["future_return"] = (df["future_price"] - df["mid_price"]) / df["mid_price"]
    df["target_class"] = np.where(df["future_return"] > 0, 1, -1)

    # Drop rows with any NA in model-critical columns.
    df_model_ready = df.dropna(
        subset=[
            "volatility_120s",
            "obi_momentum",
            "future_price",
            "future_return",
            "target_class",
        ]
    ).reset_index(drop=True)

    y = (df_model_ready["target_class"] == 1).astype(int).to_numpy()

    exclude_cols = {
        "future_price",
        "future_return",
        "target_class",
        "time_passed_forward",  # data leak: uses shift(-6), i.e. future elapsed time
    }
    feature_cols = [
        c
        for c in df_model_ready.columns
        if c not in exclude_cols and df_model_ready[c].dtype != "O"
    ]
    X = df_model_ready[feature_cols].to_numpy(dtype=float)

    return LoadedData(df=df_model_ready, X=X, y=y, feature_cols=feature_cols)


def load_data(csv_file: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    """
    Backward-compatible tuple return for existing callers.
    """
    out = load_clean_r_style(csv_file)
    return out.df, out.X, out.y, out.feature_cols

