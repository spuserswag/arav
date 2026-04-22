"""
Shared data loader (v2) — websocket / event-driven order flow.

Why this file changed (see v1/dataloader.py for the old implementation):
  - The v1 loader was calibrated to a 5-second polling cadence. `shift(24)` was
    assumed to be "~120s backward", `shift(-6)` was "~30s forward". The v2
    websocket capture is event-driven (~4 Hz median, bursty, with multi-second
    gaps) so positional shifts no longer correspond to any fixed time window.
  - v2 resamples each contiguous segment to a regular grid, then uses shifts
    expressed in **grid steps** so every window and label is defined in seconds.
  - v2 segments the stream on large gaps so windows/labels never straddle a
    reconnection or feed outage.
  - v2 drops zero-return rows within a small dead-band so ties aren't silently
    folded into "down".
  - v2 drops `Elapsed_Seconds` (a monotonic feature that extrapolates in val/test).

Defaults below match the knobs we agreed on:
    step_seconds        = 0.5    # 500 ms resample grid
    horizon_seconds     = 15.0
    backward_seconds    = 120.0  # "slow" volatility window
    gap_seconds         = 2.0    # any tick gap > 2s starts a new segment
    deadband_bps        = 0.5    # |future_return| < 0.5 bp → drop row

Multi-scale volatility (for the Vol-Transformer architecture):
    vol_fast_seconds    = 5.0    # short-horizon mid-price std
    vol_med_seconds     = 30.0   # medium-horizon mid-price std
    vol_of_vol_seconds  = 30.0   # rolling std of the fast volatility itself
Plus multi-scale log-returns over {1s, 5s, 15s} to give the model explicit
momentum signals at different horizons.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Public container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LoadedData:
    df: pd.DataFrame                 # model-ready, post-resample, post-dropna
    X: np.ndarray                    # (N, F) float64 feature matrix
    y: np.ndarray                    # (N,)   int {0,1} binary up/down label
    feature_cols: List[str]          # column order used to build X
    vol_feature_cols: List[str]      # subset of feature_cols used for FiLM vol conditioning
    segment_ids: np.ndarray          # (N,)   int segment id per row (for seq masking)
    config: dict = field(default_factory=dict)   # effective config used

    @property
    def vol_feature_indices(self) -> List[int]:
        """Indices into feature_cols for the volatility features."""
        return [self.feature_cols.index(c) for c in self.vol_feature_cols]


# ---------------------------------------------------------------------------
# Time parsing (supports MM:SS.s and HH:MM:SS, matching v1 behavior)
# ---------------------------------------------------------------------------

def _parse_time_to_seconds(series: pd.Series) -> np.ndarray:
    """
    Parse a Time column into monotonic elapsed seconds.

    Handles both formats:
      - MM:SS.s   e.g. "11:23.9"  (minute:second.tenths)
      - HH:MM:SS  e.g. "03:49:45"

    Wraps at hour boundaries (a negative diff → add 3600s) to produce a
    monotonic elapsed-seconds vector. NaNs become 0 for the diff.
    """
    s = series.astype(str).str.strip()
    parts = s.str.split(":", expand=True)
    is_hms = s.str.count(":") == 2

    mmss = (
        pd.to_numeric(parts[0], errors="coerce").fillna(0) * 60
        + pd.to_numeric(parts[1], errors="coerce").fillna(0)
    )
    if parts.shape[1] >= 3:
        hms = (
            pd.to_numeric(parts[0], errors="coerce").fillna(0) * 3600
            + pd.to_numeric(parts[1], errors="coerce").fillna(0) * 60
            + pd.to_numeric(parts[2], errors="coerce").fillna(0)
        )
        secs = np.where(is_hms, hms.values, mmss.values)
    else:
        secs = mmss.values

    # Unwrap hour rollovers: whenever the diff goes negative we assume the clock
    # wrapped from e.g. 59:58 → 00:02 and add 3600s to the step.
    steps = np.diff(secs, prepend=secs[0])
    steps = np.where(steps < 0, steps + 3600.0, steps)
    steps[0] = 0.0
    elapsed = np.cumsum(steps)
    return elapsed.astype(float)


# ---------------------------------------------------------------------------
# Column normalization (teacher → internal names)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Segment detection on gaps
# ---------------------------------------------------------------------------

def _segment_ids_from_gaps(elapsed: np.ndarray, gap_seconds: float) -> np.ndarray:
    """
    Return a per-row segment id such that adjacent rows whose Δt > gap_seconds
    land in different segments.
    """
    if len(elapsed) == 0:
        return np.zeros(0, dtype=np.int64)
    dt = np.diff(elapsed, prepend=elapsed[0])
    breaks = dt > gap_seconds
    # First row starts segment 0; every gap increments the segment id.
    return np.cumsum(breaks).astype(np.int64)


# ---------------------------------------------------------------------------
# Resample a single segment to a fixed grid
# ---------------------------------------------------------------------------

_BASE_COLS = ["mid_price", "micro_price", "fractional_price",
              "spread", "obi_1", "obi_10", "obi_diff"]


def _resample_segment(
    seg: pd.DataFrame,
    step_seconds: float,
    seg_id: int,
) -> pd.DataFrame:
    """
    Resample one contiguous segment to a regular grid.

    Strategy: bin by floor(elapsed / step) and take the **last** tick in each
    bin (the most recent book state at the grid point). Bins with no tick are
    forward-filled from the previous bin **within the same segment only**
    (because inter-segment gaps would mean stale data across a dropout).

    Returns a DataFrame indexed 0..K-1 on the grid. Adds a `segment_id` column.
    """
    if seg.empty:
        return seg.iloc[0:0].copy()

    seg = seg.reset_index(drop=True)
    t = seg["Elapsed_Seconds"].to_numpy()
    t0 = t[0]
    bin_idx = np.floor((t - t0) / step_seconds).astype(np.int64)

    cols = _BASE_COLS
    grid_len = int(bin_idx[-1]) + 1
    grid = pd.DataFrame(
        index=np.arange(grid_len),
        columns=cols,
        dtype=float,
    )

    # "Last tick per bin": groupby bin_idx and take the last observation, which
    # is the most recent book snapshot at that grid point.
    seg_tagged = seg[cols].copy()
    seg_tagged["_bin"] = bin_idx
    last_per_bin = seg_tagged.groupby("_bin", sort=True).last()
    grid.loc[last_per_bin.index, cols] = last_per_bin[cols].to_numpy()

    # Forward-fill empty bins within this segment. Bin 0 always has a tick
    # (it's the first row of the segment) so there should be no leading NaNs.
    grid[cols] = grid[cols].ffill()

    grid["Elapsed_Seconds"] = t0 + grid.index.to_numpy() * step_seconds
    grid["segment_id"] = seg_id
    return grid.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature engineering (time-based, per segment)
# ---------------------------------------------------------------------------

def _add_scale_free_book_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive scale-free replacements for raw-dollar book features so the feature
    distribution doesn't trend with BTC's price level. (Train mid_price ≠ val
    mid_price ≠ test mid_price; z-scoring with train stats leaves val/test
    ~2σ OOD. These derived features are bps-scale and stationary.)

    Adds:
      micro_mid_bps = (micro_price - mid_price) / mid_price * 1e4
      spread_bps    = spread                   / mid_price * 1e4
    """
    df["micro_mid_bps"] = (df["micro_price"] - df["mid_price"]) / df["mid_price"] * 1e4
    df["spread_bps"] = df["spread"] / df["mid_price"] * 1e4
    return df


def _per_segment_rolling(
    df: pd.DataFrame,
    step_seconds: float,
    backward_seconds: float,
    horizon_seconds: float,
    vol_fast_seconds: float,
    vol_med_seconds: float,
    vol_of_vol_seconds: float,
    ret_horizons_seconds: Tuple[float, ...],
) -> pd.DataFrame:
    """
    Compute multi-scale volatility + multi-scale return features, the backward
    OBI momentum, and the forward label — all respecting segment boundaries.

    We compute rolling ops on log(mid_price) so the volatility is a ~% std,
    which is scale-free across the dataset.

    Parameters
    ----------
    backward_seconds   : slow volatility window (and OBI momentum lag)
    horizon_seconds    : forward label horizon
    vol_fast_seconds   : short-horizon volatility window
    vol_med_seconds    : medium-horizon volatility window
    vol_of_vol_seconds : rolling std of the *fast* volatility (vol-of-vol)
    ret_horizons_seconds : tuple of backward return windows, one column each
    """
    def to_steps(sec: float) -> int:
        return max(1, int(round(sec / step_seconds)))

    slow_steps = to_steps(backward_seconds)
    fast_steps = to_steps(vol_fast_seconds)
    med_steps = to_steps(vol_med_seconds)
    vov_steps = to_steps(vol_of_vol_seconds)
    fwd_steps = to_steps(horizon_seconds)

    g = df.groupby("segment_id", sort=False, group_keys=False)

    # log-mid is stationary-ish → its rolling std is a fractional volatility
    df["_log_mid"] = np.log(df["mid_price"])

    # --- Multi-scale volatility (std of log-mid over backward windows).
    df["volatility_fast"] = g["_log_mid"].transform(
        lambda s: s.rolling(window=fast_steps, min_periods=fast_steps).std()
    )
    df["volatility_med"] = g["_log_mid"].transform(
        lambda s: s.rolling(window=med_steps, min_periods=med_steps).std()
    )
    df["volatility_slow"] = g["_log_mid"].transform(
        lambda s: s.rolling(window=slow_steps, min_periods=slow_steps).std()
    )

    # Vol-of-vol: std of volatility_fast over the vol_of_vol window.
    df["vol_of_vol"] = df.groupby("segment_id", sort=False, group_keys=False)[
        "volatility_fast"
    ].transform(lambda s: s.rolling(window=vov_steps, min_periods=vov_steps).std())

    # --- Multi-scale backward log-returns.
    for h_sec in ret_horizons_seconds:
        h_steps = to_steps(h_sec)
        col = f"logret_{int(round(h_sec * 1000))}ms"
        df[col] = df.groupby("segment_id", sort=False, group_keys=False)[
            "_log_mid"
        ].transform(lambda s, k=h_steps: s - s.shift(k))

    # OBI momentum over the slow window.
    df["obi_momentum"] = g["obi_10"].transform(lambda s: s - s.shift(slow_steps))

    # Future mid price and label return (per segment).
    df["future_price"] = g["mid_price"].transform(lambda s: s.shift(-fwd_steps))
    df["future_return"] = (df["future_price"] - df["mid_price"]) / df["mid_price"]

    df.drop(columns=["_log_mid"], inplace=True)
    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def load_clean_r_style(
    csv_file: str,
    step_seconds: float = 0.5,
    horizon_seconds: float = 15.0,
    backward_seconds: float = 120.0,
    gap_seconds: float = 2.0,
    deadband_bps: float = 0.5,
    vol_fast_seconds: float = 5.0,
    vol_med_seconds: float = 30.0,
    vol_of_vol_seconds: float = 30.0,
    ret_horizons_seconds: Tuple[float, ...] = (1.0, 5.0, 15.0),
) -> LoadedData:
    """
    Load, resample, and label a websocket order-flow CSV.

    Parameters
    ----------
    csv_file : str
        Path to the CSV (old or new schema — columns are normalized).
    step_seconds : float, default 0.5
        Resample grid spacing in seconds.
    horizon_seconds : float, default 15.0
        Forward horizon for the binary label.
    backward_seconds : float, default 120.0
        Backward window for volatility and OBI momentum features.
    gap_seconds : float, default 2.0
        Any tick-to-tick Δt larger than this starts a new segment. Rolling
        windows, shifts, and sequences never cross a segment boundary.
    deadband_bps : float, default 0.5
        Rows whose |future_return| is under this many basis points are treated
        as ties and dropped. 1 bp = 1e-4. Set to 0 to disable.

    Returns
    -------
    LoadedData with:
        df           : model-ready frame (post-resample, post-dropna, post-deadband)
        X            : (N, F) feature matrix
        y            : (N,) int {0,1} label
        feature_cols : list of column names in the order used to build X
        segment_ids  : (N,) segment id per row (use to mask sequence starts)
        config       : dict of effective knobs
    """
    cfg = {
        "step_seconds": step_seconds,
        "horizon_seconds": horizon_seconds,
        "backward_seconds": backward_seconds,
        "gap_seconds": gap_seconds,
        "deadband_bps": deadband_bps,
        "vol_fast_seconds": vol_fast_seconds,
        "vol_med_seconds": vol_med_seconds,
        "vol_of_vol_seconds": vol_of_vol_seconds,
        "ret_horizons_seconds": list(ret_horizons_seconds),
    }

    df = pd.read_csv(csv_file)
    df = _normalize_columns(df)

    missing = [c for c in _BASE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    if "Time" not in df.columns:
        raise ValueError("Expected a 'Time' column in the CSV.")

    # 1) Parse Time → monotonic Elapsed_Seconds and segment on gaps.
    df = df.copy()
    df["Elapsed_Seconds"] = _parse_time_to_seconds(df["Time"])
    df["segment_id"] = _segment_ids_from_gaps(
        df["Elapsed_Seconds"].to_numpy(), gap_seconds=gap_seconds
    )

    # 2) Resample each segment to the fixed grid. Re-index segment ids
    #    contiguously on the resampled output.
    resampled_parts: List[pd.DataFrame] = []
    for new_id, (_, seg) in enumerate(df.groupby("segment_id", sort=True)):
        part = _resample_segment(seg, step_seconds=step_seconds, seg_id=new_id)
        if not part.empty:
            resampled_parts.append(part)

    if not resampled_parts:
        return LoadedData(
            df=df.iloc[0:0].copy(),
            X=np.empty((0, 0)),
            y=np.empty((0,), dtype=int),
            feature_cols=[],
            vol_feature_cols=[],
            segment_ids=np.empty((0,), dtype=np.int64),
            config=cfg,
        )

    grid_df = pd.concat(resampled_parts, ignore_index=True)

    # 3) Scale-free book features (replaces raw-dollar mid/micro/spread in X).
    #    Raw prices trend with BTC's level between train and val; the scaler
    #    can't undo that. These are bps-scale and stationary.
    grid_df = _add_scale_free_book_features(grid_df)

    # 4) Time-based backward windows + forward label (per segment).
    grid_df = _per_segment_rolling(
        grid_df,
        step_seconds=step_seconds,
        backward_seconds=backward_seconds,
        horizon_seconds=horizon_seconds,
        vol_fast_seconds=vol_fast_seconds,
        vol_med_seconds=vol_med_seconds,
        vol_of_vol_seconds=vol_of_vol_seconds,
        ret_horizons_seconds=ret_horizons_seconds,
    )

    # 5) Drop rows with any required NA (backward rolling + future shift edges).
    ret_cols = [
        f"logret_{int(round(h * 1000))}ms" for h in ret_horizons_seconds
    ]
    required = [
        "volatility_fast",
        "volatility_med",
        "volatility_slow",
        "vol_of_vol",
        "obi_momentum",
        "future_price",
        "future_return",
    ] + ret_cols
    df_ready = grid_df.dropna(subset=required).reset_index(drop=True)

    # 6) Apply dead-band to drop ties. 1 bp = 1e-4.
    if deadband_bps > 0:
        thresh = deadband_bps * 1e-4
        keep = df_ready["future_return"].abs() >= thresh
        df_ready = df_ready.loc[keep].reset_index(drop=True)

    # 7) Binary label.
    y = (df_ready["future_return"] > 0).astype(int).to_numpy()

    # 8) Feature columns — explicit allow-list, preserves ordering.
    #    Organized into:
    #      - "price" features (raw book state + cyclical time)
    #      - multi-scale log-return features (explicit momentum signal)
    #      - "vol" features (used both as model input AND as the FiLM conditioner)
    # Book-state features. All stationary / scale-free:
    #  - micro_mid_bps / spread_bps : bps-scale replacements for raw dollars
    #    (mid_price and micro_price trend with BTC's price level, polluting
    #     train/val with the split boundary; the scaler can't fix that).
    #  - fractional_price            : naturally bounded in [0, 1).
    #  - obi_*                       : already bounded / imbalance ratios.
    # Dropped vs previous version: mid_price, micro_price, raw spread,
    #                              tod_sin, tod_cos (elapsed-time leakage).
    price_cols = [
        "micro_mid_bps",
        "spread_bps",
        "fractional_price",
        "obi_1",
        "obi_10",
        "obi_diff",
    ]
    vol_feature_cols = [
        "volatility_fast",
        "volatility_med",
        "volatility_slow",
        "vol_of_vol",
        "obi_momentum",
    ]
    feature_cols = price_cols + ret_cols + vol_feature_cols
    X = df_ready[feature_cols].to_numpy(dtype=float)
    segment_ids = df_ready["segment_id"].to_numpy(dtype=np.int64)

    return LoadedData(
        df=df_ready,
        X=X,
        y=y,
        feature_cols=feature_cols,
        vol_feature_cols=vol_feature_cols,
        segment_ids=segment_ids,
        config=cfg,
    )


# ---------------------------------------------------------------------------
# Sequence assembly (respects segment boundaries)
# ---------------------------------------------------------------------------

def valid_sequence_ends(
    segment_ids: np.ndarray,
    seq_len: int,
    stride: int = 1,
    start_offset: int = 0,
) -> np.ndarray:
    """
    Indices `t` such that rows [t - seq_len + 1 .. t] all share the same
    segment_id (i.e. the window doesn't cross a discontinuity).

    Parameters
    ----------
    segment_ids : (N,) array
    seq_len     : window length in grid steps
    stride      : step between consecutive window ends (use >1 for independent
                  test windows)
    start_offset : optional offset applied before striding (used e.g. to align
                   eval strides to the end of a split)

    Returns
    -------
    np.ndarray of end-indices `t` (inclusive) into the feature matrix.
    """
    n = len(segment_ids)
    if n < seq_len:
        return np.empty((0,), dtype=np.int64)
    # A window ending at t is valid iff segment_ids[t - seq_len + 1] == segment_ids[t].
    starts = segment_ids[: n - seq_len + 1]
    ends = segment_ids[seq_len - 1 :]
    mask = starts == ends
    all_valid_ends = np.arange(seq_len - 1, n)[mask]
    if stride <= 1 and start_offset == 0:
        return all_valid_ends
    # Apply offset + stride.
    sel = (np.arange(len(all_valid_ends)) - start_offset) % stride == 0
    sel &= np.arange(len(all_valid_ends)) >= start_offset
    return all_valid_ends[sel]


def build_sequences(
    X: np.ndarray,
    y: np.ndarray,
    segment_ids: np.ndarray,
    seq_len: int,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Materialize (num_sequences, seq_len, num_features) float32 windows whose
    rows share a segment_id. Label is taken from the window end.

    Uses numpy fancy-indexing with a precomputed valid-end list so we never
    stack a window that straddles a dropout.
    """
    ends = valid_sequence_ends(segment_ids, seq_len=seq_len, stride=stride)
    if ends.size == 0:
        return np.empty((0, seq_len, X.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.float32)
    # Vectorized gather: for each end t, take rows [t-seq_len+1 .. t].
    offsets = np.arange(-seq_len + 1, 1)  # (seq_len,), inclusive of 0
    idx = ends[:, None] + offsets[None, :]  # (num_seq, seq_len)
    X_seq = X[idx].astype(np.float32, copy=False)
    y_seq = y[ends].astype(np.float32, copy=False)
    return X_seq, y_seq


# ---------------------------------------------------------------------------
# Backward-compatible wrapper
# ---------------------------------------------------------------------------

def load_data(
    csv_file: str,
    **kwargs,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    """Tuple return kept for legacy callers (XGBoost baseline, initial.py)."""
    out = load_clean_r_style(csv_file, **kwargs)
    return out.df, out.X, out.y, out.feature_cols
