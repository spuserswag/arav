import argparse
import pickle
import time
from pathlib import Path
from typing import Optional, Tuple

import ccxt
import numpy as np
import pandas as pd
import torch
from torch import nn
from datetime import datetime

from dataloader import load_data
from transformer import TimeSeriesTransformer


def find_latest_version(base: str = "outputs") -> Path:
    """Return the outputs/vN folder with the highest N."""
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


def get_next_data_dir(base: str = "data") -> Path:
    """Create data/data_1, data/data_2, ... and return the new folder."""
    base_path = Path(base)
    base_path.mkdir(exist_ok=True)
    existing = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("data_")]
    versions = []
    for d in existing:
        try:
            versions.append(int(d.name.split("_")[1]))
        except (ValueError, IndexError):
            pass
    next_n = max(versions, default=0) + 1
    out_dir = base_path / f"data_{next_n}"
    out_dir.mkdir()
    return out_dir


def load_model_and_scaler(
    version_dir: Path,
    device: torch.device,
) -> Tuple[nn.Module, object]:
    """
    Load model weights and scaler from version_dir.
    input_dim is inferred from the scaler's n_features_in_.
    Reads config.json for encoding and attn_diagonal_bias if present.
    """
    import json

    ckpt_path = version_dir / "transformer_best.pt"
    scaler_path = version_dir / "scaler.pkl"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    input_dim = scaler.n_features_in_

    config = {}
    config_path = version_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    model = TimeSeriesTransformer(
        input_dim=input_dim,
        pos_encoding=config.get("encoding", "sinusoidal"),
        attn_diagonal_bias=config.get("attn_diagonal_bias", 0.0),
    ).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, scaler


def compute_volatility(df: pd.DataFrame, window: int = 20) -> float:
    """
    Rolling std of returns on mid price. High vol = less efficient market = more confidence to trade.
    Returns 0.0 if insufficient data.
    """
    prices = df["mid_price"] if "mid_price" in df.columns else df.get("Mid Price")
    if prices is None or len(prices) < window + 1:
        return 0.0
    prices = pd.Series(prices).astype(float)
    returns = prices.pct_change().dropna()
    if len(returns) < window:
        return 0.0
    vol = returns.iloc[-window:].std()
    return float(vol) if not np.isnan(vol) else 0.0


def predict_latest_direction(
    model: nn.Module,
    scaler: object,
    live_csv: str,
    seq_len: int,
    up_threshold: float = 0.6,
    down_threshold: float = 0.4,
) -> Tuple[float, int, str]:
    """
    Run inference on the most recent seq_len samples from live_csv (after
    applying the same clean.R-style processing and scaling).

    Returns (prob_up, class_label, action), where:
      - prob_up: P(price in ~30s > now)
      - class_label: 1 for up, 0 for non-up
      - action: "LONG", "SHORT", or "FLAT" (no trade)
    """
    device = next(model.parameters()).device
    df, X, y, feature_cols = load_data(live_csv)
    if len(X) < seq_len:
        raise ValueError(f"Not enough rows after cleaning to form a window of length {seq_len}.")

    X_scaled = scaler.transform(X)
    X_window = X_scaled[-seq_len:]
    X_window_t = torch.from_numpy(X_window).float().unsqueeze(0).to(device)  # (1, T, F)

    with torch.no_grad():
        logits = model(X_window_t)
        prob_up = float(torch.sigmoid(logits).item())

    label = int(prob_up >= 0.5)

    if prob_up >= up_threshold:
        action = "LONG"
    elif prob_up <= down_threshold:
        action = "SHORT"
    else:
        action = "FLAT"

    return prob_up, label, action


def main() -> None:
    parser = argparse.ArgumentParser(description="Live-ish continuous inference for Transformer model")
    parser.add_argument(
        "--version-dir",
        type=str,
        default=None,
        help="Path to outputs/vN directory (default: auto-detect latest outputs/vN)",
    )
    parser.add_argument(
        "--live-csv",
        type=str,
        default=None,
        help="CSV path (default: data/data_N/live_orderflow.csv, new folder each run).",
    )
    parser.add_argument(
        "--data-base",
        type=str,
        default="data",
        help="Base folder for data/data_N when creating new run directories.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=40,
        help="Sequence length used during training.",
    )
    parser.add_argument(
        "--up-threshold",
        type=float,
        default=0.6,
        help="Minimum prob(up) to go LONG.",
    )
    parser.add_argument(
        "--down-threshold",
        type=float,
        default=0.4,
        help="Maximum prob(up) to go SHORT.",
    )
    parser.add_argument(
        "--vol-threshold",
        type=float,
        default=0.0001,
        help="Minimum volatility (rolling std of returns) to allow a trade. High vol = more confidence.",
    )
    parser.add_argument(
        "--vol-window",
        type=int,
        default=20,
        help="Window for rolling volatility calculation.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Seconds between checks for new data / trades.",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=30.0,
        help="Seconds to hold a position before closing (matches model prediction horizon).",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTC/USDT",
        help="Trading pair symbol for live data (ccxt format).",
    )
    parser.add_argument(
        "--outputs-base",
        type=str,
        default="outputs",
        help="Base folder for outputs/vN when auto-detecting latest model.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.version_dir is not None:
        version_dir = Path(args.version_dir)
    else:
        version_dir = find_latest_version(args.outputs_base)
    print(f"Using model from: {version_dir}")

    model, scaler = load_model_and_scaler(version_dir, device)

    # Create fresh data folder for this run (data/data_1, data/data_2, ...)
    if args.live_csv is not None:
        live_path = Path(args.live_csv)
        live_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        data_dir = get_next_data_dir(args.data_base)
        live_path = data_dir / "live_orderflow.csv"
    print(f"Live CSV: {live_path}")

    # Live data exchange (BinanceUS, same as training data source)
    exchange = ccxt.binanceus()

    print("Starting continuous inference & trading loop. Press Ctrl+C to stop.\n")

    last_n_rows = 0
    position = "FLAT"  # "FLAT", "LONG", or "SHORT"
    entry_price: float | None = None
    entry_time: datetime | None = None
    total_pnl = 0.0
    trade_count = 0

    try:
        while True:
            # 1) Fetch a fresh live order book snapshot
            try:
                ob = exchange.fetch_order_book(args.symbol)
            except Exception as e:
                print(f"[error] fetch_order_book failed: {e}")
                time.sleep(args.poll_interval)
                continue

            # Compute the same raw features as the teacher CSV header
            bids = ob["bids"]
            asks = ob["asks"]
            if not bids or not asks:
                print("[warn] Empty order book, skipping this tick.")
                time.sleep(args.poll_interval)
                continue

            best_bid, bid_vol_1 = bids[0]
            best_ask, ask_vol_1 = asks[0]
            mid_price = (best_bid + best_ask) / 2.0
            spread = best_ask - best_bid
            fractional_price = mid_price - int(mid_price)
            micro_price = (best_bid * ask_vol_1 + best_ask * bid_vol_1) / (bid_vol_1 + ask_vol_1)

            # Depth-10 OBI metrics
            top_bids_10 = bids[:10]
            top_asks_10 = asks[:10]
            bid_vol_10 = sum(b[1] for b in top_bids_10)
            ask_vol_10 = sum(a[1] for a in top_asks_10)
            if (bid_vol_10 + ask_vol_10) == 0:
                obi_10 = 0.0
            else:
                obi_10 = (bid_vol_10 - ask_vol_10) / (bid_vol_10 + ask_vol_10)
            if (bid_vol_1 + ask_vol_1) == 0:
                obi_1 = 0.0
            else:
                obi_1 = (bid_vol_1 - ask_vol_1) / (bid_vol_1 + ask_vol_1)
            obi_diff = obi_10 - obi_1

            # Append a new row to the live CSV for this run
            header_needed = not live_path.exists()
            row_df = pd.DataFrame(
                [
                    {
                        "Time": datetime.now().strftime("%H:%M:%S"),
                        "Mid Price": float(mid_price),
                        "Micro Price": float(micro_price),
                        "fractional price": float(fractional_price),
                        "spread": float(spread),
                        "obi_1": float(obi_1),
                        "obi_10": float(obi_10),
                        "obi_diff": float(obi_diff),
                    }
                ]
            )
            row_df.to_csv(live_path, mode="a", index=False, header=header_needed)

            # 2) Reload cleaned data from the live CSV and act when enough rows exist
            df, X, y, feature_cols = load_data(str(live_path))
            n_rows = len(X)

            if n_rows < args.seq_len:
                print(f"[loop] rows={n_rows} (<{args.seq_len}) | waiting for more data...")
            else:
                current_price = float(mid_price)

                if n_rows > last_n_rows:
                    last_n_rows = n_rows
                    prob_up, label, action = predict_latest_direction(
                        model=model,
                        scaler=scaler,
                        live_csv=str(live_path),
                        seq_len=args.seq_len,
                        up_threshold=args.up_threshold,
                        down_threshold=args.down_threshold,
                    )
                    vol = compute_volatility(df, window=args.vol_window)
                    if vol < args.vol_threshold and action in ("LONG", "SHORT"):
                        action = "FLAT"
                    direction = "UP" if label == 1 else "DOWN/NO-UP"

                    # Simple position management:
                    # - If flat and strong signal → open position.
                    # - If in position and strong opposite signal → close, report PnL, and flip.
                    # - Otherwise hold.

                    msg_prefix = f"[rows={n_rows}] price={current_price:.2f} prob(up)={prob_up:.4f} vol={vol:.6f} → class: {direction} → signal: {action}"

                    if position == "FLAT":
                        if action == "LONG":
                            position = "LONG"
                            entry_price = current_price
                            entry_time = datetime.now()
                            print(f"{msg_prefix} | OPEN LONG @ {entry_price:.2f}")
                        elif action == "SHORT":
                            position = "SHORT"
                            entry_price = current_price
                            entry_time = datetime.now()
                            print(f"{msg_prefix} | OPEN SHORT @ {entry_price:.2f}")
                        else:
                            print(f"{msg_prefix} | NO POSITION")
                    else:
                        # Already in a trade
                        assert entry_price is not None
                        assert entry_time is not None
                        held_seconds = (datetime.now() - entry_time).total_seconds()
                        close_time_expired = held_seconds >= args.hold_seconds
                        close_and_flip = False

                        if position == "LONG" and action == "SHORT":
                            close_and_flip = True
                        elif position == "SHORT" and action == "LONG":
                            close_and_flip = True

                        if close_time_expired or close_and_flip:
                            # Close existing position
                            if position == "LONG":
                                trade_pnl = current_price - entry_price
                            else:  # SHORT
                                trade_pnl = entry_price - current_price

                            trade_ret_pct = (trade_pnl / entry_price) * 100.0
                            total_pnl += trade_pnl
                            trade_count += 1

                            reason = f"after {held_seconds:.0f}s" if close_time_expired else "signal flip"
                            print(
                                f"{msg_prefix} | CLOSE {position} @ {current_price:.2f} "
                                f"(entry {entry_price:.2f}) {reason} → PnL {trade_pnl:.2f} "
                                f"({trade_ret_pct:.2f}%), total PnL {total_pnl:.2f} "
                                f"over {trade_count} trades"
                            )

                            if close_and_flip:
                                # Flip into new position
                                position = "LONG" if action == "LONG" else "SHORT"
                                entry_price = current_price
                                entry_time = datetime.now()
                                print(f"    → OPEN {position} @ {entry_price:.2f}")
                            else:
                                # Time-based close: go flat
                                position = "FLAT"
                                entry_price = None
                                entry_time = None
                        else:
                            # Hold existing position
                            print(f"{msg_prefix} | HOLD {position} from {entry_price:.2f}")

                # If no new rows, still check time-based close and print heartbeat
                if n_rows == last_n_rows and n_rows >= args.seq_len:
                    # Close position if hold time expired (even without new inference)
                    if position != "FLAT" and entry_price is not None and entry_time is not None:
                        held_seconds = (datetime.now() - entry_time).total_seconds()
                        if held_seconds >= args.hold_seconds:
                            if position == "LONG":
                                trade_pnl = current_price - entry_price
                            else:
                                trade_pnl = entry_price - current_price
                            trade_ret_pct = (trade_pnl / entry_price) * 100.0
                            total_pnl += trade_pnl
                            trade_count += 1
                            print(
                                f"[loop] CLOSE {position} @ {current_price:.2f} (entry {entry_price:.2f}) "
                                f"after {held_seconds:.0f}s → PnL {trade_pnl:.2f} ({trade_ret_pct:.2f}%), "
                                f"total PnL {total_pnl:.2f} over {trade_count} trades"
                            )
                            position = "FLAT"
                            entry_price = None
                            entry_time = None

                    status = f"[loop] rows={n_rows} price={current_price:.2f} | position={position}"
                    if entry_price is not None and position != "FLAT":
                        # Mark-to-market PnL of the open position
                        if position == "LONG":
                            mtm = current_price - entry_price
                        else:
                            mtm = entry_price - current_price
                        status += f" | unrealized PnL={mtm:.2f}"
                    print(status)

            # Sleep before next check
            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print("\nStopped continuous inference loop.")


if __name__ == "__main__":
    main()

