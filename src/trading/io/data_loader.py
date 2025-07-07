"""trading.io.data_loader

Utility helpers for reading prepared OHLCV CSVs used by predictor / trader.
Extracted from `scripts/predict.py` to avoid code duplication.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["load_latest_data"]


def load_latest_data(ticker: str, data_dir: str | Path = "data") -> pd.DataFrame:
    """Return the freshest CSV for *ticker* in *data_dir*.

    Falls back to the most recent match if no `latest/` folder exists.
    Returns an **empty** DataFrame and logs a warning if nothing is found.
    """
    tdir = Path(data_dir) / ticker.upper()
    latest_dir = tdir / "latest"

    if latest_dir.exists():
        target_csv = latest_dir / "data.csv"
    else:
        # look for any data.csv inside ticker folder tree
        data_files = sorted(tdir.glob("**/data.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not data_files:
            logger.warning("No data files found for ticker %s in %s", ticker, tdir)
            return pd.DataFrame()
        target_csv = data_files[0]

    logger.info("Loading latest data from %s", target_csv)
    df = pd.read_csv(target_csv)

    # ------------------------------------------------------------------
    # 1. Normalise common OHLCV column names (case/spacing insensitive)
    # ------------------------------------------------------------------
    std_cols = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    lower_cols = {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    for low_name, std_name in std_cols.items():
        if low_name in lower_cols and std_name not in df.columns:
            df[std_name] = df[lower_cols[low_name]]

    # ------------------------------------------------------------------
    # 2. Ensure there is a numeric Price column
    # ------------------------------------------------------------------
    if "Price" not in df.columns:
        # Preferred fallbacks in order
        for alt in ("Close", "Adj Close", "Adj_Close", "AdjClose"):
            if alt in df.columns:
                df["Price"] = pd.to_numeric(df[alt], errors="coerce")
                break
        else:
            logger.warning("No price column found for %s – defaulting to zeros", ticker)
            df["Price"] = 0.0
    else:
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    # Fill remaining NaNs with 0.0 (prevents float formatting issues later)
    df["Price"] = df["Price"].fillna(0.0)

    return df
