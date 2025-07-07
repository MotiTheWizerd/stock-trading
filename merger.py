"""merger.py
Clean and merge intraday OHLCV data with generated trading signals.

This module addresses common pitfalls that arise when trying to join two
independently-generated CSV files:

1.  *Timezone mismatches* – `signals.csv` often stores `datetime` values as
    timezone-aware strings (usually UTC) whereas the OHLCV `data.csv` uses a
    timezone-naïve index.  Pandas cannot compare these directly, resulting in
    errors such as `Cannot compare tz-naive and tz-aware timestamps`.
2.  *Header rows creeping into the data area* – When an automated process
    repeatedly appends to a CSV it is easy to accidentally write the header a
    second time.  Those stray string values break `pd.to_datetime` and
    downstream sorting operations.
3.  *Unnamed index columns* – `data.csv` is stored with the timestamp as the
    *index*, not a regular column.  When we reset that index we must remember
    to give the column a name, otherwise later code that expects a column
    called ``datetime`` will raise a ``KeyError``.

The public helper in this file, :func:`prepare_labeled_dataset`, performs the
following end-to-end procedure that makes the two sources compatible and
produces a tidy, machine-learning-ready DataFrame.

Steps
-----
1.  Read **data.csv** using the first column as the index.
2.  Remove rows that contain non-timestamp garbage (e.g. header repeats).
3.  Cast the index to *timezone-naïve* ``datetime64[ns]`` and add a proper
    ``datetime`` column.
4.  Read **signals.csv**, coerce its ``datetime`` column to timezone-naïve as
    well.
5.  ``merge_asof`` the two tables on *nearest previous* timestamp within a
    configurable tolerance (default 10 minutes).
6.  Drop rows where a match could not be found (``NaN`` OHLC values).
7.  Convert textual ``signal`` values to numerical ``label`` (BUY → 1, SELL →
    -1).

Example
~~~~~~~
>>> from merger import prepare_labeled_dataset
>>> df = prepare_labeled_dataset("AAPL")
>>> df.head()

The returned DataFrame is sorted by ``datetime`` and contains every original
OHLCV column plus ``signal`` and ``label`` – ready for modelling or
back-testing.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import pandas as pd

from utils import get_date_folder

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _clean_data_csv(path: str) -> pd.DataFrame:
    """Read *data.csv* and ensure a clean, tz-naïve datetime index.

    1. Drops rows that clearly do **not** represent datetimes (e.g. the word
       "Ticker" accidentally written as the index).
    2. Forces the index to ``datetime64[ns]`` and strips any timezone.

    Returns the cleaned frame **with** a duplicate ``datetime`` column so that
    downstream ``merge_asof`` calls can operate on a normal column rather than
    the index.
    """
    df = pd.read_csv(path, index_col=0)

    # Guard clause: empty file → Nothing to merge later on.
    if df.empty:
        raise ValueError(f"[data.csv] at {path} is empty – cannot continue.")

    # Drop clearly invalid index entries (e.g. header rows injected as data)
    bad_rows = df.index.to_series().astype(str).str.contains("Ticker", na=False)
    if bad_rows.any():
        df = df[~bad_rows]

    # Convert the index to tz-naïve datetime.  Any unparsable value becomes NaT.
    df.index = pd.to_datetime(df.index, errors="coerce").tz_localize(None)
    df = df.dropna(how="any")  # also removes NaT rows

    # {future safety} ensure chronological order
    df = df.sort_index()

    # Provide an explicit column for merge_asof (index itself cannot be used)
    df["datetime"] = df.index

    return df


def _clean_signals_csv(path: str) -> pd.DataFrame:
    """Read *signals.csv* and coerce the ``datetime`` column to tz-naïve."""
    sig = pd.read_csv(path)
    if sig.empty:
        raise ValueError(f"[signals.csv] at {path} is empty – cannot continue.")

    if "datetime" not in sig.columns:
        raise KeyError("signals.csv is expected to have a 'datetime' column")

    sig["datetime"] = (
        pd.to_datetime(sig["datetime"], errors="coerce")
        .dt.tz_localize(None)  # strip timezone to match data.csv
    )

    # Drop rows with unparsable datetimes just in case
    sig = sig.dropna(subset=["datetime"]).sort_values("datetime")

    return sig

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sanitize_day_files(
    ticker: str,
    *,
    date_folder: Optional[str] = None,
    tolerance: str | pd.Timedelta = "10min",
) -> None:
    """Clean *data.csv* & *signals.csv* in-place after each scheduler cycle.

    Parameters
    ----------
    ticker : str
        Stock symbol we just processed.
    date_folder : str, optional
        Folder inside ``data/{ticker}`` – defaults to today if omitted.
    tolerance : str | pd.Timedelta
        Included only for forward-compatibility; currently unused but kept
        consistent with :func:`prepare_labeled_dataset`.
    """
    date_folder = date_folder or get_date_folder()
    root = os.path.join("data", ticker, date_folder)
    data_path = os.path.join(root, "data.csv")
    signals_path = os.path.join(root, "signals.csv")

    # Clean data.csv (if it exists)
    if os.path.exists(data_path):
        df_data = _clean_data_csv(data_path)
        # Write back with datetime index as the first column (same layout)
        df_data.to_csv(data_path)
    else:
        print(f"[sanitize] data.csv missing at {data_path}")

    # Clean signals.csv (optional – may not exist on first run)
    if os.path.exists(signals_path):
        df_sig = _clean_signals_csv(signals_path)
        df_sig.to_csv(signals_path, index=False)
    else:
        # Not critical; signals are created later by charting pipeline
        print(f"[sanitize] signals.csv missing at {signals_path}")

    print(f"🧹 Sanitized day files for {ticker} ({date_folder})")


def prepare_labeled_dataset(
    ticker: str,
    *,
    date_folder: Optional[str] = None,
    tolerance: str | pd.Timedelta = "10min",
) -> pd.DataFrame:
    """Return a merged OHLCV + *label* DataFrame for *ticker* on *date_folder*.

    Parameters
    ----------
    ticker : str
        Stock symbol (e.g. ``"AAPL"``).
    date_folder : str, optional
        Sub-folder inside ``data/{ticker}`` that contains the day-specific
        CSVs.  Defaults to *today's* folder as produced by
        :pyfunc:`utils.get_date_folder`.
    tolerance : str | pd.Timedelta, optional
        Maximum gap allowed between a signal and the matched OHLCV row.  Passed
        straight to :func:`pandas.merge_asof`.
    """
    date_folder = date_folder or get_date_folder()

    root = os.path.join("data", ticker, date_folder)
    data_path = os.path.join(root, "data.csv")
    signals_path = os.path.join(root, "signals.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"data.csv not found at {data_path}")
    if not os.path.exists(signals_path):
        raise FileNotFoundError(f"signals.csv not found at {signals_path}")

    df_data = _clean_data_csv(data_path)
    df_sig = _clean_signals_csv(signals_path)

    # Ensure chronological order before merge_asof (requirement of the API)
    df_data = df_data.sort_values("datetime")
    df_sig = df_sig.sort_values("datetime")

    merged = pd.merge_asof(
        left=df_sig,
        right=df_data.reset_index(drop=True),
        on="datetime",
        direction="backward",
        tolerance=pd.Timedelta(tolerance),
    )

    # Drop signals that did not find a matching OHLCV record.
    merged = merged.dropna(subset=["Open", "High", "Low", "Close"])  # key cols

    # Map textual BUY/SELL to +1 / -1 for ML-friendly label.
    merged["label"] = merged["signal"].map({"BUY": 1, "SELL": -1})

    return merged.sort_values("datetime").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature engineering for model training
# ---------------------------------------------------------------------------

def prepare_feature_dataset(
    ticker: str,
    *,
    date_folder: Optional[str] = None,
    tolerance: str | pd.Timedelta = "10min",
) -> pd.DataFrame:
    """Return an ML-ready feature matrix + label for *ticker* on *date_folder*.

    This helper builds on :func:`prepare_labeled_dataset` but narrows the output
    to ONLY the features required by the modelling pipeline and encodes the
    categorical columns as requested.

    Features
    --------
    rsi, macd, macd_signal, macd_histogram, price_change, price_change_pct,
    gap_from_open, volume_ratio, intraday_range, market_cap_proxy,
    signal_strength (numeric), trend_direction (one-hot encoded).

    The label column ``signal`` (BUY / SELL) is preserved **unmodified**.  All
    other extraneous columns such as ``ticker``, ``price`` and any raw OHLCV
    values are excluded.
    """

    df = prepare_labeled_dataset(
        ticker, date_folder=date_folder, tolerance=tolerance
    )

    # -------------------------------------------------------------------
    # 1. Keep only the core numerical features
    # -------------------------------------------------------------------
    base_cols = [
        "rsi",
        "macd",
        "macd_signal",
        "macd_histogram",
        "price_change",
        "price_change_pct",
        "gap_from_open",
        "volume_ratio",
        "intraday_range",
        "market_cap_proxy",
    ]

    required_cols = base_cols + ["signal_strength", "trend_direction", "signal"]
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise KeyError(
            "The following required columns are missing after merging: "
            + ", ".join(sorted(missing))
        )

    df_feat = df[required_cols].copy()

    # -------------------------------------------------------------------
    # 2. Encode *signal_strength* as numeric (STRONG → 1, WEAK → 0)
    # -------------------------------------------------------------------
    df_feat["signal_strength"] = df_feat["signal_strength"].map({"STRONG": 1, "WEAK": 0}).astype(int)

    # -------------------------------------------------------------------
    # 3. One-hot encode *trend_direction*
    # -------------------------------------------------------------------
    trend_dummies = pd.get_dummies(df_feat["trend_direction"], prefix="trend")
    df_feat = pd.concat([df_feat.drop(columns=["trend_direction"]), trend_dummies], axis=1)

    # Ensure a consistent column order: features first, label last
    feature_cols = [c for c in df_feat.columns if c != "signal"]
    ordered_cols = feature_cols + ["signal"]
    df_feat = df_feat[ordered_cols]

    return df_feat.reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI helper for ad-hoc testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover – manual usage only
    import argparse

    ap = argparse.ArgumentParser(description="Merge OHLCV data with signals")
    ap.add_argument("ticker", help="e.g. AAPL")
    ap.add_argument("--folder", dest="folder", help="date folder YYYYMMDD")
    ap.add_argument(
        "--tolerance",
        default="10min",
        help="merge_asof tolerance (default 10min)",
    )
    ns = ap.parse_args()

    df_out = prepare_labeled_dataset(ns.ticker, date_folder=ns.folder, tolerance=ns.tolerance)
    print(df_out.head())
