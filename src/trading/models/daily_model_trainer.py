from __future__ import annotations

"""Daily Model Trainer
======================
Runs once per day (configurable time) and trains / updates a RandomForest
classifier for each ticker found in the *root* directory structure:

<root>/<TICKER>/<YYYYMMDD>/data.csv
<root>/<TICKER>/<YYYYMMDD>/signals.csv

All historical `data.csv` & `signals.csv` files are aggregated, merged on
`datetime` and passed to the dynamic training pipeline from
`training_pipeline.py`.

Usage
-----
$ python daily_model_trainer.py              # trains at default 02:00 UTC
$ python daily_model_trainer.py --time 09:30 # trains every day 09:30 UTC
$ python daily_model_trainer.py --root my_data_dir

Notes
-----
* Only rows that exist in **both** data & signals files are used for
  training (inner merge).
* Missing *signal* values are filled with "NONE"; numerical columns with
  median; categorical with "UNKNOWN" (handled inside
  ``training_pipeline.train_model`` helper functions).
* A single corrupt / missing file is logged & skipped – it will **not**
  abort the whole run.
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
import schedule

# Optional progress bar – falls back gracefully if package missing
try:
    from tqdm.auto import tqdm  # type: ignore
except ImportError:  # pragma: no cover
    def tqdm(x, *args, **kwargs):  # type: ignore  # noqa: D401
        """tqdm fallback that just returns the iterable unchanged."""
        return x

# We import *runtime* because the project may not yet be installed as pkg.
import training_pipeline as tp  # type: ignore

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _collect_csv_files(ticker_path: Path) -> List[tuple[Path, Path]]:
    """Return list of (data_csv, signals_csv) tuples available for *ticker_path*.

    A tuple is included only if **both** files exist inside the same date
    sub-folder.
    """
    pairs: List[tuple[Path, Path]] = []
    for date_dir in sorted(ticker_path.iterdir()):
        if not date_dir.is_dir():
            continue
        data_file = date_dir / "data.csv"
        sig_file = date_dir / "signals.csv"
        if data_file.exists() and sig_file.exists():
            pairs.append((data_file, sig_file))
        else:
            missing = []
            if not data_file.exists():
                missing.append("data.csv")
            if not sig_file.exists():
                missing.append("signals.csv")
            logger.warning("%s – missing %s → skipped", date_dir.name, ", ".join(missing))
    return pairs


def _load_merged_dataframe(pairs: List[tuple[Path, Path]], ticker: str) -> pd.DataFrame | None:
    """Load, merge and concatenate all (data.csv, signals.csv) *pairs*.

    Returns a DataFrame with mandatory columns [`datetime`, `ticker`,
    `signal`]. If no valid pairs are provided an empty *None* is returned.
    """
    if not pairs:
        return None

    data_frames: list[pd.DataFrame] = []
    signals_frames: list[pd.DataFrame] = []

    for data_file, sig_file in pairs:
        try:
            df_d = pd.read_csv(data_file, parse_dates=["datetime"])
            df_s = pd.read_csv(sig_file, parse_dates=["datetime"])
        except Exception as exc:  # noqa: BLE001
            logger.error("Could not read %s or %s → %s", data_file, sig_file, exc)
            continue

        if "ticker" not in df_d.columns:
            df_d["ticker"] = ticker
        if "ticker" not in df_s.columns:
            df_s["ticker"] = ticker

        data_frames.append(df_d)
        signals_frames.append(df_s)

    if not data_frames or not signals_frames:
        return None

    data_df = pd.concat(data_frames, ignore_index=True)
    signals_df = pd.concat(signals_frames, ignore_index=True)

    merged = (
        pd.merge(data_df, signals_df, on=["datetime", "ticker"], how="inner", suffixes=("", "_sig"))
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    return merged

# ---------------------------------------------------------------------------
# Training job
# ---------------------------------------------------------------------------

def train_all_tickers(root: Path, models_dir: Path) -> None:
    """Iterate over ticker directories under *root* and train models."""
    logger.info("🚀 Starting training run (root=%s)", root)

    if not root.exists():
        logger.error("Root directory %s does not exist – aborting run.", root)
        return

    ticker_dirs = [p for p in root.iterdir() if p.is_dir()]

    if not ticker_dirs:
        logger.warning("No ticker directories found under %s.", root)
        return

    # Progress bar over tickers
    for ticker_dir in tqdm(ticker_dirs, desc="Tickers", unit="ticker"):
        ticker = ticker_dir.name
        logger.info("\n📂 Processing %s", ticker)
        pairs = _collect_csv_files(ticker_dir)
        if not pairs:
            logger.warning("No valid data+signals CSV pairs for %s – skipped.", ticker)
            continue

        merged_df = _load_merged_dataframe(pairs, ticker)
        if merged_df is None or merged_df.empty:
            logger.warning("Merged dataframe empty for %s – skipped.", ticker)
            continue

        # Ensure signal column present; fill missing as NONE (prep step)
        if "signal" not in merged_df.columns:
            logger.error("Column 'signal' missing for %s – skipped.", ticker)
            continue
        merged_df["signal"].fillna("NONE", inplace=True)

        try:
            # Save model directly into the ticker's dataset folder
            tp.train_model(merged_df, ticker_dir)
            logger.info("✅ Model trained for %s (rows=%d)", ticker, len(merged_df))
        except Exception as exc:  # noqa: BLE001
            logger.exception("❌ Failed training %s → %s", ticker, exc)

# ---------------------------------------------------------------------------
# Scheduler glue
# ---------------------------------------------------------------------------

def _schedule_job(run_time: str, root: Path, models_dir: Path) -> None:
    """Internal: register and start the daily schedule."""
    schedule.clear()
    schedule.every().day.at(run_time).do(train_all_tickers, root=root, models_dir=models_dir)
    logger.info("🔔 Training will run daily at %s UTC", run_time)

    try:
        while True:
            schedule.run_pending()
            time.sleep(30)
    except KeyboardInterrupt:
        logger.info("Scheduler interrupted by user – exiting.")

# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: D401
    """Parse CLI args & start scheduler."""
    parser = argparse.ArgumentParser(description="Daily ML model trainer / scheduler.")
    parser.add_argument("--time", default="02:00", help="HH:MM (24h) daily run time, UTC (default 02:00)")
    parser.add_argument("--root", default=".", help="Root directory containing ticker folders (default: current dir)")
    parser.add_argument("--models-dir", default="models", help="Output directory for trained models")
    parser.add_argument("--once", action="store_true", help="Run immediately and exit (no scheduler)")
    args = parser.parse_args()

    run_time = args.time
    try:
        datetime.strptime(run_time, "%H:%M")
    except ValueError as err:
        parser.error(f"--time must be HH:MM 24-hour format – {err}")

    root_path = Path(args.root).expanduser().resolve()
    models_dir = Path(args.models_dir).expanduser().resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    if args.once:
        train_all_tickers(root_path, models_dir)
        return

    _schedule_job(run_time, root_path, models_dir)


if __name__ == "__main__":
    main()
