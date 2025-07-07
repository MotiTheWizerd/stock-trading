"""Generate trading signals from a trained model.

Usage
-----
$ python generate_signals.py --ticker AAPL --date 20250707

This script will:
1. Load intraday OHLCV feature data from
   `data/<TICKER>/<YYYYMMDD>/data.csv`.
2. Load the corresponding scikit-learn model from
   `models/<TICKER>/model.joblib` (fallback: latest model_*.joblib nested
   anywhere below that folder).
3. Predict a `signal` for every row.
4. Save `signals.csv` next to the input data file.

The script intentionally contains *no* logic specific to a particular
feature set or model type – the pipeline returned by `joblib.load` is
assumed to perform every necessary preprocessing step.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _find_model_path(models_root: Path, ticker: str) -> Path:
    """Return path to *the* model file for *ticker*.

    The preferred location is `models/<ticker>/model.joblib`.  If that
    path does not exist we fall back to the **most recently modified**
    `*.joblib` file anywhere below `models/<ticker>` (e.g. when models
    are stored in dated sub-folders).
    """
    direct = models_root / ticker / "model.joblib"
    if direct.exists():
        return direct

    # Fallback – search recursively for joblib files
    candidates = sorted(
        (p for p in (models_root / ticker).rglob("*.joblib")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No .joblib model found for {ticker} below {models_root / ticker}."
        )
    return candidates[0]


def _ensure_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found → {path}")


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def generate_signals(
    ticker: str,
    date: str,
    models_dir: Path = Path("models"),
    data_root: Path = Path("data"),
    out_filename: str = "signals.csv",
) -> Path:
    """Predict trading *signals* for *ticker* on *date* (YYYYMMDD).

    Returns the path of the written signals CSV.
    """
    # Resolve paths
    day_dir = data_root / ticker / date
    data_path = day_dir / "data.csv"
    model_path = _find_model_path(models_dir, ticker)
    signals_path = day_dir / out_filename

    _ensure_exists(data_path, "Input data.csv")
    _ensure_exists(model_path, "Trained model")

    print(f"📥 Loading feature data → {data_path.relative_to(Path.cwd())}")
    df = pd.read_csv(data_path, parse_dates=["datetime"], low_memory=False)

    if "datetime" not in df.columns:
        raise ValueError("data.csv must contain a `datetime` column.")

    # Do NOT drop columns – the pipeline should handle preprocessing.
    X = df.drop(columns=[], errors="ignore")  # defensive placeholder

    print(f"🔮 Loading model → {model_path.relative_to(Path.cwd())}")
    pipe = joblib.load(model_path)

    print("🚀 Predicting signals …")
    preds = pipe.predict(X)

    signals_df = pd.DataFrame({
        "datetime": df["datetime"],
        "signal": preds,
    })

    signals_df.to_csv(signals_path, index=False)
    print(f"💾 Saved signals → {signals_path.relative_to(Path.cwd())}")

    return signals_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate trading signals from a trained model.")
    parser.add_argument("--ticker", required=True, help="Ticker symbol, e.g. AAPL")
    parser.add_argument("--date", required=True, help="Trading date in YYYYMMDD format")
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory containing per-ticker model folders (default: ./models)",
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Root directory where <ticker>/<YYYYMMDD>/data.csv lives (default: ./data)",
    )
    args = parser.parse_args(argv)

    try:
        generate_signals(
            ticker=args.ticker.upper(),
            date=args.date,
            models_dir=Path(args.models_dir),
            data_root=Path(args.data_root),
        )
    except Exception as exc:
        print(f"❌ Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
