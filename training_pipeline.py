"""Dynamic Training Pipeline for Multiple Tickers
-------------------------------------------------
This script loads `data.csv` (OHLCV quotes) and `signals.csv` (technical
features + labelled trading signals), automatically detects the list of
tickers contained in those files, and trains a separate `RandomForest`
classification model for each ticker.

Usage
-----
$ python training_pipeline.py  # trains models for all tickers

Configuration
-------------
By default the script expects `data.csv` and `signals.csv` to live in the
current working directory, with at least the following columns:

* `datetime` – timestamp (will be parsed to pandas `datetime64` and used
  as merge/index key)
* `ticker`   – symbol of the asset (e.g. "AAPL")
* OHLCV columns: `Open`, `High`, `Low`, `Close`, `Volume`
* feature columns such as `RSI`, `MACD`, `price_change`, `signal_strength`, …
* target column `signal` containing the class label to predict (e.g.
  "BUY", "SELL", "NONE")

The script is **fully dynamic** – it will work with an arbitrary number
of tickers and any feature-column combination. No hard-coding required.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _safe_mkdir(path: Path) -> None:
    """Create *path* (directory) if it doesn't exist."""
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def load_datasets(data_path: Path | str, signals_path: Path | str) -> pd.DataFrame:
    """Read *data.csv* and *signals.csv* and return merged DataFrame.

    The merge is performed on `datetime` + `ticker` keys using an inner
    join so that only overlapping rows are used for training.
    """
    print("📥 Loading CSVs …")
    data_df = pd.read_csv(data_path, parse_dates=["datetime"])  # type: ignore[arg-type]
    signals_df = pd.read_csv(signals_path, parse_dates=["datetime"])  # type: ignore[arg-type]

    # Sanity-check mandatory columns
    required_cols = {"datetime", "ticker"}
    if not required_cols.issubset(data_df.columns) or not required_cols.issubset(signals_df.columns):
        missing = required_cols - set(data_df.columns) - set(signals_df.columns)
        raise ValueError(f"Missing required columns {missing} in data/signals CSVs.")

    print(f"   • data.csv shape   : {data_df.shape}")
    print(f"   • signals.csv shape: {signals_df.shape}")

    # Merge on datetime + ticker
    merged = (
        pd.merge(data_df, signals_df, on=["datetime", "ticker"], how="inner", suffixes=("", "_sig"))
        .sort_values(["ticker", "datetime"])
        .reset_index(drop=True)
    )
    print(f"🔗 Merged shape: {merged.shape}")
    return merged


def split_by_ticker(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Return dict {ticker -> dataframe} where each dataframe contains only rows for the ticker."""
    grouped: Dict[str, pd.DataFrame] = {t: grp.copy() for t, grp in df.groupby("ticker", observed=True)}
    print(f"🪙 Detected tickers: {', '.join(grouped.keys())}")
    return grouped


def build_preprocessing_pipeline(df: pd.DataFrame) -> Tuple[Pipeline, List[str]]:
    """Create and return a scikit-learn ColumnTransformer preprocessing pipeline.

    Numeric columns → median imputation
    Categorical columns → impute "UNKNOWN" then OneHotEncode (handle_unknown="ignore")
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # We will *not* include target column "signal" in features – it will be dropped later.
    if "signal" in num_cols:
        num_cols.remove("signal")
    if "signal" in cat_cols:
        cat_cols.remove("signal")

    print(f"🔧 Feature columns: {len(num_cols)} numeric, {len(cat_cols)} categorical")

    numeric_transformer = SimpleImputer(strategy="median")
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
    )

    # ColumnTransformer modifies column order – we return the lists so that caller can access.
    feature_cols: List[str] = num_cols + cat_cols
    return Pipeline(steps=[("preprocess", preprocessor)]), feature_cols


def train_model(df: pd.DataFrame, output_dir: Path) -> None:
    """Train & persist a RandomForest model for *one* ticker.

    The dataframe *df* must contain the `signal` column (target)!
    """
    ticker = df["ticker"].iloc[0]
    print(f"\n🚀 Training model for {ticker} …")

    # Separate features & target
    X = df.drop(columns=["signal", "datetime", "ticker"], errors="ignore")
    y = df["signal"].fillna("NONE")  # Ensure target has no NaNs

    # Build preprocessing pipeline (fitted on training data!)
    preprocessing, feature_cols = build_preprocessing_pipeline(X)

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42,
    )

    pipe = Pipeline([
        ("pre", preprocessing),
        ("model", clf),
    ])

    pipe.fit(X, y)
    print("✅ Training finished.")

    # Optional: quick in-sample performance snapshot
    preds = pipe.predict(X)
    print(classification_report(y, preds, zero_division=0))

    # Persist model
    date_folder = pd.Timestamp.utcnow().strftime("%Y%m%d")

    # If caller passes the *ticker directory* itself as output_dir we save directly there
    if output_dir.name.lower() == ticker.lower():
        model_dir = output_dir  # no nested folder
    else:
        model_dir = output_dir / ticker / date_folder
        _safe_mkdir(model_dir)

    if not model_dir.exists():
        _safe_mkdir(model_dir)

    model_path = model_dir / f"model_{ticker}.joblib"
    joblib.dump(pipe, model_path)
    print(f"💾 Saved model → {model_path.relative_to(Path.cwd())}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Dynamic training pipeline for multiple tickers.")
    parser.add_argument("--data", default="data.csv", help="Path to OHLCV data CSV (default: data.csv)")
    parser.add_argument("--signals", default="signals.csv", help="Path to signals CSV (default: signals.csv)")
    parser.add_argument(
        "--models-dir", default="models", help="Directory where trained models will be saved (default: ./models)"
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    signals_path = Path(args.signals)
    output_dir = Path(args.models_dir)

    if not data_path.exists() or not signals_path.exists():
        raise FileNotFoundError("data.csv and/or signals.csv not found. Please provide correct paths via --data / --signals.")

    merged_df = load_datasets(data_path, signals_path)
    by_ticker = split_by_ticker(merged_df)

    for df in by_ticker.values():
        train_model(df, output_dir)

    print("\n🎉 All models trained and saved.")


if __name__ == "__main__":
    main()
