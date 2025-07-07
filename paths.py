"""Central path helpers for the Trading Stock Agents project.

This module defines *one* source of truth for how files are organised on
 disk.  Switching layout only requires touching this file.

Layout (v2 – July 2025)
=======================
    data/<TICKER>/<YYYYMM>/
        data-<YYYYMMDD>.csv          # intraday OHLCV per-day
        signals-<YYYYMMDD>.csv       # RSI signal rows (may be empty)
        charts/
            enhanced_chart_<ts>.png
    models/
        <TICKER>/
            model_<timestamp>.pkl
            model_<timestamp>_metadata.json

All other code should use these helpers instead of manual os.path.join.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

__all__ = [
    "get_data_path",
    "get_models_path",
    "get_model_metadata_path",
    "month_dir",
    "day_csv",
    "signals_csv",
    "charts_dir",
    "ensure_dir",
]

# Base directories
BASE_DIR = Path("data")
MODELS_DIR = Path("models")

def ensure_dir(path: Path) -> Path:
    """Ensure directory exists and return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_data_path(ticker: str = None) -> Path:
    """Get the data directory path.
    
    Args:
        ticker: Optional ticker symbol. If provided, returns the path to the ticker's data directory.
    """
    if ticker:
        return ensure_dir(BASE_DIR / ticker.upper())
    return ensure_dir(BASE_DIR)

def get_models_path(ticker: str = None) -> Path:
    """Get the models directory path.
    
    Args:
        ticker: Optional ticker symbol. If provided, returns the path to the ticker's models directory.
    """
    if ticker:
        return ensure_dir(MODELS_DIR / ticker.upper())
    return ensure_dir(MODELS_DIR)

def get_model_metadata_path(model_path: Path) -> Path:
    """Get the metadata path for a given model path."""
    return model_path.with_stem(f"{model_path.stem}_metadata").with_suffix(".json")

def get_ticker_models_dir(ticker: str) -> Path:
    """Get the directory path for a specific ticker's models."""
    return ensure_dir(get_models_path() / ticker.upper())

def get_latest_model_path(ticker: str) -> Path | None:
    """Get the path to the latest trained model for a ticker."""
    ticker_dir = get_ticker_models_dir(ticker)
    if not ticker_dir.exists():
        return None
    
    model_files = list(ticker_dir.glob("model_*.pkl"))
    if not model_files:
        return None
        
    # Sort by modification time, newest first
    return max(model_files, key=lambda p: p.stat().st_mtime)


def month_dir(ticker: str, yyyymmdd: str) -> Path:
    """Return directory *data/<ticker>/<YYYYMM>* (create absent)."""
    month = yyyymmdd[:6]
    path = BASE_DIR / ticker / month
    path.mkdir(parents=True, exist_ok=True)
    return path


def day_csv(ticker: str, yyyymmdd: str) -> Path:
    """Path to daily OHLCV CSV file."""
    return month_dir(ticker, yyyymmdd) / f"data-{yyyymmdd}.csv"


def signals_csv(ticker: str, yyyymmdd: str) -> Path:
    """Path to RSI signals CSV."""
    return month_dir(ticker, yyyymmdd) / f"signals-{yyyymmdd}.csv"


def charts_dir(ticker: str, yyyymmdd: str) -> Path:
    """Directory to store charts for *yyyymmdd* (inside same month folder)."""
    d = month_dir(ticker, yyyymmdd) / "charts"
    d.mkdir(exist_ok=True)
    return d
