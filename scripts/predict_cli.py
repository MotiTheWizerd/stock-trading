#!/usr/bin/env python
"""Lightweight CLI wrapper around the refactored trading modules.

Usage example:
    python scripts/predict_cli.py --ticker AAPL --simulate

This keeps the old, heavy `scripts/predict.py` intact for historical reasons
while providing a clean entry-point that leans on:
    • trading.utils.logging.setup_logging
    • trading.io.data_loader.load_latest_data
    • trading.models.StockPredictor
    • trading.agents.apply_trader

In a future clean-up pass we can deprecate the legacy script and rename this
back to `predict.py`.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import pandas as pd

from trading.utils.logging import setup_logging
from trading.io.data_loader import load_latest_data
from trading.models import StockPredictor
from trading.agents import apply_trader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Predict stock signals and (optionally) simulate trades.")
    ap.add_argument("--ticker", required=True, help="Stock ticker, e.g. AAPL")
    ap.add_argument("--model-dir", help="Directory with trained models (default: models/<TICKER>)")
    ap.add_argument("--data-dir", default="data", help="Root folder for CSV downloads (default: data)")
    ap.add_argument("--simulate", action="store_true", help="Run SimpleTrader simulation after prediction")
    ap.add_argument("--balance", type=float, default=1_000.0, help="Initial balance for simulation")
    ap.add_argument("--output", help="Optional CSV path to save predictions")
    ap.add_argument("--pretty", action="store_true", help="Render pretty Rich table output")
    return ap.parse_args(argv)


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------


def main() -> int:  # noqa: D401
    args = _parse_args()
    setup_logging()

    # ---------------------------------------------------------------------
    # 1. Load latest enriched features
    # ---------------------------------------------------------------------
    df: pd.DataFrame = load_latest_data(args.ticker, data_dir=args.data_dir)
    if df.empty:
        logger.error("No data available; aborting")
        return 1

    # ---------------------------------------------------------------------
    # 2. Make predictions
    # ---------------------------------------------------------------------
    model_dir = args.model_dir or Path("models") / args.ticker.upper()
    predictor = StockPredictor(args.ticker, model_dir=str(model_dir))
    results = predictor.predict(df)

    # ---------------------------------------------------------------------
    # 3. Optional trading simulation
    # ---------------------------------------------------------------------
    if args.simulate:
        results, trade_log = apply_trader(results, initial_balance=args.balance, return_log=True)

    # ---------------------------------------------------------------------
    # 4. Output
    # ---------------------------------------------------------------------
    pd.set_option("display.max_columns", None)
    if args.pretty:
        from trading.ui.cli_output import render_predictions
        render_predictions(results)
    else:
        print(results.head(20).to_string(index=False))

    # Pretty-print trade log if we simulated and log non-empty
    if args.simulate and trade_log:
        from rich.table import Table
        from rich import box
        table = Table(title="Trade log", box=box.SIMPLE_HEAVY)
        table.add_column("Time", justify="right")
        table.add_column("Action")
        table.add_column("Price", justify="right")
        table.add_column("Equity", justify="right")
        for entry in trade_log:
            table.add_row(
                str(entry.get("timestamp", ""))[:19],
                entry.get("action", ""),
                f"{entry.get('price', 0):.2f}",
                f"{entry.get('equity', 0):.2f}",
            )
        console = __import__("rich").console.Console()
        console.print(table)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(out_path, index=False)
        logger.info("Saved predictions to %s", out_path)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
