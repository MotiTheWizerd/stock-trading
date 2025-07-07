"""Fetch data for *all* configured tickers for a specific date.

Example
-------
poetry run python fetch_all_tickers.py --date 20250707

The script iterates over `config.TICKERS`, downloads intraday data via
`downloader.download_data`, appends it to CSV, and generates charts &
RSI-based signals – just like `fetch_daily_data.py`, but for the full
list instead of a single ticker.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from typing import List

sys.path.append(".")

from downloader import download_data, append_to_csv
from utils import plot_candlestick
from config import DEFAULT_TICKERS
from paths import month_dir, day_csv, signals_csv, charts_dir


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch intraday OHLCV for ALL tickers on a given date (YYYYMMDD).",
    )
    parser.add_argument(
        "--date",
        required=False,
        default=datetime.utcnow().strftime("%Y%m%d"),
        help="Target date folder in YYYYMMDD (default: today UTC).",
    )
    parser.add_argument(
        "--period",
        default="1d",
        help="yfinance period argument (default: 1d).",
    )
    parser.add_argument(
        "--interval",
        default="5m",
        help="yfinance interval argument (default: 5m).",
    )
    return parser.parse_args(argv)


def _validate_date_str(date_str: str) -> None:
    try:
        datetime.strptime(date_str, "%Y%m%d")
    except ValueError as exc:
        raise SystemExit(f"❌ Invalid --date format '{date_str}'. Expected YYYYMMDD.") from exc


def main(argv: List[str] | None = None) -> None:
    args = _parse_args(argv)
    _validate_date_str(args.date)

    date_folder = args.date
    print(f"\n🚀 Starting bulk fetch for {len(DEFAULT_TICKERS)} tickers on {date_folder}\n")

    success, failures = 0, 0
    for ticker in DEFAULT_TICKERS:
        try:
            print(f"📡 Processing {ticker} …")
            df = download_data(
                ticker=ticker,
                period=args.period,
                interval=args.interval,
                date_folder=date_folder,
            )
            if df.empty:
                print(f"⚠️ {ticker}: No rows returned – skipping.")
                failures += 1
                continue

            append_to_csv(df, ticker, date_folder=date_folder)
            plot_candlestick(df, ticker, save_to_file=True, date_folder=date_folder)
            success += 1
        except KeyboardInterrupt:
            print("Interrupted by user.")
            sys.exit(130)
        except Exception as exc:  # pragma: no cover – log and continue
            print(f"❌ {ticker}: {exc}")
            failures += 1

    print(f"\n🎉 Finished. ✅ {success} succeeded, ❌ {failures} failed.")


if __name__ == "__main__":
    main()
