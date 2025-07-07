"""fetch_date_range.py
=======================
Backfill daily data across a date range (inclusive) by repeatedly invoking
`fetch_daily_data.fetch_ticker_data` for every configured ticker.

Usage
-----
# Backfill from a specific start date up to **today** (UTC)
poetry run python fetch_date_range.py --start 20250101

# Backfill an explicit start/end window
poetry run python fetch_date_range.py --start 20250101 --end 20250131

# Limit to a single ticker
poetry run python fetch_date_range.py --start 20250101 --ticker AAPL

The script re-uses all the heavy lifting implemented inside
`fetch_daily_data.py` so that charts, RSI signals, and the full feature-
engineering pipeline are executed exactly the same way as a normal intraday
fetch.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from typing import List

sys.path.append(".")

from config import DEFAULT_TICKERS
from fetch_daily_data import fetch_ticker_data  # Re-use existing logic

DATE_FMT = "%Y%m%d"


def _parse_date(date_str: str, param: str) -> datetime:
    """Parse *date_str* or exit with a helpful error."""
    try:
        return datetime.strptime(date_str, DATE_FMT)
    except ValueError as exc:
        raise SystemExit(f"❌ Invalid {param} '{date_str}', expected YYYYMMDD") from exc


def _date_range(start: datetime, end: datetime) -> List[str]:
    """Yield inclusive list of YYYYMMDD strings between *start* and *end*."""
    current = start
    while current <= end:
        yield current.strftime(DATE_FMT)
        current += timedelta(days=1)


def _parse_args() -> argparse.Namespace:
    today_str = datetime.utcnow().strftime(DATE_FMT)
    parser = argparse.ArgumentParser(
        description="Backfill daily OHLCV data (incl. indicators & charts) for a date range"
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Start date YYYYMMDD (inclusive).",
    )
    parser.add_argument(
        "--end",
        default=today_str,
        help="End date YYYYMMDD (inclusive). Defaults to today UTC if omitted.",
    )
    parser.add_argument(
        "--ticker",
        help="Single ticker to process (default: all configured tickers).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    start_dt = _parse_date(args.start, "--start")
    end_dt = _parse_date(args.end, "--end")

    if start_dt > end_dt:
        raise SystemExit("❌ --start date must be before or equal to --end date")

    dates = list(_date_range(start_dt, end_dt))
    tickers = [args.ticker] if args.ticker else DEFAULT_TICKERS

    print(
        f"\n🚀 Backfill range {dates[0]} → {dates[-1]} (total {len(dates)} days) for {len(tickers)} ticker(s)\n"
    )

    total_success, total_fail = 0, 0

    for date_str in dates:
        print(f"=== {date_str} ===")
        day_success, day_fail = 0, 0
        for ticker in tickers:
            try:
                ok = fetch_ticker_data(ticker, date_str)
                if ok:
                    day_success += 1
                else:
                    day_fail += 1
            except KeyboardInterrupt:
                print("Interrupted by user – exiting.")
                sys.exit(130)
            except Exception as exc:
                print(f"❌ {ticker} @ {date_str}: {exc}")
                day_fail += 1
        print(f"Day summary {date_str}: ✅ {day_success}  ❌ {day_fail}\n")
        total_success += day_success
        total_fail += day_fail

    print(f"\n🎯 Backfill completed: ✅ {total_success}  ❌ {total_fail}\n")


if __name__ == "__main__":
    main()
