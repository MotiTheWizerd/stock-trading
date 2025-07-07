"""Backfill historical intraday data for the last *N* days.

Usage
-----
poetry run python retro_history.py --days 29
poetry run python retro_history.py --start 20250601 --end 20250630

The script re-uses downloader/download_data, append_to_csv, and
utils.plot_candlestick to replicate the same per-day folder structure
(e.g. data/<ticker>/<YYYYMMDD>/). It is essentially an automated loop
around *fetch_all_tickers.py*.
"""
from __future__ import annotations

import os
import sys
import argparse
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional

sys.path.append(".")

from config import DEFAULT_TICKERS
from downloader import download_data, append_to_csv
from utils import plot_candlestick
from signals import detect_rsi_signals
from paths import month_dir, day_csv, signals_csv, charts_dir

DATE_FMT = "%Y%m%d"


def _date_range(start: datetime, end: datetime) -> List[str]:
    """Yield inclusive list of YYYYMMDD strings between *start* and *end*."""
    current = start
    while current <= end:
        yield current.strftime(DATE_FMT)
        current += timedelta(days=1)


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill intraday data for the last N days (default 29).")
    parser.add_argument("--days", type=int, default=29, help="Number of days back from today UTC to fetch (ignored if --start specified).")
    parser.add_argument("--start", help="Start date YYYYMMDD (overrides --days).")
    parser.add_argument("--end", help="End date YYYYMMDD (default today UTC if omitted).")
    parser.add_argument("--period", default="1d", help="yfinance period (default 1d).")
    parser.add_argument("--interval", default="5m", help="yfinance interval (default 5m).")
    return parser.parse_args(argv)


def _parse_date(date_str: str | None, fallback: datetime | None = None) -> datetime:
    if date_str:
        try:
            return datetime.strptime(date_str, DATE_FMT)
        except ValueError as exc:
            raise SystemExit(f"❌ Invalid date '{date_str}', expected YYYYMMDD") from exc
    if fallback is None:
        raise ValueError("fallback must be provided when date_str is None")
    return fallback


def main(argv: List[str] | None = None) -> None:
    args = _parse_args(argv)

    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    if args.start:
        start_dt = _parse_date(args.start)
        end_dt = _parse_date(args.end, fallback=today)
    else:
        end_dt = today
        start_dt = today - timedelta(days=args.days)

    if start_dt > end_dt:
        raise SystemExit("❌ --start date must be before --end date")

    dates = list(_date_range(start_dt, end_dt))
    print(f"\n🕰  Backfill range: {dates[0]} → {dates[-1]} ({len(dates)} days) for {len(DEFAULT_TICKERS)} tickers\n")

    total_success, total_fail = 0, 0

    for date_folder in dates:
        print(f"=== {date_folder} ===")
        day_success, day_fail = 0, 0
        for ticker in DEFAULT_TICKERS:
            try:
                # Try multiple intervals if needed
                intervals = [args.interval]  # Start with the specified interval
                
                # Add fallback intervals if not already included
                if args.interval != "15m":
                    intervals.append("15m")
                if args.interval != "30m":
                    intervals.append("30m")
                if args.interval != "1h":
                    intervals.append("1h")
                
                df = pd.DataFrame()  # Empty dataframe to start
                
                for interval in intervals:
                    if interval == args.interval:
                        print(f"Trying with {interval} interval for {ticker} on {date_folder}")
                    else:
                        print(f"🔄 Retrying with {interval} interval for {ticker} on {date_folder}")
                        
                    df = download_data(
                        ticker=ticker,
                        period=args.period,
                        interval=interval,
                        date_folder=date_folder,
                        start_date=date_folder,
                    )
                    
                    if not df.empty:
                        print(f"✅ Got data with interval {interval}")
                        break
                
                if df.empty:
                    print(f"⚠️ {ticker}: no rows after all attempts – skip")
                    day_fail += 1
                    continue
                    
                append_to_csv(df, ticker, date_folder=date_folder)
                
                # Generate chart and signals
                try:
                    # Ensure chart directory exists
                    chart_dir = charts_dir(ticker, date_folder)
                    os.makedirs(chart_dir, exist_ok=True)
                    print(f"💾 Created chart directory: {chart_dir}")
                    
                    # Calculate RSI before plotting and signal detection
                    from ta.momentum import RSIIndicator
                    rsi = RSIIndicator(close=df['Close'], window=14).rsi()
                    df['RSI'] = rsi
                    
                    # Plot chart
                    plot_candlestick(df, ticker, save_to_file=True, date_folder=date_folder)
                    print(f"📈 Generated chart for {ticker}")
                    
                    # Explicitly generate signals
                    buy_signals, sell_signals, signal_df = detect_rsi_signals(df, ticker, date_folder=date_folder)
                    if signal_df is not None and not signal_df.empty:
                        print(f"🔔 Generated {len(signal_df)} signals for {ticker}")
                    else:
                        print(f"🔕 No signals detected for {ticker}")
                        
                except Exception as e:
                    print(f"❌ Error generating chart/signals for {ticker}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                
                day_success += 1
            except KeyboardInterrupt:
                print("Interrupted by user – exiting.")
                sys.exit(130)
            except Exception as exc:
                print(f"❌ {ticker} @ {date_folder}: {exc}")
                day_fail += 1
        print(f"Day summary {date_folder}: ✅ {day_success}  ❌ {day_fail}\n")
        total_success += day_success
        total_fail += day_fail

    print(f"\n🎯 Finished backfill: ✅ {total_success}  ❌ {total_fail}")


if __name__ == "__main__":
    main()
