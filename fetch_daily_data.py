"""
fetch_daily_data.py
==================
Fetches daily stock data, generates signals, and creates charts.

Usage:
    python fetch_daily_data.py [--date YYYYMMDD] [--ticker TICKER]

If no date is provided, uses current date.
If no ticker is provided, processes all tickers from config.
"""

import argparse
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys
from typing import List, Optional

sys.path.append(".")

from config import DEFAULT_TICKERS
from downloader import download_data, append_to_csv
from utils.plotting import plot_candlestick
from signals import detect_rsi_signals
from paths import month_dir, day_csv, signals_csv, charts_dir

def fetch_ticker_data(ticker: str, date_str: str) -> bool:
    """Fetch data for a single ticker and date."""
    print(f"\n📊 Fetching data for {ticker} - {date_str}")
    
    # 1. Download data - try multiple intervals if needed
    intervals = ["5m", "15m", "30m", "1h"]
    
    for interval in intervals:
        print(f"Trying interval: {interval}")
        data = download_data(
            ticker=ticker,
            period="1d",  # Get 1 day of data
            interval=interval,
            date_folder=date_str
        )
        
        if not data.empty:
            print(f"✅ Got data with interval {interval}")
            break
    
    if data.empty:
        print(f"❌ No data returned for {ticker} after trying multiple intervals")
        return False
    
    # 2. Append to CSV (will create the folder if needed)
    append_to_csv(data, ticker, date_folder=date_str)

    # 3. Reload the *final* cleaned CSV so indicators & signals work first time
    csv_path = day_csv(ticker, date_str)
    try:
        data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        # Flatten MultiIndex columns if present so 'Close' resolves to a Series
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]
        if isinstance(data['Close'], pd.DataFrame):
            data['Close'] = data['Close'].iloc[:, 0]
    except Exception as e:
        print(f"❌ Failed to reload CSV {csv_path}: {e}")
        return False

    # 4. Ensure RSI column exists on this reloaded data
    from ta.momentum import RSIIndicator
    data['RSI'] = RSIIndicator(data['Close'].astype(float), window=14).rsi()
    print(f"RSI calculated: min={data['RSI'].min():.2f}, max={data['RSI'].max():.2f}")

    # 5. Generate signals *before* chart so the CSV exists even if plotting fails
    try:
        _, _, signal_df = detect_rsi_signals(data, ticker, date_folder=date_str)
        if signal_df is not None and len(signal_df) > 0:
            print(f"🔔 Saved {len(signal_df)} signals for {ticker}")
        else:
            print(f"🔕 No signals detected for {ticker}")
    except Exception as e:
        print(f"❌ Error generating signals for {ticker}: {str(e)}")
        import traceback, sys
        traceback.print_exc()

    # 6. Finally generate the chart (optional failure)
    try:
        chart_path = plot_candlestick(data, ticker, save_to_file=True, date_folder=date_str)
        if chart_path:
            print(f"✅ Generated chart at {chart_path}")
    except Exception as e:
        print(f"⚠️  Chart generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Fetch daily stock data and generate signals")
    parser.add_argument(
        "--date",
        type=str,
        help="Date in YYYYMMDD format (default: today)",
        default=datetime.now().strftime("%Y%m%d")
    )
    parser.add_argument(
        "--ticker",
        type=str,
        help="Single ticker to process (default: all tickers)",
        default=None
    )
    
    args = parser.parse_args()
    
    # Validate date format
    try:
        datetime.strptime(args.date, "%Y%m%d")
    except ValueError:
        print(f"❌ Invalid date format: {args.date}. Use YYYYMMDD format.")
        sys.exit(1)
    
    # Get list of tickers to process
    tickers = [args.ticker] if args.ticker else DEFAULT_TICKERS
    
    print(f"🚀 Starting data fetch for {len(tickers)} tickers on {args.date}")
    
    success_count = 0
    for ticker in tickers:
        if fetch_ticker_data(ticker, args.date):
            success_count += 1
    
    print(f"\n✅ Completed: {success_count}/{len(tickers)} tickers processed successfully")

if __name__ == "__main__":
    main()
