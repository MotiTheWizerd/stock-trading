"""
create_daily_folders.py
=======================
Creates date-based folder structure for all tickers in the format:
data/<TICKER>/<YYYYMMDD>/{data.csv, signals.csv, charts/}

Usage:
    python create_daily_folders.py [--date YYYYMMDD]
    
If no date is provided, uses current date.
"""

import os
import argparse
from datetime import datetime
from pathlib import Path
from config import TICKERS  # Import tickers from central config

def create_daily_folders(date_str: str = None) -> None:
    """
    Create date-based folder structure for all tickers.
    
    Args:
        date_str: Date in YYYYMMDD format (optional, defaults to today)
    """
    # Use provided date or today's date
    if date_str:
        try:
            date_obj = datetime.strptime(date_str, "%Y%m%d")
        except ValueError:
            print(f"❌ Invalid date format: {date_str}. Use YYYYMMDD format.")
            return
    else:
        date_obj = datetime.now()
        date_str = date_obj.strftime("%Y%m%d")
    
    print(f"📅 Creating folder structure for date: {date_str}")
    
    # Create base data directory if it doesn't exist
    base_dir = Path("data")
    base_dir.mkdir(exist_ok=True)
    
    for ticker in TICKERS:
        # Create ticker directory if it doesn't exist
        ticker_dir = base_dir / ticker
        ticker_dir.mkdir(exist_ok=True)
        
        # Create date directory
        date_dir = ticker_dir / date_str
        date_dir.mkdir(exist_ok=True)
        
        # Create charts subdirectory
        charts_dir = date_dir / "charts"
        charts_dir.mkdir(exist_ok=True)
        
        # Create empty data.csv and signals.csv if they don't exist
        data_file = date_dir / "data.csv"
        signals_file = date_dir / "signals.csv"
        
        if not data_file.exists():
            data_file.touch()
            print(f"✅ Created empty data.csv for {ticker} - {date_str}")
            
        if not signals_file.exists():
            signals_file.touch()
            print(f"✅ Created empty signals.csv for {ticker} - {date_str}")

def main():
    parser = argparse.ArgumentParser(description="Create daily folder structure for stock data")
    parser.add_argument(
        "--date",
        type=str,
        help="Date in YYYYMMDD format (default: today)",
        default=None
    )
    
    args = parser.parse_args()
    create_daily_folders(args.date)

if __name__ == "__main__":
    main()
