"""
Create RSI-based trading signals for model training.

This script generates proper BUY/SELL signals based on RSI values for training data.
It will:
1. Load data from the specified directory
2. Calculate RSI if not already present
3. Generate signals based on RSI thresholds (BUY when RSI < 30, SELL when RSI > 70)
4. Save signals to a signals.csv file in the same directory

Usage:
    python create_rsi_signals.py --ticker=AAPL --data-dir=data/training
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

def calculate_rsi(data: pd.DataFrame, window: int = 14, column: str = 'Close') -> pd.Series:
    """Calculate RSI for a given DataFrame."""
    # Make sure we have the right column
    if column not in data.columns:
        column = column.lower()
        if column not in data.columns:
            raise ValueError(f"Column {column} not found in data")
    
    # Calculate price changes
    delta = data[column].diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gain and loss
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def generate_signals(df: pd.DataFrame, rsi_column: str = 'RSI_14', 
                    oversold: float = 30, overbought: float = 70) -> pd.DataFrame:
    """Generate BUY/SELL signals based on RSI thresholds."""
    # Make sure we have RSI column
    if rsi_column not in df.columns:
        logger.info(f"Calculating {rsi_column}...")
        df[rsi_column] = calculate_rsi(df)
    
    # Initialize signals DataFrame
    signals = pd.DataFrame()
    signals['datetime'] = df['datetime']
    signals['RSI_Value'] = df[rsi_column]
    
    # Generate signals based on RSI thresholds
    signals['Signal'] = 'NONE'
    signals.loc[signals['RSI_Value'] < oversold, 'Signal'] = 'BUY'
    signals.loc[signals['RSI_Value'] > overbought, 'Signal'] = 'SELL'
    
    # Count signals
    buy_count = (signals['Signal'] == 'BUY').sum()
    sell_count = (signals['Signal'] == 'SELL').sum()
    none_count = (signals['Signal'] == 'NONE').sum()
    
    logger.info(f"Generated signals: BUY: {buy_count}, SELL: {sell_count}, NONE: {none_count}")
    
    return signals

def process_directory(ticker: str, data_dir: str, 
                     rsi_column: str = 'RSI_14', 
                     oversold: float = 30, 
                     overbought: float = 70) -> None:
    """Process all data files in a directory and generate signals."""
    data_path = Path(data_dir) / ticker
    
    if not data_path.exists():
        raise FileNotFoundError(f"Directory not found: {data_path}")
    
    # Process latest directory if it exists
    latest_dir = data_path / "latest"
    if latest_dir.exists():
        process_data_directory(latest_dir, rsi_column, oversold, overbought)
    
    # Process date-based directories
    for date_dir in sorted(data_path.iterdir()):
        if not date_dir.is_dir() or date_dir.name == "latest":
            continue
        
        process_data_directory(date_dir, rsi_column, oversold, overbought)
    
    logger.info(f"✅ Completed signal generation for {ticker}")

def process_data_directory(directory: Path, 
                          rsi_column: str = 'RSI_14',
                          oversold: float = 30, 
                          overbought: float = 70) -> None:
    """Process a single data directory and generate signals."""
    data_file = directory / "data.csv"
    signals_file = directory / "signals.csv"
    
    if not data_file.exists():
        logger.warning(f"Data file not found: {data_file}")
        return
    
    logger.info(f"Processing {data_file}...")
    
    # Load data
    df = pd.read_csv(data_file, parse_dates=["datetime"])
    
    # Generate signals
    signals = generate_signals(df, rsi_column, oversold, overbought)
    
    # Save signals
    signals.to_csv(signals_file, index=False)
    logger.info(f"Saved signals to {signals_file}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Create RSI-based trading signals for model training.")
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker symbol")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing data files")
    parser.add_argument("--rsi-column", type=str, default="RSI_14", help="Column name for RSI values")
    parser.add_argument("--oversold", type=float, default=30, help="RSI threshold for oversold (BUY signal)")
    parser.add_argument("--overbought", type=float, default=70, help="RSI threshold for overbought (SELL signal)")
    
    args = parser.parse_args()
    
    try:
        process_directory(
            args.ticker, 
            args.data_dir, 
            args.rsi_column, 
            args.oversold, 
            args.overbought
        )
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
