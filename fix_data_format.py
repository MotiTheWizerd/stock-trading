"""
Fix data format for model training.

This script checks and fixes the data format for model training.
"""

import os
import pandas as pd
from pathlib import Path
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

def fix_data_file(file_path: Path) -> None:
    """Fix data file format."""
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return
    
    logger.info(f"Fixing data file: {file_path}")
    
    # Read the data file
    df = pd.read_csv(file_path)
    
    # Check and fix column names
    if 'Ticker' in df.columns:
        df = df.rename(columns={'Ticker': 'ticker'})
        logger.info("Renamed 'Ticker' column to 'ticker'")
    
    # Ensure datetime column exists and is in the right format
    if 'datetime' not in df.columns:
        logger.error(f"No 'datetime' column found in {file_path}")
        return
    
    # Convert datetime to proper format if it's not already
    try:
        df['datetime'] = pd.to_datetime(df['datetime'])
    except Exception as e:
        logger.warning(f"Could not convert datetime column: {e}")
    
    # Remove any non-numeric columns that might cause issues (except datetime and ticker)
    safe_columns = ['datetime', 'ticker']
    for col in df.columns:
        if col not in safe_columns:
            try:
                # Try to convert to numeric, coerce errors to NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Fill NaN values with 0
                if df[col].isna().any():
                    logger.info(f"Filling NaN values in column '{col}' with 0")
                    df[col] = df[col].fillna(0)
            except Exception as e:
                logger.warning(f"Could not convert column '{col}' to numeric: {e}")
    
    # Save the fixed data
    df.to_csv(file_path, index=False)
    logger.info(f"Fixed data saved to {file_path}")
    
    # Print some info about the data
    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Data columns: {df.columns.tolist()}")

def fix_signals_file(file_path: Path) -> None:
    """Fix signals file format."""
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return
    
    logger.info(f"Fixing signals file: {file_path}")
    
    # Read the signals file
    df = pd.read_csv(file_path)
    
    # Check and fix column names
    if 'Signal' not in df.columns:
        if 'signal' in df.columns:
            df = df.rename(columns={'signal': 'Signal'})
            logger.info("Renamed 'signal' column to 'Signal'")
    
    # Ensure datetime column exists and is in the right format
    if 'datetime' not in df.columns:
        logger.error(f"No 'datetime' column found in {file_path}")
        return
    
    # Convert datetime to proper format if it's not already
    try:
        df['datetime'] = pd.to_datetime(df['datetime'])
    except Exception as e:
        logger.warning(f"Could not convert datetime column: {e}")
    
    # Save the fixed signals
    df.to_csv(file_path, index=False)
    logger.info(f"Fixed signals saved to {file_path}")
    
    # Print some info about the signals
    logger.info(f"Signals shape: {df.shape}")
    logger.info(f"Signals columns: {df.columns.tolist()}")
    if 'Signal' in df.columns:
        logger.info(f"Signal distribution: {df['Signal'].value_counts().to_dict()}")

def process_directory(directory: Path) -> None:
    """Process a directory and fix data and signals files."""
    data_file = directory / "data.csv"
    signals_file = directory / "signals.csv"
    
    if data_file.exists():
        fix_data_file(data_file)
    
    if signals_file.exists():
        fix_signals_file(signals_file)

def main():
    """Main function."""
    # Define base directory
    ticker = "AAPL"
    base_dir = Path("data") / ticker
    
    # Process latest directory
    latest_dir = base_dir / "latest"
    if latest_dir.exists():
        process_directory(latest_dir)
    
    # Process date-based directories
    for date_dir in sorted(base_dir.iterdir()):
        if not date_dir.is_dir() or date_dir.name == "latest":
            continue
        
        process_directory(date_dir)
    
    logger.info(f"✅ Data format fixed for {ticker}")

if __name__ == "__main__":
    main()
