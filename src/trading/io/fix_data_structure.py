"""
Fix data structure for model training.

This script copies data from the training directory to the expected directory structure.
"""

import os
import shutil
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

def copy_data(source_dir: Path, target_dir: Path) -> None:
    """Copy data files from source to target directory."""
    # Create target directory if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy data.csv and signals.csv
    data_file = source_dir / "data.csv"
    signals_file = source_dir / "signals.csv"
    
    if data_file.exists():
        shutil.copy2(data_file, target_dir / "data.csv")
        logger.info(f"Copied {data_file} to {target_dir / 'data.csv'}")
    else:
        logger.warning(f"Data file not found: {data_file}")
    
    if signals_file.exists():
        shutil.copy2(signals_file, target_dir / "signals.csv")
        logger.info(f"Copied {signals_file} to {target_dir / 'signals.csv'}")
    else:
        logger.warning(f"Signals file not found: {signals_file}")

def main():
    """Main function."""
    # Define source and target directories
    ticker = "AAPL"
    source_base = Path("data/training") / ticker
    target_base = Path("data") / ticker
    
    # Process latest directory
    latest_source = source_base / "latest"
    latest_target = target_base / "latest"
    
    if latest_source.exists():
        copy_data(latest_source, latest_target)
    
    # Process date-based directories
    for date_dir in sorted(source_base.iterdir()):
        if not date_dir.is_dir() or date_dir.name == "latest":
            continue
        
        date_target = target_base / date_dir.name
        copy_data(date_dir, date_target)
    
    logger.info(f"✅ Data structure fixed for {ticker}")

if __name__ == "__main__":
    main()
