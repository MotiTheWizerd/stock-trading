"""
Prepare training data by consolidating daily data and signals files.

This script:
1. Scans through all date directories for a given ticker
2. Combines daily data files into a single data.csv
3. Combines daily signals files into a single signals.csv
4. Saves the consolidated files in the ticker's directory

Usage:
    python scripts/prepare_training_data.py --ticker=AAPL
"""
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_data_files(ticker: str, base_dir: Path) -> Tuple[List[Path], List[Path]]:
    """Find all data and signals files for a ticker."""
    ticker_dir = base_dir / ticker
    if not ticker_dir.exists():
        raise FileNotFoundError(f"No data directory found for {ticker}")
    
    data_files = []
    signals_files = []
    
    # Find all date directories
    for date_dir in ticker_dir.iterdir():
        if not date_dir.is_dir():
            continue
            
        # Find all data and signals files in this date directory
        date_data_files = sorted(date_dir.glob("data-*.csv"))
        date_signals_files = sorted(date_dir.glob("signals-*.csv"))
        
        if date_data_files:
            data_files.extend(date_data_files)
        if date_signals_files:
            signals_files.extend(date_signals_files)
    
    return data_files, signals_files

def load_and_combine_files(files: List[Path]) -> Optional[pd.DataFrame]:
    """Load and combine multiple CSV files into a single DataFrame."""
    if not files:
        return None
        
    dfs = []
    for file in files:
        try:
            df = pd.read_csv(file)
            if not df.empty:
                dfs.append(df)
                logger.info(f"Loaded {file} with {len(df)} rows")
        except Exception as e:
            logger.warning(f"Error loading {file}: {e}")
    
    if not dfs:
        return None
        
    return pd.concat(dfs, ignore_index=True)

def save_combined_data(df: pd.DataFrame, output_path: Path) -> bool:
    """Save combined data to a CSV file with proper formatting."""
    try:
        # Ensure the output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert datetime if exists
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime')
        
        # Save with index=False to avoid writing row numbers
        df.to_csv(output_path, index=False)
        logger.info(f"Saved combined data to {output_path} ({len(df)} rows)")
        return True
    except Exception as e:
        logger.error(f"Error saving to {output_path}: {e}")
        return False

def setup_training_directories(base_dir: Path, ticker: str) -> Path:
    """Create the directory structure required by the training pipeline."""
    # Create a dated directory for training data (using current date)
    today = datetime.now().strftime('%Y%m%d')
    training_dir = base_dir / ticker / today
    training_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a symlink to the latest training data
    latest_link = base_dir / ticker / 'latest'
    if latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(today, target_is_directory=True)
    
    return training_dir

def main():
    parser = argparse.ArgumentParser(description='Prepare training data by combining daily files')
    parser.add_argument('--ticker', type=str, required=True, help='Stock ticker symbol (e.g., AAPL)')
    parser.add_argument('--data-dir', type=Path, default=Path('data'), help='Base data directory')
    parser.add_argument('--output-dir', type=Path, help='Output directory (default: data/[TICKER]/[DATE])')
    parser.add_argument('--date', type=str, help='Date in YYYYMMDD format (default: today)')
    
    args = parser.parse_args()
    
    # Set up output directory structure
    if args.output_dir:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Use the standard training data directory structure
        training_data_dir = args.data_dir / 'training'
        training_data_dir.mkdir(exist_ok=True)
        
        # Create dated directory for this training run
        date_str = args.date or datetime.now().strftime('%Y%m%d')
        output_dir = training_data_dir / args.ticker / date_str
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a 'latest' symlink
        latest_link = training_data_dir / args.ticker / 'latest'
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(date_str, target_is_directory=True)
    
    logger.info(f"Preparing training data for {args.ticker}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Find all data and signals files
        data_files, signals_files = find_data_files(args.ticker, args.data_dir)
        
        if not data_files:
            logger.error(f"No data files found for {args.ticker}")
            return
            
        logger.info(f"Found {len(data_files)} data files and {len(signals_files)} signals files")
        
        # Combine data files
        combined_data = load_and_combine_files(data_files)
        if combined_data is not None and not combined_data.empty:
            # Save combined data in the training directory structure
            save_combined_data(combined_data, output_dir / 'data.csv')
            
            # Also save in the format expected by the training script
            if 'datetime' in combined_data.columns:
                # Extract date for subdirectory
                combined_data['date'] = pd.to_datetime(combined_data['datetime']).dt.date
                
                # Group by date and save in date-based subdirectories
                for date, group in combined_data.groupby('date'):
                    date_str = date.strftime('%Y%m%d')
                    date_dir = output_dir / date_str
                    date_dir.mkdir(exist_ok=True)
                    save_combined_data(group.drop('date', axis=1), date_dir / 'data.csv')
        
        # Combine signals files if they exist
        if signals_files:
            combined_signals = load_and_combine_files(signals_files)
            if combined_signals is not None and not combined_signals.empty:
                # Save combined signals
                save_combined_data(combined_signals, output_dir / 'signals.csv')
                
                # Also save in date-based subdirectories if datetime is available
                if 'datetime' in combined_signals.columns:
                    combined_signals['date'] = pd.to_datetime(combined_signals['datetime']).dt.date
                    
                    for date, group in combined_signals.groupby('date'):
                        date_str = date.strftime('%Y%m%d')
                        date_dir = output_dir / date_str
                        date_dir.mkdir(exist_ok=True)
                        save_combined_data(group.drop('date', axis=1), date_dir / 'signals.csv')
        else:
            logger.warning("No signals files found")
            
        logger.info(f"Data preparation complete. Training data ready in: {output_dir}")
        logger.info(f"To train the model, run: python -m train_model --ticker={args.ticker} --data-dir={output_dir}")
        
    except Exception as e:
        logger.error(f"Error preparing training data: {e}", exc_info=True)

if __name__ == "__main__":
    main()
