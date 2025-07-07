#!/usr/bin/env python3
"""
Fix script to generate signals from existing data files.
"""
import pandas as pd
from pathlib import Path
from signals import detect_rsi_signals

def process_existing_data():
    """Process all existing data files and generate signals."""
    
    # Find all existing data files
    data_dir = Path("data")
    data_files = list(data_dir.glob("**/data-*.csv"))
    
    print(f"🔍 Found {len(data_files)} data files to process")
    
    for data_file in data_files:
        print(f"\n📊 Processing {data_file}")
        
        # Extract ticker and date from path
        # e.g., data/AAPL/202407/data-20240703.csv
        parts = data_file.parts
        ticker = parts[-3]  # AAPL
        date = data_file.stem.split('-', 1)[1]  # 20240703
        
        try:
            # Read the data
            df = pd.read_csv(data_file)
            print(f"   📈 Data shape: {df.shape}")
            print(f"   📅 Date: {date}")
            
            # Check if we have actual data (not just headers)
            if len(df) <= 1:
                print(f"   ⚠️  File contains only headers, skipping")
                continue
                
            # Generate signals
            buy_signals, sell_signals, signals_df = detect_rsi_signals(df, ticker, date)
            
            if signals_df is not None and not signals_df.empty:
                print(f"   ✅ Generated {len(signals_df)} signals")
            else:
                print(f"   📭 No signals generated")
                
        except Exception as e:
            print(f"   ❌ Error processing {data_file}: {str(e)}")
            continue

if __name__ == "__main__":
    process_existing_data()
