"""
Fix missing columns in CSV data files.
This script reads CSV files, checks for missing columns, and fills them with data.
"""

import os
import sys
import pandas as pd
import argparse
from datetime import datetime
from paths import month_dir, day_csv

def fix_columns(ticker, date_str):
    """Fix missing columns in a CSV file."""
    # Get the CSV file path
    file_path = str(day_csv(ticker, date_str))
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    print(f"📂 Reading file: {file_path}")
    
    try:
        # Read the CSV file
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        print(f"✅ Successfully read CSV with shape: {df.shape}")
        
        # Check columns
        print(f"📋 Available columns: {df.columns.tolist()}")
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Check for tuple columns
        tuple_columns = [col for col in df.columns if isinstance(col, tuple) or str(col).startswith("('")]
        
        # Map tuple columns to standard columns if needed
        if tuple_columns:
            print(f"🔄 Found tuple columns: {tuple_columns}")
            for col in required_columns:
                tuple_col = next((tc for tc in tuple_columns if str(tc).lower().find(col.lower()) >= 0), None)
                if tuple_col:
                    print(f"🔗 Mapping tuple column {tuple_col} to {col}")
                    df[col] = df[tuple_col]
        
        # Check if any required columns are still missing
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"⚠️ Missing columns: {missing_columns}")
            
            # If we have Close but not Open/High/Low, use Close for all
            if 'Close' in df.columns:
                for col in ['Open', 'High', 'Low']:
                    if col in missing_columns:
                        print(f"📝 Filling {col} with Close values")
                        df[col] = df['Close']
            
            # If we have Open but not High/Low, use Open
            elif 'Open' in df.columns:
                for col in ['High', 'Low']:
                    if col in missing_columns:
                        print(f"📝 Filling {col} with Open values")
                        df[col] = df['Open']
            
            # If we're missing Volume, add a default value
            if 'Volume' in missing_columns:
                print(f"📝 Adding default Volume values (1000)")
                df['Volume'] = 1000
        
        # Save the updated CSV
        df.to_csv(file_path)
        print(f"💾 Saved updated CSV with columns: {df.columns.tolist()}")
        return True
    
    except Exception as e:
        print(f"❌ Error fixing columns: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Fix missing columns in CSV data files")
    parser.add_argument("--ticker", type=str, required=True, help="Ticker symbol")
    parser.add_argument("--date", type=str, required=True, help="Date in YYYYMMDD format")
    
    args = parser.parse_args()
    
    # Validate date format
    try:
        datetime.strptime(args.date, "%Y%m%d")
    except ValueError:
        print(f"❌ Invalid date format: {args.date}. Use YYYYMMDD format.")
        sys.exit(1)
    
    success = fix_columns(args.ticker, args.date)
    
    if success:
        print(f"✅ Successfully fixed columns for {args.ticker} on {args.date}")
    else:
        print(f"❌ Failed to fix columns for {args.ticker} on {args.date}")

if __name__ == "__main__":
    main()
