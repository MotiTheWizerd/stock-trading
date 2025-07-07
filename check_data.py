"""
Check data file contents.

This script checks the contents of the data file to see if it contains datetime and price information.
"""

import pandas as pd
from pathlib import Path
import sys

def check_file(file_path):
    """Check the contents of a file."""
    print(f"Checking file: {file_path}")
    
    if not Path(file_path).exists():
        print(f"File not found: {file_path}")
        return
    
    try:
        # Read the data file
        df = pd.read_csv(file_path)
        
        # Print basic info
        print(f"\nFile shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Check for datetime column
        if 'datetime' in df.columns:
            print("\nDatetime column exists")
            print(f"First 5 datetime values: {df['datetime'].head().tolist()}")
        else:
            print("\nNo datetime column found")
        
        # Check for price columns
        price_cols = ['Close', 'close', 'Price', 'price']
        found_price = False
        for col in price_cols:
            if col in df.columns:
                found_price = True
                print(f"\n{col} column exists")
                print(f"First 5 {col} values: {df[col].head().tolist()}")
        
        if not found_price:
            print("\nNo price columns found")
        
        # Print first few rows
        print("\nFirst 5 rows:")
        print(df.head().to_string())
        
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "data/AAPL/latest/data.csv"
    
    check_file(file_path)
