#!/usr/bin/env python3

import pandas as pd
import numpy as np
from signals import detect_rsi_signals
from utils import get_date_folder
import os

def test_signals():
    print("Testing signals generation...")
    
    # Read the CSV data
    date_folder = get_date_folder()
    csv_path = os.path.join("data", "AAPL", date_folder, "data.csv")
    
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return
        
    print(f"Reading data from: {csv_path}")
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Handle tuple columns
    tuple_columns = [col for col in df.columns if isinstance(col, tuple) or str(col).startswith("('")]
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    if tuple_columns:
        print(f"Found tuple columns: {tuple_columns}")
        for col in required_columns:
            tuple_col = next((tc for tc in tuple_columns if str(tc).lower().find(col.lower()) >= 0), None)
            if tuple_col:
                print(f"Mapping {tuple_col} to {col}")
                df[col] = df[tuple_col]
    
    # Convert to numeric
    for col in required_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop NaN rows
    df.dropna(subset=required_columns, inplace=True)
    
    print(f"Clean data shape: {df.shape}")
    print(f"Data sample:")
    print(df[required_columns].head())
    
    # Add RSI manually for testing
    from ta.momentum import RSIIndicator
    if len(df) >= 15:
        df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
        print(f"RSI added. Sample RSI values: {df['RSI'].dropna().head()}")
        
        # Test signal detection
        try:
            buy_mask, sell_mask, signal_df = detect_rsi_signals(df, "AAPL")
            if signal_df is not None:
                print(f"✅ Signals generated: {len(signal_df)} signals")
                print(signal_df.head())
            else:
                print("❌ No signals generated")
        except Exception as e:
            print(f"❌ Error in signal detection: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        print(f"❌ Not enough data for RSI calculation: {len(df)} rows")

if __name__ == "__main__":
    test_signals()
