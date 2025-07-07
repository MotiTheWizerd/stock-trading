"""
signals.py - Detect and save RSI-based trading signals
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime
from ta.momentum import RSIIndicator

from paths import signals_csv, month_dir

def detect_rsi_signals(df, ticker, date_folder=None):
    """
    Detect RSI-based signals from a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        ticker (str): Ticker symbol
        date_folder (str, optional): Date folder in YYYYMMDD format
        
    Returns:
        tuple: (buy_signals, sell_signals, signals_df)
    """
    try:
        # Debug prints
        print(f"\nDetecting RSI signals for {ticker}")
        print(f"DataFrame shape: {df.shape}")
        print(f"DataFrame columns: {df.columns.tolist()}")
        
        # Make sure we have RSI in the DataFrame
        if 'RSI' not in df.columns:
            print("RSI column not found, calculating...")
            rsi = RSIIndicator(close=df['Close'], window=14).rsi()
            df['RSI'] = rsi
        
        # Debug: print RSI stats
        print(f"RSI stats: min={df['RSI'].min():.2f}, max={df['RSI'].max():.2f}, mean={df['RSI'].mean():.2f}")
        print(f"  RSI < 30 count: {(df['RSI'] < 30).sum()}")
        print(f"  RSI > 70 count: {(df['RSI'] > 70).sum()}")
        
        # Create empty DataFrames for signals
        buy_signals = pd.DataFrame()
        sell_signals = pd.DataFrame()
        
        # Detect buy signals (RSI < 30)
        buy_mask = df['RSI'] < 30
        if buy_mask.any():
            buy_signals = df[buy_mask].copy()
            buy_signals['Signal'] = 'BUY'
            print(f"Found {len(buy_signals)} BUY signals")
        
        # Detect sell signals (RSI > 70)
        sell_mask = df['RSI'] > 70
        if sell_mask.any():
            sell_signals = df[sell_mask].copy()
            sell_signals['Signal'] = 'SELL'
            print(f"Found {len(sell_signals)} SELL signals")
        
        # Combine signals
        signal_frames = []
        if not buy_signals.empty:
            signal_frames.append(buy_signals)
        if not sell_signals.empty:
            signal_frames.append(sell_signals)
        
        if signal_frames:
            signals_df = pd.concat(signal_frames)
        else:
            signals_df = pd.DataFrame()
        
        # If we have signals, save them
        if not signals_df.empty:
            # Prepare signals DataFrame for saving
            signals_df = signals_df.sort_index()
            signals_df['Ticker'] = ticker
            signals_df['RSI_Value'] = signals_df['RSI']
            
            # Keep only necessary columns
            signals_df = signals_df[['Ticker', 'Signal', 'RSI_Value']]
            
            # Save signals to CSV
            signals_path = signals_csv(ticker, date_folder)
            os.makedirs(os.path.dirname(signals_path), exist_ok=True)
            
            print(f"Saving signals to: {signals_path}")
            
            # Check if file exists and append or create new
            if os.path.exists(signals_path):
                try:
                    existing_signals = pd.read_csv(signals_path, index_col=0, parse_dates=True)
                    combined_signals = pd.concat([existing_signals, signals_df])
                    combined_signals = combined_signals[~combined_signals.index.duplicated(keep='last')]
                    combined_signals.to_csv(signals_path)
                    print(f"Updated signals file with {len(signals_df)} new signals")
                except Exception as e:
                    print(f"Error reading existing signals file, creating new: {str(e)}")
                    signals_df.to_csv(signals_path)
                    print(f"Created signals file with {len(signals_df)} signals")
            else:
                signals_df.to_csv(signals_path)
                print(f"Created new signals file with {len(signals_df)} signals")
        else:
            print("No signals detected")
            
        return buy_signals, sell_signals, signals_df
        
    except Exception as e:
        print(f"Error detecting RSI signals: {str(e)}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), pd.DataFrame(), None
