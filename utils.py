import sys
sys.path.append(".")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import datetime
import traceback
from ta.momentum import RSIIndicator
from ta.trend import MACD
from signals import detect_rsi_signals

def get_date_folder():
    """Get today's date folder in YYYYMMDD format"""
    return datetime.now().strftime("%Y%m%d")

def plot_candlestick(data, ticker, save_to_file=True):
    try:
        print(f"Generating chart for {ticker}...")
        
        # Create a fresh DataFrame from the CSV file directly to avoid data structure issues
        date_folder = get_date_folder()
        csv_path = os.path.join("data", ticker, date_folder, "data.csv")
        
        if not os.path.exists(csv_path):
            print(f"[❌] CSV file not found at {csv_path}")
            return
            
        print(f"Reading data from CSV file: {csv_path}")
        
        try:
            # Read the CSV file directly
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            print(f"Successfully read CSV with shape: {df.shape}")
            
            # Ensure the index is a DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, errors="coerce")
                
            # Handle tuple column names and regular column names
            print(f"Available columns: {df.columns.tolist()}")
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            tuple_columns = [col for col in df.columns if isinstance(col, tuple) or str(col).startswith("('")]
            
            # Check if we have tuple columns that match our required columns
            if tuple_columns:
                print(f"Found tuple columns: {tuple_columns}")
                # Map tuple columns to standard column names
                for col in required_columns:
                    tuple_col = next((tc for tc in tuple_columns if str(tc).lower().find(col.lower()) >= 0), None)
                    if tuple_col:
                        print(f"Mapping tuple column {tuple_col} to {col}")
                        # Copy data from tuple column to standard column
                        df[col] = df[tuple_col]
            
            # Now check if we have all required columns
            if not all(col in df.columns for col in required_columns):
                print(f"[⚠️] Still missing required columns in {ticker}")
                print(f"Available columns: {df.columns.tolist()}")
                return
                
            # Convert all required columns to numeric
            for col in required_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            # Drop rows with NaN values
            df.dropna(subset=required_columns, inplace=True)
            
            if df.empty:
                print(f"[⚠️] No valid data to plot for {ticker} after cleaning")
                return
                
            print(f"Clean data shape: {df.shape}")
            print(f"Data types: {df.dtypes.to_dict()}")
        except Exception as e:
            print(f"[❌] Error reading or processing CSV: {str(e)}")
            traceback.print_exc()
            return
        
        # Print final column types after processing
        print("\nFinal column types after processing:")
        for col in df.columns:
            print(f"  - {col}: {type(df[col])}, dtype: {df[col].dtype}")
        print(f"DataFrame shape: {df.shape}")
        
        # Now proceed with the actual chart generation

        # === אינדיקטורים ===
        # Make sure we have enough data points for indicators
        if len(df) < 15:  # Need at least 14 points for RSI
            print(f"[⚠️] Not enough data points for indicators: {len(df)}")
            # Just save a simple candlestick chart without indicators
            if save_to_file:
                now = datetime.now().strftime("%Y-%m-%d_%H-%M")
                date_folder = get_date_folder()
                chart_dir = os.path.join("data", ticker, date_folder)
                os.makedirs(chart_dir, exist_ok=True)
                file_path = os.path.join(chart_dir, f"chart_{now}.png")
                
                try:
                    mpf.plot(
                        df,
                        type='candle',
                        volume=True,
                        style='yahoo',
                        title=f"{ticker} - Candlestick",
                        figratio=(16, 9),
                        tight_layout=True,
                        savefig=file_path
                    )
                    print(f"📸 Saved simple chart for {ticker} -> {file_path}")
                    return
                except Exception as e:
                    print(f"❌ Error saving simple chart: {str(e)}")
                    traceback.print_exc()
                    return
            return
            
        close_series = df['Close'].squeeze()
        
        # Calculate RSI
        try:
            df['RSI'] = RSIIndicator(close=close_series, window=14).rsi()
        except Exception as e:
            print(f"❌ Error calculating RSI: {str(e)}")
            df['RSI'] = pd.Series(np.nan, index=df.index)
            
        # Calculate MACD
        try:
            macd = MACD(close=close_series)
            df['MACD'] = macd.macd()
            df['MACD_signal'] = macd.macd_signal()
        except Exception as e:
            print(f"❌ Error calculating MACD: {str(e)}")
            df['MACD'] = pd.Series(np.nan, index=df.index)
            df['MACD_signal'] = pd.Series(np.nan, index=df.index)

        # Drop NaN values in indicator columns
        df.dropna(subset=['RSI', 'MACD', 'MACD_signal'], inplace=True)
        if df.empty:
            print(f"[⚠️] No valid data after calculating indicators")
            return

        # === סיגנלים (RSI) ===
        try:
            buy_mask, sell_mask, _ = detect_rsi_signals(df, ticker)
        except Exception as e:
            print(f"❌ Error detecting signals: {str(e)}")
            buy_mask = None
            sell_mask = None

        # === Addplots לגרף ===
        apds = []
        
        # Check if we have valid indicator data
        if 'RSI' in df.columns and not df['RSI'].isna().all():
            apds.append(mpf.make_addplot(df['RSI'], panel=1, color='purple', ylabel='RSI'))
        
        if 'MACD' in df.columns and not df['MACD'].isna().all():
            apds.append(mpf.make_addplot(df['MACD'], panel=2, color='blue', ylabel='MACD'))
            
        if 'MACD_signal' in df.columns and not df['MACD_signal'].isna().all():
            apds.append(mpf.make_addplot(df['MACD_signal'], panel=2, color='orange'))

        # חצים על הגרף - Create signal markers aligned with DataFrame index
        if buy_mask is not None and buy_mask.any():
            try:
                # Create a series with NaN for all points except buy signals
                buy_signals = pd.Series(index=df.index, dtype=float)
                buy_signals[buy_mask] = df['Close'][buy_mask]
                
                apds.append(
                    mpf.make_addplot(
                        buy_signals,
                        type='scatter',
                        marker='^',
                        markersize=100,
                        color='green',
                        panel=0
                    )
                )
            except Exception as e:
                print(f"❌ Error adding buy signals: {str(e)}")
                
        if sell_mask is not None and sell_mask.any():
            try:
                # Create a series with NaN for all points except sell signals
                sell_signals = pd.Series(index=df.index, dtype=float)
                sell_signals[sell_mask] = df['Close'][sell_mask]
                
                apds.append(
                    mpf.make_addplot(
                        sell_signals,
                        type='scatter',
                        marker='v',
                        markersize=100,
                        color='red',
                        panel=0
                    )
                )
            except Exception as e:
                print(f"❌ Error adding sell signals: {str(e)}")

        # DEBUG: הדפסת dtypes כדי לוודא שהכל מספרי לפני plot
        # ניתן לכבות במידת הצורך
        if os.environ.get('DEBUG_PLOTTING', '0') == '1':
            print('[DEBUG] dtypes before plotting:', df[required_columns].dtypes.to_dict())

        # === שמירת הגרף כתמונה ===
        if save_to_file:
            now = datetime.now().strftime("%Y-%m-%d_%H-%M")
            date_folder = get_date_folder()
            chart_dir = os.path.join("data", ticker, date_folder)
            os.makedirs(chart_dir, exist_ok=True)
            file_path = os.path.join(chart_dir, f"chart_{now}.png")

            try:
                # Determine plot parameters based on available data
                plot_params = {
                    'type': 'candle',
                    'volume': True,
                    'style': 'yahoo',
                    'figratio': (16, 9),
                    'tight_layout': True,
                    'savefig': file_path
                }
                
                # Add moving averages if we have enough data
                if len(df) >= 50:
                    plot_params['mav'] = (20, 50)
                elif len(df) >= 20:
                    plot_params['mav'] = (20,)
                
                # Add indicators if we have them
                if apds:
                    plot_params['addplot'] = apds
                    
                    # Set panel ratios based on number of indicators
                    if any('panel=2' in str(ap) for ap in apds) and any('panel=1' in str(ap) for ap in apds):
                        plot_params['panel_ratios'] = (6, 2, 2)
                        plot_params['title'] = f"{ticker} - Candlestick + RSI + MACD"
                    elif any('panel=1' in str(ap) for ap in apds):
                        plot_params['panel_ratios'] = (6, 2)
                        plot_params['title'] = f"{ticker} - Candlestick + RSI"
                    else:
                        plot_params['title'] = f"{ticker} - Candlestick"
                else:
                    plot_params['title'] = f"{ticker} - Candlestick"
                
                # Generate the plot
                mpf.plot(df, **plot_params)
                print(f"📸 Saved chart for {ticker} -> {file_path}")
            except Exception as e:
                print(f"❌ Error saving chart: {str(e)}")
                traceback.print_exc()
                
                # Try a simpler chart as fallback
                try:
                    print("Attempting to save a simpler chart as fallback...")
                    mpf.plot(
                        df,
                        type='candle',
                        volume=True,
                        style='yahoo',
                        title=f"{ticker} - Basic Chart",
                        figratio=(16, 9),
                        tight_layout=True,
                        savefig=file_path
                    )
                    print(f"📸 Saved simple fallback chart for {ticker} -> {file_path}")
                except Exception as e2:
                    print(f"❌ Error saving fallback chart: {str(e2)}")
        else:
            # Display mode (not saving)
            try:
                mpf.plot(
                    df,
                    type='candle',
                    volume=True,
                    style='yahoo',
                    mav=(20, 50),
                    addplot=apds,
                    panel_ratios=(6, 2, 2),
                    title=f"{ticker} - Candlestick + RSI + MACD",
                    figratio=(16, 9),
                    tight_layout=True
                )
            except Exception as e:
                print(f"❌ Error displaying chart: {str(e)}")
                traceback.print_exc()

    except Exception as e:
        print(f"❌ Error in plot_candlestick for {ticker}: {str(e)}")
        traceback.print_exc()
