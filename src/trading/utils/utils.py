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
from paths import day_csv, charts_dir

def get_date_folder(custom: str | None = None):
    """Return a YYYYMMDD string – use *custom* if provided else today."""
    if custom:
        return custom
    return datetime.now().strftime("%Y%m%d")

def plot_candlestick(df, ticker, save_to_file=False, date_folder=None):
    """
    Plot candlestick chart with indicators.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        ticker (str): Ticker symbol
        save_to_file (bool): Whether to save the chart to a file
        date_folder (str, optional): Date folder in YYYYMMDD format
        
    Returns:
        str: Path to the saved chart file if save_to_file is True, otherwise None
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Create a fresh DataFrame from the CSV file directly to avoid data structure issues
    folder = get_date_folder(date_folder)
    csv_path = str(day_csv(ticker, folder))
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f" Data file not found: {csv_path}")
        return
            
    print(f" Reading data from CSV file: {csv_path}")
    
    try:
        # Read the CSV file directly with enhanced error handling
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        print(f" Successfully read CSV with shape: {df.shape}")
        
        # Ensure the index is a DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")
            
        # Handle tuple column names and regular column names
        print(f" Available columns: {df.columns.tolist()}")
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        tuple_columns = [col for col in df.columns if isinstance(col, tuple) or str(col).startswith("('")]
        
        # Check if we have tuple columns that match our required columns
        if tuple_columns:
            print(f" Found tuple columns: {tuple_columns}")
            # Map tuple columns to standard column names
            for col in required_columns:
                tuple_col = next((tc for tc in tuple_columns if str(tc).lower().find(col.lower()) >= 0), None)
                if tuple_col:
                    print(f" Mapping tuple column {tuple_col} to {col}")
                    # Copy data from tuple column to standard column
                    df[col] = df[tuple_col]
        
        # Now check if we have all required columns
        if not all(col in df.columns for col in required_columns):
            print(f"[ ] Still missing required columns in {ticker}")
            print(f"Available columns: {df.columns.tolist()}")
            return
            
        # Convert all required columns to numeric with better error handling
        for col in required_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Drop rows with NaN values
        df.dropna(subset=required_columns, inplace=True)
        
        if df.empty:
            print(f"[ ] No valid data to plot for {ticker} after cleaning")
            return
            
        print(f" Clean data shape: {df.shape}")
        print(f" Data types: {df.dtypes.to_dict()}")
        
        # Now proceed with the actual chart generation
        print(f" Processing indicators for {ticker}...")

        # === Enhanced Technical Indicators ===
        apds = []  # Additional plots for indicators
        
        # Make sure we have enough data points for indicators
        if len(df) < 15:  # Need at least 14 points for RSI
            print(f"[ ] Not enough data points for indicators: {len(df)}")
            # Just save a simple candlestick chart without indicators
            if save_to_file:
                now = datetime.now().strftime("%Y-%m-%d_%H-%M")
                chart_dir_path = charts_dir(ticker, folder)
                file_path = os.path.join(chart_dir_path, f"enhanced_chart_{now}.png")
                
                try:
                    # Enhanced simple chart with better styling
                    custom_style = mpf.make_mpf_style(
                        base_mpf_style='yahoo',
                        rc={
                            'font.size': 11,
                            'axes.labelsize': 12,
                            'axes.titlesize': 16,
                            'xtick.labelsize': 10,
                            'ytick.labelsize': 10,
                            'legend.fontsize': 10,
                            'figure.facecolor': 'white',
                            'axes.facecolor': 'white'
                        },
                        gridstyle='-',
                        gridcolor='#E0E0E0',

                        y_on_right=True,
                        marketcolors=mpf.make_marketcolors(
                            up='#26A69A',
                            down='#EF5350',
                            edge='inherit',
                            wick={'up': '#26A69A', 'down': '#EF5350'},
                            volume='in'
                        )
                    )
                    
                    mpf.plot(
                        df,
                        type='candle',
                        volume=True,
                        style=custom_style,
                        title=f" {ticker} - Simple Chart (Limited Data)",
                        figratio=(16, 10),
                        tight_layout=True,
                        savefig=file_path,
                        warn_too_much_data=1000
                    )
                    print(f" Saved simple chart for {ticker} -> {file_path}")
                    return file_path
                except Exception as e:
                    print(f" Error saving simple chart: {str(e)}")
            return
            
        # Calculate RSI
        try:
            rsi = RSIIndicator(close=df['Close'], window=14).rsi()
            df['RSI'] = rsi  # Use 'RSI' as the column name for signals.py compatibility
            df['RSI_14'] = rsi  # Keep RSI_14 for backward compatibility
            
            # Create RSI plot for panel 1
            apds.append(mpf.make_addplot(df['RSI'], panel=1, color='purple', ylabel='RSI'))
            
            # Add overbought/oversold lines
            apds.append(mpf.make_addplot(pd.Series(70, index=df.index), panel=1, color='red', linestyle='--'))
            apds.append(mpf.make_addplot(pd.Series(30, index=df.index), panel=1, color='green', linestyle='--'))
            
            print(" RSI indicator added")
        except Exception as e:
            print(f" Error calculating RSI: {str(e)}")
        
        # Calculate MACD
        try:
            macd = MACD(close=df['Close'])
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
            df['MACD_Hist'] = macd.macd_diff()
            
            # Create MACD plots for panel 2
            apds.append(mpf.make_addplot(df['MACD'], panel=2, color='blue', ylabel='MACD'))
            apds.append(mpf.make_addplot(df['MACD_Signal'], panel=2, color='orange'))
            apds.append(mpf.make_addplot(df['MACD_Hist'], panel=2, type='bar', color='dimgray'))
            
            print(" MACD indicator added")
        except Exception as e:
            print(f" Error calculating MACD: {str(e)}")
        
        # Detect signals based on RSI
        try:
            buy_signals, sell_signals, signal_df = detect_rsi_signals(df, ticker, date_folder=folder)
            
            if buy_signals.any():
                buy_markers = df.loc[buy_signals, 'Close']
                apds.append(mpf.make_addplot(buy_markers, type='scatter', markersize=100, marker='^', color='green'))
                print(f" Added {buy_signals.sum()} buy signals")
                
            if sell_signals.any():
                sell_markers = df.loc[sell_signals, 'Close']
                apds.append(mpf.make_addplot(sell_markers, type='scatter', markersize=100, marker='v', color='red'))
                print(f" Added {sell_signals.sum()} sell signals")
                
        except Exception as e:
            print(f" Error detecting signals: {str(e)}")
        
        # Save or display chart
        if save_to_file:
            # Use the date_folder in filename, with timestamp for uniqueness
            timestamp = datetime.now().strftime("%H-%M-%S")
            chart_dir_path = charts_dir(ticker, folder)
            file_path = os.path.join(chart_dir_path, f"enhanced_chart_{folder}_{timestamp}.png")
            
            try:
                # Create enhanced style
                custom_style = mpf.make_mpf_style(
                    base_mpf_style='yahoo',
                    rc={
                        'font.size': 11,
                        'axes.labelsize': 12,
                        'axes.titlesize': 16,
                        'xtick.labelsize': 10,
                        'ytick.labelsize': 10,
                        'legend.fontsize': 10,
                        'figure.facecolor': 'white',
                        'axes.facecolor': 'white'
                    },
                    gridstyle='-',
                    gridcolor='#E0E0E0',
                    y_on_right=True,
                    marketcolors=mpf.make_marketcolors(
                        up='#26A69A',
                        down='#EF5350',
                        edge='inherit',
                        wick={'up': '#26A69A', 'down': '#EF5350'},
                        volume='in'
                    )
                )
                
                # Set up plot parameters
                plot_params = {
                    'type': 'candle',
                    'volume': True,
                    'style': custom_style,
                    'figratio': (16, 10),
                    'tight_layout': True,
                    'savefig': file_path,
                    'warn_too_much_data': 1000
                }
                
                # Add moving averages if we have enough data
                if len(df) >= 50:
                    plot_params['mav'] = (10, 20, 50)
                elif len(df) >= 20:
                    plot_params['mav'] = (10, 20)
                elif len(df) >= 10:
                    plot_params['mav'] = (10,)
                
                # Add indicators and set title
                if apds:
                    plot_params['addplot'] = apds
                    
                    # Set panel ratios based on indicators
                    if any('panel=2' in str(ap) for ap in apds) and any('panel=1' in str(ap) for ap in apds):
                        plot_params['panel_ratios'] = (8, 2.5, 2.5)
                        plot_params['title'] = f" {ticker} - Technical Analysis (RSI + MACD)"
                    elif any('panel=1' in str(ap) for ap in apds):
                        plot_params['panel_ratios'] = (8, 2.5)
                        plot_params['title'] = f" {ticker} - Technical Analysis (RSI)"
                    else:
                        plot_params['title'] = f" {ticker} - Candlestick Chart"
                else:
                    plot_params['title'] = f" {ticker} - Candlestick Chart"
                
                # Generate the plot
                mpf.plot(df, **plot_params)
                print(f" Successfully saved chart for {ticker} -> {file_path}")
                return file_path
            except Exception as e:
                print(f" Error saving chart: {str(e)}")
                traceback.print_exc()
        else:
            # Display mode
            try:
                print(f" Displaying chart for {ticker}...")
                
                # Create display style
                display_style = mpf.make_mpf_style(
                    base_mpf_style='yahoo',
                    rc={
                        'font.size': 11,
                        'axes.labelsize': 12,
                        'axes.titlesize': 16,
                        'figure.facecolor': 'white'
                    },
                    gridstyle='-',
                    gridcolor='#E0E0E0',
                    y_on_right=True,
                    marketcolors=mpf.make_marketcolors(
                        up='#26A69A',
                        down='#EF5350',
                        edge='inherit',
                        wick={'up': '#26A69A', 'down': '#EF5350'}
                    )
                )
                
                # Set up display parameters
                display_params = {
                    'type': 'candle',
                    'volume': True,
                    'style': display_style,
                    'figratio': (18, 12),
                    'tight_layout': True,
                    'warn_too_much_data': 1000
                }
                
                # Add moving averages
                if len(df) >= 50:
                    display_params['mav'] = (10, 20, 50)
                elif len(df) >= 20:
                    display_params['mav'] = (10, 20)
                
                # Add indicators and set title
                if apds:
                    display_params['addplot'] = apds
                    if any('panel=2' in str(ap) for ap in apds) and any('panel=1' in str(ap) for ap in apds):
                        display_params['panel_ratios'] = (8, 2.5, 2.5)
                        display_params['title'] = f" {ticker} - Live Analysis (RSI + MACD)"
                    elif any('panel=1' in str(ap) for ap in apds):
                        display_params['panel_ratios'] = (8, 2.5)
                        display_params['title'] = f" {ticker} - Live Analysis (RSI)"
                    else:
                        display_params['title'] = f" {ticker} - Live Chart"
                else:
                    display_params['title'] = f" {ticker} - Live Chart"
                
                mpf.plot(df, **display_params)
                print(f" Successfully displayed chart for {ticker}")
            except Exception as e:
                print(f" Error displaying chart: {str(e)}")
                traceback.print_exc()
    except Exception as e:
        print(f"Error plotting candlestick chart: {str(e)}")
        traceback.print_exc()
        return None
