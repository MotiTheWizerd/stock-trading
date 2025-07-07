"""Plotting utilities for financial data visualization."""
import os
import mplfinance as mpf
import matplotlib.pyplot as plt
from datetime import datetime

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
    # Make a copy of the dataframe to avoid modifying the original
    plot_df = df.copy()
    
    # Ensure the index is a DatetimeIndex
    if not isinstance(plot_df.index, pd.DatetimeIndex):
        plot_df.index = pd.to_datetime(plot_df.index)
    
    # Create figure and axes
    fig, axes = mpf.plot(
        plot_df,
        type='candle',
        style='charles',
        title=f'{ticker} Price',
        ylabel='Price ($)',
        volume=True,
        ylabel_lower='Volume',
        figratio=(12, 8),
        figscale=1.1,
        returnfig=True
    )
    
    # Save the figure if requested
    if save_to_file and date_folder:
        # Create charts directory if it doesn't exist
        chart_dir = os.path.join('data', ticker, 'charts', date_folder)
        os.makedirs(chart_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{ticker}_{date_folder}_{timestamp}.png'
        filepath = os.path.join(chart_dir, filename)
        
        # Save the figure
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        return filepath
    
    plt.close()
    return None
