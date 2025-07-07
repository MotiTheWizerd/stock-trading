import sys
sys.path.append(".")

from utils import plot_candlestick
from paths import day_csv, charts_dir
import pandas as pd
import os
from datetime import datetime

# Test chart generation directly
ticker = "AAPL"
date_folder = "20250706"
csv_path = str(day_csv(ticker, date_folder))

print(f"Testing chart generation for {ticker} using CSV at {csv_path}")

# Check if file exists
if os.path.exists(csv_path):
    print(f"CSV file exists with size {os.path.getsize(csv_path)} bytes")
    
    # Read the CSV file
    try:
        data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        print(f"CSV data shape: {data.shape}")
        print(f"CSV columns: {data.columns.tolist()}")
        
        # Generate chart
        plot_candlestick(data, ticker, save_to_file=True)
        
        # Check if chart was created
        chart_dir_path = charts_dir(ticker, date_folder)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_path = os.path.join(chart_dir_path, f"test_chart_{timestamp}.png")
        chart_files = [f for f in os.listdir(chart_dir_path) if f.startswith("chart_")]
        print(f"Chart files in directory: {chart_files}")
    except Exception as e:
        print(f"Error processing CSV: {str(e)}")
        import traceback
        traceback.print_exc()
else:
    print(f"CSV file not found at {csv_path}")
