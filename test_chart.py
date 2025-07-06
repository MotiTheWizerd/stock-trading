import sys
sys.path.append(".")

from utils import plot_candlestick
import pandas as pd
import os

# Test chart generation directly
ticker = "AAPL"
date_folder = "20250706"
csv_path = os.path.join("data", ticker, date_folder, "data.csv")

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
        chart_dir = os.path.join("data", ticker, date_folder)
        chart_files = [f for f in os.listdir(chart_dir) if f.startswith("chart_")]
        print(f"Chart files in directory: {chart_files}")
    except Exception as e:
        print(f"Error processing CSV: {str(e)}")
        import traceback
        traceback.print_exc()
else:
    print(f"CSV file not found at {csv_path}")
