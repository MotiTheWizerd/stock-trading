import os
import pandas as pd
from datetime import datetime, timedelta

# List of tickers
TICKERS = ["AAPL", "MSFT", "TSLA"]

def create_directory_structure():
    """Create the directory structure for each ticker"""
    for ticker in TICKERS:
        # Create main ticker directory
        ticker_dir = os.path.join("data", ticker)
        os.makedirs(ticker_dir, exist_ok=True)
        
        # Create charts subdirectory
        charts_dir = os.path.join(ticker_dir, "charts")
        os.makedirs(charts_dir, exist_ok=True)
        
        print(f"✅ Created directory structure for {ticker}")

def create_sample_csv():
    """Create a sample CSV file for each ticker"""
    for ticker in TICKERS:
        # Create a simple DataFrame with some sample data
        now = datetime.now()
        dates = [now - timedelta(minutes=i*5) for i in range(10)]
        
        data = {
            'Open': [100 + i for i in range(10)],
            'High': [105 + i for i in range(10)],
            'Low': [95 + i for i in range(10)],
            'Close': [102 + i for i in range(10)],
            'Volume': [1000000 + i*10000 for i in range(10)]
        }
        
        df = pd.DataFrame(data, index=dates)
        
        # Save to CSV
        file_path = os.path.join("data", ticker, "data.csv")
        df.to_csv(file_path)
        print(f"✅ Created sample CSV for {ticker} at {file_path}")

if __name__ == "__main__":
    create_directory_structure()
    create_sample_csv()
    print("All directories and sample files created successfully!")
