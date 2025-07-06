import os
import pandas as pd
import yfinance as yf
import traceback
from datetime import datetime

def get_date_folder():
    """Get today's date folder in YYYYMMDD format"""
    return datetime.now().strftime("%Y%m%d")

def download_data(ticker, period="1y", interval="1d"):
    try:
        print(f"Downloading {ticker}...")
        data = yf.download(ticker, period=period, interval=interval)
        
        # Create directory structure with date folder
        date_folder = get_date_folder()
        dir_path = os.path.join("data", ticker, date_folder)
        try:
            os.makedirs(dir_path, exist_ok=True)
            print(f"Directory created/verified: {dir_path}")
        except Exception as e:
            print(f"❌ Error creating directory {dir_path}: {str(e)}")
            traceback.print_exc()
            return data
        
        # Save data to CSV
        file_path = os.path.join(dir_path, "data.csv")
        try:
            data.to_csv(file_path)
            print(f"✅ Saved {ticker} -> {file_path}")
            
            # Verify file exists
            if os.path.exists(file_path):
                print(f"File verified: {file_path} exists with size {os.path.getsize(file_path)} bytes")
            else:
                print(f"❌ File was not created: {file_path}")
        except Exception as e:
            print(f"❌ Error saving CSV file {file_path}: {str(e)}")
            traceback.print_exc()
        
        return data
    except Exception as e:
        print(f"❌ Error in download_data for {ticker}: {str(e)}")
        traceback.print_exc()
        return pd.DataFrame()

def append_to_csv(new_data, ticker):
    try:
        # Create directory structure with date folder
        date_folder = get_date_folder()
        dir_path = os.path.join("data", ticker, date_folder)
        try:
            os.makedirs(dir_path, exist_ok=True)
            print(f"Directory created/verified for append: {dir_path}")
        except Exception as e:
            print(f"❌ Error creating directory for append {dir_path}: {str(e)}")
            traceback.print_exc()
            return

        file_path = os.path.join(dir_path, "data.csv")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(dir_path, f"data_backup_{timestamp}.csv")

        # Process existing data if file exists
        if os.path.exists(file_path):
            try:
                # Create backup of existing file
                try:
                    with open(file_path, 'r') as src, open(backup_path, 'w') as dst:
                        dst.write(src.read())
                    print(f"Created backup: {backup_path}")
                except Exception as e:
                    print(f"⚠️ Warning: Could not create backup: {str(e)}")
                
                # Read existing data
                existing = pd.read_csv(file_path, index_col=0, parse_dates=True)
                print(f"Read existing data from {file_path}, shape: {existing.shape}")

                # ודא שהאינדקס הוא DatetimeIndex לפני .tz_localize
                if not isinstance(existing.index, pd.DatetimeIndex):
                    existing.index = pd.to_datetime(existing.index, errors="coerce")

                if existing.index.tz is not None:
                    existing.index = existing.index.tz_localize(None)

                if new_data.index.tz is not None:
                    new_data.index = new_data.index.tz_localize(None)

                combined = pd.concat([existing, new_data])
                combined = combined[~combined.index.duplicated(keep='last')]
                combined.sort_index(inplace=True)
                print(f"Combined data shape: {combined.shape}")
            except Exception as e:
                print(f"❌ Error processing existing data: {str(e)}")
                traceback.print_exc()
                # Fall back to just using new data
                combined = new_data.copy()
        else:
            print(f"No existing file found at {file_path}, creating new file")
            combined = new_data.copy()

        # Save combined data
        try:
            combined.to_csv(file_path)
            print(f"✅ Updated {ticker} -> {file_path}")
            
            # Verify file exists and has content
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                print(f"File verified: {file_path} exists with size {size} bytes")
                if size == 0:
                    print("⚠️ Warning: File exists but is empty!")
            else:
                print(f"❌ File was not created: {file_path}")
        except Exception as e:
            print(f"❌ Error saving combined CSV: {str(e)}")
            traceback.print_exc()
    except Exception as e:
        print(f"❌ Unexpected error in append_to_csv for {ticker}: {str(e)}")
        traceback.print_exc()

