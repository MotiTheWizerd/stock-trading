import os
import numpy as np
import pandas as pd
import yfinance as yf
import traceback
from paths import day_csv, month_dir
from datetime import datetime

def get_date_folder(custom: str | None = None):
    """Return a YYYYMMDD date string.

    If *custom* is provided (already in YYYYMMDD), it is returned verbatim.
    Otherwise we use today's date.
    """
    if custom:
        return custom
    return datetime.now().strftime("%Y%m%d")

from datetime import timedelta

def download_data(ticker, period="1y", interval="1d", *, date_folder: str | None = None, start_date: str | None = None):
    try:
        print(f"Downloading {ticker}...")
        
        # If date_folder is provided, use it as the start_date if start_date is not provided
        if date_folder and not start_date:
            start_date = date_folder
        
        # ---------- Interval auto-fallback ----------
        SUPPORTED_MAX_DAYS = {
            "1m": 7,
            "2m": 7,
            "5m": 30,
            "15m": 60,
            "30m": 60 * 6,   # 6 months
            "60m": 365 * 2,   # 2 years
        }
        def _auto_interval(req: str, age: int) -> str:
            if req not in SUPPORTED_MAX_DAYS:
                return req  # unknown, just try
            if age <= SUPPORTED_MAX_DAYS[req]:
                return req
            # choose the *smallest* interval whose limit covers the age
            for iv, max_days in sorted(SUPPORTED_MAX_DAYS.items(), key=lambda x: x[1]):
                if age <= max_days:
                    print(f"⚠️  Interval {req} not available {age}d ago, falling back to {iv}")
                    return iv
            return "60m"

        if start_date:
            # Convert start_date to datetime
            start_dt = datetime.strptime(start_date, "%Y%m%d")
            age_days = (datetime.utcnow() - start_dt).days
            interval = _auto_interval(interval, age_days)
            # End date is the next day
            end_dt = start_dt + timedelta(days=1)
            print(f"Fetching data for specific date: {start_dt.strftime('%Y-%m-%d')} with interval {interval}")
            data = yf.download(ticker, start=start_dt, end=end_dt, interval=interval)
        else:
            # If no specific date, use period
            print(f"Fetching data for period: {period}")
            data = yf.download(ticker, period=period, interval=interval)
        
        # Create directory structure with date folder
        folder = get_date_folder(date_folder)
        file_path = str(day_csv(ticker, folder))
        
        try:
            # Ensure index has proper name so CSV gets a header
            data.index.name = 'datetime'
            data.to_csv(file_path)
            print(f"✅ Saved {ticker} -> {file_path}")
            
            # Verify file exists
            if os.path.exists(file_path):
                print(f"File verified: {file_path} exists with size {os.path.getsize(file_path)} bytes")
                
                # Automatically enrich the data with technical indicators
                try:
                    from feature_integration import enrich_after_download_hook
                    enrich_success = enrich_after_download_hook(ticker, folder)
                    if enrich_success:
                        print(f"🔧 Feature engineering completed for {ticker}")
                    else:
                        print(f"⚠️ Feature engineering failed for {ticker}")
                except ImportError:
                    print("⚠️ Feature engineering module not available")
                except Exception as e:
                    print(f"⚠️ Feature engineering error: {str(e)}")
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

def append_to_csv(new_data, ticker, *, date_folder: str | None = None):
    """Append new data to the CSV file, ensuring proper data types and format.
    
    Args:
        new_data (pd.DataFrame): New data to append
        ticker (str): Stock ticker symbol
        date_folder (str, optional): Date folder in YYYYMMDD format
    """
    try:
        # Create directory structure with date folder
        folder = get_date_folder(date_folder)
        file_path = str(day_csv(ticker, folder))
        dir_path = str(month_dir(ticker, folder))
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Clean the new data
        try:
            # Make sure we have all required columns with proper types
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Convert all price columns to numeric
            for col in required_columns:
                if col in new_data.columns:
                    new_data[col] = pd.to_numeric(new_data[col], errors='coerce')
            
            # Ensure we have all required columns
            for col in required_columns:
                if col not in new_data.columns:
                    if col == 'Volume':
                        new_data[col] = 0  # Default volume to 0 if missing
                    else:
                        # For price columns, use Close if available, otherwise use the first available price column
                        if 'Close' in new_data.columns:
                            new_data[col] = new_data['Close']
                        else:
                            # Find the first available price column
                            for price_col in ['Open', 'High', 'Low', 'Close']:
                                if price_col in new_data.columns:
                                    new_data[col] = new_data[price_col]
                                    break
            
            # Ensure datetime index
            if not isinstance(new_data.index, pd.DatetimeIndex):
                new_data.index = pd.to_datetime(new_data.index, errors='coerce')
            
            # Remove any timezone information
            if new_data.index.tz is not None:
                new_data.index = new_data.index.tz_localize(None)
            
            # Drop any rows with missing Close price
            new_data = new_data.dropna(subset=['Close'])
            
            if new_data.empty:
                print("⚠️ No valid data to save after cleaning")
                return False
                
        except Exception as e:
            print(f"❌ Error cleaning new data: {str(e)}")
            traceback.print_exc()
            return False
        
        # Process existing data if file exists
        if os.path.exists(file_path):
            try:
                # Read existing data, skipping any metadata rows
                existing = pd.read_csv(
                    file_path, 
                    index_col=0, 
                    parse_dates=True,
                    skiprows=lambda x: x > 0 and not str(x).isdigit()  # Skip non-data rows
                )
                
                # Clean existing data
                if not existing.empty:
                    # Convert index to datetime
                    if not isinstance(existing.index, pd.DatetimeIndex):
                        existing.index = pd.to_datetime(existing.index, errors='coerce')
                    
                    # Ensure all required columns exist
                    for col in required_columns:
                        if col not in existing.columns:
                            existing[col] = np.nan
                    
                    # Convert all columns to numeric
                    for col in required_columns:
                        existing[col] = pd.to_numeric(existing[col], errors='coerce')
                    
                    # Remove any rows with missing Close price
                    existing = existing.dropna(subset=['Close'])
                    
                    # Combine with new data
                    combined = pd.concat([existing, new_data])
                    combined = combined[~combined.index.duplicated(keep='last')]
                    combined = combined.sort_index()
                else:
                    combined = new_data
                    
            except Exception as e:
                print(f"❌ Error processing existing data: {str(e)}")
                traceback.print_exc()
                combined = new_data
        else:
            print(f"No existing file found at {file_path}, creating new file")
            combined = new_data
        
        # Save the data
        try:
            # Ensure we only have the required columns in the correct order
            combined = combined[required_columns]
            
            # Save to CSV without any extra rows
            combined.to_csv(file_path, float_format='%.6f')
            print(f"✅ Saved {ticker} -> {file_path}")
            
            # Verify the file was created and has content
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                print(f"File verified: {file_path} exists with size {size} bytes")
                
                # Run feature engineering if the file has content
                if size > 0:
                    try:
                        from feature_integration import enrich_after_download_hook
                        print(f"🔧 Running feature engineering for {ticker}")
                        enrich_success = enrich_after_download_hook(ticker, folder)
                        if enrich_success:
                            print(f"✅ Feature engineering completed for {ticker}")
                        else:
                            print(f"⚠️ Feature engineering failed for {ticker}")
                    except ImportError as e:
                        print(f"⚠️ Feature engineering module not available: {str(e)}")
                    except Exception as e:
                        print(f"⚠️ Feature engineering error: {str(e)}")
                        traceback.print_exc()
                else:
                    print("⚠️ File is empty, skipping feature engineering")
                return True
            else:
                print(f"❌ File was not created: {file_path}")
                return False
                
        except Exception as e:
            print(f"❌ Error saving data to {file_path}: {str(e)}")
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ Unexpected error in append_to_csv for {ticker}: {str(e)}")
        traceback.print_exc()
        return False
        traceback.print_exc()
