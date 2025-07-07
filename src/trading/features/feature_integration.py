"""
Feature Engineering Integration Module

This module integrates the FeatureEngineer with the existing download pipeline
to automatically enrich data files after they are downloaded.
"""

import os
import glob
from pathlib import Path
from typing import List, Dict, Optional
from feature_engineer import FeatureEngineer, FeatureEngineerConfig
from paths import BASE_DIR
import traceback

class FeatureIntegration:
    """
    Integration layer for automatic feature engineering in the download pipeline.
    """
    
    def __init__(self, config: Optional[FeatureEngineerConfig] = None):
        """Initialize the integration module."""
        self.engineer = FeatureEngineer(config)
        self.logger_prefix = "🔗 [FeatureIntegration]"
    
    def log(self, message: str, level: str = "INFO"):
        """Simple logging method."""
        prefix = self.logger_prefix
        if level == "ERROR":
            prefix = "❌ [FeatureIntegration]"
        elif level == "WARNING":
            prefix = "⚠️ [FeatureIntegration]"
        elif level == "SUCCESS":
            prefix = "✅ [FeatureIntegration]"
        
        print(f"{prefix} {message}")
    
    def enrich_after_download(self, ticker: str, yyyymmdd: str) -> bool:
        """
        Enrich a specific data file after it's downloaded.
        
        Args:
            ticker: Stock ticker symbol
            yyyymmdd: Date in YYYYMMDD format
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Construct file path using the same logic as paths.py
            from paths import day_csv
            file_path = str(day_csv(ticker, yyyymmdd))
            
            # Check if file exists
            if not os.path.exists(file_path):
                self.log(f"Data file not found: {file_path}", "WARNING")
                return False
            
            # Enrich the file
            self.log(f"Enriching data for {ticker} on {yyyymmdd}")
            success = self.engineer.enrich_file(file_path)
            
            if success:
                self.log(f"Successfully enriched {ticker} data for {yyyymmdd}", "SUCCESS")
            else:
                self.log(f"Failed to enrich {ticker} data for {yyyymmdd}", "ERROR")
            
            return success
            
        except Exception as e:
            self.log(f"Error enriching {ticker} data for {yyyymmdd}: {str(e)}", "ERROR")
            traceback.print_exc()
            return False
    
    def enrich_ticker_data(self, ticker: str, specific_date: Optional[str] = None) -> Dict[str, bool]:
        """
        Enrich all data files for a specific ticker.
        
        Args:
            ticker: Stock ticker symbol
            specific_date: Optional specific date (YYYYMMDD) to process
            
        Returns:
            Dict: Results mapping date to success status
        """
        results = {}
        
        try:
            # Get all data files for this ticker
            ticker_dir = BASE_DIR / ticker
            
            if not ticker_dir.exists():
                self.log(f"Ticker directory not found: {ticker_dir}", "WARNING")
                return results
            
            # Find all CSV files
            csv_files = []
            for month_dir in ticker_dir.iterdir():
                if month_dir.is_dir():
                    for csv_file in month_dir.glob("data-*.csv"):
                        if specific_date:
                            # Check if this file matches the specific date
                            if f"data-{specific_date}.csv" in csv_file.name:
                                csv_files.append(csv_file)
                        else:
                            csv_files.append(csv_file)
            
            self.log(f"Found {len(csv_files)} data files for {ticker}")
            
            # Process each file
            for csv_file in csv_files:
                try:
                    # Extract date from filename
                    filename = csv_file.name
                    date_part = filename.replace("data-", "").replace(".csv", "")
                    
                    self.log(f"Processing {csv_file}")
                    success = self.engineer.enrich_file(str(csv_file))
                    results[date_part] = success
                    
                except Exception as e:
                    self.log(f"Error processing {csv_file}: {str(e)}", "ERROR")
                    results[csv_file.name] = False
            
            successful = sum(results.values())
            total = len(results)
            self.log(f"Ticker {ticker} processing complete: {successful}/{total} files enriched", "SUCCESS")
            
        except Exception as e:
            self.log(f"Error processing ticker {ticker}: {str(e)}", "ERROR")
            traceback.print_exc()
        
        return results
    
    def enrich_all_data(self) -> Dict[str, Dict[str, bool]]:
        """
        Enrich all data files in the data directory.
        
        Returns:
            Dict: Nested results mapping ticker -> date -> success status
        """
        results = {}
        
        try:
            # Get all ticker directories
            if not BASE_DIR.exists():
                self.log(f"Data directory not found: {BASE_DIR}", "WARNING")
                return results
            
            tickers = [d.name for d in BASE_DIR.iterdir() if d.is_dir()]
            
            self.log(f"Found {len(tickers)} tickers to process")
            
            # Process each ticker
            for ticker in tickers:
                self.log(f"Processing ticker: {ticker}")
                ticker_results = self.enrich_ticker_data(ticker)
                results[ticker] = ticker_results
            
            # Calculate totals
            total_files = sum(len(ticker_results) for ticker_results in results.values())
            successful_files = sum(
                sum(ticker_results.values()) for ticker_results in results.values()
            )
            
            self.log(f"All data processing complete: {successful_files}/{total_files} files enriched", "SUCCESS")
            
        except Exception as e:
            self.log(f"Error processing all data: {str(e)}", "ERROR")
            traceback.print_exc()
        
        return results
    
    def find_unenriched_files(self) -> List[str]:
        """
        Find data files that haven't been enriched yet.
        
        Returns:
            List of file paths that need enrichment
        """
        unenriched_files = []
        
        try:
            # Get all CSV files
            if not BASE_DIR.exists():
                return unenriched_files
            
            for ticker_dir in BASE_DIR.iterdir():
                if ticker_dir.is_dir():
                    for month_dir in ticker_dir.iterdir():
                        if month_dir.is_dir():
                            for csv_file in month_dir.glob("data-*.csv"):
                                # Check if file has been enriched
                                if not self._is_file_enriched(csv_file):
                                    unenriched_files.append(str(csv_file))
            
            self.log(f"Found {len(unenriched_files)} unenriched files")
            
        except Exception as e:
            self.log(f"Error finding unenriched files: {str(e)}", "ERROR")
            traceback.print_exc()
        
        return unenriched_files
    
    def _is_file_enriched(self, file_path: Path) -> bool:
        """
        Check if a file has been enriched with technical indicators.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            bool: True if file appears to be enriched
        """
        try:
            import pandas as pd
            
            # Read just the header to check columns
            df = pd.read_csv(file_path, nrows=0)
            columns = df.columns.tolist()
            
            # Check for presence of key technical indicators
            indicator_columns = ['RSI_14', 'MACD', 'Candle_Body', 'EMA_5']
            
            # If any indicator column is present, consider it enriched
            return any(col in columns for col in indicator_columns)
            
        except Exception:
            # If we can't read the file, assume it's not enriched
            return False
    
    def enrich_unenriched_files(self) -> Dict[str, bool]:
        """
        Enrich all files that haven't been enriched yet.
        
        Returns:
            Dict: Results mapping file path to success status
        """
        unenriched_files = self.find_unenriched_files()
        
        if not unenriched_files:
            self.log("No unenriched files found", "SUCCESS")
            return {}
        
        self.log(f"Enriching {len(unenriched_files)} unenriched files")
        
        # Use the engineer's batch processing
        results = self.engineer.enrich_multiple_files(unenriched_files)
        
        return results

def enrich_after_download_hook(ticker: str, yyyymmdd: str) -> bool:
    """
    Hook function to be called after downloading data.
    
    Args:
        ticker: Stock ticker symbol
        yyyymmdd: Date in YYYYMMDD format
        
    Returns:
        bool: True if successful, False otherwise
    """
    integration = FeatureIntegration()
    return integration.enrich_after_download(ticker, yyyymmdd)

def batch_enrich_all_data() -> Dict[str, Dict[str, bool]]:
    """
    Batch enrich all existing data files.
    
    Returns:
        Dict: Nested results mapping ticker -> date -> success status
    """
    integration = FeatureIntegration()
    return integration.enrich_all_data()

def batch_enrich_unenriched() -> Dict[str, bool]:
    """
    Batch enrich only files that haven't been enriched yet.
    
    Returns:
        Dict: Results mapping file path to success status
    """
    integration = FeatureIntegration()
    return integration.enrich_unenriched_files()

def enrich_ticker_batch(ticker: str, specific_date: Optional[str] = None) -> Dict[str, bool]:
    """
    Enrich all data files for a specific ticker.
    
    Args:
        ticker: Stock ticker symbol
        specific_date: Optional specific date (YYYYMMDD) to process
        
    Returns:
        Dict: Results mapping date to success status
    """
    integration = FeatureIntegration()
    return integration.enrich_ticker_data(ticker, specific_date)

if __name__ == "__main__":
    # Command line interface for batch processing
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python feature_integration.py all           # Enrich all data")
        print("  python feature_integration.py unenriched    # Enrich only unenriched files")
        print("  python feature_integration.py ticker AAPL   # Enrich specific ticker")
        print("  python feature_integration.py file AAPL 20250707  # Enrich specific file")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "all":
        print("🚀 Starting batch enrichment of all data files...")
        results = batch_enrich_all_data()
        print(f"✅ Batch enrichment completed!")
        
    elif command == "unenriched":
        print("🚀 Starting batch enrichment of unenriched files...")
        results = batch_enrich_unenriched()
        print(f"✅ Batch enrichment completed!")
        
    elif command == "ticker" and len(sys.argv) >= 3:
        ticker = sys.argv[2].upper()
        print(f"🚀 Starting enrichment for ticker {ticker}...")
        results = enrich_ticker_batch(ticker)
        print(f"✅ Ticker enrichment completed!")
        
    elif command == "file" and len(sys.argv) >= 4:
        ticker = sys.argv[2].upper()
        date = sys.argv[3]
        print(f"🚀 Starting enrichment for {ticker} on {date}...")
        integration = FeatureIntegration()
        success = integration.enrich_after_download(ticker, date)
        if success:
            print(f"✅ File enrichment completed!")
        else:
            print(f"❌ File enrichment failed!")
            
    else:
        print("❌ Invalid command. Use 'all', 'unenriched', 'ticker', or 'file'")
        sys.exit(1)
