"""
Utility script to enrich existing data files with technical indicators

This script helps you batch process existing data files in your trading system
to add all the technical indicators for model training.
"""

import sys
import os
from feature_integration import FeatureIntegration, batch_enrich_unenriched, enrich_ticker_batch
from paths import BASE_DIR

def main():
    print("🔥 Stock Data Feature Enrichment Utility")
    print("=" * 50)
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python enrich_existing_data.py all              # Enrich all unenriched files")
        print("  python enrich_existing_data.py ticker AAPL      # Enrich specific ticker")
        print("  python enrich_existing_data.py check            # Check what needs enriching")
        return
    
    command = sys.argv[1].lower()
    
    if command == "all":
        print("🚀 Starting batch enrichment of all unenriched files...")
        print("This will only process files that haven't been enriched yet.")
        print()
        
        # Get count first
        integration = FeatureIntegration()
        unenriched_files = integration.find_unenriched_files()
        
        if not unenriched_files:
            print("✅ No unenriched files found! All data is already enriched.")
            return
        
        print(f"📊 Found {len(unenriched_files)} files to enrich")
        print()
        
        # Proceed with enrichment
        results = batch_enrich_unenriched()
        
        successful = sum(results.values())
        total = len(results)
        
        print()
        print(f"✅ Batch enrichment completed: {successful}/{total} files processed")
        
        if successful < total:
            print("❌ Some files failed to process:")
            for file_path, success in results.items():
                if not success:
                    print(f"   - {file_path}")
    
    elif command == "ticker" and len(sys.argv) >= 3:
        ticker = sys.argv[2].upper()
        print(f"🚀 Enriching data for ticker: {ticker}")
        print()
        
        results = enrich_ticker_batch(ticker)
        
        successful = sum(results.values())
        total = len(results)
        
        print()
        print(f"✅ Ticker enrichment completed: {successful}/{total} files processed")
        
        if results:
            print("📊 Processed files:")
            for date, success in results.items():
                status = "✅" if success else "❌"
                print(f"   {status} {date}")
        else:
            print(f"⚠️ No data files found for ticker {ticker}")
    
    elif command == "check":
        print("🔍 Checking which files need enrichment...")
        print()
        
        integration = FeatureIntegration()
        unenriched_files = integration.find_unenriched_files()
        
        if not unenriched_files:
            print("✅ All files are already enriched!")
            return
        
        print(f"📊 Found {len(unenriched_files)} files that need enrichment:")
        print()
        
        # Group by ticker
        ticker_files = {}
        for file_path in unenriched_files:
            # Extract ticker from path
            parts = file_path.split(os.sep)
            ticker = None
            for part in parts:
                if part.isupper() and len(part) <= 5:  # Likely a ticker
                    ticker = part
                    break
            
            if ticker:
                if ticker not in ticker_files:
                    ticker_files[ticker] = []
                ticker_files[ticker].append(file_path)
        
        for ticker, files in ticker_files.items():
            print(f"📈 {ticker}: {len(files)} files")
            for file_path in files[:3]:  # Show first 3
                print(f"   - {os.path.basename(file_path)}")
            if len(files) > 3:
                print(f"   ... and {len(files) - 3} more files")
            print()
        
        print("💡 Run 'python enrich_existing_data.py all' to enrich all files")
        print("💡 Run 'python enrich_existing_data.py ticker AAPL' to enrich specific ticker")
    
    else:
        print("❌ Invalid command. Use 'all', 'ticker', or 'check'")

if __name__ == "__main__":
    main()
