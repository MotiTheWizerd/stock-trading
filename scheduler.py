import time
from downloader import download_data, append_to_csv
from utils import plot_candlestick
from merger import sanitize_day_files

TICKERS = ["AAPL", "MSFT", "TSLA"]
INTERVAL_MINUTES = 60

def run_loop():
    while True:
        print("=== Running scheduled data fetch ===")
        for ticker in TICKERS:
            print(f"Fetching {ticker} | period=1d | interval=5m")
            data = download_data(ticker, period="1d", interval="5m")
            append_to_csv(data, ticker)

            # Generate and save chart in the same date folder
            plot_candlestick(data, ticker)

            # Clean up CSV artefacts after each cycle (headers as rows, tz issues)
            sanitize_day_files(ticker)

        print(f"Sleeping {INTERVAL_MINUTES} minutes...")
        time.sleep(INTERVAL_MINUTES * 60)

if __name__ == "__main__":
    run_loop()
