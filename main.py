# main.py
from config import TICKERS, START_DATE, END_DATE
from downloader import download_data, save_to_csv
from utils import plot_candlestick

def main():
    for ticker in TICKERS:
        data = download_data(ticker, START_DATE, END_DATE)
        if data is not None and not data.empty:
            save_to_csv(data, ticker)
            plot_candlestick(data, ticker)

if __name__ == "__main__":
    main()
