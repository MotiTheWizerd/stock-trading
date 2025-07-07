# backfill.py
import os
from datetime import datetime, timedelta
import yfinance as yf
from downloader import append_to_csv
from utils import plot_candlestick
from merger import sanitize_day_files
from config import TICKERS  # Same list used in scheduler

# You can tweak these per your needs
INTERVAL = "30m"  # 30-minute bars
SLICE_DAYS = 30       # each request covers 30 days
NUM_SLICES = 6        # 6 × 30 days ≈ 6 months

def backfill_all():
    total_days = SLICE_DAYS * NUM_SLICES
    print(f"=== Backfilling historical data: {total_days} days / {INTERVAL} ===")
    today = datetime.utcnow().date()
    start_date = today - timedelta(days=SLICE_DAYS * NUM_SLICES)

    for ticker in TICKERS:
        print(f"\n=== {ticker} | Back-filling last {NUM_SLICES*SLICE_DAYS} days ({INTERVAL}) ===")
        for i in range(NUM_SLICES):
            slice_start = start_date + timedelta(days=SLICE_DAYS * i)
            slice_end = slice_start + timedelta(days=SLICE_DAYS)

            print(f" → slice {i+1}/{NUM_SLICES}: {slice_start} → {slice_end} [{INTERVAL}]")
            data = yf.download(
                ticker,
                start=slice_start.strftime("%Y-%m-%d"),
                end=slice_end.strftime("%Y-%m-%d"),
                interval=INTERVAL,
            )
            if data.empty:
                print(f"   ⚠️  No data returned for this slice – skipping")
                continue

            append_to_csv(data, ticker)
            plot_candlestick(data, ticker)
            sanitize_day_files(ticker)

    print("✅ Backfill complete.")

if __name__ == "__main__":
    backfill_all()
