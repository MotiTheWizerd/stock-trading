import time
from datetime import datetime
import schedule
from downloader import download_data, append_to_csv
from utils import plot_candlestick
from merger import sanitize_day_files
from config import TICKERS
INTERVAL_MINUTES = 60

# ------------------------------
# Daily Task Scheduling Section
# ------------------------------
# In-memory task queues per ticker. In a real application this could be a
# database table, Redis list, etc. For demonstration purposes we keep it
# simple and in-process.
task_queues = {ticker: [] for ticker in TICKERS}

def enqueue_daily_tasks():
    """Append a new placeholder task (current UTC timestamp) to each ticker's queue."""
    now = datetime.utcnow().isoformat()
    for ticker in TICKERS:
        task = {"timestamp": now}
        task_queues[ticker].append(task)
        print(f"[{now}] Enqueued daily task for {ticker}. Queue size: {len(task_queues[ticker])}")

def start_daily_scheduler(run_time: str = "09:00"):
    """Start a background loop that runs `enqueue_daily_tasks` once per day at `run_time` (HH:MM, 24-hour)."""
    print(f"Starting daily scheduler – tasks will enqueue every day at {run_time} UTC.")
    schedule.clear()
    schedule.every().day.at(run_time).do(enqueue_daily_tasks)

    try:
        while True:
            schedule.run_pending()
            time.sleep(30)  # Poll twice per minute
    except KeyboardInterrupt:
        print("Scheduler stopped by user.")

# ------------------------------
# Interval-Based Scheduling Section
# ------------------------------

def start_interval_scheduler(interval_minutes: int = 30):
    """Start a background loop that runs `enqueue_daily_tasks` every `interval_minutes`."""
    print(f"Starting interval scheduler – tasks will enqueue every {interval_minutes} minutes.")
    schedule.clear()
    schedule.every(interval_minutes).minutes.do(enqueue_daily_tasks)

    try:
        while True:
            schedule.run_pending()
            time.sleep(5)  # Check every 5 seconds for snappier execution
    except KeyboardInterrupt:
        print("Interval scheduler stopped by user.")

# ------------------------------
# Existing Interval Fetch Loop (data download)
# ------------------------------
# ------------------------------

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
    # Uncomment ONE of the following as needed.

    # 1. Continuous intraday fetch every `INTERVAL_MINUTES` minutes
    # run_loop()

    # 2. Daily enqueue of tasks at a specific time (UTC)
    # start_daily_scheduler("09:00")

    # 3. Enqueue tasks every 30 minutes
    start_interval_scheduler(30)
