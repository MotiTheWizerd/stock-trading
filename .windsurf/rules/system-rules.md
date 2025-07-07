---
trigger: always_on
---

📈 Trading Stock Agents — Technical Documentation
🔍 Overview
Trading Stock Agents is an automated system for downloading, analyzing, and visualizing stock market data. Its primary goal is to continuously collect intraday stock data, detect technical signals (e.g., RSI, MACD), generate charts, and save insights for further algorithmic trading research or future strategy execution.

This project is modular, file-based, and designed for future extensibility — whether through AI agents, bot automation, or signal-based trading decisions.

🎯 Project Goals
Data Collection: Continuously fetch fresh stock data from Yahoo Finance at customizable intervals.

Signal Detection: Identify technical signals such as RSI overbought/oversold conditions.

Visualization: Generate candlestick charts with indicators and save them as image files.

Storage & Structure: Organize per-stock folders with all relevant data (.csv, signals.csv, charts).

Future Extensions:

Agent-based decision making

Backtesting strategies

Notification system

Integration with brokerage APIs

📁 Project Structure
yaml
Copy
Edit
trading-stock-agents/
│
├── data/
│   ├── AAPL/
│   │   ├── data.csv
│   │   ├── signals.csv
│   │   └── charts/
│   │       └── 2025-07-06_16-30.png
│   └── ...
│
├── downloader.py        # Handles downloading & appending live data
├── utils.py             # Chart generation, indicator calculation, plotting
├── signals.py           # RSI-based signal detection and saving
├── scheduler.py         # Main loop that runs every X minutes
├── config.py            # (Optional) Global config (e.g., tickers list)
├── test.py              # Manual test runner
├── main.py              # Entry point for manual runs
└── pyproject.toml       # Poetry-managed dependencies
🧠 Key Components
1. downloader.py
Uses yfinance to download data for a given ticker.

Supports different intervals (e.g., 5m, 15m, 1d).

Appends new data to existing .csv files.

Ensures deduplication and timestamp consistency.

2. utils.py
Central visualization & analysis module.

Calculates:

RSI (Relative Strength Index)

MACD (Moving Average Convergence Divergence)

Plots:

Candlestick chart

RSI and MACD panels

BUY/SELL arrows based on RSI thresholds (30 / 70)

Saves each chart image to data/<TICKER>/charts/YYYY-MM-DD_HH-MM.png.

3. signals.py
Detects RSI crossover signals:

BUY if RSI < 30

SELL if RSI > 70

Saves all detected signals to data/<TICKER>/signals.csv, avoiding duplicates.

4. scheduler.py
Infinite loop that fetches data for selected tickers every INTERVAL_MINUTES.

Calls append_to_csv() and plot_candlestick() for each stock.

Prints log of each update and sleeps between cycles.

⚙️ Dependencies
yfinance: for stock data

pandas: for data handling

mplfinance: for chart plotting

ta: technical analysis indicators

numpy, datetime, os, time, sys

Package management is handled with Poetry.

🧪 Example Output
For a given run on AAPL, this project will produce:

yaml
Copy
Edit
data/
└── AAPL/
    ├── data.csv               # Full time-series OHLCV data
    ├── signals.csv            # Timestamps + signal type (BUY/SELL)
    └── charts/
        └── 2025-07-06_16-30.png  # Chart with RSI/MACD + markers
🚀 Future Roadmap (Suggested)
✅ Multi-stock support (done)

 Backtester module

 Signal scoring and confidence levels

 Telegram/Email alerts

 Live web dashboard (e.g., via Flask)

 Brokerage API integration (e.g., Alpaca, Interactive Brokers)

