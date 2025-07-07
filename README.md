# Trading Stock Agents

Automated system for downloading, analyzing, and visualizing intraday stock data (currently AAPL for testing).

This project fetches data via Yahoo Finance, appends it to per-day CSVs, detects RSI/MACD signals, and produces candlestick charts.

Run with Poetry:
```bash
poetry install  # or: poetry sync
poetry run python fetch_daily_data.py --date YYYYMMDD
```
