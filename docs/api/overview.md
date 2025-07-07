# API Reference

This section documents the main modules and their public interfaces.

## Core Modules

### 1. Feature Engineer
```{eval-rst}
.. automodule:: feature_engineer
   :members:
   :undoc-members:
   :show-inheritance:
```

### 2. Data Downloader
```{eval-rst}
.. automodule:: downloader
   :members:
   :undoc-members:
   :show-inheritance:
```

### 3. Signal Generator
```{eval-rst}
.. automodule:: signals
   :members:
   :undoc-members:
   :show-inheritance:
```

## Configuration

### Config Module
```{eval-rst}
.. automodule:: config
   :members:
   :undoc-members:
```

## Utilities

### Paths Utility
```{eval-rst}
.. automodule:: paths
   :members:
   :undoc-members:
```

### Data Processing
```{eval-rst}
.. automodule:: utils
   :members:
   :undoc-members:
```

## Command Line Interface

### Fetch Daily Data
```
usage: fetch_daily_data.py [-h] [--date DATE] [--tickers TICKERS [TICKERS ...]]

Fetch daily stock data.

options:
  -h, --help            show this help message and exit
  --date DATE           Date in YYYYMMDD format (default: today)
  --tickers TICKERS [TICKERS ...]
                        List of tickers to fetch (default: from config)
```

### Backfill Data
```
usage: backfill.py [-h] [--start START] [--end END] [--tickers TICKERS [TICKERS ...]]

Backfill historical stock data.

options:
  -h, --help            show this help message and exit
  --start START         Start date in YYYYMMDD format (default: 20230101)
  --end END             End date in YYYYMMDD format (default: today)
  --tickers TICKERS [TICKERS ...]
                        List of tickers to fetch (default: from config)
```

## Data Structures

### Stock Data
```python
{
    'timestamp': '2023-07-07 09:30:00-04:00',
    'open': 150.0,
    'high': 151.5,
    'low': 149.8,
    'close': 150.5,
    'volume': 1000000,
    'rsi': 65.2,
    'macd': 0.5,
    'macd_signal': 0.4,
    'macd_hist': 0.1,
    'sma_5': 149.8,
    'sma_10': 148.5,
    'ema_5': 150.1,
    'ema_10': 149.2,
    'bb_upper': 152.3,
    'bb_middle': 150.0,
    'bb_lower': 147.7
}
```

## Error Handling
Common exceptions and how to handle them:

- `DataFetchError`: Failed to fetch data from Yahoo Finance
- `DataValidationError`: Invalid or missing data
- `FileOperationError`: Issues with file I/O operations

## Logging
All modules use Python's built-in logging with the following levels:
- INFO: General operational messages
- WARNING: Non-critical issues
- ERROR: Critical failures
- DEBUG: Detailed debugging information

To enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```
