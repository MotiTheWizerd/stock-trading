# Trading Stock Agents

## Overview
Trading Stock Agents is an automated system for downloading, analyzing, and visualizing intraday stock market data. The system is designed to continuously collect stock data, perform technical analysis, generate trading signals, and produce visual representations of market activity.

## Key Features
- **Automated Data Collection**: Fetches intraday stock data from Yahoo Finance API
- **Feature Engineering**: Calculates 17+ technical indicators for enhanced analysis
- **Signal Generation**: Identifies potential trading opportunities using RSI and MACD
- **Data Visualization**: Generates comprehensive candlestick charts with technical indicators
- **Modular Architecture**: Designed for extensibility and easy integration of new features

## System Architecture
1. **Data Collection Layer**: Handles data fetching and storage
2. **Feature Engineering**: Processes raw data into meaningful features
3. **Analysis Engine**: Performs technical analysis and signal generation
4. **Visualization**: Creates charts and reports
5. **Model Training**: (Future) Machine learning model training and backtesting

## Getting Started
See our [Getting Started Guide](getting_started.md) for installation and basic usage instructions.

## Documentation
- [Data Pipeline](data_pipeline/overview.md)
- [Feature Engineering](features/overview.md)
- [Model Training](models/overview.md)
- [API Reference](api/overview.md)

## License
MIT License - See [LICENSE](LICENSE) for details.
