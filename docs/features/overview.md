# Feature Engineering Pipeline

The feature engineering pipeline transforms raw OHLCV (Open, High, Low, Close, Volume) data into meaningful features for analysis and modeling.

## Available Features

### 1. Price Analysis
- **Candle Body**: Difference between Open and Close prices
- **Upper Shadow**: Difference between High and the higher of Open/Close
- **Lower Shadow**: Difference between the lower of Open/Close and Low
- **Volatility**: High - Low price range

### 2. Market Dynamics
- **Close Percentage Change**: Daily price change percentage
- **Volume Percentage Change**: Daily volume change percentage

### 3. Trend Following
- **Exponential Moving Averages (EMA)**: 5 and 10 periods
- **Simple Moving Averages (SMA)**: 5 and 10 periods

### 4. Momentum
- **RSI (14-period)**: Relative Strength Index

### 5. Convergence/Divergence
- **MACD**: Moving Average Convergence Divergence
- **MACD Signal Line**
- **MACD Histogram**

### 6. Volatility Bands
- **Bollinger Bands**: Upper, Middle, and Lower bands (20-period, 2 std)

## Usage

### Basic Usage
```python
from feature_engineer import FeatureEngineer

# Initialize with default settings
engineer = FeatureEngineer()

# Enrich a DataFrame
df = pd.read_csv('data/AAPL/202307/data-20230707.csv')
enriched_df = engineer.enrich_dataframe(df)

# Or process a file directly
engineer.enrich_file('data/AAPL/202307/data-20230707.csv')
```

### Custom Configuration
```python
from feature_engineer import FeatureEngineer, FeatureEngineerConfig

# Create custom config
config = FeatureEngineerConfig()
config.rsi = False  # Disable RSI calculation
config.bollinger_bands = False  # Disable Bollinger Bands

# Initialize with custom config
engineer = FeatureEngineer(config=config)
```

## Implementation Details

### FeatureEngineer Class
Main class that handles the feature engineering pipeline:
- `enrich_dataframe()`: Process a pandas DataFrame
- `enrich_file()`: Process a CSV file
- `enrich_multiple_files()`: Process multiple files

### Feature Configuration
All features can be toggled on/off through the `FeatureEngineerConfig` class:
```python
class FeatureEngineerConfig:
    def __init__(self):
        # Price-based features
        self.candle_body = True
        self.shadows = True
        self.volatility = True
        
        # Moving Averages
        self.ema = True
        self.sma = True
        
        # Momentum
        self.rsi = True
        
        # MACD
        self.macd = True
        
        # Bollinger Bands
        self.bollinger_bands = True
```

## Best Practices
1. **Data Validation**: Always validate input data before processing
2. **Incremental Updates**: Process new data incrementally
3. **Error Handling**: Implement proper error handling for data quality issues
4. **Logging**: Monitor the feature engineering process with appropriate logging

## Troubleshooting
Common issues and solutions:
- **Missing Data**: Ensure all required columns (Open, High, Low, Close, Volume) are present
- **Incorrect Data Types**: Verify numeric columns are properly typed
- **Date Index**: Ensure the DataFrame has a proper datetime index

For more advanced usage, see the [API Reference](../api/feature_engineer.md).
