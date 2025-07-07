# Model Training Pipeline

This document provides a comprehensive guide to training machine learning models for trading signal prediction.

## Prerequisites

Before training models, ensure you have:
1. Historical price data in `data/[TICKER]/[YYYYMM]/data-YYYYMMDD.csv`
2. Generated signals in `data/[TICKER]/[YYYYMM]/signals-YYYYMMDD.csv`
3. Required Python packages installed (`scikit-learn`, `pandas`, `numpy`, etc.)

## Training a Single Model

### Basic Training
```bash
# Train a model for a specific ticker
python -m train_model --ticker=AAPL

# Specify model type (default: randomforest)
python -m train_model --ticker=AAPL --model=randomforest

# Skip evaluation (faster training)
python -m train_model --ticker=AAPL --no-evaluate
```

### Advanced Options
```bash
# Train with custom lookback window (days)
python -m train_model --ticker=AAPL --lookback=60

# Set random seed for reproducibility
python -m train_model --ticker=AAPL --seed=42

# Control model verbosity
python -m train_model --ticker=AAPL --verbose
```

## Batch Training

### Train Models for All Tickers
```bash
# Train models for all tickers in config
python -m train_model --all-tickers

# Limit to specific tickers
python -m train_model --tickers=AAPL,MSFT,GOOG

# Parallel training (faster for multiple tickers)
python -m train_model --all-tickers --n-jobs=4
```

## Model Evaluation

### Generate Evaluation Reports
```bash
# Generate evaluation report after training
python -m train_model --ticker=AAPL --evaluate

# Save evaluation plots to custom directory
python -m train_model --ticker=AAPL --plots-dir=reports/plots
```

### Backtesting
```bash
# Run backtest after training
python -m train_model --ticker=AAPL --backtest

# Backtest with custom parameters
python -m train_model --ticker=AAPL --backtest --initial-cash=10000 --commission=0.001
```

## Model Persistence

Trained models are saved to `models/[TICKER]/` with timestamps. Example structure:
```
models/
└── AAPL/
    ├── 20250707_153045/          # Timestamped model version
    │   ├── model.joblib          # Serialized model
    │   ├── metadata.json         # Training metadata
    │   └── evaluation/           # Evaluation results
    │       ├── confusion_matrix.png
    │       ├── feature_importance.png
    │       └── metrics.json
    └── latest -> 20250707_153045/  # Symlink to latest version
```

### Loading a Trained Model
```python
import joblib
from pathlib import Path

# Load the latest model
def load_latest_model(ticker):
    model_path = Path(f"models/{ticker}/latest/model.joblib")
    return joblib.load(model_path)

model = load_latest_model('AAPL')
```

## Hyperparameter Tuning

### Grid Search
```bash
# Run grid search for hyperparameter optimization
python -m tune_hyperparameters --ticker=AAPL

# Use specific parameter grid
python -m tune_hyperparameters --ticker=AAPL --param-grid=grids/rf_params.json
```

### Cross-Validation
```bash
# Use time-series cross-validation
python -m train_model --ticker=AAPL --cv-folds=5

# Specify custom time-series split
python -m train_model --ticker=AAPL --cv-window=30
```

## Monitoring and Logging

### View Training Logs
```bash
# Follow training logs in real-time
tail -f logs/training.log

# Filter logs for specific ticker
grep "AAPL" logs/training.log
```

### MLflow Integration
```bash
# Enable MLflow tracking
export MLFLOW_TRACKING_URI=file:./mlruns
python -m train_model --ticker=AAPL --mlflow

# View MLflow UI
mlflow ui
```

## Common Issues and Solutions

1. **Insufficient Data**
   ```
   Error: Not enough samples for training
   Solution: Increase --lookback or collect more data
   ```

2. **Class Imbalance**
   ```
   Warning: Class imbalance detected
   Solution: Use --class-weight=balanced or adjust signal thresholds
   ```

3. **Memory Issues**
   ```
   Error: Unable to allocate array with shape (...)
   Solution: Reduce --lookback or use --n-jobs=1
   ```

4. **Missing Dependencies**
   ```
   ModuleNotFoundError: No module named '...'
   Solution: Run `pip install -r requirements.txt`
   ```

For additional help, run:
```bash
python -m train_model --help
```
