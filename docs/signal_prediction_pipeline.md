# Signal Prediction Pipeline

## Overview

The Signal Prediction Pipeline is a comprehensive system for generating trading signals (BUY, SELL, NONE) based on technical indicators and machine learning models. The pipeline integrates multiple components to provide reliable trading signals with confidence scores and risk assessments.

## Pipeline Architecture

The prediction pipeline consists of the following key components:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Data Loading   │────▶│ Feature         │────▶│ Model           │
│  & Preparation  │     │ Engineering     │     │ Prediction      │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Signal         │◀────│ Risk            │◀────│ Confidence      │
│  Generation     │     │ Assessment      │     │ Calibration     │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │
        ▼
┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │
│  Trading        │────▶│ Batch           │
│  Simulation     │     │ Processing      │
│                 │     │                 │
└─────────────────┘     └─────────────────┘
```

## Core Components

### 1. ModelInference (`src/trading/models/model_inference.py`)

The `ModelInference` class provides real-time inference capabilities for trained trading models:

- Loading trained models for inference
- Making predictions on new data
- Confidence scoring and filtering
- Integration with the existing data pipeline
- Support for probability calibration

Key methods:
- `load_model(model_path)`: Load a trained model for inference
- `predict(data, return_probabilities)`: Make predictions on new data
- `predict_latest()`: Make prediction on the latest available data
- `get_model_info()`: Get information about the loaded model

### 2. BatchInference (`src/trading/models/model_inference.py`)

The `BatchInference` class extends inference capabilities to multiple tickers simultaneously:

- Batch prediction for multiple tickers
- Filtering signals by confidence threshold
- Extracting BUY/SELL signals across multiple assets
- Creating signal summaries for reporting

Key methods:
- `predict_all()`: Make predictions for all tickers
- `get_high_confidence_signals(min_confidence)`: Get only high-confidence signals
- `get_buy_signals(min_confidence)`: Get BUY signals with minimum confidence
- `get_sell_signals(min_confidence)`: Get SELL signals with minimum confidence
- `create_signals_summary()`: Create a summary DataFrame of all signals

### 3. FeatureEngineer (`src/trading/features/feature_engineer.py`)

The `FeatureEngineer` class is responsible for transforming raw OHLCV data into feature-rich datasets for model training and prediction:

- Calculates technical indicators and features from raw price data
- Supports customizable feature generation through configuration
- Handles data validation and cleaning
- Can process individual dataframes or entire CSV files

Key features generated include:
- Price-based features (candle body, shadows, volatility)
- Moving averages (SMA, EMA)
- Momentum indicators (RSI)
- Trend indicators (MACD)
- Volatility indicators (Bollinger Bands)

Key methods:
- `enrich_dataframe(df)`: Apply feature engineering to a DataFrame
- `enrich_file(file_path)`: Process a CSV file and add features
- `enrich_multiple_files(file_paths)`: Batch process multiple files

### 4. StockPredictor (`src/trading/models/predictor.py`)

The `StockPredictor` class is the foundation of the prediction pipeline. It handles:

- Loading trained models from joblib/pickle files
- Feature column mapping and normalization
- Making predictions on new data
- Calculating prediction confidence scores
- Risk assessment based on prediction confidence

Key methods:
- `__init__(ticker, model_dir)`: Initialize with a ticker symbol or model path
- `_load_model()`: Load the model from file
- `predict(df)`: Generate predictions with confidence scores
- `get_feature_columns()`: Get feature columns used by the model

The `StockPredictor` handles column name normalization to ensure compatibility between training and prediction data using the `_norm()` function, which:
- Converts column names to lowercase
- Removes spaces, underscores, and hyphens
- Creates a mapping between normalized data columns and model feature columns

### 2. EnhancedStockPredictor (`src/trading/models/enhanced_prediction.py`)

The `EnhancedStockPredictor` extends the base predictor with advanced capabilities:

- Contextual feature generation
- Model calibration for improved probability estimates
- Adaptive retraining detection
- Confidence filtering with customizable thresholds

Key methods:
- `enhance_features(df)`: Add contextual features to the input data
- `calibrate_predictions(predictions_df)`: Apply calibration to raw model predictions
- `filter_by_confidence(predictions_df)`: Filter predictions based on confidence threshold
- `predict(df, apply_trading)`: Execute the full enhanced prediction pipeline
- `check_for_retraining(df)`: Detect when model retraining is needed

### 3. CustomLossTrainer (`src/trading/models/model_trainer.py`)

The `CustomLossTrainer` class is responsible for training models with specialized loss functions optimized for trading:

- Custom loss functions that penalize false signals more than missed opportunities
- Hard negative mining to focus on difficult examples
- Class balancing techniques for imbalanced datasets
- Cross-validation with proper stratification

Key methods:
- `train_with_custom_loss(model_type, balance_classes)`: Train model with custom loss function
- `evaluate_model(model, X_train, y_train, X_val, y_val)`: Evaluate model performance
- `cross_validate_model(model_type, n_splits, balance_classes)`: Perform cross-validation
- `save_model(model, model_path)`: Save trained model to file

### 4. Signal Generation (`src/trading/features/signals.py`)

The `signals.py` module detects trading signals based on technical indicators:

- RSI-based signal detection (BUY when RSI < 30, SELL when RSI > 70)
- Signal persistence to CSV files
- Deduplication of signals

Key function:
- `detect_rsi_signals(df, ticker, date_folder)`: Detect and save RSI-based signals

### 5. SimpleTrader (`scripts/predict.py`)

The `SimpleTrader` class simulates trading based on generated signals:

- Realistic position sizing based on equity
- Transaction costs and short sale margin requirements
- Proper P&L calculation
- Maximum drawdown tracking

Key methods:
- `trade(signal, price, risk, timestamp)`: Execute a trade based on signal and risk
- `_update_equity(price, timestamp)`: Update equity based on current position and price
- `_update_drawdown(timestamp)`: Update equity curve and calculate drawdown metrics

## Prediction Process Flow

1. **Data Loading**:
   - Load historical price data from CSV files
   - Ensure datetime index is properly formatted

2. **Feature Engineering**:
   - Calculate technical indicators (RSI, MACD, Bollinger Bands, etc.)
   - Generate contextual features if using enhanced prediction
   - Normalize feature names to match model expectations

3. **Model Prediction**:
   - Load the trained model (joblib/pickle format)
   - Extract feature columns from the model
   - Map input data columns to model feature columns
   - Generate raw predictions and probability scores

4. **Confidence Calibration** (Enhanced mode):
   - Apply probability calibration to improve confidence estimates
   - Handle class imbalance effects on probability scores

5. **Risk Assessment**:
   - Calculate risk scores based on prediction confidence
   - Assign risk tiers (Low Risk/High Risk)
   - Generate risk-adjusted predictions

6. **Signal Generation**:
   - Filter predictions based on confidence threshold
   - Generate final BUY/SELL/NONE signals
   - Add risk indicators to signals

7. **Trading Simulation** (Optional):
   - Apply signals to a simulated trading account
   - Track position, equity, and drawdown
   - Calculate performance metrics

## Usage Examples

### Feature Engineering

```python
from trading.features.feature_engineer import FeatureEngineer, FeatureEngineerConfig

# Create a custom configuration
config = FeatureEngineerConfig()
config.rsi = True
config.macd = True
config.bollinger = True

# Initialize the feature engineer
engineer = FeatureEngineer(config)

# Process a single DataFrame
import pandas as pd
df = pd.read_csv("data/AAPL/data.csv")
enriched_df = engineer.enrich_dataframe(df)

# Process multiple files
file_paths = ["data/AAPL/data.csv", "data/MSFT/data.csv"]
results = engineer.enrich_multiple_files(file_paths)
```

### Basic Prediction with StockPredictor

```python
from trading.models.predictor import StockPredictor

# Initialize predictor with ticker and model directory
predictor = StockPredictor(ticker="AAPL", model_dir="models/AAPL")

# Load data
import pandas as pd
df = pd.read_csv("data/AAPL/data.csv")

# Make predictions
predictions = predictor.predict(df)

# Filter for BUY/SELL signals
signals = predictions[predictions['prediction'] != 'NONE']
```

### Real-time Inference with ModelInference

```python
from trading.models.model_inference import ModelInference

# Initialize inference engine
inference = ModelInference('AAPL', confidence_threshold=0.7)
inference.load_model()

# Make prediction on latest data
latest_prediction = inference.predict_latest()
print(f"Signal: {latest_prediction['prediction']}")
print(f"Confidence: {latest_prediction['confidence']:.3f}")

# Or predict on custom data
import pandas as pd
data = pd.read_csv("data/AAPL/latest.csv")
prediction = inference.predict(data)
```

### Batch Inference for Multiple Tickers

```python
from trading.models.model_inference import BatchInference

# Initialize batch inference for multiple tickers
batch_inference = BatchInference(['AAPL', 'MSFT', 'GOOGL'], confidence_threshold=0.7)

# Get predictions for all tickers
all_predictions = batch_inference.predict_all()

# Get only high confidence BUY signals
buy_signals = batch_inference.get_buy_signals(min_confidence=0.8)
for signal in buy_signals:
    print(f"{signal['ticker']}: BUY with {signal['confidence']:.3f} confidence")

# Create a summary DataFrame
summary = batch_inference.create_signals_summary()
summary.to_csv("signals_summary.csv")
```

### Enhanced Prediction

```python
from trading.models.enhanced_prediction import EnhancedStockPredictor

# Initialize enhanced predictor
predictor = EnhancedStockPredictor(
    model_path="models/AAPL/model_AAPL.joblib",
    confidence_threshold=0.7,
    enable_contextual_features=True,
    enable_calibration=True
)

# Load data
import pandas as pd
df = pd.read_csv("data/AAPL/data.csv")

# Make enhanced predictions with trading simulation
results = predictor.predict(
    df=df,
    apply_trading=True,
    initial_balance=1000.0
)

# Access trading performance metrics
final_balance = results['final_balance'].iloc[-1]
max_drawdown = results['max_drawdown'].max()
```

### Command-Line Usage

The prediction pipeline can also be used from the command line:

```bash
python scripts/predict.py --ticker AAPL --data-file data/AAPL/data.csv --output predictions/AAPL_pred.csv
```

For enhanced prediction:

```bash
python -m trading.models.enhanced_prediction --data-file data/AAPL/data.csv --model-path models/AAPL/model_AAPL.joblib --output-file predictions/AAPL_enhanced.csv
```

## Model Training

The prediction pipeline relies on trained models. Models are trained using the `CustomLossTrainer`:

```python
from trading.models.model_trainer import CustomLossTrainer

# Initialize trainer with training data
trainer = CustomLossTrainer(data_path="data/AAPL/training_data.csv")

# Prepare features and targets
trainer.prepare_features_targets(target_col='prediction')

# Train model with custom loss function
model = trainer.train_with_custom_loss(model_type='lightgbm', balance_classes=True)

# Save model
trainer.save_model(model, "models/AAPL/model_AAPL.joblib")
```

## Visualization

Prediction results can be visualized using the CLI output module:

```python
from trading.ui.cli_output import render_predictions, render_summary

# Display predictions with rich formatting
render_predictions(predictions_df, limit=30)

# Display summary of trading performance
render_summary(results, title="Trading Performance")
```

## Integration Points

The prediction pipeline integrates with other system components:

1. **Data Pipeline**: Receives data from the data downloading and processing pipeline
2. **Feature Engineering**: Uses the feature engineering pipeline to calculate technical indicators
3. **Trading System**: Provides signals to the trading execution system
4. **Visualization**: Sends results to the visualization components for display
5. **Batch Processing**: Supports batch inference across multiple tickers
6. **Alerting System**: Can trigger alerts based on high-confidence signals

## Error Handling and Robustness

The prediction pipeline includes several mechanisms to ensure robustness:

1. **Column Name Normalization**: Handles variations in column naming conventions
2. **Feature Mapping**: Maps available features to model requirements
3. **Confidence Filtering**: Filters out low-confidence predictions
4. **Risk Assessment**: Provides risk indicators for remaining predictions
5. **Exception Handling**: Gracefully handles missing data and model errors

## Future Enhancements

Planned enhancements to the prediction pipeline include:

1. **Ensemble Models**: Combining multiple models for more robust predictions
2. **Online Learning**: Continuously updating models with new data
3. **Anomaly Detection**: Identifying unusual market conditions
4. **Multi-timeframe Analysis**: Incorporating signals from multiple timeframes
5. **Sentiment Analysis**: Adding market sentiment features from news and social media
6. **Advanced Feature Engineering**: Adding more sophisticated technical indicators
7. **Feature Selection**: Automated feature importance analysis and selection
8. **Hyperparameter Optimization**: Automated tuning of model parameters
