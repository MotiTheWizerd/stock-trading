# Enhanced Stock Prediction System

This document explains how to use the enhanced stock prediction system with advanced features for improved trading performance.

## Overview

The enhanced prediction system adds several powerful capabilities to the base stock prediction pipeline:

1. **Contextual Features** - Advanced market context indicators
2. **Model Calibration** - Ensures prediction probabilities match real-world outcomes
3. **Confidence Filtering** - Only acts on high-confidence signals
4. **Adaptive Retraining** - Automatically detects when models need updating

## Quick Start

To run the enhanced prediction pipeline on a stock:

```bash
python run_enhanced_pipeline.py --ticker AAPL --model-path models/stock_predictor.joblib
```

This will:
1. Load AAPL stock data from the data directory
2. Add contextual features
3. Calibrate the model (if enabled)
4. Make predictions with confidence filtering
5. Apply trading simulation
6. Generate visualizations in the reports directory

## Components

### 1. Contextual Features (`contextual_features.py`)

Adds advanced market context to improve prediction accuracy:

- **Window-based features**: Rolling returns, volatility, momentum, z-scores
- **Technical indicators**: RSI, MACD, Bollinger Bands, ATR, OBV
- **Temporal awareness**: Day of week, hour patterns, market open/close signals
- **Support/resistance levels**: Price barriers and distance to them

Usage:
```python
from contextual_features import ContextualFeatureGenerator

generator = ContextualFeatureGenerator()
enhanced_df = generator.add_all_features(df)
```

### 2. Model Calibration (`model_calibration.py`)

Ensures model probabilities match real-world outcomes:

- **Probability calibration**: Platt Scaling and Isotonic Regression
- **Confidence filtering**: Only act on high-confidence predictions
- **Calibration visualization**: Plots showing reliability of confidence scores
- **Threshold analysis**: Find optimal confidence cutoffs for trading

Usage:
```python
from model_calibration import ModelCalibrator

calibrator = ModelCalibrator(model_path='models/stock_predictor.joblib')
calibrator.calibrate_model(X_train, y_train, method='sigmoid')
calibrator.evaluate_calibration(X_test, y_test)
calibrator.save_calibrated_model('models/stock_predictor_calibrated.joblib')
```

### 3. Adaptive Retraining (`adaptive_retraining.py`)

Keeps models fresh as market conditions change:

- **Concept drift detection**: Automatically detect when model performance degrades
- **Sliding window retraining**: Continuously update models with recent data
- **Performance tracking**: Monitor model quality over time
- **Model versioning**: Keep history of all models with metrics

Usage:
```python
from adaptive_retraining import AdaptiveRetrainer
from model_trainer import ModelTrainer

retrainer = AdaptiveRetrainer(model_dir='models')
model_trainer = ModelTrainer()

# Check if retraining is needed
needs_retraining = retrainer.detect_concept_drift(model, X, y)

# Retrain if needed
if needs_retraining:
    retrainer.retrain_model(model_trainer, X, y, model_name='stock_predictor')
```

### 4. Enhanced Prediction (`enhanced_prediction.py`)

Integrates all components into a unified prediction pipeline:

```python
from enhanced_prediction import EnhancedStockPredictor

predictor = EnhancedStockPredictor(
    model_path='models/stock_predictor.joblib',
    confidence_threshold=0.7
)

results = predictor.predict(df, apply_trading=True, initial_balance=1000.0)
```

## Command Line Options

The `run_enhanced_pipeline.py` script accepts the following arguments:

```
--ticker TICKER           Stock ticker symbol
--model-path MODEL_PATH   Path to model file
--data-dir DATA_DIR       Directory with stock data (default: data)
--output-dir OUTPUT_DIR   Directory to save results (default: reports)
--confidence-threshold T  Confidence threshold for predictions (default: 0.7)
--initial-balance B       Initial balance for trading simulation (default: 1000.0)
--disable-contextual      Disable contextual features
--disable-calibration     Disable model calibration
```

## Example Workflow

1. **Train a base model** using `model_trainer.py`
2. **Calibrate the model** using `model_calibration.py`
3. **Run predictions** using `run_enhanced_pipeline.py`
4. **Check for retraining** periodically using `adaptive_retraining.py`

## Output

The pipeline generates:

1. CSV file with predictions and trading results
2. Price chart with buy/sell signals
3. Equity curve visualization
4. Confidence distribution histogram
5. Drawdown chart
6. Summary statistics on console

## Performance Metrics

The system tracks:

- Prediction accuracy and confidence
- Trading profit/loss
- Maximum drawdown
- Model calibration quality
- Concept drift indicators

## Advanced Usage

### Custom Feature Generation

You can add your own contextual features by extending the `ContextualFeatureGenerator` class:

```python
from contextual_features import ContextualFeatureGenerator

class MyFeatureGenerator(ContextualFeatureGenerator):
    def add_custom_features(self, df):
        # Add your custom features here
        return df
```

### Custom Calibration

For specialized calibration needs:

```python
from model_calibration import ModelCalibrator

calibrator = ModelCalibrator(model_path)
# Custom calibration parameters
calibrator.calibrate_model(X_train, y_train, method='isotonic', cv=10)
```

### Automated Retraining

Set up a scheduled job to check for concept drift and retrain as needed:

```python
from adaptive_retraining import AdaptiveRetrainer

retrainer = AdaptiveRetrainer()
retrainer.sliding_window_retrain(
    model_trainer=trainer,
    data_loader=load_data,
    model_name='stock_predictor',
    ticker='AAPL',
    window_size=30,
    step_size=7
)
```
