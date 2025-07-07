# Model Training Pipeline

## ✅ Current Implementation

### Model Architecture
- **Type**: `RandomForestClassifier` for BUY/SELL/NONE signal classification
- **Input**: 17 engineered features from OHLCV + technical indicators
- **Output**: Predictions with confidence scores and comprehensive evaluation metrics
- **Pipeline**: Preprocessing (StandardScaler) → RandomForest with optimized hyperparameters

### Training Pipeline Components

#### 1. **Data Loading** (`train_model.py`)
   - Aggregates historical data from `data/{ticker}/{date}/data.csv`
   - Merges with labeled signals from `signals.csv`
   - Supports configurable lookback periods
   - Handles missing data and deduplication

#### 2. **Feature Engineering** (Integrated with `feature_engineer.py`)
   - **17 Technical Indicators**: RSI, MACD, EMA, SMA, Bollinger Bands, Volume analysis
   - **Price Patterns**: Candle body, shadows, volatility measures
   - **Momentum**: Price/volume percentage changes
   - **Automatic Enrichment**: Seamless integration with existing data pipeline

#### 3. **Model Training** (`ModelTrainer` class)
   - **Algorithm**: `RandomForestClassifier` with optimized hyperparameters
   - **Cross-validation**: Time series split (5-fold)
   - **Class Balancing**: Automatic handling of imbalanced datasets
   - **Preprocessing**: StandardScaler for feature normalization
   - **Hyperparameters**: 300 estimators, max_depth=15, class_weight='balanced'

#### 4. **Model Evaluation** (`ModelEvaluator` class)
   - **Performance Metrics**: Accuracy, Precision, Recall, F1-Score
   - **Confidence Analysis**: Prediction probability distributions
   - **Feature Importance**: Top contributing features visualization
   - **Backtesting**: PnL simulation with configurable parameters
   - **Risk Analysis**: Drawdown, Sharpe ratio, win/loss ratios

#### 5. **Model Persistence** (Versioned Storage)
   - **Model Files**: `models/{ticker}/model_{ticker}_{timestamp}.joblib`
   - **Metadata**: Feature columns, training timestamp, evaluation results
   - **Version Control**: Timestamped model versions with metadata
   - **Plots**: Feature importance, confusion matrix, confidence histograms

## Model Evaluation

### Performance Metrics
- **Success Rate**: Accuracy of 1-candle and 3-candle predictions
- **Signal Confidence**: Using `predict_proba()` for probability estimates
- **PnL Simulation**: Virtual trading results
- **Signal Distribution**: Balance between BUY/SELL/NONE signals
- **Confidence Histograms**: Visualize prediction certainty

### Risk Analysis
- Drawdown metrics
- Win/Loss ratios
- Risk-adjusted returns (Sharpe/Sortino ratios)

## Usage

### Training a New Model
```bash
# Train model for single ticker with evaluation
python train_model.py --ticker=AAPL --model=randomforest --evaluate

# Train models for all available tickers
python train_model.py --all-tickers --model=randomforest --evaluate

# Train without evaluation (faster)
python train_model.py --ticker=AAPL --model=randomforest
```

### Making Predictions
```python
from model_inference import ModelInference

# Single ticker inference
inference = ModelInference('AAPL')
inference.load_model()  # Loads latest model automatically

# Get latest prediction
result = inference.predict_latest()
print(f"Signal: {result['prediction']}, Confidence: {result['confidence']:.3f}")

# Batch inference for multiple tickers
from model_inference import BatchInference

batch = BatchInference(['AAPL', 'GOOGL', 'MSFT'])
predictions = batch.predict_all()

# Get high-confidence signals only
high_conf_signals = batch.get_high_confidence_signals(min_confidence=0.8)
```

### Model Evaluation and Backtesting
```python
from model_evaluator import ModelEvaluator

# Initialize evaluator
evaluator = ModelEvaluator('AAPL')
evaluator.load_model()  # Loads latest model

# Run comprehensive evaluation
evaluation_results = evaluator.evaluate_model(test_data)

# Run backtesting with PnL simulation
backtest_results = evaluator.run_backtest(test_data, initial_balance=10000)

# Generate HTML report
report_path = evaluator.save_evaluation_report(evaluation_results, backtest_results)
```

### Command Line Interface
```bash
# Model training
python train_model.py --ticker=AAPL --evaluate

# Model inference
python model_inference.py --ticker=AAPL --confidence=0.7
python model_inference.py --tickers AAPL GOOGL MSFT --min-confidence=0.8

# Model evaluation
python model_evaluator.py --ticker=AAPL --test-days=30
```

## Roadmap

### Phase 1 (✅ Completed)
- [x] Advanced RandomForest implementation with hyperparameter optimization
- [x] Comprehensive evaluation metrics (accuracy, precision, recall, F1)
- [x] PnL simulation framework with backtesting
- [x] Advanced visualization (feature importance, confusion matrix, confidence histograms)
- [x] Confidence-based signal filtering
- [x] Risk analysis (drawdown, Sharpe ratio, Sortino ratio)
- [x] Model versioning and persistence
- [x] Batch inference for multiple tickers
- [x] Integration with feature engineering pipeline
- [x] Comprehensive test suite

### Phase 2 (🔄 In Progress)
- [x] Confidence-based filtering (implemented)
- [ ] Volatility-adaptive signals
- [ ] Smart holding period optimization
- [ ] Multi-timeframe analysis
- [ ] Real-time inference service API

### Future Enhancements
- **Advanced Features**:
  - Pattern matching for historical setups
  - Real-time inference service
  - Multi-timeframe analysis
  - News sentiment integration

- **Performance**:
  - Model ensembling
  - Hyperparameter optimization
  - Feature importance analysis

## Contributing
Contributions to improve the model pipeline are welcome! Please see our [contribution guidelines](../CONTRIBUTING.md) for more details.
