# Enhanced Prediction System Integration

This document explains how the enhanced prediction system integrates with the existing stock trading prediction pipeline.

## Overview

The enhanced prediction system adds advanced capabilities to the existing stock prediction pipeline:

1. **Contextual Feature Generation** - Enriches raw price data with advanced technical indicators
2. **Model Calibration** - Improves prediction probability calibration
3. **Confidence Filtering** - Filters predictions based on confidence thresholds
4. **Adaptive Retraining** - Detects concept drift and manages model retraining
5. **Trading Simulation** - Realistic trading simulation with drawdown tracking

## Integration Architecture

The integration follows a non-invasive approach that preserves the existing functionality while adding enhanced capabilities:

```
┌─────────────────┐     ┌───────────────────────┐
│                 │     │                       │
│  Existing       │     │  Enhanced             │
│  Prediction     │◄────┤  Integration Layer    │
│  System         │     │                       │
│                 │     │                       │
└─────────────────┘     └───────────┬───────────┘
                                    │
                                    ▼
                        ┌───────────────────────┐
                        │                       │
                        │  Enhanced             │
                        │  Prediction           │
                        │  Components           │
                        │                       │
                        └───────────────────────┘
```

## Key Files

- **enhanced_integration.py** - Bridge between existing and enhanced systems
- **contextual_features.py** - Advanced feature generation
- **model_calibration.py** - Probability calibration and confidence filtering
- **adaptive_retraining.py** - Concept drift detection and retraining
- **enhanced_prediction.py** - Unified enhanced prediction pipeline
- **run_enhanced_pipeline.py** - Standalone script for enhanced prediction
- **test_enhanced_integration.py** - Test script for integration

## Usage

### Option 1: Using the existing predict.py with enhanced features

```bash
python predict.py --ticker AAPL --use-enhanced --confidence-threshold 0.7
```

Additional flags:
- `--use-enhanced` - Enable enhanced prediction system
- `--disable-contextual` - Disable contextual feature generation
- `--disable-calibration` - Disable model calibration
- `--confidence-threshold` - Set minimum confidence threshold (default: 0.6)
- `--check-retraining` - Check if model retraining is needed

### Option 2: Using the dedicated enhanced pipeline

```bash
python run_enhanced_pipeline.py --ticker AAPL --model-path models/AAPL/model.joblib
```

## Testing the Integration

Run the test script to verify the integration:

```bash
python test_enhanced_integration.py --ticker AAPL
```

This will run tests with different configurations and compare the results.

## Enhanced Components

### 1. Contextual Feature Generation

The `ContextualFeatureGenerator` class in `contextual_features.py` adds advanced features:

- Rolling window features (returns, volatility, momentum, z-scores)
- Technical indicators (RSI, MACD, Bollinger Bands, ATR, OBV, etc.)
- Temporal features (day of week, hour, cyclical transformations)
- Support and resistance levels

### 2. Model Calibration

The `ModelCalibrator` class in `model_calibration.py` improves prediction probabilities:

- Platt Scaling and Isotonic Regression calibration
- Calibration curve plotting
- Brier score calculation
- Confidence threshold analysis

### 3. Adaptive Retraining

The `AdaptiveRetrainer` class in `adaptive_retraining.py` manages model retraining:

- Concept drift detection
- Sliding window retraining
- Performance monitoring
- Model versioning and history tracking

### 4. Enhanced Prediction Pipeline

The `EnhancedStockPredictor` class in `enhanced_prediction.py` integrates all components:

- Loads and manages the base prediction model
- Applies contextual feature enrichment
- Calibrates prediction probabilities
- Filters predictions by confidence thresholds
- Applies realistic trading simulation

## Integration with Existing Features

The enhanced system preserves compatibility with existing features:

- **Feature Column Normalization** - Handles column name variations using the `_norm()` function
- **Feature Engineering Pipeline** - Works with the existing feature engineering pipeline
- **Trading Engine** - Compatible with the existing trading simulation

## Fallback Mechanism

If the enhanced system fails for any reason, the integration layer will automatically fall back to the standard prediction system, ensuring robustness and backward compatibility.

## Future Enhancements

1. **Uncertainty Estimation** - Add uncertainty estimation for more robust predictions
2. **Slippage and Latency Modeling** - More realistic trading simulation
3. **Multi-Ticker Portfolio Management** - Extend for portfolio management
4. **Live Data Integration** - Automate retraining with live data feeds
