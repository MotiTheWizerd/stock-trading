"""
Model Inference Module
=====================

This module provides real-time inference capabilities for trained trading models.
It supports:

1. Loading trained models for inference
2. Making predictions on new data
3. Confidence scoring and filtering
4. Batch prediction for multiple tickers
5. Integration with existing data pipeline

Usage:
    from model_inference import ModelInference
    
    # Single ticker inference
    inference = ModelInference('AAPL')
    prediction = inference.predict(current_data)
    
    # Batch inference for multiple tickers
    batch_inference = BatchInference(['AAPL', 'GOOGL', 'MSFT'])
    predictions = batch_inference.predict_all()
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

from feature_engineer import FeatureEngineer
from paths import get_data_path, get_models_path

logger = logging.getLogger(__name__)


class ModelInference:
    """Real-time model inference for trading signals."""
    
    def __init__(self, ticker: str, confidence_threshold: float = 0.6):
        """Initialize inference engine for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            confidence_threshold: Minimum confidence for signal generation
        """
        self.ticker = ticker.upper()
        self.confidence_threshold = confidence_threshold
        self.models_path = get_models_path(ticker)
        self.feature_engineer = FeatureEngineer()
        
        self.model = None
        self.model_metadata = None
        self.feature_columns = []
        
    def load_model(self, model_path: Optional[Path] = None) -> None:
        """Load a trained model for inference.
        
        Args:
            model_path: Path to model file. If None, loads latest model.
        """
        if model_path is None:
            model_path = self._get_latest_model_path()
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = joblib.load(model_path)
        
        # Load metadata
        metadata_path = self._get_metadata_path(model_path)
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.model_metadata = json.load(f)
                self.feature_columns = self.model_metadata.get('feature_columns', [])
        
        logger.info(f"✅ Model loaded for {self.ticker}: {model_path.name}")
    
    def _get_latest_model_path(self) -> Path:
        """Get the path to the latest model file."""
        model_files = list(self.models_path.glob(f"model_{self.ticker}_*.joblib"))
        if not model_files:
            raise FileNotFoundError(f"No model files found for {self.ticker}")
        
        # Sort by timestamp and get the latest
        model_files.sort(key=lambda x: x.stem.split('_')[-1])
        return model_files[-1]
    
    def _get_metadata_path(self, model_path: Path) -> Path:
        """Get metadata path for a model file."""
        model_stem = model_path.stem
        timestamp = model_stem.split('_')[-1]
        return model_path.parent / f"metadata_{self.ticker}_{timestamp}.json"
    
    def predict(self, data: pd.DataFrame, return_probabilities: bool = False) -> Union[Dict, List[Dict]]:
        """Make predictions on new data.
        
        Args:
            data: DataFrame with OHLCV data
            return_probabilities: Whether to return prediction probabilities
            
        Returns:
            Dictionary with prediction results or list of dictionaries for multiple rows
        """
        if self.model is None:
            raise ValueError("Model must be loaded before making predictions")
        
        # Prepare features
        features_df = self._prepare_features(data)
        
        # Make predictions
        predictions = self.model.predict(features_df)
        probabilities = self.model.predict_proba(features_df)
        
        # Format results
        results = []
        is_single_prediction = len(data) == 1
        
        # Ensure predictions and probabilities are lists
        if not isinstance(predictions, (list, np.ndarray)):
            predictions = [predictions]
        if not isinstance(probabilities, (list, np.ndarray)) or (isinstance(probabilities, np.ndarray) and probabilities.ndim == 1):
            probabilities = [probabilities]
            
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            confidence = np.max(prob) if isinstance(prob, (list, np.ndarray)) else prob
            
            # Apply confidence threshold
            if confidence < self.confidence_threshold:
                pred = "NONE"
            
            result = {
                'prediction': pred,
                'confidence': float(confidence),
                'ticker': self.ticker
            }
            
            # Add timestamp if available and safe to access
            if 'datetime' in data.columns and i < len(data):
                result['timestamp'] = data.iloc[i]['datetime']
            
            if return_probabilities and hasattr(self.model, 'classes_'):
                class_labels = self.model.classes_
                result['probabilities'] = dict(zip(class_labels, prob))
            
            results.append(result)
        
        # Return single result if only one row was provided
        if is_single_prediction:
            return results[0]
        
        # Return single result if only one row
        if len(results) == 1:
            return results[0]
        
        return results
    
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for prediction."""
        # Enrich with technical indicators
        enriched_data = self.feature_engineer.enrich_dataframe(data)
        
        # Select feature columns
        if self.feature_columns:
            # Use columns from model metadata
            available_features = [col for col in self.feature_columns if col in enriched_data.columns]
            if len(available_features) != len(self.feature_columns):
                missing = set(self.feature_columns) - set(available_features)
                logger.warning(f"Missing features: {missing}")
            features_df = enriched_data[available_features]
        else:
            # Default feature selection
            exclude_cols = {"datetime", "ticker", "signal"}
            feature_cols = [col for col in enriched_data.columns if col not in exclude_cols]
            features_df = enriched_data[feature_cols]
        
        # Handle missing values
        features_df = features_df.fillna(method='ffill').fillna(method='bfill')
        
        return features_df
    
    def predict_latest(self) -> Dict:
        """Make prediction on the latest available data.
        
        Returns:
            Dictionary with prediction results
        """
        # Load latest data
        latest_data = self._load_latest_data()
        
        if latest_data.empty:
            raise ValueError(f"No data available for {self.ticker}")
        
        # Make prediction
        result = self.predict(latest_data.tail(1))
        
        logger.info(f"📈 Latest prediction for {self.ticker}: {result['prediction']} (confidence: {result['confidence']:.3f})")
        
        return result
    
    def _load_latest_data(self) -> pd.DataFrame:
        """Load the latest available data for the ticker."""
        data_path = get_data_path(self.ticker)
        
        # Find the latest date directory
        date_dirs = [d for d in data_path.iterdir() if d.is_dir()]
        if not date_dirs:
            return pd.DataFrame()
        
        latest_date_dir = sorted(date_dirs)[-1]
        
        # Load data from latest date
        data_file = latest_date_dir / "data.csv"
        if not data_file.exists():
            return pd.DataFrame()
        
        data = pd.read_csv(data_file, parse_dates=['datetime'])
        data['ticker'] = self.ticker
        
        return data
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            raise ValueError("No model loaded")
        
        info = {
            'ticker': self.ticker,
            'model_type': type(self.model).__name__,
            'confidence_threshold': self.confidence_threshold,
            'feature_count': len(self.feature_columns),
            'classes': self.model.classes_.tolist() if hasattr(self.model, 'classes_') else None
        }
        
        if self.model_metadata:
            info.update({
                'training_timestamp': self.model_metadata.get('training_timestamp'),
                'model_version': self.model_metadata.get('model_type'),
                'feature_columns': self.feature_columns
            })
        
        return info


class BatchInference:
    """Batch inference for multiple tickers."""
    
    def __init__(self, tickers: List[str], confidence_threshold: float = 0.6):
        """Initialize batch inference.
        
        Args:
            tickers: List of ticker symbols
            confidence_threshold: Minimum confidence for signal generation
        """
        self.tickers = [ticker.upper() for ticker in tickers]
        self.confidence_threshold = confidence_threshold
        self.inference_engines = {}
        
        # Initialize inference engines for each ticker
        for ticker in self.tickers:
            try:
                engine = ModelInference(ticker, confidence_threshold)
                engine.load_model()
                self.inference_engines[ticker] = engine
                logger.info(f"✅ Loaded model for {ticker}")
            except Exception as e:
                logger.warning(f"❌ Failed to load model for {ticker}: {e}")
    
    def predict_all(self) -> Dict[str, Dict]:
        """Make predictions for all tickers.
        
        Returns:
            Dictionary mapping ticker to prediction results
        """
        results = {}
        
        for ticker, engine in self.inference_engines.items():
            try:
                prediction = engine.predict_latest()
                results[ticker] = prediction
            except Exception as e:
                logger.error(f"❌ Failed to predict for {ticker}: {e}")
                results[ticker] = {
                    'prediction': 'ERROR',
                    'confidence': 0.0,
                    'error': str(e)
                }
        
        return results
    
    def get_high_confidence_signals(self, min_confidence: float = 0.8) -> Dict[str, Dict]:
        """Get only high-confidence signals.
        
        Args:
            min_confidence: Minimum confidence threshold
            
        Returns:
            Dictionary with high-confidence predictions
        """
        all_predictions = self.predict_all()
        
        high_confidence = {}
        for ticker, pred in all_predictions.items():
            if pred.get('confidence', 0) >= min_confidence and pred.get('prediction') != 'NONE':
                high_confidence[ticker] = pred
        
        return high_confidence
    
    def get_buy_signals(self, min_confidence: float = 0.7) -> List[Dict]:
        """Get BUY signals with minimum confidence.
        
        Args:
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of BUY signal dictionaries
        """
        all_predictions = self.predict_all()
        
        buy_signals = []
        for ticker, pred in all_predictions.items():
            if (pred.get('prediction') == 'BUY' and 
                pred.get('confidence', 0) >= min_confidence):
                buy_signals.append(pred)
        
        return buy_signals
    
    def get_sell_signals(self, min_confidence: float = 0.7) -> List[Dict]:
        """Get SELL signals with minimum confidence.
        
        Args:
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of SELL signal dictionaries
        """
        all_predictions = self.predict_all()
        
        sell_signals = []
        for ticker, pred in all_predictions.items():
            if (pred.get('prediction') == 'SELL' and 
                pred.get('confidence', 0) >= min_confidence):
                sell_signals.append(pred)
        
        return sell_signals
    
    def create_signals_summary(self) -> pd.DataFrame:
        """Create a summary DataFrame of all signals.
        
        Returns:
            DataFrame with signal summary
        """
        predictions = self.predict_all()
        
        summary_data = []
        for ticker, pred in predictions.items():
            summary_data.append({
                'ticker': ticker,
                'prediction': pred.get('prediction', 'ERROR'),
                'confidence': pred.get('confidence', 0.0),
                'timestamp': pred.get('timestamp', datetime.now()),
                'error': pred.get('error', None)
            })
        
        return pd.DataFrame(summary_data)


def main():
    """Example usage of inference modules."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Inference Tool")
    parser.add_argument("--ticker", type=str, help="Single ticker for inference")
    parser.add_argument("--tickers", nargs='+', help="Multiple tickers for batch inference")
    parser.add_argument("--confidence", type=float, default=0.6, help="Confidence threshold")
    parser.add_argument("--min-confidence", type=float, default=0.7, help="Minimum confidence for reporting")
    
    args = parser.parse_args()
    
    if args.ticker:
        # Single ticker inference
        inference = ModelInference(args.ticker, args.confidence)
        inference.load_model()
        
        try:
            prediction = inference.predict_latest()
            print(f"\n📈 Prediction for {args.ticker}:")
            print(f"   Signal: {prediction['prediction']}")
            print(f"   Confidence: {prediction['confidence']:.3f}")
            print(f"   Timestamp: {prediction['timestamp']}")
            
            # Model info
            model_info = inference.get_model_info()
            print(f"\n🤖 Model Info:")
            print(f"   Type: {model_info['model_type']}")
            print(f"   Features: {model_info['feature_count']}")
            print(f"   Classes: {model_info['classes']}")
            
        except Exception as e:
            print(f"❌ Error making prediction: {e}")
    
    elif args.tickers:
        # Batch inference
        batch_inference = BatchInference(args.tickers, args.confidence)
        
        # Get all predictions
        all_predictions = batch_inference.predict_all()
        print(f"\n📊 Batch Predictions:")
        for ticker, pred in all_predictions.items():
            print(f"   {ticker}: {pred['prediction']} (confidence: {pred.get('confidence', 0):.3f})")
        
        # Get high confidence signals
        high_conf_signals = batch_inference.get_high_confidence_signals(args.min_confidence)
        if high_conf_signals:
            print(f"\n🎯 High Confidence Signals (>{args.min_confidence}):")
            for ticker, pred in high_conf_signals.items():
                print(f"   {ticker}: {pred['prediction']} (confidence: {pred['confidence']:.3f})")
        
        # Get buy signals
        buy_signals = batch_inference.get_buy_signals(args.min_confidence)
        if buy_signals:
            print(f"\n📈 BUY Signals:")
            for signal in buy_signals:
                print(f"   {signal['ticker']}: confidence {signal['confidence']:.3f}")
        
        # Get sell signals
        sell_signals = batch_inference.get_sell_signals(args.min_confidence)
        if sell_signals:
            print(f"\n📉 SELL Signals:")
            for signal in sell_signals:
                print(f"   {signal['ticker']}: confidence {signal['confidence']:.3f}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
