"""
Test Suite for Model Training Pipeline
=====================================

Comprehensive tests for the model training pipeline including:
1. Model training functionality
2. Model evaluation and backtesting
3. Model inference and prediction
4. Integration with existing data pipeline

Usage:
    python test_model_pipeline.py
    python -m pytest test_model_pipeline.py -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch
import sys
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from train_model import ModelTrainer
from model_evaluator import ModelEvaluator
from model_inference import ModelInference, BatchInference
from feature_engineer import FeatureEngineer


class TestModelTrainer:
    """Test cases for ModelTrainer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        data = []
        for date in dates:
            base_price = 100 + np.random.randn() * 10
            high = base_price + np.random.rand() * 5
            low = base_price - np.random.rand() * 5
            close = base_price + np.random.randn() * 2
            volume = np.random.randint(100000, 1000000)
            
            data.append({
                'datetime': date,
                'ticker': 'TEST',
                'Open': base_price,
                'High': high,
                'Low': low,
                'Close': close,
                'Volume': volume
            })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_signals(self):
        """Create sample signals data for testing."""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        signals = []
        for date in dates:
            signal = np.random.choice(['BUY', 'SELL', 'NONE'], p=[0.3, 0.3, 0.4])
            signals.append({
                'datetime': date,
                'ticker': 'TEST',
                'signal': signal
            })
        
        return pd.DataFrame(signals)
    
    @pytest.fixture
    def temp_data_dir(self, sample_data, sample_signals):
        """Create temporary data directory structure."""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create data structure
        data_dir = temp_dir / 'data' / 'TEST'
        data_dir.mkdir(parents=True)
        
        # Create date directories and files
        for month in range(1, 13):
            date_str = f'2023{month:02d}01'
            date_dir = data_dir / date_str
            date_dir.mkdir()
            
            # Filter data for this month
            month_data = sample_data[sample_data['datetime'].dt.month == month]
            month_signals = sample_signals[sample_signals['datetime'].dt.month == month]
            
            # Save data files
            month_data.to_csv(date_dir / 'data.csv', index=False)
            month_signals.to_csv(date_dir / 'signals.csv', index=False)
        
        # Create models directory
        models_dir = temp_dir / 'models'
        models_dir.mkdir()
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_model_trainer_initialization(self):
        """Test ModelTrainer initialization."""
        trainer = ModelTrainer('TEST', 'randomforest')
        
        assert trainer.ticker == 'TEST'
        assert trainer.model_type == 'randomforest'
        assert trainer.model is None
        assert trainer.pipeline is None
        assert isinstance(trainer.feature_engineer, FeatureEngineer)
    
    @patch('train_model.get_data_path')
    @patch('train_model.get_models_path')
    def test_load_training_data(self, mock_models_path, mock_data_path, temp_data_dir):
        """Test loading training data."""
        mock_data_path.return_value = temp_data_dir / 'data' / 'TEST'
        mock_models_path.return_value = temp_data_dir / 'models' / 'TEST'
        
        trainer = ModelTrainer('TEST', 'randomforest')
        df = trainer.load_training_data()
        
        assert not df.empty
        assert 'datetime' in df.columns
        assert 'ticker' in df.columns
        assert 'signal' in df.columns
        assert all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])
    
    def test_prepare_features(self, sample_data, sample_signals):
        """Test feature preparation."""
        # Merge sample data
        merged_data = pd.merge(sample_data, sample_signals, on=['datetime', 'ticker'])
        
        trainer = ModelTrainer('TEST', 'randomforest')
        X, y = trainer.prepare_features(merged_data)
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)
        assert 'signal' not in X.columns
        assert 'datetime' not in X.columns
        assert 'ticker' not in X.columns
    
    def test_create_model(self):
        """Test model creation."""
        trainer = ModelTrainer('TEST', 'randomforest')
        pipeline = trainer.create_model()
        
        assert pipeline is not None
        assert 'scaler' in pipeline.named_steps
        assert 'model' in pipeline.named_steps
    
    def test_invalid_model_type(self):
        """Test invalid model type raises error."""
        trainer = ModelTrainer('TEST', 'invalid_model')
        
        with pytest.raises(ValueError, match="Unsupported model type"):
            trainer.create_model()


class TestModelEvaluator:
    """Test cases for ModelEvaluator class."""
    
    @pytest.fixture
    def sample_test_data(self):
        """Create sample test data."""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        data = []
        for date in dates:
            base_price = 100 + np.random.randn() * 10
            data.append({
                'datetime': date,
                'ticker': 'TEST',
                'Open': base_price,
                'High': base_price + np.random.rand() * 2,
                'Low': base_price - np.random.rand() * 2,
                'Close': base_price + np.random.randn(),
                'Volume': np.random.randint(100000, 1000000),
                'signal': np.random.choice(['BUY', 'SELL', 'NONE'])
            })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def mock_model(self):
        """Create mock trained model."""
        model = Mock()
        model.predict.return_value = np.random.choice(['BUY', 'SELL', 'NONE'], size=100)
        model.predict_proba.return_value = np.random.rand(100, 3)
        model.classes_ = np.array(['BUY', 'NONE', 'SELL'])
        return model
    
    def test_evaluator_initialization(self):
        """Test ModelEvaluator initialization."""
        evaluator = ModelEvaluator('TEST')
        
        assert evaluator.ticker == 'TEST'
        assert evaluator.model is None
        assert evaluator.model_metadata is None
    
    def test_calculate_metrics(self, sample_test_data, mock_model):
        """Test metrics calculation."""
        evaluator = ModelEvaluator('TEST')
        evaluator.model = mock_model
        
        y_true = sample_test_data['signal']
        y_pred = mock_model.predict(sample_test_data)
        y_proba = mock_model.predict_proba(sample_test_data)
        
        metrics = evaluator._calculate_metrics(y_true, y_pred, y_proba)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'classification_report' in metrics
        assert 'confusion_matrix' in metrics
    
    def test_analyze_confidence(self):
        """Test confidence analysis."""
        evaluator = ModelEvaluator('TEST')
        
        # Mock probability matrix
        y_proba = np.random.rand(100, 3)
        y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)  # Normalize
        
        confidence_analysis = evaluator._analyze_confidence(y_proba)
        
        assert 'statistics' in confidence_analysis
        assert 'histogram' in confidence_analysis
        assert 'bin_edges' in confidence_analysis
        assert 'mean' in confidence_analysis['statistics']
        assert 'std' in confidence_analysis['statistics']


class TestModelInference:
    """Test cases for ModelInference class."""
    
    @pytest.fixture
    def sample_inference_data(self):
        """Create sample data for inference."""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
        
        data = []
        for date in dates:
            base_price = 100 + np.random.randn() * 10
            data.append({
                'datetime': date,
                'ticker': 'TEST',
                'Open': base_price,
                'High': base_price + np.random.rand() * 2,
                'Low': base_price - np.random.rand() * 2,
                'Close': base_price + np.random.randn(),
                'Volume': np.random.randint(100000, 1000000)
            })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def mock_inference_model(self):
        """Create mock model for inference."""
        model = Mock()
        model.predict.return_value = np.array(['BUY', 'SELL', 'NONE'] * 4)[:10]
        model.predict_proba.return_value = np.random.rand(10, 3)
        model.classes_ = np.array(['BUY', 'NONE', 'SELL'])
        return model
    
    def test_inference_initialization(self):
        """Test ModelInference initialization."""
        inference = ModelInference('TEST')
        
        assert inference.ticker == 'TEST'
        assert inference.confidence_threshold == 0.6
        assert inference.model is None
        assert isinstance(inference.feature_engineer, FeatureEngineer)
    
    def test_predict_single_row(self, sample_inference_data, mock_inference_model):
        """Test prediction on single row."""
        inference = ModelInference('TEST')
        inference.model = mock_inference_model
        
        # Create a proper single row with all required columns
        single_row = sample_inference_data.iloc[[0]].copy()
        
        # Make sure we have all required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in single_row.columns:
                single_row[col] = 100.0  # Default value for missing columns
        
        # Test single row prediction
        result = inference.predict(single_row)
        
        # Verify the result structure
        assert isinstance(result, dict)
        assert 'prediction' in result
        assert 'confidence' in result
        assert 'ticker' in result
        assert 'prediction' in result
        assert 'confidence' in result
        assert 'ticker' in result
        assert result['ticker'] == 'TEST'
    
    def test_predict_multiple_rows(self, sample_inference_data, mock_inference_model):
        """Test prediction on multiple rows."""
        inference = ModelInference('TEST')
        inference.model = mock_inference_model
        
        results = inference.predict(sample_inference_data)
        
        assert isinstance(results, list)
        assert len(results) == len(sample_inference_data)
        assert all('prediction' in result for result in results)
        assert all('confidence' in result for result in results)
    
    def test_confidence_threshold_filtering(self, sample_inference_data, mock_inference_model):
        """Test confidence threshold filtering."""
        # Set up mock with low confidence
        mock_inference_model.predict_proba.return_value = np.full((10, 3), 0.3)
        
        inference = ModelInference('TEST', confidence_threshold=0.8)
        inference.model = mock_inference_model
        
        results = inference.predict(sample_inference_data)
        
        # All predictions should be 'NONE' due to low confidence
        if isinstance(results, list):
            assert all(result['prediction'] == 'NONE' for result in results)
        else:
            assert results['prediction'] == 'NONE'


class TestBatchInference:
    """Test cases for BatchInference class."""
    
    def test_batch_inference_initialization(self):
        """Test BatchInference initialization."""
        tickers = ['AAPL', 'GOOGL', 'MSFT']
        batch_inference = BatchInference(tickers)
        
        assert batch_inference.tickers == tickers
        assert batch_inference.confidence_threshold == 0.6
        assert isinstance(batch_inference.inference_engines, dict)
    
    def test_get_high_confidence_signals(self):
        """Test filtering high confidence signals."""
        batch_inference = BatchInference(['TEST'])
        
        # Mock predictions
        mock_predictions = {
            'TEST': {
                'prediction': 'BUY',
                'confidence': 0.9,
                'ticker': 'TEST'
            }
        }
        
        with patch.object(batch_inference, 'predict_all', return_value=mock_predictions):
            high_conf = batch_inference.get_high_confidence_signals(min_confidence=0.8)
            
            assert 'TEST' in high_conf
            assert high_conf['TEST']['prediction'] == 'BUY'
    
    def test_get_buy_signals(self):
        """Test filtering BUY signals."""
        batch_inference = BatchInference(['TEST'])
        
        mock_predictions = {
            'TEST': {
                'prediction': 'BUY',
                'confidence': 0.8,
                'ticker': 'TEST'
            }
        }
        
        with patch.object(batch_inference, 'predict_all', return_value=mock_predictions):
            buy_signals = batch_inference.get_buy_signals(min_confidence=0.7)
            
            assert len(buy_signals) == 1
            assert buy_signals[0]['prediction'] == 'BUY'
    
    def test_create_signals_summary(self):
        """Test creating signals summary DataFrame."""
        batch_inference = BatchInference(['TEST'])
        
        mock_predictions = {
            'TEST': {
                'prediction': 'BUY',
                'confidence': 0.8,
                'ticker': 'TEST',
                'timestamp': datetime.now()
            }
        }
        
        with patch.object(batch_inference, 'predict_all', return_value=mock_predictions):
            summary = batch_inference.create_signals_summary()
            
            assert isinstance(summary, pd.DataFrame)
            assert 'ticker' in summary.columns
            assert 'prediction' in summary.columns
            assert 'confidence' in summary.columns
            assert len(summary) == 1


class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    @pytest.fixture
    def temp_data_dir(self, tmp_path):
        """Create temporary data directory structure for testing."""
        # Create directory structure
        data_dir = tmp_path / "data" / "TEST"
        data_dir.mkdir(parents=True)
        
        # Create a sample data file
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        data = []
        for i, date in enumerate(dates):
            base_price = 100 + i  # Trend up
            data.append({
                'datetime': date,
                'ticker': 'TEST',
                'Open': base_price + np.random.randn() * 2,
                'High': base_price + np.random.rand() * 3,
                'Low': base_price - np.random.rand() * 3,
                'Close': base_price + np.random.randn() * 2,
                'Volume': np.random.randint(100000, 1000000)
            })
        
        df = pd.DataFrame(data)
        df.to_csv(data_dir / "data-20230101.csv", index=False)
        
        # Create models directory
        models_dir = tmp_path / "models" / "TEST"
        models_dir.mkdir(parents=True)
        
        # Save a dummy model
        dummy_model_path = models_dir / "model_20230101_120000.pkl"
        with open(dummy_model_path, 'wb') as f:
            import pickle
            pickle.dump("dummy_model", f)
        
        # Set environment variables for the test
        os.environ['DATA_DIR'] = str(data_dir.parent)
        os.environ['MODELS_DIR'] = str(models_dir.parent)
        
        return {
            'data_dir': data_dir,
            'models_dir': models_dir,
            'dummy_model': dummy_model_path
        }

    def test_end_to_end_workflow(self, temp_data_dir):
        """Test complete end-to-end workflow."""
        # This would be a comprehensive integration test
        # For now, just test that components can be instantiated together
        
        feature_engineer = FeatureEngineer()
        trainer = ModelTrainer('TEST', 'randomforest')
        evaluator = ModelEvaluator('TEST')
        inference = ModelInference('TEST')
        
        # Check that all components are properly initialized
        assert feature_engineer is not None
        assert trainer is not None
        assert evaluator is not None
        assert inference is not None
    
    def test_feature_pipeline_integration(self):
        """Test integration with feature engineering pipeline."""
        # Create sample data
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        
        data = []
        for date in dates:
            base_price = 100 + np.random.randn() * 10
            data.append({
                'datetime': date,
                'ticker': 'TEST',
                'Open': base_price,
                'High': base_price + np.random.rand() * 2,
                'Low': base_price - np.random.rand() * 2,
                'Close': base_price + np.random.randn(),
                'Volume': np.random.randint(100000, 1000000)
            })
        
        df = pd.DataFrame(data)
        
        # Test feature engineering integration
        feature_engineer = FeatureEngineer()
        enriched_df = feature_engineer.enrich_dataframe(df)
        
        # Check that technical indicators are added
        assert 'RSI_14' in enriched_df.columns
        assert 'MACD' in enriched_df.columns
        assert 'EMA_5' in enriched_df.columns
        
        # Test that ModelTrainer can use enriched data
        trainer = ModelTrainer('TEST', 'randomforest')
        
        # Add mock signals
        enriched_df['signal'] = np.random.choice(['BUY', 'SELL', 'NONE'], len(enriched_df))
        
        X, y = trainer.prepare_features(enriched_df)
        
        assert not X.empty
        assert not y.empty
        assert len(X) == len(y)


def run_tests():
    """Run all tests."""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_tests()
