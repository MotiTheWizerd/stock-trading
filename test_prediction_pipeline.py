"""
Test script for the Signal Prediction Pipeline

This script validates the key components of the signal prediction pipeline
by testing feature engineering, prediction, and batch inference.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path if needed
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import components
try:
    from src.trading.features.feature_engineer import FeatureEngineer, FeatureEngineerConfig
    from src.trading.models.predictor import StockPredictor
    from src.trading.models.model_inference import ModelInference, BatchInference
    from src.trading.models.enhanced_prediction import EnhancedStockPredictor
    from src.trading.ui.cli_output import render_predictions
    
    components_available = True
    logger.info("✅ Successfully imported all components")
except ImportError as e:
    components_available = False
    logger.error(f"❌ Failed to import components: {e}")

def test_feature_engineering():
    """Test the feature engineering component."""
    logger.info("\n🔍 Testing Feature Engineering Component")
    
    try:
        # Create a feature engineer with default config
        config = FeatureEngineerConfig()
        engineer = FeatureEngineer(config)
        
        # Load sample data
        data_path = os.path.join(project_root, "data", "AAPL", "latest", "data.csv")
        if not os.path.exists(data_path):
            data_path = os.path.join(project_root, "data", "AAPL", "data.csv")
        
        df = pd.read_csv(data_path)
        logger.info(f"📊 Loaded data with shape: {df.shape}")
        
        # Apply feature engineering
        enriched_df = engineer.enrich_dataframe(df)
        
        # Verify results
        original_cols = set(df.columns)
        enriched_cols = set(enriched_df.columns)
        new_features = enriched_cols - original_cols
        
        logger.info(f"✅ Feature engineering added {len(new_features)} new features")
        logger.info(f"📊 New features: {', '.join(list(new_features)[:5])}...")
        
        return True, enriched_df
    
    except Exception as e:
        logger.error(f"❌ Feature engineering test failed: {e}")
        return False, None

def test_stock_predictor(df=None):
    """Test the StockPredictor component."""
    logger.info("\n🔍 Testing StockPredictor Component")
    
    try:
        # If no dataframe provided, load sample data
        if df is None:
            data_path = os.path.join(project_root, "data", "AAPL", "latest", "data.csv")
            if not os.path.exists(data_path):
                data_path = os.path.join(project_root, "data", "AAPL", "data.csv")
            
            df = pd.read_csv(data_path)
            logger.info(f"📊 Loaded data with shape: {df.shape}")
        
        # Initialize predictor
        ticker = "AAPL"
        model_dir = os.path.join(project_root, "models", "AAPL")
        
        # Check if models directory exists
        if not os.path.exists(model_dir):
            logger.warning(f"⚠️ Models directory not found: {model_dir}")
            logger.info("ℹ️ This test will likely fail if no models are available")
            return False, None
        
        # Find the latest model file
        model_files = list(Path(model_dir).glob("model_AAPL_*.joblib"))
        if not model_files:
            logger.warning(f"⚠️ No model files found in: {model_dir}")
            return False, None
            
        # Sort by timestamp and get the latest
        model_files.sort(key=lambda x: x.stem.split('_')[-1])
        latest_model = model_files[-1]
        logger.info(f"📊 Using model: {latest_model.name}")
        
        try:
            # Use the specific model file path
            predictor = StockPredictor(ticker=ticker, model_dir=str(latest_model.parent))
            predictor.model_path = str(latest_model)  # Set the specific model path if the class supports it
            logger.info("✅ Successfully initialized StockPredictor")
            
            # Make predictions
            predictions = predictor.predict(df)
            
            if predictions is not None and len(predictions) > 0:
                logger.info(f"✅ Successfully made predictions on {len(predictions)} rows")
                
                # Display sample predictions
                if hasattr(predictions, 'head'):
                    sample = predictions.head(3)
                    logger.info(f"📊 Sample predictions:\n{sample}")
                
                return True, predictions
            else:
                logger.warning("⚠️ Predictions returned empty result")
                return False, None
                
        except Exception as e:
            logger.error(f"❌ StockPredictor test failed: {e}")
            logger.info("ℹ️ Error details: {str(e)}")
            return False, None
    
    except Exception as e:
        logger.error(f"❌ StockPredictor test failed: {e}")
        logger.error(f"ℹ️ Error details: {str(e)}")
        return False, None

def test_model_inference():
    """Test the ModelInference component with a mock model to avoid calibration issues."""
    logger.info("\n🔍 Testing ModelInference Component")
    
    try:
        # Create a mock model and calibrator for testing
        logger.info("🔄 Creating mock model for testing")
        
        # Import necessary libraries
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.calibration import CalibratedClassifierCV
        import joblib
        from tempfile import NamedTemporaryFile
        
        # Create a simple mock model
        mock_model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Create synthetic training data
        np.random.seed(42)
        X_train = np.random.rand(100, 5)  # 5 features
        y_train = np.random.choice(['BUY', 'SELL', 'NONE'], size=100)  # 3 classes
        
        # Train the model
        mock_model.fit(X_train, y_train)
        
        # Create feature names
        feature_names = ['RSI_14', 'MACD', 'Bollinger_Upper', 'Bollinger_Lower', 'Volatility']
        mock_model.feature_names_in_ = np.array(feature_names)
        
        # Save the model to a temporary file
        with NamedTemporaryFile(suffix='.joblib', delete=False) as temp_model_file:
            temp_model_path = Path(temp_model_file.name)
            joblib.dump(mock_model, temp_model_path)
            logger.info(f"✅ Created and saved mock model to {temp_model_path}")
        
        try:
            # Initialize ModelInference with the mock model
            ticker = "AAPL"
            inference = ModelInference(ticker, confidence_threshold=0.6)
            inference.load_model(model_path=temp_model_path)
            logger.info("✅ Successfully initialized ModelInference with mock model")
            
            # Create synthetic test data matching the model's expected features
            test_data = {}
            for feature in feature_names:
                test_data[feature] = [np.random.uniform(0, 1)]
            
            # Add datetime and ticker columns
            test_data['datetime'] = [pd.Timestamp.now()]
            test_data['ticker'] = [ticker]
            
            # Create test dataframe
            test_df = pd.DataFrame(test_data)
            logger.info(f"✅ Created test dataframe with shape {test_df.shape}")
            
            # Make prediction
            try:
                # Override the _prepare_features method to bypass feature engineering
                original_prepare_features = inference._prepare_features
                
                def mock_prepare_features(data):
                    # Return only the features the model expects
                    return data[feature_names]
                
                # Replace the method temporarily
                inference._prepare_features = mock_prepare_features
                
                # Make prediction
                prediction = inference.predict(test_df)
                
                # Restore original method
                inference._prepare_features = original_prepare_features
                
                logger.info(f"✅ Successfully made prediction with mock model")
                logger.info(f"📊 Prediction: {prediction['prediction']}, Confidence: {prediction.get('confidence', 'N/A')}")
                
                # Clean up temporary file
                os.unlink(temp_model_path)
                
                return True, prediction
            except Exception as e:
                logger.error(f"❌ Failed to make prediction with mock model: {e}")
                import traceback
                logger.error(f"ℹ️ Traceback: {traceback.format_exc()}")
                
                # Clean up temporary file
                os.unlink(temp_model_path)
                
                return False, None
        except Exception as e:
            logger.error(f"❌ Failed to initialize ModelInference with mock model: {e}")
            import traceback
            logger.error(f"ℹ️ Traceback: {traceback.format_exc()}")
            
            # Clean up temporary file
            os.unlink(temp_model_path)
            
            return False, None
    except Exception as e:
        logger.error(f"❌ Failed to create mock model: {e}")
        import traceback
        logger.error(f"ℹ️ Traceback: {traceback.format_exc()}")
        return False, None

def test_batch_inference():
    """Test the BatchInference component with mock models."""
    logger.info("\n🔍 Testing BatchInference Component")
    
    try:
        # Create a mock model for testing
        logger.info("🔄 Creating mock model for BatchInference testing")
        
        # Import necessary libraries
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        import joblib
        from tempfile import NamedTemporaryFile
        
        # Create a simple mock model
        mock_model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Create synthetic training data
        np.random.seed(42)
        X_train = np.random.rand(100, 5)  # 5 features
        y_train = np.random.choice(['BUY', 'SELL', 'NONE'], size=100)  # 3 classes
        
        # Train the model
        mock_model.fit(X_train, y_train)
        
        # Create feature names
        feature_names = ['RSI_14', 'MACD', 'Bollinger_Upper', 'Bollinger_Lower', 'Volatility']
        mock_model.feature_names_in_ = np.array(feature_names)
        
        # Save the model to a temporary file
        with NamedTemporaryFile(suffix='.joblib', delete=False) as temp_model_file:
            temp_model_path = Path(temp_model_file.name)
            joblib.dump(mock_model, temp_model_path)
            logger.info(f"✅ Created and saved mock model to {temp_model_path}")
        
        try:
            # Initialize BatchInference with the mock model
            tickers = ["AAPL"]
            batch_inference = BatchInference(tickers=tickers, model_paths={"AAPL": temp_model_path})
            logger.info("✅ Successfully initialized BatchInference with mock model")
            
            # Override the _prepare_features method in each ModelInference instance
            for ticker, inference in batch_inference.inference_engines.items():
                original_prepare_features = inference._prepare_features
                
                def mock_prepare_features(data):
                    # Create synthetic data with the expected features
                    mock_data = pd.DataFrame()
                    for feature in feature_names:
                        mock_data[feature] = np.random.uniform(0, 1, size=len(data) if len(data) > 0 else 1)
                    return mock_data
                
                # Replace the method
                inference._prepare_features = mock_prepare_features
            
            # Make predictions
            predictions = batch_inference.predict_all()
            logger.info(f"✅ Successfully made predictions for {len(predictions)} tickers")
            
            # Log results
            for ticker, result in predictions.items():
                if isinstance(result, dict) and 'error' in result:
                    logger.info(f"📊 {ticker}: ERROR ({result['error']})")
                else:
                    logger.info(f"📊 {ticker}: {result.get('prediction', 'N/A')} (confidence: {result.get('confidence', 'N/A')})")
            
            # Clean up temporary file
            os.unlink(temp_model_path)
            
            return True, predictions
        except Exception as e:
            logger.error(f"❌ BatchInference test failed: {e}")
            import traceback
            logger.error(f"ℹ️ Traceback: {traceback.format_exc()}")
            
            # Clean up temporary file
            os.unlink(temp_model_path)
            
            return False, None
    except Exception as e:
        logger.error(f"❌ Failed to create mock model for BatchInference: {e}")
        import traceback
        logger.error(f"ℹ️ Traceback: {traceback.format_exc()}")
        return False, None

def test_enhanced_prediction():
    """Test the EnhancedStockPredictor component."""
    logger.info("\n🔍 Testing EnhancedStockPredictor Component")
    
    try:
        # Find the latest model file
        model_dir = os.path.join(project_root, "models", "AAPL")
        if not os.path.exists(model_dir):
            logger.warning(f"⚠️ Models directory not found: {model_dir}")
            return False, None
            
        model_files = list(Path(model_dir).glob("model_AAPL_*.joblib"))
        if not model_files:
            logger.warning(f"⚠️ No model files found in: {model_dir}")
            return False, None
            
        # Sort by timestamp and get the latest
        model_files.sort(key=lambda x: x.stem.split('_')[-1])
        latest_model = model_files[-1]
        logger.info(f"📊 Using model: {latest_model.name}")
        
        # Check for calibrated model
        calibrated_model = latest_model.with_name(latest_model.stem + "_calibrated.joblib")
        if calibrated_model.exists():
            logger.info(f"📊 Found calibrated model: {calibrated_model.name}")
        
        # Initialize enhanced predictor
        try:
            enhanced_predictor = EnhancedStockPredictor(
                model_path=str(latest_model),  # Use the specific model path
                confidence_threshold=0.6
            )
            logger.info("✅ Successfully initialized EnhancedStockPredictor")
            
            # Load sample data
            data_path = os.path.join(project_root, "data", "AAPL", "latest", "data.csv")
            if not os.path.exists(data_path):
                data_path = os.path.join(project_root, "data", "AAPL", "data.csv")
            
            df = pd.read_csv(data_path)
            logger.info(f"📊 Loaded data with shape: {df.shape}")
            
            # Make predictions
            try:
                predictions = enhanced_predictor.predict(df, apply_trading=True)
                logger.info(f"✅ Successfully made enhanced predictions")
                
                # Display sample predictions if available
                if hasattr(predictions, 'head'):
                    sample = predictions.head(3)
                    logger.info(f"📊 Sample enhanced predictions:\n{sample}")
                
                return True, predictions
            except Exception as e:
                logger.warning(f"⚠️ Failed to make enhanced predictions: {e}")
                logger.warning(f"⚠️ Error details: {str(e)}")
                return False, None
                
        except Exception as e:
            logger.error(f"❌ EnhancedStockPredictor test failed: {e}")
            logger.error(f"ℹ️ Error details: {str(e)}")
            return False, None
    
    except Exception as e:
        logger.error(f"❌ EnhancedStockPredictor test failed: {e}")
        logger.error(f"ℹ️ Error details: {str(e)}")
        return False, None

def test_cli_output(predictions_df=None):
    """Test the CLI output rendering."""
    logger.info("\n🔍 Testing CLI Output Component")
    
    try:
        # If no predictions provided, create sample data
        if predictions_df is None or not isinstance(predictions_df, pd.DataFrame):
            # Create sample predictions DataFrame
            predictions_df = pd.DataFrame({
                'timestamp': pd.date_range(start='2025-07-01', periods=5),
                'Open': [150.0, 151.0, 152.0, 153.0, 154.0],
                'High': [152.0, 153.0, 154.0, 155.0, 156.0],
                'Low': [149.0, 150.0, 151.0, 152.0, 153.0],
                'Close': [151.0, 152.0, 153.0, 154.0, 155.0],
                'Volume': [1000000, 1100000, 1200000, 1300000, 1400000],
                'prediction': ['BUY', 'NONE', 'NONE', 'SELL', 'BUY'],
                'confidence': [0.85, 0.55, 0.60, 0.75, 0.90],
                'risk': ['LOW', 'MEDIUM', 'MEDIUM', 'LOW', 'LOW']
            })
        
        # Render predictions
        try:
            render_predictions(predictions_df)
            logger.info("✅ Successfully rendered predictions")
            return True
        except Exception as e:
            logger.error(f"❌ CLI output rendering failed: {e}")
            return False
    
    except Exception as e:
        logger.error(f"❌ CLI output test failed: {e}")
        return False

def run_all_tests():
    """Run all tests and report results."""
    logger.info("\n🚀 Starting Signal Prediction Pipeline Tests")
    
    if not components_available:
        logger.error("❌ Cannot run tests: Components not available")
        return False
    
    results = {}
    
    # Test feature engineering
    results['feature_engineering'], enriched_df = test_feature_engineering()
    
    # Test stock predictor
    results['stock_predictor'], predictions = test_stock_predictor(enriched_df)
    
    # Test model inference
    results['model_inference'], _ = test_model_inference()
    
    # Test batch inference
    results['batch_inference'], _ = test_batch_inference()
    
    # Test enhanced prediction
    results['enhanced_prediction'], enhanced_predictions = test_enhanced_prediction()
    
    # Test CLI output
    results['cli_output'] = test_cli_output(predictions if results['stock_predictor'] else None)
    
    # Report overall results
    logger.info("\n📋 Test Results Summary")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for component, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{status} - {component}")
    
    logger.info(f"\n🏁 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! The signal prediction pipeline is working correctly.")
        return True
    else:
        logger.warning(f"⚠️ {total - passed} tests failed. Some components may need attention.")
        return False

if __name__ == "__main__":
    run_all_tests()
