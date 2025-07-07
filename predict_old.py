"""
Prediction Script for Stock Trading Model

This script loads a trained model and makes predictions on new data.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class StockPredictor:
    """Class for making predictions with a trained stock trading model."""
    
    def __init__(self, ticker: str, model_dir: str = None):
        """Initialize with a ticker and optional model directory."""
        self.ticker = ticker
        self.model_dir = model_dir or f"models/{ticker}"
        self.model = None
        self.feature_columns = []
        self.model_classes = []
        self.load_model()
    
    def load_model(self):
        """Load the latest trained model for this ticker."""
        model_dir = Path(self.model_dir)
        model_files = list(model_dir.glob(f"model_{self.ticker}_*.joblib"))
        
        if not model_files:
            raise FileNotFoundError(f"No model files found for {self.ticker} in {model_dir}")
        
        # Find the latest model file
        latest_model = max(model_files, key=os.path.getmtime)
        model_timestamp = latest_model.stem.split('_')[-1]
        
        # Load the model
        logger.info(f"✅ Loaded model: {latest_model.name}")
        self.model = joblib.load(latest_model)
        
        # Store model classes
        if hasattr(self.model, 'classes_'):
            self.model_classes = self.model.classes_.tolist()
        elif hasattr(self.model, 'steps') and hasattr(self.model.steps[-1][1], 'classes_'):
            self.model_classes = self.model.steps[-1][1].classes_.tolist()
        
        logger.info(f"Model classes: {self.model_classes}")
        
        # Load metadata if available
        metadata_file = model_dir / f"metadata_{self.ticker}_{model_timestamp}.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                self.feature_columns = metadata.get('feature_columns', [])
                logger.info(f"Loaded feature columns from metadata: {len(self.feature_columns)} features")
        
        # If no feature columns from metadata, try to extract from model
        if not self.feature_columns:
            if hasattr(self.model, 'feature_names_in_'):
                self.feature_columns = self.model.feature_names_in_.tolist()
            elif hasattr(self.model, 'steps') and hasattr(self.model.steps[-1][1], 'feature_names_in_'):
                self.feature_columns = self.model.steps[-1][1].feature_names_in_.tolist()
            logger.info(f"Extracted feature columns from model: {len(self.feature_columns)} features")
        
        if self.feature_columns:
            logger.info(f"Feature columns (first 5): {self.feature_columns[:5]}...")
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions on the given data.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with predictions and confidence scores
        """
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Ensure datetime is properly parsed
        if 'datetime' in df_copy.columns and not pd.api.types.is_datetime64_any_dtype(df_copy['datetime']):
            try:
                df_copy['datetime'] = pd.to_datetime(df_copy['datetime'])
                logger.info("Converted datetime column to datetime type")
            except Exception as e:
                logger.warning(f"Failed to convert datetime column: {e}")
        
        # Check if we have the necessary feature columns
        logger.info(f"Making predictions on {len(df_copy)} rows with {len(df_copy.columns)} columns")
        
        # Check for missing feature columns
        missing_features = [col for col in self.feature_columns if col not in df_copy.columns]
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            
            # Try to apply feature engineering to get missing features
            logger.info("Applying feature engineering to enrich data...")
            try:
                from feature_engineer import FeatureEngineer
                engineer = FeatureEngineer()
                df_copy = engineer.enrich_dataframe(df_copy)
                logger.info(f"After feature engineering: {len(df_copy.columns)} columns")
            except Exception as e:
                logger.error(f"Feature engineering failed: {e}")
        
        # Check if we still have missing features after engineering
        missing_features = [col for col in self.feature_columns if col not in df_copy.columns]
        if missing_features:
            logger.warning(f"Still missing features after engineering: {missing_features}")
            
            # Add missing columns with zeros
            for col in missing_features:
                logger.info(f"Adding missing column with zeros: {col}")
                df_copy[col] = 0
        
        # Prepare feature matrix
        X = df_copy[self.feature_columns].copy()
        logger.info(f"Feature matrix shape: {X.shape}")
        
        # Make predictions
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        # Get confidence scores (probability of the predicted class)
        confidences = np.max(probabilities, axis=1)
        
        # Add predictions and confidence to the original dataframe
        # Use df_copy which has properly parsed datetime and all features
        result_df = df_copy.copy()
        result_df['prediction'] = predictions
        result_df['confidence'] = confidences
        
        # Add class probabilities
        for i, class_name in enumerate(self.model.classes_):
            result_df[f'prob_{class_name}'] = probabilities[:, i]
        
        # Log prediction distribution
        pred_counts = pd.Series(predictions).value_counts().to_dict()
        logger.info(f"Prediction distribution: {pred_counts}")
        
        # Check if all predictions are the same
        if len(pred_counts) == 1:
            only_pred = list(pred_counts.keys())[0]
            logger.warning(f"⚠️ All predictions are '{only_pred}'. This suggests an issue with the model training data.")
        
        return result_df

    def _handle_prediction_error(self, e):
        logger.error(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        logger.error(f"Prediction failed: {str(e)}")
        raise

def load_latest_data(ticker: str, data_dir: str = "data") -> pd.DataFrame:
    """Load the most recent data for a ticker."""
    ticker_dir = Path(data_dir) / ticker.upper()
    latest_dir = ticker_dir / "latest"
    
    if not latest_dir.exists():
        logger.warning(f"Latest directory not found: {latest_dir}")
        # Try to find any data file
        data_files = list(ticker_dir.glob("**/data.csv"))
        if not data_files:
            logger.error(f"No data files found for {ticker}")
            return pd.DataFrame()
        
        # Use the most recent data file
        data_file = sorted(data_files)[-1]
    else:
        data_file = latest_dir / "data.csv"
    
    if not data_file.exists():
        logger.error(f"Data file not found: {data_file}")
        return pd.DataFrame()
    
    logger.info(f"Loading from file: {data_file}")
    df = pd.read_csv(data_file)
    
    # Handle datetime column - convert to datetime if it exists
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Rename standard columns if needed
    column_mapping = {
        'Open': 'Open',
        'High': 'High', 
        'Low': 'Low',
        'Close': 'Close',
        'Volume': 'Volume'
    }
    
    # Apply mapping for columns that exist
    for old_name, new_name in column_mapping.items():
        if old_name.lower() in df.columns and old_name not in df.columns:
            df[old_name] = df[old_name.lower()]
    
    # Make sure we have a Price column (use Close if available)
    if 'Price' not in df.columns and 'Close' in df.columns:
        df['Price'] = df['Close']
    
    return df

def main():
    """Main function for the prediction script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Predict stock signals using trained model")
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker symbol")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--model-dir", type=str, default="models", help="Model directory")
    parser.add_argument("--sample-file", type=str, help="Sample data file to use instead of latest data")
    parser.add_argument("--retrain", action="store_true", help="Retrain model before prediction")
    parser.add_argument("--output", type=str, help="Output file for predictions (CSV)")
    args = parser.parse_args()
    
    # Configure logging
    setup_logging()
    
    # Retrain model if requested
    if args.retrain:
        logger.info(f"🔄 Retraining model for {args.ticker}...")
        try:
            from train_model import ModelTrainer
            trainer = ModelTrainer(args.ticker, data_dir=args.data_dir)
            trainer.run_training_pipeline()
            logger.info(f"✅ Model retrained successfully")
        except Exception as e:
            logger.error(f"❌ Model retraining failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Load model
    predictor = StockPredictor(args.ticker, model_dir=args.model_dir)
    
    # Load data
    logger.info(f"📊 Loading data for {args.ticker}")
    if args.sample_file:
        logger.info(f"Loading from sample file: {args.sample_file}")
        df = pd.read_csv(args.sample_file)
        
        # Process the data file
        # Convert datetime column if it exists
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Handle price columns
        price_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in price_columns:
            if col.lower() in df.columns and col not in df.columns:
                df[col] = df[col.lower()]
        
        # Ensure we have a Price column
        if 'Price' not in df.columns and 'Close' in df.columns:
            df['Price'] = df['Close']
    else:
        df = load_latest_data(args.ticker, args.data_dir)
    
    if df.empty:
        logger.error("No data loaded. Exiting.")
        return 1
    
    # Make predictions
    logger.info("🔮 Making predictions...")
    results = predictor.predict(df)
    
    # Display results
    print("\n📈 Prediction Results:")
    print("-" * 50)
    
    # Format datetime for display if it exists
    if 'datetime' in results.columns:
        if pd.api.types.is_datetime64_any_dtype(results['datetime']):
            # Format datetime for better display
            results['datetime_str'] = results['datetime'].dt.strftime('%Y-%m-%d %H:%M')
        else:
            results['datetime_str'] = results['datetime'].astype(str)
    else:
        results['datetime_str'] = 'N/A'
    
    # Show last 5 predictions
    display_cols = ['datetime_str']
    
    # Add price columns if they exist
    for price_col in ['Close', 'close']:
        if price_col in results.columns and results[price_col].notna().any():
            display_cols.append(price_col)
            break
    
    # Add prediction and confidence
    display_cols.extend(['prediction', 'confidence'])
    
    # Add interpretation column
    if 'prediction' in results.columns:
        # Add a human-readable interpretation column
        results['interpretation'] = results['prediction'].apply(lambda x: 
            "No Signal" if x == 'nan' else 
            "Buy Signal" if x == 'BUY' else
            "Sell Signal" if x == 'SELL' else str(x))
        display_cols.append('interpretation')
    
    # Get the last 5 rows for display
    last_predictions = results[display_cols].tail()
    
    # Format the output
    pd.set_option('display.float_format', '{:.2f}'.format)
    print(last_predictions.to_string(index=False))
    
    # Print summary
    if 'prediction' in results.columns:
        pred_counts = results['prediction'].value_counts()
        print("\nPrediction Summary:")
        for pred_type, count in pred_counts.items():
            print(f"  {pred_type}: {count} instances ({count/len(results)*100:.1f}%)")
    
    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)
        logger.info(f"📁 Predictions saved to {output_path}")
    
    return 0
        print("-" * 50)
        
        # Format datetime for display if it exists
        if 'datetime' in results.columns:
            if pd.api.types.is_datetime64_any_dtype(results['datetime']):
                # Format datetime for better display
                results['datetime_str'] = results['datetime'].dt.strftime('%Y-%m-%d %H:%M')
            else:
                results['datetime_str'] = results['datetime'].astype(str)
        else:
            results['datetime_str'] = 'N/A'
        
        # Show last 5 predictions
        display_cols = ['datetime_str']
        
        # Add price columns if they exist
        for price_col in ['Close', 'close']:
            if price_col in results.columns and results[price_col].notna().any():
                display_cols.append(price_col)
                break
            
        # Add prediction and confidence
        display_cols.extend(['prediction', 'confidence'])
        
        # Add interpretation column
        if 'prediction' in results.columns:
            # Add a human-readable interpretation column
            results['interpretation'] = results['prediction'].apply(lambda x: 
                "No Signal" if x == 'nan' else 
                "Buy Signal" if x == 'BUY' else
                "Sell Signal" if x == 'SELL' else str(x))
            display_cols.append('interpretation')
        
        # Get the last 5 rows for display
        last_predictions = results[display_cols].tail()
        
        # Format the output
        pd.set_option('display.float_format', '{:.2f}'.format)
        print(last_predictions.to_string(index=False))
        
        # Print summary
        if 'prediction' in results.columns:
            pred_counts = results['prediction'].value_counts()
            print("\nPrediction Summary:")
            for pred_type, count in pred_counts.items():
                print(f"  {pred_type}: {count} instances ({count/len(results)*100:.1f}%)")
        
        # Save to file if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            results.to_csv(output_path, index=False)
            logger.info(f"📁 Predictions saved to {output_path}")
            
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
