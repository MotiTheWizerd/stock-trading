#!/usr/bin/env python
"""
Stock Signal Predictor

This script uses a trained model to predict stock signals (BUY, SELL, NONE)
based on technical indicators.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
import logging

def _norm(col: str) -> str:
    """
    Normalize a column name:
    • lower-case
    • strip spaces
    • replace punctuation with nothing
    """
    return (
        str(col).lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )

logger = logging.getLogger(__name__)

def setup_logging():
    """Configure logging for the prediction script."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
    )
    
    # Disable verbose logs from other libraries
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("pandas").setLevel(logging.WARNING)
    logging.getLogger("sklearn").setLevel(logging.WARNING)


class StockPredictor:
    """Class to load a trained model and make predictions on new data."""
    
    def __init__(self, ticker: str, model_dir: str = None):
        """Initialize the predictor with a ticker and model directory.
        
        Args:
            ticker: Stock ticker symbol
            model_dir: Directory containing model files (default: models/<ticker>)
        """
        self.ticker = ticker.upper()
        
        # Set model directory
        if model_dir:
            self.model_dir = Path(model_dir)
        else:
            self.model_dir = Path("models") / self.ticker
        
        # Initialize model and feature columns
        self.model = None
        self.model_classes = []
        self.feature_columns = []
        self.metadata = {}
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """Load the latest model for the ticker."""
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")
        
        # Find the latest model file
        model_files = list(self.model_dir.glob("*.joblib"))
        if not model_files:
            raise FileNotFoundError(f"No model files found in {self.model_dir}")
        
        # Sort by modification time (newest first)
        model_file = sorted(model_files, key=lambda f: f.stat().st_mtime, reverse=True)[0]
        
        # Load the model
        logger.info(f"✅ Loaded model: {model_file.name}")
        self.model = joblib.load(model_file)
        
        # Get model classes
        self.model_classes = self.model.classes_.tolist()
        logger.info(f"Model classes: {self.model_classes}")
        
        # Load metadata if available
        metadata_file = model_file.with_suffix(".json")
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                self.metadata = json.load(f)
            
            # Extract feature columns from metadata
            if "feature_columns" in self.metadata:
                self.feature_columns = self.metadata["feature_columns"]
                logger.info(f"Extracted feature columns from metadata: {len(self.feature_columns)} features")
        
        # If feature columns are not in metadata, try to get them from the model
        if not self.feature_columns:
            logger.info("No feature columns found in metadata, trying to extract from model...")
            
            # Try to get feature names from the model pipeline
            if hasattr(self.model, 'feature_names_in_'):
                self.feature_columns = list(self.model.feature_names_in_)
                logger.info(f"Extracted feature columns from model.feature_names_in_: {len(self.feature_columns)} features")
            # Try to get feature names from the model inside the pipeline
            elif hasattr(self.model, 'named_steps') and hasattr(self.model.named_steps.get('model', None), 'feature_names_in_'):
                self.feature_columns = list(self.model.named_steps['model'].feature_names_in_)
                logger.info(f"Extracted feature columns from model.named_steps['model'].feature_names_in_: {len(self.feature_columns)} features")
            else:
                logger.error("❌ Could not recover feature list from model — re-train with metadata save.")
                raise ValueError("Model does not contain feature names; re-train with feature_columns in metadata.")
        
        logger.info(f"Feature columns (first 5): {self.feature_columns[:5] if len(self.feature_columns) >= 5 else self.feature_columns}...")
        logger.info(f"Total feature columns: {len(self.feature_columns)}")
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions on the given data.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with predictions and confidence scores
        """
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Log original columns for debugging
        logger.info(f"Original columns: {df_copy.columns.tolist()}")
        logger.info(f"Model expects features: {self.feature_columns}")
        
        # Normalize column names (replace spaces with underscores and standardize)
        df_copy.columns = [str(col).replace('_', ' ').strip() for col in df_copy.columns]
        
        # Ensure datetime is properly parsed
        if 'datetime' in df_copy.columns and not pd.api.types.is_datetime64_any_dtype(df_copy['datetime']):
            try:
                df_copy['datetime'] = pd.to_datetime(df_copy['datetime'])
                logger.info("Converted datetime column to datetime type")
            except Exception as e:
                logger.warning(f"Failed to convert datetime column: {e}")
        
        # Check if we have the necessary feature columns
        logger.info(f"Making predictions on {len(df_copy)} rows with {len(df_copy.columns)} columns")
        
        # Normalize feature columns from model
        normalized_feature_columns = [col.replace('_', ' ').strip() for col in self.feature_columns]
        
        # Check for missing feature columns
        missing_features = [col for col in normalized_feature_columns if col not in df_copy.columns]
        
        # Try alternative column names (with different case, underscores, etc.)
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            
            # Try alternative column name mappings
            column_mapping = {}
            for col in missing_features:
                # Try different variations of the column name
                variations = [
                    col.lower(),
                    col.upper(),
                    col.replace(' ', ''),
                    col.replace(' ', '_'),
                    col.replace('_', ' ').strip(),
                    col.lower().replace(' ', '_'),
                    col.upper().replace(' ', '_')
                ]
                
                # Find matching columns in the dataframe (case insensitive)
                for var in variations:
                    matches = [c for c in df_copy.columns if c.lower() == var.lower()]
                    if matches:
                        column_mapping[col] = matches[0]
                        logger.info(f"Mapped '{col}' to '{matches[0]}'")
                        break
            
            # Apply column mappings
            if column_mapping:
                df_copy = df_copy.rename(columns={v: k for k, v in column_mapping.items()})
                # Update missing features list
                missing_features = [col for col in normalized_feature_columns if col not in df_copy.columns]
        
        # If we still have missing features, try feature engineering
        if missing_features:
            logger.warning(f"Still missing features after column mapping: {missing_features}")
            
            # Try to apply feature engineering to get missing features
            logger.info("Applying feature engineering to enrich data...")
            try:
                from feature_engineer import FeatureEngineer
                engineer = FeatureEngineer()
                df_copy = engineer.enrich_dataframe(df_copy)
                logger.info(f"After feature engineering: {len(df_copy.columns)} columns")
                
                # Normalize column names again after feature engineering
                df_copy.columns = [str(col).replace('_', ' ').strip() for col in df_copy.columns]
                
                # Update missing features list
                missing_features = [col for col in normalized_feature_columns if col not in df_copy.columns]
                
            except Exception as e:
                logger.error(f"Feature engineering failed: {e}")
        
        # If we still have missing features, add them with zeros
        if missing_features:
            logger.warning(f"Still missing features after engineering: {missing_features}")
            logger.info(f"Available columns: {df_copy.columns.tolist()}")
            
            # Add missing columns with zeros
            for col in missing_features:
                logger.info(f"Adding missing column with zeros: {col}")
                df_copy[col] = 0.0  # Use float for consistency with numeric features
        
        # Implement robust column name normalization to fix feature name mismatch
        try:
            # --- 1. build a mapping from normalized name ➞ original in df ---
            norm_to_raw = {_norm(c): c for c in df_copy.columns}
            
            logger.info("=== COLUMN NORMALIZATION ===")
            logger.info(f"Data columns: {df_copy.columns.tolist()}")
            logger.info(f"Model features: {self.feature_columns}")
            
            # --- 2. build a list of columns the model expects, after mapping ---
            mapped_cols = []
            missing_cols = []
            
            for model_col in self.feature_columns:
                key = _norm(model_col)
                if key in norm_to_raw:
                    mapped_cols.append(norm_to_raw[key])
                    logger.info(f"✓ Mapped '{model_col}' to '{norm_to_raw[key]}'")
                else:
                    missing_cols.append(model_col)
                    logger.info(f"✗ No match found for '{model_col}'")
            
            # --- 3. warn if anything is still missing ---
            if missing_cols:
                logger.warning(f"⚠️ Still missing expected features: {missing_cols}")
            
            # --- 4. if none matched, hard-fail early so you know why ---
            if not mapped_cols:
                logger.error("No matching features between model and data after normalization")
                logger.error(f"Model features: {self.feature_columns}")
                logger.error(f"Data columns: {df_copy.columns.tolist()}")
                raise ValueError("No matching features between model and data after normalization")
            
            # --- 5. select and order the columns exactly as the model saw them ---
            # Create ordered pairs of (model_column_name, raw_data_column_name)
            ordered_pairs = [
                (model_col, norm_to_raw[_norm(model_col)])
                for model_col in self.feature_columns
                if _norm(model_col) in norm_to_raw
            ]
            
            # Extract just the raw column names for selection
            ordered_cols = [raw_col for _, raw_col in ordered_pairs]
            logger.info(f"Using ordered columns: {ordered_cols}")
            
            # Create feature matrix with only the available features
            X = df_copy[ordered_cols].copy()
            
            # CRITICAL: Rename columns back to the model's original names
            # This ensures StandardScaler sees the exact same column names it was trained with
            model_cols = [model_col for model_col, _ in ordered_pairs]
            X.columns = model_cols
            
            logger.info(f"Feature matrix shape: {X.shape}")
            logger.info(f"Feature columns after renaming: {X.columns.tolist()}")
            
            # Check for any NaN values in the feature matrix
            nan_columns = X.columns[X.isna().any()].tolist()
            if nan_columns:
                logger.warning(f"Found NaN values in columns: {nan_columns}")
                # Fill NaN values with 0 as a fallback
                X = X.fillna(0)
            
            # Log first few rows of the feature matrix
            logger.debug(f"First few rows of feature matrix:\n{X.head()}")
            
            # Make predictions
            logger.info("Starting model prediction...")
            predictions = self.model.predict(X)
            logger.info("Model prediction completed successfully")
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            logger.error(f"Feature matrix shape: {X.shape if 'X' in locals() else 'N/A'}")
            logger.error(f"Feature matrix columns: {X.columns.tolist() if 'X' in locals() else 'N/A'}")
            logger.error(f"Feature matrix dtypes: {X.dtypes if 'X' in locals() else 'N/A'}")
            raise
            
        probabilities = self.model.predict_proba(X)
        
        # Get confidence scores (probability of the predicted class)
        confidences = np.max(probabilities, axis=1)
        
        # Add predictions and confidence to the original dataframe
        # CRITICAL: We need to preserve the original datetime and price columns
        # Create a new dataframe with just the essential columns first
        essential_cols = []
        
        # Preserve datetime column if it exists
        if 'datetime' in df.columns:
            essential_cols.append('datetime')
        
        # Preserve price columns
        price_cols = ['Close', 'Open', 'High', 'Low', 'Adj Close', 'Price']
        for col in price_cols:
            if col in df.columns:
                essential_cols.append(col)
        
        # Create result dataframe with essential columns first
        if essential_cols:
            result_df = df[essential_cols].copy()
            logger.info(f"Preserved essential columns: {essential_cols}")
        else:
            result_df = pd.DataFrame(index=df.index)
            logger.warning("No essential columns found to preserve")
        
        # Add predictions and confidence
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
    
    # Ensure we have datetime column before prediction
    if 'datetime' in df.columns:
        logger.info("Found datetime column in input data")
    else:
        logger.warning("No datetime column in input data")
        
    # Make predictions while preserving datetime and price columns
    results = predictor.predict(df)
    
    # Display results
    print("\n📈 Prediction Results:")
    print("-" * 50)
    
    # Debug the columns available in results
    logger.info(f"Available columns in results: {results.columns.tolist()}")
    
    # Format datetime for display if it exists
    if 'datetime' in results.columns:
        if pd.api.types.is_datetime64_any_dtype(results['datetime']):
            # Format datetime for better display
            results['datetime_str'] = results['datetime'].dt.strftime('%Y-%m-%d %H:%M')
        else:
            results['datetime_str'] = results['datetime'].astype(str)
    else:
        # Try to find any datetime-like column
        datetime_candidates = [col for col in results.columns if 'date' in col.lower() or 'time' in col.lower()]
        if datetime_candidates:
            logger.info(f"Found datetime candidate columns: {datetime_candidates}")
            dt_col = datetime_candidates[0]
            if pd.api.types.is_datetime64_any_dtype(results[dt_col]):
                results['datetime_str'] = results[dt_col].dt.strftime('%Y-%m-%d %H:%M')
            else:
                results['datetime_str'] = results[dt_col].astype(str)
        else:
            results['datetime_str'] = 'N/A'
            logger.warning("No datetime column found in results")
    
    # Show last 5 predictions
    display_cols = ['datetime_str']
    
    # Add price columns if they exist
    price_candidates = ['Close', 'close', 'Price', 'price', 'Adj Close', 'Adj_Close']
    found_price = False
    for price_col in price_candidates:
        if price_col in results.columns and results[price_col].notna().any():
            display_cols.append(price_col)
            found_price = True
            logger.info(f"Using price column: {price_col}")
            break
    
    if not found_price:
        logger.warning(f"No price column found in results. Available columns: {results.columns.tolist()}")
    
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
    
    # Filter out rows with 'NONE' predictions
    if 'prediction' in results.columns:
        filtered_predictions = results[results['prediction'] != 'NONE'].copy()
    else:
        filtered_predictions = results.copy()
    
    if len(filtered_predictions) == 0:
        print("No non-NONE predictions to display.")
    else:
        # Add profit/loss calculation
        initial_balance = 1000.0
        balance = initial_balance
        position = 0  # 0: no position, 1: long, -1: short
        entry_price = 0.0
        signal_quality = []
        
        # Create lists to store the running balance for each row
        running_balance = []
        
        # First pass: Calculate signal quality based on next price movement
        for i in range(len(filtered_predictions)):
            current_row = filtered_predictions.iloc[i]
            current_price = current_row['Close']
            signal = current_row['prediction']
            
            # Default quality is neutral
            quality = "➖"  # Neutral indicator
            
            # If there's a next row, check price movement
            if i < len(filtered_predictions) - 1:
                next_price = filtered_predictions.iloc[i + 1]['Close']
                price_change = (next_price - current_price) / current_price * 100
                
                if signal == 'BUY':
                    quality = "✅" if price_change > 0 else "❌"  # Green check for good buy, red X for bad
                elif signal == 'SELL':
                    quality = "✅" if price_change < 0 else "❌"  # Green check for good sell, red X for bad
            
            signal_quality.append(quality)
        
        # Second pass: Calculate running balance
        for i, (_, row) in enumerate(filtered_predictions.iterrows()):
            current_price = row['Close']
            
            if row['prediction'] == 'BUY' and position <= 0:
                # Close short position if any and open long
                if position == -1:
                    pnl = (entry_price - current_price) / entry_price * balance
                    balance += pnl
                # Open long position
                position = 1
                entry_price = current_price
            elif row['prediction'] == 'SELL' and position >= 0:
                # Close long position if any and open short
                if position == 1:
                    pnl = (current_price - entry_price) / entry_price * balance
                    balance += pnl
                # Open short position
                position = -1
                entry_price = current_price
            
            running_balance.append(balance)
        
        # Add signal quality and running balance to the display
        filtered_predictions['signal_quality'] = signal_quality
        filtered_predictions['balance'] = running_balance
        filtered_predictions['P/L ($)'] = filtered_predictions['balance'] - initial_balance
        filtered_predictions['P/L (%)'] = (filtered_predictions['balance'] / initial_balance - 1) * 100
        
        # Reorder columns for better display
        display_cols = ['datetime_str', 'Close', 'prediction', 'signal_quality', 'confidence', 'balance', 'P/L ($)', 'P/L (%)']
        
        # Format the output
        pd.set_option('display.float_format', '{:,.2f}'.format)
        pd.set_option('display.max_rows', None)  # Show all rows
        
        print("\n📊 Trading Performance (Starting Balance: $1,000.00)")
        print("-" * 100)
        print("✅ Good Signal  ❌ Bad Signal  ➖ Neutral/Unknown")
        print("-" * 100)
        print(filtered_predictions[display_cols].to_string(index=False))
    
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

if __name__ == "__main__":
    sys.exit(main())
