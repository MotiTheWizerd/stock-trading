#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Stock Prediction Pipeline

This module integrates:
1. Contextual feature generation
2. Model calibration
3. Adaptive retraining
4. Confidence filtering

Into a unified prediction pipeline for improved trading decisions.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import joblib
import argparse

# Import custom modules
from contextual_features import ContextualFeatureGenerator
from model_calibration import ModelCalibrator
from adaptive_retraining import AdaptiveRetrainer
from predict import StockPredictor, apply_trader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class EnhancedStockPredictor:
    """
    Enhanced stock prediction pipeline with contextual features,
    model calibration, and adaptive retraining.
    """
    
    def __init__(self, model_path=None, model_dir='models', confidence_threshold=0.7,
                enable_contextual_features=True, enable_calibration=True):
        """
        Initialize the EnhancedStockPredictor.
        
        Args:
            model_path: Path to the model file
            model_dir: Directory to store models
            confidence_threshold: Minimum confidence for predictions
            enable_contextual_features: Whether to enable contextual features
            enable_calibration: Whether to enable model calibration
        """
        self.model_path = model_path
        self.model_dir = model_dir
        self.confidence_threshold = confidence_threshold
        self.enable_contextual_features = enable_contextual_features
        self.enable_calibration = enable_calibration
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize components
        self.feature_generator = ContextualFeatureGenerator()
        self.calibrator = ModelCalibrator(model_path)
        self.retrainer = AdaptiveRetrainer(model_dir)
        
        # Initialize base predictor
        self.base_predictor = StockPredictor(model_path) if model_path else None
    
    def load_model(self, model_path=None):
        """
        Load model from file.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Loaded model
        """
        if model_path:
            self.model_path = model_path
        
        if not self.model_path:
            raise ValueError("No model path specified")
        
        logger.info(f"Loading model from {self.model_path}")
        
        # Load model into base predictor
        self.base_predictor = StockPredictor(self.model_path)
        
        # Also load into calibrator
        self.calibrator.load_model(self.model_path)
        
        return self.base_predictor.model
    
    def enhance_features(self, df):
        """
        Add contextual features to the DataFrame.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with added contextual features
        """
        if not self.enable_contextual_features:
            logger.info("Contextual features disabled, skipping enhancement")
            return df
        
        logger.info("Adding contextual features")
        
        # Add all contextual features
        enhanced_df = self.feature_generator.add_all_features(df)
        
        logger.info(f"Added {len(enhanced_df.columns) - len(df.columns)} contextual features")
        
        return enhanced_df
    
    def calibrate_predictions(self, predictions_df):
        """
        Apply calibration to the predictions.
        
        Args:
            predictions_df: DataFrame with predictions
            
        Returns:
            DataFrame with calibrated predictions
        """
        if not hasattr(self, 'calibrator') or not self.calibrator:
            logger.debug("No calibrator available, skipping calibration")
            return predictions_df
                
        if not hasattr(self.base_predictor.model, 'predict_proba'):
            logger.debug("Model doesn't support probability prediction, skipping calibration")
            return predictions_df
            
        logger.debug("Calibrating predictions")
        
        try:
            # Get feature columns from model
            feature_cols = self.base_predictor.get_feature_columns()
            
            # Find which feature columns are actually available in the data
            available_cols = [col for col in feature_cols if col in predictions_df.columns]
            missing_cols = set(feature_cols) - set(available_cols)
            
            if missing_cols:
                logger.debug(f"Missing {len(missing_cols)} feature columns in input data")
                
                if not available_cols:
                    logger.warning("No matching feature columns found. Cannot calibrate predictions.")
                    return predictions_df
                        
                logger.debug(f"Proceeding with {len(available_cols)}/{len(feature_cols)} available features")
            
            # Create a copy of the input dataframe to avoid SettingWithCopyWarning
            result_df = predictions_df.copy()
            
            # Add missing columns with default values (0 for numeric, 'NONE' for categorical)
            for col in missing_cols:
                if col.startswith(('Bollinger_', 'EMA_', 'SMA_', 'RSI_', 'MACD_', 'Candle_', 'Close_', 'Volume_')):
                    result_df[col] = 0.0  # Default numeric value for technical indicators
                    logger.debug(f"Added missing technical indicator column with default value 0: {col}")
                else:
                    result_df[col] = 'NONE'  # Default value for other columns
            
            # Ensure all required columns are present
            X = result_df[feature_cols]
            
            # Get calibrated probabilities
            calibrated_probs = self.calibrator.calibrated_model.predict_proba(X)
            
            # Update confidence with calibrated probabilities
            if 'confidence' in result_df.columns:
                result_df['original_confidence'] = result_df['confidence']
            
            result_df['confidence'] = np.max(calibrated_probs, axis=1)
            
            # Update predictions if confidence threshold is met
            if hasattr(self, 'confidence_threshold'):
                high_confidence = result_df['confidence'] >= self.confidence_threshold
                result_df.loc[~high_confidence, 'prediction'] = 'NONE'
                
                logger.debug(f"Applied confidence threshold of {self.confidence_threshold:.2f}")
                logger.debug(f"Predictions after thresholding: {result_df['prediction'].value_counts().to_dict()}")
            
            return result_df
                
        except Exception as e:
            logger.debug(f"Error during calibration: {str(e)}")
            logger.debug("Skipping calibration due to error")
            return predictions_df
    
    def filter_by_confidence(self, predictions_df):
        """
        Filter predictions by confidence threshold.
        
        Args:
            predictions_df: DataFrame with predictions
            
        Returns:
            DataFrame with filtered predictions
        """
        logger.info(f"Filtering predictions with confidence threshold {self.confidence_threshold}")
        
        # Filter by confidence
        filtered_df = predictions_df[predictions_df['confidence'] >= self.confidence_threshold].copy()
        
        # For rows below threshold, set prediction to 'NONE'
        predictions_df.loc[predictions_df['confidence'] < self.confidence_threshold, 'prediction'] = 'NONE'
        
        # Update prediction_with_risk column
        if 'risk_emoji' in predictions_df.columns:
            predictions_df['prediction_with_risk'] = predictions_df.apply(
                lambda row: f"{row['prediction']} {row['risk_emoji']} {row['risk_tier']}" 
                if row['prediction'] != 'NONE' else 'NONE', 
                axis=1
            )
        
        logger.info(f"Filtered {len(predictions_df) - len(filtered_df)} predictions below threshold")
        logger.info(f"Remaining {len(filtered_df)} predictions above threshold")
        
        return predictions_df
    
    def predict(self, df, apply_trading=True, initial_balance=1000.0):
        """
        Make enhanced predictions with the full pipeline.
        
        Args:
            df: DataFrame with price data
            apply_trading: Whether to apply trading simulation
            initial_balance: Initial balance for trading simulation
            
        Returns:
            DataFrame with predictions and trading results
        """
        logger.info("Starting enhanced prediction pipeline")
        
        # Step 1: Enhance features
        enhanced_df = self.enhance_features(df)
        
        # Step 2: Make base predictions
        if not self.base_predictor:
            raise ValueError("No model loaded. Call load_model() first.")
        
        predictions_df = self.base_predictor.predict(enhanced_df)
        
        # Step 3: Calibrate predictions
        calibrated_df = self.calibrate_predictions(predictions_df)
        
        # Step 4: Filter by confidence
        filtered_df = self.filter_by_confidence(calibrated_df)
        
        # Step 5: Apply trading simulation if requested
        if apply_trading:
            logger.info("Applying trading simulation")
            results_df = apply_trader(filtered_df, initial_balance=initial_balance)
        else:
            results_df = filtered_df
        
        logger.info("Enhanced prediction pipeline complete")
        
        return results_df
    
    def check_for_retraining(self, df, target_col='prediction', window_size=100, threshold=0.1):
        """
        Check if model needs retraining due to concept drift.
        
        Args:
            df: DataFrame with predictions and actual outcomes
            target_col: Name of the target column
            window_size: Size of the sliding window for drift detection
            threshold: Performance drop threshold for triggering retraining
            
        Returns:
            bool: True if retraining is needed, False otherwise
        """
        if not self.base_predictor or not self.base_predictor.model:
            logger.warning("No model loaded, cannot check for retraining")
            return False
            
        if not hasattr(self, 'retrainer'):
            logger.warning("No retrainer configured, cannot check for retraining")
            return False
            
        # Extract features and target
        feature_cols = self.base_predictor.get_feature_columns()
        X = df[feature_cols]
        
        if target_col not in df.columns:
            logger.warning(f"Target column '{target_col}' not found, cannot check for retraining")
            return False
            
        y = df[target_col]
        
        # Detect concept drift
        return self.retrainer.detect_concept_drift(
            model=self.base_predictor.model,
            X=X,
            y=y,
            window_size=window_size,
            threshold=threshold
        )

def calibrate_predictions(self, predictions_df):
    """
    Apply calibration to the predictions.
    
    Args:
        predictions_df: DataFrame with predictions
        
    Returns:
        DataFrame with calibrated predictions
    """
    if not hasattr(self, 'calibrator') or not self.calibrator:
        logger.warning("No calibrator available, skipping calibration")
        return predictions_df
            
    if not hasattr(self.base_predictor.model, 'predict_proba'):
        logger.warning("Model doesn't support probability prediction, skipping calibration")
        return predictions_df
        
    logger.info("Calibrating predictions")
    
    # Get feature columns from model
    feature_cols = self.base_predictor.get_feature_columns()
    
    # Find which feature columns are actually available in the data
    available_cols = [col for col in feature_cols if col in predictions_df.columns]
    missing_cols = set(feature_cols) - set(available_cols)
    
    if missing_cols:
        logger.warning(f"Missing {len(missing_cols)} feature columns in input data. "
                     f"Available columns: {list(predictions_df.columns)}")
            
        if not available_cols:
            logger.error("No matching feature columns found. Cannot calibrate predictions.")
            return predictions_df
                
        logger.info(f"Proceeding with {len(available_cols)}/{len(feature_cols)} available features")
    
    try:
        # Try with available features
        X = predictions_df[available_cols]
        
        # Get original probabilities
        original_probs = self.base_predictor.model.predict_proba(X)
        
        # Get calibrated probabilities if calibrated model exists
        if hasattr(self.calibrator, 'calibrated_model') and self.calibrator.calibrated_model:
            calibrated_probs = self.calibrator.calibrated_model.predict_proba(X)
            
            # Update confidence with calibrated probabilities
            if 'confidence' in predictions_df.columns:
                predictions_df['original_confidence'] = predictions_df['confidence']
            predictions_df['confidence'] = np.max(calibrated_probs, axis=1)
            
            # Update predictions if confidence threshold is met
            if hasattr(self, 'confidence_threshold'):
                high_confidence = predictions_df['confidence'] >= self.confidence_threshold
                predictions_df.loc[~high_confidence, 'prediction'] = 'NONE'
                
                logger.info(f"Applied confidence threshold of {self.confidence_threshold:.2f}")
                logger.info(f"Predictions after thresholding: {predictions_df['prediction'].value_counts().to_dict()}")
        
        return predictions_df
            
    except Exception as e:
        logger.error(f"Error during calibration: {str(e)}")
        logger.warning("Skipping calibration due to error")
        return predictions_df

def filter_by_confidence(self, predictions_df):
    """
    Filter predictions by confidence threshold.
    
    Args:
        predictions_df: DataFrame with predictions
        
    Returns:
        DataFrame with filtered predictions
    """
    logger.info(f"Filtering predictions with confidence threshold {self.confidence_threshold}")
    
    # Filter by confidence
    filtered_df = predictions_df[predictions_df['confidence'] >= self.confidence_threshold].copy()
    
    # For rows below threshold, set prediction to 'NONE'
    predictions_df.loc[predictions_df['confidence'] < self.confidence_threshold, 'prediction'] = 'NONE'
    
    # Update prediction_with_risk column
    if 'risk_emoji' in predictions_df.columns and 'risk_tier' in predictions_df.columns:
        predictions_df['prediction_with_risk'] = predictions_df.apply(
            lambda row: f"{row['prediction']} {row['risk_emoji']} {row['risk_tier']}" 
            if row['prediction'] != 'NONE' else 'NONE', 
            axis=1
        )


def main():
    """Main function to run the enhanced prediction pipeline."""
    parser = argparse.ArgumentParser(description='Enhanced stock prediction pipeline')
    parser.add_argument('--data-file', type=str, required=True, help='Path to data CSV file')
    parser.add_argument('--model-path', type=str, required=True, help='Path to model file')
    parser.add_argument('--output-file', type=str, help='Path to output CSV file')
    parser.add_argument('--confidence-threshold', type=float, default=0.7, 
                       help='Confidence threshold for predictions')
    parser.add_argument('--initial-balance', type=float, default=1000.0,
                       help='Initial balance for trading simulation')
    parser.add_argument('--disable-contextual', action='store_true',
                       help='Disable contextual features')
    parser.add_argument('--disable-calibration', action='store_true',
                       help='Disable model calibration')
    
    args = parser.parse_args()
    
    # Set default output file if not provided
    if not args.output_file:
        data_dir = os.path.dirname(args.data_file)
        data_name = os.path.basename(args.data_file).split('.')[0]
        args.output_file = os.path.join(data_dir, f"{data_name}_predictions.csv")
    
    # Load data
    logger.info(f"Loading data from {args.data_file}")
    df = pd.read_csv(args.data_file)
    
    # Create predictor
    predictor = EnhancedStockPredictor(
        model_path=args.model_path,
        confidence_threshold=args.confidence_threshold,
        enable_contextual_features=not args.disable_contextual,
        enable_calibration=not args.disable_calibration
    )
    
    # Make predictions
    results = predictor.predict(
        df=df,
        apply_trading=True,
        initial_balance=args.initial_balance
    )
    
    # Save results
    logger.info(f"Saving results to {args.output_file}")
    results.to_csv(args.output_file, index=False)
    
    # Print summary
    print("\n=== Prediction Summary ===")
    print(f"Total predictions: {len(results)}")
    print(f"BUY signals: {len(results[results['prediction'] == 'BUY'])}")
    print(f"SELL signals: {len(results[results['prediction'] == 'SELL'])}")
    print(f"NONE signals: {len(results[results['prediction'] == 'NONE'])}")
    
    if 'final_balance' in results.columns:
        initial = args.initial_balance
        final = results['final_balance'].iloc[-1]
        profit = final - initial
        profit_pct = (profit / initial) * 100
        
        print("\n=== Trading Summary ===")
        print(f"Initial balance: ${initial:.2f}")
        print(f"Final balance: ${final:.2f}")
        print(f"Profit/Loss: ${profit:.2f} ({profit_pct:.2f}%)")
        
        if 'max_drawdown' in results.columns:
            max_dd = results['max_drawdown'].max()
            print(f"Maximum drawdown: {max_dd:.2f}%")
    
    logger.info("Enhanced prediction pipeline complete")


if __name__ == "__main__":
    main()
