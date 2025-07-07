#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Calibration and Confidence Filtering Module

This module implements:
1. Platt Scaling for probability calibration
2. Isotonic Regression for non-parametric calibration
3. Confidence threshold analysis
4. Brier score and calibration curve evaluation
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import joblib
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class ModelCalibrator:
    """
    A class to calibrate model probabilities and filter predictions by confidence.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the ModelCalibrator.
        
        Args:
            model_path: Path to the trained model file
        """
        self.model_path = model_path
        self.model = None
        self.calibrated_model = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """
        Load the trained model from file.
        
        Args:
            model_path: Path to the model file (str or Path object)
            
        Returns:
            Loaded model
        """
        logger.info(f"Loading model from {model_path}")
        try:
            # Convert Path object to string if needed
            model_path_str = str(model_path)
            
            if model_path_str.endswith('.pkl'):
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
            else:
                self.model = joblib.load(model_path)
            return self.model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def calibrate_model(self, X_train, y_train, method='sigmoid', cv=5):
        """
        Calibrate model probabilities using Platt Scaling or Isotonic Regression.
        
        Args:
            X_train: Training features
            y_train: Training targets
            method: 'sigmoid' for Platt Scaling or 'isotonic' for Isotonic Regression
            cv: Number of cross-validation folds
            
        Returns:
            Calibrated model
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        logger.info(f"Calibrating model using {method} method with {cv}-fold CV")
        
        # Create calibrated model
        self.calibrated_model = CalibratedClassifierCV(
            base_estimator=self.model,
            method=method,
            cv=cv
        )
        
        # Fit the calibrated model
        self.calibrated_model.fit(X_train, y_train)
        
        logger.info("Model calibration complete")
        
        return self.calibrated_model
    
    def evaluate_calibration(self, X_test, y_test, n_bins=10):
        """
        Evaluate calibration quality using calibration curves and Brier score.
        
        Args:
            X_test: Test features
            y_test: Test targets
            n_bins: Number of bins for calibration curve
            
        Returns:
            Dictionary with calibration metrics
        """
        if self.calibrated_model is None:
            if self.model is None:
                raise ValueError("No model loaded. Call load_model() first.")
            logger.warning("Using uncalibrated model for evaluation")
            model = self.model
        else:
            logger.info("Using calibrated model for evaluation")
            model = self.calibrated_model
        
        # Get predictions and probabilities
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        # For multiclass, we'll evaluate calibration for each class
        n_classes = y_prob.shape[1]
        
        # Initialize results
        results = {
            'brier_scores': {},
            'accuracy': np.mean(y_pred == y_test),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        for i in range(n_classes):
            # Convert to binary problem for this class
            y_binary = (y_test == i).astype(int)
            prob_pos = y_prob[:, i]
            
            # Calculate calibration curve
            prob_true, prob_pred = calibration_curve(y_binary, prob_pos, n_bins=n_bins)
            
            # Calculate Brier score
            brier = brier_score_loss(y_binary, prob_pos)
            results['brier_scores'][f'class_{i}'] = brier
            
            # Plot calibration curve
            plt.plot(prob_pred, prob_true, marker='o', 
                    label=f'Class {i} (Brier: {brier:.3f})')
        
        # Plot diagonal (perfect calibration)
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.title('Calibration Curve')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        os.makedirs('reports', exist_ok=True)
        plt.savefig('reports/calibration_curve.png')
        logger.info("Calibration curve saved to reports/calibration_curve.png")
        
        return results
    
    def analyze_confidence_thresholds(self, X, y, thresholds=[0.5, 0.6, 0.7, 0.8, 0.9]):
        """
        Analyze prediction performance at different confidence thresholds.
        
        Args:
            X: Features
            y: Targets
            thresholds: List of confidence thresholds to evaluate
            
        Returns:
            DataFrame with performance metrics at each threshold
        """
        if self.calibrated_model is None:
            if self.model is None:
                raise ValueError("No model loaded. Call load_model() first.")
            logger.warning("Using uncalibrated model for confidence analysis")
            model = self.model
        else:
            logger.info("Using calibrated model for confidence analysis")
            model = self.calibrated_model
        
        # Get predictions and probabilities
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)
        
        # Get confidence scores (max probability)
        confidence = np.max(y_prob, axis=1)
        
        # Evaluate at different thresholds
        results = []
        
        for threshold in thresholds:
            # Filter by confidence
            mask = confidence >= threshold
            
            # Skip if no predictions meet the threshold
            if not np.any(mask):
                logger.warning(f"No predictions meet confidence threshold {threshold}")
                continue
            
            # Calculate metrics
            filtered_y_true = y[mask] if isinstance(y, np.ndarray) else y.iloc[mask]
            filtered_y_pred = y_pred[mask]
            
            # Accuracy
            accuracy = np.mean(filtered_y_pred == filtered_y_true)
            
            # Coverage (percentage of data that meets threshold)
            coverage = np.mean(mask)
            
            # Class distribution
            class_counts = pd.Series(filtered_y_pred).value_counts(normalize=True)
            
            # Store results
            results.append({
                'threshold': threshold,
                'accuracy': accuracy,
                'coverage': coverage,
                'n_samples': np.sum(mask),
                'class_distribution': class_counts.to_dict()
            })
        
        # Create DataFrame with results
        results_df = pd.DataFrame(results)
        
        # Plot accuracy vs. coverage
        plt.figure(figsize=(10, 6))
        plt.plot(results_df['threshold'], results_df['accuracy'], 'o-', label='Accuracy')
        plt.plot(results_df['threshold'], results_df['coverage'], 'o-', label='Coverage')
        plt.xlabel('Confidence Threshold')
        plt.ylabel('Metric Value')
        plt.title('Accuracy vs. Coverage at Different Confidence Thresholds')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        os.makedirs('reports', exist_ok=True)
        plt.savefig('reports/confidence_threshold_analysis.png')
        logger.info("Confidence threshold analysis saved to reports/confidence_threshold_analysis.png")
        
        # Save the results
        results_df.to_csv('reports/confidence_threshold_analysis.csv', index=False)
        logger.info("Confidence threshold analysis saved to reports/confidence_threshold_analysis.csv")
        
        return results_df
    
    def save_calibrated_model(self, output_path=None):
        """
        Save the calibrated model to file.
        
        Args:
            output_path: Path to save the calibrated model
            
        Returns:
            Path to the saved model
        """
        if self.calibrated_model is None:
            raise ValueError("No calibrated model available. Call calibrate_model() first.")
        
        # Set default output path if not provided
        if output_path is None:
            if self.model_path:
                model_dir = os.path.dirname(self.model_path)
                model_name = os.path.basename(self.model_path).split('.')[0]
                output_path = os.path.join(model_dir, f"{model_name}_calibrated.joblib")
            else:
                output_path = "calibrated_model.joblib"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save model
        logger.info(f"Saving calibrated model to {output_path}")
        joblib.dump(self.calibrated_model, output_path)
        
        return output_path
    
    def filter_predictions(self, X, confidence_threshold=0.7):
        """
        Filter predictions based on confidence threshold.
        
        Args:
            X: Features
            confidence_threshold: Minimum confidence for predictions
            
        Returns:
            Dictionary with filtered predictions and confidence scores
        """
        if self.calibrated_model is None:
            if self.model is None:
                raise ValueError("No model loaded. Call load_model() first.")
            logger.warning("Using uncalibrated model for prediction filtering")
            model = self.model
        else:
            logger.info("Using calibrated model for prediction filtering")
            model = self.calibrated_model
        
        # Get predictions and probabilities
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)
        
        # Get confidence scores (max probability)
        confidence = np.max(y_prob, axis=1)
        
        # Filter by confidence
        mask = confidence >= confidence_threshold
        
        # Get filtered predictions
        filtered_pred = y_pred[mask] if np.any(mask) else np.array([])
        filtered_conf = confidence[mask] if np.any(mask) else np.array([])
        filtered_indices = np.where(mask)[0] if np.any(mask) else np.array([])
        
        logger.info(f"Filtered {len(y_pred)} predictions to {len(filtered_pred)} with confidence >= {confidence_threshold}")
        
        return {
            'predictions': filtered_pred,
            'confidence': filtered_conf,
            'indices': filtered_indices,
            'mask': mask
        }


def main():
    """Main function to calibrate model and analyze confidence thresholds."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Calibrate model and analyze confidence thresholds')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--train-data', type=str, required=True, help='Path to training data CSV')
    parser.add_argument('--test-data', type=str, help='Path to test data CSV')
    parser.add_argument('--method', type=str, default='sigmoid', choices=['sigmoid', 'isotonic'],
                        help='Calibration method')
    parser.add_argument('--output-path', type=str, help='Path to save calibrated model')
    parser.add_argument('--feature-cols', type=str, nargs='+', help='Feature column names')
    parser.add_argument('--target-col', type=str, default='prediction', help='Target column name')
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading training data from {args.train_data}")
    train_df = pd.read_csv(args.train_data)
    
    if args.test_data:
        logger.info(f"Loading test data from {args.test_data}")
        test_df = pd.read_csv(args.test_data)
    else:
        logger.info("No test data provided, splitting training data")
        train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=42)
    
    # Determine feature columns
    if args.feature_cols:
        feature_cols = args.feature_cols
    else:
        # Exclude known non-feature columns
        exclude_cols = [args.target_col, 'datetime', 'Open', 'High', 'Low', 'Close', 
                       'Volume', 'Adj Close', 'prediction', 'confidence']
        feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    logger.info(f"Using {len(feature_cols)} feature columns")
    
    # Extract features and targets
    X_train = train_df[feature_cols]
    y_train = train_df[args.target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[args.target_col]
    
    # Create calibrator
    calibrator = ModelCalibrator(args.model_path)
    
    # Calibrate model
    calibrator.calibrate_model(X_train, y_train, method=args.method)
    
    # Evaluate calibration
    calibration_results = calibrator.evaluate_calibration(X_test, y_test)
    
    # Analyze confidence thresholds
    threshold_results = calibrator.analyze_confidence_thresholds(X_test, y_test)
    
    # Save calibrated model
    calibrator.save_calibrated_model(args.output_path)
    
    logger.info("Model calibration and confidence analysis complete")


if __name__ == "__main__":
    main()
