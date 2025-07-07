#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Improvement Module for Stock Trading Agents

This module implements various techniques to improve model performance:
1. Confidence Calibration
2. Feature Importance Analysis
3. Class Imbalance Correction
4. Hard Negative Mining
5. Custom Loss Functions
6. Contextual Features
7. Signal Confirmation Layer
8. Adaptive Retraining
9. Risk-aware Output Layer
10. Prediction Filtering Rules
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
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import VotingClassifier
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class ModelImprover:
    """
    A class to improve model performance through various techniques.
    """
    
    def __init__(self, model_path, data_path=None):
        """
        Initialize the ModelImprover with a trained model and optional data.
        
        Args:
            model_path: Path to the trained model file
            data_path: Optional path to the data file for analysis
        """
        self.model_path = model_path
        self.data_path = data_path
        self.model = self._load_model(model_path)
        self.data = None
        self.X = None
        self.y = None
        self.feature_names = None
        
        if data_path:
            self.load_data(data_path)
    
    def _load_model(self, model_path):
        """Load the trained model from file."""
        logger.info(f"Loading model from {model_path}")
        try:
            if model_path.endswith('.pkl'):
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            else:
                model = joblib.load(model_path)
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def load_data(self, data_path):
        """Load and prepare data for analysis."""
        logger.info(f"Loading data from {data_path}")
        try:
            self.data = pd.read_csv(data_path)
            logger.info(f"Data loaded with shape: {self.data.shape}")
            return self.data
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def prepare_features_targets(self, target_col='prediction', datetime_col='datetime'):
        """
        Extract features and targets from the data.
        
        Args:
            target_col: Name of the target column
            datetime_col: Name of the datetime column
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Extract feature names from model if available
        try:
            if hasattr(self.model, 'feature_names_in_'):
                self.feature_names = self.model.feature_names_in_
            elif hasattr(self.model, 'feature_names'):
                self.feature_names = self.model.feature_names
            # Handle pipeline case
            elif hasattr(self.model, 'steps'):
                for _, step in self.model.steps:
                    if hasattr(step, 'feature_names_in_'):
                        self.feature_names = step.feature_names_in_
                        break
        except Exception as e:
            logger.warning(f"Could not extract feature names from model: {e}")
        
        # If we still don't have feature names, try to infer from data
        if self.feature_names is None:
            # Exclude known non-feature columns
            exclude_cols = [target_col, datetime_col, 'Open', 'High', 'Low', 'Close', 
                           'Volume', 'Adj Close', 'prediction', 'confidence']
            self.feature_names = [col for col in self.data.columns 
                                 if col not in exclude_cols]
            logger.warning(f"Inferred {len(self.feature_names)} feature names from data")
        
        # Extract features and target
        self.X = self.data[self.feature_names].copy()
        
        # Handle missing target (for prediction data)
        if target_col in self.data.columns:
            self.y = self.data[target_col].copy()
            # Convert string labels to numeric if needed
            if self.y.dtype == 'object':
                label_map = {'BUY': 0, 'NONE': 1, 'SELL': 2}
                self.y = self.y.map(label_map)
        else:
            self.y = None
            logger.warning(f"Target column '{target_col}' not found in data")
        
        logger.info(f"Prepared features with shape: {self.X.shape}")
        if self.y is not None:
            logger.info(f"Target distribution: {self.y.value_counts(normalize=True)}")
        
        return self.X, self.y

    def calibrate_model(self, method='sigmoid', cv=5):
        """
        Calibrate model confidence scores using Platt Scaling or Isotonic Regression.
        
        Args:
            method: 'sigmoid' for Platt Scaling or 'isotonic' for Isotonic Regression
            cv: Number of cross-validation folds
            
        Returns:
            Calibrated model
        """
        if self.X is None or self.y is None:
            raise ValueError("Features and targets not prepared. Call prepare_features_targets() first.")
        
        logger.info(f"Calibrating model using {method} method with {cv}-fold CV")
        
        # Create calibrated model
        calibrated_model = CalibratedClassifierCV(
            base_estimator=self.model,
            method=method,
            cv=cv
        )
        
        # Fit the calibrated model
        calibrated_model.fit(self.X, self.y)
        
        # Evaluate calibration
        y_prob = calibrated_model.predict_proba(self.X)
        
        # For multiclass, we'll evaluate calibration for each class
        n_classes = y_prob.shape[1]
        
        plt.figure(figsize=(10, 8))
        
        for i in range(n_classes):
            # Convert to binary problem for this class
            y_binary = (self.y == i).astype(int)
            prob_pos = y_prob[:, i]
            
            # Calculate calibration curve
            prob_true, prob_pred = calibration_curve(y_binary, prob_pos, n_bins=10)
            
            # Calculate Brier score
            brier = brier_score_loss(y_binary, prob_pos)
            
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
        
        # Save the calibrated model
        model_dir = os.path.dirname(self.model_path)
        calibrated_model_path = os.path.join(model_dir, 'calibrated_model.joblib')
        joblib.dump(calibrated_model, calibrated_model_path)
        logger.info(f"Calibrated model saved to {calibrated_model_path}")
        
        return calibrated_model

    def analyze_feature_importance(self, n_repeats=10, random_state=42):
        """
        Analyze feature importance using permutation importance.
        
        Args:
            n_repeats: Number of times to permute each feature
            random_state: Random seed for reproducibility
            
        Returns:
            DataFrame with feature importance scores
        """
        if self.X is None or self.y is None:
            raise ValueError("Features and targets not prepared. Call prepare_features_targets() first.")
        
        logger.info("Analyzing feature importance using permutation importance")
        
        # Calculate permutation importance
        perm_importance = permutation_importance(
            self.model, self.X, self.y,
            n_repeats=n_repeats,
            random_state=random_state
        )
        
        # Create DataFrame with importance scores
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': perm_importance.importances_mean,
            'Std': perm_importance.importances_std
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(12, 10))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
        plt.title('Feature Importance (Top 20)')
        plt.tight_layout()
        
        # Save the plot
        os.makedirs('reports', exist_ok=True)
        plt.savefig('reports/feature_importance.png')
        logger.info("Feature importance plot saved to reports/feature_importance.png")
        
        # Save the importance DataFrame
        importance_df.to_csv('reports/feature_importance.csv', index=False)
        logger.info("Feature importance data saved to reports/feature_importance.csv")
        
        return importance_df

    def analyze_feature_distributions(self):
        """
        Analyze feature distributions to identify potential issues.
        
        Returns:
            DataFrame with feature statistics
        """
        if self.X is None:
            raise ValueError("Features not prepared. Call prepare_features_targets() first.")
        
        logger.info("Analyzing feature distributions")
        
        # Calculate statistics for each feature
        stats_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Mean': self.X.mean().values,
            'Std': self.X.std().values,
            'Min': self.X.min().values,
            'Max': self.X.max().values,
            'Zeros': (self.X == 0).sum().values,
            'NaNs': self.X.isna().sum().values,
            'Unique': [self.X[col].nunique() for col in self.X.columns]
        })
        
        # Identify potential issues
        stats_df['ZeroVariance'] = stats_df['Std'] < 1e-6
        stats_df['HighNaNs'] = stats_df['NaNs'] > len(self.X) * 0.1  # >10% NaNs
        stats_df['LowUnique'] = stats_df['Unique'] < 10  # <10 unique values
        
        # Save the statistics
        os.makedirs('reports', exist_ok=True)
        stats_df.to_csv('reports/feature_statistics.csv', index=False)
        logger.info("Feature statistics saved to reports/feature_statistics.csv")
        
        # Plot histograms for features with issues
        problem_features = stats_df[
            stats_df['ZeroVariance'] | stats_df['HighNaNs'] | stats_df['LowUnique']
        ]['Feature'].tolist()
        
        if problem_features:
            logger.warning(f"Found {len(problem_features)} features with potential issues")
            
            # Plot histograms for problem features
            n_cols = 2
            n_rows = (len(problem_features) + 1) // 2
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 4))
            axes = axes.flatten()
            
            for i, feature in enumerate(problem_features):
                if i < len(axes):
                    sns.histplot(self.X[feature], ax=axes[i])
                    axes[i].set_title(feature)
            
            plt.tight_layout()
            plt.savefig('reports/problem_features.png')
            logger.info("Problem feature histograms saved to reports/problem_features.png")
        
        return stats_df

    def correct_class_imbalance(self, method='smote', random_state=42):
        """
        Correct class imbalance using oversampling or undersampling.
        
        Args:
            method: 'smote', 'random_over', or 'random_under'
            random_state: Random seed for reproducibility
            
        Returns:
            Resampled features and targets
        """
        if self.X is None or self.y is None:
            raise ValueError("Features and targets not prepared. Call prepare_features_targets() first.")
        
        logger.info(f"Correcting class imbalance using {method}")
        
        # Calculate class weights
        classes = np.unique(self.y)
        class_weights = compute_class_weight('balanced', classes=classes, y=self.y)
        class_weight_dict = {c: w for c, w in zip(classes, class_weights)}
        
        logger.info(f"Class weights: {class_weight_dict}")
        
        # Apply resampling
        if method == 'smote':
            resampler = SMOTE(random_state=random_state)
        elif method == 'random_over':
            resampler = RandomOverSampler(random_state=random_state)
        elif method == 'random_under':
            resampler = RandomUnderSampler(random_state=random_state)
        else:
            raise ValueError(f"Unknown resampling method: {method}")
        
        X_resampled, y_resampled = resampler.fit_resample(self.X, self.y)
        
        logger.info(f"Original class distribution: {pd.Series(self.y).value_counts()}")
        logger.info(f"Resampled class distribution: {pd.Series(y_resampled).value_counts()}")
        
        return X_resampled, y_resampled

    def collect_hard_negatives(self, confidence_threshold=0.7):
        """
        Collect high-confidence wrong predictions (hard negatives).
        
        Args:
            confidence_threshold: Minimum confidence for hard negatives
            
        Returns:
            DataFrame with hard negatives
        """
        if self.X is None or self.y is None:
            raise ValueError("Features and targets not prepared. Call prepare_features_targets() first.")
        
        logger.info(f"Collecting hard negatives with confidence threshold {confidence_threshold}")
        
        # Get predictions and probabilities
        y_pred = self.model.predict(self.X)
        y_proba = self.model.predict_proba(self.X)
        
        # Get confidence scores (max probability)
        confidence = np.max(y_proba, axis=1)
        
        # Find high-confidence wrong predictions
        wrong_mask = (y_pred != self.y)
        high_conf_mask = (confidence >= confidence_threshold)
        hard_neg_mask = wrong_mask & high_conf_mask
        
        # Create DataFrame with hard negatives
        hard_neg_indices = np.where(hard_neg_mask)[0]
        hard_neg_df = pd.DataFrame({
            'index': hard_neg_indices,
            'true_label': self.y.iloc[hard_neg_indices] if hasattr(self.y, 'iloc') else self.y[hard_neg_indices],
            'pred_label': y_pred[hard_neg_indices],
            'confidence': confidence[hard_neg_indices]
        })
        
        logger.info(f"Found {len(hard_neg_df)} hard negatives")
        
        # Save hard negatives
        os.makedirs('reports', exist_ok=True)
        hard_neg_df.to_csv('reports/hard_negatives.csv', index=False)
        logger.info("Hard negatives saved to reports/hard_negatives.csv")
        
        return hard_neg_df

    def create_ensemble(self, models, voting='soft'):
        """
        Create an ensemble of models for signal confirmation.
        
        Args:
            models: List of (name, model) tuples
            voting: 'soft' for probability averaging, 'hard' for majority vote
            
        Returns:
            Ensemble model
        """
        logger.info(f"Creating {voting} voting ensemble with {len(models)} models")
        
        # Create voting classifier
        ensemble = VotingClassifier(
            estimators=models,
            voting=voting
        )
        
        # If we have data, fit the ensemble
        if self.X is not None and self.y is not None:
            logger.info("Fitting ensemble on available data")
            ensemble.fit(self.X, self.y)
        
        return ensemble

    def add_contextual_features(self, window_sizes=[5, 10, 20]):
        """
        Add contextual features like rolling windows and trend indicators.
        
        Args:
            window_sizes: List of window sizes for rolling features
            
        Returns:
            DataFrame with added contextual features
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        logger.info(f"Adding contextual features with window sizes {window_sizes}")
        
        # Make a copy of the data
        enhanced_data = self.data.copy()
        
        # Ensure data is sorted by datetime if available
        if 'datetime' in enhanced_data.columns:
            enhanced_data['datetime'] = pd.to_datetime(enhanced_data['datetime'])
            enhanced_data = enhanced_data.sort_values('datetime')
        
        # Price and volume columns
        price_col = 'Close'
        volume_col = 'Volume' if 'Volume' in enhanced_data.columns else None
        
        # Add rolling window features
        for window in window_sizes:
            # Price momentum
            enhanced_data[f'return_{window}d'] = enhanced_data[price_col].pct_change(window)
            
            # Price volatility
            enhanced_data[f'volatility_{window}d'] = enhanced_data[price_col].pct_change().rolling(window).std()
            
            # Price trend (positive days ratio)
            enhanced_data[f'up_ratio_{window}d'] = enhanced_data[price_col].pct_change().gt(0).rolling(window).mean()
            
            # Price acceleration
            enhanced_data[f'acceleration_{window}d'] = enhanced_data[price_col].pct_change().diff().rolling(window).mean()
            
            # Volume trend
            if volume_col:
                enhanced_data[f'volume_trend_{window}d'] = enhanced_data[volume_col].pct_change().rolling(window).mean()
        
        # Add distance from moving averages
        for window in window_sizes:
            ma_col = f'MA_{window}'
            if ma_col in enhanced_data.columns:
                enhanced_data[f'dist_from_{ma_col}'] = (enhanced_data[price_col] / enhanced_data[ma_col] - 1) * 100
            else:
                # Calculate if not present
                enhanced_data[ma_col] = enhanced_data[price_col].rolling(window).mean()
                enhanced_data[f'dist_from_{ma_col}'] = (enhanced_data[price_col] / enhanced_data[ma_col] - 1) * 100
        
        # Add RSI divergence if RSI is available
        if 'RSI_14' in enhanced_data.columns:
            # Price making higher highs but RSI making lower highs = bearish divergence
            enhanced_data['RSI_slope_5d'] = enhanced_data['RSI_14'].diff(5)
            enhanced_data['price_slope_5d'] = enhanced_data[price_col].diff(5)
            enhanced_data['RSI_divergence'] = np.where(
                (enhanced_data['price_slope_5d'] > 0) & (enhanced_data['RSI_slope_5d'] < 0), 
                -1,  # Bearish divergence
                np.where(
                    (enhanced_data['price_slope_5d'] < 0) & (enhanced_data['RSI_slope_5d'] > 0),
                    1,   # Bullish divergence
                    0    # No divergence
                )
            )
        
        # Add day of week if datetime is available
        if 'datetime' in enhanced_data.columns:
            enhanced_data['day_of_week'] = enhanced_data['datetime'].dt.dayofweek
            enhanced_data['hour_of_day'] = enhanced_data['datetime'].dt.hour
        
        # Fill NaN values created by rolling windows
        enhanced_data = enhanced_data.fillna(method='bfill').fillna(0)
        
        # Log new features
        new_features = [col for col in enhanced_data.columns if col not in self.data.columns]
        logger.info(f"Added {len(new_features)} new contextual features")
        
        return enhanced_data

    def filter_predictions_by_confidence(self, confidence_thresholds=[0.5, 0.6, 0.7, 0.8, 0.9]):
        """
        Analyze prediction performance at different confidence thresholds.
        
        Args:
            confidence_thresholds: List of confidence thresholds to evaluate
            
        Returns:
            DataFrame with performance metrics at each threshold
        """
        if self.X is None or self.y is None:
            raise ValueError("Features and targets not prepared. Call prepare_features_targets() first.")
        
        logger.info(f"Filtering predictions by confidence thresholds: {confidence_thresholds}")
        
        # Get predictions and probabilities
        y_pred = self.model.predict(self.X)
        y_proba = self.model.predict_proba(self.X)
        
        # Get confidence scores (max probability)
        confidence = np.max(y_proba, axis=1)
        
        # Evaluate at different thresholds
        results = []
        
        for threshold in confidence_thresholds:
            # Filter by confidence
            mask = confidence >= threshold
            
            # Skip if no predictions meet the threshold
            if not np.any(mask):
                logger.warning(f"No predictions meet confidence threshold {threshold}")
                continue
            
            # Calculate metrics
            filtered_y_true = self.y[mask] if isinstance(self.y, np.ndarray) else self.y.iloc[mask]
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


def main():
    """Main function to run model improvement analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Improve stock prediction model')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--data-path', type=str, required=True, help='Path to the data file')
    parser.add_argument('--calibrate', action='store_true', help='Calibrate model confidence')
    parser.add_argument('--analyze-features', action='store_true', help='Analyze feature importance')
    parser.add_argument('--analyze-distributions', action='store_true', help='Analyze feature distributions')
    parser.add_argument('--correct-imbalance', action='store_true', help='Correct class imbalance')
    parser.add_argument('--collect-hard-negatives', action='store_true', help='Collect hard negatives')
    parser.add_argument('--add-context', action='store_true', help='Add contextual features')
    parser.add_argument('--filter-predictions', action='store_true', help='Filter predictions by confidence')
    parser.add_argument('--all', action='store_true', help='Run all improvement steps')
    
    args = parser.parse_args()
    
    # Create model improver
    improver = ModelImprover(args.model_path, args.data_path)
    
    # Prepare features and targets
    improver.prepare_features_targets()
    
    # Run requested analyses
    if args.all or args.calibrate:
        logger.info("Running model calibration")
        improver.calibrate_model()
    
    if args.all or args.analyze_features:
        logger.info("Running feature importance analysis")
        improver.analyze_feature_importance()
    
    if args.all or args.analyze_distributions:
        logger.info("Running feature distribution analysis")
        improver.analyze_feature_distributions()
    
    if args.all or args.correct_imbalance:
        logger.info("Running class imbalance correction")
        improver.correct_class_imbalance()
    
    if args.all or args.collect_hard_negatives:
        logger.info("Collecting hard negatives")
        improver.collect_hard_negatives()
    
    if args.all or args.add_context:
        logger.info("Adding contextual features")
        enhanced_data = improver.add_contextual_features()
        
        # Save enhanced data
        enhanced_data_path = os.path.join(os.path.dirname(args.data_path), 'enhanced_data.csv')
        enhanced_data.to_csv(enhanced_data_path, index=False)
        logger.info(f"Enhanced data saved to {enhanced_data_path}")
    
    if args.all or args.filter_predictions:
        logger.info("Filtering predictions by confidence")
        improver.filter_predictions_by_confidence()
    
    logger.info("Model improvement analysis complete")


if __name__ == "__main__":
    main()
