"""
Advanced Model Training Pipeline
===============================

This module implements a comprehensive model training pipeline for trading signals
based on the specifications in docs/models/overview.md. It supports:

1. Data Loading with historical data aggregation
2. Feature Engineering integration with existing pipeline
3. Model Training with RandomForestClassifier
4. Performance Evaluation with multiple metrics
5. Model Persistence with versioning
6. Backtesting and PnL simulation

Usage:
    python train_model.py --ticker=AAPL --model=randomforest
    python train_model.py --ticker=AAPL --model=randomforest --evaluate
    python train_model.py --all-tickers --model=randomforest
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# Import project modules
from feature_engineer import FeatureEngineer
from paths import get_data_path, get_models_path
import training_pipeline as tp

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Advanced model training pipeline with evaluation and persistence."""
    
    def __init__(self, ticker: str, model_type: str = "randomforest", data_dir: str = None):
        """Initialize the model trainer.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            model_type: Type of model to train ('randomforest' supported)
            data_dir: Optional custom directory containing training data
        """
        self.ticker = ticker.upper()
        self.model_type = model_type
        self.feature_engineer = FeatureEngineer()
        self.model = None
        self.pipeline = None
        self.feature_columns = []
        
        # Set up data and model paths
        if data_dir:
            self.data_path = Path(data_dir)
        else:
            self.data_path = get_data_path(ticker)
            
        self.models_path = get_models_path(ticker)
        self.evaluation_results = {}
        
        # Ensure models directory exists
        self.models_path.mkdir(parents=True, exist_ok=True)
        
    def load_training_data(self, days_back: int = 30) -> pd.DataFrame:
        """Load and aggregate historical data for training.
        
        Args:
            days_back: Number of days to look back for data
            
        Returns:
            Merged DataFrame with data and signals
        """
        logger.info(f"🔄 Loading training data for {self.ticker}")
        
        # Collect all available data files
        data_files = []
        signals_files = []
        
        for date_dir in sorted(self.data_path.iterdir()):
            if not date_dir.is_dir():
                continue
                
            data_file = date_dir / "data.csv"
            signals_file = date_dir / "signals.csv"
            
            if data_file.exists() and signals_file.exists():
                data_files.append(data_file)
                signals_files.append(signals_file)
        
        if not data_files:
            raise FileNotFoundError(f"No data files found for {self.ticker}")
        
        # Load and concatenate data
        combined_data = []
        combined_signals = []
        
        for data_file, signals_file in zip(data_files, signals_files):
            try:
                # Load data and signals
                df_data = pd.read_csv(data_file, parse_dates=["datetime"])
                df_signals = pd.read_csv(signals_file, parse_dates=["datetime"])
                
                # Ensure ticker column exists
                if "ticker" not in df_data.columns:
                    df_data["ticker"] = self.ticker
                if "ticker" not in df_signals.columns:
                    df_signals["ticker"] = self.ticker
                
                # Merge data and signals on datetime
                df_merged = pd.merge(
                    df_data, 
                    df_signals[["datetime", "Signal", "RSI_Value"]], 
                    on="datetime", 
                    how="left"
                )
                
                # Rename Signal column to signal for consistency
                if "Signal" in df_merged.columns:
                    df_merged = df_merged.rename(columns={"Signal": "signal"})
                
                combined_data.append(df_merged)
                combined_signals.append(df_signals)
                
            except Exception as e:
                logger.warning(f"Failed to load or merge {data_file}: {e}")
                continue
        
        # Combine all data
        if not combined_data:
            raise ValueError(f"No valid data files found for {self.ticker}")
            
        # Concatenate all merged dataframes
        merged_df = pd.concat(combined_data, ignore_index=True)
        
        # Sort by datetime
        merged_df = merged_df.sort_values("datetime").reset_index(drop=True)
        
        # Log some statistics
        logger.info(f"✅ Loaded {len(merged_df)} rows of training data")
        if "signal" in merged_df.columns:
            signal_counts = merged_df["signal"].value_counts().to_dict()
            logger.info(f"✅ Signal distribution: {signal_counts}")
        
        return merged_df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for training.
        
        Args:
            df: Raw DataFrame with OHLCV and signals
            
        Returns:
            Tuple of (features_df, target_series)
        """
        logger.info("🔧 Preparing features and target")
        
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Ensure we have the signal column, if not create a dummy one
        if "signal" not in df.columns and "Signal" in df.columns:
            df["signal"] = df["Signal"]
        elif "signal" not in df.columns:
            logger.warning("No 'signal' column found in the data. Using 'NONE' for all samples.")
            df["signal"] = "NONE"
        
        # Define feature columns (exclude metadata and target)
        exclude_cols = {"datetime", "ticker", "signal", "Signal", "RSI_Value"}
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Prepare features
        X = df[feature_cols].copy()
        
        # Handle missing values in features
        X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Prepare target - convert to string and fill missing values with 'NONE'
        y = df["signal"].astype(str).fillna("NONE")
        
        # Store feature columns for later use
        self.feature_columns = feature_cols
        
        logger.info(f"✅ Prepared {len(feature_cols)} features")
        logger.info(f"✅ Target distribution:\n{y.value_counts().to_dict()}")
        return X, y
    
    def create_model(self) -> Pipeline:
        """Create the model pipeline.
        
        Returns:
            Sklearn Pipeline with preprocessing and model
        """
        logger.info(f"🤖 Creating {self.model_type} model")
        
        if self.model_type == "randomforest":
            model = RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                n_jobs=-1,
                class_weight='balanced',
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Create pipeline with preprocessing
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
        return pipeline
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model with cross-validation.
        
        Args:
            X: Feature matrix
            y: Target vector
        """
        logger.info("🚀 Training model with cross-validation")
        
        # Create model pipeline
        self.pipeline = self.create_model()
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(self.pipeline, X, y, cv=tscv, scoring='accuracy')
        
        logger.info(f"Cross-validation scores: {cv_scores}")
        logger.info(f"Average CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Train final model on all data
        self.pipeline.fit(X, y)
        
        # Store model reference
        self.model = self.pipeline.named_steps['model']
        
        logger.info("✅ Model training completed")
    
    def evaluate_model(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Evaluate model performance with comprehensive metrics.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info("📊 Evaluating model performance")
        
        if self.pipeline is None:
            raise ValueError("Model must be trained before evaluation")
        
        # Predictions
        y_pred = self.pipeline.predict(X)
        y_proba = self.pipeline.predict_proba(X)
        
        # Basic metrics
        accuracy = accuracy_score(y, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='weighted')
        
        # Classification report
        class_report = classification_report(y, y_pred, output_dict=True)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y, y_pred)
        
        # Feature importance
        feature_importance = dict(zip(
            self.feature_columns, 
            self.model.feature_importances_
        ))
        
        # Signal distribution
        signal_dist = y.value_counts(normalize=True).to_dict()
        
        # Confidence analysis
        confidence_stats = self._analyze_confidence(y_proba)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'feature_importance': feature_importance,
            'signal_distribution': signal_dist,
            'confidence_stats': confidence_stats,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
        
        self.evaluation_results = results
        logger.info("✅ Model evaluation completed")
        
        return results
    
    def _analyze_confidence(self, y_proba: np.ndarray) -> Dict:
        """Analyze prediction confidence statistics.
        
        Args:
            y_proba: Prediction probabilities
            
        Returns:
            Dictionary with confidence statistics
        """
        max_proba = np.max(y_proba, axis=1)
        
        return {
            'mean_confidence': np.mean(max_proba),
            'std_confidence': np.std(max_proba),
            'min_confidence': np.min(max_proba),
            'max_confidence': np.max(max_proba),
            'median_confidence': np.median(max_proba),
            'confidence_bins': np.histogram(max_proba, bins=10)[0].tolist()
        }
    
    def save_model(self) -> Path:
        """Save the trained model with metadata.
        
        Returns:
            Path to saved model file
        """
        if self.pipeline is None:
            raise ValueError("Model must be trained before saving")
        
        # Create model filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"model_{self.ticker}_{timestamp}.joblib"
        model_path = self.models_path / model_filename
        
        # Save model
        joblib.dump(self.pipeline, model_path)
        
        # Save metadata
        metadata = {
            'ticker': self.ticker,
            'model_type': self.model_type,
            'feature_columns': self.feature_columns,
            'training_timestamp': timestamp,
            'evaluation_results': self.evaluation_results
        }
        
        metadata_path = self.models_path / f"metadata_{self.ticker}_{timestamp}.json"
        
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"💾 Model saved: {model_path}")
        logger.info(f"📝 Metadata saved: {metadata_path}")
        
        return model_path
    
    def create_evaluation_plots(self) -> None:
        """Create and save evaluation plots."""
        if not self.evaluation_results:
            logger.warning("No evaluation results available for plotting")
            return
        
        logger.info("📈 Creating evaluation plots")
        
        # Create plots directory
        plots_dir = self.models_path / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Feature importance plot
        self._plot_feature_importance(plots_dir, timestamp)
        
        # Confusion matrix plot
        self._plot_confusion_matrix(plots_dir, timestamp)
        
        # Confidence histogram
        self._plot_confidence_histogram(plots_dir, timestamp)
        
        logger.info("✅ Evaluation plots created")
    
    def _plot_feature_importance(self, plots_dir: Path, timestamp: str) -> None:
        """Create feature importance plot."""
        importance = self.evaluation_results['feature_importance']
        
        # Sort by importance
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        features, values = zip(*sorted_features[:15])  # Top 15 features
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(features)), values)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title(f'Top 15 Feature Importance - {self.ticker}')
        plt.tight_layout()
        
        plt.savefig(plots_dir / f"feature_importance_{self.ticker}_{timestamp}.png", dpi=300)
        plt.close()
    
    def _plot_confusion_matrix(self, plots_dir: Path, timestamp: str) -> None:
        """Create confusion matrix plot."""
        conf_matrix = self.evaluation_results['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {self.ticker}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.savefig(plots_dir / f"confusion_matrix_{self.ticker}_{timestamp}.png", dpi=300)
        plt.close()
    
    def _plot_confidence_histogram(self, plots_dir: Path, timestamp: str) -> None:
        """Create confidence histogram plot."""
        y_proba = self.evaluation_results['y_proba']
        max_proba = np.max(y_proba, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.hist(max_proba, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Frequency')
        plt.title(f'Prediction Confidence Distribution - {self.ticker}')
        plt.axvline(np.mean(max_proba), color='red', linestyle='--', label=f'Mean: {np.mean(max_proba):.3f}')
        plt.legend()
        
        plt.savefig(plots_dir / f"confidence_histogram_{self.ticker}_{timestamp}.png", dpi=300)
        plt.close()
    
    def run_training_pipeline(self, evaluate: bool = True) -> None:
        """Run the complete training pipeline.
        
        Args:
            evaluate: Whether to run evaluation and create plots
        """
        logger.info(f"🚀 Starting training pipeline for {self.ticker}")
        
        # Load data
        df = self.load_training_data()
        
        # Prepare features
        X, y = self.prepare_features(df)
        
        # Train model
        self.train_model(X, y)
        
        # Evaluate model
        if evaluate:
            self.evaluate_model(X, y)
            self.create_evaluation_plots()
        
        # Save model
        model_path = self.save_model()
        
        logger.info(f"✅ Training pipeline completed for {self.ticker}")
        logger.info(f"📁 Model saved at: {model_path}")


def train_all_tickers(model_type: str = "randomforest", evaluate: bool = True, data_dir: str = None):
    """Train models for all available tickers.
        
    Args:
        model_type: Type of model to train
        evaluate: Whether to evaluate models
        data_dir: Optional base directory containing ticker subdirectories
    """
    logger.info(" Starting batch training for all tickers")
        
    # Determine the data directory
    base_dir = Path(data_dir) if data_dir else Path("data")
        
    # Get all tickers from the data directory
    tickers = [d.name for d in base_dir.iterdir() if d.is_dir() and d.name != "__pycache__"]
        
    if not tickers:
        logger.error(f" No ticker directories found in {base_dir}/")
        return
            
    logger.info(f" Found {len(tickers)} tickers: {', '.join(tickers)}")
        
    success_count = 0
    failure_count = 0
    
    for ticker in tickers:
        try:
            logger.info(f"\n{'='*50}")
            logger.info(f" Starting training for {ticker}")
            logger.info(f"{'='*50}")
                
            # If data_dir is provided, use ticker-specific subdirectory
            ticker_data_dir = base_dir / ticker if data_dir else None
            trainer = ModelTrainer(ticker, model_type, str(ticker_data_dir) if ticker_data_dir else None)
            trainer.run_training_pipeline(evaluate)
                
            logger.info(f" Completed training for {ticker}")
            success_count += 1
                
        except Exception as e:
            logger.error(f" Error training model for {ticker}: {str(e)}", exc_info=True)
            failure_count += 1
    
    logger.info("\n Batch training completed!")
    logger.info(f" Successful: {success_count}")
    logger.info(f" Failed: {failure_count}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Advanced Model Training Pipeline")
    parser.add_argument("--ticker", type=str, help="Stock ticker to train model for")
    parser.add_argument("--model", type=str, default="randomforest", choices=["randomforest"], 
                       help="Model type to train")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation and create plots")
    parser.add_argument("--all-tickers", action="store_true", help="Train models for all tickers")
    parser.add_argument("--data-dir", type=str, help="Custom directory containing training data")
    
    args = parser.parse_args()
    
    if args.all_tickers:
        train_all_tickers(args.model, args.evaluate, args.data_dir)
    elif args.ticker:
        trainer = ModelTrainer(args.ticker, args.model, args.data_dir)
        trainer.run_training_pipeline(args.evaluate)
    else:
        parser.print_help()
        return
    
    logger.info("🎯 Training pipeline completed successfully!")

if __name__ == "__main__":
    main()
