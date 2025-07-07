#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Adaptive Retraining Module for Stock Trading Models

This module implements:
1. Concept drift detection
2. Sliding window retraining
3. Performance monitoring
4. Automated model updates
5. Model versioning and history tracking
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pickle
import joblib
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class AdaptiveRetrainer:
    """
    A class to implement adaptive retraining for stock prediction models.
    """
    
    def __init__(self, model_dir='models', history_file=None):
        """
        Initialize the AdaptiveRetrainer.
        
        Args:
            model_dir: Directory to store models
            history_file: File to store model history
        """
        self.model_dir = model_dir
        self.history_file = history_file or os.path.join(model_dir, 'model_history.json')
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Load model history if it exists
        self.model_history = self._load_model_history()
    
    def _load_model_history(self):
        """
        Load model history from file.
        
        Returns:
            Dictionary with model history
        """
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load model history: {e}")
                return {'models': []}
        else:
            return {'models': []}
    
    def _save_model_history(self):
        """
        Save model history to file.
        """
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.model_history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save model history: {e}")
    
    def detect_concept_drift(self, model, X, y, window_size=100, threshold=0.1):
        """
        Detect concept drift in model performance.
        
        Args:
            model: Trained model
            X: Features
            y: Targets
            window_size: Size of sliding window
            threshold: Performance drop threshold to trigger retraining
            
        Returns:
            Boolean indicating whether concept drift was detected
        """
        logger.info(f"Detecting concept drift with window size {window_size}")
        
        # Check if we have enough data
        if len(X) < window_size * 2:
            logger.warning(f"Not enough data for concept drift detection (need {window_size * 2}, got {len(X)})")
            return False
        
        # Get predictions
        y_pred = model.predict(X)
        
        # Calculate performance in sliding windows
        performances = []
        
        for i in range(0, len(X) - window_size + 1, window_size // 2):  # 50% overlap
            # Get window
            end_idx = min(i + window_size, len(X))
            window_y = y[i:end_idx]
            window_pred = y_pred[i:end_idx]
            
            # Calculate performance metrics
            accuracy = accuracy_score(window_y, window_pred)
            
            # Store performance
            performances.append({
                'start_idx': i,
                'end_idx': end_idx,
                'accuracy': accuracy
            })
        
        # Check for performance drop
        if len(performances) < 2:
            logger.warning("Not enough windows for concept drift detection")
            return False
        
        # Calculate moving average of performance
        window_count = 3
        if len(performances) >= window_count:
            # Calculate moving average
            moving_avg = []
            for i in range(len(performances) - window_count + 1):
                avg_acc = np.mean([p['accuracy'] for p in performances[i:i+window_count]])
                moving_avg.append(avg_acc)
            
            # Check for significant drop
            if len(moving_avg) >= 2:
                first_avg = moving_avg[0]
                last_avg = moving_avg[-1]
                
                perf_drop = first_avg - last_avg
                
                logger.info(f"Performance change: {perf_drop:.4f} (first: {first_avg:.4f}, last: {last_avg:.4f})")
                
                if perf_drop > threshold:
                    logger.warning(f"Concept drift detected! Performance drop: {perf_drop:.4f}")
                    return True
        
        logger.info("No significant concept drift detected")
        return False
    
    def retrain_model(self, model_trainer, X, y, model_name, ticker=None, 
                     test_size=0.2, random_state=42):
        """
        Retrain model with new data.
        
        Args:
            model_trainer: ModelTrainer instance
            X: Features
            y: Targets
            model_name: Base name for the model
            ticker: Stock ticker symbol
            test_size: Test split ratio
            random_state: Random seed
            
        Returns:
            Dictionary with retrained model and performance metrics
        """
        logger.info(f"Retraining model {model_name}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Train model
        model = model_trainer.train_model(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
        
        # Generate timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Generate model version
        version = len([m for m in self.model_history['models'] if m['base_name'] == model_name]) + 1
        
        # Generate model filename
        if ticker:
            model_filename = f"{model_name}_{ticker}_v{version}_{timestamp}.joblib"
        else:
            model_filename = f"{model_name}_v{version}_{timestamp}.joblib"
        
        model_path = os.path.join(self.model_dir, model_filename)
        
        # Save model
        joblib.dump(model, model_path)
        
        # Update model history
        model_info = {
            'base_name': model_name,
            'ticker': ticker,
            'version': version,
            'timestamp': timestamp,
            'path': model_path,
            'metrics': metrics,
            'data_size': len(X),
            'feature_count': X.shape[1]
        }
        
        self.model_history['models'].append(model_info)
        self._save_model_history()
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Performance: accuracy={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}")
        
        return {
            'model': model,
            'path': model_path,
            'metrics': metrics,
            'info': model_info
        }
    
    def get_best_model(self, model_name, ticker=None, metric='accuracy'):
        """
        Get the best model based on a performance metric.
        
        Args:
            model_name: Base name of the model
            ticker: Stock ticker symbol
            metric: Metric to use for comparison
            
        Returns:
            Dictionary with best model info
        """
        # Filter models by name and ticker
        filtered_models = [
            m for m in self.model_history['models'] 
            if m['base_name'] == model_name and 
            (ticker is None or m['ticker'] == ticker)
        ]
        
        if not filtered_models:
            logger.warning(f"No models found for {model_name}" + (f" and {ticker}" if ticker else ""))
            return None
        
        # Sort by metric
        sorted_models = sorted(filtered_models, key=lambda m: m['metrics'].get(metric, 0), reverse=True)
        
        best_model_info = sorted_models[0]
        
        logger.info(f"Best model: {os.path.basename(best_model_info['path'])}")
        logger.info(f"Performance: {best_model_info['metrics']}")
        
        return best_model_info
    
    def load_model(self, model_path):
        """
        Load model from file.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Loaded model
        """
        logger.info(f"Loading model from {model_path}")
        try:
            return joblib.load(model_path)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def sliding_window_retrain(self, model_trainer, data_loader, model_name, ticker=None,
                             window_size=30, step_size=7, min_samples=1000,
                             feature_cols=None, target_col='prediction'):
        """
        Retrain model using sliding window approach.
        
        Args:
            model_trainer: ModelTrainer instance
            data_loader: Function to load data for a date range
            model_name: Base name for the model
            ticker: Stock ticker symbol
            window_size: Window size in days
            step_size: Step size in days
            min_samples: Minimum number of samples required for training
            feature_cols: Feature column names
            target_col: Target column name
            
        Returns:
            List of retrained models
        """
        logger.info(f"Starting sliding window retraining for {model_name}")
        
        # Get current date
        end_date = datetime.now()
        
        # Calculate start date
        start_date = end_date - timedelta(days=window_size)
        
        retrained_models = []
        
        # Retrain in sliding windows
        while True:
            # Format dates
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            logger.info(f"Training window: {start_str} to {end_str}")
            
            try:
                # Load data for window
                df = data_loader(start_date=start_str, end_date=end_str, ticker=ticker)
                
                # Check if we have enough data
                if len(df) < min_samples:
                    logger.warning(f"Not enough data for training: {len(df)} < {min_samples}")
                    break
                
                # Determine feature columns if not provided
                if feature_cols is None:
                    # Exclude known non-feature columns
                    exclude_cols = [target_col, 'datetime', 'Open', 'High', 'Low', 'Close', 
                                  'Volume', 'Adj Close', 'prediction', 'confidence']
                    feature_cols = [col for col in df.columns if col not in exclude_cols]
                
                # Extract features and target
                X = df[feature_cols]
                y = df[target_col]
                
                # Retrain model
                retrained = self.retrain_model(
                    model_trainer=model_trainer,
                    X=X,
                    y=y,
                    model_name=model_name,
                    ticker=ticker
                )
                
                retrained_models.append(retrained)
                
                # Move window
                end_date = start_date - timedelta(days=1)
                start_date = end_date - timedelta(days=window_size)
                
            except Exception as e:
                logger.error(f"Error during retraining: {e}")
                break
        
        logger.info(f"Sliding window retraining complete. Retrained {len(retrained_models)} models.")
        
        return retrained_models
    
    def plot_model_performance_history(self, model_name, ticker=None, metric='accuracy'):
        """
        Plot performance history of models.
        
        Args:
            model_name: Base name of the model
            ticker: Stock ticker symbol
            metric: Metric to plot
            
        Returns:
            Path to saved plot
        """
        # Filter models by name and ticker
        filtered_models = [
            m for m in self.model_history['models'] 
            if m['base_name'] == model_name and 
            (ticker is None or m['ticker'] == ticker)
        ]
        
        if not filtered_models:
            logger.warning(f"No models found for {model_name}" + (f" and {ticker}" if ticker else ""))
            return None
        
        # Sort by timestamp
        sorted_models = sorted(filtered_models, key=lambda m: m['timestamp'])
        
        # Extract timestamps and metrics
        timestamps = [datetime.strptime(m['timestamp'], '%Y%m%d_%H%M%S') for m in sorted_models]
        metrics = [m['metrics'].get(metric, 0) for m in sorted_models]
        
        # Create plot
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, metrics, 'o-', label=metric)
        plt.xlabel('Date')
        plt.ylabel(metric.capitalize())
        plt.title(f'{metric.capitalize()} History for {model_name}' + (f' ({ticker})' if ticker else ''))
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        os.makedirs('reports', exist_ok=True)
        plot_path = f'reports/{model_name}_performance_history.png'
        plt.savefig(plot_path)
        
        logger.info(f"Performance history plot saved to {plot_path}")
        
        return plot_path


def main():
    """Main function to demonstrate adaptive retraining."""
    import argparse
    from model_trainer import ModelTrainer
    
    parser = argparse.ArgumentParser(description='Adaptive retraining for stock prediction models')
    parser.add_argument('--model-dir', type=str, default='models', help='Directory to store models')
    parser.add_argument('--data-dir', type=str, default='data', help='Directory with stock data')
    parser.add_argument('--ticker', type=str, help='Stock ticker symbol')
    parser.add_argument('--model-name', type=str, default='stock_predictor', help='Base name for the model')
    parser.add_argument('--window-size', type=int, default=30, help='Window size in days')
    parser.add_argument('--step-size', type=int, default=7, help='Step size in days')
    
    args = parser.parse_args()
    
    # Create retrainer
    retrainer = AdaptiveRetrainer(model_dir=args.model_dir)
    
    # Define data loader function
    def load_data(start_date, end_date, ticker):
        """Load data for a date range."""
        if ticker:
            data_path = os.path.join(args.data_dir, ticker, 'data.csv')
        else:
            # Find any data file
            for root, _, files in os.walk(args.data_dir):
                for file in files:
                    if file == 'data.csv':
                        data_path = os.path.join(root, file)
                        break
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Load data
        df = pd.read_csv(data_path)
        
        # Convert datetime column
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        elif 'Date' in df.columns:
            df['datetime'] = pd.to_datetime(df['Date'])
        else:
            raise ValueError("No datetime column found in data")
        
        # Filter by date range
        df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]
        
        return df
    
    # Create model trainer
    model_trainer = ModelTrainer()
    
    # Perform sliding window retraining
    retrainer.sliding_window_retrain(
        model_trainer=model_trainer,
        data_loader=load_data,
        model_name=args.model_name,
        ticker=args.ticker,
        window_size=args.window_size,
        step_size=args.step_size
    )
    
    # Plot performance history
    retrainer.plot_model_performance_history(args.model_name, args.ticker)


if __name__ == "__main__":
    main()
