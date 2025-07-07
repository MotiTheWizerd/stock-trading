#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run Enhanced Stock Prediction Pipeline

This script demonstrates the full enhanced prediction pipeline:
1. Load and prepare stock data
2. Add contextual features
3. Calibrate the model
4. Make predictions with confidence filtering
5. Apply trading simulation
6. Visualize results
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse

# Import custom modules
from enhanced_prediction import EnhancedStockPredictor
from model_calibration import ModelCalibrator
from contextual_features import ContextualFeatureGenerator
from model_trainer import ModelTrainer
from adaptive_retraining import AdaptiveRetrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_stock_data(ticker, data_dir='data'):
    """
    Load stock data for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        data_dir: Directory with stock data
        
    Returns:
        DataFrame with stock data
    """
    data_path = os.path.join(data_dir, ticker, 'data.csv')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    logger.info(f"Loading data for {ticker} from {data_path}")
    df = pd.read_csv(data_path)
    
    # Convert datetime column if present
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    
    return df


def prepare_model(model_path, X_train=None, y_train=None, calibrate=True):
    """
    Prepare and optionally calibrate the model.
    
    Args:
        model_path: Path to the model file
        X_train: Training features for calibration
        y_train: Training targets for calibration
        calibrate: Whether to calibrate the model
        
    Returns:
        ModelCalibrator instance
    """
    logger.info(f"Preparing model from {model_path}")
    
    # Create calibrator
    calibrator = ModelCalibrator(model_path)
    
    # Calibrate if requested and training data provided
    if calibrate and X_train is not None and y_train is not None:
        logger.info("Calibrating model")
        calibrator.calibrate_model(X_train, y_train, method='sigmoid')
    
    return calibrator


def visualize_results(results_df, output_dir='reports'):
    """
    Visualize prediction and trading results.
    
    Args:
        results_df: DataFrame with prediction and trading results
        output_dir: Directory to save visualizations
        
    Returns:
        Dictionary with paths to saved visualizations
    """
    logger.info("Visualizing results")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store visualization paths
    viz_paths = {}
    
    # 1. Plot price and predictions
    if all(col in results_df.columns for col in ['datetime', 'Close', 'prediction']):
        plt.figure(figsize=(12, 6))
        
        # Plot price
        plt.plot(results_df['datetime'], results_df['Close'], label='Price')
        
        # Plot buy signals
        buy_signals = results_df[results_df['prediction'] == 'BUY']
        plt.scatter(buy_signals['datetime'], buy_signals['Close'], 
                   marker='^', color='green', s=100, label='BUY')
        
        # Plot sell signals
        sell_signals = results_df[results_df['prediction'] == 'SELL']
        plt.scatter(sell_signals['datetime'], sell_signals['Close'], 
                   marker='v', color='red', s=100, label='SELL')
        
        plt.title('Stock Price and Predictions')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        price_plot_path = os.path.join(output_dir, 'price_predictions.png')
        plt.savefig(price_plot_path)
        viz_paths['price_plot'] = price_plot_path
    
    # 2. Plot trading equity curve
    if all(col in results_df.columns for col in ['datetime', 'running_balance']):
        plt.figure(figsize=(12, 6))
        
        # Plot equity curve
        plt.plot(results_df['datetime'], results_df['running_balance'], label='Equity')
        
        # Add buy/sell markers if available
        if 'trade_action' in results_df.columns:
            # Buy points
            buy_points = results_df[results_df['trade_action'] == 'BUY']
            if not buy_points.empty:
                plt.scatter(buy_points['datetime'], buy_points['running_balance'], 
                           marker='^', color='green', s=100, label='BUY')
            
            # Sell points
            sell_points = results_df[results_df['trade_action'] == 'SELL']
            if not sell_points.empty:
                plt.scatter(sell_points['datetime'], sell_points['running_balance'], 
                           marker='v', color='red', s=100, label='SELL')
        
        plt.title('Trading Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Equity ($)')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        equity_plot_path = os.path.join(output_dir, 'equity_curve.png')
        plt.savefig(equity_plot_path)
        viz_paths['equity_plot'] = equity_plot_path
    
    # 3. Plot confidence distribution
    if 'confidence' in results_df.columns:
        plt.figure(figsize=(10, 6))
        
        # Plot histogram of confidence scores
        plt.hist(results_df['confidence'], bins=20, alpha=0.7)
        
        plt.title('Prediction Confidence Distribution')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.grid(True)
        
        # Save plot
        conf_plot_path = os.path.join(output_dir, 'confidence_distribution.png')
        plt.savefig(conf_plot_path)
        viz_paths['confidence_plot'] = conf_plot_path
    
    # 4. Plot drawdown
    if 'drawdown' in results_df.columns:
        plt.figure(figsize=(12, 6))
        
        # Plot drawdown
        plt.plot(results_df['datetime'], results_df['drawdown'], color='red')
        
        plt.title('Equity Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True)
        
        # Save plot
        dd_plot_path = os.path.join(output_dir, 'drawdown.png')
        plt.savefig(dd_plot_path)
        viz_paths['drawdown_plot'] = dd_plot_path
    
    logger.info(f"Saved {len(viz_paths)} visualizations to {output_dir}")
    
    return viz_paths


def run_pipeline(ticker, model_path, data_dir='data', output_dir='reports',
               confidence_threshold=0.7, initial_balance=1000.0,
               enable_contextual=True, enable_calibration=True):
    """
    Run the full enhanced prediction pipeline.
    
    Args:
        ticker: Stock ticker symbol
        model_path: Path to the model file
        data_dir: Directory with stock data
        output_dir: Directory to save results
        confidence_threshold: Confidence threshold for predictions
        initial_balance: Initial balance for trading simulation
        enable_contextual: Whether to enable contextual features
        enable_calibration: Whether to enable model calibration
        
    Returns:
        Dictionary with results
    """
    logger.info(f"Running enhanced pipeline for {ticker}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load stock data
    df = load_stock_data(ticker, data_dir)
    
    # Step 2: Create enhanced predictor
    predictor = EnhancedStockPredictor(
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        enable_contextual_features=enable_contextual,
        enable_calibration=enable_calibration
    )
    
    # Step 3: Make predictions
    results_df = predictor.predict(
        df=df,
        apply_trading=True,
        initial_balance=initial_balance
    )
    
    # Step 4: Save results
    results_path = os.path.join(output_dir, f"{ticker}_enhanced_predictions.csv")
    results_df.to_csv(results_path, index=False)
    
    # Step 5: Visualize results
    viz_paths = visualize_results(results_df, output_dir)
    
    # Step 6: Check if retraining is needed
    needs_retraining = False
    if 'prediction' in results_df.columns and 'signal_quality' in results_df.columns:
        # Convert signal quality to binary target
        results_df['correct'] = results_df['signal_quality'].apply(
            lambda x: 1 if x == '✅' else 0 if x == '❌' else np.nan
        )
        
        # Check for concept drift
        needs_retraining = predictor.check_for_retraining(
            df=results_df.dropna(subset=['correct']),
            target_col='correct'
        )
    
    # Prepare summary
    summary = {
        'ticker': ticker,
        'data_points': len(df),
        'predictions': {
            'total': len(results_df),
            'buy': len(results_df[results_df['prediction'] == 'BUY']),
            'sell': len(results_df[results_df['prediction'] == 'SELL']),
            'none': len(results_df[results_df['prediction'] == 'NONE'])
        },
        'trading': {
            'initial_balance': initial_balance,
            'final_balance': results_df['running_balance'].iloc[-1] if 'running_balance' in results_df.columns else None,
            'profit': results_df['running_balance'].iloc[-1] - initial_balance if 'running_balance' in results_df.columns else None,
            'profit_pct': ((results_df['running_balance'].iloc[-1] / initial_balance) - 1) * 100 if 'running_balance' in results_df.columns else None,
            'max_drawdown': results_df['max_drawdown'].max() if 'max_drawdown' in results_df.columns else None
        },
        'needs_retraining': needs_retraining,
        'results_path': results_path,
        'visualization_paths': viz_paths
    }
    
    # Print summary
    print("\n=== Enhanced Prediction Pipeline Summary ===")
    print(f"Ticker: {summary['ticker']}")
    print(f"Data points: {summary['data_points']}")
    print(f"\nPredictions:")
    print(f"  Total: {summary['predictions']['total']}")
    print(f"  BUY signals: {summary['predictions']['buy']}")
    print(f"  SELL signals: {summary['predictions']['sell']}")
    print(f"  NONE signals: {summary['predictions']['none']}")
    
    if summary['trading']['final_balance']:
        print(f"\nTrading Results:")
        print(f"  Initial balance: ${summary['trading']['initial_balance']:.2f}")
        print(f"  Final balance: ${summary['trading']['final_balance']:.2f}")
        print(f"  Profit/Loss: ${summary['trading']['profit']:.2f} ({summary['trading']['profit_pct']:.2f}%)")
        
        if summary['trading']['max_drawdown']:
            print(f"  Maximum drawdown: {summary['trading']['max_drawdown']:.2f}%")
    
    print(f"\nModel Status:")
    print(f"  Needs retraining: {'Yes' if summary['needs_retraining'] else 'No'}")
    
    print(f"\nResults saved to: {summary['results_path']}")
    print(f"Visualizations saved to: {output_dir}")
    
    return summary


def main():
    """Main function to run the enhanced prediction pipeline."""
    parser = argparse.ArgumentParser(description='Run enhanced stock prediction pipeline')
    parser.add_argument('--ticker', type=str, required=True, help='Stock ticker symbol')
    parser.add_argument('--model-path', type=str, required=True, help='Path to model file')
    parser.add_argument('--data-dir', type=str, default='data', help='Directory with stock data')
    parser.add_argument('--output-dir', type=str, default='reports', help='Directory to save results')
    parser.add_argument('--confidence-threshold', type=float, default=0.7, 
                       help='Confidence threshold for predictions')
    parser.add_argument('--initial-balance', type=float, default=1000.0,
                       help='Initial balance for trading simulation')
    parser.add_argument('--disable-contextual', action='store_true',
                       help='Disable contextual features')
    parser.add_argument('--disable-calibration', action='store_true',
                       help='Disable model calibration')
    
    args = parser.parse_args()
    
    # Run pipeline
    run_pipeline(
        ticker=args.ticker,
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        confidence_threshold=args.confidence_threshold,
        initial_balance=args.initial_balance,
        enable_contextual=not args.disable_contextual,
        enable_calibration=not args.disable_calibration
    )


if __name__ == "__main__":
    main()
