#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Prediction Integration Test

This script tests different prediction strategies and evaluates their performance
using comprehensive metrics and visualizations.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
import sys
import os
from typing import Dict, Any, Optional

# Add project root and src directory to path
project_root = Path(__file__).parent
src_dir = project_root / "src"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_dir))

from trading.models.enhanced_prediction import EnhancedStockPredictor
from trading.backtest.backtest_analyzer import BacktestAnalyzer
from results_ui import ResultsRenderer
from trading.models.enhanced_trader import EnhancedTrader

# Configure enhanced Rich logging
from trading.utils.rich_logger import setup_rich_logging

# Set up Rich-based logging with enhanced visual formatting
logger = setup_rich_logging(level=logging.INFO)

# Configure specific loggers to be less verbose
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('pandas').setLevel(logging.WARNING)
logging.getLogger('ta').setLevel(logging.WARNING)
logging.getLogger('feature_engineer').setLevel(logging.WARNING)
logging.getLogger('contextual_features').setLevel(logging.WARNING)

class PredictionTester:
    def __init__(self, ticker: str, data_dir: str = "data", models_dir: str = "models"):
        self.ticker = ticker
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir) / ticker
        self.initial_balance = 10000.0
        
        # Initialize analyzer and renderer
        self.analyzer = BacktestAnalyzer(initial_balance=self.initial_balance)
        self.renderer = ResultsRenderer()
        
    def load_data(self, data_file: Optional[str] = None) -> pd.DataFrame:
        """Load and prepare market data."""
        if data_file:
            data_path = Path(data_file)
            if not data_path.exists():
                raise FileNotFoundError(f"Data file not found: {data_path}")
            df = pd.read_csv(data_path)
            logger.info(f"Loaded data from {data_path}")
        else:
            # Try to load data from the data directory
            try:
                from predict import load_latest_data
                df = load_latest_data(self.ticker, str(self.data_dir))
                if df is None or df.empty:
                    raise ValueError(f"No data found for {self.ticker}")
                logger.info(f"Loaded latest data for {self.ticker}")
            except ImportError:
                # Fallback to direct file loading
                data_files = list((self.data_dir / self.ticker).glob("*.csv"))
                if not data_files:
                    raise FileNotFoundError(f"No data files found in {self.data_dir/self.ticker}")
                latest_file = max(data_files, key=lambda x: x.stat().st_mtime)
                df = pd.read_csv(latest_file)
                logger.info(f"Loaded {latest_file}")
                
        # Ensure required columns exist
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column not found in data: {col}")
                
        return df
    
    def find_latest_model(self) -> Path:
        """Find the latest trained model."""
        if not self.models_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.models_dir}")
            
        model_files = list(self.models_dir.glob("*.joblib"))
        if not model_files:
            raise FileNotFoundError(f"No model files found in {self.models_dir}")
            
        # Sort by modification time (newest first)
        return sorted(model_files, key=lambda f: f.stat().st_mtime, reverse=True)[0]
    
    def run_strategy(self, df: pd.DataFrame, strategy_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single strategy with enhanced trader and return metrics."""
        try:
            # Make a copy of the input data
            df = df.copy()
            
            # Initialize enhanced trader for this strategy
            trader = EnhancedTrader(
                ticker=self.ticker,
                initial_balance=self.initial_balance,
                journal_output_dir=f"trade_journals/{strategy_config['name']}"
            )
            
            if strategy_config["use_enhanced"]:
                # Use enhanced prediction with trader integration
                predictions = self._run_enhanced_strategy_with_trader(df, strategy_config, trader)
            else:
                # Use standard prediction with trader integration
                predictions = self._run_standard_strategy_with_trader(df, strategy_config, trader)
            
            if predictions is None or predictions.empty:
                logger.warning(f"No predictions generated for {strategy_config['name']}")
                return {}
                
            # Calculate metrics using the trader's performance
            price_series = df['Close'] if 'Close' in df.columns else None
            metrics = self.analyzer.calculate_metrics(predictions, price_series)
            
            # Add enhanced trader metrics
            trader_performance = trader.get_performance_summary()
            metrics.update({
                'strategy': strategy_config['name'],
                'confidence_threshold': strategy_config['confidence_threshold'],
                'trader_performance': trader_performance,
                'journal_entries': len(trader.journal.entries)
            })
            
            # Save results
            output_dir = Path("test_results") / strategy_config['name']
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save predictions
            predictions_file = output_dir / f"{self.ticker}_predictions.csv"
            predictions.to_csv(predictions_file, index=False)
            
            # Save trader results and journal
            trades_file, journal_file = trader.save_results(str(output_dir))
            
            logger.info(f"Strategy {strategy_config['name']} completed:")
            logger.info(f"  - Predictions: {predictions_file}")
            logger.info(f"  - Journal: {journal_file}")
            if trades_file:
                logger.info(f"  - Trades: {trades_file}")
            
            # Print trader summary
            trader.print_summary()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error running strategy {strategy_config['name']}: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            return {}
    
    def _run_enhanced_strategy_with_trader(self, df: pd.DataFrame, strategy_config: Dict[str, Any], trader: EnhancedTrader) -> pd.DataFrame:
        """Run enhanced prediction strategy with trader integration."""
        try:
            # Initialize enhanced predictor
            enhanced_pred = EnhancedStockPredictor(
                model_path=strategy_config.get("model_path"),
                confidence_threshold=strategy_config["confidence_threshold"],
                enable_contextual_features=strategy_config["enable_contextual"],
                enable_calibration=strategy_config["enable_calibration"]
            )
            
            # Load the model
            enhanced_pred.load_model()
            
            # Get predictions for the entire dataset
            predictions_df = enhanced_pred.predict(df, apply_trading=False)
            
            if predictions_df is None or predictions_df.empty:
                logger.warning("No predictions generated from enhanced model")
                return pd.DataFrame()
            
            # Process each prediction with the trader
            results = []
            for idx, pred_row in predictions_df.iterrows():
                signal = pred_row.get('prediction', 'NONE')
                confidence = pred_row.get('confidence', 0.0)
                risk_score = pred_row.get('risk_score')
                
                # Apply confidence threshold
                if confidence < strategy_config["confidence_threshold"]:
                    signal = 'NONE'
                
                # Get corresponding market data
                market_data = df.iloc[idx] if idx < len(df) else df.iloc[-1]
                
                # Process with trader (includes journal logging)
                trade_result = trader.process_signal(
                    data_row=market_data,
                    signal=signal,
                    confidence=confidence,
                    risk_score=risk_score,
                    model_metadata={
                        'model_type': 'enhanced',
                        'contextual_enabled': strategy_config["enable_contextual"],
                        'calibration_enabled': strategy_config["enable_calibration"]
                    }
                )
                
                # Combine prediction and trade results
                result_row = {
                    'timestamp': idx,
                    'prediction': signal,
                    'confidence': confidence,
                    'risk_score': risk_score,
                    'position_taken': trade_result['position_action'],
                    'trade_executed': trade_result['trade_result'].get('executed', False),
                    'portfolio_value': trade_result['portfolio_value'],
                    'price': market_data.get('Close', 0),
                    'balance': trader.current_balance,
                    'position': trader.position
                }
                
                # Add original prediction columns
                for col in pred_row.index:
                    if col not in result_row:
                        result_row[col] = pred_row[col]
                
                # Add original market data columns
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    if col in market_data:
                        result_row[f'market_{col}'] = market_data[col]
                
                results.append(result_row)
            
            return pd.DataFrame(results)
            
        except Exception as e:
            logger.error(f"Error in enhanced strategy: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            return pd.DataFrame()
    
    def _run_standard_strategy_with_trader(self, df: pd.DataFrame, strategy_config: Dict[str, Any], trader: EnhancedTrader) -> pd.DataFrame:
        """Run standard prediction strategy with trader integration."""
        try:
            from predict import StockPredictor
            
            # Initialize standard predictor
            predictor = StockPredictor(self.ticker, model_dir=str(self.models_dir))
            
            # Get predictions for all data
            predictions_df = predictor.predict(df)
            
            if predictions_df is None or predictions_df.empty:
                return pd.DataFrame()
            
            # Process each prediction with the trader
            results = []
            for idx, row in predictions_df.iterrows():
                signal = row.get('prediction', 'NONE')
                confidence = row.get('confidence', 0.0)
                
                # Apply confidence threshold
                if confidence < strategy_config["confidence_threshold"]:
                    signal = 'NONE'
                
                # Get corresponding market data
                market_data = df.iloc[idx] if idx < len(df) else df.iloc[-1]
                
                # Process with trader (includes journal logging)
                trade_result = trader.process_signal(
                    data_row=market_data,
                    signal=signal,
                    confidence=confidence,
                    risk_score=None,  # Standard predictor doesn't provide risk scores
                    model_metadata={'model_type': 'standard'}
                )
                
                # Combine prediction and trade results
                result_row = {
                    'timestamp': idx,
                    'prediction': signal,
                    'confidence': confidence,
                    'risk_score': None,
                    'position_taken': trade_result['position_action'],
                    'trade_executed': trade_result['trade_result'].get('executed', False),
                    'portfolio_value': trade_result['portfolio_value'],
                    'price': market_data.get('Close', 0),
                    'balance': trader.current_balance,
                    'position': trader.position
                }
                
                # Add original prediction columns
                for col in row.index:
                    if col not in result_row:
                        result_row[col] = row[col]
                
                results.append(result_row)
            
            return pd.DataFrame(results)
            
        except Exception as e:
            logger.error(f"Error in standard strategy: {str(e)}")
            return pd.DataFrame()

def main():
    """Main function to run the strategy comparison."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test and compare prediction strategies")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol")
    parser.add_argument("--model", type=str, help="Model file to use (default: latest model)")
    parser.add_argument("--data-file", type=str, help="Data file to use for testing")
    parser.add_argument("--output-dir", type=str, default="test_results", 
                       help="Output directory for test results")
    
    args = parser.parse_args()
    
    try:
        # Initialize tester
        tester = PredictionTester(
            ticker=args.ticker,
            data_dir="data",
            models_dir="models"
        )
        
        # Load data
        df = tester.load_data(args.data_file)
        
        # Find model if not specified
        model_path = args.model if args.model else tester.find_latest_model()
        logger.info(f"Using model: {model_path}")
        
        # Define test configurations
        test_configs = [
            {
                "name": "standard_prediction",
                "use_enhanced": False,
                "model_path": model_path,
                "confidence_threshold": 0.6,
                "enable_contextual": False,
                "enable_calibration": False
            },
            {
                "name": "enhanced_basic",
                "use_enhanced": True,
                "model_path": model_path,
                "confidence_threshold": 0.6,
                "enable_contextual": True,
                "enable_calibration": True
            },
            {
                "name": "enhanced_no_contextual",
                "use_enhanced": True,
                "model_path": model_path,
                "confidence_threshold": 0.6,
                "enable_contextual": False,
                "enable_calibration": True
            },
            {
                "name": "enhanced_no_calibration",
                "use_enhanced": True,
                "model_path": model_path,
                "confidence_threshold": 0.6,
                "enable_contextual": True,
                "enable_calibration": False
            },
            {
                "name": "enhanced_high_confidence",
                "use_enhanced": True,
                "model_path": model_path,
                "confidence_threshold": 0.8,
                "enable_contextual": True,
                "enable_calibration": True
            }
        ]
        
        # Run all strategies
        all_results = {}
        for config in test_configs:
            logger.info(f"\n{'='*80}")
            logger.info(f"Running strategy: {config['name']}")
            logger.info(f"Configuration: {config}")
            
            metrics = tester.run_strategy(df, config)
            if metrics:
                all_results[config['name']] = metrics
                logger.info(f"Completed {config['name']}")
            else:
                logger.warning(f"Skipping {config['name']} - no results")
        
        # Display results
        if all_results:
            tester.renderer.display_results(
                all_results,
                strategy_names=list(all_results.keys()),
                title=f"Strategy Comparison - {args.ticker}"
            )
        else:
            logger.error("No valid results to display")
            return 1
            
        return 0
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())
