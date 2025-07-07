#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for the enhanced prediction integration.

This script tests the integration between the existing stock prediction system
and the enhanced prediction components.
"""

import os
import sys
import logging
import pandas as pd
from pathlib import Path
import argparse

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Reduced from INFO to WARNING to reduce verbosity
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Configure specific loggers to be less verbose
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('pandas').setLevel(logging.WARNING)
logging.getLogger('ta').setLevel(logging.WARNING)
logging.getLogger('feature_engineer').setLevel(logging.WARNING)
logging.getLogger('contextual_features').setLevel(logging.WARNING)

# Custom formatter to remove timestamps
class CleanFormatter(logging.Formatter):
    def format(self, record):
        if record.levelno >= logging.WARNING:
            return f"[!] {record.msg}"
        return record.msg

# Apply custom formatter to root logger
for handler in logging.root.handlers:
    handler.setFormatter(CleanFormatter())

def main():
    """Main function to test the enhanced prediction integration."""
    parser = argparse.ArgumentParser(description="Test enhanced prediction integration")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol")
    parser.add_argument("--model", type=str, help="Model file to use (default: latest model)")
    parser.add_argument("--data-file", type=str, help="Data file to use for testing")
    parser.add_argument("--output-dir", type=str, default="test_results", help="Output directory for test results")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if enhanced modules are available
    try:
        import enhanced_integration
        logger.info("Enhanced integration module found")
        
        # Check which enhanced modules are available
        modules_available = enhanced_integration.check_enhanced_modules()
        for module, available in enhanced_integration.ENHANCED_MODULES.items():
            status = "✅" if available else "❌"
            logger.info(f"{status} {module}")
        
        if not all(enhanced_integration.ENHANCED_MODULES.values()):
            logger.warning("Not all enhanced modules are available")
    except ImportError:
        logger.error("Enhanced integration module not found")
        return 1
    
    # Load data
    if args.data_file:
        data_path = Path(args.data_file)
        if not data_path.exists():
            logger.error(f"Data file not found: {data_path}")
            return 1
        
        df = pd.read_csv(data_path)
        logger.info(f"Loaded data from {data_path}")
    else:
        # Try to load data from the data directory
        try:
            from predict import load_latest_data
            df = load_latest_data(args.ticker, "data")
            if df is None or df.empty:
                logger.error(f"No data found for {args.ticker}")
                return 1
            logger.info(f"Loaded latest data for {args.ticker}")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    # Determine model path
    if args.model:
        model_path = Path(args.model)
    else:
        # Try to find the latest model
        model_dir = Path("models") / args.ticker
        if not model_dir.exists():
            logger.error(f"Model directory not found: {model_dir}")
            return 1
        
        model_files = list(model_dir.glob("*.joblib"))
        if not model_files:
            logger.error(f"No model files found in {model_dir}")
            return 1
        
        # Sort by modification time (newest first)
        model_path = sorted(model_files, key=lambda f: f.stat().st_mtime, reverse=True)[0]
    
    logger.info(f"Using model: {model_path}")
    
    # Run tests for different configurations
    test_configs = [
        {
            "name": "standard_prediction",
            "use_enhanced": False,
            "confidence_threshold": 0.6,
            "enable_contextual": False,
            "enable_calibration": False
        },
        {
            "name": "enhanced_basic",
            "use_enhanced": True,
            "confidence_threshold": 0.6,
            "enable_contextual": True,
            "enable_calibration": True
        },
        {
            "name": "enhanced_no_contextual",
            "use_enhanced": True,
            "confidence_threshold": 0.6,
            "enable_contextual": False,
            "enable_calibration": True
        },
        {
            "name": "enhanced_no_calibration",
            "use_enhanced": True,
            "confidence_threshold": 0.6,
            "enable_contextual": True,
            "enable_calibration": False
        },
        {
            "name": "enhanced_high_confidence",
            "use_enhanced": True,
            "confidence_threshold": 0.8,
            "enable_contextual": True,
            "enable_calibration": True
        }
    ]
    
    results = {}
    
    for config in test_configs:
        logger.info(f"Running test: {config['name']}")
        
        if config["use_enhanced"]:
            # Use enhanced prediction
            predictions = enhanced_integration.run_enhanced_prediction(
                df=df,
                model_path=model_path,
                confidence_threshold=config["confidence_threshold"],
                enable_contextual=config["enable_contextual"],
                enable_calibration=config["enable_calibration"],
                apply_trading=True,
                initial_balance=1000.0
            )
            
            if predictions is None:
                logger.error(f"Enhanced prediction failed for {config['name']}")
                continue
        else:
            # Use standard prediction
            try:
                from predict import StockPredictor, apply_trader
                
                predictor = StockPredictor(args.ticker, model_dir=str(model_path.parent))
                predictions = predictor.predict(df)
                
                # Filter by confidence
                predictions.loc[predictions['confidence'] < config["confidence_threshold"], 'prediction'] = 'NONE'
                
                # Apply trading engine
                predictions = apply_trader(predictions, initial_balance=1000.0)
            except Exception as e:
                logger.error(f"Standard prediction failed for {config['name']}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Save predictions
        output_file = output_dir / f"{args.ticker}_{config['name']}_predictions.csv"
        predictions.to_csv(output_file, index=False)
        logger.info(f"Saved predictions to {output_file}")
        
        # Calculate metrics
        metrics = {
            "total_predictions": len(predictions),
            "buy_signals": len(predictions[predictions['prediction'] == 'BUY']),
            "sell_signals": len(predictions[predictions['prediction'] == 'SELL']),
            "none_signals": len(predictions[predictions['prediction'] == 'NONE']),
        }
        
        # Add trading metrics if available
        if 'final_balance' in predictions.columns:
            initial_balance = 1000.0
            final_balance = predictions['final_balance'].iloc[-1]
            profit = final_balance - initial_balance
            profit_pct = (profit / initial_balance) * 100
            
            metrics.update({
                "initial_balance": initial_balance,
                "final_balance": final_balance,
                "profit": profit,
                "profit_pct": profit_pct
            })
            
            if 'max_drawdown' in predictions.columns:
                metrics["max_drawdown"] = predictions['max_drawdown'].max()
        
        results[config['name']] = metrics
    
    # Print comparison
    print("\n=== Test Results Comparison ===")
    print("-" * 80)
    
    # Print header
    header = ["Metric"]
    for config in test_configs:
        header.append(config['name'])
    print(" | ".join(header))
    print("-" * 80)
    
    # Print metrics
    all_metrics = set()
    for metrics in results.values():
        all_metrics.update(metrics.keys())
    
    for metric in sorted(all_metrics):
        row = [metric]
        for config in test_configs:
            if config['name'] in results and metric in results[config['name']]:
                value = results[config['name']][metric]
                if isinstance(value, float):
                    row.append(f"{value:.2f}")
                else:
                    row.append(str(value))
            else:
                row.append("N/A")
        print(" | ".join(row))
    
    print("-" * 80)
    print("Test completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())
