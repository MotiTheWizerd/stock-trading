#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Prediction Integration Module

This module provides integration between the existing stock prediction system
and the enhanced prediction components (contextual features, model calibration,
adaptive retraining).
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import importlib.util

# Configure logging
logger = logging.getLogger(__name__)

# Check if enhanced modules are available
ENHANCED_MODULES = {
    "contextual_features": False,
    "model_calibration": False,
    "adaptive_retraining": False,
    "enhanced_prediction": False
}

def check_enhanced_modules():
    """Check which enhanced modules are available."""
    for module_name in ENHANCED_MODULES:
        try:
            spec = importlib.util.find_spec(module_name)
            if spec is not None:
                ENHANCED_MODULES[module_name] = True
                logger.info(f"Enhanced module available: {module_name}")
        except ImportError:
            logger.warning(f"Enhanced module not available: {module_name}")
    
    return all(ENHANCED_MODULES.values())

def get_enhanced_predictor(model_path, confidence_threshold=0.6, 
                          enable_contextual=True, enable_calibration=True):
    """
    Get an instance of the EnhancedStockPredictor if available.
    
    Args:
        model_path: Path to the model file
        confidence_threshold: Minimum confidence threshold for predictions
        enable_contextual: Whether to enable contextual features
        enable_calibration: Whether to enable model calibration
        
    Returns:
        EnhancedStockPredictor instance or None if not available
    """
    if not check_enhanced_modules():
        logger.warning("Enhanced prediction system not fully available")
        return None
    
    try:
        from enhanced_prediction import EnhancedStockPredictor
        
        predictor = EnhancedStockPredictor(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            enable_contextual_features=enable_contextual,
            enable_calibration=enable_calibration
        )
        
        logger.info("Enhanced prediction system initialized successfully")
        return predictor
    except Exception as e:
        logger.error(f"Error initializing enhanced prediction system: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_enhanced_prediction(df, model_path, confidence_threshold=0.6, 
                           enable_contextual=True, enable_calibration=True,
                           apply_trading=True, initial_balance=1000.0):
    """
    Run enhanced prediction on the given data.
    
    Args:
        df: DataFrame with stock data
        model_path: Path to the model file
        confidence_threshold: Minimum confidence threshold for predictions
        enable_contextual: Whether to enable contextual features
        enable_calibration: Whether to enable model calibration
        apply_trading: Whether to apply trading simulation
        initial_balance: Initial balance for trading simulation
        
    Returns:
        DataFrame with predictions and trading results, or None if enhanced system not available
    """
    predictor = get_enhanced_predictor(
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        enable_contextual=enable_contextual,
        enable_calibration=enable_calibration
    )
    
    if predictor is None:
        return None
    
    try:
        # Make predictions with enhanced system
        logger.info("Making predictions with enhanced system")
        results = predictor.predict(
            df=df,
            apply_trading=apply_trading,
            initial_balance=initial_balance
        )
        
        return results
    except Exception as e:
        logger.error(f"Error making enhanced predictions: {e}")
        import traceback
        traceback.print_exc()
        return None

def add_enhanced_args(parser):
    """
    Add enhanced prediction arguments to an existing ArgumentParser.
    
    Args:
        parser: ArgumentParser instance
        
    Returns:
        Updated ArgumentParser instance
    """
    group = parser.add_argument_group('Enhanced Prediction Options')
    group.add_argument("--use-enhanced", action="store_true", 
                      help="Use enhanced prediction system if available")
    group.add_argument("--disable-contextual", action="store_true", 
                      help="Disable contextual features (enhanced mode only)")
    group.add_argument("--disable-calibration", action="store_true", 
                      help="Disable model calibration (enhanced mode only)")
    group.add_argument("--confidence-threshold", type=float, default=0.6, 
                      help="Minimum confidence threshold for predictions")
    group.add_argument("--check-retraining", action="store_true",
                      help="Check if model retraining is needed (enhanced mode only)")
    
    return parser

def print_enhanced_info():
    """Print information about the enhanced prediction system."""
    all_available = check_enhanced_modules()
    
    if all_available:
        print("\n🚀 Enhanced prediction system is fully available!")
    else:
        print("\n⚠️ Some enhanced modules are available:")
        for module, available in ENHANCED_MODULES.items():
            status = "✅" if available else "❌"
            print(f"  {status} {module}")
    
    print("\nRun with --use-enhanced flag to enable advanced features:")
    print("  --use-enhanced         Use enhanced prediction system")
    print("  --disable-contextual   Disable contextual features")
    print("  --disable-calibration  Disable model calibration")
    print("  --confidence-threshold Set minimum confidence threshold (default: 0.6)")
    print("  --check-retraining     Check if model retraining is needed")
    
    print("\nOr run the dedicated enhanced pipeline script:")
    print("  python run_enhanced_pipeline.py --ticker SYMBOL --model-path models/your_model.joblib")

if __name__ == "__main__":
    # Simple test to check if enhanced modules are available
    logging.basicConfig(level=logging.INFO)
    check_enhanced_modules()
    print_enhanced_info()
