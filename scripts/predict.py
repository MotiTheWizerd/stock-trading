#!/usr/bin/env python
"""
Stock Signal Predictor

This script uses a trained model to predict stock signals (BUY, SELL, NONE)
based on technical indicators.
It can also use the enhanced prediction system if available.
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
import argparse

# Configure logging
import logging

# Try to import enhanced integration module
try:
    import enhanced_integration
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False



logger = logging.getLogger(__name__)


def _norm(col: str) -> str:
    """Normalize a column name for loose matching (lower, strip, no punctuation)."""
    return (
        str(col).lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )

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
            ticker: Stock ticker symbol or model path
            model_dir: Directory containing model files (default: models/<ticker>)
        """
        # Check if ticker is actually a model path
        ticker_str = str(ticker)
        if (ticker_str.endswith('.joblib') or ticker_str.endswith('.pkl')) and Path(ticker_str).exists():
            # If ticker is a model path, extract ticker from filename
            model_path = Path(ticker_str)
            self.model_dir = model_path.parent
            self.ticker = model_path.stem.split('_')[0].upper()  # Extract ticker from filename like 'model_AAPL_...'
        else:
            # Handle as a regular ticker
            self.ticker = ticker_str.upper()
            # Set model directory - look in models/ticker/
            if model_dir:
                self.model_dir = Path(model_dir)
            else:
                self.model_dir = Path("models") / self.ticker
                # Ensure directory exists
                self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model and feature columns
        self.model = None
        self.model_classes = []
        self.feature_columns = []
        self.metadata = {}
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """Load the model from the specified path or directory."""
        # Check if model_dir is actually a model file
        if self.model_dir.is_file() and (self.model_dir.suffix == '.joblib' or self.model_dir.suffix == '.pkl'):
            model_file = self.model_dir
            self.model_dir = model_file.parent
        else:
            # Handle directory case
            if not self.model_dir.exists():
                raise FileNotFoundError(f"Model directory not found: {self.model_dir}")
            
            # Find candidate model files (exclude pure calibrators)
            logger.info(f"Searching for model files in: {self.model_dir.absolute()}")
            all_files = list(self.model_dir.glob("*.joblib")) + list(self.model_dir.glob("*.pkl"))
            logger.info(f"Found {len(all_files)} model files: {[f.name for f in all_files]}")
            if not all_files:
                raise FileNotFoundError(f"No model files found in {self.model_dir.absolute()}")

            # Prefer base models (those NOT ending with '_calibrated.joblib')
            base_models = [f for f in all_files if not f.name.endswith('_calibrated.joblib')]
            search_pool = base_models if base_models else all_files  # fallback to any if only calibrators present

            # Pick the newest file in the chosen pool
            model_file = max(search_pool, key=lambda f: f.stat().st_mtime)
        
        # Load the model
        logger.info(f"✅ Loading model from: {model_file}")
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
            
            # Only log column details at debug level
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("=== COLUMN NORMALIZATION ===")
                logger.debug(f"Data columns: {df_copy.columns.tolist()}")
                logger.debug(f"Model features: {self.feature_columns}")
            
            # --- 2. build a list of columns the model expects, after mapping ---
            mapped_cols = []
            missing_cols = []
            
            for model_col in self.feature_columns:
                key = _norm(model_col)
                if key in norm_to_raw:
                    mapped_cols.append(norm_to_raw[key])
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Mapped '{model_col}' to '{norm_to_raw[key]}'")
                else:
                    missing_cols.append(model_col)
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"No match found for '{model_col}'")
            
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
        
        # Calculate risk score and risk tier
        # Risk score is based on the uncertainty of the prediction
        # Higher uncertainty (lower confidence) means higher risk
        result_df['risk_score'] = 1 - result_df['confidence']
        
        # Classify risk tier based on risk score
        # Low Risk (LR): risk_score <= 0.4
        # High Risk (HR): risk_score > 0.4
        result_df['risk_tier'] = np.where(
            result_df['risk_score'] <= 0.4, 
            'LR',  # Low Risk
            'HR'   # High Risk
        )
        
        # Add risk emoji for display
        result_df['risk_emoji'] = np.where(
            result_df['risk_tier'] == 'LR',
            '🟢',  # Green circle for Low Risk
            '🔴'   # Red circle for High Risk
        )
        
        # Add prediction with risk indicator
        result_df['prediction_with_risk'] = result_df.apply(
            lambda row: f"{row['prediction']} {row['risk_emoji']} {row['risk_tier']}" 
            if row['prediction'] != 'NONE' else 'NONE', 
            axis=1
        )
        
        # Log prediction distribution
        pred_counts = pd.Series(predictions).value_counts().to_dict()
        logger.info(f"Prediction distribution: {pred_counts}")
        
        # Log risk tier distribution
        if 'risk_tier' in result_df.columns:
            risk_counts = result_df['risk_tier'].value_counts().to_dict()
            logger.info(f"Risk tier distribution: {risk_counts}")
        
        # Check if all predictions are the same
        if len(pred_counts) == 1:
            only_pred = list(pred_counts.keys())[0]
            logger.warning(f"⚠️ All predictions are '{only_pred}'. This suggests an issue with the model training data.")
        
        return result_df
        
    def get_feature_columns(self) -> list:
        """Get the list of feature columns used by the model.
        
        Returns:
            List of feature column names
        """
        return self.feature_columns.copy()
        
    def _handle_prediction_error(self, e):
        logger.error(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()

class SimpleTrader:
    """
    A realistic trading simulator that properly handles:
    - Short sale margin requirements
    - Position sizing based on equity, not cash
    - Transaction costs
    - Proper P&L calculation
    - Max drawdown tracking
    """
    def __init__(self, equity, commission_per_trade=0.0, borrow_rate_per_bar=0.0001):
        self.cash = equity          # free cash
        self.equity = equity        # cash + unrealised P/L
        self.position = 0           # shares (+long, -short)
        self.avg_px = 0             # average entry price
        self.log = []               # trade log
        self.initial_equity = equity # for P&L calculation
        self.commission = commission_per_trade  # fixed commission per trade
        self.borrow_rate = borrow_rate_per_bar  # short borrow fee per bar
        
        # Drawdown tracking
        self.equity_curve = [equity]  # List of historical equity values
        self.peak_equity = equity     # Highest equity value seen so far
        self.max_drawdown = 0.0       # Maximum percentage drawdown
        self.max_drawdown_start = None # Datetime of peak before max drawdown
        self.max_drawdown_end = None   # Datetime of trough in max drawdown
    
    def _update_equity(self, price, timestamp=None):
        """
        Update equity based on current position and price
        
        Args:
            price: Current price
            timestamp: Optional timestamp for drawdown tracking
        """
        if self.position != 0:
            # For long positions: unrealized P&L = shares * (current - entry)
            # For short positions: unrealized P&L = -shares * (current - entry)
            unrealised = self.position * (price - self.avg_px)
            self.equity = self.cash + unrealised
        else:
            self.equity = self.cash
        
        # Update equity curve and track drawdown
        self._update_drawdown(timestamp)
        
    def _update_drawdown(self, timestamp=None):
        """
        Update equity curve and calculate drawdown metrics
        
        Args:
            timestamp: Optional timestamp for tracking when drawdown occurred
        """
        # Add current equity to the curve
        self.equity_curve.append(self.equity)
        
        # Store the timestamp of the peak for drawdown tracking
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
            # Store timestamp of new peak
            if timestamp is not None:
                self.max_drawdown_start = timestamp
        
        # Calculate current drawdown
        if self.peak_equity > 0:  # Avoid division by zero
            current_drawdown = (self.peak_equity - self.equity) / self.peak_equity * 100
            
            # Update max drawdown if this is worse
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown
                # We already stored the peak timestamp when we found the peak
                # Just store the current timestamp as the trough
                if timestamp is not None:
                    self.max_drawdown_end = timestamp
    
    def trade(self, signal, price, risk, timestamp=None):
        """
        Execute a trade based on signal, price and risk tier
        
        Args:
            signal: 'BUY', 'SELL', or 'NONE'
            price: Current price
            risk: Risk tier ('LR' or 'HR')
            timestamp: Optional timestamp for drawdown tracking
            
        Returns:
            Dictionary with trade details
        """
        trade_action = ''
        old_position = self.position
        old_cash = self.cash
        old_equity = self.equity
        
        # Skip if no signal
        if signal == 'NONE':
            self._update_equity(price, timestamp)
            return {'action': '', 'position': self.position, 'cash': self.cash, 'equity': self.equity}
        
        # Close opposite position first
        if (self.position > 0 and signal == 'SELL') or (self.position < 0 and signal == 'BUY'):
            # Calculate P&L
            if self.position > 0:
                # Closing long position
                pnl = self.position * (price - self.avg_px)
                self.cash += self.position * price  # Return investment + profit
                trade_action = f'Close LONG {self.position:.0f} @ ${price:.2f} (P/L: ${pnl:.2f})'
            else:
                # Closing short position
                pnl = -self.position * (self.avg_px - price)
                self.cash += -self.position * price + pnl  # Return margin + profit
                trade_action = f'Close SHORT {abs(self.position):.0f} @ ${price:.2f} (P/L: ${pnl:.2f})'
            
            # Apply commission
            self.cash -= self.commission
            
            # Reset position
            self.position = 0
            self.avg_px = 0
        
        # Update equity after closing position
        self._update_equity(price, timestamp)
        
        # Determine position sizing based on risk tier
        # 🟢 LR (Low Risk): Use 100% of equity
        # 🔴 HR (High Risk): Use 50% of equity
        size_frac = 1.0 if risk == 'LR' else 0.5
        target_dollars = self.equity * size_frac
        
        # Calculate quantity (round down to whole shares)
        qty = int(target_dollars / price)
        
        # Execute new position if quantity > 0
        if qty > 0:
            if signal == 'BUY':
                # Check if we can afford it
                cost = qty * price + self.commission
                if cost <= self.cash:
                    self.cash -= cost
                    self.position = qty
                    self.avg_px = price
                    
                    if trade_action:
                        trade_action += f' + Open LONG {qty:.0f} @ ${price:.2f}'
                    else:
                        trade_action = f'Open LONG {qty:.0f} @ ${price:.2f}'
                else:
                    # Adjust quantity if we can't afford full position
                    affordable_qty = int((self.cash - self.commission) / price)
                    if affordable_qty > 0:
                        cost = affordable_qty * price + self.commission
                        self.cash -= cost
                        self.position = affordable_qty
                        self.avg_px = price
                        
                        if trade_action:
                            trade_action += f' + Open LONG {affordable_qty:.0f} @ ${price:.2f} (reduced size)'
                        else:
                            trade_action = f'Open LONG {affordable_qty:.0f} @ ${price:.2f} (reduced size)'
            
            elif signal == 'SELL':
                # For shorts, we need to reserve margin (50% of position value)
                margin_req = qty * price * 0.5 + self.commission
                
                if margin_req <= self.cash:
                    self.cash -= margin_req  # Lock margin
                    self.position = -qty
                    self.avg_px = price
                    
                    if trade_action:
                        trade_action += f' + Open SHORT {qty:.0f} @ ${price:.2f}'
                    else:
                        trade_action = f'Open SHORT {qty:.0f} @ ${price:.2f}'
                else:
                    # Adjust quantity if we can't afford margin
                    affordable_qty = int((self.cash - self.commission) / (price * 0.5))
                    if affordable_qty > 0:
                        margin_req = affordable_qty * price * 0.5 + self.commission
                        self.cash -= margin_req
                        self.position = -affordable_qty
                        self.avg_px = price
                        
                        if trade_action:
                            trade_action += f' + Open SHORT {affordable_qty:.0f} @ ${price:.2f} (reduced size)'
                        else:
                            trade_action = f'Open SHORT {affordable_qty:.0f} @ ${price:.2f} (reduced size)'
        
        # Apply borrow fee for short positions
        if self.position < 0:
            borrow_fee = abs(self.position) * price * self.borrow_rate
            self.cash -= borrow_fee
            if borrow_fee > 0.01:  # Only log if significant
                trade_action += f' (Borrow fee: ${borrow_fee:.2f})'
        
        # Update equity after new position
        self._update_equity(price, timestamp)
        
        # Log the trade
        if trade_action:
            self.log.append({
                'action': trade_action,
                'price': price,
                'position_change': self.position - old_position,
                'cash_change': self.cash - old_cash,
                'equity_change': self.equity - old_equity
            })
        
        return {
            'action': trade_action,
            'position': self.position,
            'cash': self.cash,
            'equity': self.equity,
            'pl_dollar': self.equity - self.initial_equity,
            'pl_percent': (self.equity / self.initial_equity - 1) * 100,
            'max_drawdown': self.max_drawdown
        }


def apply_trader(df: pd.DataFrame, initial_balance: float = 1000.0) -> pd.DataFrame:
    """
    Apply a realistic trading engine to execute signals and track positions.
    
    Args:
        df: DataFrame with predictions, risk_tier, and price data
        initial_balance: Starting cash balance
        
    Returns:
        DataFrame with updated balance, position, and P/L columns
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Initialize trading state with SimpleTrader
    trader = SimpleTrader(
        equity=initial_balance,
        commission_per_trade=0.5,  # $0.50 per trade
        borrow_rate_per_bar=0.0001  # 0.01% per bar for short positions
    )
    
    # Initialize result columns
    result_df['cash'] = 0.0
    result_df['position_size'] = 0.0
    result_df['position_value'] = 0.0
    result_df['total_equity'] = initial_balance
    result_df['balance'] = initial_balance
    result_df['P/L ($)'] = 0.0
    result_df['P/L (%)'] = 0.0
    result_df['trade_action'] = ''
    result_df['max_drawdown'] = 0.0
    
    logger.info(f"🏦 Starting trading simulation with ${initial_balance:,.2f}")
    
    for i in range(len(result_df)):
        row = result_df.iloc[i]
        current_price = row['Close']
        signal = row['prediction']
        risk_tier = row.get('risk_tier', 'HR')  # Default to high risk if not available
        timestamp = row.get('datetime', None)  # Get timestamp if available
        
        # Execute trade
        trade_result = trader.trade(signal, current_price, risk_tier, timestamp)
        
        # Calculate position value
        position_value = trader.position * current_price if trader.position != 0 else 0.0
        
        # Update result DataFrame with trade results or defaults if no trade occurred
        result_df.at[result_df.index[i], 'cash'] = trader.cash
        result_df.at[result_df.index[i], 'position_size'] = trader.position
        result_df.at[result_df.index[i], 'position_value'] = position_value
        result_df.at[result_df.index[i], 'total_equity'] = trader.equity
        result_df.at[result_df.index[i], 'balance'] = trader.equity
        
        # Safely get values with defaults if they don't exist
        result_df.at[result_df.index[i], 'P/L ($)'] = trade_result.get('pl_dollar', trader.equity - initial_balance)
        result_df.at[result_df.index[i], 'P/L (%)'] = trade_result.get('pl_percent', 
                                                                    (trader.equity / initial_balance - 1) * 100)
        result_df.at[result_df.index[i], 'trade_action'] = trade_result.get('action', '')
        result_df.at[result_df.index[i], 'max_drawdown'] = trade_result.get('max_drawdown', trader.max_drawdown)
        
        # Log significant trades
        if trade_result['action']:
            logger.info(f"💰 {trade_result['action']} | Equity: ${trader.equity:.2f}")
    
    # Log final results
    final_equity = trader.equity
    final_pnl = final_equity - initial_balance
    final_pnl_pct = (final_equity / initial_balance - 1) * 100
    
    logger.info(f"🏁 Trading simulation complete:")
    logger.info(f"   Initial Balance: ${initial_balance:,.2f}")
    logger.info(f"   Final Balance: ${final_equity:,.2f}")
    logger.info(f"   Total P/L: ${final_pnl:,.2f} ({final_pnl_pct:.2f}%)")
    logger.info(f"   Max Drawdown: {trader.max_drawdown:.2f}%")
    
    return result_df


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
    parser = argparse.ArgumentParser(description="Stock prediction script")
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker symbol")
    parser.add_argument("--model", type=str, help="Model file to use (default: latest model)")
    parser.add_argument("--days", type=int, default=30, help="Number of days to predict")
    parser.add_argument("--interval", type=str, default="1d", help="Data interval (1d, 1h, etc.)")
    parser.add_argument("--sample-file", type=str, help="Sample data file to use instead of latest data")
    parser.add_argument("--retrain", action="store_true", help="Retrain model before prediction")
    parser.add_argument("--output", type=str, help="Output file for predictions (CSV)")
    parser.add_argument("--initial-balance", type=float, default=1000.0, help="Initial balance for trading simulation")
    
    # Add enhanced prediction arguments if available
    if ENHANCED_AVAILABLE:
        parser = enhanced_integration.add_enhanced_args(parser)
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

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
    logger.info(f"🔍 Loading model for {args.ticker}")
    try:
        # If model file is specified, use it
        if args.model:
            model_dir = Path(args.model)
            if model_dir.is_file():
                model_path = model_dir
            else:
                model_path = model_dir / args.ticker
        else:
            # Otherwise use default model directory
            model_path = Path("models") / args.ticker
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
    
    # Load data
    logger.info(f"📊 Loading data for {args.ticker}")
    if args.sample_file:
        # Use sample file if provided
        sample_path = Path(args.sample_file)
        if not sample_path.exists():
            logger.error(f"Sample file not found: {sample_path}")
            return 1
        
        df = pd.read_csv(sample_path)
        logger.info(f"Loaded sample data from {sample_path}")
    else:
        # Otherwise load latest data
        df = load_latest_data(args.ticker, "data")
        if df is None:
            logger.error(f"No data found for {args.ticker}")
            return 1
    
    # Check if we have data
    if df.empty:
        logger.error("No data loaded. Exiting.")
        return 1
        
    # Check if we should use enhanced prediction system
    use_enhanced = False
    if ENHANCED_AVAILABLE and hasattr(args, 'use_enhanced') and args.use_enhanced:
        logger.info("Using enhanced prediction system")
        
        # Get confidence threshold from args if available
        confidence_threshold = getattr(args, 'confidence_threshold', 0.6)
        
        # Get contextual and calibration flags from args if available
        enable_contextual = not getattr(args, 'disable_contextual', False)
        enable_calibration = not getattr(args, 'disable_calibration', False)
        
        # Try to run enhanced prediction
        results = enhanced_integration.run_enhanced_prediction(
            df=df,
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            enable_contextual=enable_contextual,
            enable_calibration=enable_calibration,
            apply_trading=False,  # We'll apply trading later
            initial_balance=args.initial_balance
        )
        
        if results is not None:
            use_enhanced = True
            logger.info("Enhanced prediction completed successfully")
        else:
            logger.warning("Enhanced prediction failed, falling back to standard prediction")
    
    if not use_enhanced:
        # Create standard predictor
        logger.info("Using standard prediction system")
        # Use the correct model directory structure: models/AAPL/
        ticker_model_dir = Path("models") / args.ticker
        logger.info(f"Looking for models in: {ticker_model_dir.absolute()}")
        predictor = StockPredictor(args.ticker, model_dir=str(ticker_model_dir))
        
        # Make predictions
        logger.info("Making predictions")
        results = predictor.predict(df)
        
        # Get confidence threshold from args if available
        confidence_threshold = getattr(args, 'confidence_threshold', 0.6)
        
        # Filter predictions by confidence
        logger.info(f"Filtering predictions with confidence threshold {confidence_threshold}")
        filtered_predictions = results[results['confidence'] >= confidence_threshold].copy()
        
        # For rows below threshold, set prediction to 'NONE'
        results.loc[results['confidence'] < confidence_threshold, 'prediction'] = 'NONE'
        
        # Update prediction_with_risk column
        results['prediction_with_risk'] = results.apply(
            lambda row: f"{row['prediction']} {row['risk_emoji']} {row['risk_tier']}" 
            if row['prediction'] != 'NONE' else 'NONE', 
            axis=1
        )
    
    # Display results
    print("\n📈 Prediction Results:")
    print("-" * 50)
    
    # Calculate signal quality if not already present
    if 'signal_quality' not in results.columns:
        logger.info("Calculating signal quality")
        signal_quality = []
        for i, row in results.iterrows():
            if row['prediction'] == 'NONE':
                signal_quality.append('➖')
            else:
                if i < len(results) - 1:
                    next_row = results.iloc[i + 1]
                    current_price = row['Close']
                    next_price = next_row['Close']
                    is_correct = (row['prediction'] == 'BUY' and next_price > current_price) or (row['prediction'] == 'SELL' and next_price < current_price)
                    signal_quality.append('✅' if is_correct else '❌')
                else:
                    signal_quality.append('❓')
        
        results['signal_quality'] = signal_quality
    
    # Apply trading engine
    logger.info("Applying trading engine")
    results = apply_trader(results, initial_balance=args.initial_balance)
    
    # Build display columns dynamically based on available columns
    display_cols = ['datetime_str', 'Close']
    
    # Add prediction column (with or without risk)
    if 'prediction_with_risk' in results.columns:
        display_cols.append('prediction_with_risk')
    else:
        display_cols.append('prediction')
    
    # Add signal quality and confidence
    display_cols.append('signal_quality')
    
    # Add risk indicators if available
    if 'risk_emoji' in results.columns:
        display_cols.append('risk_emoji')
    if 'risk_tier' in results.columns:
        display_cols.append('risk_tier')
    if 'risk_score' in results.columns:
        display_cols.append('risk_score')
        if 'prediction_with_risk' in filtered_predictions.columns:
            display_cols.append('prediction_with_risk')
        else:
            display_cols.append('prediction')
        
        # Add signal quality and confidence
        display_cols.append('signal_quality')
        
        # Add risk indicators if available
        if 'risk_emoji' in filtered_predictions.columns:
            display_cols.append('risk_emoji')
        if 'risk_tier' in filtered_predictions.columns:
            display_cols.append('risk_tier')
        if 'risk_score' in filtered_predictions.columns:
            display_cols.append('risk_score')
            
        # Add confidence
        display_cols.append('confidence')
        
        # Add trading information
        if 'trade_action' in filtered_predictions.columns:
            display_cols.append('trade_action')
        if 'position_size' in filtered_predictions.columns:
            display_cols.append('position_size')
        if 'cash' in filtered_predictions.columns:
            display_cols.append('cash')
            
        # Only keep columns that exist in the DataFrame
        display_cols = [col for col in display_cols if col in filtered_predictions.columns]
        
        # Filter out NONE predictions if they exist
        if 'prediction' in filtered_predictions.columns:
            signals_df = filtered_predictions[filtered_predictions['prediction'] != 'NONE'].copy()
        else:
            signals_df = filtered_predictions.copy()
            
        # Format the output
        pd.set_option('display.float_format', '{:,.2f}'.format)
        pd.set_option('display.max_rows', 100)  # Show up to 100 rows
        
        if not signals_df.empty:
            print(f"\n📊 Trading Signals (Showing {len(signals_df)} non-NONE signals)")
            print("-" * 120)
            print("Signal Quality:")
            print("  ✅ Correct Prediction  ❌ Incorrect Prediction  ➖ No Signal")
            
            # Only show risk indicators if we have risk data
            if any(col in signals_df.columns for col in ['risk_tier', 'risk_score']):
                print("\nRisk Indicators:")
                print("  🟢 Low Risk (LR)  🔴 High Risk (HR)")
                
            print("-" * 120)
            
            # Ensure datetime is in the display if available
            if 'datetime' in signals_df.columns:
                display_cols = ['datetime'] + [col for col in display_cols if col != 'datetime']
                
            print(signals_df[display_cols].to_string(index=False))
            print(f"\nTotal signals found: {len(signals_df)}")
        else:
            print("\n⚠️ No trading signals found (all predictions are NONE)")
    
    # Print summary with risk tiers if available, otherwise fall back to basic summary
    if 'prediction' in results.columns:
        if 'risk_tier' in results.columns and 'risk_score' in results.columns:
            # Summary by prediction and risk tier
            print("\n📊 Prediction Summary by Risk Tier:")
            print("-" * 60)
            
            # Get unique predictions (excluding NONE)
            unique_preds = results[results['prediction'] != 'NONE']['prediction'].unique()
            
            for pred in sorted(unique_preds):
                pred_subset = results[results['prediction'] == pred]
                total = len(pred_subset)
                
                if total == 0:
                    continue
                    
                lr_count = len(pred_subset[pred_subset['risk_tier'] == 'LR'])
                hr_count = len(pred_subset[pred_subset['risk_tier'] == 'HR'])
                
                print(f"{pred}:")
                print(f"  🟢 Low Risk (LR):  {lr_count:3d} ({lr_count/max(1, total)*100:5.1f}%)")
                print(f"  🔴 High Risk (HR): {hr_count:3d} ({hr_count/max(1, total)*100:5.1f}%)")
                print(f"  {'-' * 50}")
            
            # Add overall statistics
            total_signals = len(results[results['prediction'] != 'NONE'])
            if total_signals > 0:
                lr_pct = len(results[results['risk_tier'] == 'LR']) / total_signals * 100
                hr_pct = len(results[results['risk_tier'] == 'HR']) / total_signals * 100
                
                print(f"\n📈 Risk Distribution (Excluding NONE signals):")
                print(f"  🟢 Low Risk (LR):  {lr_pct:5.1f}%")
                print(f"  🔴 High Risk (HR): {hr_pct:5.1f}%")
                
                # Calculate average risk score
                avg_risk = results[results['prediction'] != 'NONE']['risk_score'].mean()
                risk_level = "Low" if avg_risk <= 0.4 else "Medium" if avg_risk <= 0.7 else "High"
                print(f"\n📊 Average Risk Score: {avg_risk:.2f} ({risk_level} Risk)")
        else:
            # Fallback to simple summary if risk tiers not available
            pred_counts = results['prediction'].value_counts()
            print("\n📊 Prediction Summary:")
            print("-" * 40)
            for pred_type, count in pred_counts.items():
                print(f"  {pred_type}: {count} instances ({count/len(results)*100:.1f}%)")
    elif 'prediction' in results.columns:
        # Fallback to simple summary if risk tiers not available
        pred_counts = results['prediction'].value_counts()
        print("\n📊 Prediction Summary:")
        print("-" * 40)
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
    result = main()
    
    # Print enhanced system info if available
    if ENHANCED_AVAILABLE:
        enhanced_integration.print_enhanced_info()
    
    sys.exit(result)
