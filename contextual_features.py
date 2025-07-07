#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Contextual Features Module for Stock Trading Agents

This module adds advanced contextual features to improve model performance:
1. Window-based features (past N-bar returns, volatility)
2. Market context features (trend strength, support/resistance)
3. Temporal awareness (time-based patterns)
4. Adaptive feature generation
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
# Try to import talib, but provide fallback implementations if not available
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    import warnings
    warnings.warn("TA-Lib not installed. Using pandas/numpy implementations for technical indicators.")
    # We'll implement our own versions of the indicators below
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Fallback implementations for TA-Lib functions using pandas and numpy
def fallback_rsi(close, timeperiod=14):
    """Calculate RSI using pandas."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=timeperiod).mean()
    avg_loss = loss.rolling(window=timeperiod).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def fallback_sma(close, timeperiod=30):
    """Calculate Simple Moving Average using pandas."""
    return close.rolling(window=timeperiod).mean()

def fallback_ema(close, timeperiod=30):
    """Calculate Exponential Moving Average using pandas."""
    return close.ewm(span=timeperiod, adjust=False).mean()

def fallback_macd(close, fastperiod=12, slowperiod=26, signalperiod=9):
    """Calculate MACD using pandas."""
    ema_fast = fallback_ema(close, fastperiod)
    ema_slow = fallback_ema(close, slowperiod)
    macd_line = ema_fast - ema_slow
    signal_line = fallback_ema(macd_line, signalperiod)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def fallback_bbands(close, timeperiod=5, nbdevup=2, nbdevdn=2):
    """Calculate Bollinger Bands using pandas."""
    middle = close.rolling(window=timeperiod).mean()
    std = close.rolling(window=timeperiod).std()
    upper = middle + (std * nbdevup)
    lower = middle - (std * nbdevdn)
    return upper, middle, lower

def fallback_atr(high, low, close, timeperiod=14):
    """Calculate Average True Range using pandas."""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    atr = tr.rolling(window=timeperiod).mean()
    return atr

def fallback_obv(close, volume):
    """Calculate On-Balance Volume using pandas."""
    obv = pd.Series(index=close.index, dtype='float64')
    obv.iloc[0] = volume.iloc[0]
    
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv

def fallback_cci(high, low, close, timeperiod=14):
    """Calculate Commodity Channel Index using pandas."""
    tp = (high + low + close) / 3
    tp_sma = tp.rolling(window=timeperiod).mean()
    mad = tp.rolling(window=timeperiod).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    cci = (tp - tp_sma) / (0.015 * mad)
    return cci

def fallback_adx(high, low, close, timeperiod=14):
    """Calculate Average Directional Index using pandas."""
    # This is a simplified version
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm.abs()), 0)
    minus_dm = minus_dm.abs().where((minus_dm > 0) & (minus_dm > plus_dm), 0)
    
    tr = fallback_atr(high, low, close, timeperiod)
    plus_di = 100 * fallback_ema(plus_dm, timeperiod) / tr
    minus_di = 100 * fallback_ema(minus_dm, timeperiod) / tr
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = fallback_ema(dx, timeperiod)
    return adx

def fallback_aroon(high, low, timeperiod=14):
    """Calculate Aroon Oscillator using pandas."""
    aroon_up = pd.Series(index=high.index)
    aroon_down = pd.Series(index=low.index)
    
    for i in range(timeperiod, len(high)):
        period_high = high.iloc[i-timeperiod:i+1]
        period_low = low.iloc[i-timeperiod:i+1]
        
        aroon_up.iloc[i] = 100 * (timeperiod - period_high.argmax()) / timeperiod
        aroon_down.iloc[i] = 100 * (timeperiod - period_low.argmin()) / timeperiod
    
    aroon_osc = aroon_up - aroon_down
    return aroon_up, aroon_down, aroon_osc

def fallback_cmf(high, low, close, volume, timeperiod=20):
    """Calculate Chaikin Money Flow using pandas."""
    mfv = volume * ((close - low) - (high - close)) / (high - low)
    cmf = mfv.rolling(window=timeperiod).sum() / volume.rolling(window=timeperiod).sum()
    return cmf


class ContextualFeatureGenerator:
    """
    A class to generate contextual features for stock prediction.
    """
    
    def __init__(self):
        """Initialize the ContextualFeatureGenerator."""
        self.scaler = StandardScaler()
    
    def add_window_features(self, df, price_col='Close', volume_col='Volume', 
                           windows=[5, 10, 20, 50]):
        """
        Add rolling window features for price and volume.
        
        Args:
            df: DataFrame with price and volume data
            price_col: Name of the price column
            volume_col: Name of the volume column
            windows: List of window sizes
            
        Returns:
            DataFrame with added window features
        """
        logger.info(f"Adding window features with sizes: {windows}")
        
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Ensure datetime index for proper rolling windows
        if 'datetime' in result.columns:
            result['datetime'] = pd.to_datetime(result['datetime'])
            result = result.set_index('datetime')
        
        # Add price-based features
        for window in windows:
            # Returns
            result[f'return_{window}d'] = result[price_col].pct_change(window)
            
            # Volatility (standard deviation of returns)
            result[f'volatility_{window}d'] = result[price_col].pct_change().rolling(window).std()
            
            # Acceleration (change in returns)
            result[f'acceleration_{window}d'] = result[price_col].pct_change().diff().rolling(window).mean()
            
            # Z-score (how many standard deviations from the mean)
            rolling_mean = result[price_col].rolling(window).mean()
            rolling_std = result[price_col].rolling(window).std()
            result[f'zscore_{window}d'] = (result[price_col] - rolling_mean) / rolling_std
            
            # Momentum (current price / price N periods ago - 1)
            result[f'momentum_{window}d'] = (result[price_col] / result[price_col].shift(window) - 1) * 100
            
            # Trend strength (R-squared of linear regression)
            if window >= 5:  # Need enough points for meaningful regression
                result[f'trend_strength_{window}d'] = self._calculate_trend_strength(result[price_col], window)
        
        # Add volume-based features if volume column exists
        if volume_col in result.columns:
            for window in windows:
                # Volume change
                result[f'volume_change_{window}d'] = result[volume_col].pct_change(window)
                
                # Relative volume (current volume / average volume)
                result[f'relative_volume_{window}d'] = result[volume_col] / result[volume_col].rolling(window).mean()
                
                # Price-volume correlation
                if window >= 5:  # Need enough points for meaningful correlation
                    result[f'price_volume_corr_{window}d'] = self._rolling_correlation(
                        result[price_col], result[volume_col], window
                    )
        
        # Reset index if we set it earlier
        if 'datetime' not in result.columns and isinstance(result.index, pd.DatetimeIndex):
            result = result.reset_index()
        
        # Fill NaN values created by rolling windows
        result = result.bfill().fillna(0)
        
        logger.info(f"Added {len(result.columns) - len(df.columns)} window features")
        
        return result
    
    def add_technical_indicators(self, df, price_col='Close', high_col='High', 
                               low_col='Low', volume_col='Volume'):
        """
        Add technical indicators using TA-Lib or fallback implementations.
        
        Args:
            df: DataFrame with OHLCV data
            price_col: Name of the close price column
            high_col: Name of the high price column
            low_col: Name of the low price column
            volume_col: Name of the volume column
            
        Returns:
            DataFrame with added technical indicators
        """
        logger.info("Adding technical indicators")
        
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Check if required columns exist
        required_cols = [price_col]
        if high_col not in result.columns or low_col not in result.columns:
            logger.warning(f"High/Low columns not found. Some indicators will be skipped.")
        else:
            required_cols.extend([high_col, low_col])
        
        if volume_col not in result.columns:
            logger.warning(f"Volume column not found. Volume-based indicators will be skipped.")
        else:
            required_cols.append(volume_col)
        
        # Convert columns to float if needed
        for col in required_cols:
            if col in result.columns:
                result[col] = result[col].astype(float)
        
        # Add momentum indicators
        if price_col in result.columns:
            # RSI (Relative Strength Index)
            if TALIB_AVAILABLE:
                result['RSI_14'] = talib.RSI(result[price_col].values, timeperiod=14)
            else:
                result['RSI_14'] = fallback_rsi(result[price_col], timeperiod=14)
            
            # MACD (Moving Average Convergence Divergence)
            if TALIB_AVAILABLE:
                macd, macd_signal, macd_hist = talib.MACD(
                    result[price_col].values, fastperiod=12, slowperiod=26, signalperiod=9
                )
            else:
                macd, macd_signal, macd_hist = fallback_macd(
                    result[price_col], fastperiod=12, slowperiod=26, signalperiod=9
                )
            result['MACD'] = macd
            result['MACD_Signal'] = macd_signal
            result['MACD_Hist'] = macd_hist
            
            # ROC (Rate of Change)
            if TALIB_AVAILABLE:
                result['ROC_10'] = talib.ROC(result[price_col].values, timeperiod=10)
            else:
                # Simple implementation of ROC
                result['ROC_10'] = result[price_col].pct_change(10) * 100
            
            # CCI (Commodity Channel Index)
            if all(col in result.columns for col in [high_col, low_col]):
                if TALIB_AVAILABLE:
                    result['CCI_14'] = talib.CCI(
                        result[high_col].values, result[low_col].values, 
                        result[price_col].values, timeperiod=14
                    )
                else:
                    result['CCI_14'] = fallback_cci(
                        result[high_col], result[low_col], result[price_col], timeperiod=14
                    )
        
        # Add volatility indicators
        if all(col in result.columns for col in [high_col, low_col, price_col]):
            # ATR (Average True Range)
            if TALIB_AVAILABLE:
                result['ATR_14'] = talib.ATR(
                    result[high_col].values, result[low_col].values, 
                    result[price_col].values, timeperiod=14
                )
            else:
                result['ATR_14'] = fallback_atr(
                    result[high_col], result[low_col], result[price_col], timeperiod=14
                )
            
            # Bollinger Bands
            if TALIB_AVAILABLE:
                upper, middle, lower = talib.BBANDS(
                    result[price_col].values, timeperiod=20, 
                    nbdevup=2, nbdevdn=2, matype=0
                )
            else:
                upper, middle, lower = fallback_bbands(
                    result[price_col], timeperiod=20, nbdevup=2, nbdevdn=2
                )
            result['BB_Upper'] = upper
            result['BB_Middle'] = middle
            result['BB_Lower'] = lower
            
            # Calculate BB width and %B
            result['BB_Width'] = (result['BB_Upper'] - result['BB_Lower']) / result['BB_Middle']
            result['BB_PercentB'] = (result[price_col] - result['BB_Lower']) / (result['BB_Upper'] - result['BB_Lower'])
        
        # Add trend indicators
        if price_col in result.columns:
            # ADX (Average Directional Index)
            if all(col in result.columns for col in [high_col, low_col]):
                if TALIB_AVAILABLE:
                    result['ADX_14'] = talib.ADX(
                        result[high_col].values, result[low_col].values, 
                        result[price_col].values, timeperiod=14
                    )
                else:
                    result['ADX_14'] = fallback_adx(
                        result[high_col], result[low_col], result[price_col], timeperiod=14
                    )
            
            # Aroon Oscillator
            if all(col in result.columns for col in [high_col, low_col]):
                if TALIB_AVAILABLE:
                    aroon_down, aroon_up = talib.AROON(
                        result[high_col].values, result[low_col].values, timeperiod=14
                    )
                    result['Aroon_Oscillator'] = aroon_up - aroon_down
                else:
                    aroon_up, aroon_down, aroon_osc = fallback_aroon(
                        result[high_col], result[low_col], timeperiod=14
                    )
                    result['Aroon_Oscillator'] = aroon_osc
        
        # Add volume indicators
        if all(col in result.columns for col in [price_col, volume_col]):
            # OBV (On-Balance Volume)
            if TALIB_AVAILABLE:
                result['OBV'] = talib.OBV(result[price_col].values, result[volume_col].values)
            else:
                result['OBV'] = fallback_obv(result[price_col], result[volume_col])
            
            # CMF (Chaikin Money Flow)
            if all(col in result.columns for col in [high_col, low_col]):
                if TALIB_AVAILABLE:
                    result['CMF_20'] = talib.ADOSC(
                        result[high_col].values, result[low_col].values,
                        result[price_col].values, result[volume_col].values,
                        fastperiod=3, slowperiod=10
                    )
                else:
                    result['CMF_20'] = fallback_cmf(
                        result[high_col], result[low_col], result[price_col], result[volume_col], timeperiod=20
                    )
        
        # Fill NaN values
        result = result.bfill().fillna(0)
        
        logger.info(f"Added {len(result.columns) - len(df.columns)} technical indicators")
        
        return result
    
    def add_temporal_features(self, df, datetime_col='datetime'):
        """
        Add time-based features to capture temporal patterns.
        
        Args:
            df: DataFrame with datetime column
            datetime_col: Name of the datetime column
            
        Returns:
            DataFrame with added temporal features
        """
        logger.info("Adding temporal features")
        
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Check if datetime column exists
        if datetime_col not in result.columns:
            logger.warning(f"Datetime column '{datetime_col}' not found. Skipping temporal features.")
            return result
        
        # Convert to datetime if needed
        result[datetime_col] = pd.to_datetime(result[datetime_col])
        
        # Extract time components
        result['day_of_week'] = result[datetime_col].dt.dayofweek
        result['hour_of_day'] = result[datetime_col].dt.hour
        result['minute_of_hour'] = result[datetime_col].dt.minute
        result['month'] = result[datetime_col].dt.month
        result['quarter'] = result[datetime_col].dt.quarter
        
        # Create cyclical features for periodic components
        # Day of week (0-6) -> sin and cos
        result['day_of_week_sin'] = np.sin(2 * np.pi * result['day_of_week'] / 7)
        result['day_of_week_cos'] = np.cos(2 * np.pi * result['day_of_week'] / 7)
        
        # Hour of day (0-23) -> sin and cos
        result['hour_of_day_sin'] = np.sin(2 * np.pi * result['hour_of_day'] / 24)
        result['hour_of_day_cos'] = np.cos(2 * np.pi * result['hour_of_day'] / 24)
        
        # Month (1-12) -> sin and cos
        result['month_sin'] = np.sin(2 * np.pi * result['month'] / 12)
        result['month_cos'] = np.cos(2 * np.pi * result['month'] / 12)
        
        # Is market open/close hour
        result['is_market_open'] = ((result['hour_of_day'] >= 9) & 
                                   (result['hour_of_day'] < 16)).astype(int)
        result['is_near_close'] = ((result['hour_of_day'] == 15) & 
                                  (result['minute_of_hour'] >= 30)).astype(int)
        result['is_near_open'] = ((result['hour_of_day'] == 9) & 
                                 (result['minute_of_hour'] <= 30)).astype(int)
        
        logger.info(f"Added {len(result.columns) - len(df.columns)} temporal features")
        
        return result
    
    def add_support_resistance(self, df, price_col='Close', window=20, num_levels=3):
        """
        Add support and resistance level features.
        
        Args:
            df: DataFrame with price data
            price_col: Name of the price column
            window: Window size for identifying levels
            num_levels: Number of support/resistance levels to identify
            
        Returns:
            DataFrame with added support/resistance features
        """
        logger.info(f"Adding support/resistance features with window {window}")
        
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Check if price column exists
        if price_col not in result.columns:
            logger.warning(f"Price column '{price_col}' not found. Skipping support/resistance features.")
            return result
        
        # Initialize support/resistance columns
        for i in range(1, num_levels + 1):
            result[f'support_{i}'] = np.nan
            result[f'resistance_{i}'] = np.nan
            result[f'dist_to_support_{i}'] = np.nan
            result[f'dist_to_resistance_{i}'] = np.nan
        
        # Calculate support and resistance levels for each point
        for i in range(window, len(result)):
            # Get the window of prices
            window_prices = result[price_col].iloc[i-window:i].values
            
            # Find local minima (support) and maxima (resistance)
            # A point is a local minimum if it's less than both neighbors
            is_min = np.r_[True, window_prices[1:] < window_prices[:-1]] & \
                     np.r_[window_prices[:-1] < window_prices[1:], True]
            
            # A point is a local maximum if it's greater than both neighbors
            is_max = np.r_[True, window_prices[1:] > window_prices[:-1]] & \
                     np.r_[window_prices[:-1] > window_prices[1:], True]
            
            # Get support and resistance levels
            support_levels = window_prices[is_min]
            resistance_levels = window_prices[is_max]
            
            # Sort levels
            support_levels = np.sort(support_levels)
            resistance_levels = np.sort(resistance_levels)[::-1]  # Descending
            
            # Current price
            current_price = result[price_col].iloc[i]
            
            # Store levels and distances
            for j in range(1, num_levels + 1):
                # Skip if current price is zero or NaN to avoid division by zero
                if pd.isna(current_price) or current_price <= 0:
                    continue
                    
                # Support levels (below current price)
                if j <= len(support_levels):
                    support = support_levels[j-1]
                    if support > 0 and support < current_price:  # Ensure support is positive
                        result.at[result.index[i], f'support_{j}'] = support
                        result.at[result.index[i], f'dist_to_support_{j}'] = (current_price / support - 1) * 100
                
                # Resistance levels (above current price)
                if j <= len(resistance_levels):
                    resistance = resistance_levels[j-1]
                    if resistance > current_price:  # resistance is already > current_price > 0
                        result.at[result.index[i], f'resistance_{j}'] = resistance
                        result.at[result.index[i], f'dist_to_resistance_{j}'] = (resistance / current_price - 1) * 100
        
        # Fill NaN values with forward fill then backward fill
        for i in range(1, num_levels + 1):
            result[f'support_{i}'] = result[f'support_{i}'].ffill().bfill()
            result[f'resistance_{i}'] = result[f'resistance_{i}'].ffill().bfill()
            result[f'dist_to_support_{i}'] = result[f'dist_to_support_{i}'].ffill().fillna(0)
            result[f'dist_to_resistance_{i}'] = result[f'dist_to_resistance_{i}'].ffill().fillna(0)
        
        logger.info(f"Added {num_levels * 4} support/resistance features")
        
        return result
    
    def add_all_features(self, df, price_col='Close', high_col='High', low_col='Low', 
                        volume_col='Volume', datetime_col='datetime'):
        """
        Add all contextual features to the DataFrame.
        
        Args:
            df: DataFrame with price and volume data
            price_col: Name of the price column
            high_col: Name of the high price column
            low_col: Name of the low price column
            volume_col: Name of the volume column
            datetime_col: Name of the datetime column
            
        Returns:
            DataFrame with all added features
        """
        logger.info("Adding all contextual features")
        
        # Add features in sequence
        result = df.copy()
        
        # Add window features
        result = self.add_window_features(result, price_col, volume_col)
        
        # Add technical indicators
        result = self.add_technical_indicators(result, price_col, high_col, low_col, volume_col)
        
        # Add temporal features
        result = self.add_temporal_features(result, datetime_col)
        
        # Add support/resistance features
        result = self.add_support_resistance(result, price_col)
        
        logger.info(f"Added a total of {len(result.columns) - len(df.columns)} contextual features")
        
        return result
    
    def _calculate_trend_strength(self, series, window):
        """
        Calculate trend strength using R-squared of linear regression.
        
        Args:
            series: Price series
            window: Window size
            
        Returns:
            Series with trend strength values
        """
        result = pd.Series(index=series.index, dtype=float)
        
        for i in range(window, len(series) + 1):
            # Get window of prices
            y = series.iloc[i-window:i].values
            x = np.arange(window)
            
            # Calculate linear regression
            n = window
            sum_x = np.sum(x)
            sum_y = np.sum(y)
            sum_xy = np.sum(x * y)
            sum_xx = np.sum(x * x)
            
            # Calculate slope and intercept
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
            intercept = (sum_y - slope * sum_x) / n
            
            # Calculate predicted values
            y_pred = intercept + slope * x
            
            # Calculate R-squared
            ss_total = np.sum((y - np.mean(y)) ** 2)
            ss_residual = np.sum((y - y_pred) ** 2)
            
            # Avoid division by zero
            if ss_total == 0:
                r_squared = 0
            else:
                r_squared = 1 - ss_residual / ss_total
            
            # Store result
            result.iloc[i-1] = r_squared
        
        return result
    
    def _rolling_correlation(self, series1, series2, window):
        """
        Calculate rolling correlation between two series.
        
        Args:
            series1: First series
            series2: Second series
            window: Window size
            
        Returns:
            Series with correlation values
        """
        return series1.rolling(window).corr(series2)


def main():
    """Main function to add contextual features to stock data."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Add contextual features to stock data')
    parser.add_argument('--input-file', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--output-file', type=str, help='Path to output CSV file')
    parser.add_argument('--price-col', type=str, default='Close', help='Name of price column')
    parser.add_argument('--high-col', type=str, default='High', help='Name of high price column')
    parser.add_argument('--low-col', type=str, default='Low', help='Name of low price column')
    parser.add_argument('--volume-col', type=str, default='Volume', help='Name of volume column')
    parser.add_argument('--datetime-col', type=str, default='datetime', help='Name of datetime column')
    
    args = parser.parse_args()
    
    # Set default output file if not provided
    if not args.output_file:
        input_dir = os.path.dirname(args.input_file)
        input_name = os.path.basename(args.input_file).split('.')[0]
        args.output_file = os.path.join(input_dir, f"{input_name}_enhanced.csv")
    
    # Load data
    logger.info(f"Loading data from {args.input_file}")
    df = pd.read_csv(args.input_file)
    
    # Create feature generator
    generator = ContextualFeatureGenerator()
    
    # Add all features
    enhanced_df = generator.add_all_features(
        df, 
        price_col=args.price_col,
        high_col=args.high_col,
        low_col=args.low_col,
        volume_col=args.volume_col,
        datetime_col=args.datetime_col
    )
    
    # Save enhanced data
    logger.info(f"Saving enhanced data to {args.output_file}")
    enhanced_df.to_csv(args.output_file, index=False)
    
    logger.info(f"Original shape: {df.shape}, Enhanced shape: {enhanced_df.shape}")
    logger.info("Feature generation complete")


if __name__ == "__main__":
    main()
