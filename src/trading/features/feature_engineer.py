"""
Feature Engineering Pipeline for Stock Trading Data

This module provides feature engineering capabilities for stock market data,
automatically calculating and appending technical indicators to OHLCV data.
"""

import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import warnings
import traceback
from typing import Optional, List, Dict, Any
from pathlib import Path
import os

warnings.filterwarnings('ignore')

class FeatureEngineerConfig:
    """Configuration class for feature engineering parameters."""
    
    def __init__(self):
        # Price-based features
        self.candle_body = True
        self.shadows = True
        self.volatility = True
        
        # Percentage changes
        self.price_changes = True
        self.volume_changes = True
        
        # Moving averages
        self.moving_averages = True
        self.ema_periods = [5, 10]
        self.sma_periods = [5, 10]
        
        # Momentum indicators
        self.rsi = True
        self.rsi_period = 14
        
        # MACD
        self.macd = True
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        
        # Bollinger Bands
        self.bollinger = True
        self.bollinger_period = 20
        self.bollinger_std = 2
        
        # Required columns
        self.required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

class FeatureEngineer:
    """
    Feature Engineering Pipeline for Stock Market Data
    
    This class handles the automatic calculation and appending of technical
    indicators to stock market OHLCV data files.
    """
    
    def __init__(self, config: Optional[FeatureEngineerConfig] = None):
        """Initialize the Feature Engineer with configuration."""
        self.config = config or FeatureEngineerConfig()
        self.logger_prefix = "🔧 [FeatureEngineer]"
    
    def log(self, message: str, level: str = "INFO"):
        """Simple logging method."""
        prefix = self.logger_prefix
        if level == "ERROR":
            prefix = "❌ [FeatureEngineer]"
        elif level == "WARNING":
            prefix = "⚠️ [FeatureEngineer]"
        elif level == "SUCCESS":
            prefix = "✅ [FeatureEngineer]"
        
        print(f"{prefix} {message}")
    
    def validate_data(self, df: pd.DataFrame) -> tuple[bool, str]:
        """
        Validate that the dataframe has required columns and structure.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            tuple: (is_valid, error_message)
        """
        if df.empty:
            return False, "DataFrame is empty"
        
        # Check required columns
        missing_cols = [col for col in self.config.required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
        
        # Check for numeric data
        for col in self.config.required_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                return False, f"Column {col} is not numeric"
        
        # Check for minimum data points (need at least 30 for most indicators)
        if len(df) < 30:
            return False, f"Insufficient data points: {len(df)} (minimum 30 required)"
        
        return True, ""
    
    def calculate_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate price-based features."""
        if not self.config.candle_body and not self.config.shadows and not self.config.volatility:
            return df
        
        try:
            if self.config.candle_body:
                df['Candle_Body'] = abs(df['Close'] - df['Open'])
            
            if self.config.shadows:
                df['Upper_Shadow'] = df['High'] - df[['Close', 'Open']].max(axis=1)
                df['Lower_Shadow'] = df[['Close', 'Open']].min(axis=1) - df['Low']
            
            if self.config.volatility:
                df['Volatility'] = df['High'] - df['Low']
            
            self.log("✓ Price-based features calculated")
            
        except Exception as e:
            self.log(f"Error calculating price features: {str(e)}", "ERROR")
            
        return df
    
    def calculate_percentage_changes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate percentage changes."""
        if not self.config.price_changes and not self.config.volume_changes:
            return df
        
        try:
            if self.config.price_changes:
                df['Close_pct_change'] = df['Close'].pct_change() * 100
            
            if self.config.volume_changes:
                df['Volume_pct_change'] = df['Volume'].pct_change() * 100
            
            self.log("✓ Percentage changes calculated")
            
        except Exception as e:
            self.log(f"Error calculating percentage changes: {str(e)}", "ERROR")
            
        return df
    
    def calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate moving averages."""
        if not self.config.moving_averages:
            return df
        
        try:
            # EMA calculations
            for period in self.config.ema_periods:
                ema = EMAIndicator(close=df['Close'], window=period)
                df[f'EMA_{period}'] = ema.ema_indicator()
            
            # SMA calculations
            for period in self.config.sma_periods:
                sma = SMAIndicator(close=df['Close'], window=period)
                df[f'SMA_{period}'] = sma.sma_indicator()
            
            self.log("✓ Moving averages calculated")
            
        except Exception as e:
            self.log(f"Error calculating moving averages: {str(e)}", "ERROR")
            
        return df
    
    def calculate_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum indicators."""
        if not self.config.rsi:
            return df
        
        try:
            # RSI
            rsi = RSIIndicator(close=df['Close'], window=self.config.rsi_period)
            df[f'RSI_{self.config.rsi_period}'] = rsi.rsi()
            
            self.log("✓ Momentum indicators calculated")
            
        except Exception as e:
            self.log(f"Error calculating momentum indicators: {str(e)}", "ERROR")
            
        return df
    
    def calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD indicators."""
        if not self.config.macd:
            return df
        
        try:
            macd = MACD(
                close=df['Close'],
                window_fast=self.config.macd_fast,
                window_slow=self.config.macd_slow,
                window_sign=self.config.macd_signal
            )
            
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
            df['MACD_Hist'] = macd.macd_diff()
            
            self.log("✓ MACD indicators calculated")
            
        except Exception as e:
            self.log(f"Error calculating MACD: {str(e)}", "ERROR")
            
        return df
    
    def calculate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        if not self.config.bollinger:
            return df
        
        try:
            bollinger = BollingerBands(
                close=df['Close'],
                window=self.config.bollinger_period,
                window_dev=self.config.bollinger_std
            )
            
            df['Bollinger_Upper'] = bollinger.bollinger_hband()
            df['Bollinger_Lower'] = bollinger.bollinger_lband()
            df['Bollinger_Mid'] = bollinger.bollinger_mavg()
            
            self.log("✓ Bollinger Bands calculated")
            
        except Exception as e:
            self.log(f"Error calculating Bollinger Bands: {str(e)}", "ERROR")
            
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare the final dataset."""
        try:
            # Replace infinite values with NaN
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # Forward fill NaN values for continuity
            df = df.fillna(method='ffill')
            
            # For any remaining NaN values at the beginning, backward fill
            df = df.fillna(method='bfill')
            
            # Round numerical values to reasonable precision
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].round(6)
            
            self.log("✓ Data cleaned and finalized")
            
        except Exception as e:
            self.log(f"Error cleaning data: {str(e)}", "ERROR")
            
        return df
    
    def enrich_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering steps to a DataFrame.
        
        Args:
            df: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with enriched features
        """
        # Validate input data
        is_valid, error_msg = self.validate_data(df)
        if not is_valid:
            self.log(f"Data validation failed: {error_msg}", "ERROR")
            return df
        
        # Make a copy to avoid modifying original
        enriched_df = df.copy()
        
        # Apply all feature engineering steps
        enriched_df = self.calculate_price_features(enriched_df)
        enriched_df = self.calculate_percentage_changes(enriched_df)
        enriched_df = self.calculate_moving_averages(enriched_df)
        enriched_df = self.calculate_momentum_indicators(enriched_df)
        enriched_df = self.calculate_macd(enriched_df)
        enriched_df = self.calculate_bollinger_bands(enriched_df)
        enriched_df = self.clean_data(enriched_df)
        
        # Log feature count
        original_cols = len(df.columns)
        new_cols = len(enriched_df.columns)
        added_features = new_cols - original_cols
        
        self.log(f"Feature engineering completed: {added_features} new features added", "SUCCESS")
        
        return enriched_df
    
    def enrich_file(self, file_path: str) -> bool:
        """
        Enrich a CSV file with technical indicators.
        
        Args:
            file_path: Path to the CSV file to enrich
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Validate file exists
            if not os.path.exists(file_path):
                self.log(f"File not found: {file_path}", "ERROR")
                return False
            
            # Read the CSV file
            self.log(f"Reading data from: {file_path}")
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            
            # Enrich the dataframe
            enriched_df = self.enrich_dataframe(df)
            
            # Save back to the same file
            self.log(f"Saving enriched data to: {file_path}")
            enriched_df.to_csv(file_path)
            
            self.log(f"Successfully enriched file: {file_path}", "SUCCESS")
            return True
            
        except Exception as e:
            self.log(f"Error processing file {file_path}: {str(e)}", "ERROR")
            traceback.print_exc()
            return False
    
    def enrich_multiple_files(self, file_paths: List[str]) -> Dict[str, bool]:
        """
        Enrich multiple CSV files with technical indicators.
        
        Args:
            file_paths: List of paths to CSV files to enrich
            
        Returns:
            Dict: Mapping of file_path to success status
        """
        results = {}
        
        for file_path in file_paths:
            self.log(f"Processing file {file_path}")
            results[file_path] = self.enrich_file(file_path)
        
        successful = sum(results.values())
        total = len(file_paths)
        
        self.log(f"Batch processing completed: {successful}/{total} files processed successfully", "SUCCESS")
        
        return results
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names that will be added."""
        features = []
        
        if self.config.candle_body:
            features.append('Candle_Body')
        
        if self.config.shadows:
            features.extend(['Upper_Shadow', 'Lower_Shadow'])
        
        if self.config.volatility:
            features.append('Volatility')
        
        if self.config.price_changes:
            features.append('Close_pct_change')
        
        if self.config.volume_changes:
            features.append('Volume_pct_change')
        
        if self.config.moving_averages:
            for period in self.config.ema_periods:
                features.append(f'EMA_{period}')
            for period in self.config.sma_periods:
                features.append(f'SMA_{period}')
        
        if self.config.rsi:
            features.append(f'RSI_{self.config.rsi_period}')
        
        if self.config.macd:
            features.extend(['MACD', 'MACD_Signal', 'MACD_Hist'])
        
        if self.config.bollinger:
            features.extend(['Bollinger_Upper', 'Bollinger_Lower', 'Bollinger_Mid'])
        
        return features

def create_feature_engineer(
    enable_all: bool = True,
    custom_config: Optional[Dict[str, Any]] = None
) -> FeatureEngineer:
    """
    Factory function to create a FeatureEngineer instance.
    
    Args:
        enable_all: If True, enable all features
        custom_config: Optional dictionary to override default config
        
    Returns:
        FeatureEngineer instance
    """
    config = FeatureEngineerConfig()
    
    if not enable_all:
        # Disable all features by default
        config.candle_body = False
        config.shadows = False
        config.volatility = False
        config.price_changes = False
        config.volume_changes = False
        config.moving_averages = False
        config.rsi = False
        config.macd = False
        config.bollinger = False
    
    # Apply custom configuration if provided
    if custom_config:
        for key, value in custom_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    return FeatureEngineer(config)

# Convenience functions for common use cases
def enrich_csv_file(file_path: str, config: Optional[FeatureEngineerConfig] = None) -> bool:
    """
    Convenience function to enrich a single CSV file.
    
    Args:
        file_path: Path to the CSV file
        config: Optional configuration
        
    Returns:
        bool: Success status
    """
    engineer = FeatureEngineer(config)
    return engineer.enrich_file(file_path)

def enrich_csv_files(file_paths: List[str], config: Optional[FeatureEngineerConfig] = None) -> Dict[str, bool]:
    """
    Convenience function to enrich multiple CSV files.
    
    Args:
        file_paths: List of CSV file paths
        config: Optional configuration
        
    Returns:
        Dict: Results mapping
    """
    engineer = FeatureEngineer(config)
    return engineer.enrich_multiple_files(file_paths)
