"""
Unit tests for the Feature Engineering Pipeline

This module provides comprehensive tests for the FeatureEngineer class
to ensure all technical indicators are calculated correctly.
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from datetime import datetime, timedelta
from feature_engineer import FeatureEngineer, FeatureEngineerConfig, create_feature_engineer

class TestFeatureEngineer:
    """Test class for FeatureEngineer functionality."""
    
    def setup_method(self):
        """Setup test data before each test."""
        # Create sample OHLCV data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        n_days = len(dates)
        
        # Generate realistic stock data
        np.random.seed(42)  # For reproducible tests
        
        # Generate realistic price series
        initial_price = 100
        returns = np.random.normal(0.001, 0.02, n_days)
        prices = [initial_price]
        
        for i in range(1, n_days):
            prices.append(prices[-1] * (1 + returns[i]))
        
        # Create OHLCV data
        self.sample_data = pd.DataFrame({
            'Open': [p * np.random.uniform(0.99, 1.01) for p in prices],
            'High': [p * np.random.uniform(1.00, 1.05) for p in prices],
            'Low': [p * np.random.uniform(0.95, 1.00) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, n_days)
        }, index=dates)
        
        # Ensure High >= max(Open, Close) and Low <= min(Open, Close)
        for i in range(len(self.sample_data)):
            row = self.sample_data.iloc[i]
            self.sample_data.iloc[i, 1] = max(row['High'], row['Open'], row['Close'])  # High
            self.sample_data.iloc[i, 2] = min(row['Low'], row['Open'], row['Close'])   # Low
    
    def test_initialization(self):
        """Test FeatureEngineer initialization."""
        # Test default initialization
        engineer = FeatureEngineer()
        assert engineer.config is not None
        assert isinstance(engineer.config, FeatureEngineerConfig)
        
        # Test custom config initialization
        config = FeatureEngineerConfig()
        config.rsi = False
        engineer = FeatureEngineer(config)
        assert engineer.config.rsi == False
    
    def test_validate_data(self):
        """Test data validation functionality."""
        engineer = FeatureEngineer()
        
        # Test valid data
        is_valid, error = engineer.validate_data(self.sample_data)
        assert is_valid == True
        assert error == ""
        
        # Test empty DataFrame
        empty_df = pd.DataFrame()
        is_valid, error = engineer.validate_data(empty_df)
        assert is_valid == False
        assert "empty" in error.lower()
        
        # Test missing columns
        incomplete_df = self.sample_data[['Open', 'Close']].copy()
        is_valid, error = engineer.validate_data(incomplete_df)
        assert is_valid == False
        assert "missing" in error.lower()
        
        # Test insufficient data
        small_df = self.sample_data.head(10)
        is_valid, error = engineer.validate_data(small_df)
        assert is_valid == False
        assert "insufficient" in error.lower()
    
    def test_price_features(self):
        """Test price-based feature calculations."""
        engineer = FeatureEngineer()
        enriched = engineer.calculate_price_features(self.sample_data.copy())
        
        # Check that new columns are added
        assert 'Candle_Body' in enriched.columns
        assert 'Upper_Shadow' in enriched.columns
        assert 'Lower_Shadow' in enriched.columns
        assert 'Volatility' in enriched.columns
        
        # Check calculations are correct
        row = enriched.iloc[0]
        expected_candle_body = abs(row['Close'] - row['Open'])
        assert abs(row['Candle_Body'] - expected_candle_body) < 1e-6
        
        expected_volatility = row['High'] - row['Low']
        assert abs(row['Volatility'] - expected_volatility) < 1e-6
    
    def test_percentage_changes(self):
        """Test percentage change calculations."""
        engineer = FeatureEngineer()
        enriched = engineer.calculate_percentage_changes(self.sample_data.copy())
        
        # Check that new columns are added
        assert 'Close_pct_change' in enriched.columns
        assert 'Volume_pct_change' in enriched.columns
        
        # Check first row is NaN (no previous value)
        assert pd.isna(enriched.iloc[0]['Close_pct_change'])
        assert pd.isna(enriched.iloc[0]['Volume_pct_change'])
        
        # Check calculation for second row
        row1 = enriched.iloc[1]
        row0 = enriched.iloc[0]
        expected_close_pct = ((row1['Close'] - row0['Close']) / row0['Close']) * 100
        assert abs(row1['Close_pct_change'] - expected_close_pct) < 1e-6
    
    def test_moving_averages(self):
        """Test moving average calculations."""
        engineer = FeatureEngineer()
        enriched = engineer.calculate_moving_averages(self.sample_data.copy())
        
        # Check that new columns are added
        assert 'EMA_5' in enriched.columns
        assert 'EMA_10' in enriched.columns
        assert 'SMA_5' in enriched.columns
        assert 'SMA_10' in enriched.columns
        
        # Check values are reasonable (not NaN after warmup period)
        assert not pd.isna(enriched.iloc[20]['EMA_5'])
        assert not pd.isna(enriched.iloc[20]['SMA_5'])
    
    def test_momentum_indicators(self):
        """Test momentum indicator calculations."""
        engineer = FeatureEngineer()
        enriched = engineer.calculate_momentum_indicators(self.sample_data.copy())
        
        # Check RSI column is added
        assert 'RSI_14' in enriched.columns
        
        # Check RSI values are in valid range (0-100)
        rsi_values = enriched['RSI_14'].dropna()
        assert all(0 <= val <= 100 for val in rsi_values)
    
    def test_macd(self):
        """Test MACD calculations."""
        engineer = FeatureEngineer()
        enriched = engineer.calculate_macd(self.sample_data.copy())
        
        # Check MACD columns are added
        assert 'MACD' in enriched.columns
        assert 'MACD_Signal' in enriched.columns
        assert 'MACD_Hist' in enriched.columns
        
        # Check histogram is difference of MACD and Signal
        row = enriched.iloc[50]  # After warmup period
        if not pd.isna(row['MACD']) and not pd.isna(row['MACD_Signal']):
            expected_hist = row['MACD'] - row['MACD_Signal']
            assert abs(row['MACD_Hist'] - expected_hist) < 1e-6
    
    def test_bollinger_bands(self):
        """Test Bollinger Bands calculations."""
        engineer = FeatureEngineer()
        enriched = engineer.calculate_bollinger_bands(self.sample_data.copy())
        
        # Check Bollinger columns are added
        assert 'Bollinger_Upper' in enriched.columns
        assert 'Bollinger_Lower' in enriched.columns
        assert 'Bollinger_Mid' in enriched.columns
        
        # Check band relationships (Upper > Mid > Lower)
        row = enriched.iloc[50]  # After warmup period
        if not pd.isna(row['Bollinger_Upper']) and not pd.isna(row['Bollinger_Lower']):
            assert row['Bollinger_Upper'] > row['Bollinger_Mid']
            assert row['Bollinger_Mid'] > row['Bollinger_Lower']
    
    def test_full_enrichment(self):
        """Test full feature enrichment pipeline."""
        engineer = FeatureEngineer()
        enriched = engineer.enrich_dataframe(self.sample_data.copy())
        
        # Check that original columns are preserved
        for col in self.sample_data.columns:
            assert col in enriched.columns
        
        # Check that new features are added
        original_cols = len(self.sample_data.columns)
        new_cols = len(enriched.columns)
        assert new_cols > original_cols
        
        # Check data integrity
        assert len(enriched) == len(self.sample_data)
        assert enriched.index.equals(self.sample_data.index)
    
    def test_file_processing(self):
        """Test file-based processing."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.sample_data.to_csv(f.name)
            temp_file = f.name
        
        try:
            # Process the file
            engineer = FeatureEngineer()
            success = engineer.enrich_file(temp_file)
            assert success == True
            
            # Read back and verify
            enriched = pd.read_csv(temp_file, index_col=0, parse_dates=True)
            assert len(enriched.columns) > len(self.sample_data.columns)
            assert 'RSI_14' in enriched.columns
            assert 'MACD' in enriched.columns
            
        finally:
            # Clean up
            os.unlink(temp_file)
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = FeatureEngineerConfig()
        config.rsi = False
        config.macd = False
        config.bollinger = False
        
        engineer = FeatureEngineer(config)
        enriched = engineer.enrich_dataframe(self.sample_data.copy())
        
        # These indicators should NOT be present
        assert 'RSI_14' not in enriched.columns
        assert 'MACD' not in enriched.columns
        assert 'Bollinger_Upper' not in enriched.columns
        
        # But price features should still be there
        assert 'Candle_Body' in enriched.columns
        assert 'Volatility' in enriched.columns
    
    def test_feature_names(self):
        """Test feature name generation."""
        engineer = FeatureEngineer()
        feature_names = engineer.get_feature_names()
        
        # Should contain expected features
        assert 'Candle_Body' in feature_names
        assert 'RSI_14' in feature_names
        assert 'MACD' in feature_names
        assert 'EMA_5' in feature_names
        assert 'SMA_10' in feature_names
    
    def test_factory_function(self):
        """Test factory function for creating engineers."""
        # Test with all features enabled
        engineer_all = create_feature_engineer(enable_all=True)
        assert engineer_all.config.rsi == True
        assert engineer_all.config.macd == True
        
        # Test with custom config
        custom_config = {'rsi': False, 'macd': False}
        engineer_custom = create_feature_engineer(custom_config=custom_config)
        assert engineer_custom.config.rsi == False
        assert engineer_custom.config.macd == False
    
    def test_data_cleaning(self):
        """Test data cleaning functionality."""
        # Create data with problematic values
        dirty_data = self.sample_data.copy()
        dirty_data.iloc[10, 0] = np.inf  # Add infinity
        dirty_data.iloc[15, 1] = np.nan  # Add NaN
        
        engineer = FeatureEngineer()
        cleaned = engineer.clean_data(dirty_data)
        
        # Check that infinite values are handled
        assert not np.isinf(cleaned.iloc[10, 0])
        
        # Check that NaN values are handled
        assert not pd.isna(cleaned.iloc[15, 1])

def test_integration():
    """Integration test to verify complete pipeline."""
    # Create test data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    test_data = pd.DataFrame({
        'Open': np.random.uniform(95, 105, 100),
        'High': np.random.uniform(100, 110, 100),
        'Low': np.random.uniform(90, 100, 100),
        'Close': np.random.uniform(95, 105, 100),
        'Volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)
    
    # Ensure data consistency
    for i in range(len(test_data)):
        row = test_data.iloc[i]
        test_data.iloc[i, 1] = max(row['High'], row['Open'], row['Close'])
        test_data.iloc[i, 2] = min(row['Low'], row['Open'], row['Close'])
    
    # Test complete pipeline
    engineer = FeatureEngineer()
    enriched = engineer.enrich_dataframe(test_data)
    
    # Verify results
    assert len(enriched) == len(test_data)
    assert len(enriched.columns) > len(test_data.columns)
    
    # Verify no infinite or NaN values in final result
    assert not enriched.isin([np.inf, -np.inf]).any().any()
    
    # Check that most values are not NaN (some may be at the beginning due to indicators)
    non_nan_ratio = enriched.notna().sum().sum() / (len(enriched) * len(enriched.columns))
    assert non_nan_ratio > 0.8  # At least 80% of values should be non-NaN

if __name__ == "__main__":
    # Run tests manually
    test = TestFeatureEngineer()
    test.setup_method()
    
    print("🧪 Running Feature Engineer Tests...")
    
    try:
        test.test_initialization()
        print("✅ Initialization test passed")
        
        test.test_validate_data()
        print("✅ Data validation test passed")
        
        test.test_price_features()
        print("✅ Price features test passed")
        
        test.test_percentage_changes()
        print("✅ Percentage changes test passed")
        
        test.test_moving_averages()
        print("✅ Moving averages test passed")
        
        test.test_momentum_indicators()
        print("✅ Momentum indicators test passed")
        
        test.test_macd()
        print("✅ MACD test passed")
        
        test.test_bollinger_bands()
        print("✅ Bollinger Bands test passed")
        
        test.test_full_enrichment()
        print("✅ Full enrichment test passed")
        
        test.test_file_processing()
        print("✅ File processing test passed")
        
        test.test_custom_config()
        print("✅ Custom config test passed")
        
        test.test_feature_names()
        print("✅ Feature names test passed")
        
        test.test_factory_function()
        print("✅ Factory function test passed")
        
        test.test_data_cleaning()
        print("✅ Data cleaning test passed")
        
        test_integration()
        print("✅ Integration test passed")
        
        print("\n🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
