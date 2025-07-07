"""
Test script for the complete Feature Engineering Pipeline

This script demonstrates and tests the complete feature engineering pipeline
including integration with the existing download system.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import shutil
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_engineer import FeatureEngineer, FeatureEngineerConfig, create_feature_engineer
from feature_integration import FeatureIntegration, batch_enrich_unenriched
from paths import BASE_DIR, day_csv, month_dir

def create_test_data():
    """Create realistic test data for demonstration."""
    print("📊 Creating test data...")
    
    # Create sample stock data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    # Generate realistic stock prices
    np.random.seed(42)
    initial_price = 100
    
    prices = []
    for i in range(len(dates)):
        if i == 0:
            prices.append(initial_price)
        else:
            # Random walk with slight upward bias
            change = np.random.normal(0.002, 0.02)
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1))  # Ensure positive prices
    
    # Create OHLCV data
    test_data = pd.DataFrame({
        'Open': [p * np.random.uniform(0.98, 1.02) for p in prices],
        'High': [p * np.random.uniform(1.00, 1.06) for p in prices],
        'Low': [p * np.random.uniform(0.94, 1.00) for p in prices],
        'Close': prices,
        'Volume': np.random.randint(100000, 1000000, len(dates))
    }, index=dates)
    
    # Ensure realistic OHLC relationships
    for i in range(len(test_data)):
        row = test_data.iloc[i]
        # High should be max of all prices
        test_data.iloc[i, 1] = max(row['High'], row['Open'], row['Close'])
        # Low should be min of all prices
        test_data.iloc[i, 2] = min(row['Low'], row['Open'], row['Close'])
    
    return test_data

def test_feature_engineer():
    """Test the core FeatureEngineer class."""
    print("\n🔧 Testing FeatureEngineer...")
    
    # Create test data
    test_data = create_test_data()
    
    # Test default configuration
    engineer = FeatureEngineer()
    enriched = engineer.enrich_dataframe(test_data)
    
    print(f"✅ Original columns: {len(test_data.columns)}")
    print(f"✅ Enriched columns: {len(enriched.columns)}")
    print(f"✅ New features added: {len(enriched.columns) - len(test_data.columns)}")
    
    # Check for specific features
    expected_features = ['Candle_Body', 'RSI_14', 'MACD', 'EMA_5', 'Bollinger_Upper']
    for feature in expected_features:
        if feature in enriched.columns:
            print(f"✅ {feature} feature present")
        else:
            print(f"❌ {feature} feature missing")
    
    # Test custom configuration
    config = FeatureEngineerConfig()
    config.rsi = False
    config.macd = False
    
    custom_engineer = FeatureEngineer(config)
    custom_enriched = custom_engineer.enrich_dataframe(test_data)
    
    print(f"✅ Custom config columns: {len(custom_enriched.columns)}")
    
    return enriched

def test_file_processing():
    """Test file-based processing."""
    print("\n📁 Testing file processing...")
    
    # Create temporary directory structure
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test CSV file
        test_data = create_test_data()
        test_file = os.path.join(temp_dir, "test_data.csv")
        test_data.to_csv(test_file)
        
        print(f"✅ Created test file: {test_file}")
        
        # Test enrichment
        engineer = FeatureEngineer()
        success = engineer.enrich_file(test_file)
        
        if success:
            print("✅ File enrichment successful")
            
            # Read back and verify
            enriched = pd.read_csv(test_file, index_col=0, parse_dates=True)
            print(f"✅ Enriched file columns: {len(enriched.columns)}")
            
            # Check for technical indicators
            if 'RSI_14' in enriched.columns:
                print("✅ RSI indicator present")
            if 'MACD' in enriched.columns:
                print("✅ MACD indicator present")
                
        else:
            print("❌ File enrichment failed")
        
        return success

def test_integration():
    """Test integration with the existing system."""
    print("\n🔗 Testing integration...")
    
    # Create a temporary data directory structure
    with tempfile.TemporaryDirectory() as temp_dir:
        # Override BASE_DIR for testing
        original_base_dir = str(BASE_DIR)
        
        # Create test directory structure
        test_ticker = "TEST"
        test_date = "20230101"
        test_month = test_date[:6]  # 202301
        
        ticker_dir = Path(temp_dir) / test_ticker / test_month
        ticker_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test data file
        test_data = create_test_data()
        test_file = ticker_dir / f"data-{test_date}.csv"
        test_data.to_csv(test_file)
        
        print(f"✅ Created test structure: {test_file}")
        
        # Test integration
        integration = FeatureIntegration()
        
        # Temporarily override BASE_DIR
        import paths
        original_base_dir_obj = paths.BASE_DIR
        paths.BASE_DIR = Path(temp_dir)
        
        try:
            success = integration.enrich_after_download(test_ticker, test_date)
            
            if success:
                print("✅ Integration test successful")
                
                # Verify enrichment
                enriched = pd.read_csv(test_file, index_col=0, parse_dates=True)
                print(f"✅ Integrated enrichment columns: {len(enriched.columns)}")
                
            else:
                print("❌ Integration test failed")
                
        finally:
            # Restore original BASE_DIR
            paths.BASE_DIR = original_base_dir_obj
        
        return success

def test_batch_processing():
    """Test batch processing functionality."""
    print("\n📦 Testing batch processing...")
    
    # Create multiple test files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test data
        test_data = create_test_data()
        
        # Create multiple test files
        test_files = []
        for i in range(3):
            test_file = os.path.join(temp_dir, f"test_data_{i}.csv")
            test_data.to_csv(test_file)
            test_files.append(test_file)
        
        print(f"✅ Created {len(test_files)} test files")
        
        # Test batch processing
        engineer = FeatureEngineer()
        results = engineer.enrich_multiple_files(test_files)
        
        successful = sum(results.values())
        total = len(results)
        
        print(f"✅ Batch processing: {successful}/{total} files processed")
        
        # Verify results
        for file_path, success in results.items():
            if success:
                enriched = pd.read_csv(file_path, index_col=0, parse_dates=True)
                print(f"✅ {os.path.basename(file_path)}: {len(enriched.columns)} columns")
            else:
                print(f"❌ {os.path.basename(file_path)}: Failed")
        
        return successful == total

def test_feature_names():
    """Test feature name generation."""
    print("\n🏷️ Testing feature names...")
    
    engineer = FeatureEngineer()
    feature_names = engineer.get_feature_names()
    
    print(f"✅ Total features available: {len(feature_names)}")
    print("✅ Feature categories:")
    
    categories = {
        'Price Features': ['Candle_Body', 'Upper_Shadow', 'Lower_Shadow', 'Volatility'],
        'Change Features': ['Close_pct_change', 'Volume_pct_change'],
        'Moving Averages': ['EMA_5', 'EMA_10', 'SMA_5', 'SMA_10'],
        'Momentum': ['RSI_14'],
        'MACD': ['MACD', 'MACD_Signal', 'MACD_Hist'],
        'Bollinger': ['Bollinger_Upper', 'Bollinger_Lower', 'Bollinger_Mid']
    }
    
    for category, features in categories.items():
        present = [f for f in features if f in feature_names]
        print(f"   {category}: {len(present)}/{len(features)} features")
    
    return len(feature_names) > 0

def test_data_quality():
    """Test data quality and validation."""
    print("\n🔍 Testing data quality...")
    
    # Test with good data
    good_data = create_test_data()
    engineer = FeatureEngineer()
    
    is_valid, error = engineer.validate_data(good_data)
    print(f"✅ Good data validation: {'Pass' if is_valid else 'Fail'}")
    
    # Test with bad data
    bad_data = pd.DataFrame({
        'Open': [100, 101],
        'Close': [102, 103]
    })
    
    is_valid, error = engineer.validate_data(bad_data)
    print(f"✅ Bad data validation: {'Pass' if not is_valid else 'Fail'}")
    
    # Test data cleaning
    dirty_data = good_data.copy()
    dirty_data.iloc[10, 0] = np.inf
    dirty_data.iloc[15, 1] = np.nan
    
    cleaned = engineer.clean_data(dirty_data)
    
    has_inf = np.isinf(cleaned.values).any()
    has_excessive_nan = (cleaned.isna().sum().sum() / (len(cleaned) * len(cleaned.columns))) > 0.3
    
    print(f"✅ Data cleaning: {'Pass' if not has_inf and not has_excessive_nan else 'Fail'}")
    
    return True

def run_comprehensive_test():
    """Run all tests comprehensively."""
    print("🚀 Starting comprehensive Feature Engineering Pipeline test...")
    print("=" * 60)
    
    test_results = {}
    
    # Run all tests
    try:
        test_results['feature_engineer'] = test_feature_engineer() is not None
        test_results['file_processing'] = test_file_processing()
        test_results['integration'] = test_integration()
        test_results['batch_processing'] = test_batch_processing()
        test_results['feature_names'] = test_feature_names()
        test_results['data_quality'] = test_data_quality()
        
    except Exception as e:
        print(f"❌ Test error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Feature Engineering Pipeline is ready to use.")
        return True
    else:
        print("⚠️ Some tests failed. Please review the output above.")
        return False

def demonstrate_features():
    """Demonstrate the feature engineering capabilities."""
    print("\n🎪 Feature Engineering Pipeline Demonstration")
    print("=" * 60)
    
    # Create sample data
    print("📊 Creating sample stock data...")
    sample_data = create_test_data()
    print(f"✅ Created {len(sample_data)} days of OHLCV data")
    
    # Show original data
    print("\n📋 Original data columns:")
    for i, col in enumerate(sample_data.columns, 1):
        print(f"   {i}. {col}")
    
    # Apply feature engineering
    print("\n🔧 Applying feature engineering...")
    engineer = FeatureEngineer()
    enriched_data = engineer.enrich_dataframe(sample_data)
    
    # Show enriched data
    print(f"\n✨ Enriched data columns ({len(enriched_data.columns)} total):")
    new_features = [col for col in enriched_data.columns if col not in sample_data.columns]
    
    for i, col in enumerate(enriched_data.columns, 1):
        marker = "🆕" if col in new_features else "📊"
        print(f"   {i:2d}. {marker} {col}")
    
    # Show feature categories
    print(f"\n🏷️ New features added ({len(new_features)} total):")
    categories = {
        'Price Analysis': ['Candle_Body', 'Upper_Shadow', 'Lower_Shadow', 'Volatility'],
        'Market Dynamics': ['Close_pct_change', 'Volume_pct_change'],
        'Trend Following': ['EMA_5', 'EMA_10', 'SMA_5', 'SMA_10'],
        'Momentum': ['RSI_14'],
        'Convergence/Divergence': ['MACD', 'MACD_Signal', 'MACD_Hist'],
        'Volatility Bands': ['Bollinger_Upper', 'Bollinger_Lower', 'Bollinger_Mid']
    }
    
    for category, features in categories.items():
        present_features = [f for f in features if f in new_features]
        if present_features:
            print(f"\n   {category}:")
            for feature in present_features:
                print(f"      • {feature}")
    
    # Show sample values
    print(f"\n📈 Sample technical indicator values (last 5 days):")
    display_cols = ['Close', 'RSI_14', 'MACD', 'EMA_5', 'Bollinger_Upper']
    display_data = enriched_data[display_cols].tail(5).round(2)
    print(display_data.to_string())
    
    print(f"\n🎯 Feature engineering pipeline ready for model training!")
    
    return enriched_data

if __name__ == "__main__":
    print("🔥 Feature Engineering Pipeline Test Suite")
    print("=" * 60)
    
    # Check if running tests or demonstration
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demonstrate_features()
    else:
        # Run comprehensive tests
        success = run_comprehensive_test()
        
        if success:
            print("\n🎪 Running demonstration...")
            demonstrate_features()
        else:
            print("\n❌ Tests failed. Skipping demonstration.")
            sys.exit(1)
    
    print("\n✅ Feature Engineering Pipeline setup complete!")
    print("💡 To use in your code:")
    print("   from feature_engineer import FeatureEngineer")
    print("   engineer = FeatureEngineer()")
    print("   enriched_data = engineer.enrich_dataframe(your_data)")
    print("   # or")
    print("   success = engineer.enrich_file('path/to/your/file.csv')")
