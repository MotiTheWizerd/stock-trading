#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Trader Test Script

This script demonstrates the enhanced trading agent with comprehensive
signal logging, technical indicator analysis, and trade intelligence.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from enhanced_trader import EnhancedTrader
from trade_journal import TradeJournal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def create_sample_data(ticker: str = "AAPL", days: int = 100) -> pd.DataFrame:
    """Create sample market data with technical indicators for testing."""
    
    # Generate base price data
    np.random.seed(42)
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), periods=days, freq='D')
    
    # Simulate price movement
    initial_price = 150.0
    returns = np.random.normal(0.001, 0.02, days)  # Daily returns with slight upward bias
    prices = [initial_price]
    
    for i in range(1, days):
        new_price = prices[-1] * (1 + returns[i])
        prices.append(max(new_price, 1.0))  # Ensure price doesn't go negative
    
    # Create OHLCV data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Generate realistic OHLC from close price
        volatility = abs(np.random.normal(0, 0.01))
        high = close * (1 + volatility)
        low = close * (1 - volatility)
        open_price = close + np.random.normal(0, close * 0.005)
        volume = int(np.random.normal(1000000, 200000))
        
        data.append({
            'Date': date,
            'Open': max(open_price, 1.0),
            'High': max(high, close, open_price),
            'Low': min(low, close, open_price),
            'Close': close,
            'Volume': max(volume, 1000)
        })
    
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    
    # Add technical indicators
    df = add_technical_indicators(df)
    
    return df

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to the dataframe."""
    
    # RSI
    def calculate_rsi(prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    df['RSI_14'] = calculate_rsi(df['Close'])
    
    # Moving averages
    df['EMA_5'] = df['Close'].ewm(span=5).mean()
    df['EMA_10'] = df['Close'].ewm(span=10).mean()
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    
    # MACD
    ema_12 = df['Close'].ewm(span=12).mean()
    ema_26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Bollinger Bands
    df['Bollinger_Mid'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['Bollinger_Upper'] = df['Bollinger_Mid'] + (bb_std * 2)
    df['Bollinger_Lower'] = df['Bollinger_Mid'] - (bb_std * 2)
    
    # Volatility and changes
    df['Close_pct_change'] = df['Close'].pct_change() * 100
    df['Volume_pct_change'] = df['Volume'].pct_change() * 100
    df['Volatility'] = df['Close'].rolling(window=10).std() / df['Close'].rolling(window=10).mean()
    
    # Candle patterns
    df['Candle_Body'] = df['Close'] - df['Open']
    df['Upper_Shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['Lower_Shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    
    return df

def generate_sample_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Generate sample trading signals based on technical indicators."""
    
    signals = []
    
    for idx, row in df.iterrows():
        # Skip if we don't have enough data for indicators
        if pd.isna(row['RSI_14']) or pd.isna(row['MACD']):
            signals.append({
                'signal': 'NONE',
                'confidence': 0.5,
                'risk_score': 0.5
            })
            continue
        
        # Simple signal generation logic
        signal = 'NONE'
        confidence = 0.5
        risk_score = 0.5
        
        # RSI-based signals
        rsi = row['RSI_14']
        if rsi <= 30:
            signal = 'BUY'
            confidence = min(0.9, 0.6 + (30 - rsi) / 30 * 0.3)
            risk_score = max(0.2, 0.5 - (30 - rsi) / 30 * 0.3)
        elif rsi >= 70:
            signal = 'SELL'
            confidence = min(0.9, 0.6 + (rsi - 70) / 30 * 0.3)
            risk_score = max(0.2, 0.5 - (rsi - 70) / 30 * 0.3)
        
        # MACD confirmation
        if row['MACD'] > row['MACD_Signal'] and signal == 'BUY':
            confidence = min(0.95, confidence + 0.1)
        elif row['MACD'] < row['MACD_Signal'] and signal == 'SELL':
            confidence = min(0.95, confidence + 0.1)
        
        # Volume-based risk adjustment
        if abs(row['Volume_pct_change']) > 50:
            risk_score = min(0.8, risk_score + 0.2)
        
        # Volatility-based risk adjustment
        if row['Volatility'] > 0.03:
            risk_score = min(0.9, risk_score + 0.1)
        
        signals.append({
            'signal': signal,
            'confidence': confidence,
            'risk_score': risk_score
        })
    
    return pd.DataFrame(signals)

def test_enhanced_trader():
    """Test the enhanced trader with sample data."""
    
    logger.info("🔥 Starting Enhanced Trader Test")
    logger.info("="*60)
    
    # Create sample data
    ticker = "AAPL"
    logger.info(f"Creating sample data for {ticker}...")
    df = create_sample_data(ticker, days=50)
    
    # Generate sample signals
    logger.info("Generating sample trading signals...")
    signals_df = generate_sample_signals(df)
    
    # Initialize enhanced trader
    logger.info("Initializing Enhanced Trader...")
    trader = EnhancedTrader(
        ticker=ticker,
        initial_balance=10000.0,
        journal_output_dir="test_trade_journals"
    )
    
    # Process signals with the trader
    logger.info("Processing signals with Enhanced Trader...")
    logger.info("-" * 60)
    
    results = []
    for idx, (market_row, signal_row) in enumerate(zip(df.iterrows(), signals_df.iterrows())):
        date, market_data = market_row
        _, signal_data = signal_row
        
        # Process signal with trader
        trade_result = trader.process_signal(
            data_row=market_data,
            signal=signal_data['signal'],
            confidence=signal_data['confidence'],
            risk_score=signal_data['risk_score'],
            model_metadata={
                'model_type': 'test_model',
                'signal_source': 'technical_indicators'
            }
        )
        
        # Store result
        result = {
            'date': date,
            'signal': signal_data['signal'],
            'confidence': signal_data['confidence'],
            'risk_score': signal_data['risk_score'],
            'position_action': trade_result['position_action'],
            'trade_executed': trade_result['trade_result'].get('executed', False),
            'portfolio_value': trade_result['portfolio_value'],
            'price': market_data['Close']
        }
        results.append(result)
        
        # Print progress every 10 trades
        if (idx + 1) % 10 == 0:
            logger.info(f"Processed {idx + 1} signals...")
    
    logger.info("-" * 60)
    logger.info("Trading simulation completed!")
    
    # Print comprehensive summary
    trader.print_summary()
    
    # Save results
    logger.info("\\nSaving results...")
    trades_file, journal_file = trader.save_results("test_results")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_file = Path("test_results") / f"{ticker}_test_results.csv"
    results_df.to_csv(results_file, index=False)
    
    logger.info(f"Results saved:")
    logger.info(f"  - Trading results: {results_file}")
    logger.info(f"  - Trade journal: {journal_file}")
    if trades_file:
        logger.info(f"  - Individual trades: {trades_file}")
    
    # Display journal summary
    logger.info("\\n" + "="*60)
    logger.info("TRADE JOURNAL ANALYSIS")
    logger.info("="*60)
    
    # Get journal statistics
    journal_stats = trader.journal.get_summary_stats()
    
    if journal_stats:
        logger.info(f"Total Journal Entries: {journal_stats.get('total_entries', 0)}")
        
        if 'signal_distribution' in journal_stats:
            logger.info("\\nSignal Distribution:")
            for signal, count in journal_stats['signal_distribution'].items():
                logger.info(f"  {signal}: {count}")
        
        if 'confidence_stats' in journal_stats:
            conf_stats = journal_stats['confidence_stats']
            logger.info(f"\\nConfidence Statistics:")
            logger.info(f"  Average: {conf_stats.get('mean_confidence', 0):.3f}")
            logger.info(f"  Range: {conf_stats.get('min_confidence', 0):.3f} - {conf_stats.get('max_confidence', 0):.3f}")
        
        if 'signal_strength_distribution' in journal_stats:
            logger.info("\\nSignal Strength Distribution:")
            for strength, count in journal_stats['signal_strength_distribution'].items():
                logger.info(f"  {strength}: {count}")
    
    logger.info("\\n🎉 Enhanced Trader Test Completed Successfully!")
    
    return trader, results_df

if __name__ == "__main__":
    try:
        trader, results = test_enhanced_trader()
        
        # Optional: Display some sample journal entries
        if trader.journal.entries:
            logger.info("\\n" + "="*60)
            logger.info("SAMPLE JOURNAL ENTRIES (First 3)")
            logger.info("="*60)
            
            for i, entry in enumerate(trader.journal.entries[:3]):
                logger.info(f"\\nEntry {i+1}:")
                logger.info(f"  Time: {entry['timestamp']}")
                logger.info(f"  Signal: {entry['signal']} (Confidence: {entry['confidence']:.2f})")
                logger.info(f"  Strength: {entry['signal_strength']}")
                logger.info(f"  Risk Level: {entry['risk_level']}")
                logger.info(f"  Reason: {entry['trade_reason']}")
                
                if entry['indicators']:
                    key_indicators = {k: v for k, v in entry['indicators'].items() 
                                    if k in ['RSI_14', 'MACD', 'Close']}
                    if key_indicators:
                        logger.info(f"  Key Indicators: {key_indicators}")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
