import os
import pandas as pd
from datetime import datetime


def detect_rsi_signals(df: pd.DataFrame, ticker: str):
    """Detect RSI-based BUY/SELL signals and persist them.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that **already** contains an 'RSI' column.
    ticker : str
        Ticker symbol (used for logging and for the CSV path).

    Returns
    -------
    tuple[pd.Series | None, pd.Series | None, pd.DataFrame | None]
        buy_mask, sell_mask, and a DataFrame of the generated/updated signals
        (or (None, None, None) if no signals were found).
    """
    if 'RSI' not in df.columns:
        raise ValueError("DataFrame must contain an 'RSI' column before calling detect_rsi_signals()")

    # Debug: Show RSI values
    print(f"RSI values for {ticker}:")
    print(f"  Min RSI: {df['RSI'].min():.2f}")
    print(f"  Max RSI: {df['RSI'].max():.2f}")
    print(f"  Mean RSI: {df['RSI'].mean():.2f}")
    print(f"  RSI < 30 count: {(df['RSI'] < 30).sum()}")
    print(f"  RSI > 70 count: {(df['RSI'] > 70).sum()}")
    print(f"  RSI < 45 count: {(df['RSI'] < 45).sum()}")
    print(f"  RSI > 55 count: {(df['RSI'] > 55).sum()}")

    buy_mask = df['RSI'] < 45  # oversold (very sensitive for testing)
    sell_mask = df['RSI'] > 55  # overbought (very sensitive for testing)

    signal_rows = []
    for i, idx in enumerate(df.index):
        if buy_mask.iloc[i] or sell_mask.iloc[i]:
            try:
                # Get current row data using iloc for safer access
                current_close = df['Close'].iloc[i]
                current_open = df['Open'].iloc[i]
                current_high = df['High'].iloc[i]
                current_low = df['Low'].iloc[i]
                current_volume = df['Volume'].iloc[i]
                current_rsi = df['RSI'].iloc[i]
                
                # Calculate price changes and percentages
                prev_close = df['Close'].iloc[i-1] if i > 0 else current_close
                price_change = current_close - prev_close
                price_change_pct = (price_change / prev_close * 100) if prev_close != 0 else 0
            
                # Calculate volatility (rolling std of last 5 periods)
                if i >= 4:
                    volatility = df['Close'].iloc[max(0, i-4):i+1].std()
                else:
                    volatility = 0
                
                # Calculate volume ratio (current vs average)
                if i >= 9:
                    avg_volume = df['Volume'].iloc[max(0, i-9):i+1].mean()
                    volume_ratio = current_volume / avg_volume if avg_volume != 0 else 1
                else:
                    volume_ratio = 1
                
                # Get MACD values if available
                macd_value = df['MACD'].iloc[i] if 'MACD' in df.columns else 0
                macd_signal_value = df['MACD_signal'].iloc[i] if 'MACD_signal' in df.columns else 0
                macd_histogram = macd_value - macd_signal_value
                
                # Determine signal type
                signal_type = 'BUY' if buy_mask.iloc[i] else 'SELL'
                
                # Create comprehensive signal record
                signal_data = {
                    'datetime': idx,
                    'ticker': ticker,
                    'signal': signal_type,
                    'rsi': round(float(current_rsi), 4),
                    'price': round(float(current_close), 4),
                    'open': round(float(current_open), 4),
                    'high': round(float(current_high), 4),
                    'low': round(float(current_low), 4),
                    'volume': int(current_volume),
                    'price_change': round(float(price_change), 4),
                    'price_change_pct': round(float(price_change_pct), 4),
                    'volatility': round(float(volatility), 4),
                    'volume_ratio': round(float(volume_ratio), 4),
                    'macd': round(float(macd_value), 4),
                    'macd_signal': round(float(macd_signal_value), 4),
                    'macd_histogram': round(float(macd_histogram), 4),
                    'market_cap_proxy': round(float(current_close * current_volume), 2),
                    'intraday_range': round(float((current_high - current_low) / current_close * 100), 4),
                    'gap_from_open': round(float((current_close - current_open) / current_open * 100), 4),
                    'signal_strength': 'STRONG' if (signal_type == 'BUY' and current_rsi < 40) or (signal_type == 'SELL' and current_rsi > 60) else 'WEAK',
                    'market_session': 'REGULAR',
                    'trend_direction': 'UP' if price_change > 0 else 'DOWN' if price_change < 0 else 'FLAT'
                }
                
                signal_rows.append(signal_data)
                
            except Exception as e:
                print(f"❌ Error processing signal at index {i}: {str(e)}")
                continue

    if not signal_rows:
        print(f"🔕 No RSI signals for {ticker}")
        return None, None, None

    signal_df = pd.DataFrame(signal_rows)
    
    # Get today's date folder in YYYYMMDD format
    date_folder = datetime.now().strftime("%Y%m%d")
    
    # Create path with date folder structure
    signal_path = os.path.join('data', ticker, date_folder, 'signals.csv')
    os.makedirs(os.path.dirname(signal_path), exist_ok=True)

    # Merge with existing to prevent duplicates
    if os.path.exists(signal_path):
        existing = pd.read_csv(signal_path, parse_dates=['datetime'])
        combined = pd.concat([existing, signal_df], ignore_index=True)
        combined.drop_duplicates(subset=['datetime', 'signal'], inplace=True)
        combined.sort_values('datetime', inplace=True)
        combined.to_csv(signal_path, index=False)
        print(f"🔔 Updated {len(signal_rows)} RSI signals for {ticker}")
    else:
        signal_df.to_csv(signal_path, index=False)
        print(f"🔔 Saved {len(signal_rows)} new RSI signals for {ticker}")

    return buy_mask, sell_mask, signal_df
