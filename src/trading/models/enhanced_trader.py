"""
Enhanced Trader Module - Intelligent Trading with Comprehensive Logging

This module provides an enhanced trading agent that logs all decisions,
technical indicators, and reasoning for maximum transparency and intelligence.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import logging
from pathlib import Path

from trading.utils.trade_journal import TradeJournal

logger = logging.getLogger(__name__)

class EnhancedTrader:
    """
    Enhanced trading agent with comprehensive decision logging and intelligence.
    """
    
    def __init__(self, 
                 ticker: str,
                 initial_balance: float = 10000.0,
                 transaction_cost: float = 0.001,
                 journal_output_dir: str = "trade_journals"):
        
        self.ticker = ticker
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.transaction_cost = transaction_cost
        
        # Position tracking
        self.position = 0  # Number of shares
        self.position_value = 0.0
        self.last_price = 0.0
        
        # Trade tracking
        self.trades = []
        self.trade_history = []
        
        # Initialize trade journal
        self.journal = TradeJournal(ticker, journal_output_dir)
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        
        logger.info(f"Enhanced Trader initialized for {ticker} with ${initial_balance:,.2f}")
    
    def process_signal(self, 
                      data_row: pd.Series,
                      signal: str,
                      confidence: float,
                      risk_score: Optional[float] = None,
                      model_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a trading signal with full logging and decision tracking.
        
        Args:
            data_row: Current market data row
            signal: BUY, SELL, or NONE
            confidence: Model confidence (0-1)
            risk_score: Risk assessment (0-1)
            model_metadata: Additional model information
            
        Returns:
            Dict with trade execution results and logging info
        """
        
        timestamp = data_row.name if hasattr(data_row, 'name') else datetime.now()
        current_price = data_row.get('Close', data_row.get('close', 0))
        
        # Extract technical indicators from the data row
        indicators = self._extract_indicators(data_row)
        
        # Determine position action
        position_action = self._determine_position_action(signal, confidence, risk_score)
        
        # Generate trade reasoning
        trade_reason = self._generate_enhanced_reasoning(
            signal, confidence, risk_score, indicators, data_row, model_metadata
        )
        
        # Execute the trade
        trade_result = self._execute_trade(
            position_action, current_price, timestamp, data_row
        )
        
        # Log the decision to journal
        self.journal.log_signal_decision(
            timestamp=timestamp,
            signal=signal,
            confidence=confidence,
            risk_score=risk_score,
            indicators=indicators,
            price=current_price,
            position_taken=position_action,
            trade_reason=trade_reason,
            additional_data={
                'model_metadata': model_metadata,
                'trade_result': trade_result,
                'portfolio_value': self.get_portfolio_value(current_price)
            }
        )
        
        # Update tracking
        self.last_price = current_price
        
        return {
            'signal': signal,
            'position_action': position_action,
            'confidence': confidence,
            'risk_score': risk_score,
            'trade_result': trade_result,
            'indicators': indicators,
            'trade_reason': trade_reason,
            'portfolio_value': self.get_portfolio_value(current_price)
        }
    
    def _extract_indicators(self, data_row: pd.Series) -> Dict[str, float]:
        """Extract technical indicators from data row."""
        indicators = {}
        
        # Standard price/volume indicators
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in data_row:
                indicators[col] = float(data_row[col])
            elif col.lower() in data_row:
                indicators[col] = float(data_row[col.lower()])
        
        # Technical indicators
        indicator_columns = [
            'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist',
            'Bollinger_Upper', 'Bollinger_Lower', 'Bollinger_Mid',
            'EMA_5', 'EMA_10', 'SMA_5', 'SMA_10',
            'Volatility', 'Close_pct_change', 'Volume_pct_change',
            'Candle_Body', 'Upper_Shadow', 'Lower_Shadow'
        ]
        
        for col in indicator_columns:
            if col in data_row and pd.notna(data_row[col]):
                indicators[col] = float(data_row[col])
        
        return indicators
    
    def _determine_position_action(self, signal: str, confidence: float, risk_score: Optional[float]) -> str:
        """Determine the actual position action based on signal and risk."""
        
        # Risk-based filtering
        if risk_score and risk_score > 0.8:
            logger.debug(f"High risk ({risk_score:.2f}) - reducing signal strength")
            if signal in ['BUY', 'SELL'] and confidence < 0.9:
                return 'NONE'  # Skip high-risk, low-confidence trades
        
        # Confidence-based filtering
        if confidence < 0.5:
            return 'NONE'  # Skip low-confidence signals
        
        # Position size based on confidence and risk
        if signal == 'BUY':
            if self.position <= 0:  # Can buy
                return 'BUY'
            else:
                return 'NONE'  # Already long
        
        elif signal == 'SELL':
            if self.position > 0:  # Can sell
                return 'SELL'
            else:
                return 'NONE'  # No position to sell
        
        return 'NONE'
    
    def _generate_enhanced_reasoning(self, 
                                   signal: str, 
                                   confidence: float, 
                                   risk_score: Optional[float],
                                   indicators: Dict[str, float],
                                   data_row: pd.Series,
                                   model_metadata: Optional[Dict[str, Any]]) -> str:
        """Generate comprehensive trade reasoning."""
        
        reasons = []
        
        # Model-based reasoning
        reasons.append(f"Model signal: {signal} (confidence: {confidence:.2f})")
        
        if risk_score:
            reasons.append(f"Risk assessment: {risk_score:.2f}")
        
        # Technical analysis reasoning
        tech_reasons = self._analyze_technical_context(indicators)
        reasons.extend(tech_reasons)
        
        # Market context reasoning
        market_reasons = self._analyze_market_context(data_row, indicators)
        reasons.extend(market_reasons)
        
        # Position context
        position_context = self._analyze_position_context()
        if position_context:
            reasons.append(position_context)
        
        # Model metadata insights
        if model_metadata:
            meta_reasons = self._analyze_model_metadata(model_metadata)
            reasons.extend(meta_reasons)
        
        return " | ".join(reasons)
    
    def _analyze_technical_context(self, indicators: Dict[str, float]) -> List[str]:
        """Analyze technical indicators for context."""
        reasons = []
        
        # RSI analysis
        if 'RSI_14' in indicators:
            rsi = indicators['RSI_14']
            if rsi <= 25:
                reasons.append(f"RSI extremely oversold ({rsi:.1f})")
            elif rsi <= 30:
                reasons.append(f"RSI oversold ({rsi:.1f})")
            elif rsi >= 75:
                reasons.append(f"RSI extremely overbought ({rsi:.1f})")
            elif rsi >= 70:
                reasons.append(f"RSI overbought ({rsi:.1f})")
        
        # MACD analysis
        if all(k in indicators for k in ['MACD', 'MACD_Signal']):
            macd = indicators['MACD']
            macd_signal = indicators['MACD_Signal']
            macd_diff = macd - macd_signal
            
            if macd_diff > 0.01:
                reasons.append("MACD strong bullish")
            elif macd_diff > 0:
                reasons.append("MACD bullish crossover")
            elif macd_diff < -0.01:
                reasons.append("MACD strong bearish")
            elif macd_diff < 0:
                reasons.append("MACD bearish crossover")
        
        # Bollinger Bands analysis
        if all(k in indicators for k in ['Close', 'Bollinger_Upper', 'Bollinger_Lower']):
            close = indicators['Close']
            bb_upper = indicators['Bollinger_Upper']
            bb_lower = indicators['Bollinger_Lower']
            bb_width = bb_upper - bb_lower
            
            if close <= bb_lower:
                reasons.append("Price at/below lower BB")
            elif close >= bb_upper:
                reasons.append("Price at/above upper BB")
            
            # Bollinger squeeze detection
            if bb_width < (close * 0.05):  # Less than 5% of price
                reasons.append("Bollinger squeeze detected")
        
        # Moving average analysis
        if all(k in indicators for k in ['Close', 'EMA_5', 'EMA_10']):
            close = indicators['Close']
            ema5 = indicators['EMA_5']
            ema10 = indicators['EMA_10']
            
            if ema5 > ema10 and close > ema5:
                reasons.append("Strong uptrend (EMA alignment)")
            elif ema5 < ema10 and close < ema5:
                reasons.append("Strong downtrend (EMA alignment)")
        
        return reasons
    
    def _analyze_market_context(self, data_row: pd.Series, indicators: Dict[str, float]) -> List[str]:
        """Analyze market context and volatility."""
        reasons = []
        
        # Volume analysis
        if 'Volume_pct_change' in indicators:
            vol_change = indicators['Volume_pct_change']
            if vol_change > 100:
                reasons.append(f"Exceptional volume spike (+{vol_change:.0f}%)")
            elif vol_change > 50:
                reasons.append(f"High volume (+{vol_change:.0f}%)")
            elif vol_change < -50:
                reasons.append(f"Low volume ({vol_change:.0f}%)")
        
        # Price volatility
        if 'Volatility' in indicators:
            volatility = indicators['Volatility']
            if volatility > 0.05:  # 5% volatility
                reasons.append(f"High volatility ({volatility:.1%})")
            elif volatility < 0.01:  # 1% volatility
                reasons.append(f"Low volatility ({volatility:.1%})")
        
        # Price change analysis
        if 'Close_pct_change' in indicators:
            price_change = indicators['Close_pct_change']
            if abs(price_change) > 5:
                direction = "surge" if price_change > 0 else "drop"
                reasons.append(f"Price {direction} ({price_change:+.1f}%)")
        
        # Candle pattern analysis
        if all(k in indicators for k in ['Candle_Body', 'Upper_Shadow', 'Lower_Shadow']):
            body = indicators['Candle_Body']
            upper_shadow = indicators['Upper_Shadow']
            lower_shadow = indicators['Lower_Shadow']
            
            # Doji pattern
            if abs(body) < 0.001:
                reasons.append("Doji candle pattern")
            
            # Hammer/Shooting star patterns
            total_range = abs(body) + upper_shadow + lower_shadow
            if total_range > 0:
                if lower_shadow > 2 * abs(body) and upper_shadow < abs(body):
                    reasons.append("Hammer pattern")
                elif upper_shadow > 2 * abs(body) and lower_shadow < abs(body):
                    reasons.append("Shooting star pattern")
        
        return reasons
    
    def _analyze_position_context(self) -> Optional[str]:
        """Analyze current position context."""
        if self.position > 0:
            unrealized_pnl = (self.last_price * self.position) - self.position_value
            pnl_pct = (unrealized_pnl / self.position_value) * 100 if self.position_value > 0 else 0
            return f"Long position: {self.position} shares, P&L: {pnl_pct:+.1f}%"
        elif self.position < 0:
            unrealized_pnl = self.position_value - (self.last_price * abs(self.position))
            pnl_pct = (unrealized_pnl / abs(self.position_value)) * 100 if self.position_value > 0 else 0
            return f"Short position: {abs(self.position)} shares, P&L: {pnl_pct:+.1f}%"
        else:
            return "No position"
    
    def _analyze_model_metadata(self, metadata: Dict[str, Any]) -> List[str]:
        """Analyze model metadata for additional insights."""
        reasons = []
        
        if 'feature_importance' in metadata:
            # Get top contributing features
            importance = metadata['feature_importance']
            if isinstance(importance, dict):
                top_features = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
                feature_str = ", ".join([f"{k}:{v:.2f}" for k, v in top_features])
                reasons.append(f"Key features: {feature_str}")
        
        if 'prediction_probability' in metadata:
            prob = metadata['prediction_probability']
            if isinstance(prob, (list, tuple)) and len(prob) >= 2:
                reasons.append(f"Class probabilities: {prob[0]:.2f}/{prob[1]:.2f}")
        
        if 'model_version' in metadata:
            reasons.append(f"Model: {metadata['model_version']}")
        
        return reasons
    
    def _execute_trade(self, 
                      action: str, 
                      price: float, 
                      timestamp: datetime,
                      data_row: pd.Series) -> Dict[str, Any]:
        """Execute the actual trade and update positions."""
        
        if action == 'NONE':
            return {'action': 'NONE', 'executed': False, 'reason': 'No action taken'}
        
        # Calculate position size (for now, use fixed size)
        position_size = self._calculate_position_size(action, price)
        
        if position_size == 0:
            return {'action': action, 'executed': False, 'reason': 'Insufficient funds or position'}
        
        # Calculate transaction cost
        transaction_value = position_size * price
        transaction_fee = transaction_value * self.transaction_cost
        
        trade_result = {
            'action': action,
            'executed': True,
            'price': price,
            'quantity': position_size,
            'value': transaction_value,
            'fee': transaction_fee,
            'timestamp': timestamp
        }
        
        if action == 'BUY':
            # Buy shares
            total_cost = transaction_value + transaction_fee
            if self.current_balance >= total_cost:
                self.current_balance -= total_cost
                self.position += position_size
                self.position_value += transaction_value
                
                trade_result['new_balance'] = self.current_balance
                trade_result['new_position'] = self.position
                
                self.trades.append(trade_result)
                self.total_trades += 1
                
                logger.info(f"BUY executed: {position_size} shares at ${price:.2f}")
            else:
                trade_result['executed'] = False
                trade_result['reason'] = 'Insufficient balance'
        
        elif action == 'SELL':
            # Sell shares
            if self.position >= position_size:
                proceeds = transaction_value - transaction_fee
                self.current_balance += proceeds
                
                # Calculate P&L for this trade
                avg_cost_per_share = self.position_value / self.position if self.position > 0 else 0
                trade_pnl = (price - avg_cost_per_share) * position_size - transaction_fee
                
                self.position -= position_size
                self.position_value -= (avg_cost_per_share * position_size)
                self.total_pnl += trade_pnl
                
                if trade_pnl > 0:
                    self.winning_trades += 1
                
                trade_result['new_balance'] = self.current_balance
                trade_result['new_position'] = self.position
                trade_result['pnl'] = trade_pnl
                
                self.trades.append(trade_result)
                self.total_trades += 1
                
                logger.info(f"SELL executed: {position_size} shares at ${price:.2f}, P&L: ${trade_pnl:.2f}")
            else:
                trade_result['executed'] = False
                trade_result['reason'] = 'Insufficient position'
        
        return trade_result
    
    def _calculate_position_size(self, action: str, price: float) -> int:
        """Calculate position size based on available capital and risk management."""
        
        if action == 'BUY':
            # Use a percentage of available balance
            max_investment = self.current_balance * 0.95  # Leave 5% buffer
            max_shares = int(max_investment / (price * (1 + self.transaction_cost)))
            
            # For now, use a fixed position size or percentage
            target_investment = min(max_investment, self.current_balance * 0.2)  # 20% of balance
            position_size = int(target_investment / (price * (1 + self.transaction_cost)))
            
            return min(position_size, max_shares)
        
        elif action == 'SELL':
            # Sell all or partial position
            return self.position  # For now, sell entire position
        
        return 0
    
    def get_portfolio_value(self, current_price: float) -> float:
        """Calculate total portfolio value."""
        position_market_value = self.position * current_price
        return self.current_balance + position_market_value
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        current_portfolio_value = self.get_portfolio_value(self.last_price)
        total_return = current_portfolio_value - self.initial_balance
        total_return_pct = (total_return / self.initial_balance) * 100
        
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        return {
            'initial_balance': self.initial_balance,
            'current_balance': self.current_balance,
            'current_position': self.position,
            'position_value': self.position * self.last_price if self.position > 0 else 0,
            'portfolio_value': current_portfolio_value,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': win_rate,
            'total_pnl': self.total_pnl
        }
    
    def save_results(self, output_dir: str = "trading_results") -> Tuple[Path, Path]:
        """Save trading results and journal."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save trade history
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_file = output_path / f"{self.ticker}_trades_{timestamp}.csv"
            trades_df.to_csv(trades_file, index=False)
            logger.info(f"Trades saved to {trades_file}")
        else:
            trades_file = None
        
        # Save journal
        journal_file = self.journal.save_journal(f"{self.ticker}_journal_{timestamp}.json")
        
        return trades_file, journal_file
    
    def print_summary(self) -> None:
        """Print comprehensive trading summary."""
        perf = self.get_performance_summary()
        
        print(f"\n{'='*80}")
        print(f"ENHANCED TRADING SUMMARY - {self.ticker}")
        print(f"{'='*80}")
        
        print(f"Portfolio Performance:")
        print(f"  Initial Balance: ${perf['initial_balance']:,.2f}")
        print(f"  Current Balance: ${perf['current_balance']:,.2f}")
        print(f"  Position Value:  ${perf['position_value']:,.2f}")
        print(f"  Portfolio Value: ${perf['portfolio_value']:,.2f}")
        print(f"  Total Return:    ${perf['total_return']:+,.2f} ({perf['total_return_pct']:+.2f}%)")
        
        print(f"\nTrading Statistics:")
        print(f"  Total Trades:    {perf['total_trades']}")
        print(f"  Winning Trades:  {perf['winning_trades']}")
        print(f"  Win Rate:        {perf['win_rate']:.1f}%")
        print(f"  Total P&L:       ${perf['total_pnl']:+,.2f}")
        
        print(f"\nCurrent Position:")
        print(f"  Shares Held:     {perf['current_position']}")
        print(f"  Last Price:      ${self.last_price:.2f}")
        
        # Print journal summary
        self.journal.print_summary()
