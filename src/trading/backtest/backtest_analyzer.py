"""
Backtest analysis module for evaluating trading strategy performance.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class BacktestAnalyzer:
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        
    def calculate_metrics(self, trades_df: pd.DataFrame, 
                         price_series: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive trading metrics from backtest results.
        
        Args:
            trades_df: DataFrame containing trade records with required columns
            price_series: Optional price series for buy-and-hold comparison
            
        Returns:
            Dictionary containing all calculated metrics
        """
        if trades_df.empty:
            logger.warning("Empty trades DataFrame provided")
            return {}
            
        metrics = {}
        
        # Basic trade statistics
        metrics.update(self._calculate_trade_stats(trades_df))
        
        # Balance metrics
        metrics.update(self._calculate_balance_metrics(trades_df))
        
        # Risk metrics
        metrics.update(self._calculate_risk_metrics(trades_df))
        
        # Add buy-and-hold metrics if price series is provided
        if price_series is not None and not price_series.empty:
            metrics.update(self._calculate_buy_hold_metrics(price_series))
            
        return metrics
    
    def _calculate_trade_stats(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate trade execution statistics."""
        if len(trades_df) == 0:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'avg_holding_period_bars': 0,
                'avg_win': 0,
                'avg_loss': 0
            }
            
        # Calculate wins/losses
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        total_trades = len(trades_df)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # Calculate profit factor
        gross_profit = winning_trades['pnl'].sum()
        gross_loss = abs(losing_trades['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate holding periods if timestamps are available
        if all(col in trades_df.columns for col in ['entry_time', 'exit_time']):
            holding_periods = (pd.to_datetime(trades_df['exit_time']) - 
                             pd.to_datetime(trades_df['entry_time'])).dt.days
            avg_holding_period = holding_periods.mean()
        else:
            avg_holding_period = 0
            
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_holding_period_bars': avg_holding_period,
            'avg_win': winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0,
            'avg_loss': losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        }
    
    def _calculate_balance_metrics(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate balance and return metrics."""
        if 'balance' not in trades_df.columns or len(trades_df) == 0:
            return {
                'final_balance': self.initial_balance,
                'total_pnl': 0,
                'total_return_percent': 0,
                'max_drawdown_percent': 0,
                'avg_trade_return_percent': 0
            }
            
        balance_series = trades_df['balance'].dropna()
        if len(balance_series) == 0:
            return {}
            
        final_balance = balance_series.iloc[-1]
        total_pnl = final_balance - self.initial_balance
        total_return_percent = (total_pnl / self.initial_balance) * 100
        
        # Calculate max drawdown
        rolling_max = balance_series.cummax()
        drawdowns = (balance_series - rolling_max) / rolling_max * 100
        max_drawdown_percent = abs(drawdowns.min()) if len(drawdowns) > 0 else 0
        
        # Calculate average trade return
        if 'return_percent' in trades_df.columns:
            avg_trade_return = trades_df['return_percent'].mean()
        else:
            # Estimate from P&L if return_percent not available
            avg_trade_return = (trades_df['pnl'] / trades_df['entry_balance']).mean() * 100 \
                if 'entry_balance' in trades_df.columns else 0
                
        return {
            'final_balance': final_balance,
            'total_pnl': total_pnl,
            'total_return_percent': total_return_percent,
            'max_drawdown_percent': max_drawdown_percent,
            'avg_trade_return_percent': avg_trade_return
        }
    
    def _calculate_risk_metrics(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate risk-related metrics."""
        metrics = {
            'avg_risk_score': np.nan,
            'number_of_high_risk_trades': 0,
            'average_confidence': np.nan
        }
        
        if 'risk_score' in trades_df.columns:
            metrics['avg_risk_score'] = trades_df['risk_score'].mean()
            metrics['number_of_high_risk_trades'] = len(trades_df[trades_df['risk_score'] > 0.7])
            
        if 'confidence' in trades_df.columns:
            metrics['average_confidence'] = trades_df['confidence'].mean()
            
        return metrics
    
    def _calculate_buy_hold_metrics(self, price_series: pd.Series) -> Dict[str, Any]:
        """Calculate buy-and-hold metrics for comparison."""
        if len(price_series) < 2:
            return {}
            
        initial_price = price_series.iloc[0]
        final_price = price_series.iloc[-1]
        buy_hold_return = ((final_price - initial_price) / initial_price) * 100
        
        return {
            'buy_and_hold_return': buy_hold_return
        }
