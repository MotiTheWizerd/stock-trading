"""
Trade Journal Module - Enhanced Trade Intelligence and Transparency

This module provides comprehensive logging and analysis of trading decisions,
including signal rationale, technical indicators, and trade outcomes.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging
import json
from pathlib import Path

# Import our custom Rich logger if available, otherwise use standard logger
try:
    from trading.utils.rich_logger import setup_rich_logging
    logger = setup_rich_logging()
    HAS_RICH_LOGGER = True
except ImportError:
    logger = logging.getLogger(__name__)
    HAS_RICH_LOGGER = False

class TradeJournal:
    """
    Comprehensive trade journal for logging and analyzing trading decisions.
    """
    
    def __init__(self, ticker: str, output_dir: str = "trade_journals"):
        self.ticker = ticker
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize journal entries
        self.entries: List[Dict[str, Any]] = []
        
        # Signal strength thresholds
        self.strong_signal_threshold = 0.8
        self.high_risk_threshold = 0.7
        
    def log_signal_decision(self, 
                          timestamp: datetime,
                          signal: str,
                          confidence: float,
                          risk_score: Optional[float] = None,
                          indicators: Optional[Dict[str, float]] = None,
                          price: Optional[float] = None,
                          position_taken: Optional[str] = None,
                          trade_reason: Optional[str] = None,
                          additional_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a trading signal decision with full context.
        
        Args:
            timestamp: Decision timestamp
            signal: BUY, SELL, or NONE
            confidence: Model confidence score (0-1)
            risk_score: Risk assessment score (0-1)
            indicators: Technical indicators dict
            price: Current price
            position_taken: Actual position taken
            trade_reason: Human-readable reason for the trade
            additional_data: Any additional context
        """
        
        # Determine signal strength and risk level
        signal_strength = self._classify_signal_strength(signal, confidence, risk_score)
        risk_level = self._classify_risk_level(risk_score)
        
        # Generate trade reason if not provided
        if not trade_reason:
            trade_reason = self._generate_trade_reason(signal, confidence, risk_score, indicators)
        
        # Create journal entry
        entry = {
            'timestamp': timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp),
            'signal': signal,
            'confidence': confidence,
            'risk_score': risk_score,
            'signal_strength': signal_strength,
            'risk_level': risk_level,
            'indicators': indicators or {},
            'price': price,
            'position_taken': position_taken or signal,
            'trade_reason': trade_reason,
            'additional_data': additional_data or {}
        }
        
        self.entries.append(entry)
        
        # Log the decision
        self._log_decision_to_console(entry)
        
    def _classify_signal_strength(self, signal: str, confidence: float, risk_score: Optional[float]) -> str:
        """Classify signal strength based on confidence and risk."""
        if signal == 'NONE':
            return 'NEUTRAL'
        
        if confidence >= self.strong_signal_threshold:
            if risk_score and risk_score <= 0.3:
                return 'VERY_STRONG'
            return 'STRONG'
        elif confidence >= 0.6:
            return 'MODERATE'
        else:
            return 'WEAK'
    
    def _classify_risk_level(self, risk_score: Optional[float]) -> str:
        """Classify risk level."""
        if not risk_score:
            return 'UNKNOWN'
        
        if risk_score >= self.high_risk_threshold:
            return 'HIGH'
        elif risk_score >= 0.4:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _generate_trade_reason(self, signal: str, confidence: float, 
                             risk_score: Optional[float], indicators: Optional[Dict[str, float]]) -> str:
        """Generate a human-readable trade reason."""
        reasons = []
        
        # Confidence-based reasoning
        if confidence >= 0.8:
            reasons.append(f"High confidence ({confidence:.2f})")
        elif confidence >= 0.6:
            reasons.append(f"Moderate confidence ({confidence:.2f})")
        else:
            reasons.append(f"Low confidence ({confidence:.2f})")
        
        # Risk-based reasoning
        if risk_score:
            if risk_score >= 0.7:
                reasons.append(f"High risk ({risk_score:.2f})")
            elif risk_score <= 0.3:
                reasons.append(f"Low risk ({risk_score:.2f})")
        
        # Indicator-based reasoning
        if indicators:
            indicator_reasons = self._analyze_indicators(signal, indicators)
            reasons.extend(indicator_reasons)
        
        return "; ".join(reasons) if reasons else f"{signal} signal"
    
    def _analyze_indicators(self, signal: str, indicators: Dict[str, float]) -> List[str]:
        """Analyze technical indicators for trade reasoning."""
        reasons = []
        
        # RSI analysis
        if 'RSI_14' in indicators:
            rsi = indicators['RSI_14']
            if rsi <= 30:
                reasons.append(f"RSI oversold ({rsi:.1f})")
            elif rsi >= 70:
                reasons.append(f"RSI overbought ({rsi:.1f})")
        
        # MACD analysis
        if all(k in indicators for k in ['MACD', 'MACD_Signal']):
            macd = indicators['MACD']
            macd_signal = indicators['MACD_Signal']
            if macd > macd_signal:
                reasons.append("MACD bullish crossover")
            elif macd < macd_signal:
                reasons.append("MACD bearish crossover")
        
        # Bollinger Bands analysis
        if all(k in indicators for k in ['Bollinger_Upper', 'Bollinger_Lower']) and 'Close' in indicators:
            close = indicators['Close']
            bb_upper = indicators['Bollinger_Upper']
            bb_lower = indicators['Bollinger_Lower']
            
            if close <= bb_lower:
                reasons.append("Price at lower Bollinger Band")
            elif close >= bb_upper:
                reasons.append("Price at upper Bollinger Band")
        
        # Volume analysis
        if 'Volume_pct_change' in indicators:
            vol_change = indicators['Volume_pct_change']
            if abs(vol_change) > 50:
                reasons.append(f"High volume spike ({vol_change:+.1f}%)")
        
        return reasons
    
    def _log_decision_to_console(self, entry: Dict[str, Any]) -> None:
        """Log the trading decision to console with formatting."""
        signal = entry['signal']
        confidence = entry['confidence']
        risk_score = entry['risk_score']
        signal_strength = entry['signal_strength']
        risk_level = entry['risk_level']
        price = entry.get('price')
        
        # Choose emoji and color based on signal and strength
        if signal == 'BUY':
            if signal_strength in ['VERY_STRONG', 'STRONG']:
                emoji = "🟢"
                color_tag = "STRONG BUY"
            else:
                emoji = "🔵"
                color_tag = "BUY"
        elif signal == 'SELL':
            if risk_level == 'HIGH':
                emoji = "🔴"
                color_tag = "HIGH RISK SELL"
            elif signal_strength in ['VERY_STRONG', 'STRONG']:
                emoji = "🟠"
                color_tag = "STRONG SELL"
            else:
                emoji = "🟡"
                color_tag = "SELL"
        else:
            emoji = "⚪"
            color_tag = "HOLD"
        
        # Format indicators for display
        indicators_dict = {}
        if entry['indicators']:
            for key, value in entry['indicators'].items():
                if key in ['RSI_14', 'MACD', 'Close', 'Volume_pct_change']:
                    if isinstance(value, (int, float)):
                        indicators_dict[key] = value
        
        # Format the log message in a way that our Rich logger can parse
        log_message = f"{signal} - Conf: {confidence:.2f}"
        
        if risk_score is not None:
            log_message += f", Risk: {risk_score:.2f}"
            
        if price is not None:
            log_message += f", Close: {price:.2f}"
            
        # Add indicators
        for key, value in indicators_dict.items():
            log_message += f", {key}: {value:.2f}"
            
        # Add reason if available
        if entry['trade_reason']:
            log_message += f", Reason: {entry['trade_reason']}"
            
        # Log the formatted decision
        logger.info(log_message)
    
    def save_journal(self, filename: Optional[str] = None) -> Path:
        """Save the trade journal to a file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.ticker}_trade_journal_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        # Save as JSON
        with open(filepath, 'w') as f:
            json.dump(self.entries, f, indent=2, default=str)
        
        logger.info(f"Trade journal saved to {filepath}")
        return filepath
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert journal entries to a pandas DataFrame."""
        if not self.entries:
            return pd.DataFrame()
        
        # Flatten the entries
        flattened_entries = []
        for entry in self.entries:
            flat_entry = entry.copy()
            
            # Flatten indicators
            if 'indicators' in flat_entry and isinstance(flat_entry['indicators'], dict):
                for key, value in flat_entry['indicators'].items():
                    flat_entry[f'indicator_{key}'] = value
                del flat_entry['indicators']
            
            # Flatten additional_data
            if 'additional_data' in flat_entry and isinstance(flat_entry['additional_data'], dict):
                for key, value in flat_entry['additional_data'].items():
                    flat_entry[f'data_{key}'] = value
                del flat_entry['additional_data']
            
            flattened_entries.append(flat_entry)
        
        return pd.DataFrame(flattened_entries)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics from the trade journal."""
        if not self.entries:
            return {}
        
        df = self.to_dataframe()
        
        # Signal distribution
        signal_counts = df['signal'].value_counts().to_dict()
        
        # Confidence statistics
        confidence_stats = {
            'mean_confidence': df['confidence'].mean(),
            'median_confidence': df['confidence'].median(),
            'min_confidence': df['confidence'].min(),
            'max_confidence': df['confidence'].max()
        }
        
        # Risk statistics
        risk_stats = {}
        if 'risk_score' in df.columns and df['risk_score'].notna().any():
            risk_stats = {
                'mean_risk': df['risk_score'].mean(),
                'median_risk': df['risk_score'].median(),
                'high_risk_trades': len(df[df['risk_score'] >= self.high_risk_threshold])
            }
        
        # Signal strength distribution
        strength_counts = df['signal_strength'].value_counts().to_dict()
        
        return {
            'total_entries': len(self.entries),
            'signal_distribution': signal_counts,
            'confidence_stats': confidence_stats,
            'risk_stats': risk_stats,
            'signal_strength_distribution': strength_counts,
            'date_range': {
                'start': df['timestamp'].min(),
                'end': df['timestamp'].max()
            }
        }
    
    def print_summary(self) -> None:
        """Print a summary of the trade journal."""
        stats = self.get_summary_stats()
        
        print(f"\n{'='*60}")
        print(f"TRADE JOURNAL SUMMARY - {self.ticker}")
        print(f"{'='*60}")
        
        print(f"Total Entries: {stats.get('total_entries', 0)}")
        
        if 'signal_distribution' in stats:
            print(f"\nSignal Distribution:")
            for signal, count in stats['signal_distribution'].items():
                print(f"  {signal}: {count}")
        
        if 'confidence_stats' in stats:
            conf_stats = stats['confidence_stats']
            print(f"\nConfidence Statistics:")
            print(f"  Mean: {conf_stats.get('mean_confidence', 0):.3f}")
            print(f"  Range: {conf_stats.get('min_confidence', 0):.3f} - {conf_stats.get('max_confidence', 0):.3f}")
        
        if 'risk_stats' in stats and stats['risk_stats']:
            risk_stats = stats['risk_stats']
            print(f"\nRisk Statistics:")
            print(f"  Mean Risk: {risk_stats.get('mean_risk', 0):.3f}")
            print(f"  High Risk Trades: {risk_stats.get('high_risk_trades', 0)}")
        
        if 'signal_strength_distribution' in stats:
            print(f"\nSignal Strength Distribution:")
            for strength, count in stats['signal_strength_distribution'].items():
                print(f"  {strength}: {count}")
