"""
Rich-based logger formatter for enhanced trading logs.

This module provides a custom logging formatter that uses Rich to create
visually appealing and well-structured log messages for trading operations.
"""

import logging
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.style import Style
from rich.logging import RichHandler
from rich.theme import Theme
from rich.columns import Columns
from rich.box import ROUNDED
from typing import Dict, Any, Optional

# Custom theme for trading logs
TRADING_THEME = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "critical": "bold white on red",
    "buy": "bold green",
    "sell": "bold red",
    "hold": "bold yellow",
    "bullish": "green",
    "bearish": "red",
    "neutral": "yellow",
    "high_risk": "bold red",
    "low_risk": "green",
    "high_conf": "bold green",
    "low_conf": "dim yellow",
})

class TradingLogFormatter:
    """Rich-based formatter for trading logs."""
    
    def __init__(self):
        self.console = Console(theme=TRADING_THEME)
        
    def format_signal_log(self, message: str) -> None:
        """Format and print a trading signal log message."""
        # Parse the log message
        parts = {}
        
        # Extract action (HOLD, BUY, SELL)
        if "HOLD" in message:
            action = "HOLD"
            action_style = "hold"
        elif "BUY" in message:
            action = "BUY"
            action_style = "buy"
        elif "SELL" in message:
            action = "SELL"
            action_style = "sell"
        else:
            action = "INFO"
            action_style = "info"
            
        # Extract confidence and risk if available
        conf_val = self._extract_value(message, "Conf:", ",")
        risk_val = self._extract_value(message, "Risk:", " ")
        
        # Extract price data
        close = self._extract_value(message, "Close:", ",")
        rsi = self._extract_value(message, "RSI_14:", ",")
        macd = self._extract_value(message, "MACD:", ",")
        volume_change = self._extract_value(message, "Volume_pct_change:", ")")
        
        # Extract reason components
        reason = ""
        if "Reason:" in message:
            reason = message.split("Reason:")[1].strip()
            
        # Create a structured panel for the signal
        self._display_signal_panel(
            action=action,
            action_style=action_style,
            confidence=conf_val,
            risk=risk_val,
            close=close,
            rsi=rsi,
            macd=macd,
            volume_change=volume_change,
            reason=reason
        )
    
    def _display_signal_panel(self, action: str, action_style: str, 
                             confidence: Optional[float] = None,
                             risk: Optional[float] = None,
                             close: Optional[float] = None,
                             rsi: Optional[float] = None,
                             macd: Optional[float] = None,
                             volume_change: Optional[float] = None,
                             reason: Optional[str] = None) -> None:
        """Display a nicely formatted panel for a trading signal."""
        # Create the main table
        table = Table(show_header=False, box=ROUNDED, border_style="dim")
        table.add_column("Key", style="cyan", width=12)
        table.add_column("Value", style="white")
        
        # Add price data
        if close:
            table.add_row("Close", f"${close}")
        
        # Add technical indicators
        indicators_text = ""
        if rsi:
            rsi_float = float(rsi)
            rsi_style = "bearish" if rsi_float < 30 else "bullish" if rsi_float > 70 else "neutral"
            indicators_text += f"[{rsi_style}]RSI: {rsi}[/] "
            
        if macd:
            macd_float = float(macd)
            macd_style = "bearish" if macd_float < 0 else "bullish"
            indicators_text += f"[{macd_style}]MACD: {macd}[/] "
            
        if volume_change:
            vol_float = float(volume_change)
            vol_style = "bullish" if vol_float > 0 else "bearish" if vol_float < 0 else "neutral"
            indicators_text += f"[{vol_style}]Vol Δ: {volume_change}%[/] "
            
        if indicators_text:
            table.add_row("Indicators", indicators_text)
        
        # Add confidence and risk
        if confidence:
            conf_float = float(confidence)
            conf_style = "high_conf" if conf_float > 0.7 else "low_conf"
            table.add_row("Confidence", f"[{conf_style}]{confidence}[/]")
            
        if risk:
            risk_float = float(risk)
            risk_style = "high_risk" if risk_float > 0.5 else "low_risk"
            table.add_row("Risk", f"[{risk_style}]{risk}[/]")
        
        # Add reason if available
        if reason:
            table.add_row("Analysis", reason)
        
        # Create the panel with the action as the title
        action_icon = "🔶" if action == "HOLD" else "🟢" if action == "BUY" else "🔴" if action == "SELL" else "ℹ️"
        panel = Panel(
            table,
            title=f"{action_icon} [{action_style}]{action}[/]",
            border_style=action_style,
            width=100
        )
        
        self.console.print(panel)
    
    def _extract_value(self, message: str, prefix: str, suffix: str) -> Optional[str]:
        """Extract a value from a message between prefix and suffix."""
        if prefix not in message:
            return None
            
        start_idx = message.find(prefix) + len(prefix)
        end_idx = message.find(suffix, start_idx) if suffix in message[start_idx:] else len(message)
        
        if start_idx < end_idx:
            return message[start_idx:end_idx].strip()
        return None


class RichTradingHandler(RichHandler):
    """Custom Rich handler for trading logs."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.formatter = TradingLogFormatter()
        
    def emit(self, record):
        """Format and emit a log record."""
        # For INFO level messages that look like trading signals
        if record.levelno == logging.INFO and any(x in record.getMessage() for x in ["HOLD", "BUY", "SELL"]):
            self.formatter.format_signal_log(record.getMessage())
        else:
            # Use default Rich handler for other messages
            super().emit(record)


def setup_rich_logging(level=logging.INFO):
    """Set up Rich-based logging for trading applications."""
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichTradingHandler(rich_tracebacks=True, markup=True)]
    )
    return logging.getLogger(__name__)
