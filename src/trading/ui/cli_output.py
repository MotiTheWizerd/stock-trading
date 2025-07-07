"""trading.ui.cli_output

Unified Rich-based helpers for console output in CLIs & notebooks.

Currently wraps `results_ui.ResultsRenderer` (legacy) so existing visual
components keep working after the refactor.  In the future we can move the
class here directly but importing keeps the diff minimal.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.progress import Progress, SpinnerColumn, TextColumn

# Re-use existing renderer without moving its heavy code right now
try:
    from results_ui import ResultsRenderer  # type: ignore  # noqa: WPS433
except ModuleNotFoundError:  # fallback if file was removed or not on path
    ResultsRenderer = None  # type: ignore

logger = logging.getLogger(__name__)

__all__ = [
    "render_predictions",
    "render_summary",
]

console = Console()


def render_predictions(df: pd.DataFrame, limit: int = 30) -> None:  # noqa: D401
    """Pretty-print the *df* predictions using Rich.

    Shows the first *limit* rows with prediction probabilities and technical indicators.
    """
    if df.empty:
        console.print("[bold red]No predictions to show.[/]")
        return

    # Create a comprehensive table with all requested information
    table = Table(box=box.SIMPLE_HEAVY)
    
    # Core columns
    table.add_column("Datetime", justify="right", no_wrap=True)
    table.add_column("Price", justify="right")
    table.add_column("Prediction", justify="center")
    
    # Probability columns
    prob_cols = [col for col in df.columns if col.startswith("prob_")]
    if prob_cols:
        for col in sorted(prob_cols):
            display_name = col.replace("prob_", "")
            table.add_column(f"{display_name} Prob", justify="right")
    
    # Confidence and risk columns
    table.add_column("Conf", justify="right")
    table.add_column("Conf Level", justify="center")
    table.add_column("Risk Score", justify="right")
    table.add_column("Risk Tier", justify="center")
    
    # Technical indicator columns
    indicator_cols = ["RSI", "Bollinger_Upper", "Bollinger_Lower", "Bollinger_Mid", "MACD"]
    for col in indicator_cols:
        if col in df.columns:
            table.add_column(col, justify="right")

    palette = {"BUY": "green", "SELL": "red", "NONE": "yellow"}

    # Check if all prices are zero or missing
    all_prices_zero = True
    price_columns = ["Price", "Close", "Open", "Adj Close", "High", "Low"]
    
    for _, row in df.head(limit).iterrows():
        # Get prediction and style
        pred = str(row.get("prediction", "NONE"))
        color = palette.get(pred, "white")
        
        # Get confidence level and style
        conf = row.get("confidence", 0)
        conf_level = row.get("confidence_level", "")
        if conf >= 0.95:
            conf_style = "bold green"
        elif conf >= 0.8:
            conf_style = "green"
        elif conf >= 0.5:
            conf_style = "yellow"
        else:
            conf_style = "red"
            
        # Get price from any available price column
        price = 0
        for price_col in price_columns:
            if price_col in row and pd.notna(row[price_col]) and row[price_col] != 0:
                price = row[price_col]
                all_prices_zero = False
                break
        
        # Build row data starting with core columns
        row_data = [
            str(row.get("datetime", ""))[:19],
            f"{price:.2f}",
            f"[{color}]{pred}[/]",
        ]
        
        # Add probability columns
        for col in sorted(prob_cols):
            prob_value = row.get(col, 0)
            # Highlight the highest probability
            if prob_value == conf:
                row_data.append(f"[bold]{prob_value:.3f}[/]")
            else:
                row_data.append(f"{prob_value:.3f}")
        
        # Add confidence and risk columns
        row_data.extend([
            f"[{conf_style}]{conf:.3f}[/]",
            str(conf_level),
            f"{row.get('risk_score', 0):.3f}",
            row.get("risk_tier", "LR"),
        ])
        
        # Add technical indicators
        for col in indicator_cols:
            if col in row and pd.notna(row[col]):
                indicator_value = row[col]
                # Format RSI with color coding
                if col == "RSI" and indicator_value is not None:
                    if indicator_value > 70:
                        row_data.append(f"[red]{indicator_value:.1f}[/]")
                    elif indicator_value < 30:
                        row_data.append(f"[green]{indicator_value:.1f}[/]")
                    else:
                        row_data.append(f"{indicator_value:.1f}")
                else:
                    row_data.append(f"{indicator_value:.1f}" if isinstance(indicator_value, (int, float)) else str(indicator_value))
            else:
                row_data.append("N/A")
                
        table.add_row(*row_data)
    
    if all_prices_zero:
        logger.warning("All price values are zero or missing in the displayed results!")
        console.print("[bold yellow]Warning: All price values are zero or missing![/]")

    console.print(table)


def render_summary(results: Dict[str, Dict[str, Any]], title: str = "Results") -> None:  # noqa: D401
    """Display aggregate back-test results with fancy Rich visuals."""
    if ResultsRenderer is None:
        console.print("[red]Rich summary renderer unavailable (results_ui missing).[/]")
        return
    renderer = ResultsRenderer()
    renderer.display_results(results, strategy_names=list(results.keys()), title=title)
