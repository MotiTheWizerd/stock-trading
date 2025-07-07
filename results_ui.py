"""
Results visualization module using Rich for console output.
Enhanced with advanced visual elements for better data presentation.
"""
from rich.console import Console
from rich.table import Table, Column
from rich import box
from rich.panel import Panel
from rich.layout import Layout
from rich.progress import Progress, BarColumn, TextColumn, TaskProgressColumn
from rich.text import Text
from rich.style import Style
from rich.rule import Rule
from rich.columns import Columns
from typing import Dict, List, Any, Optional
import numpy as np
from datetime import datetime

class ResultsRenderer:
    def __init__(self):
        self.console = Console()
        
    def display_results(self, results: Dict[str, Dict[str, Any]], 
                      strategy_names: List[str],
                      title: str = "Backtest Results"):
        """Display backtest results with enhanced visual formatting."""
        # Clear the screen for a clean display
        self.console.clear()
        
        # Display header with timestamp
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header_text = f"{title} - {current_time}"
        
        self.console.print(Rule(f"[bold blue]{header_text}[/]", style="blue"))
        
        # Create summary panel with key metrics
        self._display_summary_panel(results, strategy_names)
        
        # Display each section with enhanced visuals
        self._display_section(
            "📊 Balance Metrics", 
            self._get_balance_metrics(), 
            results, 
            strategy_names
        )
        
        self._display_section(
            "📈 Trade Statistics", 
            self._get_trade_metrics(), 
            results, 
            strategy_names
        )
        
        self._display_section(
            "⚠️  Risk Metrics", 
            self._get_risk_metrics(), 
            results, 
            strategy_names
        )
        
        # Display legend
        legend = Panel(
            "[green]■[/] Best performance  [red]■[/] Worst performance  \n[cyan]▓[/] Progress bars show relative values  [yellow]↑↓[/] Trend indicators",
            title="[bold]Legend[/]",
            border_style="dim",
            padding=(1, 2)
        )
        self.console.print(legend)
        
    def _display_summary_panel(self, results: Dict[str, Dict[str, Any]], strategy_names: List[str]):
        """Display a summary panel with key performance metrics."""
        panels = []
        
        for name in strategy_names:
            strategy_results = results.get(name, {})
            
            # Get key metrics for summary
            final_balance = strategy_results.get('final_balance', 0)
            total_return = strategy_results.get('total_return_percent', 0)
            win_rate = strategy_results.get('win_rate', 0)
            profit_factor = strategy_results.get('profit_factor', 0)
            max_drawdown = strategy_results.get('max_drawdown_percent', 0)
            
            # Create return indicator
            return_indicator = "🔼" if total_return > 0 else "🔽" if total_return < 0 else "➖"
            return_color = "green" if total_return > 0 else "red" if total_return < 0 else "yellow"
            
            # Create win rate progress bar
            win_rate_value = win_rate if isinstance(win_rate, (int, float)) else 0
            win_rate_display = f"[cyan]{'█' * int(win_rate_value/10)}{'░' * (10-int(win_rate_value/10))}[/] {win_rate_value:.1f}%"
            
            # Format content
            content = f"""[bold]Final Balance:[/] ${final_balance:,.2f}
[bold]Return:[/] [{return_color}]{return_indicator} {total_return:.2f}%[/]
[bold]Win Rate:[/] {win_rate_display}
[bold]Profit Factor:[/] {profit_factor:.2f}
[bold]Max Drawdown:[/] [red]{max_drawdown:.2f}%[/]"""
            
            # Create panel
            panel = Panel(
                content,
                title=f"[bold]{name}[/]",
                border_style="blue" if total_return > 0 else "red",
                padding=(1, 2),
                width=40
            )
            panels.append(panel)
        
        # Display panels in columns
        self.console.print(Columns(panels))
        self.console.print()

    def _display_section(self, title: str, metrics: List[Dict], 
                        results: Dict[str, Dict[str, Any]], 
                        strategy_names: List[str]):
        """Display a section of metrics in a table with enhanced visuals."""
        table = Table(
            title=title,
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
            title_style="bold cyan",
            border_style="blue"
        )
        
        # Add columns
        table.add_column("Metric", style="cyan", no_wrap=True, width=30)
        for name in strategy_names:
            table.add_column(name, justify="right", width=25)
            
        # Add rows for each metric
        for metric in metrics:
            row = [metric['name']]
            values = []
            raw_values = []
            
            # Collect values for each strategy
            for name in strategy_names:
                value = results.get(name, {}).get(metric['key'], np.nan)
                raw_values.append(value)
                formatted_value = self._format_value(value, metric.get('format', '.2f'))
                values.append(formatted_value)
            
            # Find best and worst values for highlighting
            if metric.get('highlight', True):
                valid_values = [v for v in raw_values if not isinstance(v, str) and not np.isnan(v)]
                if valid_values:
                    best = max(valid_values) if metric.get('higher_better', True) else min(valid_values)
                    worst = min(valid_values) if metric.get('higher_better', True) else max(valid_values)
                    
                    # Apply enhanced highlighting with visual indicators
                    for i, (val, formatted_val) in enumerate(zip(raw_values, values)):
                        # Skip if not a number
                        if isinstance(val, (str)) or np.isnan(val):
                            continue
                            
                        # Add progress bar for percentage values
                        if metric.get('format', '').endswith('%') and 0 <= val <= 100:
                            bar_length = int(val / 10) if val <= 100 else 10
                            bar = f"[cyan]{'█' * bar_length}{'░' * (10-bar_length)}[/]"
                            
                            if val == best and val != worst:
                                values[i] = f"[green]{formatted_val}[/green] {bar}"
                            elif val == worst and len(valid_values) > 1:
                                values[i] = f"[red]{formatted_val}[/red] {bar}"
                            else:
                                values[i] = f"{formatted_val} {bar}"
                        # Add trend indicators for other numeric values
                        else:
                            trend = ""
                            if val == best and val != worst:
                                values[i] = f"[green]{formatted_val} ↑[/green]"
                            elif val == worst and len(valid_values) > 1:
                                values[i] = f"[red]{formatted_val} ↓[/red]"
            
            row.extend(values)
            table.add_row(*row)
        
        # Add a border around the table
        panel = Panel(table, border_style="blue", padding=(0, 1))
        self.console.print(panel)
        self.console.print()
        
    def _format_value(self, value, format_spec: str = '.2f'):
        """Format a value according to its type and format specification."""
        if isinstance(value, (int, np.integer)):
            return value
        elif isinstance(value, (float, np.floating)):
            if np.isnan(value):
                return "N/A"
            if format_spec.endswith('%'):
                return float(f"{value:{format_spec[:-1]}}")
            elif format_spec.startswith('$'):
                return float(f"{value:{format_spec[1:]}}")
            return float(f"{value:{format_spec}}")
        return value
        
    def _get_balance_metrics(self) -> List[Dict]:
        return [
            {
                'name': 'Final Balance ($)', 
                'key': 'final_balance', 
                'format': '$,.2f', 
                'higher_better': True
            },
            {
                'name': 'Total P&L ($)', 
                'key': 'total_pnl', 
                'format': '$,.2f', 
                'higher_better': True
            },
            {
                'name': 'Total Return (%)', 
                'key': 'total_return_percent', 
                'format': '.2f%', 
                'higher_better': True
            },
            {
                'name': 'Max Drawdown (%)', 
                'key': 'max_drawdown_percent', 
                'format': '.2f%', 
                'higher_better': False
            },
            {
                'name': 'Avg Trade Return (%)', 
                'key': 'avg_trade_return_percent', 
                'format': '.2f%', 
                'higher_better': True
            },
            {
                'name': 'Buy & Hold Return (%)', 
                'key': 'buy_and_hold_return', 
                'format': '.2f%', 
                'highlight': False
            }
        ]
        
    def _get_risk_metrics(self) -> List[Dict]:
        return [
            {
                'name': 'Avg Risk Score', 
                'key': 'avg_risk_score', 
                'format': '.2f', 
                'higher_better': False
            },
            {
                'name': 'High Risk Trades', 
                'key': 'number_of_high_risk_trades', 
                'format': ',.0f', 
                'higher_better': False
            },
            {
                'name': 'Avg Confidence', 
                'key': 'average_confidence', 
                'format': '.2f', 
                'higher_better': True
            }
        ]
        
    def _get_trade_metrics(self) -> List[Dict]:
        return [
            {
                'name': 'Total Trades', 
                'key': 'total_trades', 
                'format': ',.0f',
                'higher_better': None
            },
            {
                'name': 'Win Rate (%)', 
                'key': 'win_rate', 
                'format': '.1f%', 
                'higher_better': True
            },
            {
                'name': 'Profit Factor', 
                'key': 'profit_factor', 
                'format': '.2f', 
                'higher_better': True
            },
            {
                'name': 'Avg Win ($)', 
                'key': 'avg_win', 
                'format': '$,.2f', 
                'higher_better': True
            },
            {
                'name': 'Avg Loss ($)', 
                'key': 'avg_loss', 
                'format': '$,.2f', 
                'higher_better': False
            },
            {
                'name': 'Avg Holding (days)', 
                'key': 'avg_holding_period_bars', 
                'format': '.1f', 
                'higher_better': False
            }
        ]
