"""Trading agents package (decision-making, execution simulators)."""

from .trader import SimpleTrader, apply_trader

__all__: list[str] = [
    "SimpleTrader",
    "apply_trader",
]
