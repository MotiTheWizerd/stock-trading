"""trading.agents.trader

Contains:
• SimpleTrader – realistic trade simulator
• apply_trader(df, initial_balance) – convenience wrapper that applies the
  simulator to a DataFrame produced by the predictor.

This code is a verbatim extraction from `scripts/predict.py` so that trading
logic lives in the `trading.agents` package and can be reused in notebooks,
back-tests or a live engine.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "SimpleTrader",
    "apply_trader",
]


class SimpleTrader:
    """A realistic trading simulator with equity, P&L and drawdown tracking."""

    def __init__(
        self,
        equity: float,
        commission_per_trade: float = 0.0,
        borrow_rate_per_bar: float = 0.0001,
    ) -> None:
        self.cash = equity  # free cash
        self.equity = equity  # cash + unrealised P/L
        self.position = 0  # shares (+long, -short)
        self.avg_px = 0.0  # average entry price
        self.log: list[Dict[str, Any]] = []
        self.initial_equity = equity
        self.commission = commission_per_trade
        self.borrow_rate = borrow_rate_per_bar

        # Drawdown tracking
        self.equity_curve = [equity]
        self.peak_equity = equity
        self.max_drawdown = 0.0
        self.max_drawdown_start = None
        self.max_drawdown_end = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_drawdown(self, timestamp=None) -> None:
        self.equity_curve.append(self.equity)
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
            if timestamp is not None:
                self.max_drawdown_start = timestamp

        if self.peak_equity > 0:
            current_dd = (self.peak_equity - self.equity) / self.peak_equity * 100
            if current_dd > self.max_drawdown:
                self.max_drawdown = current_dd
                if timestamp is not None:
                    self.max_drawdown_end = timestamp

    def _update_equity(self, price, timestamp=None) -> None:
        if self.position != 0:
            unrealised = self.position * (price - self.avg_px)
            self.equity = self.cash + unrealised
        else:
            self.equity = self.cash
        self._update_drawdown(timestamp)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def trade(self, signal, price, risk, timestamp=None):
        trade_action = ""
        if signal == "NONE":
            self._update_equity(price, timestamp)
            return {}

        # Determine trade size (simple example: full position each trade)
        qty = self.equity // price if price > 0 else 0
        qty = int(qty)
        if qty == 0:
            logger.warning("Not enough equity to take any position")
            self._update_equity(price, timestamp)
            return {}

        # Execute based on risk tier
        if signal == "BUY":
            if self.position >= 0:
                # Add to / open long
                self.cash -= qty * price
                self.position += qty
                self.avg_px = price if self.position == qty else (
                    (self.avg_px * (self.position - qty) + price * qty) / self.position
                )
                trade_action = f"Long {qty} @ ${price:.2f}"
            else:
                # Closing short
                close_qty = min(qty, -self.position)
                pnl = close_qty * (self.avg_px - price)
                self.cash += pnl - self.commission + close_qty * (self.avg_px - price)
                self.position += close_qty
                trade_action = f"Close {close_qty} short @ ${price:.2f} (P/L: ${pnl:.2f})"
        elif signal == "SELL":
            if self.position <= 0:
                # Add to / open short
                self.cash += qty * price  # receive cash for short sale
                self.position -= qty
                self.avg_px = price if self.position == -qty else (
                    (self.avg_px * (-self.position - qty) + price * qty) / -self.position
                )
                trade_action = f"Short {qty} @ ${price:.2f}"
            else:
                # Closing long
                close_qty = min(qty, self.position)
                pnl = close_qty * (price - self.avg_px)
                self.cash += pnl - self.commission + close_qty * price
                self.position -= close_qty
                trade_action = f"Close {close_qty} long @ ${price:.2f} (P/L: ${pnl:.2f})"
        else:
            logger.error("Unknown signal: %s", signal)

        # Commission cost
        self.cash -= self.commission

        # Borrow fee for shorts
        if self.position < 0:
            borrow_fee = abs(self.position) * price * self.borrow_rate
            self.cash -= borrow_fee
            if borrow_fee > 0.01:
                trade_action += f" (borrow fee ${borrow_fee:.2f})"

        # Update equity
        self._update_equity(price, timestamp)

        if trade_action:
            self.log.append({
                "timestamp": timestamp,
                "action": trade_action,
                "price": price,
                "equity": self.equity,
            })
            logger.info("🔄 %s | Equity: $%.2f", trade_action, self.equity)

        return {
            "action": trade_action,
            "equity": self.equity,
            "cash": self.cash,
            "position": self.position,
        }


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def apply_trader(
    df: pd.DataFrame,
    initial_balance: float = 1_000.0,
    *,
    return_log: bool = False,
):  # noqa: D401
    """Apply `SimpleTrader` row-by-row; append equity & drawdown.

    Parameters
    ----------
    df : DataFrame
        Prediction dataframe.
    initial_balance : float, default 1000
        Starting cash.
    return_log : bool, default False
        If *True*, return a tuple ``(result_df, trade_log)``.
    """
    result_df = df.copy()
    trader = SimpleTrader(equity=initial_balance)

    for idx, row in result_df.iterrows():
        trader.trade(
            signal=row.get("prediction", "NONE"),
            price=row.get("Price", row.get("Close", 0)),
            risk=row.get("risk_tier", "LR"),
            timestamp=row.get("datetime"),
        )
        result_df.loc[idx, "balance"] = trader.equity
        result_df.loc[idx, "position"] = trader.position

    logger.info(
        "🏁 Simulation complete – final equity: $%.2f | P/L: $%.2f (%.2f%%)",
        trader.equity,
        trader.equity - initial_balance,
        (trader.equity / initial_balance - 1) * 100,
    )

    if return_log:
        return result_df, trader.log
    return result_df
