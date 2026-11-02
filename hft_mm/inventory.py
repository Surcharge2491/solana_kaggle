"""Position and PnL. Position only changes from fills (or an exchange read)."""

from __future__ import annotations

from .events import Fill, Side


class Inventory:
    def __init__(self) -> None:
        self.position = 0.0   # base asset: + long, - short
        self.fees = 0.0
        self.cash = 0.0       # USD in and out from fills
        self.volume = 0.0     # base traded (for stats)
        self.n_fills = 0
        self._unvalued = 0.0  # adopted position not yet priced
        self._seen: set = set()

    def on_fill(self, fill: Fill) -> bool:
        """Apply a fill. False if we already counted it."""
        if fill.trade_id in self._seen:
            return False
        self._seen.add(fill.trade_id)
        signed = fill.size if fill.side is Side.BUY else -fill.size
        self.position += signed
        self.cash -= signed * fill.price
        self.fees += fill.fee
        self.volume += fill.size
        self.n_fills += 1
        return True

    def adopt(self, venue: float) -> None:
        """Believe the exchange's position; value the difference at the next price."""
        self._unvalued += venue - self.position
        self.position = venue

    def mark(self, mid: float) -> None:
        self.cash -= self._unvalued * mid
        self._unvalued = 0.0

    def pnl(self, mid: float) -> float:
        """Profit since start in USD (fees included), holdings valued at mid."""
        return self.cash + (self.position - self._unvalued) * mid - self.fees
