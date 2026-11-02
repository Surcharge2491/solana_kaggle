"""Our copy of Bullet's order book: price -> size per side."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple


@dataclass(frozen=True)
class Book:
    bids: Dict[float, float] = field(default_factory=dict)
    asks: Dict[float, float] = field(default_factory=dict)

    @classmethod
    def from_levels(cls, bids: Iterable[Tuple[float, float]],
                    asks: Iterable[Tuple[float, float]]) -> "Book":
        return cls({p: s for p, s in bids if s > 0}, {p: s for p, s in asks if s > 0})

    @classmethod
    def top(cls, bid: float, ask: float, bid_size: float = 1.0, ask_size: float = 1.0) -> "Book":
        """Top-of-book only: what most historical data gives you."""
        return cls({bid: bid_size}, {ask: ask_size})

    def best_bid(self) -> Optional[float]:
        return max(self.bids) if self.bids else None

    def best_ask(self) -> Optional[float]:
        return min(self.asks) if self.asks else None

    def mid(self) -> Optional[float]:
        b, a = self.best_bid(), self.best_ask()
        return None if b is None or a is None else (b + a) / 2
