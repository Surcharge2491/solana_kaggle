"""Events flow in (market data, fills, timers); intents flow out (place, cancel).

Every event carries `t`, the time in seconds we received it. In a backtest
that's the replayed timestamp; live it's time.monotonic().
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

from .book import Book


class Side(Enum):
    BUY = "Buy"
    SELL = "Sell"


# ---------- events (in) ----------

@dataclass(frozen=True)
class Reference:
    """Binance best bid/ask: our price reference."""
    bid: float
    ask: float


@dataclass(frozen=True)
class BookEvent:
    """Bullet's order book changed."""
    book: Book


@dataclass(frozen=True)
class BookLost:
    """Our copy of the Bullet book is out of sync; don't trust it."""


@dataclass(frozen=True)
class Fill:
    trade_id: int
    client_id: Optional[int]
    side: Side
    price: float
    size: float
    fee: float
    order_finished: bool  # the whole order is done and off the book


@dataclass(frozen=True)
class FillEvent:
    fill: Fill


@dataclass(frozen=True)
class OrderGone:
    """One of our orders left the book without filling (cancelled, expired, rejected)."""
    client_id: int


@dataclass(frozen=True)
class PlaceFailed:
    client_ids: tuple


@dataclass(frozen=True)
class CancelFailed:
    """A cancel failed, so the order may still be resting."""


@dataclass(frozen=True)
class PositionEvent:
    """Our position according to the exchange (answer to SyncPosition)."""
    position: float


@dataclass(frozen=True)
class Reconnecting:
    """The Bullet websocket dropped; we may have missed things."""


@dataclass(frozen=True)
class Tick:
    """Once a second, so staleness checks run when the market is quiet."""


Event = Union[Reference, BookEvent, BookLost, FillEvent, OrderGone, PlaceFailed,
              CancelFailed, PositionEvent, Reconnecting, Tick]


# ---------- intents (out) ----------

@dataclass(frozen=True)
class Order:
    side: Side
    price: float
    size: float
    client_id: int


@dataclass(frozen=True)
class Place:
    order: Order


@dataclass(frozen=True)
class Cancel:
    client_id: int


@dataclass(frozen=True)
class CancelAll:
    """Cancel every order we have on the market, tracked or not."""


@dataclass(frozen=True)
class SyncPosition:
    """Ask the exchange for our position (comes back as PositionEvent)."""


@dataclass(frozen=True)
class SyncFills:
    """Ask the exchange for our fills since start (they come back as FillEvents)."""


Intent = Union[Place, Cancel, CancelAll, SyncPosition, SyncFills]
