"""Python port of bulletxyz/hft-for-dummies: a SOL-USD market maker quoting
on Bullet, priced off Binance's SOL perp.

Same shape as the Rust bot: events in, intents out, no I/O in the strategy.
Time is a plain float (seconds) passed in by the caller, so the exact same
code runs live or on replayed history (backtest).
"""

from .config import Config
from .events import (
    BookEvent, BookLost, CancelFailed, Fill, FillEvent, OrderGone, PlaceFailed,
    PositionEvent, Reconnecting, Reference, Tick,
    Cancel, CancelAll, Order, Place, Side, SyncFills, SyncPosition,
)
from .book import Book
from .fair_value import FairValue, NotReady, WARMUP_SECS
from .strategy import Quotes, quote
from .inventory import Inventory
from .market_maker import MarketMaker

__all__ = [n for n in dir() if not n.startswith("_")]
