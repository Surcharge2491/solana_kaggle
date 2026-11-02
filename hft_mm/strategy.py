"""Where to put our bid and ask.

1. Spread: quote spread_bps either side of fair value.
2. Skew: long -> lean both quotes down (more likely to sell); short -> up.
3. Limits: don't quote a side that would pass max_position, and never cross
   the book (that would take liquidity, not make it).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .book import Book
from .config import Config

BPS = 10_000.0
_EPS = 1e-9  # float slack so 99.9/0.01 = 9989.999999 still rounds to 9990


@dataclass(frozen=True)
class Quotes:
    bid: Optional[float] = None
    ask: Optional[float] = None


def quote(fair: float, position: float, config: Config, book: Book, tick: float) -> Quotes:
    # How full our inventory is, from -1 (max short) to +1 (max long).
    fullness = max(-1.0, min(1.0, position / config.max_position))
    lean_bps = config.skew_bps * fullness

    bid = fair * (1 - (config.spread_bps + lean_bps) / BPS)
    ask = fair * (1 + (config.spread_bps - lean_bps) / BPS)

    room_to_buy = position + config.order_size <= config.max_position + _EPS
    room_to_sell = position - config.order_size >= -config.max_position - _EPS

    out_bid = out_ask = None
    if room_to_buy:
        best_ask = book.best_ask()
        if best_ask is not None:
            bid = min(bid, best_ask - tick)  # stay at least one tick under the best ask
        out_bid = round_to_tick(bid, tick, up=False)
    if room_to_sell:
        best_bid = book.best_bid()
        if best_bid is not None:
            ask = max(ask, best_bid + tick)
        out_ask = round_to_tick(ask, tick, up=True)
    return Quotes(out_bid, out_ask)


def round_to_tick(price: float, tick: float, up: bool) -> float:
    """Bids round down, asks round up: rounding never makes a quote more aggressive."""
    if tick <= 0:
        return price
    n = price / tick
    n = math.ceil(n - _EPS) if up else math.floor(n + _EPS)
    decimals = max(0, -math.floor(math.log10(tick)) + 2)
    return round(n * tick, decimals)
