"""The market maker's brain: events in, intents out. A pure state machine.

It never waits on the network or talks to an exchange, so a backtest just
feeds it historical events plus simulated fills, and gets its intents back.
It tracks what it *believes* rests on the book (one bid, one ask at most),
updating that belief when it asks for a change and correcting it on failure.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from .book import Book
from .config import Config
from .events import (
    BookEvent, BookLost, Cancel, CancelAll, CancelFailed, Fill, FillEvent, Intent,
    Order, OrderGone, Place, PlaceFailed, PositionEvent, Reconnecting, Reference,
    Side, SyncFills, SyncPosition, Tick,
)
from .fair_value import FairValue, NotReady
from .inventory import Inventory
from .strategy import Quotes, quote

log = logging.getLogger("hft_mm")

FILL_CHECK_EVERY = 10.0  # seconds between fill reconciliations
SUMMARY_EVERY = 10.0


class MarketMaker:
    def __init__(self, config: Config, tick: float, first_client_id: int = 1) -> None:
        self.config = config
        self.tick = tick
        self.fair = FairValue()
        self.book: Optional[tuple] = None  # (Book, received_at)
        self.bid: Optional[Order] = None
        self.ask: Optional[Order] = None
        self.bid_since: Optional[float] = None
        self.ask_since: Optional[float] = None
        self.inventory = Inventory()
        self.next_client_id = first_client_id
        self.last_requote: Optional[float] = None
        self.last_summary: Optional[float] = None
        self.last_fill_check: Optional[float] = None
        self.stopped: Optional[str] = None
        self.status = ""

    # ---- lifecycle ----

    def start(self) -> List[Intent]:
        """Clear out stale orders and learn our position."""
        return [CancelAll(), SyncPosition()]

    def stop(self) -> List[Intent]:
        """Cancel everything, then catch up on a fill that raced the cancel."""
        return self._pull() + [SyncFills()]

    # ---- events ----

    def on_event(self, event, now: float) -> List[Intent]:
        """Update what we know. Returns intents only when an event demands an
        immediate reaction; normal quoting happens in requote()."""
        c = self.config
        if isinstance(event, Reference):
            self.fair.on_binance(event.bid, event.ask, now)
        elif isinstance(event, BookEvent):
            mid = event.book.mid()
            if mid is not None:
                self.fair.on_bullet_mid(mid, now, c.reference_stale_secs, c.basis_halflife_secs)
                self.inventory.mark(mid)
            self.book = (event.book, now)
        elif isinstance(event, BookLost):
            self.book = None
        elif isinstance(event, FillEvent):
            self._on_fill(event.fill)
        elif isinstance(event, OrderGone):
            self._forget(event.client_id)
        elif isinstance(event, PlaceFailed):
            for cid in event.client_ids:
                self._forget(cid)
        elif isinstance(event, CancelFailed):
            log.warning("a cancel failed; cancelling all our orders")
            return self._pull()
        elif isinstance(event, PositionEvent):
            venue = event.position
            if venue != self.inventory.position:
                log.warning("using the exchange's position %s (ours %s)", venue, self.inventory.position)
                self.inventory.adopt(venue)
            if abs(venue) > c.max_position and self.stopped is None:
                # The bot never holds more than max_position: this was opened by hand.
                log.error("position %s exceeds max_position %s; not trading it", venue, c.max_position)
                self.stopped = "stopped: position bigger than max_position (opened by hand?)"
                return self._pull()
        elif isinstance(event, Reconnecting):
            log.warning("Bullet reconnecting: pulling quotes and checking our fills")
            self.book = None
            return self._pull() + [SyncFills()]
        elif isinstance(event, Tick):
            if self.last_summary is None or now - self.last_summary >= SUMMARY_EVERY:
                self.log_summary(now)
                self.last_summary = now
            if self.last_fill_check is None or now - self.last_fill_check >= FILL_CHECK_EVERY:
                self.last_fill_check = now
                return [SyncFills()]
        return []

    # ---- quoting ----

    def requote(self, now: float) -> List[Intent]:
        """Compare the quotes we want with what rests, and say what to change."""
        wanted, status = self.wanted_quotes(now)
        if status != self.status:
            log.info(status)
            self.status = status

        # Rate-limit replacing quotes, but never delay pulling them.
        pulling = wanted.bid is None and wanted.ask is None
        min_gap = self.config.min_requote_ms / 1000.0
        if not pulling and self.last_requote is not None and now - self.last_requote < min_gap:
            return []

        refresh_after = self.config.quote_ttl_secs / 2  # replace at half its lifetime
        cancels: List[Intent] = []
        places: List[Intent] = []
        for side, want in ((Side.BUY, wanted.bid), (Side.SELL, wanted.ask)):
            resting = self._slot(side)
            since = self._since(side)
            young = since is not None and now - since < refresh_after
            if resting is not None and want is not None:
                keep = young and drift_bps(resting.price, want) < self.config.requote_threshold_bps
            else:
                keep = resting is None and want is None
            if keep:
                continue
            if resting is not None:
                cancels.append(Cancel(resting.client_id))
            new = None
            if want is not None:
                new = Order(side, want, self.config.order_size, self._new_client_id())
                places.append(Place(new))
            self._set_slot(side, new, now)
        if not cancels and not places:
            return []
        self.last_requote = now
        # Cancels first: never have the old and new quote resting at once.
        return cancels + places

    def wanted_quotes(self, now: float):
        none = Quotes()
        if self.stopped:
            return none, self.stopped
        if self.book is None:
            return none, "waiting for the Bullet order book"
        book, at = self.book
        if now - at > self.config.book_stale_secs:
            return none, "Bullet order book is stale, not quoting"
        fair = self.fair.get(now, self.config.reference_stale_secs)
        if isinstance(fair, NotReady):
            return none, fair.value
        return quote(fair, self.inventory.position, self.config, book, self.tick), "quoting"

    # ---- internals ----

    def _on_fill(self, fill: Fill) -> None:
        if not self.inventory.on_fill(fill):
            return
        if fill.order_finished and fill.client_id is not None:
            self._forget(fill.client_id)
        pnl = self.pnl()
        log.debug("FILL %s %s @ %s | position %s | pnl %s", fill.side.value, fill.size,
                  fill.price, self.inventory.position, pnl)
        if self.stopped is None and pnl is not None and pnl <= -self.config.max_loss_usd:
            log.error("loss limit of $%s hit: no more quoting this run", self.config.max_loss_usd)
            self.stopped = "stopped: loss limit hit"

    def _pull(self) -> List[Intent]:
        self.bid = self.ask = None
        return [CancelAll()]

    def _forget(self, client_id: int) -> None:
        if self.bid is not None and self.bid.client_id == client_id:
            self.bid = None
        if self.ask is not None and self.ask.client_id == client_id:
            self.ask = None

    def _slot(self, side: Side) -> Optional[Order]:
        return self.bid if side is Side.BUY else self.ask

    def _since(self, side: Side) -> Optional[float]:
        return self.bid_since if side is Side.BUY else self.ask_since

    def _set_slot(self, side: Side, order: Optional[Order], now: float) -> None:
        if side is Side.BUY:
            self.bid, self.bid_since = order, now
        else:
            self.ask, self.ask_since = order, now

    def _new_client_id(self) -> int:
        self.next_client_id += 1
        return self.next_client_id

    def mid(self) -> Optional[float]:
        return None if self.book is None else self.book[0].mid()

    def pnl(self) -> Optional[float]:
        mid = self.mid()
        return None if mid is None else self.inventory.pnl(mid)

    def log_summary(self, now: float) -> None:
        fair = self.fair.get(now, self.config.reference_stale_secs)
        log.info("bullet mid %s | fair %s | our bid %s ask %s | position %s | pnl $%s (fees $%s)",
                 self.mid(), None if isinstance(fair, NotReady) else round(fair, 4),
                 self.bid and self.bid.price, self.ask and self.ask.price,
                 self.inventory.position, self.pnl(), self.inventory.fees)


def drift_bps(a: float, b: float) -> float:
    """How far apart two prices are, in basis points."""
    if b == 0:
        return float("inf")
    return abs(a - b) / b * 10_000
