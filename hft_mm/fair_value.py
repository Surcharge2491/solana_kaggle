"""Fair value = Binance mid + slowly averaged (Bullet mid - Binance mid).

Binance moves first, so we follow its mid tick by tick. The steady gap
between venues (USDT vs USD, different flows) is the basis; we average it
with a time-based EMA so fair value doesn't drift from where Bullet trades.
No trustworthy price (Binance quiet, basis not learned yet) -> don't quote.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Union

WARMUP_SECS = 60.0  # watch the basis this long before trusting it


class NotReady(Enum):
    NO_BINANCE_PRICE = "waiting for Binance"
    BINANCE_STALE = "Binance price is stale, not quoting"
    WARMING_UP = "learning the Binance-to-Bullet price gap (takes about a minute)"


@dataclass
class _Basis:
    average: float
    last_update: float
    started: float


class FairValue:
    def __init__(self) -> None:
        self.binance_mid: Optional[Tuple[float, float]] = None  # (mid, received_at)
        self.basis: Optional[_Basis] = None

    def on_binance(self, bid: float, ask: float, at: float) -> None:
        if bid > 0 and ask >= bid:
            self.binance_mid = ((bid + ask) / 2, at)

    def on_bullet_mid(self, bullet_mid: float, now: float, stale: float, halflife: float) -> None:
        ref = self._fresh_binance_mid(now, stale)
        if isinstance(ref, NotReady):
            # Without a live Binance price the gap means nothing; start over later.
            self.basis = None
            return
        sample = bullet_mid - ref
        b = self.basis
        if b is None:
            self.basis = _Basis(sample, now, now)
        else:
            w = ema_weight(now - b.last_update, now - b.started, halflife)
            self.basis = _Basis(b.average + w * (sample - b.average), now, b.started)

    def get(self, now: float, stale: float) -> Union[float, NotReady]:
        """The price to quote around, or why there isn't one."""
        ref = self._fresh_binance_mid(now, stale)
        if isinstance(ref, NotReady):
            return ref
        if self.basis is not None and now - self.basis.started >= WARMUP_SECS:
            return ref + self.basis.average
        return NotReady.WARMING_UP

    def basis_average(self) -> Optional[float]:
        return None if self.basis is None else self.basis.average

    def _fresh_binance_mid(self, now: float, stale: float) -> Union[float, NotReady]:
        if self.binance_mid is None:
            return NotReady.NO_BINANCE_PRICE
        mid, at = self.binance_mid
        if now - at > stale:
            return NotReady.BINANCE_STALE
        return mid


def ema_weight(since_last: float, since_start: float, halflife: float) -> float:
    """A sample `halflife` after the previous one moves the average halfway.
    Early on, use the plain time-weighted mean so the first sample doesn't dominate."""
    if since_last <= 0:
        return 0.0
    ema = 1.0 - 0.5 ** (since_last / max(halflife, 1.0))
    mean = since_last / since_start if since_start > 0 else 1.0
    return min(max(ema, mean, 0.0), 1.0)
