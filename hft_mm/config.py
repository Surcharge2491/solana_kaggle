"""All tunables, mirroring the Rust bot's bot.toml (same names, same defaults)."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace


@dataclass(frozen=True)
class Config:
    symbol: str = "SOL-USD"
    binance_symbol: str = "SOLUSDT"

    # --- Strategy ---
    spread_bps: float = 10.0            # quote 0.10% either side of fair value
    order_size: float = 0.05            # SOL per order
    max_position: float = 0.1           # never hold more than this long or short
    skew_bps: float = 5.0               # at max position, lean both quotes this far
    requote_threshold_bps: float = 3.0  # replace a quote once it is this far off
    min_requote_ms: int = 500           # but at most this often
    quote_ttl_secs: int = 30            # orders expire; refreshed at half this age

    # --- Safety ---
    max_loss_usd: float = 2.0           # stop quoting for the run after this loss
    reference_stale_secs: float = 5.0   # no Binance price this long: pull quotes
    book_stale_secs: float = 10.0       # no Bullet book update this long: pull quotes
    basis_halflife_secs: float = 300.0  # average the Binance-Bullet gap over ~5 min

    def __post_init__(self) -> None:
        # Same sanity checks as config.rs: refuse numbers that make no sense.
        positive = ["order_size", "max_position", "quote_ttl_secs",
                    "reference_stale_secs", "book_stale_secs", "basis_halflife_secs",
                    "max_loss_usd"]
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ["spread_bps", "skew_bps", "requote_threshold_bps", "min_requote_ms"]:
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative")
        if self.order_size > self.max_position:
            raise ValueError("order_size must not exceed max_position")

    def with_(self, **kw) -> "Config":
        return replace(self, **kw)

    @classmethod
    def from_toml(cls, path: str) -> "Config":
        """Read a bot.toml from the Rust repo; unknown keys (network, keys, ports) are ignored."""
        import tomllib

        with open(path, "rb") as f:
            raw = tomllib.load(f)
        known = {f.name: f.type for f in fields(cls)}
        kw = {}
        for k, v in raw.items():
            if k in known:
                kw[k] = v if known[k] == "str" else (int(v) if known[k] == "int" else float(v))
        return cls(**kw)
