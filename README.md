# solana_kaggle
solana kaggle

## hft_mm: the HFT-for-dummies market maker, in Python

A port of [bulletxyz/hft-for-dummies](https://github.com/bulletxyz/hft-for-dummies)
(Rust), ready to drive from historical data for backtesting. It quotes SOL-USD
on Bullet, priced off Binance's SOL perp.

| File | |
|---|---|
| `hft_mm/events.py` | Events in (`Reference`, `BookEvent`, `FillEvent`, `Tick`, ...) and intents out (`Place`, `Cancel`, `CancelAll`, ...) |
| `hft_mm/fair_value.py` | Binance mid + time-EMA of the Bullet-Binance basis, 60 s warm-up, staleness checks |
| `hft_mm/strategy.py` | Bid/ask: spread, inventory skew, position limits, no-cross, tick rounding |
| `hft_mm/market_maker.py` | Pure state machine: `on_event(e, now)` / `requote(now)` return intents |
| `hft_mm/inventory.py` | Position, cash, fees, PnL (changed only by fills) |
| `hft_mm/config.py` | Same settings and defaults as the Rust `bot.toml`; `Config.from_toml` reads it |

Time is a float `now` in seconds passed in by the caller, so a backtest just
replays events in time order, calls `requote(now)`, and feeds simulated fills
back as `FillEvent`s.

```bash
python3 -m unittest discover tests
```
