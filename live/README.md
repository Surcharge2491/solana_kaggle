# Live paper trading (Helius)

Runs the v2 model on live Solana data. **Paper trading only: it never sends a transaction.**

For every pump.fun → PumpSwap migration it:

1. streams pump.fun and PumpSwap program logs from Helius (`logsSubscribe`) and decodes the Anchor events
   (`TradeEvent`, `CreateEvent`, `CreatePoolEvent`, `BuyEvent`, `SellEvent`) with the programs' on-chain IDLs;
2. applies the competition universe filter (pool ≥ 60 SOL, ≥ 100 trades before t = 60 s, age ≤ 30 days, not mayhem mode);
3. at t = 60 s rebuilds the features exactly as in the Kaggle data, scores the token and logs a decision
   (EV policy: buy if expected PnL > margin, tp chosen per token, daily cap);
4. after 1 hour labels the outcome the competition's way (max multiple over (60, 3660] s with the top
   max(1%, 1) prints trimmed; final multiple at 3660 s) and adds the token to the wallet-reputation history.

## How the live data matches the Kaggle data

| Kaggle column | Live source | Check |
|---|---|---|
| AMM `direction`, `sol`, `tok`, `wallet`, `ts` | `BuyEvent` / `SellEvent` (`quote_amount_in/out`/1e9, base/1e6, `user`, `timestamp`) | 100% match on sampled Kaggle transactions |
| AMM `new_price` | (pool quote after + 17,584,505,288) / pool base after; quote after = before + `quote_amount_in_with_lp_fee` (buy) or − `quote_amount_out_without_lp_fee` (sell) | 100% match |
| curve `sol`, `tok`, `wallet`, `new_price` | `TradeEvent` (virtual SOL / virtual token reserves) | 100% match |
| `mt` | `CreatePoolEvent.timestamp` in a pump.fun migration tx | matches `pfamm_migrations.mt` |
| `gas_used`, instruction / lookup counts, `creation_ix_index`, `pf_program_index`, `direct_pf_invocation` | creation transaction (`getTransaction`) | 100% match on sampled tokens |
| `dev_balance` | creator SOL balance **before** the creation tx | matches |
| universe | pool SOL ≥ 60, pre-snapshot trades ≥ 100, age ≤ 30 d, not mayhem | reproduces all 40,893 Kaggle tokens, no false positives |

Not reproducible live, so the live model is trained without them (backtest accuracy unchanged):
Jito-bundle features (`bundle_size`, `bundled_buys*`, `bs_*`) and failed-transaction counts
(`curve_failed`, `curve_failed_ratio`, `pre_failed`; failed transactions' logs don't name the token).

Known gaps: tokens created before the collector started have incomplete curve history
(`history_complete = 0`), and trades during a websocket reconnect are missed (`stream_gap = 1`).
Wallet reputation is bootstrapped from Kaggle labels through the end of the training data; days between
that and the start of the run are missing.

## Setup

```bash
pip install pandas pyarrow lightgbm scikit-learn websockets base58 solders
```

Build the model and wallet history once (needs the competition files in `./data`, ~35 min):

```bash
python live/build_artifacts.py
```

Run (key from https://dashboard.helius.dev; never commit it):

```bash
HELIUS_API_KEY=your-key python live/paper_trader.py --out live/runs/today
```

`--hours N` stops after N hours. Outputs in the run folder:

- `decisions.csv`: one row per universe token at t = 60 s (tp, EV, calibrated probabilities, flags)
- `outcomes.csv`: 1-hour result per scored token (`max_x`, `final_x`, `pnl`, and `pnl_late_fill` using the
  first price seen ≥ 2 s after the decision, a rough slippage check)
- `migrations.csv`: migrations outside the universe; `features.csv`, `rows60.jsonl`: audit data; `status.json`

Check stream completeness against chain history (run alongside the trader; audits the next N tokens right after their decision):

```bash
HELIUS_API_KEY=your-key python live/audit.py live/runs/today --follow --n 3
```

## Helius usage

The stream is ~900 transactions/s (~1.8 MB/s) across both programs, plus one `getTransaction` per migration.
Check your plan's credit usage in the Helius dashboard during the first hours.
