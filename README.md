# solana_kaggle

Notebooks for the Kaggle competition **memecoins-pump-or-dump**: at t = 60 s after a pump.fun → PumpSwap
migration, decide whether to buy each token and at which take-profit (tp = 0 or 3–100x).

| Notebook | What it does |
|---|---|
| `solana-amm-microstructure-take-profit-modeling.ipynb` | v1. EDA, ~60 first-minute features, leak-free wallet reputation, two-stage LightGBM cascade, top-q% of tokens per day at a fixed 5x. |
| `solana-amm-take-profit-v2-ev-policy.ipynb` | v2. Same features. Adds `train_archive.csv` to training, one cascade per tp level (3/4/5/7/10/20x), calibrated probabilities, per-token tp chosen by expected value, and a nested walk-forward where every policy setting is chosen on earlier days only. |

## Running

On Kaggle: Code → New Notebook → Import the `.ipynb`, then Add Input → the competition. CPU is enough.
v2 takes about 35 minutes (the walk-forward dominates; set `N_SEEDS = 3` to speed it up).

Locally: put the competition files in `data/` next to the notebook (flat layout, as downloaded), install
`pandas pyarrow lightgbm scikit-learn matplotlib seaborn scipy`, and run it.

## Live paper-trading results

- [Nov 8–9, 2026: 8.5 hours on live Solana data](live/reports/live_nov8_9/README.md): charts and explanations
