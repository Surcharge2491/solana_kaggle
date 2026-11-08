"""Build the live model bundle and wallet-reputation table from the Kaggle data, reusing the v2 notebook's code.

Runs the v2 notebook's data/feature/model/policy cells with the live feature set (features that can be rebuilt
from Helius data: no Jito-bundle and no failed-transaction features), then fits the final models on all labelled
tokens. Takes ~35 min on 4 CPUs.

Usage (from the repo root, competition files in ./data):  python live/build_artifacts.py
Writes live/model_bundle.pkl and live/wallet_stats.parquet.
"""
import glob, json, os, pickle, sys
import matplotlib; matplotlib.use('Agg')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'live')
NB = os.path.join(ROOT, 'solana-amm-take-profit-v2-ev-policy.ipynb')
LIVE_DROP = ['bs_ops', 'bs_buys', 'bs_amt_sum', 'bs_amt_max', 'bundle_size', 'bundled_buys', 'bundled_buys_count',
             'curve_failed', 'curve_failed_ratio', 'pre_failed']

cells = [''.join(c['source']) for c in json.load(open(NB))['cells'] if c['cell_type'] == 'code']
setup, data, tape, launch, rep, model, walk, policy, results, _submission = cells
g = {'display': print, '__name__': 'build'}
os.chdir(ROOT)
for code in (setup, data, tape, launch, rep):
    exec(code, g)
g['ALL_FEATURE_COLS'] = [f for f in g['ALL_FEATURE_COLS'] if f not in LIVE_DROP]
print(f"Live feature set: {len(g['ALL_FEATURE_COLS'])} features")
exec(model, g)
exec(walk.replace('oof_v2.parquet', 'oof_live.parquet'), g)
exec(policy, g)
exec(results, g)

exec('''
X = TR_DF[ALL_FEATURE_COLS].astype(float); mx = TR_DF['max_x'].values; surv = mx >= SURVIVE_K
fit = lambda Xa, y: [lgb.LGBMClassifier(random_state=s, **LGB_PARAMS).fit(Xa, y) for s in SEEDS]
cal_all, ml_all = fit_calibration(OOF)
margin, cap = (float(x) for x in EV_GRID.sum().idxmax())
BUNDLE = dict(features=ALL_FEATURE_COLS, ladder=TP_LADDER, survive_k=SURVIVE_K,
              stage1=fit(X, surv.astype(int)), stage2={k: fit(X[surv], (mx[surv] >= k).astype(int)) for k in TP_LADDER},
              calibrators=cal_all, miss_loss=ml_all, margin=margin, cap_pct=cap, cost=COST,
              trained_through=TR_DF['day'].max(), n_train=len(TR_DF),
              daily_universe_median=int(TR_DF.groupby('day').size().tail(14).median()))
# wallet reputation state: appearances / 5x hits of first-minute AMM buyers over all labelled tokens
E = EDGE_DF[['mint', 'wallet']].merge(TR_DF[['mint', 'max_x']], on='mint')
WALLETS = E.assign(hit=(E['max_x'] >= 5).astype(int)).groupby('wallet').agg(appear=('mint', 'size'), hit=('hit', 'sum')).reset_index()
''', g)
pickle.dump(g['BUNDLE'], open(os.path.join(OUT, 'model_bundle.pkl'), 'wb'))
g['WALLETS'].to_parquet(os.path.join(OUT, 'wallet_stats.parquet'), index=False)
print(f"Saved live/model_bundle.pkl (margin={g['margin']}, cap={g['cap']}%) and live/wallet_stats.parquet ({len(g['WALLETS']):,} wallets)")
