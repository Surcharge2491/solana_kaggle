"""Plots and summary numbers for a paper-trading run.

Usage:  python live/report.py live/runs/day24 live/reports/day24
Writes PNG charts, decisions.csv / outcomes.csv copies and summary.json into the report folder.
"""
import json, os, re, shutil, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from sklearn.metrics import roc_auc_score

COST, MARGIN = 0.3, 0.1
LADDER = [3, 4, 5, 7, 10, 20]
# Kaggle training base rates (40,893 tokens, Aug 21 - Oct 31)
KAGGLE = {3: 13.03, 4: 7.74, 5: 5.00, 7: 2.59, 10: 1.35, 20: 0.33}

BLUE, ORANGE, AQUA, GRAY = '#2a78d6', '#eb6834', '#1baf7a', '#8a8984'
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'axes.edgecolor': GRID,
                     'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.axisbelow': True,
                     'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10.5,
                     'axes.titlesize': 12.5, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
                     'legend.frameon': False, 'savefig.dpi': 150, 'savefig.bbox': 'tight'})


def pnl(tp, mx, fx):
    return (tp - 1 - COST) if mx >= tp else max(fx - 1 - COST, -1.0)


def load(run):
    d = pd.read_csv(os.path.join(run, 'decisions.csv'))
    o = pd.read_csv(os.path.join(run, 'outcomes.csv'))
    log = next((os.path.join(run, f) for f in ('trader.log',) if os.path.exists(os.path.join(run, f))), None)
    tp_ev = {}
    if log:
        for line in open(log):
            m = re.search(r'(?:skip|BUY ) (\w+)\.\. ev=\S+ tp=([\d.]+)', line)
            if m:
                tp_ev[m.group(1)] = float(m.group(2))
    m = o.drop(columns=['ev']).rename(columns={'tp': 'tp_taken'}).merge(d.drop(columns=['tp']), on=['mint', 'mt'])
    m['tp_model'] = [tp_ev.get(x[:8], 3.0) for x in m.mint]
    m['t'] = pd.to_datetime(m.mt, unit='s')
    m = m.sort_values('mt').reset_index(drop=True)
    m['pnl_model_tp'] = [pnl(a, b, c) for a, b, c in zip(m.tp_model, m.max_x, m.final_x)]
    m['pnl_tp3'] = [pnl(3, b, c) for b, c in zip(m.max_x, m.final_x)]
    return d, o, m


def main(run, out):
    os.makedirs(out, exist_ok=True)
    d, o, m = load(run)
    status = json.load(open(os.path.join(run, 'status.json')))
    S = {'decisions': len(d), 'resolved': len(m), 'buys': int((d.tp > 0).sum()), 'status': status,
         'start_utc': str(pd.to_datetime(d.mt.min(), unit='s')), 'end_utc': str(pd.to_datetime(d.mt.max(), unit='s'))}

    # 1. Base rates: live vs Kaggle
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(LADDER)); w = 0.38
    live = [100 * (m.max_x >= k).mean() for k in LADDER]
    ax.bar(x - w / 2 - 0.01, [KAGGLE[k] for k in LADDER], w, color=GRAY, label='Kaggle training (40,893 tokens)')
    b = ax.bar(x + w / 2 + 0.01, live, w, color=BLUE, label=f'Live run ({len(m)} tokens)')
    for r, v in zip(b, live):
        ax.text(r.get_x() + r.get_width() / 2, v + 0.3, f'{v:.1f}%', ha='center', va='bottom', fontsize=9, color=INK2)
    ax.set_xticks(x, [f'≥{k}x' for k in LADDER]); ax.set_ylabel('share of tokens reaching the multiple (%)')
    ax.set_title('How often tokens pump within the hour: live vs training data'); ax.legend()
    fig.savefig(os.path.join(out, '1_base_rates.png')); plt.close(fig)
    S['base_rates_live'] = dict(zip([f'>={k}x' for k in LADDER], [round(v, 2) for v in live]))

    # 2. Calibration: predicted probability vs realised hit rate, by quintile of p_k
    fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
    S['calibration'] = {}
    for ax, k in zip(axs, (3, 5, 10)):
        p, y = m[f'p_{k}'], (m.max_x >= k).astype(int)
        q = pd.qcut(p.rank(method='first'), 5, labels=False)
        g = pd.DataFrame({'p': p, 'y': y, 'q': q}).groupby('q').agg(pred=('p', 'mean'), real=('y', 'mean'), n=('y', 'size'))
        se = np.sqrt(g.real.clip(0.02, 0.98) * (1 - g.real.clip(0.02, 0.98)) / g.n)
        lim = max(g.pred.max(), g.real.max()) * 1.25 + 0.01
        ax.plot([0, lim], [0, lim], color=GRAY, lw=1.2, ls='--', label='perfect calibration')
        ax.errorbar(g.pred, g.real, yerr=[np.minimum(1.96 * se, g.real), 1.96 * se], fmt='o', color=BLUE, ms=8, lw=1.5, capsize=3, label='live, by quintile')
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_title(f'≥{k}x: predicted {p.sum():.1f}, actual {y.sum()}')
        ax.set_xlabel('predicted probability'); ax.set_ylabel('actual hit rate' if k == 3 else '')
        auc = roc_auc_score(y, p) if 0 < y.sum() < len(y) else float('nan')
        ax.text(0.97, 0.04, f'AUC {auc:.2f}', transform=ax.transAxes, ha='right', color=INK2)
        S['calibration'][f'{k}x'] = {'pred': round(float(p.sum()), 2), 'actual': int(y.sum()), 'auc': round(float(auc), 3)}
    axs[0].legend(loc='upper left', fontsize=9)
    fig.suptitle('Are the model\'s probabilities right?', x=0.01, ha='left', fontweight='bold', fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(out, '2_calibration.png')); plt.close(fig)

    # 3. EV distribution and the buy threshold
    fig, ax = plt.subplots(figsize=(8, 4))
    ev = d.ev.dropna()
    ax.hist(ev, bins=np.arange(np.floor(ev.min() * 20) / 20, max(ev.max(), MARGIN) + 0.05, 0.025),
            color=BLUE, edgecolor=SURF, linewidth=1.5)
    ax.axvline(MARGIN, color=ORANGE, lw=2)
    ax.text(MARGIN, ax.get_ylim()[1] * 0.92, '  buy threshold (EV > 0.1 SOL)', color=INK, va='top')
    ax.set_xlabel('model expected PnL per 1-SOL trade at the best take-profit (SOL)'); ax.set_ylabel('tokens')
    ax.set_title(f'Expected value at decision time ({len(ev)} tokens, {S["buys"]} buys)')
    fig.savefig(os.path.join(out, '3_ev_distribution.png')); plt.close(fig)
    S['ev'] = {'max': round(float(ev.max()), 4), 'median': round(float(ev.median()), 4),
               'p90': round(float(ev.quantile(0.9)), 4)}

    # 4. Does a higher score mean more pumps? Outcomes by EV quintile
    fig, ax = plt.subplots(figsize=(8, 4))
    m['evq'] = pd.qcut(m.ev.rank(method='first'), 5, labels=False)
    g = m.groupby('evq').agg(hit3=('max_x', lambda s: 100 * (s >= 3).mean()),
                             hit5=('max_x', lambda s: 100 * (s >= 5).mean()), avg=('pnl_model_tp', 'mean'))
    x = np.arange(5)
    ax.bar(x - 0.2, g.hit3, 0.38, color=BLUE, label='reached ≥3x')
    ax.bar(x + 0.2, g.hit5, 0.38, color=ORANGE, label='reached ≥5x')
    ax.set_xticks(x, ['lowest EV', '2', '3', '4', 'highest EV']); ax.set_ylabel('% of tokens in the group')
    ax.set_title('Pump rate by model score (tokens split into 5 equal groups)'); ax.legend()
    fig.savefig(os.path.join(out, '4_score_vs_outcome.png')); plt.close(fig)
    S['by_ev_quintile'] = g.round(2).to_dict()

    # 5. Hypothetical cumulative PnL
    fig, ax = plt.subplots(figsize=(9, 4.2))
    top = m.ev >= m.ev.quantile(0.9)
    series = [('model policy (actual trades)', m.pnl.cumsum(), INK),
              ('buy every token, tp=3x', m.pnl_tp3.cumsum(), GRAY),
              ('buy every token, model tp', m.pnl_model_tp.cumsum(), ORANGE),
              ('buy top 10% by EV, model tp', (m.pnl_model_tp * top).cumsum(), BLUE)]
    for lab, s, c in series:
        ax.plot(m.t, s, color=c, lw=2, label=lab)
        ax.annotate(f'{s.iloc[-1]:+.1f}', (m.t.iloc[-1], s.iloc[-1]), xytext=(4, 0), textcoords='offset points',
                    va='center', color=INK2, fontsize=9)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylabel('cumulative PnL (SOL, 1 SOL per trade)'); ax.set_xlabel('migration time (UTC)')
    ax.set_title('What different policies would have made'); ax.legend(loc='lower left', fontsize=9)
    fig.autofmt_xdate(); fig.savefig(os.path.join(out, '5_cumulative_pnl.png')); plt.close(fig)
    S['pnl'] = {'actual': round(float(m.pnl.sum()), 2), 'all_tp3': round(float(m.pnl_tp3.sum()), 2),
                'all_model_tp': round(float(m.pnl_model_tp.sum()), 2),
                'top10_model_tp': round(float((m.pnl_model_tp * top).sum()), 2), 'top10_n': int(top.sum()),
                'top10_hit_rate': round(float((m.max_x >= m.tp_model)[top].mean()), 3),
                'late_fill_median': round(float(m.late_fill_x.median()), 4)}
    if 'pnl_late_fill' in m:
        S['pnl']['actual_late_fill'] = round(float(m.pnl_late_fill.fillna(0).sum()), 2)

    # 6. Where tokens end: max multiple vs final multiple
    fig, ax = plt.subplots(figsize=(7, 5))
    hit = m.max_x >= 3
    ax.scatter(m.max_x[~hit], m.final_x[~hit], s=36, color=GRAY, alpha=0.7, edgecolor=SURF, lw=0.8, label='peak < 3x')
    ax.scatter(m.max_x[hit], m.final_x[hit], s=48, color=BLUE, edgecolor=SURF, lw=0.8, label='peak ≥ 3x')
    ax.set_xscale('log'); ax.set_yscale('log')
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}x')
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_formatter(fmt); a.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks([1, 2, 3, 5, 10, 20][:1 + sum(m.max_x.max() >= v for v in (2, 3, 5, 10, 20))])
    ax.set_yticks([v for v in (0.01, 0.1, 1, 10) if m.final_x.min() / 2 <= v <= m.final_x.max() * 2])
    ax.axhline(1, color=INK2, lw=0.8); ax.axvline(3, color=ORANGE, lw=1.2, ls='--')
    ax.set_xlabel('peak multiple in the hour (max_x, log)'); ax.set_ylabel('multiple at 1 hour (final_x, log)')
    ax.set_title('Peak vs where the token ended'); ax.legend(loc='upper left')
    fig.savefig(os.path.join(out, '6_peak_vs_final.png')); plt.close(fig)
    S['final_below_0.05'] = round(float((m.final_x < 0.05).mean()), 3)

    # 7. Activity over time
    allm = pd.concat([d[['mt']].assign(u=1), pd.read_csv(os.path.join(run, 'migrations.csv'))[['mt']].assign(u=0)])
    allm['h'] = pd.to_datetime(allm.mt, unit='s').dt.floor('h')
    g = allm.groupby('h').agg(all=('u', 'size'), uni=('u', 'sum'))
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.bar(g.index, g['all'], width=0.035, color=GRAY, label='all migrations')
    ax.bar(g.index, g['uni'], width=0.035, color=BLUE, label='in competition universe (scored)')
    ax.set_ylabel('tokens per hour'); ax.set_title('Pump.fun → PumpSwap migrations per hour (UTC)'); ax.legend()
    fig.autofmt_xdate(); fig.savefig(os.path.join(out, '7_activity.png')); plt.close(fig)

    for f in ('decisions.csv', 'outcomes.csv'):
        shutil.copy(os.path.join(run, f), os.path.join(out, f))
    json.dump(S, open(os.path.join(out, 'summary.json'), 'w'), indent=1, default=str)
    print(json.dumps(S, indent=1, default=str))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
