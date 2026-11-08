"""Feature computation for live tokens. `extract_tape_features` is copied verbatim from the v1/v2 notebooks
so live AMM features are computed exactly as in training."""
import re
import numpy as np
import pandas as pd

VIRTUAL_QUOTE_RESERVES = 17584505288  # constant PumpSwap virtual quote reserve (lamports) used in Kaggle's new_price
CURVE_COLS = ['failed', 'mint', 'ts_rel', 'slot', 'tx_idx', 'direction', 'sol', 'tok', 'wallet', 'new_price']
AMM_COLS = ['mint', 'ts_rel', 'slot', 'tx_idx', 'direction', 'sol', 'tok', 'wallet', 'new_price']
REPUTATION_FEATURE_NAMES = ['w_known', 'w_hit_max', 'w_hit_mean', 'w_hit_top3', 'w_elite_n', 'w_elite_share', 'w_eb_score']
C_PRIOR, P_PRIOR = 10.0, 0.06

def extract_tape_features(c_df, a_df):
    # High-frequency microstructure metrics from bonding curve and AMM tapes
    # Invariant: Prices taken from pool reserves (new_price), not sol/tok ratios
    # Invariant: AMM direction is normalized to token ('B' = buy token, 'S' = sell token)
    for df in (c_df, a_df):
        df['is_buy'] = df['direction'].astype(str).str.upper().str.startswith('B').astype(np.int8)
        df['price'] = np.where(df['new_price'] > 0, df['new_price'], df['sol'] / np.maximum(df['tok'], 1e-18))

    # Stable sort by mint, slot, tx_idx preserving exact chain order and intra-tx swap file order
    # Invariant: NEVER sort by ts or block_time, as doing so scrambles trades within the same second
    a_df = a_df.sort_values(['mint', 'slot', 'tx_idx'], kind='mergesort')
    g_amm = a_df.groupby('mint')
    
    F = pd.DataFrame(index=g_amm.size().index)
    
    # 1. AMM Participation & Volume
    F['amm_n'] = g_amm.size()
    F['amm_sol'] = g_amm['sol'].sum()
    F['amm_uw'] = g_amm['wallet'].nunique()
    F['trades_per_wallet'] = F['amm_n'] / F['amm_uw'].replace(0, np.nan)
    
    # Participant concentration (Top-1 wallet volume share and HHI)
    wallet_sol = a_df.groupby(['mint', 'wallet'])['sol'].sum()
    wallet_tot = wallet_sol.groupby('mint').sum().replace(0, np.nan)
    F['amm_top1'] = wallet_sol.groupby('mint').max() / wallet_tot
    
    # Herfindahl-Hirschman Index (HHI)
    wallet_shares_sq = (wallet_sol / wallet_sol.groupby('mint').transform('sum').replace(0, np.nan)) ** 2
    F['amm_hhi'] = wallet_shares_sq.groupby('mint').sum()
    
    # 2. Temporal Flow Dynamics (Early vs. Late)
    late_mask = a_df['ts_rel'] >= 45
    early_mask = a_df['ts_rel'] < 15
    late_df = a_df[late_mask]
    early_df = a_df[early_mask]
    
    F['late_sol'] = late_df.groupby('mint')['sol'].sum()
    early_sol = early_df.groupby('mint')['sol'].sum().replace(0, np.nan)
    F['late_over_early'] = F['late_sol'] / early_sol
    
    late_buys = late_df[late_df['is_buy'] == 1]
    F['late_buyshare'] = late_buys.groupby('mint')['sol'].sum() / F['late_sol'].replace(0, np.nan)
    
    tail_df = a_df[a_df['ts_rel'] > 30]
    tail_buys = tail_df[tail_df['is_buy'] == 1]
    F['tail_buyshare'] = tail_buys.groupby('mint')['sol'].sum() / tail_df.groupby('mint')['sol'].sum().replace(0, np.nan)
    
    # Fresh wallet arrivals in terminal window
    early_wallets = early_df.groupby('mint')['wallet'].apply(set)
    late_wallets = late_df.groupby('mint')['wallet'].apply(set)
    F['fresh_wallets'] = pd.Series({
        m: len(s - early_wallets.get(m, set())) for m, s in late_wallets.items()
    })
    
    # 3. Predatory Fast Flipping Detection
    early_buyers = a_df[(a_df['ts_rel'] <= 15) & (a_df['is_buy'] == 1)].groupby('mint')['wallet'].apply(set)
    first_min_sellers = a_df[a_df['is_buy'] == 0].groupby('mint')['wallet'].apply(set)
    F['flip_rate'] = pd.Series({
        m: len(s & first_min_sellers.get(m, set())) / max(len(s), 1) for m, s in early_buyers.items()
    })
    
    # 4. Order Size & Whale Participation
    buys_df = a_df[a_df['is_buy'] == 1]
    F['buy_max'] = buys_df.groupby('mint')['sol'].max()
    F['whale_buys'] = buys_df[buys_df['sol'] >= 5.0].groupby('mint').size().reindex(F.index).fillna(0)
    F['buy_count'] = buys_df.groupby('mint').size().reindex(F.index).fillna(0)
    F['sell_count'] = (F['amm_n'] - F['buy_count']).fillna(0)
    F['net_buy_ratio'] = (F['buy_count'] - F['sell_count']) / F['amm_n'].replace(0, np.nan)
    
    # 5. Price Path & Microstructure Drawdown (from reserve new_price)
    p_first = g_amm['price'].first()
    p_last = g_amm['price'].last()
    p_max = g_amm['price'].max()
    
    F['ret60'] = p_last / p_first.replace(0, np.nan)
    F['runup'] = p_max / p_first.replace(0, np.nan)
    F['at_high'] = p_last / p_max.replace(0, np.nan)
    F['drawdown_from_peak'] = 1.0 - F['at_high']
    F['vol'] = a_df.assign(lp=np.log(a_df['price'].replace(0, np.nan))).groupby('mint')['lp'].std()
    
    # Kyle's Lambda Microstructure Price Impact Proxy
    aa = a_df[a_df['price'] > 0].copy()
    aa['dlp'] = aa.groupby('mint')['price'].transform(lambda s: np.log(s).diff())
    aa['signed_sol'] = np.where(aa['is_buy'] == 1, aa['sol'], -aa['sol'])
    aa = aa.dropna(subset=['dlp'])
    F['impact_per_sol'] = ((aa['signed_sol'] * aa['dlp']).groupby(aa['mint']).sum()
                           / (aa['signed_sol'] ** 2).groupby(aa['mint']).sum().replace(0, np.nan))
    
    # 6. Bonding Curve Dynamics & Latent Failed Instructions
    ok_curve = c_df[c_df['failed'] == 0]
    g_curve = ok_curve.groupby('mint')
    F['curve_sol'] = g_curve['sol'].sum()
    F['curve_uw'] = g_curve['wallet'].nunique()
    F['curve_dur'] = -g_curve['ts_rel'].first()
    F['curve_failed'] = c_df[c_df['failed'] != 0].groupby('mint').size().reindex(F.index).fillna(0)
    F['curve_failed_ratio'] = F['curve_failed'] / (F['curve_failed'] + ok_curve.groupby('mint').size().reindex(F.index).fillna(0))
    F['migration_pop'] = p_first / g_curve['price'].last().replace(0, np.nan)
    
    # Reconstructed helper columns per 2026-10-01 competition dataset specification
    F['pre_failed'] = F['curve_failed']
    curve_ok_trades = ok_curve.groupby('mint').size().reindex(F.index).fillna(0)
    amm_trades_pos = a_df[a_df['new_price'] > 0].groupby('mint').size().reindex(F.index).fillna(0)
    F['pre_total'] = curve_ok_trades + amm_trades_pos
    F['entry'] = (g_amm['new_price'].last() / 1000.0).reindex(F.index)
    
    return F


def amm_features(mint, amm_rows):
    """amm_rows: list of dicts with AMM_COLS (ts_rel <= 60), in on-chain order. Returns a 1-row feature dict."""
    a = pd.DataFrame(amm_rows, columns=AMM_COLS)
    c = pd.DataFrame(columns=CURVE_COLS).astype({'failed': int, 'ts_rel': float, 'slot': int, 'tx_idx': int,
                                                  'sol': float, 'tok': float, 'new_price': float})
    F = extract_tape_features(c, a)
    return F.loc[mint].to_dict() if mint in F.index else {}


def curve_features(agg, mt, p_first_amm, n_amm_priced):
    """Curve features from streamed aggregates, same definitions as extract_tape_features (successful trades only)."""
    out = {'curve_sol': agg['sol'], 'curve_uw': len(agg['wallets']), 'curve_dur': float(mt - agg['first_ts']),
           'migration_pop': p_first_amm / agg['last_price'] if agg['last_price'] else np.nan,
           'pre_total': agg['n'] + n_amm_priced}
    return out


def launch_features(cre, tx_feats, mt):
    """Creation metadata features (LAUNCH_M in the notebook), minus the Jito-bundle ones."""
    name, sym = cre.get('name') or '', cre.get('symbol') or ''
    f = {'name_len': len(name), 'symbol_len': len(sym),
         'name_upper_ratio': sum(ch.isupper() for ch in name) / max(len(name), 1),
         'name_eq_symbol': int(name.lower()[:4] == sym.lower()[:4]),
         'is_cashback_enabled': int(bool(cre.get('is_cashback_enabled'))),
         'token2022': int(not str(cre.get('token_program', '')).startswith('Tokenkeg')),
         'age_s': mt - cre['created_ts']}
    f.update(tx_feats or {})
    return f


def creation_tx_features(tx, creator, pump_program):
    """gas_used, instruction/lookup counts, pump program position and dev_balance (creator SOL before the tx)."""
    msg, meta = tx['transaction']['message'], tx['meta']
    la = meta.get('loadedAddresses') or {}
    keys = list(msg['accountKeys']) + la.get('writable', []) + la.get('readonly', [])
    ixs = msg['instructions']; lut = msg.get('addressTableLookups') or []
    pf = [i for i, ix in enumerate(ixs) if keys[ix['programIdIndex']] == pump_program]
    ci = keys.index(creator) if creator in keys else None
    return {'gas_used': meta.get('computeUnitsConsumed'), 'amount_of_instructions': len(ixs),
            'amount_of_lookup_reads': sum(len(l['readonlyIndexes']) for l in lut),
            'amount_of_lookup_writes': sum(len(l['writableIndexes']) for l in lut),
            'creation_ix_index': pf[0] if pf else np.nan,
            'pf_program_index': keys.index(pump_program) if pump_program in keys else np.nan,
            'direct_pf_invocation': int(bool(pf)),
            'dev_balance': meta['preBalances'][ci] if ci is not None and ci < len(meta['preBalances']) else np.nan}


def reputation_features(buyer_sol, wallet_stats):
    """buyer_sol: {wallet: SOL bought in first 60s}. wallet_stats: {wallet: [appear, hit]} of resolved tokens."""
    if not buyer_sol:
        return dict.fromkeys(REPUTATION_FEATURE_NAMES, 0.0)
    ws = list(buyer_sol)
    ap = np.array([wallet_stats.get(w, (0, 0))[0] for w in ws], float)
    ht = np.array([wallet_stats.get(w, (0, 0))[1] for w in ws], float)
    sols = np.array([buyer_sol[w] for w in ws], float)
    known = ap >= 3
    if known.sum() == 0:
        return dict.fromkeys(REPUTATION_FEATURE_NAMES, 0.0)
    rates = ht[known] / ap[known]
    elite = (ap >= 5) & (ht >= 1)
    eb = (ht[known] + C_PRIOR * P_PRIOR) / (ap[known] + C_PRIOR)
    return dict(zip(REPUTATION_FEATURE_NAMES, [float(known.sum()), float(rates.max()), float(rates.mean()),
                float(np.sort(rates)[-3:].mean()), float(elite.sum()), float(sols[elite].sum() / max(sols.sum(), 1e-9)),
                float(eb.max())]))
