"""Audit the live stream: for a few tokens, rebuild the first-minute AMM trades from the chain's transaction
history and compare trade counts and features with what the stream captured.

Usage:  HELIUS_API_KEY=... python audit.py runs/<run_dir> [--n 3] [--max-trades 400]
"""
import argparse, json, os, sys, time, urllib.request
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoder import events_from_logs
from features import VIRTUAL_QUOTE_RESERVES, amm_features


def rpc(url, method, params):
    for i in range(7):
        try:
            req = urllib.request.Request(url, json.dumps({'jsonrpc': '2.0', 'id': 1, 'method': method,
                                         'params': params}).encode(), {'Content-Type': 'application/json'})
            d = json.load(urllib.request.urlopen(req, timeout=30))
            if 'result' in d:
                time.sleep(0.12)
                return d['result']
        except Exception:
            pass
        time.sleep(0.5 * 2 ** i)
    raise RuntimeError(method)


def chain_rows(url, mint, pool, mt):
    sigs, before = [], None
    while True:   # newest -> oldest until we pass the migration time
        page = rpc(url, 'getSignaturesForAddress', [pool, {'limit': 1000, **({'before': before} if before else {})}])
        if not page:
            break
        sigs += page; before = page[-1]['signature']
        if page[-1].get('blockTime') and page[-1]['blockTime'] < mt:
            break
    sigs = [s for s in sigs if s.get('err') is None and s.get('blockTime') is not None and mt <= s['blockTime'] <= mt + 60]
    rows = []
    for s in reversed(sigs):
        tx = rpc(url, 'getTransaction', [s['signature'], {'encoding': 'json', 'maxSupportedTransactionVersion': 1}])
        for _, name, f in events_from_logs(tx['meta']['logMessages']):
            if name not in ('BuyEvent', 'SellEvent') or f['pool'] != pool:
                continue
            buy = name == 'BuyEvent'
            b = f['pool_base_token_reserves'] + (-f['base_amount_out'] if buy else f['base_amount_in'])
            q = f['pool_quote_token_reserves'] + (f['quote_amount_in_with_lp_fee'] if buy else -f['quote_amount_out_without_lp_fee'])
            rows.append({'mint': mint, 'ts_rel': f['timestamp'] - mt, 'slot': tx['slot'], 'tx_idx': len(rows),
                         'direction': 'B' if buy else 'S', 'sol': (f['quote_amount_in'] if buy else f['quote_amount_out']) / 1e9,
                         'tok': (f['base_amount_out'] if buy else f['base_amount_in']) / 1e6, 'wallet': f['user'],
                         'new_price': (q + VIRTUAL_QUOTE_RESERVES) / b if b > 0 else 0.0})
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('run'); ap.add_argument('--n', type=int, default=3)
    ap.add_argument('--max-trades', type=int, default=400)
    a = ap.parse_args()
    url = f"https://mainnet.helius-rpc.com/?api-key={os.environ['HELIUS_API_KEY']}"
    toks = [json.loads(l) for l in open(os.path.join(a.run, 'rows60.jsonl'))]
    toks = [t for t in toks if 0 < len(t['rows']) <= a.max_trades][:a.n]
    for t in toks:
        live = t['rows']; chain = chain_rows(url, t['mint'], t['pool'], t['mt'])
        fl, fc = amm_features(t['mint'], live), amm_features(t['mint'], chain)
        diff = {k: (round(fl.get(k, np.nan), 4), round(fc.get(k, np.nan), 4)) for k in fc
                if not np.isclose(fl.get(k, np.nan), fc[k], rtol=1e-3, equal_nan=True)}
        print(f"{t['mint'][:10]}.. live trades={len(live)} chain trades={len(chain)} | features differing: {len(diff)}/{len(fc)}")
        for k, v in list(diff.items())[:8]:
            print(f"     {k}: live={v[0]} chain={v[1]}")
