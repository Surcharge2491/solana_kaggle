"""Live paper trader for pump.fun -> PumpSwap migrations, using the v2 model.

Streams pump.fun and PumpSwap program logs from Helius, rebuilds the competition's features at t = 60 s after
each migration, scores the token, logs a paper decision (never trades), and labels the outcome after 1 hour
exactly like the competition (trimmed max multiple over (60, 3660] s, final multiple at 3660 s).

Usage:  HELIUS_API_KEY=... python paper_trader.py --out runs/live [--hours 6]
"""
import argparse, asyncio, collections, csv, json, math, os, pickle, sys, time, urllib.request
import numpy as np
import pandas as pd
import websockets

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoder import PROGRAMS, events_from_logs
from features import (VIRTUAL_QUOTE_RESERVES, amm_features, creation_tx_features, curve_features,
                      launch_features, reputation_features)

HERE = os.path.dirname(os.path.abspath(__file__))
PUMP, AMM = PROGRAMS['pump'], PROGRAMS['pump_amm']
WSOL = 'So11111111111111111111111111111111111111112'
DECISION_S, HORIZON_S, SETTLE_S = 60, 3660, 8        # decide at t=60 s; label at 3660 s; wait for late messages
MIN_POOL_SOL, MIN_PRE_TRADES, MAX_AGE_S = 60.0, 100, 30 * 86400
CURVE_KEEP_S = 6 * 3600                                # forget curve history of tokens idle this long


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


class Model:
    def __init__(self, path):
        b = pickle.load(open(path, 'rb'))
        self.__dict__.update(b)

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def score(self, feats):
        X = pd.DataFrame([feats]).reindex(columns=self.features).astype(float)
        p1 = np.mean([m.predict_proba(X)[:, 1] for m in self.stage1])
        out, P = {}, []
        for k in self.ladder:
            s = p1 * np.mean([m.predict_proba(X)[:, 1] for m in self.stage2[k]])
            cal = self.calibrators[k]
            P.append(cal.predict_proba(self._logit([s]))[0, 1] if cal is not None else 0.0)
            out[f's_{k:g}'] = float(s)
        P = np.minimum.accumulate(np.array(P))
        ev = [p * (k - 1 - self.cost) + (1 - p) * self.miss_loss[k] for p, k in zip(P, self.ladder)]
        j = int(np.argmax(ev))
        out.update({f'p_{k:g}': float(p) for k, p in zip(self.ladder, P)})
        out.update(ev=float(ev[j]), tp_ev=float(self.ladder[j]))
        return out


class Rpc:
    def __init__(self, url):
        self.url, self.lock = url, asyncio.Semaphore(4)

    def _call(self, method, params):
        for i in range(6):
            try:
                req = urllib.request.Request(self.url, json.dumps({'jsonrpc': '2.0', 'id': 1, 'method': method,
                                             'params': params}).encode(), {'Content-Type': 'application/json'})
                d = json.load(urllib.request.urlopen(req, timeout=30))
                if 'result' in d:
                    return d['result']
            except Exception:
                pass
            time.sleep(0.5 * 2 ** i)
        return None

    async def get_tx(self, sig):
        async with self.lock:
            return await asyncio.to_thread(self._call, 'getTransaction',
                                           [sig, {'encoding': 'json', 'maxSupportedTransactionVersion': 1,
                                                  'commitment': 'confirmed'}])


class Trader:
    def __init__(self, args):
        self.args = args
        os.makedirs(args.out, exist_ok=True)
        self.model = Model(args.model or os.path.join(HERE, 'model_bundle.pkl'))
        ws = pd.read_parquet(os.path.join(HERE, 'wallet_stats.parquet'))
        self.wallets = {w: [a, h] for w, a, h in zip(ws['wallet'], ws['appear'], ws['hit'])}
        self.rpc = Rpc(f"https://mainnet.helius-rpc.com/?api-key={args.key}")
        self.started = time.time()
        self.seq = 0
        self.creations = {}                       # mint -> creation info (seen live)
        self.curve = {}                           # mint -> streamed bonding-curve aggregates
        self.pools = {}                           # pool -> mint for tracked migrations
        self.tokens = {}                          # mint -> tracked migration state
        self.trades_today = collections.Counter()
        self.daily_cap = max(1, math.ceil(self.model.daily_universe_median * self.model.cap_pct / 100))
        self.stats = collections.Counter()
        self.gaps = []                            # (start, end) wall times with no stream
        self.down_since = None
        self.files = {}
        log(f"model trained through {self.model.trained_through} on {self.model.n_train:,} tokens; "
            f"margin={self.model.margin} cap={self.model.cap_pct}% (~{self.daily_cap}/day); wallets={len(self.wallets):,}")

    # ---------- output
    def write(self, name, row):
        path = os.path.join(self.args.out, f'{name}.csv')
        new = not os.path.exists(path)
        if name not in self.files:
            self.files[name] = open(path, 'a', newline='')
        f = self.files[name]
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row); f.flush()

    # ---------- stream handling
    def on_logs(self, value, slot):
        if value.get('err') is not None:
            return
        self.seq += 1
        for prog, name, f in events_from_logs(value['logs']):
            if name == 'TradeEvent':
                self.on_curve_trade(f, slot)
            elif name == 'CreateEvent':
                self.creations.setdefault(f['mint'], {
                    'sig': value['signature'], 'created_ts': f['timestamp'], 'name': f['name'], 'symbol': f['symbol'],
                    'creator': f['creator'] or f['user'], 'token_program': f['token_program'],
                    'is_cashback_enabled': f.get('is_cashback_enabled', False), 'mayhem': f.get('is_mayhem_mode', False)})
            elif name == 'CreatePoolEvent':
                if any(l.startswith(f'Program {PUMP} invoke') for l in value['logs']):
                    self.on_migration(f, value['signature'])
            elif name in ('BuyEvent', 'SellEvent'):
                self.on_amm_trade(name, f, slot)

    def on_curve_trade(self, f, slot):
        a = self.curve.get(f['mint'])
        if a is None:
            a = self.curve[f['mint']] = {'sol': 0.0, 'wallets': set(), 'first_ts': f['timestamp'], 'n': 0,
                                         'last_price': None, 'last_seen': 0}
        a['sol'] += f['sol_amount'] / 1e9; a['wallets'].add(f['user']); a['n'] += 1
        a['last_price'] = f['virtual_sol_reserves'] / f['virtual_token_reserves'] if f['virtual_token_reserves'] else None
        a['last_seen'] = time.time()

    def on_migration(self, f, sig):
        mint, pool = f['base_mint'], f['pool']
        if f['quote_mint'] != WSOL or mint in self.tokens:
            return
        self.stats['migrations'] += 1
        t = {'mint': mint, 'pool': pool, 'mt': f['timestamp'], 'pool_sol': f['quote_amount_in'] / 1e9,
             'mayhem': bool(f.get('is_mayhem_mode')), 'rows': [], 'sig': sig}
        self.tokens[mint] = t; self.pools[pool] = mint
        asyncio.get_running_loop().create_task(self.lifecycle(t))

    def on_amm_trade(self, name, f, slot):
        mint = self.pools.get(f['pool'])
        if mint is None:
            return
        t = self.tokens[mint]
        if name == 'BuyEvent':
            b = f['pool_base_token_reserves'] - f['base_amount_out']
            q = f['pool_quote_token_reserves'] + f['quote_amount_in_with_lp_fee']
            d, sol, tok = 'B', f['quote_amount_in'], f['base_amount_out']
        else:
            b = f['pool_base_token_reserves'] + f['base_amount_in']
            q = f['pool_quote_token_reserves'] - f['quote_amount_out_without_lp_fee']
            d, sol, tok = 'S', f['quote_amount_out'], f['base_amount_in']
        t['rows'].append({'mint': mint, 'ts_rel': f['timestamp'] - t['mt'], 'slot': slot, 'tx_idx': self.seq,
                          'direction': d, 'sol': sol / 1e9, 'tok': tok / 1e6, 'wallet': f['user'],
                          'new_price': (q + VIRTUAL_QUOTE_RESERVES) / b if b > 0 else 0.0, 'recv': time.time()})

    # ---------- per-token lifecycle
    async def lifecycle(self, t):
        await asyncio.sleep(max(0, t['mt'] + DECISION_S + SETTLE_S - time.time()))
        try:
            await self.decide(t)
        except Exception as e:
            log('decide error', t['mint'], repr(e)); self.stats['errors'] += 1
        await asyncio.sleep(max(0, t['mt'] + HORIZON_S + SETTLE_S - time.time()))
        try:
            self.resolve(t)
        except Exception as e:
            log('resolve error', t['mint'], repr(e)); self.stats['errors'] += 1
        self.pools.pop(t['pool'], None); self.tokens.pop(t['mint'], None)

    async def decide(self, t):
        mint, mt = t['mint'], t['mt']
        rows = sorted([r for r in t['rows'] if r['ts_rel'] <= DECISION_S], key=lambda r: (r['slot'], r['tx_idx']))
        cre, agg = self.creations.get(mint), self.curve.get(mint)
        complete = cre is not None and cre['created_ts'] >= self.started   # whole curve history streamed live
        priced = [r for r in rows if r['new_price'] > 0]
        n_amm_priced = len(priced)
        pre_total = (agg['n'] if agg else 0) + n_amm_priced
        age = mt - cre['created_ts'] if cre else None
        in_universe = (t['pool_sol'] >= MIN_POOL_SOL and not t['mayhem'] and pre_total >= MIN_PRE_TRADES
                       and (age is None or age <= MAX_AGE_S))
        base = {'mint': mint, 'mt': mt, 'utc': pd.to_datetime(mt, unit='s').strftime('%Y-%m-%d %H:%M:%S'),
                'pool_sol': round(t['pool_sol'], 3), 'pre_total': pre_total, 'history_complete': int(complete),
                'n_amm_60s': len(rows), 'stream_gap': int(self.had_gap((cre['created_ts'] if cre else mt) - 5, mt + DECISION_S + 5))}
        if not in_universe or not priced:
            self.stats['skipped_universe'] += 1
            self.write('migrations', {**base, 'in_universe': 0})
            return
        self.stats['universe'] += 1
        t['entry'] = priced[-1]['new_price']
        feats = amm_features(mint, rows)
        if agg:
            feats.update(curve_features(agg, mt, priced[0]['new_price'], n_amm_priced))
        tx_feats = {}
        if cre:
            tx = await self.rpc.get_tx(cre['sig'])
            if tx:
                tx_feats = creation_tx_features(tx, cre['creator'], PUMP)
            feats.update(launch_features(cre, tx_feats, mt))
        buyers = collections.defaultdict(float)
        for r in rows:
            if r['direction'] == 'B':
                buyers[r['wallet']] += r['sol']
        t['buyers'] = list(buyers)
        feats.update(reputation_features(buyers, self.wallets))
        sc = self.model.score(feats)
        day = time.strftime('%Y-%m-%d', time.gmtime(mt))
        take = sc['ev'] > self.model.margin and self.trades_today[day] < self.daily_cap
        if take:
            self.trades_today[day] += 1
        t.update(tp=sc['tp_ev'] if take else 0.0, decided_at=time.time(), score=sc)
        missing = [c for c in self.model.features if c not in feats or feats[c] is None or
                   (isinstance(feats[c], float) and math.isnan(feats[c]))]
        row = {**base, 'in_universe': 1, 'entry': t['entry'], 'decision_lag_s': round(time.time() - mt - DECISION_S, 1),
               'tp': t['tp'], 'ev': round(sc['ev'], 4), **{k: round(v, 5) for k, v in sc.items() if k.startswith('p_')},
               'n_missing_features': len(missing), 'missing': ' '.join(missing[:8])}
        self.write('decisions', row)
        self.write('features', {'mint': mint, 'mt': mt, **{c: feats.get(c) for c in self.model.features}})
        with open(os.path.join(self.args.out, 'rows60.jsonl'), 'a') as fh:
            fh.write(json.dumps({'mint': mint, 'pool': t['pool'], 'mt': mt, 'rows': rows,
                                 'curve': {k: (len(v) if k == 'wallets' else v) for k, v in (agg or {}).items()}}) + '\n')
        log(f"{'BUY ' if take else 'skip'} {mint[:8]}.. ev={sc['ev']:+.3f} tp={sc['tp_ev']:g} p5={sc['p_5']:.3f} "
            f"pool={t['pool_sol']:.0f} trades60={len(rows)} complete={int(complete)} missing={len(missing)}")

    def resolve(self, t):
        if 'entry' not in t:
            return
        rows = sorted(t['rows'], key=lambda r: (r['slot'], r['tx_idx']))
        after = [r['new_price'] / t['entry'] for r in rows if DECISION_S < r['ts_rel'] <= HORIZON_S and r['new_price'] > 0]
        upto = [r['new_price'] for r in rows if r['ts_rel'] <= HORIZON_S and r['new_price'] > 0]
        final_x = upto[-1] / t['entry'] if upto else 1.0
        if after:
            xs = sorted(after, reverse=True)
            n_trim = max(1, len(xs) // 100)
            max_x = xs[n_trim] if len(xs) > n_trim else xs[-1]
            raw_max = xs[0]
        else:
            max_x = raw_max = 1.0
        # realistic fill: first price we could have hit after deciding (decision time + 2 s)
        fill = next((r['new_price'] for r in rows if r['recv'] >= t.get('decided_at', 0) + 2 and r['new_price'] > 0), None)
        tp, c = t.get('tp', 0.0), self.model.cost
        pnl = lambda mx: 0.0 if tp <= 0 else (tp - 1 - c if mx >= tp else max(final_x - 1 - c, -1.0))
        row = {'mint': t['mint'], 'mt': t['mt'], 'tp': tp, 'ev': round(t['score']['ev'], 4), 'max_x': round(max_x, 4),
               'raw_max_x': round(raw_max, 4), 'final_x': round(final_x, 4), 'pnl': round(pnl(max_x), 4),
               'n_trades_1h': len(after), 'late_fill_x': round(fill / t['entry'], 4) if fill else None}
        if fill and tp > 0:   # same trade but entered at the realistic fill price
            mx_f, fx_f = max_x * t['entry'] / fill, final_x * t['entry'] / fill
            row['pnl_late_fill'] = round(tp - 1 - c if mx_f >= tp else max(fx_f - 1 - c, -1.0), 4)
        self.write('outcomes', row)
        for w in t.get('buyers', []):          # wallet reputation: commit the resolved token (1 h embargo)
            s = self.wallets.setdefault(w, [0, 0]); s[0] += 1; s[1] += int(max_x >= 5)
        if tp > 0:
            log(f"RESULT {t['mint'][:8]}.. tp={tp:g} max_x={max_x:.2f} final_x={final_x:.2f} pnl={row['pnl']:+.2f}")

    def had_gap(self, a, b):
        gaps = self.gaps + ([(self.down_since, time.time())] if self.down_since else [])
        return a < self.started or any(s < b and e > a for s, e in gaps)

    # ---------- housekeeping
    async def janitor(self):
        while True:
            await asyncio.sleep(300)
            now = time.time()
            for m in [m for m, a in self.curve.items() if now - a['last_seen'] > CURVE_KEEP_S and m not in self.tokens]:
                del self.curve[m]
            for m in [m for m, c in self.creations.items() if now - c['created_ts'] > 48 * 3600 and m not in self.tokens]:
                del self.creations[m]
            st = {**self.stats, 'tracked': len(self.tokens), 'curve_tokens': len(self.curve), 'uptime_h': round((now - self.started) / 3600, 2),
                  'trades_today': dict(self.trades_today)}
            json.dump(st, open(os.path.join(self.args.out, 'status.json'), 'w'), indent=1)
            log('status', st)

    async def stream(self):
        url = f"wss://mainnet.helius-rpc.com/?api-key={self.args.key}"
        while True:
            try:
                async with websockets.connect(url, max_size=2 ** 24, ping_interval=20, ping_timeout=60) as ws:
                    subs = {}
                    for i, p in enumerate((PUMP, AMM)):
                        await ws.send(json.dumps({'jsonrpc': '2.0', 'id': i, 'method': 'logsSubscribe',
                                                  'params': [{'mentions': [p]}, {'commitment': 'confirmed'}]}))
                    log('stream connected')
                    if self.down_since:
                        self.gaps.append((self.down_since, time.time())); self.down_since = None
                    async for msg in ws:
                        d = json.loads(msg)
                        if 'params' in d:
                            r = d['params']['result']
                            self.on_logs(r['value'], r['context']['slot'])
                        elif 'error' in d:
                            log('subscribe error', d['error'])
            except Exception as e:
                self.stats['reconnects'] += 1
                self.down_since = self.down_since or time.time()
                log('stream dropped:', repr(e), '- reconnecting in 3 s')
                await asyncio.sleep(3)

    async def run(self):
        tasks = [asyncio.create_task(self.stream()), asyncio.create_task(self.janitor())]
        if self.args.hours:
            await asyncio.sleep(self.args.hours * 3600)
            log('time limit reached; waiting for open positions to resolve is skipped')
            for t in tasks:
                t.cancel()
        else:
            await asyncio.gather(*tasks)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(HERE, 'runs', time.strftime('%Y%m%d_%H%M%S')))
    ap.add_argument('--hours', type=float, default=0, help='stop after this many hours (0 = run forever)')
    ap.add_argument('--key', default=os.environ.get('HELIUS_API_KEY'))
    ap.add_argument('--model', default=None, help='path to model bundle (default: model_bundle.pkl)')
    args = ap.parse_args()
    if not args.key:
        sys.exit('Set HELIUS_API_KEY or pass --key')
    asyncio.run(Trader(args).run())
