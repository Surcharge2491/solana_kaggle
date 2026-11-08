"""Decode pump.fun / PumpSwap Anchor events from transaction logs or inner instructions, using the on-chain IDLs."""
import base64, json, os, struct
import base58

HERE = os.path.dirname(os.path.abspath(__file__))
EVENT_IX_TAG = bytes([0xe4, 0x45, 0xa5, 0x2e, 0x51, 0xcb, 0x9a, 0x1d])  # anchor emit_cpi! prefix
PROGRAMS = {'pump': '6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P', 'pump_amm': 'pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA'}
WANTED = {'pump': {'TradeEvent', 'CreateEvent', 'CompletePumpAmmMigrationEvent'},
          'pump_amm': {'BuyEvent', 'SellEvent', 'CreatePoolEvent'}}

class _Reader:
    def __init__(self, b): self.b, self.o = b, 0
    def take(self, n):
        v = self.b[self.o:self.o + n]
        if len(v) < n: raise EOFError
        self.o += n; return v

def _load(name):
    idl = json.load(open(os.path.join(HERE, f'idl_{name}.json')))
    types = {t['name']: t for t in idl['types']}
    events = {bytes(e['discriminator']): e['name'] for e in idl['events'] if e['name'] in WANTED[name]}
    return types, events

_IDL = {n: _load(n) for n in PROGRAMS}

def _read(r, t, types):
    if isinstance(t, str):
        if t == 'pubkey': return base58.b58encode(r.take(32)).decode()
        if t == 'bool': return r.take(1)[0] != 0
        if t == 'string': n = struct.unpack('<I', r.take(4))[0]; return r.take(n).decode('utf-8', 'replace')
        fmt = {'u8': '<B', 'i8': '<b', 'u16': '<H', 'i16': '<h', 'u32': '<I', 'i32': '<i', 'u64': '<Q', 'i64': '<q'}.get(t)
        if fmt: return struct.unpack(fmt, r.take(struct.calcsize(fmt)))[0]
        if t in ('u128', 'i128'): return int.from_bytes(r.take(16), 'little', signed=t == 'i128')
        raise ValueError(t)
    if 'vec' in t: n = struct.unpack('<I', r.take(4))[0]; return [_read(r, t['vec'], types) for _ in range(n)]
    if 'option' in t: return _read(r, t['option'], types) if r.take(1)[0] else None
    if 'array' in t: et, n = t['array']; return [_read(r, et, types) for _ in range(n)]
    if 'defined' in t:
        d = types[t['defined']['name']]['type']
        if d['kind'] == 'struct': return {f['name']: _read(r, f['type'], types) for f in d['fields']}
        if d['kind'] == 'enum': return d['variants'][r.take(1)[0]]['name']
    raise ValueError(t)

def decode_event(data: bytes, program: str):
    """Return (event_name, fields) or None. Trailing fields missing in older events are tolerated."""
    types, events = _IDL[program]
    name = events.get(data[:8])
    if name is None: return None
    r, out = _Reader(data[8:]), {}
    for f in types[name]['type']['fields']:
        try: out[f['name']] = _read(r, f['type'], types)
        except EOFError: break
    return name, out

def events_from_logs(logs):
    """Events emitted with emit! appear as 'Program data: <base64>' lines inside the emitting program's frame."""
    out, stack = [], []
    for line in logs:
        if line.startswith('Program ') and ' invoke [' in line: stack.append(line.split()[1])
        elif line.startswith('Program ') and (line.endswith(' success') or ' failed' in line) and stack: stack.pop()
        elif line.startswith('Program data: ') and stack:
            prog = next((n for n, p in PROGRAMS.items() if p == stack[-1]), None)
            if prog:
                ev = decode_event(base64.b64decode(line[14:]), prog)
                if ev: out.append((prog,) + ev)
    return out

def events_from_inner_ix(tx):
    """Events emitted with emit_cpi! are self-CPIs whose data starts with EVENT_IX_TAG (needs getTransaction)."""
    msg = tx['transaction']['message']; keys = list(msg['accountKeys'])
    la = tx['meta'].get('loadedAddresses') or {}
    keys += la.get('writable', []) + la.get('readonly', [])
    out = []
    for grp in tx['meta'].get('innerInstructions') or []:
        for ix in grp['instructions']:
            prog_id = keys[ix['programIdIndex']]
            prog = next((n for n, p in PROGRAMS.items() if p == prog_id), None)
            if not prog: continue
            data = base58.b58decode(ix['data'])
            if data[:8] == EVENT_IX_TAG:
                ev = decode_event(data[8:], prog)
                if ev: out.append((prog,) + ev)
    return out
