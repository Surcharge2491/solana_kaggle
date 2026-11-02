"""Ports of the Rust bot's unit tests (strategy, fair_value, inventory, market_maker)."""

import unittest

from hft_mm import (
    Book, BookEvent, Cancel, CancelAll, Config, FairValue, Fill, FillEvent, Inventory,
    MarketMaker, NotReady, Place, PlaceFailed, PositionEvent, Quotes, Reconnecting,
    Reference, Side, SyncFills, Tick, WARMUP_SECS, quote,
)
from hft_mm.market_maker import FILL_CHECK_EVERY

TICK = 0.01
WIDE = Book.from_levels([(90, 1)], [(110, 1)])


def sconfig():
    return Config(spread_bps=10, skew_bps=10, order_size=0.1, max_position=1)


class StrategyTests(unittest.TestCase):
    def test_flat_quotes_spread_either_side(self):
        self.assertEqual(quote(100, 0, sconfig(), WIDE, TICK), Quotes(99.90, 100.10))

    def test_long_leans_both_quotes_down(self):
        self.assertEqual(quote(100, 0.5, sconfig(), WIDE, TICK), Quotes(99.85, 100.05))

    def test_short_leans_both_quotes_up(self):
        self.assertEqual(quote(100, -0.5, sconfig(), WIDE, TICK), Quotes(99.95, 100.15))

    def test_stops_buying_at_max_long(self):
        q = quote(100, 1, sconfig(), WIDE, TICK)
        self.assertIsNone(q.bid)
        self.assertIsNotNone(q.ask)

    def test_stops_selling_at_max_short(self):
        q = quote(100, -1, sconfig(), WIDE, TICK)
        self.assertIsNone(q.ask)
        self.assertIsNotNone(q.bid)

    def test_stops_buying_when_one_more_order_would_pass_the_max(self):
        self.assertIsNone(quote(100, 0.95, sconfig(), WIDE, TICK).bid)

    def test_never_crosses_the_book(self):
        book = Book.from_levels([(99.40, 1)], [(99.50, 1)])
        q = quote(100, 0, sconfig(), book, TICK)
        self.assertEqual(q.bid, 99.49)
        self.assertEqual(q.ask, 100.10)

    def test_rounds_bids_down_and_asks_up(self):
        self.assertEqual(quote(100.003, 0, sconfig(), WIDE, TICK), Quotes(99.90, 100.11))


def watched_for(secs):
    fv = FairValue()
    for s in range(secs + 1):
        fv.on_binance(99, 101, s)
        fv.on_bullet_mid(101, s, 5, 300)
    return fv, float(secs)


class FairValueTests(unittest.TestCase):
    def test_nothing_until_binance_speaks(self):
        self.assertIs(FairValue().get(0, 5), NotReady.NO_BINANCE_PRICE)

    def test_waits_for_warmup(self):
        fv, now = watched_for(30)
        self.assertIs(fv.get(now, 5), NotReady.WARMING_UP)

    def test_binance_mid_plus_basis_once_warm(self):
        fv, now = watched_for(60)
        self.assertAlmostEqual(fv.get(now, 5), 101)
        fv.on_binance(101, 103, now)
        self.assertAlmostEqual(fv.get(now, 5), 103)

    def test_stops_when_binance_goes_quiet(self):
        fv, now = watched_for(60)
        self.assertIs(fv.get(now + 6, 5), NotReady.BINANCE_STALE)


def fill(tid, side, price, size, fee=0.01, cid=None, finished=True):
    return Fill(tid, cid, side, price, size, fee, finished)


class InventoryTests(unittest.TestCase):
    def test_buy_low_sell_high(self):
        inv = Inventory()
        inv.on_fill(fill(1, Side.BUY, 100, 1))
        inv.on_fill(fill(2, Side.SELL, 101, 1))
        self.assertEqual(inv.position, 0)
        self.assertAlmostEqual(inv.pnl(500), 0.98)

    def test_open_position_valued_at_current_price(self):
        inv = Inventory()
        inv.on_fill(fill(1, Side.BUY, 100, 2))
        self.assertAlmostEqual(inv.pnl(99), -2.01)

    def test_same_fill_twice_counts_once(self):
        inv = Inventory()
        self.assertTrue(inv.on_fill(fill(1, Side.BUY, 100, 1)))
        self.assertFalse(inv.on_fill(fill(1, Side.BUY, 100, 1)))
        self.assertEqual(inv.position, 1)

    def test_adopted_position_starts_at_zero_profit(self):
        inv = Inventory()
        inv.adopt(3)
        self.assertEqual(inv.pnl(100), 0)
        inv.mark(100)
        self.assertEqual(inv.pnl(100), 0)
        self.assertAlmostEqual(inv.pnl(101), 3)


BOOK = Book.from_levels([(99.5, 10)], [(100.5, 10)])


def warmed_up():
    cfg = Config(spread_bps=10, skew_bps=0, order_size=1, max_position=2,
                 requote_threshold_bps=5, min_requote_ms=0)
    mm = MarketMaker(cfg, TICK)
    for s in range(int(WARMUP_SECS) + 1):
        mm.on_event(Reference(99.5, 100.5), s)
        mm.on_event(BookEvent(BOOK), s)
    return mm, WARMUP_SECS


def places(intents):
    return [(i.order.side, i.order.price) for i in intents if isinstance(i, Place)]


class MarketMakerTests(unittest.TestCase):
    def test_shutdown_cancels_then_checks_fills(self):
        mm, now = warmed_up()
        mm.requote(now)
        self.assertEqual(mm.stop(), [CancelAll(), SyncFills()])

    def test_quotes_both_sides_once_warmed_up(self):
        mm, now = warmed_up()
        self.assertEqual(places(mm.requote(now)), [(Side.BUY, 99.90), (Side.SELL, 100.10)])
        self.assertEqual(mm.requote(now), [])

    def test_no_quotes_before_basis_learned(self):
        mm, now = warmed_up()
        mm.fair = FairValue()
        mm.on_event(Reference(99.5, 100.5), now)
        self.assertEqual(mm.requote(now), [])

    def test_small_moves_keep_big_moves_replace(self):
        mm, now = warmed_up()
        mm.requote(now)
        mm.on_event(Reference(99.52, 100.52), now)
        self.assertEqual(mm.requote(now), [])
        mm.on_event(Reference(99.6, 100.6), now)
        intents = mm.requote(now)
        self.assertTrue(all(isinstance(i, Cancel) for i in intents[:2]), "cancels go first")
        self.assertEqual(places(intents), [(Side.BUY, 99.99), (Side.SELL, 100.21)])

    def test_refreshes_before_expiry(self):
        mm, now = warmed_up()
        mm.requote(now)
        half = mm.config.quote_ttl_secs / 2
        for t, expect_empty in ((now + half - 1, True), (now + half, False)):
            mm.on_event(Reference(99.5, 100.5), t)
            mm.on_event(BookEvent(BOOK), t)
            intents = mm.requote(t)
            if expect_empty:
                self.assertEqual(intents, [])
            else:
                self.assertEqual(places(intents), [(Side.BUY, 99.90), (Side.SELL, 100.10)])
                self.assertEqual(sum(isinstance(i, Cancel) for i in intents), 2)

    def test_pulls_when_binance_quiet(self):
        mm, now = warmed_up()
        mm.requote(now)
        later = now + mm.config.reference_stale_secs + 1
        mm.on_event(BookEvent(BOOK), later)
        intents = mm.requote(later)
        self.assertEqual(len(intents), 2)
        self.assertTrue(all(isinstance(i, Cancel) for i in intents))

    def test_fill_moves_position_and_requotes_filled_side(self):
        mm, now = warmed_up()
        bid = mm.requote(now)[0].order
        mm.on_event(FillEvent(fill(1, Side.BUY, bid.price, 1, 0, bid.client_id, True)), now)
        self.assertEqual(mm.inventory.position, 1)
        self.assertEqual(places(mm.requote(now)), [(Side.BUY, 99.90)])

    def test_loss_limit_pulls_for_good(self):
        mm, now = warmed_up()
        mm.requote(now)
        mm.on_event(FillEvent(fill(1, Side.BUY, 110, 1, 0, None, False)), now)
        intents = mm.requote(now)
        self.assertTrue(intents and all(isinstance(i, Cancel) for i in intents))
        self.assertEqual(mm.requote(now + 60), [])

    def test_failed_place_is_retried(self):
        mm, now = warmed_up()
        ids = tuple(i.order.client_id for i in mm.requote(now) if isinstance(i, Place))
        mm.on_event(PlaceFailed(ids), now)
        self.assertEqual(len(places(mm.requote(now))), 2)

    def test_adopts_small_position(self):
        mm, now = warmed_up()
        self.assertEqual(mm.on_event(PositionEvent(1), now), [])
        self.assertEqual(mm.inventory.position, 1)
        self.assertNotEqual(mm.requote(now), [])

    def test_refuses_hand_opened_position(self):
        mm, now = warmed_up()
        mm.requote(now)
        self.assertEqual(mm.on_event(PositionEvent(5), now), [CancelAll()])
        self.assertEqual(mm.requote(now), [])
        self.assertEqual(mm.requote(now + 60), [])

    def test_fill_check_every_ten_seconds(self):
        mm, now = warmed_up()
        self.assertEqual(mm.on_event(Tick(), now), [SyncFills()])
        self.assertEqual(mm.on_event(Tick(), now + 5), [])
        self.assertEqual(mm.on_event(Tick(), now + FILL_CHECK_EVERY), [SyncFills()])

    def test_missed_fill_recovered_once(self):
        mm, now = warmed_up()
        f = fill(42, Side.SELL, 100, 1, 0, None, False)
        mm.on_event(FillEvent(f), now)
        mm.on_event(FillEvent(f), now)
        self.assertEqual(mm.inventory.position, -1)

    def test_reconnect_pulls_and_checks_fills(self):
        mm, now = warmed_up()
        mm.requote(now)
        self.assertEqual(mm.on_event(Reconnecting(), now), [CancelAll(), SyncFills()])
        self.assertEqual(mm.requote(now), [])


class ConfigTests(unittest.TestCase):
    def test_reads_the_rust_bot_toml(self):
        import os
        import tempfile
        toml = 'network = "mainnet"\ndry_run = true\nspread_bps = "10"\norder_size = "0.05"\nmin_requote_ms = 500\n'
        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
            f.write(toml)
        try:
            c = Config.from_toml(f.name)
        finally:
            os.unlink(f.name)
        self.assertEqual((c.spread_bps, c.order_size, c.min_requote_ms), (10.0, 0.05, 500))

    def test_rejects_nonsense(self):
        with self.assertRaises(ValueError):
            Config(order_size=1, max_position=0.5)


if __name__ == "__main__":
    unittest.main()
