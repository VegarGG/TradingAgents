import json
import uuid
from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def conn(tmp_path):
    from tradingagents.persistence.db import connect
    return connect(str(tmp_path / "iic.db"))


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    return str(d)


@pytest.fixture
def fake_graph_runner():
    """Mock graph runner: returns a fresh run_id and a deterministic decision
    per (ticker, persona)."""
    from tradingagents.persistence import store

    class FakeRunner:
        def __init__(self):
            self.invocations = []

        def run(self, *, ticker, trade_date, persona_id, conn):
            run_id = uuid.uuid4().hex
            now = datetime.now(timezone.utc).isoformat()
            store.insert_run(conn, run_id=run_id, ticker=ticker,
                             persona_id=persona_id, started_ts=now,
                             artifact_dir=f"runs/{run_id}")
            decision = {"macro": "BUY", "value": "HOLD", "momentum": "SELL"}.get(
                persona_id, "HOLD"
            )
            store.finalize_run(conn, run_id=run_id, ended_ts=now,
                               status="complete", decision=decision)
            self.invocations.append((ticker, trade_date, persona_id))
            return run_id, decision
    return FakeRunner()


@pytest.fixture
def fake_price_chain():
    """Mock chain returning a 2-bar synthetic series (start, end)."""
    from tradingagents.backtest.prices import Bars, Resolution

    class FakeChain:
        def __init__(self):
            self.calls = []

        def get_bars(self, ticker, start, end, resolution):
            self.calls.append((ticker, start, end, resolution))
            if start == end:
                return Bars(
                    ticker=ticker, resolution=resolution,
                    bars=[(datetime.combine(start, datetime.min.time()), 100.0)],
                    source="fake",
                )
            return Bars(
                ticker=ticker, resolution=resolution,
                bars=[
                    (datetime.combine(start, datetime.min.time()), 100.0),
                    (datetime.combine(end,   datetime.min.time()), 110.0),
                ],
                source="fake",
            )

    return FakeChain()


@pytest.mark.unit
def test_watchlist_open_inserts_backtests_and_backtest_runs(
    conn, data_dir, fake_graph_runner, fake_price_chain
):
    from tradingagents.backtest.harness import BacktestHarness
    h = BacktestHarness(conn=conn, data_dir=data_dir,
                         graph_runner=fake_graph_runner,
                         price_chain=fake_price_chain)
    backtest_id = h.run_watchlist(
        tickers=["AAPL", "MSFT"],
        personas=["macro", "value"],
        start_date=date(2030, 1, 1),  # future → don't auto-mature
        end_date=date(2030, 1, 31),
    )

    backtests = list(conn.execute("SELECT * FROM backtests WHERE backtest_id=?",
                                    (backtest_id,)))
    assert len(backtests) == 1
    assert backtests[0]["status"] == "open"
    assert json.loads(backtests[0]["universe"]) == ["AAPL", "MSFT"]

    runs = list(conn.execute("SELECT * FROM backtest_runs WHERE backtest_id=?",
                              (backtest_id,)))
    assert len(runs) == 4  # 2 tickers × 2 personas
    for r in runs:
        m = json.loads(r["metrics"])
        assert m["status"] == "open"
        assert m["decision"] in ("BUY", "HOLD", "SELL")
        assert m["entry_price"] == pytest.approx(100.0)
        assert m["entry_date"] == "2030-01-01"
        assert m["scheduled_close_date"] == "2030-01-31"
        assert m["resolution"] == "1d"
        assert m["price_source"] == "fake"


@pytest.mark.unit
def test_watchlist_open_calls_graph_runner_per_ticker_persona(
    conn, data_dir, fake_graph_runner, fake_price_chain
):
    from tradingagents.backtest.harness import BacktestHarness
    h = BacktestHarness(conn=conn, data_dir=data_dir,
                         graph_runner=fake_graph_runner,
                         price_chain=fake_price_chain)
    h.run_watchlist(tickers=["AAPL", "MSFT"],
                     personas=["macro", "value", "momentum"],
                     start_date=date(2030, 1, 1),
                     end_date=date(2030, 1, 31))
    assert len(fake_graph_runner.invocations) == 6  # 2 × 3
    personas_seen = {p for (_, _, p) in fake_graph_runner.invocations}
    tickers_seen = {t for (t, _, _) in fake_graph_runner.invocations}
    assert personas_seen == {"macro", "value", "momentum"}
    assert tickers_seen == {"AAPL", "MSFT"}


@pytest.mark.unit
def test_watchlist_open_does_not_mature_future_window(
    conn, data_dir, fake_graph_runner, fake_price_chain
):
    """When end_date > today, all backtest_runs stay status=open."""
    from tradingagents.backtest.harness import BacktestHarness
    h = BacktestHarness(conn=conn, data_dir=data_dir,
                         graph_runner=fake_graph_runner,
                         price_chain=fake_price_chain)
    backtest_id = h.run_watchlist(tickers=["AAPL"], personas=["macro"],
                                    start_date=date(2030, 1, 1),
                                    end_date=date(2030, 1, 31))
    runs = list(conn.execute("SELECT metrics FROM backtest_runs WHERE backtest_id=?",
                              (backtest_id,)))
    for r in runs:
        m = json.loads(r["metrics"])
        assert m["status"] == "open"
        assert "close_date" not in m


@pytest.fixture
def historical_price_chain():
    """Mock chain returning a deterministic 6-bar series for any ticker."""
    from tradingagents.backtest.prices import Bars, Resolution

    class HistoricalChain:
        def __init__(self):
            self.bars_for = {
                ("AAPL", "entry"):    [(datetime(2026, 4, 26), 200.0)],
                ("AAPL", "full"):     [
                    (datetime(2026, 4, 26), 200.0),
                    (datetime(2026, 4, 30), 202.0),
                    (datetime(2026, 5,  5), 198.0),
                    (datetime(2026, 5, 12), 210.0),
                    (datetime(2026, 5, 20), 215.0),
                    (datetime(2026, 5, 26), 220.0),
                ],
                ("SPY",  "entry"):    [(datetime(2026, 4, 26), 500.0)],
                ("SPY",  "full"):     [
                    (datetime(2026, 4, 26), 500.0),
                    (datetime(2026, 4, 30), 501.0),
                    (datetime(2026, 5,  5), 499.0),
                    (datetime(2026, 5, 12), 505.0),
                    (datetime(2026, 5, 20), 507.0),
                    (datetime(2026, 5, 26), 510.0),
                ],
            }

        def get_bars(self, ticker, start, end, resolution):
            kind = "entry" if start == end else "full"
            bars = self.bars_for.get((ticker, kind), [])
            return Bars(ticker=ticker, resolution=resolution, bars=bars,
                        source="historical")

    return HistoricalChain()


@pytest.mark.unit
def test_watchlist_with_past_end_date_matures_inline(
    conn, data_dir, fake_graph_runner, historical_price_chain
):
    """When end_date <= today, all backtest_runs close inline."""
    from tradingagents.backtest.harness import BacktestHarness
    h = BacktestHarness(conn=conn, data_dir=data_dir,
                         graph_runner=fake_graph_runner,
                         price_chain=historical_price_chain)
    backtest_id = h.run_watchlist(
        tickers=["AAPL"], personas=["macro"],
        start_date=date(2026, 4, 26), end_date=date(2026, 5, 26),
    )

    backtests_row = conn.execute(
        "SELECT * FROM backtests WHERE backtest_id=?", (backtest_id,)
    ).fetchone()
    assert backtests_row["status"] == "closed"

    runs = list(conn.execute(
        "SELECT metrics FROM backtest_runs WHERE backtest_id=?", (backtest_id,)
    ))
    assert len(runs) == 1
    m = json.loads(runs[0]["metrics"])
    assert m["status"] == "closed"
    assert m["close_date"] == "2026-05-26"
    assert m["exit_price"] == pytest.approx(220.0)
    assert m["total_return"] == pytest.approx(0.10, rel=1e-6)
    assert m["benchmark_return"] == pytest.approx((510 - 500) / 500, rel=1e-6)
    assert m["alpha"] == pytest.approx(0.10 - 0.02, rel=1e-6)
    assert isinstance(m["returns"], list) and len(m["returns"]) >= 1
    assert "sharpe" in m and "max_drawdown" in m and "win_rate" in m
    assert m["holding_days_elapsed"] == 30


@pytest.mark.unit
def test_maturation_writes_outcome_log_row(
    conn, data_dir, fake_graph_runner, historical_price_chain
):
    from tradingagents.backtest.harness import BacktestHarness
    h = BacktestHarness(conn=conn, data_dir=data_dir,
                         graph_runner=fake_graph_runner,
                         price_chain=historical_price_chain)
    h.run_watchlist(tickers=["AAPL"], personas=["macro"],
                     start_date=date(2026, 4, 26), end_date=date(2026, 5, 26))
    rows = list(conn.execute("SELECT * FROM outcome_log"))
    assert len(rows) == 1
    assert rows[0]["ticker"] == "AAPL"
    tags = json.loads(rows[0]["tags"])
    assert tags["persona_id"] == "macro"
    assert tags["source"] == "forward_test"


@pytest.mark.unit
def test_maturation_handles_missing_exit_price_as_errored(
    conn, data_dir, fake_graph_runner
):
    """If the price chain raises for the exit fetch, the row is errored,
    not silently zero-returns."""
    from tradingagents.backtest.harness import BacktestHarness
    from tradingagents.backtest.prices import Bars, Resolution

    class FlakyChain:
        def get_bars(self, ticker, start, end, resolution):
            if start == end:  # entry fetch
                return Bars(ticker=ticker, resolution=resolution,
                            bars=[(datetime(2026, 4, 26), 100.0)], source="x")
            raise RuntimeError("exit fetch failed")

    h = BacktestHarness(conn=conn, data_dir=data_dir,
                         graph_runner=fake_graph_runner,
                         price_chain=FlakyChain())
    backtest_id = h.run_watchlist(
        tickers=["AAPL"], personas=["macro"],
        start_date=date(2026, 4, 26), end_date=date(2026, 5, 26),
    )
    runs = list(conn.execute("SELECT metrics FROM backtest_runs WHERE backtest_id=?",
                              (backtest_id,)))
    m = json.loads(runs[0]["metrics"])
    assert m["status"] == "errored"
    assert "exit" in m["error"].lower() or "fetch" in m["error"].lower()
