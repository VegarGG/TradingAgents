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
