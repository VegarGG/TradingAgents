"""F2 exit-gate boundary smoke test (no real LLM).

This is the structural / boundary check — it asserts the harness writes
the expected rows and the report is byte-equal on re-render. The actual
exit-gate run that spends LLM tokens lives in the runbook / Task 24.
"""

import json
import re
import uuid
from datetime import date, datetime, timedelta, timezone

import pytest


F2_EXIT_TICKERS = ["AAPL", "MSFT", "GOOG", "NVDA", "TSLA"]
F2_EXIT_PERSONAS = ["macro", "value", "momentum"]


@pytest.mark.smoke
def test_f2_exit_gate_structural_checks(tmp_path, monkeypatch):
    """Structural check — runs the harness with a mock graph runner and
    a deterministic price chain. Asserts:
      * `backtests` row inserted, status=closed at the end.
      * 15 `backtest_runs` rows, all status=closed (no errored).
      * 15 `outcome_log` rows with `tags.source='forward_test'`.
      * Re-rendering the report produces byte-equal content modulo `generated_ts`.
    """
    iic_db = tmp_path / "iic.db"
    iic_data = tmp_path / "data"
    memory_log = tmp_path / "memory" / "trading_memory.md"
    memory_log.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("TRADINGAGENTS_IIC_DB_PATH", str(iic_db))
    monkeypatch.setenv("TRADINGAGENTS_IIC_DATA_DIR", str(iic_data))
    monkeypatch.setenv("TRADINGAGENTS_MEMORY_LOG_PATH", str(memory_log))

    import importlib
    import tradingagents.default_config as dc
    importlib.reload(dc)

    from tradingagents.backtest.harness import BacktestHarness
    from tradingagents.backtest.prices import Bars, Resolution
    from tradingagents.persistence.db import connect
    from tradingagents.persistence import store

    # Deterministic price chain — same prices for every ticker.
    class DeterministicChain:
        def get_bars(self, ticker, start, end, resolution):
            if start == end:
                return Bars(
                    ticker=ticker, resolution=resolution,
                    bars=[(datetime.combine(start, datetime.min.time()), 100.0)],
                    source="det",
                )
            # 5-bar synthetic series across the window
            prices = [100.0, 105.0, 102.0, 108.0, 110.0]
            step = max(1, (end - start).days // (len(prices) - 1))
            bars = []
            for i, p in enumerate(prices):
                day = min(start + timedelta(days=i * step), end)
                bars.append((datetime.combine(day, datetime.min.time()), p))
            # Ensure last bar lands exactly on end
            bars[-1] = (datetime.combine(end, datetime.min.time()), prices[-1])
            return Bars(ticker=ticker, resolution=resolution, bars=bars,
                        source="det")

    # Fake graph runner — deterministic decision per persona.
    class FakeRunner:
        DECISION_BY_PERSONA = {"macro": "BUY", "value": "HOLD", "momentum": "SELL"}

        def run(self, *, ticker, trade_date, persona_id, conn):
            run_id = uuid.uuid4().hex
            now = datetime.now(timezone.utc).isoformat()
            store.insert_run(conn, run_id=run_id, ticker=ticker,
                             persona_id=persona_id, started_ts=now,
                             artifact_dir=f"runs/{run_id}")
            store.finalize_run(conn, run_id=run_id, ended_ts=now,
                               status="complete",
                               decision=self.DECISION_BY_PERSONA[persona_id])
            return run_id, self.DECISION_BY_PERSONA[persona_id]

    conn = connect(str(iic_db))
    harness = BacktestHarness(
        conn=conn, data_dir=str(iic_data),
        graph_runner=FakeRunner(), price_chain=DeterministicChain(),
    )

    end_date = date.today()
    start_date = end_date - timedelta(days=30)

    backtest_id = harness.run_watchlist(
        tickers=F2_EXIT_TICKERS,
        personas=F2_EXIT_PERSONAS,
        start_date=start_date,
        end_date=end_date,
    )

    # --- structural assertions ---
    bt = conn.execute("SELECT * FROM backtests WHERE backtest_id = ?",
                       (backtest_id,)).fetchone()
    assert bt["status"] == "closed"

    runs = list(conn.execute(
        "SELECT metrics FROM backtest_runs WHERE backtest_id = ?",
        (backtest_id,)))
    assert len(runs) == 15, f"expected 15 rows, got {len(runs)}"
    statuses = [json.loads(r["metrics"])["status"] for r in runs]
    assert statuses.count("closed") == 15, f"non-closed rows: {statuses}"

    outcome_rows = list(conn.execute("SELECT * FROM outcome_log"))
    assert len(outcome_rows) == 15
    for r in outcome_rows:
        tags = json.loads(r["tags"])
        assert tags["source"] == "forward_test"
        assert tags["persona_id"] in F2_EXIT_PERSONAS
        assert tags["backtest_id"] == backtest_id

    # --- report byte-equality ---
    from tradingagents.backtest.report import render_report
    md1 = render_report(conn, backtest_id=backtest_id)
    md2 = render_report(conn, backtest_id=backtest_id)
    rx = re.compile(r"^generated_ts:.*$", re.MULTILINE)
    assert rx.sub("", md1) == rx.sub("", md2), "report not byte-equal on rerun"
    for persona in F2_EXIT_PERSONAS:
        assert persona in md1
    for ticker in F2_EXIT_TICKERS:
        assert ticker in md1


@pytest.mark.smoke
@pytest.mark.integration
def test_f2_exit_gate_real_run(tmp_path, monkeypatch):
    """Real LLM, real yfinance — back-dated 30-day window.

    Runs only when explicitly selected (``pytest -m integration`` or
    ``pytest tests/smoke/test_f2_exit_gate.py::test_f2_exit_gate_real_run``).
    Spends real LLM tokens; expected runtime ~15 minutes.
    """
    import os
    if not os.getenv("F2_RUN_REAL_EXIT_GATE"):
        pytest.skip("set F2_RUN_REAL_EXIT_GATE=1 to spend LLM budget")

    iic_db = tmp_path / "iic.db"
    iic_data = tmp_path / "data"
    memory_log = tmp_path / "memory" / "trading_memory.md"
    memory_log.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("TRADINGAGENTS_IIC_DB_PATH", str(iic_db))
    monkeypatch.setenv("TRADINGAGENTS_IIC_DATA_DIR", str(iic_data))
    # Isolate memory_log so we don't trigger reflection against the user's
    # actual prior runs (pending entries would try to score via the LLM
    # before the backtest's own decision-making starts).
    monkeypatch.setenv("TRADINGAGENTS_MEMORY_LOG_PATH", str(memory_log))

    # tests/conftest.py's `_dummy_api_keys` autouse fixture sets every API
    # key env var to "placeholder" if not already present. For this
    # integration test we need the REAL keys from .env, so explicitly
    # reload .env with override=True after monkeypatch has run.
    from dotenv import find_dotenv, load_dotenv
    load_dotenv(find_dotenv(usecwd=True), override=True)

    import importlib
    import tradingagents.default_config as dc; importlib.reload(dc)
    import cli.forge as forge_mod; importlib.reload(forge_mod)

    from typer.testing import CliRunner
    runner = CliRunner()
    end_date = date.today()
    start_date = end_date - timedelta(days=30)
    result = runner.invoke(
        forge_mod.forge_app,
        ["backtest", "start",
         "--watchlist", "AAPL,MSFT,GOOG,NVDA,TSLA",
         "--start-date", start_date.isoformat()],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    m = re.search(r"backtest_id:\s*(\d+)", result.output)
    assert m, f"no backtest_id in: {result.output}"
    backtest_id = int(m.group(1))

    result = runner.invoke(forge_mod.forge_app,
                            ["backtest", "report", str(backtest_id)])
    assert result.exit_code == 0
    report_path = iic_data / "backtests" / str(backtest_id) / "report.md"
    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    for persona in ("macro", "value", "momentum"):
        assert persona in content
    assert "AAPL" in content
