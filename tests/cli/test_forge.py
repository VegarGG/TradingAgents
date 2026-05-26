import pytest
from typer.testing import CliRunner


@pytest.mark.unit
def test_forge_command_exists():
    from cli.forge import forge_app
    runner = CliRunner()
    result = runner.invoke(forge_app, ["--help"])
    assert result.exit_code == 0
    assert "backtest" in result.stdout


@pytest.mark.unit
def test_backtest_start_help_lists_required_flags():
    from cli.forge import forge_app
    runner = CliRunner()
    result = runner.invoke(forge_app, ["backtest", "start", "--help"])
    assert result.exit_code == 0
    out = result.stdout
    assert "--watchlist" in out
    assert "--brief-id" in out
    assert "--start-date" in out


@pytest.mark.unit
def test_backtest_leaderboard_help():
    from cli.forge import forge_app
    runner = CliRunner()
    result = runner.invoke(forge_app, ["backtest", "leaderboard", "--help"])
    assert result.exit_code == 0
    assert "--persona" in result.stdout


@pytest.mark.unit
def test_backtest_report_help():
    from cli.forge import forge_app
    runner = CliRunner()
    result = runner.invoke(forge_app, ["backtest", "report", "--help"])
    assert result.exit_code == 0


@pytest.mark.unit
def test_backtest_close_help():
    from cli.forge import forge_app
    runner = CliRunner()
    result = runner.invoke(forge_app, ["backtest", "close", "--help"])
    assert result.exit_code == 0


@pytest.mark.unit
def test_backtest_report_writes_file(tmp_path, monkeypatch):
    """End-to-end: seed a closed backtest, run `report`, verify file."""
    import json, uuid
    from datetime import datetime, timezone

    iic_db = tmp_path / "iic.db"
    iic_data = tmp_path / "data"
    monkeypatch.setenv("TRADINGAGENTS_IIC_DB_PATH", str(iic_db))
    monkeypatch.setenv("TRADINGAGENTS_IIC_DATA_DIR", str(iic_data))

    # Force DEFAULT_CONFIG reload so env-var overrides apply.
    import importlib
    import tradingagents.default_config as dc
    importlib.reload(dc)
    import cli.forge as forge_mod
    importlib.reload(forge_mod)

    from tradingagents.persistence.db import connect
    from tradingagents.persistence import store
    conn = connect(str(iic_db))

    cur = conn.execute(
        "INSERT INTO backtests (universe, start_date, end_date, status, created_ts)"
        " VALUES (?, '2026-04-26', '2026-05-26', 'closed', ?)",
        (json.dumps(["AAPL"]), datetime.now(timezone.utc).isoformat()),
    )
    backtest_id = cur.lastrowid
    run_id = uuid.uuid4().hex
    store.insert_run(conn, run_id=run_id, ticker="AAPL", persona_id="macro",
                     started_ts=datetime.now(timezone.utc).isoformat(),
                     artifact_dir=f"runs/{run_id}")
    conn.execute(
        "INSERT INTO backtest_runs (backtest_id, persona_id, ticker, metrics)"
        " VALUES (?, ?, ?, ?)",
        (backtest_id, "macro", "AAPL", json.dumps({
            "status": "closed", "run_id": run_id, "decision": "BUY",
            "position": 1, "entry_date": "2026-04-26", "entry_price": 200.0,
            "close_date": "2026-05-26", "exit_price": 220.0,
            "total_return": 0.10, "benchmark_return": 0.02, "alpha": 0.08,
            "sharpe": 1.4, "max_drawdown": -0.02, "win_rate": 0.6,
            "returns": [], "holding_days_elapsed": 30,
        })),
    )
    conn.commit()
    conn.close()

    runner = CliRunner()
    result = runner.invoke(forge_mod.forge_app,
                            ["backtest", "report", str(backtest_id)])
    assert result.exit_code == 0, result.output
    report_path = iic_data / "backtests" / str(backtest_id) / "report.md"
    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "macro" in content


@pytest.mark.unit
def test_backtest_sweep_help():
    from cli.forge import forge_app
    runner = CliRunner()
    result = runner.invoke(forge_app, ["backtest", "sweep", "--help"])
    assert result.exit_code == 0


@pytest.mark.unit
def test_backtest_watch_help():
    from cli.forge import forge_app
    runner = CliRunner()
    result = runner.invoke(forge_app, ["backtest", "watch", "--help"])
    assert result.exit_code == 0
    assert "--interval" in result.stdout


@pytest.mark.unit
def test_backtest_sweep_prints_counts(tmp_path, monkeypatch):
    """End-to-end: empty DB → sweep prints zero counts."""
    iic_db = tmp_path / "iic.db"
    iic_data = tmp_path / "data"
    monkeypatch.setenv("TRADINGAGENTS_IIC_DB_PATH", str(iic_db))
    monkeypatch.setenv("TRADINGAGENTS_IIC_DATA_DIR", str(iic_data))

    import importlib
    import tradingagents.default_config as dc; importlib.reload(dc)
    import cli.forge as forge_mod; importlib.reload(forge_mod)

    # touch DB so schema lands
    from tradingagents.persistence.db import connect
    connect(str(iic_db)).close()

    runner = CliRunner()
    result = runner.invoke(forge_mod.forge_app, ["backtest", "sweep"])
    assert result.exit_code == 0
    assert "closed=0" in result.output
    assert "skipped=0" in result.output


@pytest.mark.unit
def test_cli_main_registers_forge():
    """cli.main must register the forge sub-app under `forge`."""
    from cli.main import app
    runner = CliRunner()
    result = runner.invoke(app, ["forge", "--help"])
    assert result.exit_code == 0
    assert "backtest" in result.stdout
