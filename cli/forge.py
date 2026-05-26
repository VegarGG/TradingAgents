"""IIC-FORGE F2 CLI: ``forge backtest ...`` commands."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import typer

from tradingagents.backtest.harness import BacktestHarness
from tradingagents.backtest.prices import (
    PriceFallbackChain, Resolution,
)
from tradingagents.backtest.strict_historical import StrictHistoricalChain
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.persistence.db import connect as iic_connect

forge_app = typer.Typer(help="IIC-FORGE backtest commands")
backtest_app = typer.Typer(help="Forward-test harness and leaderboard")
forge_app.add_typer(backtest_app, name="backtest")


# --------------------------------------------------------------------
# Helpers: build a PriceFallbackChain from a list of source names.
# --------------------------------------------------------------------

_SOURCE_FACTORIES = {
    "yfinance":      "tradingagents.backtest.sources.yfinance_source:YFinanceSource",
    "polygon":       "tradingagents.backtest.sources.polygon_source:PolygonSource",
    "alpha_vantage": "tradingagents.backtest.sources.alpha_vantage_source:AlphaVantageSource",
    "futu":          "tradingagents.backtest.sources.futu_source:FutuSource",
}


def _build_price_chain(source_names: list[str]) -> PriceFallbackChain:
    sources = []
    for name in source_names:
        spec = _SOURCE_FACTORIES.get(name)
        if not spec:
            typer.echo(f"warning: unknown price source {name!r}; skipping", err=True)
            continue
        mod_name, class_name = spec.split(":")
        import importlib
        cls = getattr(importlib.import_module(mod_name), class_name)
        sources.append(cls())
    return PriceFallbackChain(sources)


# --------------------------------------------------------------------
# Graph runner adapter — wraps TradingAgentsGraph.
# --------------------------------------------------------------------

class TradingAgentsGraphRunner:
    """Adapter implementing the GraphRunner Protocol against the real graph."""

    def __init__(self, config: dict):
        self.config = config

    def run(self, *, ticker, trade_date, persona_id, conn: sqlite3.Connection):
        # Import lazily to keep CLI startup fast.
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        from tradingagents.personas.loader import load_persona_from_file

        overlay = dict(self.config)
        overlay["persona_id"] = persona_id

        # Apply per-persona LLM overrides (matches cli/deepdive.py pattern).
        personas_dir = Path(__file__).resolve().parent.parent / "tradingagents" / "personas"
        persona_file = personas_dir / f"{persona_id}.yaml"
        if persona_file.exists():
            p = load_persona_from_file(str(persona_file))
            overlay["deep_think_llm"]  = p.llm.deep_think_llm
            overlay["quick_think_llm"] = p.llm.quick_think_llm
            if p.llm.deepseek_reasoning_effort is not None:
                overlay["deepseek_reasoning_effort"] = p.llm.deepseek_reasoning_effort
            selected = list(p.analysts.include)
        else:
            selected = ["market", "news", "fundamentals"]

        graph = TradingAgentsGraph(config=overlay, selected_analysts=selected)
        graph.propagate(ticker, trade_date)
        # Pull the decision back from the runs table the Run Recorder just wrote.
        row = conn.execute(
            "SELECT decision FROM runs WHERE run_id = ?", (graph.run_id,),
        ).fetchone()
        decision = (row["decision"] if row and row["decision"] else "HOLD")
        return graph.run_id, decision


# --------------------------------------------------------------------
# `forge backtest start`
# --------------------------------------------------------------------

@backtest_app.command("start")
def backtest_start(
    watchlist: Optional[str] = typer.Option(
        None, "--watchlist", help="Comma-separated tickers (watchlist mode)"
    ),
    brief_id: Optional[str] = typer.Option(
        None, "--brief-id", help="Brief ID (brief-scoped mode); mutually exclusive with --watchlist"
    ),
    start_date_s: Optional[str] = typer.Option(
        None, "--start-date", help="YYYY-MM-DD (defaults to today)"
    ),
    end_date_s: Optional[str] = typer.Option(
        None, "--end-date", help="YYYY-MM-DD (defaults to start+30 days)"
    ),
    resolution_s: Optional[str] = typer.Option(
        None, "--resolution", help="1d | 1m  (1m raises until a 1m source is registered)"
    ),
    sources_s: Optional[str] = typer.Option(
        None, "--sources", help="Comma-separated source priority order; default from config"
    ),
    personas_s: Optional[str] = typer.Option(
        None, "--personas", help="Comma-separated persona ids; default = all loaded"
    ),
):
    """Start a forward-test batch. Exactly one of --watchlist or --brief-id must be set."""
    if bool(watchlist) == bool(brief_id):
        typer.echo("error: provide exactly one of --watchlist or --brief-id", err=True)
        raise typer.Exit(code=2)

    config = dict(DEFAULT_CONFIG)
    resolution = Resolution(resolution_s or config["backtest_resolution_default"])
    source_names = (
        sources_s.split(",") if sources_s else config["backtest_price_sources"]
    )

    conn = iic_connect(config["iic_db_path"])
    chain = _build_price_chain(source_names)

    if brief_id:
        # No graph runner needed — brief-scoped reuses persisted decisions.
        harness = BacktestHarness(
            conn=conn, data_dir=config["iic_data_dir"],
            graph_runner=_NullGraphRunner(),
            price_chain=chain,
            resolution=resolution,
        )
        backtest_id = harness.run_brief_scoped(brief_id=brief_id)
    else:
        # Watchlist mode — parse inputs + maybe enable strict_historical.
        tickers = [t.strip().upper() for t in watchlist.split(",") if t.strip()]
        start_date = (
            date.fromisoformat(start_date_s) if start_date_s else date.today()
        )
        end_date = (
            date.fromisoformat(end_date_s) if end_date_s
            else start_date + timedelta(days=30)
        )

        # Auto-on strict historical when start_date < today (unless config forces).
        strict_cfg = config["backtest_strict_historical"]
        strict_on = (start_date < date.today()) if strict_cfg is None else bool(strict_cfg)
        effective_chain = (
            StrictHistoricalChain(chain, cutoff=end_date) if strict_on else chain
        )

        personas = (
            personas_s.split(",") if personas_s
            else _all_persona_ids()
        )

        harness = BacktestHarness(
            conn=conn, data_dir=config["iic_data_dir"],
            graph_runner=TradingAgentsGraphRunner(config),
            price_chain=effective_chain,
            resolution=resolution,
        )
        backtest_id = harness.run_watchlist(
            tickers=tickers, personas=personas,
            start_date=start_date, end_date=end_date,
        )

    typer.echo(f"backtest_id: {backtest_id}")
    return backtest_id


def _all_persona_ids() -> list[str]:
    from tradingagents.personas.loader import load_all_personas
    personas_dir = Path(__file__).resolve().parent.parent / "tradingagents" / "personas"
    return [p.id for p in load_all_personas(str(personas_dir))]


class _NullGraphRunner:
    def run(self, **kw):
        raise RuntimeError("brief-scoped mode must not invoke the graph")


# --------------------------------------------------------------------
# `forge backtest leaderboard`
# --------------------------------------------------------------------

@backtest_app.command("leaderboard")
def backtest_leaderboard(
    persona: Optional[str] = typer.Option(None, "--persona"),
    status: Optional[str] = typer.Option(
        None, "--status", help="open | closed (default: all)"
    ),
    no_mtm: bool = typer.Option(
        False, "--no-mtm", help="Skip live mark-to-market for open rows (faster)"
    ),
):
    """Show open + closed forward-test rows with current performance."""
    from tradingagents.backtest.leaderboard import build_leaderboard

    config = dict(DEFAULT_CONFIG)
    conn = iic_connect(config["iic_db_path"])
    chain = None if no_mtm else _build_price_chain(config["backtest_price_sources"])

    rows = build_leaderboard(conn, price_chain=chain, persona=persona,
                              status_filter=status)
    if not rows:
        typer.echo("(no rows)")
        return

    typer.echo(
        f"{'btr':>4}  {'persona':<10} {'ticker':<6} {'status':<8} "
        f"{'decision':<5} {'TR':>8} {'alpha':>8}"
    )
    for r in rows:
        tr = r.get("total_return") if r["status"] == "closed" else r.get("mtm_return")
        al = r.get("alpha") if r["status"] == "closed" else r.get("mtm_alpha")
        tr_s = f"{tr:+.4f}" if tr is not None else " - "
        al_s = f"{al:+.4f}" if al is not None else " - "
        typer.echo(
            f"{r['btr_id']:>4}  {r['persona_id'] or '-':<10} {r['ticker']:<6} "
            f"{r['status']:<8} {(r.get('decision') or '-'):<5} "
            f"{tr_s:>8} {al_s:>8}"
        )


# --------------------------------------------------------------------
# `forge backtest report`
# --------------------------------------------------------------------

@backtest_app.command("report")
def backtest_report(
    backtest_id: int = typer.Argument(..., help="backtest_id from `backtest start`"),
):
    """Render the deterministic Markdown report for a backtest."""
    from tradingagents.backtest.report import render_report

    config = dict(DEFAULT_CONFIG)
    conn = iic_connect(config["iic_db_path"])
    md = render_report(conn, backtest_id=backtest_id)

    out_dir = Path(config["iic_data_dir"]) / "backtests" / str(backtest_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.md"
    report_path.write_text(md, encoding="utf-8")

    conn.execute(
        "UPDATE backtests SET report_path = ? WHERE backtest_id = ?",
        (str(report_path.relative_to(config["iic_data_dir"])), backtest_id),
    )
    conn.commit()
    typer.echo(f"wrote {report_path}")


# --------------------------------------------------------------------
# `forge backtest close` — manual single-row maturation
# --------------------------------------------------------------------

@backtest_app.command("close")
def backtest_close(
    btr_id: int = typer.Argument(..., help="backtest_runs.btr_id"),
):
    """Manually mature a single open forward test by btr_id."""
    import json
    from tradingagents.backtest.harness import BacktestHarness

    config = dict(DEFAULT_CONFIG)
    conn = iic_connect(config["iic_db_path"])
    row = conn.execute(
        "SELECT backtest_id, persona_id, ticker, metrics FROM backtest_runs WHERE btr_id = ?",
        (btr_id,),
    ).fetchone()
    if not row:
        typer.echo(f"btr_id {btr_id} not found", err=True)
        raise typer.Exit(code=1)
    m = json.loads(row["metrics"])
    if m.get("status") != "open":
        typer.echo(f"btr_id {btr_id} has status {m.get('status')!r}; nothing to close")
        return

    chain = _build_price_chain(config["backtest_price_sources"])
    harness = BacktestHarness(
        conn=conn, data_dir=config["iic_data_dir"],
        graph_runner=_NullGraphRunner(), price_chain=chain,
    )
    harness._mature_one(
        btr_id=btr_id,
        persona_id=row["persona_id"],
        ticker=row["ticker"],
        metrics=m,
        end_date=date.fromisoformat(m["scheduled_close_date"]),
    )
    typer.echo(f"closed btr_id {btr_id}")
