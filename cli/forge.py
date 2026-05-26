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
