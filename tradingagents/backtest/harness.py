"""F2 BacktestHarness — orchestrator for the two invocation modes.

Watchlist mode: open + maturation (this file, Tasks 14-15).
Brief-scoped mode: open from persisted runs + maturation (Task 16).

The harness is decoupled from TradingAgentsGraph via the GraphRunner
Protocol so tests can substitute a fast mock. Same for prices via
PriceFallbackChain (or anything quacking get_bars(...) → Bars).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import List, Optional, Protocol, Tuple

from tradingagents.backtest.prices import Resolution
from tradingagents.backtest.simulator import position_from_decision


class GraphRunner(Protocol):
    """Anything that can invoke a per-persona graph and persist a runs row."""
    def run(self, *, ticker: str, trade_date: str, persona_id: str,
            conn: sqlite3.Connection) -> Tuple[str, str]:
        """Return ``(run_id, decision)``. The runs row must already be written."""
        ...


@dataclass
class BacktestHarness:
    conn: sqlite3.Connection
    data_dir: str
    graph_runner: GraphRunner
    price_chain: object   # any get_bars(...) → Bars producer
    resolution: Resolution = Resolution.DAILY
    benchmark: str = "SPY"

    def run_watchlist(
        self,
        *,
        tickers: List[str],
        personas: List[str],
        start_date: date,
        end_date: date,
    ) -> int:
        """Open one backtest covering tickers × personas. Auto-mature if
        ``end_date <= today``. Returns the new ``backtest_id``."""
        backtest_id = self._insert_backtests_row(
            triggered_by_brief_id=None,
            universe=tickers,
            start_date=start_date,
            end_date=end_date,
        )

        for ticker in tickers:
            for persona_id in personas:
                self._open_forward_test(
                    backtest_id=backtest_id,
                    ticker=ticker,
                    persona_id=persona_id,
                    start_date=start_date,
                    end_date=end_date,
                )

        if end_date <= date.today():
            self._mature_all_open(backtest_id=backtest_id, end_date=end_date)
            self._close_backtest(backtest_id)

        return backtest_id

    # ---------- internals ----------

    def _insert_backtests_row(
        self,
        *,
        triggered_by_brief_id: Optional[str],
        universe: List[str],
        start_date: date,
        end_date: date,
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO backtests "
            "(triggered_by_brief_id, universe, start_date, end_date, status, "
            " report_path, created_ts) VALUES (?, ?, ?, ?, 'open', NULL, ?)",
            (
                triggered_by_brief_id,
                json.dumps(universe),
                start_date.isoformat(),
                end_date.isoformat(),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def _open_forward_test(
        self,
        *,
        backtest_id: int,
        ticker: str,
        persona_id: str,
        start_date: date,
        end_date: date,
    ) -> None:
        # 1. Invoke the graph at start_date — produces runs row + decision.
        run_id, decision = self.graph_runner.run(
            ticker=ticker,
            trade_date=start_date.isoformat(),
            persona_id=persona_id,
            conn=self.conn,
        )

        # 2. Fetch entry price (single-bar window @ start_date).
        try:
            bars = self.price_chain.get_bars(
                ticker, start_date, start_date, self.resolution,
            )
        except Exception as e:
            self._insert_backtest_run_errored(
                backtest_id=backtest_id, ticker=ticker, persona_id=persona_id,
                run_id=run_id, error=f"entry price fetch failed: {e!r}",
            )
            return
        if not bars.bars:
            self._insert_backtest_run_errored(
                backtest_id=backtest_id, ticker=ticker, persona_id=persona_id,
                run_id=run_id, error=f"no bars for entry {start_date}",
            )
            return
        entry_price = bars.bars[0][1]
        price_source = bars.source

        # 3. Fetch entry benchmark price too (best-effort).
        try:
            bench_bars = self.price_chain.get_bars(
                self.benchmark, start_date, start_date, self.resolution,
            )
            benchmark_entry_price = bench_bars.bars[0][1] if bench_bars.bars else None
        except Exception:
            benchmark_entry_price = None

        # 4. Translate decision → position.
        try:
            position = position_from_decision(decision)
        except ValueError:
            position = 0  # unknown → flat (HOLD-ish)

        metrics = {
            "status": "open",
            "run_id": run_id,
            "decision": decision,
            "position": position,
            "entry_date": start_date.isoformat(),
            "entry_price": entry_price,
            "benchmark": self.benchmark,
            "benchmark_entry_price": benchmark_entry_price,
            "scheduled_close_date": end_date.isoformat(),
            "resolution": str(self.resolution.value),
            "price_source": price_source,
        }
        self.conn.execute(
            "INSERT INTO backtest_runs (backtest_id, persona_id, ticker, metrics)"
            " VALUES (?, ?, ?, ?)",
            (backtest_id, persona_id, ticker, json.dumps(metrics)),
        )
        self.conn.commit()

    def _insert_backtest_run_errored(
        self,
        *,
        backtest_id: int,
        ticker: str,
        persona_id: str,
        run_id: Optional[str],
        error: str,
    ) -> None:
        metrics = {
            "status": "errored",
            "run_id": run_id,
            "error": error,
            "errored_sources": [],
        }
        self.conn.execute(
            "INSERT INTO backtest_runs (backtest_id, persona_id, ticker, metrics)"
            " VALUES (?, ?, ?, ?)",
            (backtest_id, persona_id, ticker, json.dumps(metrics)),
        )
        self.conn.commit()

    # Maturation lives in Task 15.
    def _mature_all_open(self, *, backtest_id: int, end_date: date) -> None:
        raise NotImplementedError("Maturation lands in Task 15.")

    def _close_backtest(self, backtest_id: int) -> None:
        self.conn.execute(
            "UPDATE backtests SET status='closed' WHERE backtest_id=?",
            (backtest_id,),
        )
        self.conn.commit()
