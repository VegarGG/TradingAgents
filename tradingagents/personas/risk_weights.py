"""Persona-weighted risk-debate formatter (ADR-NEW-2 / spec §7).

The Portfolio Manager consumes the risk-debate history as a formatted
string. When a persona is active, each side is labelled with its weight
so the PM naturally emphasises higher-weighted views.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from tradingagents.personas.loader import Persona


def _section(label: str, body: str, weight: Optional[float]) -> str:
    body = (body or "").strip() or "(no entries)"
    if weight is None:
        return f"### {label}\n{body}"
    return f"### {label} (weight {weight:.2f})\n{body}"


def format_weighted_risk_debate(
    state: Mapping[str, Any],
    persona: Optional[Persona],
) -> str:
    """Render the three risk-debate sides as a single string.

    When ``persona`` is set, prefixes each side's header with its weight
    from ``persona.risk_debate.weights``. When ``None``, omits the weight.
    """
    aggr = state.get("aggressive_history", "")
    cons = state.get("conservative_history", "")
    neut = state.get("neutral_history", "")

    w = persona.risk_debate.weights if persona is not None else {}
    return (
        _section("Aggressive",     aggr, w.get("aggressive")) + "\n\n" +
        _section("Conservative",   cons, w.get("conservative")) + "\n\n" +
        _section("Neutral",        neut, w.get("neutral"))
    )
