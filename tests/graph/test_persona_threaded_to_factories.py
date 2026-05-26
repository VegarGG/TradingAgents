"""Boundary test for D4: when a persona is active, every factory accepts
the persona kwarg and constructs without raising."""

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def persona():
    from tradingagents.personas.loader import load_persona_from_string
    return load_persona_from_string("""
id: macro_test
name: Macro Test
description: x
system_prompt_fragment: |
  PERSONA-FRAGMENT-SENTINEL-XYZ123
llm: {deep_think_llm: m, quick_think_llm: m}
analysts: {include: [market], exclude: []}
risk_debate: {weights: {aggressive: 0.5, conservative: 1.5, neutral: 1.0}}
memory_scope: hybrid
""")


_FACTORIES = [
    ("tradingagents.agents.analysts.market_analyst",       "create_market_analyst"),
    ("tradingagents.agents.analysts.news_analyst",         "create_news_analyst"),
    ("tradingagents.agents.analysts.sentiment_analyst",    "create_sentiment_analyst"),
    ("tradingagents.agents.analysts.sentiment_analyst",    "create_social_media_analyst"),
    ("tradingagents.agents.analysts.fundamentals_analyst", "create_fundamentals_analyst"),
    ("tradingagents.agents.analysts.derivative_analyst",   "create_derivative_analyst"),
    ("tradingagents.agents.researchers.bull_researcher",   "create_bull_researcher"),
    ("tradingagents.agents.researchers.bear_researcher",   "create_bear_researcher"),
    ("tradingagents.agents.managers.research_manager",     "create_research_manager"),
    ("tradingagents.agents.trader.trader",                 "create_trader"),
    ("tradingagents.agents.risk_mgmt.aggressive_debator",  "create_aggressive_debator"),
    ("tradingagents.agents.risk_mgmt.conservative_debator","create_conservative_debator"),
    ("tradingagents.agents.risk_mgmt.neutral_debator",     "create_neutral_debator"),
]


@pytest.mark.unit
@pytest.mark.parametrize("module_name,factory_name", _FACTORIES)
def test_factory_accepts_persona_kwarg(module_name, factory_name, persona):
    """All factories must accept ``persona=...`` without raising."""
    import importlib
    import inspect
    mod = importlib.import_module(module_name)
    factory = getattr(mod, factory_name)
    sig = inspect.signature(factory)
    assert "persona" in sig.parameters, (
        f"{factory_name} must accept a `persona` kwarg"
    )
    # Calling with persona=None should match the pre-F2 behaviour (no error).
    node = factory(MagicMock(), persona=None)
    assert callable(node)
    # Calling with the persona should also not raise at construction.
    node = factory(MagicMock(), persona=persona)
    assert callable(node)


@pytest.mark.unit
@pytest.mark.parametrize("module_name,factory_name", _FACTORIES)
def test_factory_imports_apply_fragment(module_name, factory_name):
    """Every factory must import apply_fragment into its namespace so the
    fragment can be applied. (The actual fragment-in-prompt assertion lives
    in the PM-specific test for Task 11 and in the Task 24 integration run.)"""
    import importlib
    mod = importlib.import_module(module_name)
    assert hasattr(mod, "apply_fragment"), (
        f"{module_name} must import apply_fragment from "
        "tradingagents.personas.prompt_overlay"
    )
