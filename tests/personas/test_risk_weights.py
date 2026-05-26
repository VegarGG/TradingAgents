import pytest


@pytest.fixture
def macro_persona():
    from tradingagents.personas.loader import load_persona_from_string
    return load_persona_from_string("""
id: macro
name: Macro
description: top-down
system_prompt_fragment: "x"
llm: {deep_think_llm: m, quick_think_llm: m}
analysts: {include: [market], exclude: []}
risk_debate: {weights: {aggressive: 0.5, conservative: 1.5, neutral: 1.0}}
memory_scope: hybrid
""")


@pytest.fixture
def debate_state():
    return {
        "aggressive_history": "Aggressive view: rate hikes will spike growth tickers.",
        "conservative_history": "Conservative view: rate path uncertain; trim risk.",
        "neutral_history":      "Neutral view: data is mixed, hold positions.",
    }


@pytest.mark.unit
def test_format_includes_each_side_and_weights(macro_persona, debate_state):
    from tradingagents.personas.risk_weights import format_weighted_risk_debate
    out = format_weighted_risk_debate(debate_state, macro_persona)
    assert "Aggressive" in out and "0.5" in out
    assert "Conservative" in out and "1.5" in out
    assert "Neutral" in out and "1.0" in out
    assert "rate hikes" in out and "trim risk" in out and "Hold positions".lower() in out.lower()


@pytest.mark.unit
def test_format_none_persona_omits_weights(debate_state):
    """No persona → no weight annotations; sections still present."""
    from tradingagents.personas.risk_weights import format_weighted_risk_debate
    out = format_weighted_risk_debate(debate_state, None)
    assert "weight" not in out.lower()
    assert "Aggressive" in out
    assert "Conservative" in out
    assert "Neutral" in out


@pytest.mark.unit
def test_format_missing_history_keys_are_safe(macro_persona):
    """A risk_debate_state missing some keys should not raise."""
    from tradingagents.personas.risk_weights import format_weighted_risk_debate
    out = format_weighted_risk_debate({"aggressive_history": "x"}, macro_persona)
    assert "Aggressive" in out
    # Missing sides render as empty (or "(no entries)") — either is acceptable;
    # the assertion is just that no exception was raised.
