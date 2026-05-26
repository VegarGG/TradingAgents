import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def persona():
    from tradingagents.personas.loader import load_persona_from_string
    return load_persona_from_string("""
id: macro_test
name: Macro Test
description: x
system_prompt_fragment: ""
llm: {deep_think_llm: m, quick_think_llm: m}
analysts: {include: [market], exclude: []}
risk_debate: {weights: {aggressive: 0.5, conservative: 1.5, neutral: 1.0}}
memory_scope: hybrid
""")


def _fake_state():
    return {
        "company_of_interest": "AAPL",
        "investment_plan": "research plan",
        "trader_investment_plan": "trader plan",
        "past_context": "",
        "risk_debate_state": {
            "history": "<should be ignored when persona overrides format>",
            "aggressive_history": "Aggressive: buy.",
            "conservative_history": "Conservative: hold.",
            "neutral_history": "Neutral: monitor.",
            "current_aggressive_response": "",
            "current_conservative_response": "",
            "current_neutral_response": "",
            "latest_speaker": "Neutral",
            "count": 0,
        },
    }


@pytest.mark.unit
def test_pm_uses_weighted_format_when_persona_set(persona):
    from tradingagents.agents.managers.portfolio_manager import (
        create_portfolio_manager,
    )

    captured = {}

    def fake_invoke(structured_llm, llm, prompt, render, role_label):
        captured["prompt"] = prompt
        return "FINAL TRANSACTION PROPOSAL: **HOLD**"

    fake_llm = MagicMock()
    with patch("tradingagents.agents.managers.portfolio_manager.bind_structured",
               return_value=MagicMock()), \
         patch("tradingagents.agents.managers.portfolio_manager.invoke_structured_or_freetext",
               side_effect=fake_invoke):
        node = create_portfolio_manager(fake_llm, persona=persona)
        node(_fake_state())

    prompt = captured["prompt"]
    assert "Aggressive" in prompt and "0.50" in prompt
    assert "Conservative" in prompt and "1.50" in prompt
    assert "Neutral" in prompt and "1.00" in prompt


@pytest.mark.unit
def test_pm_unweighted_when_no_persona():
    from tradingagents.agents.managers.portfolio_manager import (
        create_portfolio_manager,
    )

    captured = {}

    def fake_invoke(structured_llm, llm, prompt, render, role_label):
        captured["prompt"] = prompt
        return "FINAL TRANSACTION PROPOSAL: **HOLD**"

    fake_llm = MagicMock()
    with patch("tradingagents.agents.managers.portfolio_manager.bind_structured",
               return_value=MagicMock()), \
         patch("tradingagents.agents.managers.portfolio_manager.invoke_structured_or_freetext",
               side_effect=fake_invoke):
        node = create_portfolio_manager(fake_llm, persona=None)
        node(_fake_state())

    prompt = captured["prompt"]
    # The "(weight X.XX)" annotation is unique to the persona-weighted format.
    # The substring "weight" appears legitimately in "Overweight"/"Underweight",
    # so we match the parenthesised form instead.
    assert "(weight " not in prompt
    # The three sides still appear because format_weighted_risk_debate writes
    # them regardless of persona; just without weight annotations.
    assert "Aggressive" in prompt
    assert "Conservative" in prompt
    assert "Neutral" in prompt
