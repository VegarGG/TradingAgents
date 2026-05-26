import pytest


@pytest.fixture
def macro_persona():
    from tradingagents.personas.loader import load_persona_from_string
    return load_persona_from_string("""
id: macro
name: Macro
description: top-down
system_prompt_fragment: |
  You think top-down. Stretch your horizon to quarters.
llm: {deep_think_llm: m, quick_think_llm: m}
analysts: {include: [market], exclude: []}
risk_debate: {weights: {aggressive: 0.5, conservative: 1.5, neutral: 1.0}}
memory_scope: hybrid
""")


@pytest.mark.unit
def test_apply_fragment_appends_to_base(macro_persona):
    from tradingagents.personas.prompt_overlay import apply_fragment
    base = "You are a market analyst. Pick 8 indicators."
    result = apply_fragment(base, macro_persona)
    assert base in result
    assert "You think top-down" in result
    # Order: base first, fragment after — analyst-specific instructions stay primary.
    assert result.index("market analyst") < result.index("top-down")


@pytest.mark.unit
def test_apply_fragment_none_is_passthrough():
    from tradingagents.personas.prompt_overlay import apply_fragment
    base = "You are an analyst."
    assert apply_fragment(base, None) == base


@pytest.mark.unit
def test_apply_fragment_strips_trailing_whitespace(macro_persona):
    """No trailing blank lines after concatenation."""
    from tradingagents.personas.prompt_overlay import apply_fragment
    result = apply_fragment("base.\n\n", macro_persona)
    assert not result.endswith("\n\n\n")


@pytest.mark.unit
def test_apply_fragment_empty_fragment_returns_base(macro_persona):
    """A persona with an empty fragment is treated like None."""
    from tradingagents.personas.loader import load_persona_from_string
    p = load_persona_from_string("""
id: blank
name: Blank
description: x
system_prompt_fragment: ""
llm: {deep_think_llm: m, quick_think_llm: m}
analysts: {include: [market], exclude: []}
risk_debate: {weights: {aggressive: 1.0, conservative: 1.0, neutral: 1.0}}
memory_scope: hybrid
""")
    from tradingagents.personas.prompt_overlay import apply_fragment
    base = "You are an analyst."
    assert apply_fragment(base, p) == base
