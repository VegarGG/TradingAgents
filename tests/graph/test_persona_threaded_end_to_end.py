"""End-to-end: setting config['persona_id'] reaches every factory invocation."""

import pytest
from unittest.mock import patch, MagicMock


@pytest.mark.unit
def test_graph_setup_accepts_persona():
    from tradingagents.graph.setup import GraphSetup
    sig = __import__("inspect").signature(GraphSetup.__init__)
    assert "persona" in sig.parameters


@pytest.mark.unit
def test_trading_agents_graph_loads_persona_from_config(tmp_path, monkeypatch):
    """When config['persona_id']='macro', TradingAgentsGraph loads macro.yaml
    and forwards a Persona to GraphSetup."""
    from tradingagents.default_config import DEFAULT_CONFIG
    config = dict(DEFAULT_CONFIG)
    config["iic_db_path"] = str(tmp_path / "iic.db")
    config["iic_data_dir"] = str(tmp_path / "data")
    config["persona_id"] = "macro"
    # Avoid heavy LLM construction.
    with patch("tradingagents.graph.trading_graph.create_llm_client",
               return_value=MagicMock(get_llm=MagicMock(return_value=MagicMock()))), \
         patch("tradingagents.graph.trading_graph.GraphSetup") as mock_setup:
        mock_setup.return_value.setup_graph.return_value = MagicMock()
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        TradingAgentsGraph(selected_analysts=["market"], config=config)

    # GraphSetup must have been called with persona=<a Persona whose id=='macro'>
    call = mock_setup.call_args
    persona = call.kwargs.get("persona")
    assert persona is not None
    assert persona.id == "macro"


@pytest.mark.unit
def test_graph_setup_threads_persona_into_factory_calls():
    """GraphSetup.setup_graph must pass persona into the analyst & PM factories."""
    from tradingagents.personas.loader import load_persona_from_string
    p = load_persona_from_string("""
id: macro_test
name: Macro Test
description: x
system_prompt_fragment: "X"
llm: {deep_think_llm: m, quick_think_llm: m}
analysts: {include: [market], exclude: []}
risk_debate: {weights: {aggressive: 0.5, conservative: 1.5, neutral: 1.0}}
memory_scope: hybrid
""")

    from tradingagents.graph.setup import GraphSetup
    from tradingagents.graph.conditional_logic import ConditionalLogic

    captured = {}
    real_market = __import__("tradingagents.agents.analysts.market_analyst",
                              fromlist=["create_market_analyst"]).create_market_analyst
    real_pm = __import__("tradingagents.agents.managers.portfolio_manager",
                          fromlist=["create_portfolio_manager"]).create_portfolio_manager

    def spy_market(llm, persona=None):
        captured["market_persona"] = persona
        return real_market(MagicMock(), persona=None)  # avoid touching real LLMs

    def spy_pm(llm, persona=None):
        captured["pm_persona"] = persona
        return real_pm(MagicMock(), persona=None)

    with patch("tradingagents.graph.setup.create_market_analyst", side_effect=spy_market), \
         patch("tradingagents.graph.setup.create_portfolio_manager", side_effect=spy_pm):
        gs = GraphSetup(
            quick_thinking_llm=MagicMock(),
            deep_thinking_llm=MagicMock(),
            tool_nodes={"market": MagicMock()},
            conditional_logic=ConditionalLogic(max_debate_rounds=1, max_risk_discuss_rounds=1),
            persona=p,
        )
        gs.setup_graph(selected_analysts=["market"])

    assert captured["market_persona"] is p
    assert captured["pm_persona"] is p
