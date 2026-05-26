import pytest


@pytest.mark.unit
def test_default_config_has_f2_keys():
    from tradingagents.default_config import DEFAULT_CONFIG as C
    assert "backtest_price_sources" in C
    assert C["backtest_price_sources"] == ["yfinance", "polygon", "alpha_vantage", "futu"]
    assert "backtest_resolution_default" in C
    assert C["backtest_resolution_default"] == "1d"
    assert "sweep_interval_seconds" in C
    assert C["sweep_interval_seconds"] == 300
    assert "backtest_max_concurrent_graph_runs" in C
    assert C["backtest_max_concurrent_graph_runs"] == 5
    assert "backtest_strict_historical" in C
    # Auto-on by date check; documented default is None (auto)
    assert C["backtest_strict_historical"] in (None, "auto")
