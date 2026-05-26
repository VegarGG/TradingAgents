import pytest
from datetime import datetime, date


@pytest.mark.unit
def test_resolution_enum_values():
    from tradingagents.backtest.prices import Resolution
    assert Resolution.DAILY.value == "1d"
    assert Resolution.ONE_MIN.value == "1m"


@pytest.mark.unit
def test_bars_dataclass_round_trip():
    from tradingagents.backtest.prices import Bars, Resolution
    b = Bars(
        ticker="AAPL",
        resolution=Resolution.DAILY,
        bars=[(datetime(2026, 4, 26), 213.45), (datetime(2026, 4, 27), 214.10)],
        source="yfinance",
    )
    assert b.ticker == "AAPL"
    assert b.resolution is Resolution.DAILY
    assert len(b.bars) == 2
    assert b.bars[0][1] == pytest.approx(213.45)
    assert b.source == "yfinance"


@pytest.mark.unit
def test_bars_is_frozen():
    from tradingagents.backtest.prices import Bars, Resolution
    b = Bars(ticker="A", resolution=Resolution.DAILY, bars=[], source="x")
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        b.ticker = "B"  # type: ignore


@pytest.mark.unit
def test_price_data_unavailable_is_exception():
    from tradingagents.backtest.prices import PriceDataUnavailable
    assert issubclass(PriceDataUnavailable, Exception)
    err = PriceDataUnavailable("AAPL", date(2026, 4, 26), date(2026, 5, 26),
                                tried_sources=["yfinance", "polygon"])
    assert "AAPL" in str(err)
    assert err.tried_sources == ["yfinance", "polygon"]


@pytest.mark.unit
def test_price_source_protocol_has_required_attrs():
    """A class with name/supports/get_bars should satisfy the Protocol."""
    from tradingagents.backtest.prices import PriceSource, Resolution, Bars

    class FakeSource:
        name = "fake"
        supports = {Resolution.DAILY}
        def get_bars(self, ticker, start, end, resolution):
            return Bars(ticker=ticker, resolution=resolution, bars=[], source=self.name)

    # Protocol is runtime-checkable in F2 — this should pass without raising.
    src: PriceSource = FakeSource()
    assert src.name == "fake"
    assert src.get_bars("AAPL", date(2026, 4, 26), date(2026, 5, 26),
                        Resolution.DAILY).source == "fake"
