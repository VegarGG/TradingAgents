import pytest
from typer.testing import CliRunner


@pytest.mark.unit
def test_forge_command_exists():
    from cli.forge import forge_app
    runner = CliRunner()
    result = runner.invoke(forge_app, ["--help"])
    assert result.exit_code == 0
    assert "backtest" in result.stdout


@pytest.mark.unit
def test_backtest_start_help_lists_required_flags():
    from cli.forge import forge_app
    runner = CliRunner()
    result = runner.invoke(forge_app, ["backtest", "start", "--help"])
    assert result.exit_code == 0
    out = result.stdout
    assert "--watchlist" in out
    assert "--brief-id" in out
    assert "--start-date" in out
