from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from zoneinfo import ZoneInfo

from quanthack.trading.preflight import PreflightStatus, run_preflight


class PreflightTest(TestCase):
    def test_default_preflight_is_ready_for_dry_run(self) -> None:
        report = run_preflight(
            config_path="configs/default.toml",
            now=datetime(2026, 6, 17, 12, 0, tzinfo=ZoneInfo("Europe/London")),
        )

        self.assertEqual(report.overall, "READY_FOR_DRY_RUN")
        self.assertTrue(all(check.status == PreflightStatus.OK for check in report.checks))

    def test_stale_quote_requires_attention(self) -> None:
        report = run_preflight(
            config_path="configs/default.toml",
            now=datetime(2026, 6, 17, 12, 0, tzinfo=ZoneInfo("Europe/London")),
            quote_as_of=datetime(2026, 6, 22, 10, 20, 10, tzinfo=ZoneInfo("Europe/London")),
        )

        self.assertEqual(report.overall, "ATTENTION_REQUIRED")
        market_quality = [check for check in report.checks if check.name == "Market quality"][0]
        self.assertEqual(market_quality.status, PreflightStatus.FAIL)
        self.assertIn("stale", market_quality.details)

    def test_aging_quote_warns_before_it_fails(self) -> None:
        report = run_preflight(
            config_path="configs/default.toml",
            now=datetime(2026, 6, 17, 12, 0, tzinfo=ZoneInfo("Europe/London")),
            quote_as_of=datetime(2026, 6, 22, 10, 20, 3, tzinfo=ZoneInfo("Europe/London")),
        )

        self.assertEqual(report.overall, "READY_WITH_WARNINGS")
        market_quality = [check for check in report.checks if check.name == "Market quality"][0]
        self.assertEqual(market_quality.status, PreflightStatus.WARN)
        self.assertIn("quote age", market_quality.details)

    def test_missing_config_requires_attention(self) -> None:
        report = run_preflight(config_path="configs/does_not_exist.toml")

        self.assertEqual(report.overall, "ATTENTION_REQUIRED")
        self.assertEqual(report.checks[-1].name, "Config")
        self.assertEqual(report.checks[-1].status, PreflightStatus.FAIL)

    def test_borderline_risk_limits_warn_without_blocking(self) -> None:
        with TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "borderline.toml"
            text = Path("configs/default.toml").read_text(encoding="utf-8")
            replacements = {
                "max_gross_leverage = 2.0": "max_gross_leverage = 2.5",
                "max_symbol_notional_pct = 0.25": "max_symbol_notional_pct = 0.30",
                "max_daily_loss_pct = 0.025": "max_daily_loss_pct = 0.028",
                "max_drawdown_pct = 0.06": "max_drawdown_pct = 0.07",
                "min_margin_level_pct = 300.0": "min_margin_level_pct = 320.0",
            }
            for old, new in replacements.items():
                text = text.replace(old, new)
            config_path.write_text(text, encoding="utf-8")

            report = run_preflight(
                config_path=config_path,
                now=datetime(2026, 6, 17, 12, 0, tzinfo=ZoneInfo("Europe/London")),
            )

        risk = [check for check in report.checks if check.name == "Risk limits"][0]
        self.assertEqual(report.overall, "READY_WITH_WARNINGS")
        self.assertEqual(risk.status, PreflightStatus.WARN)
        self.assertIn("max_gross_leverage", risk.details)

    def test_aggressive_risk_limits_require_attention(self) -> None:
        with TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "aggressive.toml"
            config_path.write_text(
                """
[competition]
timezone = "Europe/London"
starting_equity = 1000000.0
open_at = "2026-06-21T22:00:00+01:00"
checkpoints = ["2026-06-22T22:00:00+01:00"]
protect_minutes_before = 90.0
protect_minutes_after = 5.0

[risk]
max_gross_leverage = 10.0
max_symbol_notional_pct = 0.75
max_daily_loss_pct = 0.10
max_drawdown_pct = 0.20
checkpoint_risk_multiplier = 0.5
min_margin_level_pct = 100.0

[market_quality]
max_spread_bps = 10.0
max_quote_age_seconds = 5.0

[strategy.simple_momentum]
symbol = "EURUSD"
lookback = 5
threshold_bps = 8.0
target_notional_usd = 50000.0

[market_data]
price_csv = "data/sample_prices.csv"
quote_csv = "data/sample_quotes.csv"

[execution]
route = "dry_run"
journal_path = "outputs/dry_run_journal.jsonl"

[backtest]
price_csv = "data/backtest_prices.csv"
quote_csv = "data/backtest_quotes.csv"
slippage_bps = 1.0
periods_per_year = 252.0
equity_curve_csv = "outputs/backtests/equity_curve.csv"

[sweep]
lookbacks = [3, 5]
threshold_bps = [4.0, 8.0]
train_fraction = 0.6
results_csv = "outputs/backtests/parameter_sweep.csv"
""",
                encoding="utf-8",
            )

            report = run_preflight(
                config_path=config_path,
                now=datetime(2026, 6, 17, 12, 0, tzinfo=ZoneInfo("Europe/London")),
            )

        risk = [check for check in report.checks if check.name == "Risk limits"][0]
        self.assertEqual(report.overall, "ATTENTION_REQUIRED")
        self.assertEqual(risk.status, PreflightStatus.FAIL)
        self.assertIn("max_gross_leverage", risk.details)
