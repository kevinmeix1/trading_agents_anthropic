from datetime import datetime
from unittest import TestCase
from zoneinfo import ZoneInfo

from quanthack.backtesting.backtest import BacktestEngine, FillModel
from quanthack.core.config import load_config
from quanthack.market.data_health import validate_market_data
from quanthack.market.market_data import load_price_history, load_quote_history
from quanthack.trading.preflight import run_preflight
from quanthack.reporting.research_report import build_research_report
from quanthack.backtesting.strategy_compare import compare_strategies
from quanthack.backtesting.sweep import run_parameter_sweep
from quanthack.backtesting.walk_forward import run_walk_forward


class ResearchReportTest(TestCase):
    def test_builds_research_report_with_core_sections(self) -> None:
        config = load_config("configs/default.toml")
        prices = load_price_history(config.backtest.price_csv)
        quotes = load_quote_history(config.backtest.quote_csv)
        fill_model = FillModel(slippage_bps=config.backtest.slippage_bps)
        backtest = BacktestEngine(
            strategy=config.build_strategy("simple_momentum"),
            risk_limits=config.risk,
            quality_limits=config.market_quality,
            clock=config.competition.to_clock(),
            fill_model=fill_model,
            periods_per_year=config.backtest.periods_per_year,
        ).run(
            prices=prices,
            quotes=quotes,
            symbol=config.simple_momentum.symbol,
            starting_equity=config.competition.starting_equity,
        )
        comparison = compare_strategies(
            config=config,
            prices=prices,
            quotes=quotes,
            strategy_names=("simple_momentum", "mean_reversion"),
        )
        sweep = run_parameter_sweep(
            prices=prices,
            quotes=quotes,
            symbol=config.simple_momentum.symbol,
            base_config=config.simple_momentum,
            lookbacks=(3,),
            threshold_bps=(4.0,),
            train_fraction=0.6,
            starting_equity=config.competition.starting_equity,
            risk_limits=config.risk,
            quality_limits=config.market_quality,
            clock=config.competition.to_clock(),
            fill_model=fill_model,
            periods_per_year=config.backtest.periods_per_year,
        )

        report = build_research_report(
            config=config,
            preflight=run_preflight(
                config_path="configs/default.toml",
                now=datetime(2026, 6, 17, 12, 0, tzinfo=ZoneInfo("Europe/London")),
            ),
            backtest=backtest,
            comparison=comparison,
            sweep=sweep,
            strategy_name="simple_momentum",
            data_health=validate_market_data(
                prices=prices,
                quotes=quotes,
                symbols=(config.simple_momentum.symbol,),
                max_spread_bps=config.market_quality.max_spread_bps,
            ),
            walk_forward=run_walk_forward(
                config=config,
                prices=prices,
                quotes=quotes,
                strategy_names=("simple_momentum", "mean_reversion"),
                symbol=config.simple_momentum.symbol,
                train_size=10,
                test_size=5,
                step_size=5,
                momentum_lookbacks=(3,),
                momentum_threshold_bps=(4.0,),
            ),
            generated_at=datetime(2026, 6, 17, 14, 0, tzinfo=ZoneInfo("Europe/London")),
        )

        self.assertIn("<!doctype html>", report.html)
        self.assertIn("Decision Snapshot", report.html)
        self.assertIn("Preflight", report.html)
        self.assertIn("Market Data Health", report.html)
        self.assertIn("Backtest", report.html)
        self.assertIn("Strategy Comparison", report.html)
        self.assertIn("Walk-Forward Evaluation", report.html)
        self.assertIn("Momentum Parameter Sweep", report.html)
        self.assertIn("Risk Settings", report.html)
        self.assertIn("simple_momentum", report.html)
        self.assertIn("Realized P&amp;L", report.html)

    def test_escapes_report_title(self) -> None:
        config = load_config("configs/default.toml")
        prices = load_price_history(config.backtest.price_csv)
        quotes = load_quote_history(config.backtest.quote_csv)
        fill_model = FillModel(slippage_bps=config.backtest.slippage_bps)
        backtest = BacktestEngine(
            strategy=config.build_strategy("simple_momentum"),
            risk_limits=config.risk,
            quality_limits=config.market_quality,
            clock=config.competition.to_clock(),
            fill_model=fill_model,
            periods_per_year=config.backtest.periods_per_year,
        ).run(
            prices=prices,
            quotes=quotes,
            symbol=config.simple_momentum.symbol,
            starting_equity=config.competition.starting_equity,
        )
        comparison = compare_strategies(
            config=config,
            prices=prices,
            quotes=quotes,
            strategy_names=("simple_momentum",),
        )
        sweep = run_parameter_sweep(
            prices=prices,
            quotes=quotes,
            symbol=config.simple_momentum.symbol,
            base_config=config.simple_momentum,
            lookbacks=(3,),
            threshold_bps=(4.0,),
            train_fraction=0.6,
            starting_equity=config.competition.starting_equity,
            risk_limits=config.risk,
            quality_limits=config.market_quality,
            clock=config.competition.to_clock(),
            fill_model=fill_model,
            periods_per_year=config.backtest.periods_per_year,
        )

        report = build_research_report(
            config=config,
            preflight=run_preflight(config_path="configs/default.toml"),
            backtest=backtest,
            comparison=comparison,
            sweep=sweep,
            strategy_name="simple_momentum",
            title="<bad>",
        )

        self.assertIn("&lt;bad&gt;", report.html)
        self.assertNotIn("<bad>", report.html)
