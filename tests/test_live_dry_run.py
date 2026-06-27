from __future__ import annotations

import json
from contextlib import redirect_stdout
from io import StringIO
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.backtesting.portfolio_allocator import AllocationPolicy
from quanthack.backtesting.portfolio_session import SessionGatePolicy
from quanthack.cli import live_dry_run
from quanthack.core.clock import CompetitionMode, UTC
from quanthack.core.config import load_config
from quanthack.market.market_data import PriceBar, QuoteSnapshot
from quanthack.market.sample_data import (
    generate_synthetic_market_data,
    write_price_history_csv,
    write_quote_history_csv,
)
from quanthack.trading.execution import DryRunExecutor, read_journal
from quanthack.trading.live_dry_run import LiveDryRunEngine, LiveDryRunSettings
from quanthack.trading.risk import AccountSnapshot, PortfolioSnapshot, Position


class LiveDryRunTest(TestCase):
    def test_live_dry_run_builds_allocated_dry_run_records(self) -> None:
        with TemporaryDirectory() as tmpdir:
            journal = Path(tmpdir) / "journal.jsonl"
            config = load_config("configs/default.toml")
            settings = LiveDryRunSettings(
                symbols=("eurusd", "USDJPY"),
                strategy_name="simple_momentum",
                bars=5,
                iterations=1,
                journal_path=str(journal),
            )
            engine = LiveDryRunEngine(
                config=config,
                settings=settings,
                market_data=_BalancedMomentumMarket(),
                account_adapter=_StaticAccount(),
                executor=DryRunExecutor(journal),
            )

            result = engine.run()
            saved_records = read_journal(journal)

        self.assertEqual(settings.symbols, ("EURUSD", "USDJPY"))
        self.assertEqual(len(result.iterations), 1)
        self.assertEqual(len(result.records), 2)
        self.assertEqual(len(saved_records), 2)
        self.assertTrue(all(record.status == "DRY_RUN_ACCEPTED" for record in result.records))
        self.assertEqual(result.monitor_report.latest.accepted_trade_count, 2)
        self.assertGreater(result.monitor_report.latest.gross_notional_usd, 0)
        self.assertLess(result.monitor_report.latest.net_directional_exposure, 0.80)

    def test_live_dry_run_settings_validate_loop_controls(self) -> None:
        with self.assertRaisesRegex(ValueError, "bars"):
            LiveDryRunSettings(symbols=("EURUSD",), strategy_name="simple_momentum", bars=1)
        with self.assertRaisesRegex(ValueError, "iterations"):
            LiveDryRunSettings(
                symbols=("EURUSD",),
                strategy_name="simple_momentum",
                iterations=0,
            )

    def test_live_dry_run_settings_validate_strategy_map_symbols(self) -> None:
        with self.assertRaisesRegex(ValueError, "override symbol"):
            LiveDryRunSettings(
                symbols=("EURUSD",),
                strategy_name="simple_momentum",
                strategy_by_symbol=(("USDJPY", "macd_momentum"),),
            )

    def test_live_dry_run_builds_per_symbol_strategy_overrides(self) -> None:
        with TemporaryDirectory() as tmpdir:
            journal = Path(tmpdir) / "journal.jsonl"
            config = load_config("configs/default.toml")
            settings = LiveDryRunSettings(
                symbols=("EURUSD", "USDJPY"),
                strategy_name="simple_momentum",
                strategy_by_symbol=(("USDJPY", "macd_momentum"),),
                journal_path=str(journal),
            )
            engine = LiveDryRunEngine(
                config=config,
                settings=settings,
                market_data=_BalancedMomentumMarket(),
                account_adapter=_StaticAccount(),
                executor=DryRunExecutor(journal),
            )

        self.assertEqual(settings.strategy_name, "simple_momentum")
        self.assertEqual(settings.strategy_by_symbol, (("USDJPY", "macd_momentum"),))
        self.assertEqual(settings.strategy_for_symbol("EURUSD"), "simple_momentum")
        self.assertEqual(settings.strategy_for_symbol("USDJPY"), "macd_momentum")
        self.assertEqual(engine._strategies["EURUSD"].__class__.__name__, "SimpleMomentumStrategy")
        self.assertEqual(engine._strategies["USDJPY"].__class__.__name__, "MacdMomentumStrategy")

    def test_live_dry_run_applies_profile_multipliers_before_allocation(self) -> None:
        with TemporaryDirectory() as tmpdir:
            journal = Path(tmpdir) / "journal.jsonl"
            config = load_config("configs/default.toml")
            settings = LiveDryRunSettings(
                symbols=("EURUSD",),
                strategy_name="simple_momentum",
                target_notional_multipliers_by_symbol=(("EURUSD", 0.5),),
                bars=5,
                journal_path=str(journal),
            )
            engine = LiveDryRunEngine(
                config=config,
                settings=settings,
                market_data=_BalancedMomentumMarket(),
                account_adapter=_StaticAccount(),
                allocation_policy=AllocationPolicy(
                    max_symbol_gross_pct=1.0,
                    max_net_directional_pct=1.0,
                    min_active_symbols=1,
                    apply_diversification_scale=False,
                ),
                executor=DryRunExecutor(journal),
            )

            result = engine.run_once()

        (target,) = result.allocation.targets
        self.assertEqual(settings.target_multiplier_for_symbol("EURUSD"), 0.5)
        self.assertAlmostEqual(target.requested_notional_usd, 25_000.0)
        self.assertAlmostEqual(target.adjusted_notional_usd, 25_000.0)

    def test_live_dry_run_applies_profile_session_gate(self) -> None:
        with TemporaryDirectory() as tmpdir:
            journal = Path(tmpdir) / "journal.jsonl"
            config = load_config("configs/default.toml")
            settings = LiveDryRunSettings(
                symbols=("EURUSD",),
                strategy_name="simple_momentum",
                session_gate_policy=SessionGatePolicy(allowed_utc_hours=(0,)),
                bars=5,
                journal_path=str(journal),
            )
            engine = LiveDryRunEngine(
                config=config,
                settings=settings,
                market_data=_BalancedMomentumMarket(),
                account_adapter=_StaticAccount(),
                allocation_policy=AllocationPolicy(
                    max_symbol_gross_pct=1.0,
                    max_net_directional_pct=1.0,
                    min_active_symbols=1,
                    apply_diversification_scale=False,
                ),
                executor=DryRunExecutor(journal),
            )

            result = engine.run_once()

        (target,) = result.allocation.targets
        self.assertEqual(target.requested_notional_usd, 0.0)
        self.assertEqual(target.primary_signal, "session_gate")
        self.assertEqual(len(result.records), 0)

    def test_live_dry_run_cli_loads_deployment_profile(self) -> None:
        data = generate_synthetic_market_data(
            symbols=("EURUSD",),
            periods=12,
            interval_minutes=1,
            seed=191,
        )
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            profile_path = root / "pack.json"
            price_path = root / "prices.csv"
            quote_path = root / "quotes.csv"
            journal_path = root / "journal.jsonl"
            monitor_path = root / "monitor.csv"
            allocation_path = root / "allocation.csv"
            _write_profile_pack(profile_path)
            write_price_history_csv(data.prices, price_path)
            write_quote_history_csv(data.quotes, quote_path)

            stdout = StringIO()
            with redirect_stdout(stdout):
                live_dry_run.main(
                    [
                        "--config",
                        "configs/default.toml",
                        "--adapter",
                        "csv",
                        "--profile-pack-json",
                        str(profile_path),
                        "--profile-slot",
                        "conservative",
                        "--price-csv",
                        str(price_path),
                        "--quote-csv",
                        str(quote_path),
                        "--journal",
                        str(journal_path),
                        "--monitor-output",
                        str(monitor_path),
                        "--allocation-output",
                        str(allocation_path),
                        "--bars",
                        "5",
                    ]
                )

            output = stdout.getvalue()
            monitor_text = monitor_path.read_text(encoding="utf-8")
            allocation_text = allocation_path.read_text(encoding="utf-8")

            self.assertIn(
                "Deployment profile: conservative (live_test_profile)",
                output,
            )
            self.assertIn("Strategy map: EURUSD=simple_momentum", output)
            self.assertIn("Multipliers: EURUSD=0.500", output)
            self.assertIn(f"Allocation CSV: {allocation_path}", output)
            self.assertIn("timestamp,equity,daily_pnl_pct", monitor_text)
            self.assertIn("requested_gross_notional_usd", allocation_text)

    def test_live_dry_run_routes_allocated_exit_through_risk_engine(self) -> None:
        with TemporaryDirectory() as tmpdir:
            journal = Path(tmpdir) / "journal.jsonl"
            config = load_config("configs/default.toml")
            settings = LiveDryRunSettings(
                symbols=("EURUSD",),
                strategy_name="simple_momentum",
                journal_path=str(journal),
            )
            engine = LiveDryRunEngine(
                config=config,
                settings=settings,
                market_data=_BalancedMomentumMarket(),
                account_adapter=_StaticAccount(),
                executor=DryRunExecutor(journal),
            )

            request, decision = engine._request_and_decision(
                target_symbol="EURUSD",
                current_notional=50_000,
                adjusted_target=0.0,
                account=AccountSnapshot(equity=1_000_000, margin_level_pct=2_000),
                portfolio=PortfolioSnapshot(
                    positions=(Position(symbol="EURUSD", notional_usd=50_000),)
                ),
                mode=CompetitionMode.QUALIFY,
                reason="test allocator exit",
            )

        self.assertEqual(request.reason, "test allocator exit; allocated exit")
        self.assertTrue(decision.approved)
        self.assertEqual(decision.adjusted_notional_usd, 0.0)
        self.assertIn("exit approved", decision.reason)

    def test_live_dry_run_journals_market_quality_holds(self) -> None:
        with TemporaryDirectory() as tmpdir:
            journal = Path(tmpdir) / "journal.jsonl"
            config = load_config("configs/default.toml")
            settings = LiveDryRunSettings(
                symbols=("EURUSD",),
                strategy_name="simple_momentum",
                bars=5,
                journal_path=str(journal),
            )
            engine = LiveDryRunEngine(
                config=config,
                settings=settings,
                market_data=_WideSpreadMarket(),
                account_adapter=_StaticAccount(),
                executor=DryRunExecutor(journal),
            )

            result = engine.run()
            saved_records = read_journal(journal)

        self.assertEqual(len(result.records), 1)
        self.assertEqual(result.records[0].status, "DRY_RUN_BLOCKED")
        self.assertIn("market quality hold", result.records[0].decision.reason)
        self.assertEqual(len(saved_records), 1)
        self.assertEqual(saved_records[0]["status"], "DRY_RUN_BLOCKED")
        self.assertEqual(result.monitor_report.latest.accepted_trade_count, 0)

    def test_live_dry_run_records_quote_failure_and_continues_polling(self) -> None:
        with TemporaryDirectory() as tmpdir:
            journal = Path(tmpdir) / "journal.jsonl"
            config = load_config("configs/default.toml")
            settings = LiveDryRunSettings(
                symbols=("EURUSD", "USDJPY"),
                strategy_name="simple_momentum",
                bars=5,
                iterations=2,
                journal_path=str(journal),
            )
            engine = LiveDryRunEngine(
                config=config,
                settings=settings,
                market_data=_FailOnceQuoteMarket(),
                account_adapter=_StaticAccount(),
                executor=DryRunExecutor(journal),
            )

            result = engine.run()
            saved_records = read_journal(journal)

        self.assertEqual(len(result.iterations), 2)
        self.assertIn("tick outage", result.iterations[0].error or "")
        self.assertIsNone(result.iterations[1].error)
        self.assertEqual(result.iterations[0].records[0].status, "DRY_RUN_BLOCKED")
        self.assertIn(
            "live dry-run polling failure",
            result.iterations[0].records[0].decision.reason,
        )
        self.assertTrue(
            all(record.status == "DRY_RUN_ACCEPTED" for record in result.iterations[1].records)
        )
        self.assertEqual(len(saved_records), 3)
        self.assertEqual(result.monitor_report.latest.accepted_trade_count, 2)

    def test_live_dry_run_records_account_failure_and_continues_polling(self) -> None:
        with TemporaryDirectory() as tmpdir:
            journal = Path(tmpdir) / "journal.jsonl"
            config = load_config("configs/default.toml")
            settings = LiveDryRunSettings(
                symbols=("EURUSD", "USDJPY"),
                strategy_name="simple_momentum",
                bars=5,
                iterations=2,
                journal_path=str(journal),
            )
            engine = LiveDryRunEngine(
                config=config,
                settings=settings,
                market_data=_BalancedMomentumMarket(),
                account_adapter=_FailOnceAccount(),
                executor=DryRunExecutor(journal),
            )

            result = engine.run()
            saved_records = read_journal(journal)

        self.assertEqual(len(result.iterations), 2)
        self.assertIn("account outage", result.iterations[0].error or "")
        self.assertIsNone(result.iterations[1].error)
        self.assertEqual(result.iterations[0].records[0].status, "DRY_RUN_BLOCKED")
        self.assertTrue(
            all(record.status == "DRY_RUN_ACCEPTED" for record in result.iterations[1].records)
        )
        self.assertEqual(len(saved_records), 3)
        self.assertEqual(result.monitor_report.latest.accepted_trade_count, 2)


class _BalancedMomentumMarket:
    def __init__(self) -> None:
        start = datetime(2026, 6, 22, 12, 0, tzinfo=UTC)
        self._bars = {
            "EURUSD": _bars("EURUSD", (1.1000, 1.1020, 1.1040, 1.1060, 1.1080), start),
            "USDJPY": _bars("USDJPY", (155.00, 154.70, 154.40, 154.10, 153.80), start),
        }
        self._quotes = {
            "EURUSD": QuoteSnapshot(
                timestamp=start + timedelta(minutes=4),
                symbol="EURUSD",
                bid=1.10795,
                ask=1.10805,
            ),
            "USDJPY": QuoteSnapshot(
                timestamp=start + timedelta(minutes=4),
                symbol="USDJPY",
                bid=153.799,
                ask=153.801,
            ),
        }

    def supported_symbols(self) -> tuple[str, ...]:
        return ("EURUSD", "USDJPY")

    def get_latest_quote(self, symbol: str) -> QuoteSnapshot:
        return self._quotes[symbol]

    def get_recent_bars(
        self,
        symbol: str,
        *,
        timeframe: str,
        count: int,
    ) -> tuple[PriceBar, ...]:
        return self._bars[symbol][-count:]


class _FailOnceQuoteMarket(_BalancedMomentumMarket):
    def __init__(self) -> None:
        super().__init__()
        self._failed = False

    def get_latest_quote(self, symbol: str) -> QuoteSnapshot:
        if not self._failed:
            self._failed = True
            raise RuntimeError("tick outage")
        return super().get_latest_quote(symbol)


class _WideSpreadMarket:
    def __init__(self) -> None:
        start = datetime(2026, 6, 22, 12, 0, tzinfo=UTC)
        self._bars = {
            "EURUSD": _bars("EURUSD", (1.1000, 1.1020, 1.1040, 1.1060, 1.1080), start),
        }
        self._quotes = {
            "EURUSD": QuoteSnapshot(
                timestamp=start + timedelta(minutes=4),
                symbol="EURUSD",
                bid=1.1000,
                ask=1.1200,
            ),
        }

    def supported_symbols(self) -> tuple[str, ...]:
        return ("EURUSD",)

    def get_latest_quote(self, symbol: str) -> QuoteSnapshot:
        return self._quotes[symbol]

    def get_recent_bars(
        self,
        symbol: str,
        *,
        timeframe: str,
        count: int,
    ) -> tuple[PriceBar, ...]:
        return self._bars[symbol][-count:]


class _StaticAccount:
    def get_account_snapshot(
        self,
        *,
        starting_equity: float,
        day_start_equity: float,
        peak_equity: float,
    ) -> AccountSnapshot:
        return AccountSnapshot(
            equity=1_000_000,
            starting_equity=starting_equity,
            day_start_equity=day_start_equity,
            peak_equity=peak_equity,
            margin_level_pct=2_000,
        )


class _FailOnceAccount(_StaticAccount):
    def __init__(self) -> None:
        self._failed = False

    def get_account_snapshot(
        self,
        *,
        starting_equity: float,
        day_start_equity: float,
        peak_equity: float,
    ) -> AccountSnapshot:
        if not self._failed:
            self._failed = True
            raise RuntimeError("account outage")
        return super().get_account_snapshot(
            starting_equity=starting_equity,
            day_start_equity=day_start_equity,
            peak_equity=peak_equity,
        )


def _bars(symbol: str, closes: tuple[float, ...], start: datetime) -> tuple[PriceBar, ...]:
    return tuple(
        PriceBar(
            timestamp=start + timedelta(minutes=index),
            symbol=symbol,
            close=close,
        )
        for index, close in enumerate(closes)
    )


def _write_profile_pack(path: Path) -> None:
    payload = {
        "recommended_slot": "paper_only",
        "profiles": [
            {
                "slot": "conservative",
                "label": "live_test_profile",
                "evidence_status": "PAPER_ONLY",
                "use_case": "test",
                "reason": "test",
                "return_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "sharpe_15m": 0.0,
                "fold_contribution": 0.0,
                "strategy_map": "EURUSD=simple_momentum",
                "multiplier_map": "EURUSD=0.500",
                "crypto_allowed_utc_hours": "all",
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
