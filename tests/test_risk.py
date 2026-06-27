from unittest import TestCase

from quanthack.core.clock import CompetitionMode
from quanthack.trading.risk import (
    AccountSnapshot,
    PortfolioSnapshot,
    Position,
    RiskEngine,
    RiskLimits,
    RiskState,
    Side,
    TradeRequest,
)


def make_request(target_notional_usd: float = 50_000) -> TradeRequest:
    return TradeRequest(
        symbol="EURUSD",
        side=Side.BUY,
        target_notional_usd=target_notional_usd,
        reason="test request",
    )


class AccountSnapshotTest(TestCase):
    def test_account_calculates_pnl_and_drawdown(self) -> None:
        account = AccountSnapshot(
            equity=980_000,
            starting_equity=1_000_000,
            day_start_equity=1_000_000,
            peak_equity=1_050_000,
        )

        self.assertAlmostEqual(account.total_pnl_pct, -0.02)
        self.assertAlmostEqual(account.daily_pnl_pct, -0.02)
        self.assertAlmostEqual(account.drawdown_pct, 1 - (980_000 / 1_050_000))


class RiskEngineTest(TestCase):
    def test_normal_request_is_approved(self) -> None:
        engine = RiskEngine()

        decision = engine.evaluate(
            account=AccountSnapshot(equity=1_000_000, margin_level_pct=2_000),
            portfolio=PortfolioSnapshot(),
            request=make_request(),
            mode=CompetitionMode.QUALIFY,
        )

        self.assertTrue(decision.approved)
        self.assertEqual(decision.adjusted_notional_usd, 50_000)
        self.assertEqual(decision.state, RiskState.NORMAL)

    def test_large_symbol_request_is_capped(self) -> None:
        engine = RiskEngine(RiskLimits(max_symbol_notional_pct=0.25))

        decision = engine.evaluate(
            account=AccountSnapshot(equity=1_000_000, margin_level_pct=2_000),
            portfolio=PortfolioSnapshot(),
            request=make_request(900_000),
            mode=CompetitionMode.QUALIFY,
        )

        self.assertTrue(decision.approved)
        self.assertEqual(decision.adjusted_notional_usd, 250_000)

    def test_checkpoint_protect_reduces_winning_account_size(self) -> None:
        engine = RiskEngine(RiskLimits(checkpoint_risk_multiplier=0.5))

        decision = engine.evaluate(
            account=AccountSnapshot(
                equity=1_020_000,
                starting_equity=1_000_000,
                margin_level_pct=2_000,
            ),
            portfolio=PortfolioSnapshot(),
            request=make_request(100_000),
            mode=CompetitionMode.CHECKPOINT_PROTECT,
        )

        self.assertTrue(decision.approved)
        self.assertEqual(decision.adjusted_notional_usd, 50_000)

    def test_daily_loss_blocks_and_freezes(self) -> None:
        engine = RiskEngine(RiskLimits(max_daily_loss_pct=0.025))

        decision = engine.evaluate(
            account=AccountSnapshot(
                equity=974_000,
                day_start_equity=1_000_000,
                margin_level_pct=2_000,
            ),
            portfolio=PortfolioSnapshot(),
            request=make_request(),
            mode=CompetitionMode.QUALIFY,
        )

        self.assertFalse(decision.approved)
        self.assertEqual(decision.state, RiskState.FROZEN)
        self.assertIn("daily loss", decision.reason)

    def test_drawdown_blocks_and_moves_to_reduce_only(self) -> None:
        engine = RiskEngine(RiskLimits(max_drawdown_pct=0.06))

        decision = engine.evaluate(
            account=AccountSnapshot(
                equity=930_000,
                day_start_equity=930_000,
                peak_equity=1_000_000,
                margin_level_pct=2_000,
            ),
            portfolio=PortfolioSnapshot(),
            request=make_request(),
            mode=CompetitionMode.QUALIFY,
        )

        self.assertFalse(decision.approved)
        self.assertEqual(decision.state, RiskState.REDUCE_ONLY)
        self.assertIn("drawdown", decision.reason)

    def test_margin_warning_blocks_and_freezes(self) -> None:
        engine = RiskEngine(RiskLimits(min_margin_level_pct=300))

        decision = engine.evaluate(
            account=AccountSnapshot(equity=1_000_000, margin_level_pct=250),
            portfolio=PortfolioSnapshot(),
            request=make_request(),
            mode=CompetitionMode.QUALIFY,
        )

        self.assertFalse(decision.approved)
        self.assertEqual(decision.state, RiskState.FROZEN)
        self.assertIn("margin", decision.reason)

    def test_exit_is_approved_even_when_loss_stop_is_active(self) -> None:
        engine = RiskEngine(RiskLimits(max_daily_loss_pct=0.025))

        decision = engine.evaluate_exit(
            account=AccountSnapshot(
                equity=974_000,
                day_start_equity=1_000_000,
                margin_level_pct=2_000,
            ),
            portfolio=PortfolioSnapshot(
                positions=(Position(symbol="EURUSD", notional_usd=50_000),)
            ),
            symbol="EURUSD",
            mode=CompetitionMode.QUALIFY,
        )

        self.assertTrue(decision.approved)
        self.assertEqual(decision.adjusted_notional_usd, 0.0)
        self.assertEqual(decision.state, RiskState.FROZEN)
        self.assertIn("daily loss", decision.reason)
        self.assertIn("exit approved", decision.reason)

    def test_reduce_only_margin_band_blocks_without_freezing(self) -> None:
        engine = RiskEngine(
            RiskLimits(
                min_margin_level_pct=300,
                reduce_only_margin_level_pct=500,
            )
        )

        decision = engine.evaluate(
            account=AccountSnapshot(equity=1_000_000, margin_level_pct=400),
            portfolio=PortfolioSnapshot(),
            request=make_request(),
            mode=CompetitionMode.QUALIFY,
        )

        self.assertFalse(decision.approved)
        self.assertEqual(decision.state, RiskState.REDUCE_ONLY)
        self.assertIn("reduce-only", decision.reason)

    def test_daily_loss_can_reduce_only_instead_of_freeze(self) -> None:
        engine = RiskEngine(
            RiskLimits(max_daily_loss_pct=0.025, freeze_on_daily_loss=False)
        )

        decision = engine.evaluate(
            account=AccountSnapshot(
                equity=974_000,
                day_start_equity=1_000_000,
                margin_level_pct=2_000,
            ),
            portfolio=PortfolioSnapshot(),
            request=make_request(),
            mode=CompetitionMode.QUALIFY,
        )

        self.assertFalse(decision.approved)
        self.assertEqual(decision.state, RiskState.REDUCE_ONLY)

    def test_drawdown_brake_scales_size_before_hard_throttle(self) -> None:
        engine = RiskEngine(
            RiskLimits(
                max_drawdown_pct=0.20,
                drawdown_derisk_start_pct=0.04,
                drawdown_derisk_full_pct=0.10,
            )
        )

        decision = engine.evaluate(
            account=AccountSnapshot(
                equity=930_000,
                day_start_equity=930_000,
                peak_equity=1_000_000,
                margin_level_pct=2_000,
            ),
            portfolio=PortfolioSnapshot(),
            request=make_request(50_000),
            mode=CompetitionMode.QUALIFY,
        )

        self.assertTrue(decision.approved)
        self.assertAlmostEqual(decision.adjusted_notional_usd, 25_000, delta=200)

    def test_exit_remains_approved_in_reduce_only_margin_band(self) -> None:
        engine = RiskEngine(
            RiskLimits(
                min_margin_level_pct=300,
                reduce_only_margin_level_pct=500,
            )
        )

        decision = engine.evaluate_exit(
            account=AccountSnapshot(equity=1_000_000, margin_level_pct=400),
            portfolio=PortfolioSnapshot(
                positions=(Position(symbol="EURUSD", notional_usd=50_000),)
            ),
            symbol="EURUSD",
            mode=CompetitionMode.QUALIFY,
        )

        self.assertTrue(decision.approved)
        self.assertEqual(decision.state, RiskState.REDUCE_ONLY)
        self.assertIn("reduce-only", decision.reason)

    def test_competition_safe_preset_stays_below_official_danger_zones(self) -> None:
        limits = RiskLimits.competition_safe()

        self.assertLess(limits.max_gross_leverage, 28.0)
        self.assertLessEqual(limits.max_symbol_notional_pct, 0.90)
        self.assertGreater(limits.min_margin_level_pct, 30.0)
        self.assertEqual(limits.max_position_loss_for_symbol("EURUSD"), 0.01)
        self.assertEqual(limits.max_position_loss_for_symbol("XAGUSD"), 0.02)
        self.assertEqual(limits.max_position_loss_for_symbol("BTCUSD"), 0.025)
        self.assertGreater(
            limits.reduce_only_margin_level_pct or 0.0,
            limits.min_margin_level_pct,
        )

    def test_asset_class_position_loss_overrides_fall_back_to_global(self) -> None:
        limits = RiskLimits(
            max_position_loss_pct=0.01,
            max_forex_position_loss_pct=0.005,
            max_metal_position_loss_pct=0.02,
            max_crypto_position_loss_pct=0.0,
        )

        self.assertEqual(limits.max_position_loss_for_symbol("EURUSD"), 0.005)
        self.assertEqual(limits.max_position_loss_for_symbol("XAUUSD"), 0.02)
        self.assertEqual(limits.max_position_loss_for_symbol("ETHUSD"), 0.0)
        self.assertEqual(limits.max_position_loss_for_symbol("UNKNOWN"), 0.01)

    def test_reduce_only_floor_must_exceed_freeze_floor(self) -> None:
        with self.assertRaisesRegex(ValueError, "reduce_only_margin_level_pct"):
            RiskLimits(min_margin_level_pct=300, reduce_only_margin_level_pct=200)

    def test_drawdown_brake_thresholds_must_be_set_together(self) -> None:
        with self.assertRaisesRegex(ValueError, "set together"):
            RiskLimits(drawdown_derisk_start_pct=0.04)

    def test_asset_class_position_loss_overrides_must_be_non_negative(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_metal_position_loss_pct"):
            RiskLimits(max_metal_position_loss_pct=-0.01)
