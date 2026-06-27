from unittest import TestCase

from quanthack.trading.position_risk import (
    PositionCostBasis,
    evaluate_position_stop,
)


class PositionCostBasisTest(TestCase):
    def test_tracks_add_partial_close_and_full_exit(self) -> None:
        basis = PositionCostBasis()
        basis = basis.apply_fill(trade_units=100, fill_price=1.00)
        basis = basis.apply_fill(trade_units=100, fill_price=1.10)

        self.assertEqual(basis.position_units, 200)
        self.assertAlmostEqual(basis.average_entry_price or 0.0, 1.05)

        basis = basis.apply_fill(trade_units=-50, fill_price=1.20)

        self.assertEqual(basis.position_units, 150)
        self.assertAlmostEqual(basis.average_entry_price or 0.0, 1.05)

        basis = basis.apply_fill(trade_units=-150, fill_price=1.15)

        self.assertEqual(basis.position_units, 0.0)
        self.assertIsNone(basis.average_entry_price)

    def test_reversal_uses_fill_price_as_new_entry(self) -> None:
        basis = PositionCostBasis().apply_fill(trade_units=100, fill_price=1.00)

        basis = basis.apply_fill(trade_units=-150, fill_price=0.98)

        self.assertEqual(basis.position_units, -50)
        self.assertEqual(basis.average_entry_price, 0.98)

    def test_position_stop_triggers_on_entry_notional_loss(self) -> None:
        basis = PositionCostBasis().apply_fill(trade_units=100_000, fill_price=1.00)

        decision = evaluate_position_stop(
            symbol="EURUSD",
            cost_basis=basis,
            mark_price=0.99,
            max_position_loss_pct=0.005,
        )

        self.assertTrue(decision.triggered)
        self.assertAlmostEqual(decision.loss_pct, 0.01)
        self.assertIn("position stop-loss", decision.reason)

    def test_position_stop_does_not_trigger_when_disabled(self) -> None:
        basis = PositionCostBasis().apply_fill(trade_units=100_000, fill_price=1.00)

        decision = evaluate_position_stop(
            symbol="EURUSD",
            cost_basis=basis,
            mark_price=0.50,
            max_position_loss_pct=0.0,
        )

        self.assertFalse(decision.triggered)
