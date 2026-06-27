from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.backtesting.portfolio_allocator import (
    AllocationPolicy,
    PortfolioAllocator,
    SymbolIntent,
    write_allocation_report_csv,
)


class PortfolioAllocatorTest(TestCase):
    def test_caps_gross_leverage_budget(self) -> None:
        allocator = PortfolioAllocator(
            AllocationPolicy(
                max_gross_leverage=1.0,
                max_symbol_gross_pct=1.0,
                max_forex_gross_pct=1.0,
                max_net_directional_pct=1.0,
                min_active_symbols=1,
            )
        )

        allocation = allocator.allocate(
            (
                SymbolIntent("EURUSD", 800_000),
                SymbolIntent("GBPUSD", -800_000),
            ),
            equity=1_000_000,
        )

        self.assertAlmostEqual(allocation.requested_gross_notional_usd, 1_600_000)
        self.assertAlmostEqual(allocation.adjusted_gross_notional_usd, 1_000_000)
        self.assertIn("gross leverage budget", allocation.trim_reasons[0])

    def test_caps_asset_class_budget(self) -> None:
        allocator = PortfolioAllocator(
            AllocationPolicy(
                max_gross_leverage=2.0,
                max_symbol_gross_pct=1.0,
                max_crypto_gross_pct=0.25,
                max_net_directional_pct=1.0,
                min_active_symbols=1,
            )
        )

        allocation = allocator.allocate(
            (
                SymbolIntent("BTCUSD", 500_000),
                SymbolIntent("ETHUSD", -500_000),
            ),
            equity=1_000_000,
        )

        self.assertAlmostEqual(allocation.adjusted_gross_notional_usd, 500_000)
        self.assertTrue(
            any("crypto asset-class cap" in reason for reason in allocation.trim_reasons)
        )

    def test_caps_single_symbol_budget(self) -> None:
        allocator = PortfolioAllocator(
            AllocationPolicy(
                max_gross_leverage=2.0,
                max_symbol_gross_pct=0.10,
                max_forex_gross_pct=1.0,
                max_net_directional_pct=1.0,
                min_active_symbols=1,
            )
        )

        allocation = allocator.allocate(
            (
                SymbolIntent("EURUSD", 500_000),
                SymbolIntent("GBPUSD", -100_000),
            ),
            equity=1_000_000,
        )

        eurusd = _target(allocation, "EURUSD")
        self.assertAlmostEqual(eurusd.adjusted_abs_notional_usd, 200_000)
        self.assertTrue(any("symbol cap" in reason for reason in eurusd.reasons))

    def test_caps_net_directional_exposure(self) -> None:
        allocator = PortfolioAllocator(
            AllocationPolicy(
                max_gross_leverage=2.0,
                max_symbol_gross_pct=1.0,
                max_forex_gross_pct=1.0,
                max_net_directional_pct=0.80,
                min_active_symbols=1,
            )
        )

        allocation = allocator.allocate(
            (
                SymbolIntent("EURUSD", 100_000),
                SymbolIntent("GBPUSD", -10_000),
            ),
            equity=1_000_000,
        )

        self.assertLessEqual(allocation.net_directional_exposure, 0.80)
        self.assertAlmostEqual(_target(allocation, "EURUSD").adjusted_notional_usd, 90_000)

    def test_scales_when_below_diversification_target(self) -> None:
        allocator = PortfolioAllocator(
            AllocationPolicy(
                max_gross_leverage=2.0,
                max_symbol_gross_pct=1.0,
                max_forex_gross_pct=1.0,
                max_net_directional_pct=1.0,
                min_active_symbols=3,
            )
        )

        allocation = allocator.allocate(
            (
                SymbolIntent("EURUSD", 90_000),
                SymbolIntent("GBPUSD", -90_000),
            ),
            equity=1_000_000,
        )

        self.assertAlmostEqual(allocation.adjusted_gross_notional_usd, 120_000)
        self.assertTrue(
            any("diversification preference" in reason for reason in allocation.trim_reasons)
        )

    def test_diversification_preference_does_not_shave_existing_holds(self) -> None:
        allocator = PortfolioAllocator(
            AllocationPolicy(
                max_gross_leverage=2.0,
                max_symbol_gross_pct=1.0,
                max_forex_gross_pct=1.0,
                max_net_directional_pct=1.0,
                min_active_symbols=3,
            )
        )

        allocation = allocator.allocate(
            (
                SymbolIntent("EURUSD", 90_000, current_notional_usd=90_000),
                SymbolIntent("GBPUSD", -90_000, current_notional_usd=-90_000),
            ),
            equity=1_000_000,
        )

        self.assertAlmostEqual(_target(allocation, "EURUSD").adjusted_notional_usd, 90_000)
        self.assertAlmostEqual(_target(allocation, "GBPUSD").adjusted_notional_usd, -90_000)
        self.assertFalse(
            any("diversification preference" in reason for reason in allocation.trim_reasons)
        )

    def test_blocks_single_symbol_concentration(self) -> None:
        allocator = PortfolioAllocator(
            AllocationPolicy(
                max_gross_leverage=2.0,
                max_symbol_gross_pct=1.0,
                max_forex_gross_pct=1.0,
                max_net_directional_pct=1.0,
                min_active_symbols=3,
            )
        )

        allocation = allocator.allocate(
            (SymbolIntent("EURUSD", 90_000),),
            equity=1_000_000,
        )

        target = _target(allocation, "EURUSD")
        self.assertAlmostEqual(target.adjusted_notional_usd, 0.0)
        self.assertTrue(
            any("single-symbol concentration guard" in reason for reason in target.reasons)
        )

    def test_rebalance_deadband_ignores_small_same_direction_changes(self) -> None:
        allocator = PortfolioAllocator(
            AllocationPolicy(
                max_gross_leverage=2.0,
                max_symbol_gross_pct=1.0,
                max_forex_gross_pct=1.0,
                max_net_directional_pct=1.0,
                min_active_symbols=1,
                min_rebalance_notional_usd=500.0,
                min_rebalance_change_pct=0.02,
            )
        )

        allocation = allocator.allocate(
            (
                SymbolIntent("EURUSD", 50_400, current_notional_usd=50_000),
            ),
            equity=1_000_000,
        )

        target = _target(allocation, "EURUSD")
        self.assertAlmostEqual(target.adjusted_notional_usd, 50_000)
        self.assertAlmostEqual(target.change_notional_usd, 0.0)
        self.assertTrue(any("rebalance deadband" in reason for reason in target.reasons))

    def test_rebalance_deadband_does_not_block_exits(self) -> None:
        allocator = PortfolioAllocator(
            AllocationPolicy(
                max_gross_leverage=2.0,
                max_symbol_gross_pct=1.0,
                max_forex_gross_pct=1.0,
                max_net_directional_pct=1.0,
                min_active_symbols=1,
                min_rebalance_notional_usd=1_000.0,
            )
        )

        allocation = allocator.allocate(
            (
                SymbolIntent("EURUSD", 0.0, current_notional_usd=500.0),
            ),
            equity=1_000_000,
        )

        self.assertAlmostEqual(_target(allocation, "EURUSD").adjusted_notional_usd, 0.0)

    def test_writes_allocation_report_csv(self) -> None:
        allocator = PortfolioAllocator()
        allocation = allocator.allocate(
            (
                SymbolIntent("EURUSD", 50_000, reason="long signal"),
                SymbolIntent("GBPUSD", -50_000, reason="short signal"),
            ),
            equity=1_000_000,
            timestamp="2026-06-22T10:00:00+01:00",
        )

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "allocation.csv"
            write_allocation_report_csv((allocation,), path)
            text = path.read_text(encoding="utf-8")

        self.assertIn("timestamp,requested_gross_notional_usd", text)
        self.assertIn("estimated_risk_status", text)
        self.assertIn("EURUSD", text)


def _target(allocation, symbol: str):
    for target in allocation.targets:
        if target.symbol == symbol:
            return target
    raise AssertionError(f"missing {symbol}")
