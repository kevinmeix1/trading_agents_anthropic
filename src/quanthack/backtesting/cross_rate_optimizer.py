from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path

from quanthack.backtesting.signal_diagnostics import (
    SignalDiagnosticRow,
    evaluate_signal_diagnostics,
)
from quanthack.core.config import AppConfig
from quanthack.core.instruments import AssetClass, instrument_for
from quanthack.market.market_data import PriceHistory, QuoteHistory


@dataclass(frozen=True)
class CrossRateParameterSet:
    label: str
    lookback: int
    entry_zscore: float
    exit_zscore: float = 0.25
    min_abs_deviation_bps: float = 0.5
    max_abs_deviation_bps: float = 80.0
    slippage_bps: float = 1.0
    fee_bps: float = 0.0
    cost_buffer: float = 1.0
    max_spread_bps: float | None = None

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("cross-rate parameter label is required")
        if self.lookback < 4:
            raise ValueError("cross-rate lookback must be at least 4")
        if self.entry_zscore <= 0:
            raise ValueError("entry_zscore must be positive")
        if self.exit_zscore < 0 or self.exit_zscore >= self.entry_zscore:
            raise ValueError("exit_zscore must be non-negative and below entry_zscore")
        if self.min_abs_deviation_bps < 0:
            raise ValueError("min_abs_deviation_bps cannot be negative")
        if self.max_abs_deviation_bps <= self.min_abs_deviation_bps:
            raise ValueError(
                "max_abs_deviation_bps must be greater than min_abs_deviation_bps"
            )
        if self.slippage_bps < 0:
            raise ValueError("slippage_bps cannot be negative")
        if self.fee_bps < 0:
            raise ValueError("fee_bps cannot be negative")
        if self.cost_buffer <= 0:
            raise ValueError("cost_buffer must be positive")
        if self.max_spread_bps is not None and self.max_spread_bps <= 0:
            raise ValueError("max_spread_bps must be positive when set")


DEFAULT_CROSS_RATE_PARAMETER_SETS: tuple[CrossRateParameterSet, ...] = (
    CrossRateParameterSet(
        label="baseline_l12_z1_dev1_slip1",
        lookback=12,
        entry_zscore=1.00,
        exit_zscore=0.25,
        min_abs_deviation_bps=1.00,
        slippage_bps=1.00,
        cost_buffer=1.00,
        max_spread_bps=10.0,
    ),
    CrossRateParameterSet(
        label="permissive_l12_z0_75_dev0_5_slip0_5",
        lookback=12,
        entry_zscore=0.75,
        exit_zscore=0.20,
        min_abs_deviation_bps=0.50,
        slippage_bps=0.50,
        cost_buffer=0.75,
        max_spread_bps=14.0,
    ),
    CrossRateParameterSet(
        label="fast_l8_z1_dev0_5_slip0_5",
        lookback=8,
        entry_zscore=1.00,
        exit_zscore=0.25,
        min_abs_deviation_bps=0.50,
        slippage_bps=0.50,
        cost_buffer=0.75,
        max_spread_bps=14.0,
    ),
    CrossRateParameterSet(
        label="slow_l24_z1_dev0_5_slip0_5",
        lookback=24,
        entry_zscore=1.00,
        exit_zscore=0.25,
        min_abs_deviation_bps=0.50,
        slippage_bps=0.50,
        cost_buffer=0.75,
        max_spread_bps=14.0,
    ),
    CrossRateParameterSet(
        label="selective_l12_z1_5_dev1_slip0_5",
        lookback=12,
        entry_zscore=1.50,
        exit_zscore=0.30,
        min_abs_deviation_bps=1.00,
        slippage_bps=0.50,
        cost_buffer=0.75,
        max_spread_bps=14.0,
    ),
    CrossRateParameterSet(
        label="raw_research_l12_z1_dev0_25_slip0",
        lookback=12,
        entry_zscore=1.00,
        exit_zscore=0.25,
        min_abs_deviation_bps=0.25,
        slippage_bps=0.00,
        cost_buffer=0.50,
        max_spread_bps=None,
    ),
)


@dataclass(frozen=True)
class CrossRateOptimizationCandidate:
    parameters: CrossRateParameterSet
    row: SignalDiagnosticRow
    min_active_signals: int
    min_hit_rate: float
    min_average_signed_return_bps: float

    @property
    def eligible(self) -> bool:
        return (
            self.row.active_count >= self.min_active_signals
            and self.row.hit_rate >= self.min_hit_rate
            and self.row.average_signed_forward_return_bps
            >= self.min_average_signed_return_bps
        )

    @property
    def quality_score(self) -> float:
        if self.row.active_count <= 0:
            return 0.0
        sample_factor = self.row.active_count / (
            self.row.active_count + max(self.min_active_signals, 1)
        )
        hit_bonus = max(self.row.hit_rate - 0.50, 0.0) * 2.0
        return (
            self.row.average_signed_forward_return_bps * sample_factor
            + hit_bonus
            + max(self.row.average_edge_after_cost_bps, 0.0) * 0.10
        )

    @property
    def rank_key(self) -> tuple[bool, float, float, float, int]:
        return (
            self.eligible,
            self.quality_score,
            self.row.average_signed_forward_return_bps,
            self.row.hit_rate,
            self.row.active_count,
        )


@dataclass(frozen=True)
class CrossRateOptimizationResult:
    symbols: tuple[str, ...]
    horizon_bars: int
    candidates: tuple[CrossRateOptimizationCandidate, ...]

    @property
    def best(self) -> CrossRateOptimizationCandidate | None:
        if not self.candidates:
            return None
        return self.candidates[0]

    @property
    def recommended_symbols(self) -> tuple[str, ...]:
        symbols: list[str] = []
        for candidate in self.candidates:
            if not candidate.eligible or candidate.row.symbol in symbols:
                continue
            symbols.append(candidate.row.symbol)
        return tuple(symbols)


def optimize_cross_rate_parameters(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    symbols: tuple[str, ...] | None = None,
    parameter_sets: tuple[
        CrossRateParameterSet, ...
    ] = DEFAULT_CROSS_RATE_PARAMETER_SETS,
    horizon_bars: int = 4,
    min_confidence: float = 0.0,
    min_edge_after_cost_bps: float = 0.0,
    min_active_signals: int = 10,
    min_hit_rate: float = 0.50,
    min_average_signed_return_bps: float = 0.0,
) -> CrossRateOptimizationResult:
    if not parameter_sets:
        raise ValueError("cross-rate optimizer needs at least one parameter set")
    if horizon_bars < 1:
        raise ValueError("horizon_bars must be at least 1")
    if min_active_signals < 0:
        raise ValueError("min_active_signals cannot be negative")
    if not 0 <= min_hit_rate <= 1:
        raise ValueError("min_hit_rate must be between 0 and 1")

    selected_symbols = _selected_fx_symbols(prices=prices, quotes=quotes, symbols=symbols)
    candidates: list[CrossRateOptimizationCandidate] = []
    for parameters in parameter_sets:
        candidate_config = _config_with_parameters(config, parameters)
        report = evaluate_signal_diagnostics(
            config=candidate_config,
            prices=prices,
            quotes=quotes,
            strategy_name="cross_rate_reversion",
            symbols=selected_symbols,
            horizon_bars=horizon_bars,
            min_confidence=min_confidence,
            min_edge_after_cost_bps=min_edge_after_cost_bps,
        )
        candidates.extend(
            CrossRateOptimizationCandidate(
                parameters=parameters,
                row=row,
                min_active_signals=min_active_signals,
                min_hit_rate=min_hit_rate,
                min_average_signed_return_bps=min_average_signed_return_bps,
            )
            for row in report.rows
        )

    candidates.sort(key=lambda candidate: candidate.rank_key, reverse=True)
    return CrossRateOptimizationResult(
        symbols=selected_symbols,
        horizon_bars=horizon_bars,
        candidates=tuple(candidates),
    )


def write_cross_rate_optimization_csv(
    result: CrossRateOptimizationResult,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "eligible",
                "quality_score",
                "symbol",
                "label",
                "horizon_bars",
                "lookback",
                "entry_zscore",
                "exit_zscore",
                "min_abs_deviation_bps",
                "max_abs_deviation_bps",
                "slippage_bps",
                "fee_bps",
                "cost_buffer",
                "max_spread_bps",
                "observations",
                "active_count",
                "long_count",
                "short_count",
                "hit_rate",
                "average_signed_forward_return_bps",
                "average_abs_forward_return_bps",
                "average_confidence",
                "average_weight",
                "average_edge_after_cost_bps",
            ],
        )
        writer.writeheader()
        for rank, candidate in enumerate(result.candidates, start=1):
            parameters = candidate.parameters
            row = candidate.row
            writer.writerow(
                {
                    "rank": rank,
                    "eligible": candidate.eligible,
                    "quality_score": candidate.quality_score,
                    "symbol": row.symbol,
                    "label": parameters.label,
                    "horizon_bars": result.horizon_bars,
                    "lookback": parameters.lookback,
                    "entry_zscore": parameters.entry_zscore,
                    "exit_zscore": parameters.exit_zscore,
                    "min_abs_deviation_bps": parameters.min_abs_deviation_bps,
                    "max_abs_deviation_bps": parameters.max_abs_deviation_bps,
                    "slippage_bps": parameters.slippage_bps,
                    "fee_bps": parameters.fee_bps,
                    "cost_buffer": parameters.cost_buffer,
                    "max_spread_bps": (
                        "" if parameters.max_spread_bps is None else parameters.max_spread_bps
                    ),
                    "observations": row.observations,
                    "active_count": row.active_count,
                    "long_count": row.long_count,
                    "short_count": row.short_count,
                    "hit_rate": row.hit_rate,
                    "average_signed_forward_return_bps": (
                        row.average_signed_forward_return_bps
                    ),
                    "average_abs_forward_return_bps": row.average_abs_forward_return_bps,
                    "average_confidence": row.average_confidence,
                    "average_weight": row.average_weight,
                    "average_edge_after_cost_bps": row.average_edge_after_cost_bps,
                }
            )


def _config_with_parameters(
    config: AppConfig,
    parameters: CrossRateParameterSet,
) -> AppConfig:
    cross_rate_reversion = replace(
        config.cross_rate_reversion,
        lookback=parameters.lookback,
        entry_zscore=parameters.entry_zscore,
        exit_zscore=parameters.exit_zscore,
        min_abs_deviation_bps=parameters.min_abs_deviation_bps,
        max_abs_deviation_bps=parameters.max_abs_deviation_bps,
        slippage_bps=parameters.slippage_bps,
        fee_bps=parameters.fee_bps,
        cost_buffer=parameters.cost_buffer,
        max_spread_bps=parameters.max_spread_bps,
    )
    return replace(config, cross_rate_reversion=cross_rate_reversion)


def _selected_fx_symbols(
    *,
    prices: PriceHistory,
    quotes: QuoteHistory,
    symbols: tuple[str, ...] | None,
) -> tuple[str, ...]:
    candidate_symbols = tuple(symbols or sorted(set(prices.symbols()) & set(quotes.symbols())))
    selected: list[str] = []
    for symbol in candidate_symbols:
        instrument = instrument_for(symbol)
        if instrument.asset_class != AssetClass.FOREX:
            continue
        if instrument.symbol not in set(prices.symbols()) & set(quotes.symbols()):
            continue
        selected.append(instrument.symbol)
    if not selected:
        raise ValueError("cross-rate optimizer needs at least one FX symbol with data")
    return tuple(selected)
