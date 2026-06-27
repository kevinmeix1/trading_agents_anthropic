from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from quanthack.backtesting.backtest import BacktestEngine, BacktestResult, FillModel
from quanthack.core.clock import CompetitionClock
from quanthack.market.market_data import PriceHistory, QuoteHistory
from quanthack.market.market_quality import MarketQualityLimits
from quanthack.trading.risk import RiskLimits
from quanthack.strategies.strategy import MomentumConfig, SimpleMomentumStrategy


@dataclass(frozen=True)
class SweepCandidate:
    lookback: int
    threshold_bps: float
    train: BacktestResult
    test: BacktestResult

    @property
    def rank_key(self) -> tuple[float, float, float]:
        return (
            self.test.metrics.sharpe_ratio,
            self.test.metrics.total_return_pct,
            -self.test.metrics.max_drawdown_pct,
        )


@dataclass(frozen=True)
class SweepResult:
    candidates: tuple[SweepCandidate, ...]

    @property
    def best(self) -> SweepCandidate | None:
        if not self.candidates:
            return None
        return self.candidates[0]


def run_parameter_sweep(
    *,
    prices: PriceHistory,
    quotes: QuoteHistory,
    symbol: str,
    base_config: MomentumConfig,
    lookbacks: tuple[int, ...],
    threshold_bps: tuple[float, ...],
    train_fraction: float,
    starting_equity: float,
    risk_limits: RiskLimits,
    quality_limits: MarketQualityLimits,
    clock: CompetitionClock,
    fill_model: FillModel,
    periods_per_year: float,
) -> SweepResult:
    train_prices, test_prices = split_price_history(
        prices=prices,
        symbol=symbol,
        train_fraction=train_fraction,
    )
    train_quotes, test_quotes = split_quote_history(
        quotes=quotes,
        symbol=symbol,
        train_count=len(train_prices.bars),
    )

    candidates: list[SweepCandidate] = []
    for lookback in lookbacks:
        for threshold in threshold_bps:
            if len(train_prices.bars) < lookback or len(test_prices.bars) < lookback:
                continue

            strategy_config = MomentumConfig(
                symbol=symbol,
                lookback=lookback,
                threshold_bps=threshold,
                target_notional_usd=base_config.target_notional_usd,
            )
            train = _run_one(
                prices=train_prices,
                quotes=train_quotes,
                symbol=symbol,
                strategy_config=strategy_config,
                starting_equity=starting_equity,
                risk_limits=risk_limits,
                quality_limits=quality_limits,
                clock=clock,
                fill_model=fill_model,
                periods_per_year=periods_per_year,
            )
            test = _run_one(
                prices=test_prices,
                quotes=test_quotes,
                symbol=symbol,
                strategy_config=strategy_config,
                starting_equity=starting_equity,
                risk_limits=risk_limits,
                quality_limits=quality_limits,
                clock=clock,
                fill_model=fill_model,
                periods_per_year=periods_per_year,
            )
            candidates.append(
                SweepCandidate(
                    lookback=lookback,
                    threshold_bps=threshold,
                    train=train,
                    test=test,
                )
            )

    candidates.sort(key=lambda candidate: candidate.rank_key, reverse=True)
    return SweepResult(tuple(candidates))


def split_price_history(
    *,
    prices: PriceHistory,
    symbol: str,
    train_fraction: float,
) -> tuple[PriceHistory, PriceHistory]:
    bars = prices.for_symbol(symbol).bars
    split_index = _split_index(len(bars), train_fraction)
    return PriceHistory(bars[:split_index]), PriceHistory(bars[split_index:])


def split_quote_history(
    *,
    quotes: QuoteHistory,
    symbol: str,
    train_count: int,
) -> tuple[QuoteHistory, QuoteHistory]:
    symbol_quotes = quotes.for_symbol(symbol).quotes
    return QuoteHistory(symbol_quotes[:train_count]), QuoteHistory(symbol_quotes[train_count:])


def write_sweep_csv(result: SweepResult, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "lookback",
                "threshold_bps",
                "train_return_pct",
                "train_sharpe",
                "train_max_drawdown_pct",
                "train_fills",
                "test_return_pct",
                "test_sharpe",
                "test_max_drawdown_pct",
                "test_fills",
                "test_turnover_notional",
            ],
        )
        writer.writeheader()
        for index, candidate in enumerate(result.candidates, start=1):
            writer.writerow(
                {
                    "rank": index,
                    "lookback": candidate.lookback,
                    "threshold_bps": candidate.threshold_bps,
                    "train_return_pct": candidate.train.metrics.total_return_pct,
                    "train_sharpe": candidate.train.metrics.sharpe_ratio,
                    "train_max_drawdown_pct": candidate.train.metrics.max_drawdown_pct,
                    "train_fills": len(candidate.train.fills),
                    "test_return_pct": candidate.test.metrics.total_return_pct,
                    "test_sharpe": candidate.test.metrics.sharpe_ratio,
                    "test_max_drawdown_pct": candidate.test.metrics.max_drawdown_pct,
                    "test_fills": len(candidate.test.fills),
                    "test_turnover_notional": candidate.test.metrics.turnover_notional,
                }
            )


def _run_one(
    *,
    prices: PriceHistory,
    quotes: QuoteHistory,
    symbol: str,
    strategy_config: MomentumConfig,
    starting_equity: float,
    risk_limits: RiskLimits,
    quality_limits: MarketQualityLimits,
    clock: CompetitionClock,
    fill_model: FillModel,
    periods_per_year: float,
) -> BacktestResult:
    engine = BacktestEngine(
        strategy=SimpleMomentumStrategy(strategy_config),
        risk_limits=risk_limits,
        quality_limits=quality_limits,
        clock=clock,
        fill_model=fill_model,
        periods_per_year=periods_per_year,
    )
    return engine.run(
        prices=prices,
        quotes=quotes,
        symbol=symbol,
        starting_equity=starting_equity,
    )


def _split_index(length: int, train_fraction: float) -> int:
    if length < 2:
        raise ValueError("need at least two bars to split train/test")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")

    split_index = int(length * train_fraction)
    return min(max(split_index, 1), length - 1)

