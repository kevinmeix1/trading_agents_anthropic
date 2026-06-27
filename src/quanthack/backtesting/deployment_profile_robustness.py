from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path

from quanthack.backtesting.backtest import FillModel
from quanthack.backtesting.competition_score import (
    CompetitionMetrics,
    RiskDisciplineReport,
    build_competition_metrics,
    build_risk_discipline_report,
    risk_samples_from_portfolio_equity,
)
from quanthack.backtesting.deployment_profile_backtest import (
    LoadedDeploymentProfile,
    load_deployment_profile,
    session_gate_policy_for_profile,
)
from quanthack.backtesting.portfolio_backtest import PortfolioBacktestEngine
from quanthack.core.config import AppConfig
from quanthack.core.instruments import instrument_for
from quanthack.market.market_data import PriceHistory, QuoteHistory


@dataclass(frozen=True)
class DeploymentProfileRobustnessRow:
    scenario_type: str
    scenario: str
    excluded_symbol: str
    slippage_multiplier: float
    slippage_bps: float
    symbols: str
    symbol_count: int
    return_pct: float
    return_delta_pct: float
    max_drawdown_pct: float
    drawdown_delta_pct: float
    sharpe_15m: float
    sharpe_delta: float
    fills: int
    risk_discipline_score: float
    total_pnl_usd: float
    total_pnl_delta_usd: float
    decision: str
    note: str

    @property
    def rank_key(self) -> tuple[float, ...]:
        return (
            _decision_rank(self.decision),
            self.return_pct,
            -self.max_drawdown_pct,
            self.sharpe_15m,
        )


@dataclass(frozen=True)
class DeploymentProfileRobustnessResult:
    profile: LoadedDeploymentProfile
    rows: tuple[DeploymentProfileRobustnessRow, ...]

    @property
    def baseline(self) -> DeploymentProfileRobustnessRow:
        return self.rows[0]

    @property
    def stress_rows(self) -> tuple[DeploymentProfileRobustnessRow, ...]:
        return self.rows[1:]

    @property
    def weakest_row(self) -> DeploymentProfileRobustnessRow | None:
        if not self.stress_rows:
            return None
        return min(self.stress_rows, key=lambda row: row.return_pct)

    @property
    def most_dependent_symbol_row(self) -> DeploymentProfileRobustnessRow | None:
        symbol_rows = tuple(row for row in self.rows if row.scenario_type == "leave_one_symbol")
        if not symbol_rows:
            return None
        return min(symbol_rows, key=lambda row: row.return_delta_pct)


def evaluate_deployment_profile_robustness(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    profile_pack_json: str | Path,
    slot: str,
    slippage_multipliers: tuple[float, ...] = (1.5, 2.0, 3.0),
) -> DeploymentProfileRobustnessResult:
    if any(multiplier <= 0 for multiplier in slippage_multipliers):
        raise ValueError("slippage multipliers must be positive")
    profile = load_deployment_profile(
        profile_pack_json=profile_pack_json,
        slot=slot,
    )
    symbols = tuple(symbol for symbol, _ in profile.strategy_by_symbol)
    baseline_run = _run_profile_scenario(
        config=config,
        prices=prices,
        quotes=quotes,
        profile=profile,
        symbols=symbols,
        slippage_multiplier=1.0,
    )
    baseline_metrics, baseline_risk, baseline_pnl, baseline_fills = baseline_run
    rows = [
        _row(
            scenario_type="baseline",
            scenario="baseline",
            excluded_symbol="",
            symbols=symbols,
            slippage_multiplier=1.0,
            slippage_bps=config.backtest.slippage_bps,
            metrics=baseline_metrics,
            risk=baseline_risk,
            total_pnl_usd=baseline_pnl,
            fills=baseline_fills,
            baseline_metrics=baseline_metrics,
            baseline_pnl_usd=baseline_pnl,
        )
    ]
    for multiplier in tuple(dict.fromkeys(slippage_multipliers)):
        if multiplier == 1.0:
            continue
        metrics, risk, pnl, fills = _run_profile_scenario(
            config=config,
            prices=prices,
            quotes=quotes,
            profile=profile,
            symbols=symbols,
            slippage_multiplier=multiplier,
        )
        rows.append(
            _row(
                scenario_type="cost_stress",
                scenario=f"slippage_{multiplier:g}x",
                excluded_symbol="",
                symbols=symbols,
                slippage_multiplier=multiplier,
                slippage_bps=config.backtest.slippage_bps * multiplier,
                metrics=metrics,
                risk=risk,
                total_pnl_usd=pnl,
                fills=fills,
                baseline_metrics=baseline_metrics,
                baseline_pnl_usd=baseline_pnl,
            )
        )
    for excluded_symbol in symbols:
        kept_symbols = tuple(symbol for symbol in symbols if symbol != excluded_symbol)
        if not kept_symbols:
            continue
        metrics, risk, pnl, fills = _run_profile_scenario(
            config=config,
            prices=prices,
            quotes=quotes,
            profile=profile,
            symbols=kept_symbols,
            slippage_multiplier=1.0,
        )
        rows.append(
            _row(
                scenario_type="leave_one_symbol",
                scenario=f"exclude_{excluded_symbol}",
                excluded_symbol=excluded_symbol,
                symbols=kept_symbols,
                slippage_multiplier=1.0,
                slippage_bps=config.backtest.slippage_bps,
                metrics=metrics,
                risk=risk,
                total_pnl_usd=pnl,
                fills=fills,
                baseline_metrics=baseline_metrics,
                baseline_pnl_usd=baseline_pnl,
            )
        )
    return DeploymentProfileRobustnessResult(profile=profile, rows=tuple(rows))


def write_deployment_profile_robustness_csv(
    result: DeploymentProfileRobustnessResult,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "scenario_type",
                "scenario",
                "excluded_symbol",
                "slippage_multiplier",
                "slippage_bps",
                "symbols",
                "symbol_count",
                "return_pct",
                "return_delta_pct",
                "max_drawdown_pct",
                "drawdown_delta_pct",
                "sharpe_15m",
                "sharpe_delta",
                "fills",
                "risk_discipline_score",
                "total_pnl_usd",
                "total_pnl_delta_usd",
                "decision",
                "note",
            ),
        )
        writer.writeheader()
        for row in result.rows:
            writer.writerow(row.__dict__)


def _run_profile_scenario(
    *,
    config: AppConfig,
    prices: PriceHistory,
    quotes: QuoteHistory,
    profile: LoadedDeploymentProfile,
    symbols: tuple[str, ...],
    slippage_multiplier: float,
) -> tuple[CompetitionMetrics, RiskDisciplineReport, float, int]:
    strategy_map = dict(profile.strategy_by_symbol)
    multiplier_map = dict(profile.multipliers_by_symbol)
    engine = PortfolioBacktestEngine(
        strategies={
            symbol: config.build_strategy(strategy_map[symbol], symbol=symbol)
            for symbol in symbols
        },
        risk_limits=config.risk,
        quality_limits=config.market_quality,
        quality_limits_by_symbol={
            symbol: replace(
                config.market_quality,
                max_spread_bps=instrument_for(symbol).max_spread_bps,
            )
            for symbol in symbols
        },
        clock=config.competition.to_clock(),
        fill_model=FillModel(
            slippage_bps=config.backtest.slippage_bps * slippage_multiplier
        ),
        periods_per_year=config.backtest.periods_per_year,
        target_notional_multipliers_by_symbol={
            symbol: multiplier_map.get(symbol, 1.0) for symbol in symbols
        },
        session_gate_policy=session_gate_policy_for_profile(profile),
    )
    result = engine.run(
        prices=prices,
        quotes=quotes,
        starting_equity=config.competition.starting_equity,
    )
    metrics = build_competition_metrics(
        equity_points=result.equity_curve,
        fills=result.fills,
    )
    risk = build_risk_discipline_report(
        risk_samples_from_portfolio_equity(result.equity_curve)
    )
    return metrics, risk, result.total_pnl_usd, len(result.fills)


def _row(
    *,
    scenario_type: str,
    scenario: str,
    excluded_symbol: str,
    symbols: tuple[str, ...],
    slippage_multiplier: float,
    slippage_bps: float,
    metrics: CompetitionMetrics,
    risk: RiskDisciplineReport,
    total_pnl_usd: float,
    fills: int,
    baseline_metrics: CompetitionMetrics,
    baseline_pnl_usd: float,
) -> DeploymentProfileRobustnessRow:
    return_delta = metrics.return_pct - baseline_metrics.return_pct
    drawdown_delta = metrics.max_drawdown_pct - baseline_metrics.max_drawdown_pct
    sharpe_delta = metrics.sharpe_15m - baseline_metrics.sharpe_15m
    pnl_delta = total_pnl_usd - baseline_pnl_usd
    decision, note = _decision(
        scenario_type=scenario_type,
        excluded_symbol=excluded_symbol,
        return_pct=metrics.return_pct,
        return_delta=return_delta,
        drawdown_delta=drawdown_delta,
        risk_score=risk.score,
        baseline_return=baseline_metrics.return_pct,
    )
    return DeploymentProfileRobustnessRow(
        scenario_type=scenario_type,
        scenario=scenario,
        excluded_symbol=excluded_symbol,
        slippage_multiplier=slippage_multiplier,
        slippage_bps=slippage_bps,
        symbols=";".join(symbols),
        symbol_count=len(symbols),
        return_pct=metrics.return_pct,
        return_delta_pct=return_delta,
        max_drawdown_pct=metrics.max_drawdown_pct,
        drawdown_delta_pct=drawdown_delta,
        sharpe_15m=metrics.sharpe_15m,
        sharpe_delta=sharpe_delta,
        fills=fills,
        risk_discipline_score=risk.score,
        total_pnl_usd=total_pnl_usd,
        total_pnl_delta_usd=pnl_delta,
        decision=decision,
        note=note,
    )


def _decision(
    *,
    scenario_type: str,
    excluded_symbol: str,
    return_pct: float,
    return_delta: float,
    drawdown_delta: float,
    risk_score: float,
    baseline_return: float,
) -> tuple[str, str]:
    if scenario_type == "baseline":
        return "BASELINE", "reference scenario"
    if risk_score < 95:
        return "FAIL", "risk discipline below 95/100"
    if return_pct <= 0:
        return "FAIL", "stress scenario return is not positive"
    if baseline_return > 0 and return_delta <= -0.50 * baseline_return:
        detail = (
            f"excluding {excluded_symbol} removes at least half of baseline return"
            if excluded_symbol
            else "cost stress removes at least half of baseline return"
        )
        return "FRAGILE", detail
    if drawdown_delta > 0.01:
        return "FRAGILE", "drawdown increases by more than 1 percentage point"
    if return_delta < 0:
        return "PASS_WEAKER", "positive under stress but weaker than baseline"
    return "PASS", "stress scenario is positive and not weaker than baseline"


def _decision_rank(decision: str) -> float:
    if decision in {"BASELINE", "PASS"}:
        return 4.0
    if decision == "PASS_WEAKER":
        return 3.0
    if decision == "FRAGILE":
        return 2.0
    return 1.0
