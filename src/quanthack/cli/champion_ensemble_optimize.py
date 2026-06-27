from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.backtesting.champion_ensemble_optimizer import (
    DEFAULT_CHAMPION_ENSEMBLE_PARAMETER_SETS,
    ChampionEnsembleParameterSet,
    optimize_champion_ensemble_parameters,
    write_champion_ensemble_optimization_csv,
)
from quanthack.cli._format import money
from quanthack.core.config import load_config
from quanthack.market.market_data import load_price_history, load_quote_history


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optimize champion ensemble weights with portfolio backtests."
    )
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--symbol", action="append", default=None)
    parser.add_argument("--price-csv", default=None)
    parser.add_argument("--quote-csv", default=None)
    parser.add_argument(
        "--candidate",
        action="append",
        default=None,
        help=(
            "Candidate as label,kalman_weight,asset_squeeze_weight,"
            "dual_squeeze_weight,trend_pullback_weight,entry_score,"
            "strong_lead_score,conflict_penalty[,fixing_reversal_weight[,macd_momentum_weight]]"
        ),
    )
    parser.add_argument("--include-walk-forward", action="store_true")
    parser.add_argument("--train-size", type=int, default=960)
    parser.add_argument("--test-size", type=int, default=192)
    parser.add_argument("--step-size", type=int, default=192)
    parser.add_argument(
        "--output",
        default="outputs/backtests/champion_ensemble_optimization.csv",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    price_csv = args.price_csv or config.backtest.price_csv
    quote_csv = args.quote_csv or config.backtest.quote_csv
    parameter_sets = (
        tuple(_parse_candidate(value) for value in args.candidate)
        if args.candidate
        else DEFAULT_CHAMPION_ENSEMBLE_PARAMETER_SETS
    )
    result = optimize_champion_ensemble_parameters(
        config=config,
        prices=load_price_history(price_csv),
        quotes=load_quote_history(quote_csv),
        symbols=tuple(args.symbol) if args.symbol else None,
        parameter_sets=parameter_sets,
        include_walk_forward=args.include_walk_forward,
        train_size=args.train_size,
        test_size=args.test_size,
        step_size=args.step_size,
    )
    write_champion_ensemble_optimization_csv(result, args.output)

    print("Champion Ensemble Optimization")
    print(f"  Symbols: {', '.join(result.symbols)}")
    print(f"  Price CSV: {price_csv}")
    print(f"  Quote CSV: {quote_csv}")
    print(f"  Walk-forward: {'yes' if args.include_walk_forward else 'no'}")
    print(f"  Output CSV: {args.output}")
    print("Ranked candidates")
    for rank, candidate in enumerate(result.candidates, start=1):
        params = candidate.parameters
        metrics = candidate.comparison_row.competition_metrics
        wf = candidate.walk_forward
        wf_text = (
            ""
            if wf is None
            else (
                f", wf_pos={wf.positive_fold_fraction:.1%}, "
                f"wf_active_pos={wf.active_positive_fold_fraction:.1%}, "
                f"wf_nonneg={wf.non_negative_fold_fraction:.1%}, "
                f"wf_med={wf.median_test_return_pct:.3%}, "
                f"wf_active_med={wf.median_active_test_return_pct:.3%}, "
                f"wf_conc={wf.largest_positive_fold_contribution:.1%}"
            )
        )
        print(
            f"  {rank}. {params.label}: "
            f"k={params.kalman_trend_weight:.2f}, "
            f"a={params.asset_adaptive_dual_squeeze_weight:.2f}, "
            f"d={params.dual_squeeze_weight:.2f}, "
            f"t={params.trend_pullback_weight:.2f}, "
            f"f={params.fixing_reversal_weight:.2f}, "
            f"m={params.macd_momentum_weight:.2f}, "
            f"entry={params.entry_score:.2f}, "
            f"lead={params.strong_lead_score:.2f}, "
            f"proxy={candidate.comparison_row.proxy_score:.1f}, "
            f"return={metrics.return_pct:.3%}, "
            f"drawdown={metrics.max_drawdown_pct:.3%}, "
            f"sharpe15={metrics.sharpe_15m:.3f}, "
            f"final={money(metrics.final_equity)}, "
            f"trades={metrics.trade_count}"
            f"{wf_text}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


def _parse_candidate(raw: str) -> ChampionEnsembleParameterSet:
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) not in {8, 9, 10}:
        raise argparse.ArgumentTypeError(
            "candidate must be label,kalman_weight,asset_squeeze_weight,"
            "dual_squeeze_weight,trend_pullback_weight,entry_score,"
            "strong_lead_score,conflict_penalty[,fixing_reversal_weight[,macd_momentum_weight]]"
        )
    (
        label,
        kalman_weight,
        asset_weight,
        dual_weight,
        pullback_weight,
        entry_score,
        strong_lead_score,
        conflict_penalty,
    ) = parts[:8]
    fixing_reversal_weight = parts[8] if len(parts) >= 9 else "0.0"
    macd_momentum_weight = parts[9] if len(parts) == 10 else "0.0"
    try:
        return ChampionEnsembleParameterSet(
            label=label,
            kalman_trend_weight=float(kalman_weight),
            asset_adaptive_dual_squeeze_weight=float(asset_weight),
            dual_squeeze_weight=float(dual_weight),
            trend_pullback_weight=float(pullback_weight),
            entry_score=float(entry_score),
            strong_lead_score=float(strong_lead_score),
            conflict_penalty=float(conflict_penalty),
            fixing_reversal_weight=float(fixing_reversal_weight),
            macd_momentum_weight=float(macd_momentum_weight),
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
