from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.backtesting.adaptive_strategy_selector import (
    AdaptiveStrategyCandidate,
    build_adaptive_strategy_promotion_audit,
    decide_adaptive_strategy_selection_promotion,
    run_adaptive_strategy_selection,
    write_adaptive_strategy_selection_folds_csv,
    write_adaptive_strategy_selection_scores_csv,
    write_adaptive_strategy_selection_summary_csv,
    write_adaptive_strategy_promotion_audit_csv,
    write_adaptive_strategy_stitched_equity_csv,
)
from quanthack.cli._format import money
from quanthack.core.config import load_config
from quanthack.core.instruments import instrument_for
from quanthack.market.market_data import load_price_history, load_quote_history
from quanthack.strategies.strategy import STRATEGY_NAMES, normalize_strategy_name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Walk-forward select the best recent portfolio strategy."
    )
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--strategy", action="append", choices=STRATEGY_NAMES, default=None)
    parser.add_argument(
        "--no-default-strategies",
        action="store_true",
        help="Use only --candidate-map/--recipe-map entries when no --strategy is supplied.",
    )
    parser.add_argument(
        "--candidate-map",
        action="append",
        default=None,
        metavar="LABEL:SYMBOL=STRATEGY,...",
        help=(
            "Add a deployable per-symbol candidate map. The map must cover every "
            "selected symbol. Repeat to compare multiple portfolio recipes."
        ),
    )
    parser.add_argument(
        "--recipe-map",
        action="append",
        default=None,
        metavar="LABEL:SYMBOL=STRATEGY,...",
        help=(
            "Add a deployable candidate recipe that trades only the listed symbols. "
            "Repeat to compare recipes with different symbol universes."
        ),
    )
    parser.add_argument("--symbol", action="append", default=None)
    parser.add_argument("--price-csv", default=None)
    parser.add_argument("--quote-csv", default=None)
    parser.add_argument("--train-size", type=int, default=960)
    parser.add_argument("--test-size", type=int, default=192)
    parser.add_argument("--step-size", type=int, default=192)
    parser.add_argument(
        "--loss-cooldown-folds",
        type=int,
        default=0,
        help=(
            "Temporarily skip a selected strategy for this many future folds after "
            "it produces a negative evaluation fold."
        ),
    )
    parser.add_argument(
        "--min-train-fills",
        type=int,
        default=0,
        help=(
            "Only allow a candidate to be selected when its training window has "
            "at least this many fills. If every candidate fails, the best raw "
            "training score is used as a fallback."
        ),
    )
    parser.add_argument(
        "--min-train-adjusted-return-pct",
        type=float,
        default=None,
        help=(
            "Only allow a candidate when training return minus drawdown penalty "
            "is at least this decimal return value, for example 0.0001 for 0.01%%."
        ),
    )
    parser.add_argument(
        "--train-fill-penalty-pct",
        type=float,
        default=0.0,
        help=(
            "Subtract this decimal return amount per training-window fill when "
            "ranking candidates, for example 0.000002 for a 0.0002%% per-fill "
            "churn penalty."
        ),
    )
    parser.add_argument(
        "--train-stability-splits",
        type=int,
        default=0,
        help=(
            "Split each training window into this many chronological subwindows "
            "and record fold-stability diagnostics. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--prefer-train-stability",
        action="store_true",
        help=(
            "When training-stability splits are enabled, rank candidates by "
            "subwindow active-positive and non-negative consistency before raw "
            "training return."
        ),
    )
    parser.add_argument(
        "--transition-risk-multiplier",
        type=float,
        default=1.0,
        help=(
            "Scale risk budgets for the evaluation fold immediately after the "
            "selector changes strategy. Use 1.0 to disable."
        ),
    )
    parser.add_argument(
        "--allow-cash-fallback",
        action="store_true",
        help=(
            "When every candidate fails the training gates, sit out the next "
            "fold with a flat cash allocation instead of forcing the best "
            "failed candidate."
        ),
    )
    parser.add_argument(
        "--per-symbol-selection",
        action="store_true",
        help=(
            "Add a dynamic candidate that picks the best recent strategy "
            "separately for each symbol before evaluating the next fold."
        ),
    )
    parser.add_argument(
        "--per-symbol-only",
        action="store_true",
        help=(
            "Force the dynamic per-symbol candidate to be selected every fold. "
            "Requires --per-symbol-selection and cannot be combined with maps."
        ),
    )
    parser.add_argument(
        "--summary-output",
        default="outputs/backtests/adaptive_strategy_selection_summary.csv",
    )
    parser.add_argument(
        "--folds-output",
        default="outputs/backtests/adaptive_strategy_selection_folds.csv",
    )
    parser.add_argument(
        "--scores-output",
        default="outputs/backtests/adaptive_strategy_selection_scores.csv",
    )
    parser.add_argument(
        "--stitched-equity-output",
        default="outputs/backtests/adaptive_strategy_selection_stitched_equity.csv",
    )
    parser.add_argument(
        "--promotion-output",
        default="outputs/backtests/adaptive_strategy_selection_promotion.csv",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    price_csv = args.price_csv or config.backtest.price_csv
    quote_csv = args.quote_csv or config.backtest.quote_csv
    if args.strategy:
        strategies = tuple(args.strategy)
    elif args.no_default_strategies:
        strategies = ()
    else:
        strategies = (
            "kalman_trend",
            "champion_ensemble",
            "macd_momentum",
        )
    try:
        candidate_maps = tuple(
            _parse_candidate_map(value) for value in args.candidate_map or ()
        )
        recipe_maps = tuple(
            _parse_candidate_map(value) for value in args.recipe_map or ()
        )
    except argparse.ArgumentTypeError as exc:
        raise SystemExit(str(exc)) from exc
    result = run_adaptive_strategy_selection(
        config=config,
        prices=load_price_history(price_csv),
        quotes=load_quote_history(quote_csv),
        strategy_names=strategies,
        candidate_maps=candidate_maps,
        recipe_maps=recipe_maps,
        symbols=tuple(args.symbol) if args.symbol else None,
        train_size=args.train_size,
        test_size=args.test_size,
        step_size=args.step_size,
        loss_cooldown_folds=args.loss_cooldown_folds,
        min_train_fills=args.min_train_fills,
        min_train_drawdown_adjusted_return_pct=args.min_train_adjusted_return_pct,
        train_fill_penalty_pct=args.train_fill_penalty_pct,
        train_stability_splits=args.train_stability_splits,
        prefer_train_stability=args.prefer_train_stability,
        transition_risk_multiplier=args.transition_risk_multiplier,
        allow_cash_fallback=args.allow_cash_fallback,
        per_symbol_selection=args.per_symbol_selection,
        per_symbol_only=args.per_symbol_only,
    )
    write_adaptive_strategy_selection_summary_csv(result, args.summary_output)
    write_adaptive_strategy_selection_folds_csv(result, args.folds_output)
    write_adaptive_strategy_selection_scores_csv(result, args.scores_output)
    stitched_curve = write_adaptive_strategy_stitched_equity_csv(
        result,
        args.stitched_equity_output,
        starting_equity=config.competition.starting_equity,
    )
    audit = build_adaptive_strategy_promotion_audit(result)
    write_adaptive_strategy_promotion_audit_csv(audit, args.promotion_output)
    decision = audit.decision
    stitched_final_equity = (
        stitched_curve[-1].equity
        if stitched_curve
        else config.competition.starting_equity
    )

    print("Adaptive Strategy Selection")
    print(f"  Symbols: {', '.join(result.symbols)}")
    print(f"  Candidates: {', '.join(result.strategy_names)}")
    print(f"  Price CSV: {price_csv}")
    print(f"  Quote CSV: {quote_csv}")
    print(f"  Folds: {len(result.folds)}")
    print(f"  Loss cooldown folds: {result.loss_cooldown_folds}")
    print(f"  Min train fills: {result.min_train_fills}")
    print(
        "  Min train adjusted return: "
        f"{result.min_train_drawdown_adjusted_return_pct}"
    )
    print(f"  Train fill penalty: {result.train_fill_penalty_pct}")
    print(f"  Train stability splits: {result.train_stability_splits}")
    print(
        "  Prefer train stability: "
        f"{'yes' if result.prefer_train_stability else 'no'}"
    )
    print(f"  Transition risk multiplier: {result.transition_risk_multiplier}")
    print(f"  Cash fallback: {'yes' if result.allow_cash_fallback else 'no'}")
    print(f"  Per-symbol selection: {'yes' if result.per_symbol_selection else 'no'}")
    print(f"  Per-symbol only: {'yes' if result.per_symbol_only else 'no'}")
    print(f"  Positive fold fraction: {result.positive_fold_fraction:.1%}")
    print(f"  Active fold fraction: {result.active_fold_fraction:.1%}")
    print(
        "  Active positive fold fraction: "
        f"{result.active_positive_fold_fraction:.1%}"
    )
    print(f"  Non-negative fold fraction: {result.non_negative_fold_fraction:.1%}")
    print(f"  Compounded OOS return: {result.compounded_test_return_pct:.3%}")
    print(f"  Median active test return: {result.median_active_test_return_pct:.3%}")
    print(f"  Median test Sharpe 15m: {result.median_test_sharpe_15m:.3f}")
    print(f"  Worst test drawdown: {result.worst_test_drawdown_pct:.3%}")
    print(f"  Average risk discipline: {result.average_risk_discipline_score:.1f}/100")
    print(f"  Evaluation fills: {result.total_evaluation_fills}")
    print(f"  Stitched OOS final equity: {money(stitched_final_equity)}")
    print(f"  Promotion: {decision.status} ({decision.reason})")
    print(f"  Summary CSV: {args.summary_output}")
    print(f"  Folds CSV: {args.folds_output}")
    print(f"  Scores CSV: {args.scores_output}")
    print(f"  Stitched OOS equity CSV: {args.stitched_equity_output}")
    print(f"  Promotion audit CSV: {args.promotion_output}")
    print("Selection counts")
    for count in result.selection_counts:
        print(f"  {count.strategy_name}: {count.folds}")
    print("Folds")
    for fold in result.folds:
        metrics = fold.metrics
        score = fold.selected_train_score
        print(
            f"  {fold.fold_index}. {fold.selected_strategy}: "
            f"blocked={','.join(fold.cooldown_blocked_strategies) or 'none'}, "
            f"train_gate_blocked={','.join(fold.train_gate_blocked_strategies) or 'none'}, "
            f"train_ret={score.return_pct:.3%}, "
            f"train_adj={score.drawdown_adjusted_return_pct:.3%}, "
            f"train_stable_active_pos={score.stability.active_positive_fraction:.1%}, "
            f"train_stable_nonneg={score.stability.non_negative_fraction:.1%}, "
            f"risk_mult={fold.evaluation_risk_multiplier:.2f}, "
            f"test_ret={metrics.return_pct:.3%}, "
            f"drawdown={metrics.max_drawdown_pct:.3%}, "
            f"sharpe15={metrics.sharpe_15m:.3f}, "
            f"risk={fold.risk_discipline.score}/100, "
            f"final={money(metrics.final_equity)}, "
            f"fills={len(fold.evaluation.fills)}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


def _parse_candidate_map(raw: str) -> AdaptiveStrategyCandidate:
    if ":" not in raw:
        raise argparse.ArgumentTypeError(
            "candidate map must use LABEL:SYMBOL=STRATEGY,..."
        )
    label, raw_map = raw.split(":", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("candidate map label cannot be empty")
    entries: list[tuple[str, str]] = []
    for raw_entry in raw_map.replace(";", ",").split(","):
        entry = raw_entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise argparse.ArgumentTypeError(
                f"candidate map entry must use SYMBOL=STRATEGY, got {entry!r}"
            )
        raw_symbol, raw_strategy = entry.split("=", 1)
        try:
            symbol = instrument_for(raw_symbol.strip()).symbol
            strategy = normalize_strategy_name(raw_strategy.strip())
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc
        entries.append((symbol, strategy))
    if not entries:
        raise argparse.ArgumentTypeError("candidate map cannot be empty")
    return AdaptiveStrategyCandidate(
        label=label,
        strategy_by_symbol=tuple(entries),
    )
