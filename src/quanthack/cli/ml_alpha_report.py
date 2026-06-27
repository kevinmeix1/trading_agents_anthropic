from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace

from quanthack.core.config import load_config
from quanthack.market.market_data import load_price_history
from quanthack.strategies.ml_alpha import (
    calibrate_ml_alpha_filters,
    decide_ml_alpha_promotion,
    evaluate_ml_alpha,
    evaluate_ml_alpha_portfolio,
    score_ml_alpha_filter_by_symbol,
    walk_forward_calibrate_ml_alpha_filters,
    write_ml_alpha_calibration_csv,
    write_ml_alpha_portfolio_predictions_csv,
    write_ml_alpha_predictions_csv,
    write_ml_alpha_symbol_calibration_csv,
    write_ml_alpha_walk_forward_csv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the ML alpha router signal.")
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--price-csv", default=None)
    parser.add_argument("--symbol", default=None)
    parser.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated symbols to evaluate with --all-symbols.",
    )
    parser.add_argument(
        "--all-symbols",
        action="store_true",
        help="Evaluate every symbol in the price CSV instead of one configured symbol.",
    )
    parser.add_argument("--output", default="outputs/backtests/ml_alpha_predictions.csv")
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Sweep ML probability and quality gates on rolling predictions.",
    )
    parser.add_argument(
        "--calibration-output",
        default="outputs/backtests/ml_alpha_calibration.csv",
    )
    parser.add_argument(
        "--symbol-calibration-output",
        default="outputs/backtests/ml_alpha_symbol_calibration.csv",
    )
    parser.add_argument(
        "--walk-forward-calibration-output",
        default="outputs/backtests/ml_alpha_walk_forward_calibration.csv",
    )
    parser.add_argument(
        "--walk-forward-calibrate",
        action="store_true",
        help="Select ML filter gates on earlier predictions and score them later.",
    )
    parser.add_argument("--wf-train-timestamps", type=int, default=80)
    parser.add_argument("--wf-test-timestamps", type=int, default=20)
    parser.add_argument("--wf-step-timestamps", type=int, default=20)
    parser.add_argument("--entry-probabilities", default="0.55,0.58,0.60,0.62")
    parser.add_argument("--min-training-accuracies", default="0.50,0.55,0.60")
    parser.add_argument("--min-expected-edges-bps", default="0,2,3")
    parser.add_argument("--min-samples-for-trade", default="8,12,20")
    parser.add_argument("--ml-lookback", type=int, default=None)
    parser.add_argument("--ml-train-window", type=int, default=None)
    parser.add_argument("--ml-min-train-samples", type=int, default=None)
    parser.add_argument("--ml-epochs", type=int, default=None)
    parser.add_argument("--ml-learning-rate", type=float, default=None)
    parser.add_argument("--ml-l2", type=float, default=None)
    parser.add_argument("--ml-label-threshold-bps", type=float, default=None)
    return parser


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    alpha_router = _alpha_router_with_overrides(config.alpha_router, args)
    price_csv = args.price_csv or config.backtest.price_csv
    prices = load_price_history(price_csv)
    if args.all_symbols:
        symbols = _selected_symbols(args.symbols, tuple(prices.symbols()))
        portfolio_evaluation = evaluate_ml_alpha_portfolio(
            prices=prices,
            symbols=symbols,
            alpha_router=alpha_router,
            momentum=config.simple_momentum,
            breakout=config.breakout,
            mean_reversion=config.mean_reversion,
        )
        write_ml_alpha_portfolio_predictions_csv(portfolio_evaluation, args.output)
        print("ML Alpha Portfolio Evaluation")
        print(f"  Symbols: {', '.join(portfolio_evaluation.symbols)}")
        print(f"  Price CSV: {price_csv}")
        print(f"  Total rows: {portfolio_evaluation.total_rows}")
        print(f"  Scored predictions: {len(portfolio_evaluation.predictions)}")
        _print_portfolio_summary(portfolio_evaluation)
        if args.calibrate:
            calibration = calibrate_ml_alpha_filters(
                portfolio_evaluation,
                entry_probabilities=_parse_float_tuple(args.entry_probabilities),
                min_training_accuracies=_parse_float_tuple(args.min_training_accuracies),
                min_expected_edges_bps=_parse_float_tuple(args.min_expected_edges_bps),
                min_samples_for_trade=_parse_int_tuple(args.min_samples_for_trade),
                disable_on_negative_signed_return=(
                    alpha_router.ml_disable_on_negative_signed_return
                ),
            )
            write_ml_alpha_calibration_csv(calibration, args.calibration_output)
            _print_calibration_summary(calibration)
            symbol_rows = score_ml_alpha_filter_by_symbol(
                portfolio_evaluation,
                filter_config=calibration[0].filter_config,
            )
            write_ml_alpha_symbol_calibration_csv(
                symbol_rows,
                args.symbol_calibration_output,
            )
            _print_symbol_calibration_summary(symbol_rows)
            print(f"  Calibration CSV: {args.calibration_output}")
            print(f"  Symbol calibration CSV: {args.symbol_calibration_output}")
        if args.walk_forward_calibrate:
            walk_forward = walk_forward_calibrate_ml_alpha_filters(
                portfolio_evaluation,
                train_timestamps=args.wf_train_timestamps,
                test_timestamps=args.wf_test_timestamps,
                step_timestamps=args.wf_step_timestamps,
                entry_probabilities=_parse_float_tuple(args.entry_probabilities),
                min_training_accuracies=_parse_float_tuple(args.min_training_accuracies),
                min_expected_edges_bps=_parse_float_tuple(args.min_expected_edges_bps),
                min_samples_for_trade=_parse_int_tuple(args.min_samples_for_trade),
                disable_on_negative_signed_return=(
                    alpha_router.ml_disable_on_negative_signed_return
                ),
            )
            write_ml_alpha_walk_forward_csv(
                walk_forward,
                args.walk_forward_calibration_output,
            )
            _print_walk_forward_summary(walk_forward)
            decision = decide_ml_alpha_promotion(walk_forward)
            print(f"    Promotion decision: {decision.status}")
            print(f"    Decision reason: {decision.reason}")
            print(f"  Walk-forward calibration CSV: {args.walk_forward_calibration_output}")
        print(f"  Predictions CSV: {args.output}")
        return

    symbol = args.symbol or config.alpha_router.symbol
    evaluation = evaluate_ml_alpha(
        prices=prices,
        symbol=symbol,
        alpha_router=alpha_router,
        momentum=config.simple_momentum,
        breakout=config.breakout,
        mean_reversion=config.mean_reversion,
    )
    write_ml_alpha_predictions_csv(evaluation, args.output)

    print("ML Alpha Evaluation")
    print(f"  Symbol: {evaluation.symbol}")
    print(f"  Price CSV: {price_csv}")
    print(f"  Rows: {evaluation.total_rows}")
    print(f"  Scored predictions: {len(evaluation.predictions)}")
    print(f"  Actionable predictions: {len(evaluation.actionable_predictions)}")
    print(f"  Coverage: {evaluation.coverage:.1%}")
    print(f"  Accuracy: {evaluation.accuracy:.1%}")
    print(f"  Actionable accuracy: {evaluation.actionable_accuracy:.1%}")
    print(f"  Long / short / flat: {evaluation.long_count} / {evaluation.short_count} / {evaluation.flat_count}")
    print(f"  Average next return: {evaluation.average_forward_return_bps:.2f} bps")
    print(f"  Average signed return: {evaluation.average_signed_return_bps:.2f} bps")
    print(f"  Cumulative signed return: {evaluation.cumulative_signed_return_bps:.2f} bps")
    print(f"  Predictions CSV: {args.output}")


def _print_portfolio_summary(evaluation) -> None:
    total_actionable = sum(
        len(symbol_evaluation.actionable_predictions)
        for symbol_evaluation in evaluation.evaluations
    )
    total_predictions = len(evaluation.predictions)
    coverage = total_actionable / total_predictions if total_predictions else 0.0
    cumulative_signed = sum(
        symbol_evaluation.cumulative_signed_return_bps
        for symbol_evaluation in evaluation.evaluations
    )
    average_signed = cumulative_signed / total_actionable if total_actionable else 0.0
    print(f"  Actionable predictions: {total_actionable}")
    print(f"  Coverage: {coverage:.1%}")
    print(f"  Average signed return: {average_signed:.2f} bps")
    print(f"  Cumulative signed return: {cumulative_signed:.2f} bps")


def _print_calibration_summary(results) -> None:
    if not results:
        print("  Calibration: no candidates")
        return
    best = results[0]
    print("  Best calibration candidate:")
    print(f"    {best.filter_config.label}")
    print(f"    Actionable: {best.actionable_predictions}/{best.total_predictions}")
    print(f"    Coverage: {best.coverage:.1%}")
    print(f"    Accuracy: {best.actionable_accuracy:.1%}")
    print(f"    Average signed return: {best.average_signed_return_bps:.2f} bps")
    print(f"    Cumulative signed return: {best.cumulative_signed_return_bps:.2f} bps")


def _print_walk_forward_summary(result) -> None:
    print("  Walk-forward calibration:")
    print(f"    Folds: {len(result.folds)}")
    print(f"    Positive fold rate: {result.positive_fold_rate:.1%}")
    print(f"    Average test signed return: {result.average_test_signed_return_bps:.2f} bps")
    print(f"    Cumulative test signed return: {result.cumulative_test_signed_return_bps:.2f} bps")


def _print_symbol_calibration_summary(rows) -> None:
    print("  Top symbols for best calibration:")
    for row in rows[:3]:
        print(
            f"    {row.symbol}: avg_signed={row.result.average_signed_return_bps:.2f} bps, "
            f"coverage={row.result.coverage:.1%}, "
            f"actions={row.result.actionable_predictions}"
        )


def _parse_float_tuple(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("expected at least one float value")
    return values


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("expected at least one integer value")
    return values


def _selected_symbols(raw_symbols: str | None, available_symbols: tuple[str, ...]) -> tuple[str, ...]:
    if raw_symbols is None:
        return available_symbols
    selected = tuple(
        symbol.strip().upper()
        for symbol in raw_symbols.split(",")
        if symbol.strip()
    )
    if not selected:
        raise ValueError("expected at least one symbol")
    missing = tuple(symbol for symbol in selected if symbol not in available_symbols)
    if missing:
        missing_text = ", ".join(missing)
        available_text = ", ".join(available_symbols)
        raise ValueError(
            f"symbols not found in price data: {missing_text}; available: {available_text}"
        )
    return selected


def _alpha_router_with_overrides(alpha_router, args: argparse.Namespace):
    overrides = {
        "ml_lookback": args.ml_lookback,
        "ml_train_window": args.ml_train_window,
        "ml_min_train_samples": args.ml_min_train_samples,
        "ml_epochs": args.ml_epochs,
        "ml_learning_rate": args.ml_learning_rate,
        "ml_l2": args.ml_l2,
        "ml_label_threshold_bps": args.ml_label_threshold_bps,
    }
    clean_overrides = {
        key: value
        for key, value in overrides.items()
        if value is not None
    }
    if not clean_overrides:
        return alpha_router
    return replace(alpha_router, **clean_overrides)


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
