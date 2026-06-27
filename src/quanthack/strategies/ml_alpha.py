from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from quanthack.market.market_data import PriceHistory
from quanthack.strategies.strategy import (
    AlphaRouterConfig,
    AlphaRouterStrategy,
    BreakoutConfig,
    MeanReversionConfig,
    MomentumConfig,
    SignalDirection,
)


@dataclass(frozen=True)
class MLAlphaPrediction:
    timestamp: str
    symbol: str
    close: float
    next_close: float
    probability_up: float
    score: float
    prediction: SignalDirection
    actual: SignalDirection
    forward_return_bps: float
    signed_return_bps: float
    correct: bool
    sample_count: int
    training_accuracy: float
    training_signed_return_bps: float
    expected_edge_bps: float


@dataclass(frozen=True)
class MLAlphaEvaluation:
    symbol: str
    predictions: tuple[MLAlphaPrediction, ...]
    total_rows: int
    label_threshold_bps: float

    @property
    def actionable_predictions(self) -> tuple[MLAlphaPrediction, ...]:
        return tuple(
            prediction
            for prediction in self.predictions
            if prediction.prediction != SignalDirection.FLAT
        )

    @property
    def coverage(self) -> float:
        if not self.predictions:
            return 0.0
        return len(self.actionable_predictions) / len(self.predictions)

    @property
    def accuracy(self) -> float:
        if not self.predictions:
            return 0.0
        correct = [prediction for prediction in self.predictions if prediction.correct]
        return len(correct) / len(self.predictions)

    @property
    def actionable_accuracy(self) -> float:
        actionable = self.actionable_predictions
        if not actionable:
            return 0.0
        correct = [prediction for prediction in actionable if prediction.correct]
        return len(correct) / len(actionable)

    @property
    def average_forward_return_bps(self) -> float:
        if not self.predictions:
            return 0.0
        return sum(prediction.forward_return_bps for prediction in self.predictions) / len(
            self.predictions
        )

    @property
    def average_signed_return_bps(self) -> float:
        actionable = self.actionable_predictions
        if not actionable:
            return 0.0
        return sum(prediction.signed_return_bps for prediction in actionable) / len(actionable)

    @property
    def cumulative_signed_return_bps(self) -> float:
        return sum(prediction.signed_return_bps for prediction in self.actionable_predictions)

    @property
    def long_count(self) -> int:
        return len(
            [
                prediction
                for prediction in self.predictions
                if prediction.prediction == SignalDirection.LONG
            ]
        )

    @property
    def short_count(self) -> int:
        return len(
            [
                prediction
                for prediction in self.predictions
                if prediction.prediction == SignalDirection.SHORT
            ]
        )

    @property
    def flat_count(self) -> int:
        return len(
            [
                prediction
                for prediction in self.predictions
                if prediction.prediction == SignalDirection.FLAT
            ]
        )


@dataclass(frozen=True)
class MLAlphaPortfolioEvaluation:
    evaluations: tuple[MLAlphaEvaluation, ...]

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(evaluation.symbol for evaluation in self.evaluations)

    @property
    def predictions(self) -> tuple[MLAlphaPrediction, ...]:
        return tuple(
            prediction
            for evaluation in self.evaluations
            for prediction in evaluation.predictions
        )

    @property
    def total_rows(self) -> int:
        return sum(evaluation.total_rows for evaluation in self.evaluations)


@dataclass(frozen=True)
class MLAlphaFilterConfig:
    entry_probability: float
    min_training_accuracy: float
    min_expected_edge_bps: float
    min_samples_for_trade: int
    disable_on_negative_signed_return: bool = True

    @property
    def label(self) -> str:
        negative_gate = "pos_train" if self.disable_on_negative_signed_return else "any_train"
        return (
            f"p{self.entry_probability:.2f}_"
            f"acc{self.min_training_accuracy:.2f}_"
            f"edge{self.min_expected_edge_bps:g}_"
            f"n{self.min_samples_for_trade}_"
            f"{negative_gate}"
        )


@dataclass(frozen=True)
class MLAlphaCalibrationResult:
    filter_config: MLAlphaFilterConfig
    total_predictions: int
    actionable_predictions: int
    correct_actionable_predictions: int
    long_count: int
    short_count: int
    cumulative_signed_return_bps: float

    @property
    def coverage(self) -> float:
        if self.total_predictions == 0:
            return 0.0
        return self.actionable_predictions / self.total_predictions

    @property
    def actionable_accuracy(self) -> float:
        if self.actionable_predictions == 0:
            return 0.0
        return self.correct_actionable_predictions / self.actionable_predictions

    @property
    def average_signed_return_bps(self) -> float:
        if self.actionable_predictions == 0:
            return 0.0
        return self.cumulative_signed_return_bps / self.actionable_predictions


@dataclass(frozen=True)
class MLAlphaSymbolCalibrationResult:
    symbol: str
    result: MLAlphaCalibrationResult


@dataclass(frozen=True)
class MLAlphaCalibrationFold:
    fold_index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    selected_filter: MLAlphaFilterConfig
    train_result: MLAlphaCalibrationResult
    test_result: MLAlphaCalibrationResult


@dataclass(frozen=True)
class MLAlphaWalkForwardCalibration:
    folds: tuple[MLAlphaCalibrationFold, ...]

    @property
    def positive_fold_rate(self) -> float:
        if not self.folds:
            return 0.0
        positive = [
            fold
            for fold in self.folds
            if fold.test_result.average_signed_return_bps > 0
        ]
        return len(positive) / len(self.folds)

    @property
    def average_test_signed_return_bps(self) -> float:
        if not self.folds:
            return 0.0
        return sum(
            fold.test_result.average_signed_return_bps for fold in self.folds
        ) / len(self.folds)

    @property
    def cumulative_test_signed_return_bps(self) -> float:
        return sum(
            fold.test_result.cumulative_signed_return_bps for fold in self.folds
        )


@dataclass(frozen=True)
class MLAlphaPromotionDecision:
    status: str
    reason: str
    positive_fold_rate: float
    average_test_signed_return_bps: float


def evaluate_ml_alpha(
    *,
    prices: PriceHistory,
    symbol: str,
    alpha_router: AlphaRouterConfig,
    momentum: MomentumConfig,
    breakout: BreakoutConfig,
    mean_reversion: MeanReversionConfig,
) -> MLAlphaEvaluation:
    bars = prices.for_symbol(symbol).bars
    if len(bars) < alpha_router.ml_lookback + 2:
        raise ValueError(f"not enough price bars for ML alpha evaluation on {symbol}")

    strategy = AlphaRouterStrategy(
        config=alpha_router,
        momentum=momentum,
        breakout=breakout,
        mean_reversion=mean_reversion,
    )
    closes = [bar.close for bar in bars]
    predictions: list[MLAlphaPrediction] = []
    for index in range(alpha_router.ml_lookback, len(bars) - 1):
        history = closes[: index + 1]
        reading = strategy.read_ml_alpha(history)
        if reading is None:
            continue

        current_close = closes[index]
        next_close = closes[index + 1]
        forward_return_bps = _simple_return_bps(current_close, next_close)
        prediction = _prediction_direction(
            probability_up=reading.probability_up,
            entry_probability=alpha_router.ml_entry_probability,
        )
        actual = _actual_direction(
            forward_return_bps=forward_return_bps,
            threshold_bps=alpha_router.ml_label_threshold_bps,
        )
        signed_return_bps = _signed_return_bps(
            prediction=prediction,
            forward_return_bps=forward_return_bps,
        )
        predictions.append(
            MLAlphaPrediction(
                timestamp=bars[index].timestamp.isoformat(timespec="seconds"),
                symbol=symbol,
                close=current_close,
                next_close=next_close,
                probability_up=reading.probability_up,
                score=reading.score,
                prediction=prediction,
                actual=actual,
                forward_return_bps=forward_return_bps,
                signed_return_bps=signed_return_bps,
                correct=prediction == actual,
                sample_count=reading.sample_count,
                training_accuracy=reading.training_accuracy,
                training_signed_return_bps=reading.training_signed_return_bps,
                expected_edge_bps=reading.expected_edge_bps,
            )
        )

    return MLAlphaEvaluation(
        symbol=symbol,
        predictions=tuple(predictions),
        total_rows=len(bars),
        label_threshold_bps=alpha_router.ml_label_threshold_bps,
    )


def evaluate_ml_alpha_portfolio(
    *,
    prices: PriceHistory,
    symbols: tuple[str, ...] | list[str] | None,
    alpha_router: AlphaRouterConfig,
    momentum: MomentumConfig,
    breakout: BreakoutConfig,
    mean_reversion: MeanReversionConfig,
) -> MLAlphaPortfolioEvaluation:
    selected_symbols = tuple(symbols or prices.symbols())
    if not selected_symbols:
        raise ValueError("no symbols available for ML alpha portfolio evaluation")
    return MLAlphaPortfolioEvaluation(
        tuple(
            evaluate_ml_alpha(
                prices=prices,
                symbol=symbol,
                alpha_router=alpha_router,
                momentum=momentum,
                breakout=breakout,
                mean_reversion=mean_reversion,
            )
            for symbol in selected_symbols
        )
    )


def calibrate_ml_alpha_filters(
    evaluation: MLAlphaPortfolioEvaluation,
    *,
    entry_probabilities: tuple[float, ...] = (0.55, 0.58, 0.60, 0.62),
    min_training_accuracies: tuple[float, ...] = (0.50, 0.55, 0.60),
    min_expected_edges_bps: tuple[float, ...] = (0.0, 2.0, 3.0),
    min_samples_for_trade: tuple[int, ...] = (8, 12, 20),
    disable_on_negative_signed_return: bool = True,
) -> tuple[MLAlphaCalibrationResult, ...]:
    results: list[MLAlphaCalibrationResult] = []
    for entry_probability in entry_probabilities:
        for min_training_accuracy in min_training_accuracies:
            for min_expected_edge_bps in min_expected_edges_bps:
                for min_samples in min_samples_for_trade:
                    filter_config = MLAlphaFilterConfig(
                        entry_probability=entry_probability,
                        min_training_accuracy=min_training_accuracy,
                        min_expected_edge_bps=min_expected_edge_bps,
                        min_samples_for_trade=min_samples,
                        disable_on_negative_signed_return=disable_on_negative_signed_return,
                    )
                    results.append(
                        score_ml_alpha_filter(
                            predictions=evaluation.predictions,
                            filter_config=filter_config,
                        )
                    )
    return tuple(
        sorted(
            results,
            key=lambda result: (
                result.average_signed_return_bps,
                result.actionable_accuracy,
                result.coverage,
                result.cumulative_signed_return_bps,
            ),
            reverse=True,
        )
    )


def walk_forward_calibrate_ml_alpha_filters(
    evaluation: MLAlphaPortfolioEvaluation,
    *,
    train_timestamps: int = 80,
    test_timestamps: int = 20,
    step_timestamps: int = 20,
    entry_probabilities: tuple[float, ...] = (0.55, 0.58, 0.60, 0.62),
    min_training_accuracies: tuple[float, ...] = (0.50, 0.55, 0.60),
    min_expected_edges_bps: tuple[float, ...] = (0.0, 2.0, 3.0),
    min_samples_for_trade: tuple[int, ...] = (8, 12, 20),
    disable_on_negative_signed_return: bool = True,
) -> MLAlphaWalkForwardCalibration:
    if train_timestamps < 1:
        raise ValueError("train_timestamps must be positive")
    if test_timestamps < 1:
        raise ValueError("test_timestamps must be positive")
    if step_timestamps < 1:
        raise ValueError("step_timestamps must be positive")

    unique_times = sorted({_prediction_datetime(prediction) for prediction in evaluation.predictions})
    if len(unique_times) < train_timestamps + test_timestamps:
        raise ValueError("not enough prediction timestamps for ML walk-forward calibration")

    folds: list[MLAlphaCalibrationFold] = []
    fold_index = 1
    start = 0
    while start + train_timestamps + test_timestamps <= len(unique_times):
        train_window = unique_times[start : start + train_timestamps]
        test_window = unique_times[
            start + train_timestamps : start + train_timestamps + test_timestamps
        ]
        train_predictions = _predictions_between(evaluation.predictions, train_window)
        test_predictions = _predictions_between(evaluation.predictions, test_window)
        train_results = calibrate_ml_alpha_filters(
            MLAlphaPortfolioEvaluation(
                (
                    MLAlphaEvaluation(
                        symbol="PORTFOLIO",
                        predictions=train_predictions,
                        total_rows=len(train_predictions),
                        label_threshold_bps=0.0,
                    ),
                )
            ),
            entry_probabilities=entry_probabilities,
            min_training_accuracies=min_training_accuracies,
            min_expected_edges_bps=min_expected_edges_bps,
            min_samples_for_trade=min_samples_for_trade,
            disable_on_negative_signed_return=disable_on_negative_signed_return,
        )
        selected = train_results[0]
        test_result = score_ml_alpha_filter(
            predictions=test_predictions,
            filter_config=selected.filter_config,
        )
        folds.append(
            MLAlphaCalibrationFold(
                fold_index=fold_index,
                train_start=train_window[0].isoformat(timespec="seconds"),
                train_end=train_window[-1].isoformat(timespec="seconds"),
                test_start=test_window[0].isoformat(timespec="seconds"),
                test_end=test_window[-1].isoformat(timespec="seconds"),
                selected_filter=selected.filter_config,
                train_result=selected,
                test_result=test_result,
            )
        )
        fold_index += 1
        start += step_timestamps

    return MLAlphaWalkForwardCalibration(tuple(folds))


def decide_ml_alpha_promotion(
    walk_forward: MLAlphaWalkForwardCalibration,
    *,
    min_folds: int = 3,
    min_positive_fold_rate: float = 0.60,
    min_average_test_signed_return_bps: float = 0.0,
) -> MLAlphaPromotionDecision:
    if len(walk_forward.folds) < min_folds:
        return MLAlphaPromotionDecision(
            status="INSUFFICIENT_EVIDENCE",
            reason=f"needs at least {min_folds} walk-forward folds",
            positive_fold_rate=walk_forward.positive_fold_rate,
            average_test_signed_return_bps=walk_forward.average_test_signed_return_bps,
        )
    if walk_forward.positive_fold_rate < min_positive_fold_rate:
        return MLAlphaPromotionDecision(
            status="REJECT",
            reason=(
                f"positive fold rate {walk_forward.positive_fold_rate:.1%} below "
                f"{min_positive_fold_rate:.1%}"
            ),
            positive_fold_rate=walk_forward.positive_fold_rate,
            average_test_signed_return_bps=walk_forward.average_test_signed_return_bps,
        )
    if walk_forward.average_test_signed_return_bps <= min_average_test_signed_return_bps:
        return MLAlphaPromotionDecision(
            status="REJECT",
            reason=(
                "average out-of-sample signed return "
                f"{walk_forward.average_test_signed_return_bps:.2f} bps is not positive"
            ),
            positive_fold_rate=walk_forward.positive_fold_rate,
            average_test_signed_return_bps=walk_forward.average_test_signed_return_bps,
        )
    return MLAlphaPromotionDecision(
        status="PROMOTE_TO_EXPERIMENT",
        reason="walk-forward filter is positive enough for dry-run experimentation",
        positive_fold_rate=walk_forward.positive_fold_rate,
        average_test_signed_return_bps=walk_forward.average_test_signed_return_bps,
    )


def score_ml_alpha_filter(
    *,
    predictions: tuple[MLAlphaPrediction, ...],
    filter_config: MLAlphaFilterConfig,
) -> MLAlphaCalibrationResult:
    actionable = 0
    correct = 0
    long_count = 0
    short_count = 0
    cumulative_signed_return_bps = 0.0
    for prediction in predictions:
        direction = _prediction_direction(
            probability_up=prediction.probability_up,
            entry_probability=filter_config.entry_probability,
        )
        if direction == SignalDirection.FLAT:
            continue
        if prediction.sample_count < filter_config.min_samples_for_trade:
            continue
        if prediction.training_accuracy < filter_config.min_training_accuracy:
            continue
        if prediction.expected_edge_bps < filter_config.min_expected_edge_bps:
            continue
        if (
            filter_config.disable_on_negative_signed_return
            and prediction.training_signed_return_bps <= 0
        ):
            continue

        actionable += 1
        if direction == SignalDirection.LONG:
            long_count += 1
        else:
            short_count += 1
        if direction == prediction.actual:
            correct += 1
        cumulative_signed_return_bps += _signed_return_bps(
            prediction=direction,
            forward_return_bps=prediction.forward_return_bps,
        )

    return MLAlphaCalibrationResult(
        filter_config=filter_config,
        total_predictions=len(predictions),
        actionable_predictions=actionable,
        correct_actionable_predictions=correct,
        long_count=long_count,
        short_count=short_count,
        cumulative_signed_return_bps=cumulative_signed_return_bps,
    )


def score_ml_alpha_filter_by_symbol(
    evaluation: MLAlphaPortfolioEvaluation,
    *,
    filter_config: MLAlphaFilterConfig,
) -> tuple[MLAlphaSymbolCalibrationResult, ...]:
    rows = tuple(
        MLAlphaSymbolCalibrationResult(
            symbol=symbol_evaluation.symbol,
            result=score_ml_alpha_filter(
                predictions=symbol_evaluation.predictions,
                filter_config=filter_config,
            ),
        )
        for symbol_evaluation in evaluation.evaluations
    )
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                row.result.average_signed_return_bps,
                row.result.actionable_predictions,
            ),
            reverse=True,
        )
    )


def write_ml_alpha_predictions_csv(
    evaluation: MLAlphaEvaluation,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timestamp",
                "symbol",
                "close",
                "next_close",
                "probability_up",
                "score",
                "prediction",
                "actual",
                "forward_return_bps",
                "signed_return_bps",
                "correct",
                "sample_count",
                "training_accuracy",
                "training_signed_return_bps",
                "expected_edge_bps",
            ],
        )
        writer.writeheader()
        for prediction in evaluation.predictions:
            writer.writerow(
                {
                    "timestamp": prediction.timestamp,
                    "symbol": prediction.symbol,
                    "close": prediction.close,
                    "next_close": prediction.next_close,
                    "probability_up": prediction.probability_up,
                    "score": prediction.score,
                    "prediction": prediction.prediction.value,
                    "actual": prediction.actual.value,
                    "forward_return_bps": prediction.forward_return_bps,
                    "signed_return_bps": prediction.signed_return_bps,
                    "correct": prediction.correct,
                    "sample_count": prediction.sample_count,
                    "training_accuracy": prediction.training_accuracy,
                    "training_signed_return_bps": prediction.training_signed_return_bps,
                    "expected_edge_bps": prediction.expected_edge_bps,
                }
            )


def write_ml_alpha_symbol_calibration_csv(
    rows: tuple[MLAlphaSymbolCalibrationResult, ...],
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "symbol",
                "actionable_predictions",
                "coverage",
                "actionable_accuracy",
                "long_count",
                "short_count",
                "average_signed_return_bps",
                "cumulative_signed_return_bps",
            ],
        )
        writer.writeheader()
        for row in rows:
            result = row.result
            writer.writerow(
                {
                    "symbol": row.symbol,
                    "actionable_predictions": result.actionable_predictions,
                    "coverage": result.coverage,
                    "actionable_accuracy": result.actionable_accuracy,
                    "long_count": result.long_count,
                    "short_count": result.short_count,
                    "average_signed_return_bps": result.average_signed_return_bps,
                    "cumulative_signed_return_bps": result.cumulative_signed_return_bps,
                }
            )


def write_ml_alpha_portfolio_predictions_csv(
    evaluation: MLAlphaPortfolioEvaluation,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timestamp",
                "symbol",
                "close",
                "next_close",
                "probability_up",
                "score",
                "prediction",
                "actual",
                "forward_return_bps",
                "signed_return_bps",
                "correct",
                "sample_count",
                "training_accuracy",
                "training_signed_return_bps",
                "expected_edge_bps",
            ],
        )
        writer.writeheader()
        for prediction in evaluation.predictions:
            writer.writerow(
                {
                    "timestamp": prediction.timestamp,
                    "symbol": prediction.symbol,
                    "close": prediction.close,
                    "next_close": prediction.next_close,
                    "probability_up": prediction.probability_up,
                    "score": prediction.score,
                    "prediction": prediction.prediction.value,
                    "actual": prediction.actual.value,
                    "forward_return_bps": prediction.forward_return_bps,
                    "signed_return_bps": prediction.signed_return_bps,
                    "correct": prediction.correct,
                    "sample_count": prediction.sample_count,
                    "training_accuracy": prediction.training_accuracy,
                    "training_signed_return_bps": prediction.training_signed_return_bps,
                    "expected_edge_bps": prediction.expected_edge_bps,
                }
            )


def write_ml_alpha_calibration_csv(
    results: tuple[MLAlphaCalibrationResult, ...],
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "label",
                "entry_probability",
                "min_training_accuracy",
                "min_expected_edge_bps",
                "min_samples_for_trade",
                "disable_on_negative_signed_return",
                "total_predictions",
                "actionable_predictions",
                "coverage",
                "actionable_accuracy",
                "long_count",
                "short_count",
                "average_signed_return_bps",
                "cumulative_signed_return_bps",
            ],
        )
        writer.writeheader()
        for rank, result in enumerate(results, start=1):
            filter_config = result.filter_config
            writer.writerow(
                {
                    "rank": rank,
                    "label": filter_config.label,
                    "entry_probability": filter_config.entry_probability,
                    "min_training_accuracy": filter_config.min_training_accuracy,
                    "min_expected_edge_bps": filter_config.min_expected_edge_bps,
                    "min_samples_for_trade": filter_config.min_samples_for_trade,
                    "disable_on_negative_signed_return": (
                        filter_config.disable_on_negative_signed_return
                    ),
                    "total_predictions": result.total_predictions,
                    "actionable_predictions": result.actionable_predictions,
                    "coverage": result.coverage,
                    "actionable_accuracy": result.actionable_accuracy,
                    "long_count": result.long_count,
                    "short_count": result.short_count,
                    "average_signed_return_bps": result.average_signed_return_bps,
                    "cumulative_signed_return_bps": result.cumulative_signed_return_bps,
                }
            )


def write_ml_alpha_walk_forward_csv(
    result: MLAlphaWalkForwardCalibration,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "fold",
                "train_start",
                "train_end",
                "test_start",
                "test_end",
                "selected_filter",
                "train_actionable",
                "train_coverage",
                "train_accuracy",
                "train_avg_signed_return_bps",
                "test_actionable",
                "test_coverage",
                "test_accuracy",
                "test_avg_signed_return_bps",
                "test_cumulative_signed_return_bps",
            ],
        )
        writer.writeheader()
        for fold in result.folds:
            writer.writerow(
                {
                    "fold": fold.fold_index,
                    "train_start": fold.train_start,
                    "train_end": fold.train_end,
                    "test_start": fold.test_start,
                    "test_end": fold.test_end,
                    "selected_filter": fold.selected_filter.label,
                    "train_actionable": fold.train_result.actionable_predictions,
                    "train_coverage": fold.train_result.coverage,
                    "train_accuracy": fold.train_result.actionable_accuracy,
                    "train_avg_signed_return_bps": (
                        fold.train_result.average_signed_return_bps
                    ),
                    "test_actionable": fold.test_result.actionable_predictions,
                    "test_coverage": fold.test_result.coverage,
                    "test_accuracy": fold.test_result.actionable_accuracy,
                    "test_avg_signed_return_bps": (
                        fold.test_result.average_signed_return_bps
                    ),
                    "test_cumulative_signed_return_bps": (
                        fold.test_result.cumulative_signed_return_bps
                    ),
                }
            )


def _prediction_direction(
    *,
    probability_up: float,
    entry_probability: float,
) -> SignalDirection:
    if probability_up >= entry_probability:
        return SignalDirection.LONG
    if probability_up <= 1.0 - entry_probability:
        return SignalDirection.SHORT
    return SignalDirection.FLAT


def _actual_direction(
    *,
    forward_return_bps: float,
    threshold_bps: float,
) -> SignalDirection:
    if forward_return_bps > threshold_bps:
        return SignalDirection.LONG
    if forward_return_bps < -threshold_bps:
        return SignalDirection.SHORT
    return SignalDirection.FLAT


def _signed_return_bps(
    *,
    prediction: SignalDirection,
    forward_return_bps: float,
) -> float:
    if prediction == SignalDirection.LONG:
        return forward_return_bps
    if prediction == SignalDirection.SHORT:
        return -forward_return_bps
    return 0.0


def _simple_return_bps(current: float, future: float) -> float:
    if current <= 0:
        raise ValueError("current close must be positive")
    return ((future / current) - 1.0) * 10_000


def _prediction_datetime(prediction: MLAlphaPrediction) -> datetime:
    return datetime.fromisoformat(prediction.timestamp)


def _predictions_between(
    predictions: tuple[MLAlphaPrediction, ...],
    timestamps: list[datetime],
) -> tuple[MLAlphaPrediction, ...]:
    timestamp_set = set(timestamps)
    return tuple(
        prediction
        for prediction in predictions
        if _prediction_datetime(prediction) in timestamp_set
    )
