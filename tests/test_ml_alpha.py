from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from zoneinfo import ZoneInfo

from quanthack.market.market_data import PriceBar, PriceHistory
from quanthack.strategies.ml_alpha import (
    MLAlphaFilterConfig,
    MLAlphaCalibrationFold,
    MLAlphaCalibrationResult,
    MLAlphaWalkForwardCalibration,
    calibrate_ml_alpha_filters,
    decide_ml_alpha_promotion,
    evaluate_ml_alpha,
    evaluate_ml_alpha_portfolio,
    score_ml_alpha_filter,
    score_ml_alpha_filter_by_symbol,
    walk_forward_calibrate_ml_alpha_filters,
    write_ml_alpha_calibration_csv,
    write_ml_alpha_portfolio_predictions_csv,
    write_ml_alpha_predictions_csv,
    write_ml_alpha_symbol_calibration_csv,
    write_ml_alpha_walk_forward_csv,
)
from quanthack.strategies.strategy import (
    AlphaRouterConfig,
    BreakoutConfig,
    MeanReversionConfig,
    MomentumConfig,
    SignalDirection,
)


LONDON = ZoneInfo("Europe/London")


class MLAlphaEvaluationTest(TestCase):
    def test_rising_history_produces_long_actionable_predictions(self) -> None:
        evaluation = evaluate_ml_alpha(
            prices=_prices([1.0000 + index * 0.0010 for index in range(30)]),
            symbol="EURUSD",
            alpha_router=_config(),
            momentum=MomentumConfig(symbol="EURUSD"),
            breakout=BreakoutConfig(symbol="EURUSD"),
            mean_reversion=MeanReversionConfig(symbol="EURUSD"),
        )

        self.assertGreater(len(evaluation.predictions), 0)
        self.assertGreater(len(evaluation.actionable_predictions), 0)
        self.assertGreater(evaluation.actionable_accuracy, 0.5)
        self.assertGreater(evaluation.average_signed_return_bps, 0.0)
        self.assertTrue(
            any(
                prediction.prediction == SignalDirection.LONG
                for prediction in evaluation.actionable_predictions
            )
        )

    def test_falling_history_produces_short_actionable_predictions(self) -> None:
        evaluation = evaluate_ml_alpha(
            prices=_prices([1.0000 - index * 0.0010 for index in range(30)]),
            symbol="EURUSD",
            alpha_router=_config(),
            momentum=MomentumConfig(symbol="EURUSD"),
            breakout=BreakoutConfig(symbol="EURUSD"),
            mean_reversion=MeanReversionConfig(symbol="EURUSD"),
        )

        self.assertGreater(len(evaluation.actionable_predictions), 0)
        self.assertTrue(
            any(
                prediction.prediction == SignalDirection.SHORT
                for prediction in evaluation.actionable_predictions
            )
        )
        self.assertGreater(evaluation.average_signed_return_bps, 0.0)

    def test_prediction_csv_is_written(self) -> None:
        evaluation = evaluate_ml_alpha(
            prices=_prices([1.0000 + index * 0.0010 for index in range(30)]),
            symbol="EURUSD",
            alpha_router=_config(),
            momentum=MomentumConfig(symbol="EURUSD"),
            breakout=BreakoutConfig(symbol="EURUSD"),
            mean_reversion=MeanReversionConfig(symbol="EURUSD"),
        )

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ml_predictions.csv"
            write_ml_alpha_predictions_csv(evaluation, path)
            text = path.read_text(encoding="utf-8")

        self.assertIn("timestamp,symbol,close,next_close,probability_up", text)
        self.assertIn("training_accuracy", text)
        self.assertIn("training_signed_return_bps", text)
        self.assertIn("EURUSD", text)

    def test_too_little_data_fails_loudly(self) -> None:
        with self.assertRaisesRegex(ValueError, "not enough price bars"):
            evaluate_ml_alpha(
                prices=_prices([1.0000, 1.0010, 1.0020]),
                symbol="EURUSD",
                alpha_router=_config(),
                momentum=MomentumConfig(symbol="EURUSD"),
                breakout=BreakoutConfig(symbol="EURUSD"),
                mean_reversion=MeanReversionConfig(symbol="EURUSD"),
            )

    def test_portfolio_evaluation_scores_each_symbol(self) -> None:
        evaluation = evaluate_ml_alpha_portfolio(
            prices=_multi_symbol_prices(
                {
                    "EURUSD": [1.0000 + index * 0.0010 for index in range(30)],
                    "GBPUSD": [1.3000 - index * 0.0010 for index in range(30)],
                }
            ),
            symbols=None,
            alpha_router=_config(),
            momentum=MomentumConfig(symbol="EURUSD"),
            breakout=BreakoutConfig(symbol="EURUSD"),
            mean_reversion=MeanReversionConfig(symbol="EURUSD"),
        )

        self.assertEqual(evaluation.symbols, ("EURUSD", "GBPUSD"))
        self.assertEqual(len(evaluation.evaluations), 2)
        self.assertGreater(len(evaluation.predictions), 0)

    def test_calibration_scores_probability_and_quality_gates(self) -> None:
        evaluation = evaluate_ml_alpha_portfolio(
            prices=_multi_symbol_prices(
                {
                    "EURUSD": [1.0000 + index * 0.0010 for index in range(30)],
                    "GBPUSD": [1.3000 - index * 0.0010 for index in range(30)],
                }
            ),
            symbols=("EURUSD", "GBPUSD"),
            alpha_router=_config(),
            momentum=MomentumConfig(symbol="EURUSD"),
            breakout=BreakoutConfig(symbol="EURUSD"),
            mean_reversion=MeanReversionConfig(symbol="EURUSD"),
        )
        result = score_ml_alpha_filter(
            predictions=evaluation.predictions,
            filter_config=MLAlphaFilterConfig(
                entry_probability=0.55,
                min_training_accuracy=0.50,
                min_expected_edge_bps=0.0,
                min_samples_for_trade=1,
            ),
        )

        self.assertGreater(result.actionable_predictions, 0)
        self.assertGreater(result.average_signed_return_bps, 0.0)
        self.assertEqual(result.actionable_predictions, result.long_count + result.short_count)

    def test_calibration_csv_is_written(self) -> None:
        evaluation = evaluate_ml_alpha_portfolio(
            prices=_multi_symbol_prices(
                {
                    "EURUSD": [1.0000 + index * 0.0010 for index in range(30)],
                    "GBPUSD": [1.3000 - index * 0.0010 for index in range(30)],
                }
            ),
            symbols=("EURUSD", "GBPUSD"),
            alpha_router=_config(),
            momentum=MomentumConfig(symbol="EURUSD"),
            breakout=BreakoutConfig(symbol="EURUSD"),
            mean_reversion=MeanReversionConfig(symbol="EURUSD"),
        )
        calibration = calibrate_ml_alpha_filters(
            evaluation,
            entry_probabilities=(0.55, 0.60),
            min_training_accuracies=(0.50,),
            min_expected_edges_bps=(0.0,),
            min_samples_for_trade=(1,),
        )

        with TemporaryDirectory() as tmpdir:
            prediction_path = Path(tmpdir) / "portfolio_predictions.csv"
            calibration_path = Path(tmpdir) / "calibration.csv"
            write_ml_alpha_portfolio_predictions_csv(evaluation, prediction_path)
            write_ml_alpha_calibration_csv(calibration, calibration_path)
            prediction_text = prediction_path.read_text(encoding="utf-8")
            calibration_text = calibration_path.read_text(encoding="utf-8")

        self.assertIn("timestamp,symbol,close,next_close,probability_up", prediction_text)
        self.assertIn("rank,label,entry_probability", calibration_text)
        self.assertIn("average_signed_return_bps", calibration_text)

    def test_symbol_calibration_csv_is_written(self) -> None:
        evaluation = evaluate_ml_alpha_portfolio(
            prices=_multi_symbol_prices(
                {
                    "EURUSD": [1.0000 + index * 0.0010 for index in range(30)],
                    "GBPUSD": [1.3000 - index * 0.0010 for index in range(30)],
                }
            ),
            symbols=("EURUSD", "GBPUSD"),
            alpha_router=_config(),
            momentum=MomentumConfig(symbol="EURUSD"),
            breakout=BreakoutConfig(symbol="EURUSD"),
            mean_reversion=MeanReversionConfig(symbol="EURUSD"),
        )
        rows = score_ml_alpha_filter_by_symbol(
            evaluation,
            filter_config=MLAlphaFilterConfig(
                entry_probability=0.55,
                min_training_accuracy=0.50,
                min_expected_edge_bps=0.0,
                min_samples_for_trade=1,
            ),
        )

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "symbol_calibration.csv"
            write_ml_alpha_symbol_calibration_csv(rows, path)
            text = path.read_text(encoding="utf-8")

        self.assertIn("symbol,actionable_predictions,coverage", text)
        self.assertIn("EURUSD", text)
        self.assertIn("GBPUSD", text)

    def test_walk_forward_calibration_scores_later_predictions(self) -> None:
        evaluation = evaluate_ml_alpha_portfolio(
            prices=_multi_symbol_prices(
                {
                    "EURUSD": [1.0000 + index * 0.0010 for index in range(60)],
                    "GBPUSD": [1.3000 - index * 0.0010 for index in range(60)],
                }
            ),
            symbols=("EURUSD", "GBPUSD"),
            alpha_router=_config(),
            momentum=MomentumConfig(symbol="EURUSD"),
            breakout=BreakoutConfig(symbol="EURUSD"),
            mean_reversion=MeanReversionConfig(symbol="EURUSD"),
        )

        walk_forward = walk_forward_calibrate_ml_alpha_filters(
            evaluation,
            train_timestamps=20,
            test_timestamps=10,
            step_timestamps=10,
            entry_probabilities=(0.55, 0.60),
            min_training_accuracies=(0.50,),
            min_expected_edges_bps=(0.0,),
            min_samples_for_trade=(1,),
        )

        self.assertGreater(len(walk_forward.folds), 0)
        self.assertGreaterEqual(walk_forward.positive_fold_rate, 0.0)
        self.assertLessEqual(walk_forward.positive_fold_rate, 1.0)

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ml_wf.csv"
            write_ml_alpha_walk_forward_csv(walk_forward, path)
            text = path.read_text(encoding="utf-8")

        self.assertIn("fold,train_start,train_end,test_start", text)
        self.assertIn("test_avg_signed_return_bps", text)

    def test_promotion_decision_rejects_unstable_ml_filter(self) -> None:
        decision = decide_ml_alpha_promotion(
            MLAlphaWalkForwardCalibration(
                (
                    _fold(1, -1.0),
                    _fold(2, -0.5),
                    _fold(3, 1.0),
                )
            )
        )

        self.assertEqual(decision.status, "REJECT")
        self.assertIn("positive fold rate", decision.reason)

    def test_promotion_decision_allows_stable_positive_filter(self) -> None:
        decision = decide_ml_alpha_promotion(
            MLAlphaWalkForwardCalibration(
                (
                    _fold(1, 1.0),
                    _fold(2, 0.5),
                    _fold(3, -0.1),
                    _fold(4, 0.2),
                )
            )
        )

        self.assertEqual(decision.status, "PROMOTE_TO_EXPERIMENT")


def _config() -> AlphaRouterConfig:
    return AlphaRouterConfig(
        ml_enabled=True,
        ml_lookback=4,
        ml_min_train_samples=6,
        ml_entry_probability=0.55,
        ml_label_threshold_bps=0.1,
        ml_train_window=40,
    )


def _prices(values: list[float]) -> PriceHistory:
    return PriceHistory(
        tuple(
            PriceBar(
                timestamp=datetime(2026, 6, 22, 10, 0, tzinfo=LONDON)
                + timedelta(minutes=index * 5),
                symbol="EURUSD",
                close=value,
            )
            for index, value in enumerate(values)
        )
    )


def _multi_symbol_prices(values_by_symbol: dict[str, list[float]]) -> PriceHistory:
    bars: list[PriceBar] = []
    for symbol, values in values_by_symbol.items():
        bars.extend(
            PriceBar(
                timestamp=datetime(2026, 6, 22, 10, 0, tzinfo=LONDON)
                + timedelta(minutes=index * 5),
                symbol=symbol,
                close=value,
            )
            for index, value in enumerate(values)
        )
    return PriceHistory(tuple(bars))


def _fold(index: int, test_average_signed_return_bps: float) -> MLAlphaCalibrationFold:
    filter_config = MLAlphaFilterConfig(
        entry_probability=0.60,
        min_training_accuracy=0.50,
        min_expected_edge_bps=0.0,
        min_samples_for_trade=1,
    )
    train_result = MLAlphaCalibrationResult(
        filter_config=filter_config,
        total_predictions=10,
        actionable_predictions=5,
        correct_actionable_predictions=3,
        long_count=3,
        short_count=2,
        cumulative_signed_return_bps=5.0,
    )
    test_result = MLAlphaCalibrationResult(
        filter_config=filter_config,
        total_predictions=10,
        actionable_predictions=5,
        correct_actionable_predictions=3,
        long_count=3,
        short_count=2,
        cumulative_signed_return_bps=test_average_signed_return_bps * 5,
    )
    return MLAlphaCalibrationFold(
        fold_index=index,
        train_start="2026-05-11T00:00:00+00:00",
        train_end="2026-05-11T01:00:00+00:00",
        test_start="2026-05-11T01:15:00+00:00",
        test_end="2026-05-11T02:00:00+00:00",
        selected_filter=filter_config,
        train_result=train_result,
        test_result=test_result,
    )
