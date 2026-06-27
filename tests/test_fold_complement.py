from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.backtesting.fold_complement import (
    FoldReturn,
    evaluate_fold_complement,
    read_fold_returns,
    write_fold_complement_csv,
    write_fold_complement_summary_csv,
)


class FoldComplementTest(TestCase):
    def test_evaluates_candidate_help_on_flat_and_losing_folds(self) -> None:
        baseline = (
            FoldReturn(1, "a", "b", 0.0, 0),
            FoldReturn(2, "b", "c", -0.01, 4),
            FoldReturn(3, "c", "d", 0.02, 6),
        )
        candidate = (
            FoldReturn(1, "a", "b", 0.005, 2),
            FoldReturn(2, "b", "c", 0.003, 2),
            FoldReturn(3, "c", "d", -0.001, 2),
        )

        summary = evaluate_fold_complement(
            baseline=baseline,
            candidate=candidate,
            label="candidate",
        )

        self.assertEqual(summary.baseline_flat_folds, 1)
        self.assertEqual(summary.baseline_losing_folds, 1)
        self.assertEqual(summary.candidate_positive_on_baseline_flat, 1)
        self.assertEqual(summary.candidate_positive_on_baseline_losing, 1)
        self.assertEqual(summary.candidate_hurt_baseline_positive, 1)
        self.assertAlmostEqual(summary.combined_positive_fraction, 2 / 3)

    def test_requires_matching_fold_windows(self) -> None:
        baseline = (FoldReturn(1, "a", "b", 0.0, 0),)
        candidate = (FoldReturn(1, "x", "b", 0.01, 1),)

        with self.assertRaisesRegex(ValueError, "windows"):
            evaluate_fold_complement(
                baseline=baseline,
                candidate=candidate,
                label="bad",
            )

    def test_reads_and_writes_csv_outputs(self) -> None:
        with TemporaryDirectory() as tmpdir:
            folds_path = Path(tmpdir) / "folds.csv"
            summary_path = Path(tmpdir) / "summary.csv"
            detail_path = Path(tmpdir) / "detail.csv"
            folds_path.write_text(
                "fold,test_start,test_end,return_pct,evaluation_fills\n"
                "1,a,b,0.01,2\n",
                encoding="utf-8",
            )

            folds = read_fold_returns(folds_path)
            summary = evaluate_fold_complement(
                baseline=folds,
                candidate=folds,
                label="self",
            )
            write_fold_complement_summary_csv((summary,), summary_path)
            write_fold_complement_csv((summary,), detail_path)

            summary_text = summary_path.read_text(encoding="utf-8")
            detail_text = detail_path.read_text(encoding="utf-8")

        self.assertEqual(folds[0].fold, 1)
        self.assertIn("combined_positive_fraction", summary_text)
        self.assertIn("candidate_helped_flat_or_losing", detail_text)
