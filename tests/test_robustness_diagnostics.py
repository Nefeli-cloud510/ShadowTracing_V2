import unittest

import numpy as np
import pandas as pd

from core.robustness_diagnostics import (
    _block_ids,
    empty_bootstrap_summary,
    paired_calendar_block_bootstrap,
    summary_to_robustness_report,
)
from core.unified_schema import BootstrapReport


def _paired_frames(
    n: int = 40,
    *,
    start: str = "2021-03-01",
    horizon: int = 1,
):
    dates = pd.date_range(start, periods=n, freq="D")
    y_true = np.array([300 + 2.0 * i + np.sin(i) for i in range(n)], dtype=float)
    common = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(n)],
            "issue_date": dates.strftime("%Y-%m-%d"),
            "target_date": (dates + pd.Timedelta(days=horizon)).strftime("%Y-%m-%d"),
            "y_true": y_true,
        }
    )
    baseline = common.assign(y_pred=y_true + np.cos(np.arange(n)))
    treatment = common.assign(y_pred=y_true + 100.0)
    return baseline, treatment


class RobustnessDiagnosticsTest(unittest.TestCase):
    def test_block_ids_restart_each_year(self) -> None:
        issues = pd.Series(
            [
                "2021-01-01",
                "2021-01-07",
                "2022-01-01",
                "2022-01-07",
            ]
        )
        ids = _block_ids(issues, block_size=7)

        self.assertNotEqual(int(ids[0]), int(ids[2]))
        self.assertNotEqual(int(ids[1]), int(ids[3]))
        self.assertEqual(int(ids[0]), int(ids[1]))
        self.assertEqual(int(ids[2]), int(ids[3]))
        self.assertEqual(sorted(set(ids)), [0, 1])

    def test_block_ids_remain_contiguous_across_years(self) -> None:
        issues = pd.Series(
            [
                "2020-12-30",
                "2021-01-01",
                "2022-01-01",
            ]
        )
        ids = _block_ids(issues, block_size=7)

        self.assertEqual(list(ids), [0, 1, 2])

    def test_bootstrap_pearson_ci_for_constant_shift(self) -> None:
        baseline_pred, treatment_pred = _paired_frames()
        # treatment 是 baseline 的常数偏移：两条臂的 Pearson 都等于 1，
        # 相关系数增益恒为 0，CI 必须近似 [0, 0]。
        baseline = baseline_pred.assign(y_pred=baseline_pred["y_true"].to_numpy())
        treatment = treatment_pred

        summary = paired_calendar_block_bootstrap(
            baseline,
            treatment,
            block_size=7,
            n_draws=100,
            seed=20261008,
            verify_pearson=True,
        )

        self.assertEqual(summary["n_valid_pearson_draws"], 100)
        self.assertAlmostEqual(summary["point_gain_pearson_r"], 0.0, places=9)
        lower, upper = summary["pearson_r_ci"]
        self.assertLess(abs(lower), 1e-6)
        self.assertLess(abs(upper), 1e-6)

    def test_vectorized_pearson_matches_weighted_pearson(self) -> None:
        baseline, treatment = _paired_frames()
        summary = paired_calendar_block_bootstrap(
            baseline,
            treatment,
            block_size=7,
            n_draws=100,
            seed=11,
            verify_pearson=True,
        )
        self.assertEqual(summary["status"], "completed")
        self.assertGreater(summary["n_valid_pearson_draws"], 0)
        # verify_pearson=True 时，前 25 次抽样会与 weighted_pearson() 逐项比对，
        # 未抛出 AssertionError 即证明向量化实现与参考实现一致。

    def test_unavailable_pearson_ci_is_none_not_null_pair(self) -> None:
        baseline, treatment = _paired_frames(n=40)
        baseline = baseline.assign(y_pred=350.0)
        treatment = treatment.assign(y_pred=351.0)

        summary = paired_calendar_block_bootstrap(
            baseline,
            treatment,
            block_size=7,
            n_draws=100,
            seed=20261017,
        )

        self.assertEqual(summary["n_valid_pearson_draws"], 0)
        self.assertIsNone(summary["pearson_r_ci"])

        report = summary_to_robustness_report(summary)
        self.assertIsNone(report.bootstrap.pearson_r_ci)
        reloaded = BootstrapReport.model_validate(report.bootstrap.model_dump())
        self.assertIsNone(reloaded.pearson_r_ci)

    def test_summary_to_robustness_report_does_not_report_time_slice(self) -> None:
        summary = {
            "status": "completed",
            "n_iterations": 100,
            "block_size": 7,
            "seed": 1,
            "ci_level": 95.0,
            "point_gain_rmse": -0.5,
            "point_gain_mae": -0.4,
            "point_gain_pearson_r": 0.01,
            "rmse_ci": [-0.9, -0.1],
            "mae_ci": [-0.8, -0.2],
            "pearson_r_ci": [-0.02, 0.04],
            "stable_rmse": True,
        }

        report = summary_to_robustness_report(summary)

        self.assertTrue(report.overall.stable)
        self.assertEqual(report.overall.status, "tested")
        self.assertFalse(report.time_slice.stable)
        self.assertEqual(report.time_slice.status, "not_run")
        self.assertIn("未运行分时段检验", report.recommendation)

    def test_single_calendar_block_never_reports_directional_stability(self) -> None:
        baseline, treatment = _paired_frames(n=3, start="2021-03-01")

        summary = paired_calendar_block_bootstrap(
            baseline,
            treatment,
            block_size=7,
            n_draws=100,
            seed=1,
        )

        self.assertEqual(summary["n_nonempty_blocks"], 1)
        self.assertTrue(summary["insufficient_blocks"])
        self.assertFalse(summary["stable_rmse"])

        report = summary_to_robustness_report(summary)

        self.assertFalse(report.overall.stable)
        self.assertEqual(report.overall.status, "inconclusive")
        self.assertEqual(report.time_slice.status, "not_run")
        self.assertTrue(report.bootstrap.insufficient_blocks)
        self.assertEqual(report.bootstrap.n_nonempty_blocks, 1)
        self.assertEqual(report.bootstrap.min_blocks, 2)
        self.assertIn("低于预登记最小块数", report.overall.concern or "")
        self.assertIn("不能据此作方向性稳定结论", report.recommendation or "")

    def test_missing_robustness_summary_is_not_tested(self) -> None:
        report = summary_to_robustness_report(None)

        self.assertFalse(report.overall.stable)
        self.assertEqual(report.overall.status, "not_tested")
        self.assertEqual(report.bootstrap.status, "not_run")
        self.assertEqual(report.time_slice.status, "not_run")

    def test_empty_bootstrap_summary_is_not_tested(self) -> None:
        report = summary_to_robustness_report(
            empty_bootstrap_summary("缺少共同评价样本")
        )

        self.assertEqual(report.overall.status, "not_tested")
        self.assertFalse(report.overall.stable)
        self.assertEqual(report.bootstrap.status, "not_run")
        self.assertIn("缺少共同评价样本", report.overall.concern or "")


if __name__ == "__main__":
    unittest.main()
