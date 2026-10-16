import unittest

from core.evaluation_unified import _get_run
from core.support_update_rules import (
    EVIDENCE_IDENTICAL_PREDICTIONS,
    classify_rmse_gain_evidence,
    compute_support_update_from_evidence,
)
from core.unified_schema import (
    ExecutionSummary,
    ExperimentResult,
    ExperimentRun,
    RunMetrics,
)
import scripts.report_exporter as report_exporter


def _decisions_bundle(*decisions: dict) -> dict:
    return {
        "round": 2,
        "decision_log": {"decisions": list(decisions)},
    }


def _review_decision() -> dict:
    return {
        "decision_id": "D018",
        "timestamp": "2026-10-13T22:03:31",
        "round_id": 2,
        "decision_type": "round_review_requested",
        "summary": "系统请求 human_pi 对 round 2 作出下一步决策",
        "details": {},
    }


class ReportDecisionConsistencyTest(unittest.TestCase):
    def test_stop_decision_is_latest_and_controls_report(self) -> None:
        bundle = _decisions_bundle(
            _review_decision(),
            {
                "decision_id": "D019",
                "timestamp": "2026-10-13T22:03:39",
                "round_id": 2,
                "decision_type": "terminate_workflow",
                "made_by": "human_pi",
                "summary": "用户在 round 2 结束后停止工作流",
                "details": {"human_feedback": "已完成两轮闭环，固化全部结果。"},
            },
            {
                "decision_id": "D011",
                "timestamp": "2026-10-13T21:30:53",
                "round_id": 1,
                "decision_type": "continue_next_round",
                "details": {"human_feedback": "进入下一轮。"},
            },
        )
        latest = report_exporter._latest_commit_decision(bundle, round_id=2)
        self.assertEqual(latest["decision_type"], "terminate_workflow")
        self.assertEqual(
            report_exporter._commit_decision_label(latest),
            "停止工作流",
        )
        html = report_exporter._conclusion_section(bundle)
        self.assertIn("停止工作流", html)
        self.assertIn("已完成两轮闭环，固化全部结果。", html)
        self.assertIn("22:03:39", html)

    def test_review_without_commit_waits_for_pi(self) -> None:
        bundle = _decisions_bundle(
            _review_decision(),
            {
                "decision_id": "D011",
                "timestamp": "2026-10-13T21:30:53",
                "round_id": 1,
                "decision_type": "continue_next_round",
                "details": {"human_feedback": "进入下一轮。"},
            },
        )
        html = report_exporter._conclusion_section(bundle)
        self.assertIn("待人工决策", html)
        self.assertNotIn("停止工作流", html)

    def test_interpretation_source_label_is_honest(self) -> None:
        self.assertEqual(
            report_exporter._interpretation_source_tag(
                {"interpretation_source": "llm_interpreter"}
            ),
            "真实 LLM 原文",
        )
        self.assertEqual(
            report_exporter._interpretation_source_tag({}),
            "程序规则回退",
        )

    def test_four_layer_uses_source_tag(self) -> None:
        bundle = {
            "round": 1,
            "evaluation": {
                "metrics": {},
                "scientific": {
                    "three_layer_conclusion": {
                        "data_layer": {
                            "rmse_attribution": "数值归因",
                            "interpretation_source": "llm_interpreter",
                        },
                        "scientific_layer": {
                            "answer": "程序答案",
                            "interpretation_source": "program_fallback",
                        },
                    }
                },
            },
            "result": {},
        }
        html = report_exporter._four_layer_section(bundle)
        self.assertIn("RMSE 变化归因（真实 LLM 原文）", html)
        self.assertIn("科学问题回答（程序规则回退）", html)


class IdenticalPredictionsEvidenceTest(unittest.TestCase):
    def test_zero_ci_classifies_as_identical_and_does_not_update_support(self) -> None:
        evidence = classify_rmse_gain_evidence(
            rmse_delta=0.0,
            rmse_ci=[0.0, 0.0],
        )
        self.assertEqual(evidence.classification, EVIDENCE_IDENTICAL_PREDICTIONS)
        self.assertIn("两组预测完全一致", evidence.labels[0])

        outcome = compute_support_update_from_evidence(
            current_support=0.50,
            evidence_classification=evidence.classification,
            evidence_labels=evidence.labels,
            expected_effect="positive",
            expected_range=[0.02, 0.10],
            observed_delta=0.0,
        )
        self.assertFalse(outcome.updated)
        self.assertEqual(outcome.support_after, 0.50)


class RunLookupByIdentityTest(unittest.TestCase):
    def test_run_id_identity_wins_over_display_name(self) -> None:
        result = ExperimentResult(
            experiment_id="E1",
            round_id=1,
            status="completed",
            execution=ExecutionSummary(status="completed"),
            runs=[
                ExperimentRun(
                    run_id="baseline",
                    name="对照组",
                    metrics=RunMetrics(rmse=3.59, pearson_r=0.06),
                ),
                ExperimentRun(
                    run_id="treatment",
                    name="实验组",
                    metrics=RunMetrics(rmse=3.59, pearson_r=0.06),
                ),
            ],
        )
        baseline = _get_run(result, "baseline")
        treatment = _get_run(result, "treatment")
        self.assertIsNotNone(baseline)
        self.assertIsNotNone(treatment)
        self.assertEqual(baseline.name, "对照组")
        self.assertEqual(treatment.name, "实验组")
        self.assertIsNone(_get_run(result, "missing"))


class ReportRunOrderingAndCandidateTypeTest(unittest.TestCase):
    def test_metric_columns_keep_baseline_treatment_first(self) -> None:
        order = report_exporter._ordered_run_ids(
            ["treatment", "ridge_treatment", "baseline", "ridge_baseline"]
        )
        self.assertEqual(order[:2], ["baseline", "treatment"])
        self.assertEqual(set(order[2:]), {"ridge_treatment", "ridge_baseline"})

    def test_pdf_stem_drops_duplicate_extension(self) -> None:
        self.assertEqual(
            report_exporter._pdf_stem("ShadowTracing_Round_02_Report.pdf"),
            "ShadowTracing_Round_02_Report",
        )
        self.assertEqual(report_exporter._pdf_stem("report"), "report")

    def test_candidate_type_uses_structured_type_and_never_stale_lag_label(self) -> None:
        bundle = {
            "round": 1,
            "uncertainties": {"records": []},
            "decision_log": {"decisions": []},
            "protocol": {},
            "candidate_experiments": {
                "candidates": [
                    {
                        "experiment_id": "E_R01_01",
                        "utility_score": "0.8",
                        "experiment_type": "model_sensitivity",
                        "type": "lag_sensitivity",
                    },
                    {
                        "experiment_id": "E_R01_02",
                        "utility_score": "0.6",
                        "type": "baseline_benchmark",
                    },
                    {
                        "experiment_id": "E_R01_03",
                        "utility_score": "0.5",
                        "type": "lag_sensitivity",
                    },
                ]
            },
        }
        html = report_exporter._uncertainty_candidate_section(bundle)
        self.assertIn("模型敏感性", html)
        self.assertIn("单臂基准", html)
        self.assertIn("未标注", html)
        self.assertNotIn("lag_sensitivity", html)


class InvalidatedEvidenceReportTest(unittest.TestCase):
    def test_conclusion_section_blocked_when_evidence_invalidated(self) -> None:
        bundle = {
            "round": 2,
            "task": {
                "payload": {
                    "research_question": {
                        "variables": {"y": "太阳风速度"},
                    }
                }
            },
            "evaluation": {
                "metrics": {},
                "scientific": {
                    "three_layer_conclusion": {
                        "data_layer": {},
                        "scientific_layer": {"answer": "不应进入结论"},
                    }
                },
            },
            "result": {},
        }
        invalidation = {
            "status": "evidence_invalidated",
            "root_cause": "OMNI 字段错位导致真实预测目标是质子密度",
            "invalidated_experiments": ["E_R01_05", "E_R02_03"],
        }
        banner = report_exporter._invalidation_banner(invalidation)
        self.assertIn("本节能科学结论已阻断", banner)
        self.assertIn("质子密度", banner)
        self.assertIn("E_R01_05、E_R02_03", banner)
        self.assertIn("仅作审计复现", banner)


class HypothesisDisplayNameTest(unittest.TestCase):
    def _bundle(self) -> dict:
        return {
            "round": 2,
            "hypothesis_tree": {
                "nodes": [
                    {
                        "hypothesis_id": "H_shadow_incremental_gain",
                        "display_hypothesis_id": "H1",
                        "statement": "在共同前景模型中，日影南北偏移对太阳风速度预测有增量贡献",
                    },
                    {
                        "hypothesis_id": "H_by_mediated_path",
                        "display_hypothesis_id": "H2",
                        "statement": "日影改善通过行星际磁场中介传递",
                    },
                ]
            },
            "evaluation": {
                "metrics": {},
                "scientific": {
                    "three_layer_conclusion": {
                        "data_layer": {},
                        "scientific_layer": {},
                        "hypothesis_layer": [
                            {
                                "hypothesis_id": "H_shadow_incremental_gain",
                                "predicted_direction": "positive",
                                "actual_delta": None,
                                "direction_matched": False,
                                "magnitude_matched": False,
                                "conclusion": "证据不足",
                                "support_after": 0.5,
                            }
                        ],
                    }
                },
            },
            "result": {},
        }

    def test_assessment_table_uses_display_id(self) -> None:
        bundle = self._bundle()
        html = report_exporter._four_layer_section(bundle)
        self.assertIn("<td>H1</td>", html)
        self.assertNotIn("H_shadow_incremental_gain", html)

    def test_related_hypotheses_use_display_ids(self) -> None:
        bundle = self._bundle()
        bundle["uncertainties"] = {
            "records": [
                {
                    "uncertainty_id": "U1",
                    "related_hypotheses": "H_shadow_incremental_gain、H_by_mediated_path",
                    "priority": "high",
                    "status": "active",
                    "question": "日影是否提供增量信息？",
                }
            ]
        }
        bundle["candidate_experiments"] = {"candidates": []}
        html = report_exporter._uncertainty_candidate_section(bundle)
        self.assertIn("<td>H1、H2</td>", html)
        self.assertNotIn("H_shadow_incremental_gain", html)


class PlottingFontConfigurationTest(unittest.TestCase):
    def test_runner_configures_chinese_font_for_charts(self) -> None:
        import matplotlib.pyplot as plt
        import models.elasticnet_runner  # noqa: F401  (module applies rcParams on import)

        self.assertIn("Microsoft YaHei", plt.rcParams["font.sans-serif"])
        self.assertFalse(plt.rcParams["axes.unicode_minus"])
        self.assertEqual(plt.rcParams["axes.titlesize"], 17)

    def test_coefficient_plot_label_tracks_actual_model_family(self) -> None:
        import tempfile
        from pathlib import Path
        from unittest import mock

        import pandas as pd

        from models.elasticnet_runner import plot_feature_coefficients

        coef_df = pd.DataFrame(
            {
                "feature": ["ss_br_mean", "p2_y_center"],
                "coefficient": [0.12, -0.05],
                "abs_coefficient": [0.12, 0.05],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("matplotlib.pyplot.title") as mocked_title, mock.patch(
                "matplotlib.pyplot.savefig"
            ):
                output = plot_feature_coefficients(
                    coef_df, Path(tmp), model_family="elasticnet"
                )
        self.assertEqual(output.name, "elasticnet_coefficients.png")
        self.assertIn("ElasticNet", mocked_title.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
