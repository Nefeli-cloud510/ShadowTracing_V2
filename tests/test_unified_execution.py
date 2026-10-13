import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from core.evaluation_unified import evaluate_experiment
from core.harness_unified import UnifiedExperimentHarness
from core.protocol_bridge import (
    ElasticNetDataSourceConfig,
    ElasticNetProtocolBridge,
    legacy_config_to_protocol,
)
from core.schema import ExperimentConfig as LegacyExperimentConfig
from core.unified_schema import ExperimentProtocol, ModelSpec


class UnifiedExecutionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        (self.root / "config").mkdir()
        (self.root / "outputs").mkdir()
        (self.root / "results").mkdir()

        time_values = pd.date_range("2021-01-01", periods=36, freq="D")
        omni = pd.DataFrame(
            {
                "TIME": time_values.strftime("%Y%m%d"),
                "By": [(-1) ** i * (i % 5 + 1) for i in range(36)],
                "Bz": [(i % 7) - 3 for i in range(36)],
                "Vsw": [350 + i * 2 for i in range(36)],
            }
        )
        lhaaso = pd.DataFrame(
            {
                "TIME": time_values.strftime("%Y%m%d"),
                "DeltaDec": [0.1 * ((i % 6) - 3) for i in range(36)],
            }
        )

        self.omni_path = self.root / "omni.csv"
        self.lhaaso_path = self.root / "lhaaso.csv"
        omni.to_csv(self.omni_path, index=False)
        lhaaso.to_csv(self.lhaaso_path, index=False)

        self.data_source = ElasticNetDataSourceConfig(
            omni_file_path=self.omni_path,
            lhaaso_file_path=self.lhaaso_path,
            time_column="TIME",
            target_column="Vsw",
            test_split_ratio=0.25,
            random_state=7,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_bridge_builds_runner_config(self) -> None:
        protocol = ExperimentProtocol(
            experiment_id="E01",
            round_id=1,
            task_id="ST_001",
            target="Vsw",
            features={
                "target": "Vsw",
                "control": ["By", "Bz"],
                "treatment": ["By", "Bz", "DeltaDec"],
                "lags": {"DeltaDec": [1, 2]},
            },
            model=ModelSpec(
                name="ElasticNet",
                parameters={
                    "window_size": 3,
                    "use_lag_feature": True,
                    "max_lag_day": 2,
                    "alpha": 0.2,
                    "l1_ratio": 0.6,
                    "random_state": 7,
                    "test_split_ratio": 0.25,
                },
            ),
        )
        bridge = ElasticNetProtocolBridge(self.data_source)
        config = bridge.build_runner_config(protocol, ["By", "Bz", "DeltaDec"])

        self.assertEqual(config.file_config.target_column, "Vsw")
        self.assertEqual(config.time_window_config.window_size, 3)
        self.assertAlmostEqual(config.elasticnet_config.alpha, 0.2)

    def test_legacy_adapter(self) -> None:
        legacy = LegacyExperimentConfig(
            window_size=5,
            use_lag_feature=True,
            max_lag_day=2,
            alpha=0.25,
            l1_ratio=0.65,
            reasoning="legacy config",
        )
        protocol = legacy_config_to_protocol(
            legacy,
            round_id=2,
            task_id="ST_002",
            target="Vsw",
            control_features=["By", "Bz"],
            treatment_features=["By", "Bz", "DeltaDec"],
            experiment_id="E02",
        )

        self.assertEqual(protocol.model.parameters["window_size"], 5)
        self.assertEqual(protocol.features.treatment[-1], "DeltaDec")

    def test_unified_harness_runs_real_runner(self) -> None:
        protocol = ExperimentProtocol(
            experiment_id="E03",
            round_id=1,
            task_id="ST_003",
            target="Vsw",
            features={
                "target": "Vsw",
                "control": ["By", "Bz"],
                "treatment": ["By", "Bz", "DeltaDec"],
                "lags": {"DeltaDec": [1, 2]},
            },
            model=ModelSpec(
                name="ElasticNet",
                parameters={
                    "window_size": 4,
                    "use_lag_feature": True,
                    "max_lag_day": 2,
                    "alpha": 0.1,
                    "l1_ratio": 0.5,
                    "random_state": 7,
                    "test_split_ratio": 0.25,
                    "forecast_horizon_days": 1,
                    "past_lag_days": 4,
                },
            ),
        )
        harness = UnifiedExperimentHarness(
            data_source=self.data_source,
            project_root=self.root,
            run_shap=False,
        )
        result = harness.run(protocol)
        evaluation = evaluate_experiment(
            result,
            protocol=protocol,
            remaining_uncertainties=[{"question": "DeltaDec 的增益是否跨时间段稳定"}],
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(len(result.runs), 2)
        self.assertTrue((self.root / "results" / "round_01" / "result_unified.json").exists())
        self.assertTrue((self.root / "config" / "latest_protocol.json").exists())
        self.assertTrue((self.root / "outputs" / "latest_metrics.json").exists())
        self.assertFalse(Path(result.runs[0].prediction_artifact).is_absolute())
        self.assertTrue(result.visualizations)
        self.assertFalse(Path(result.visualizations[0].path).is_absolute())
        self.assertEqual(evaluation.metrics.experiment_id, "E03")
        self.assertIsNotNone(evaluation.metrics.pg_actual_signed)
        self.assertGreaterEqual(evaluation.metrics.pg_actual_clipped, 0.0)
        self.assertEqual(
            evaluation.scientific.evidence_summary.remaining_uncertainties[0]["question"],
            "DeltaDec 的增益是否跨时间段稳定",
        )
        self.assertEqual(evaluation.visualizations[0].path, result.visualizations[0].path)

    def test_frozen_evaluation_plan_is_shared_across_candidates(self) -> None:
        def protocol(experiment_id: str, treatment: list[str]):
            return ExperimentProtocol(
                experiment_id=experiment_id,
                round_id=1,
                task_id="ST_004",
                target="Vsw",
                features={
                    "target": "Vsw",
                    "control": ["By", "Bz"],
                    "treatment": treatment,
                    "lags": {},
                },
                model=ModelSpec(
                    name="ElasticNet",
                    parameters={
                        "window_size": 4,
                        "use_lag_feature": False,
                        "alpha": 0.1,
                        "l1_ratio": 0.5,
                        "random_state": 7,
                        "test_split_ratio": 0.25,
                        "forecast_horizon_days": 1,
                        "past_lag_days": 4,
                    },
                ),
            )

        harness = UnifiedExperimentHarness(
            data_source=self.data_source,
            project_root=self.root,
            run_shap=False,
        )
        first = harness.run(protocol("E04", ["By", "Bz", "DeltaDec"]))

        plan_path = self.root / "config" / "evaluation_plan.json"
        self.assertTrue(plan_path.exists())
        first_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        self.assertEqual(first_plan["split_label"], "dev_backtest")
        self.assertIsNotNone(first_plan["test_start"])

        harness.run(protocol("E05", ["By", "Bz"]))
        second_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        self.assertEqual(
            second_plan["test_start"],
            first_plan["test_start"],
            "候选实验改变特征集合后不得移动评估分区边界",
        )

        archived_plan = (
            self.root
            / "results"
            / "round_01"
            / "archive"
            / first.run_id
            / "evaluation_plan.json"
        )
        self.assertTrue(archived_plan.exists())
        archived = json.loads(archived_plan.read_text(encoding="utf-8"))
        self.assertEqual(archived["split_label"], "dev_backtest")
        self.assertEqual(archived["test_start"], first_plan["test_start"])

        archive_dir = self.root / "results" / "round_01" / "archive" / first.run_id
        for archive_name in (
            "data_manifest.json",
            "sample_manifest.csv",
            "diagnostics.json",
            "run_notes.json",
            "compiled_plan.json",
        ):
            self.assertTrue((archive_dir / archive_name).exists(), archive_name)
        self.assertTrue((self.root / "outputs" / "latest_metrics.json").exists())

    def test_pfss_missing_days_only_affects_arms_using_pfss_features(self) -> None:
        base_dates = pd.date_range("2021-01-01", periods=40, freq="D")
        omni = pd.DataFrame(
            {
                "TIME": base_dates.strftime("%Y%m%d"),
                "By": [(-1) ** i * (i % 5 + 1) for i in range(40)],
                "Bz": [(i % 7) - 3 for i in range(40)],
                "Vsw": [350 + i * 2 for i in range(40)],
            }
        )
        lhaaso = pd.DataFrame(
            {
                "TIME": base_dates.strftime("%Y%m%d"),
                "DeltaDec": [0.1 * ((i % 6) - 3) for i in range(40)],
            }
        )
        pfss_dates = list(base_dates)
        missing_days = pd.date_range("2021-01-10", periods=7, freq="D")
        missing_keys = {day.strftime("%Y%m%d") for day in missing_days}
        pfss_frame = pd.DataFrame(
            {
                "TIME": [day.strftime("%Y%m%d") for day in pfss_dates if day.strftime("%Y%m%d") not in missing_keys],
                "PFSS_Bt": [0.5 * ((i % 9) - 4) for i in range(len(pfss_dates) - len(missing_days))],
            }
        )

        omni_path = self.root / "omni_pfss.csv"
        lhaaso_path = self.root / "lhaaso_pfss.csv"
        pfss_path = self.root / "pfss.csv"
        omni.to_csv(omni_path, index=False)
        lhaaso.to_csv(lhaaso_path, index=False)
        pfss_frame.to_csv(pfss_path, index=False)

        with_pfss_source = ElasticNetDataSourceConfig(
            omni_file_path=omni_path,
            lhaaso_file_path=lhaaso_path,
            extra_file_paths=[pfss_path],
            time_column="TIME",
            target_column="Vsw",
            test_split_ratio=0.25,
            random_state=7,
            sparse_sources=["pfss"],
        )

        protocol_with_pfss = ExperimentProtocol(
            experiment_id="E_PFSS",
            round_id=1,
            task_id="ST_PFSS_001",
            target="Vsw",
            features={
                "target": "Vsw",
                "control": ["By", "Bz"],
                "treatment": ["By", "Bz", "PFSS_Bt"],
                "lags": {"PFSS_Bt": [1, 2]},
            },
            model=ModelSpec(
                name="ElasticNet",
                parameters={
                    "window_size": 4,
                    "use_lag_feature": True,
                    "max_lag_day": 2,
                    "alpha": 0.1,
                    "l1_ratio": 0.5,
                    "random_state": 7,
                    "test_split_ratio": 0.25,
                },
            ),
        )
        harness = UnifiedExperimentHarness(
            data_source=with_pfss_source,
            project_root=self.root,
            run_shap=False,
        )
        result = harness.run(protocol_with_pfss)

        pfss_coverage = [
            item
            for item in result.data_coverage
            if item.source == "pfss"
        ]
        self.assertTrue(pfss_coverage, "使用 PFSS 特征的实验臂应输出 PFSS 覆盖审计")
        self.assertEqual(
            {item.run_id for item in pfss_coverage},
            {"baseline", "treatment"},
            "共同样本语义下 PFSS 缺失日同时作用于双臂",
        )
        for item in pfss_coverage:
            self.assertEqual(item.expected_days, 40)
            self.assertEqual(item.observed_days, 33)
            self.assertEqual(item.missing_days, 7)
            self.assertAlmostEqual(item.coverage_ratio, 33 / 40, places=6)
            self.assertEqual(item.dropped_gap_windows, 0)
            self.assertEqual(
                item.note,
                "缺失记录保留至日历样本层，按历史特征与目标可用性判定保留或排除",
            )

        protocol_without_pfss = ExperimentProtocol(
            experiment_id="E_NO_PFSS",
            round_id=1,
            task_id="ST_PFSS_002",
            target="Vsw",
            features={
                "target": "Vsw",
                "control": ["By", "Bz"],
                "treatment": ["By", "Bz", "DeltaDec"],
                "lags": {"DeltaDec": [1, 2]},
            },
            model=ModelSpec(
                name="ElasticNet",
                parameters={
                    "window_size": 4,
                    "use_lag_feature": True,
                    "max_lag_day": 2,
                    "alpha": 0.1,
                    "l1_ratio": 0.5,
                    "random_state": 7,
                    "test_split_ratio": 0.25,
                    "forecast_horizon_days": 1,
                    "past_lag_days": 4,
                },
            ),
        )
        result_without_pfss = harness.run(protocol_without_pfss)
        self.assertFalse(
            any(item.source == "pfss" for item in result_without_pfss.data_coverage),
            "未使用 PFSS 特征的实验不应输出 PFSS 覆盖审计",
        )


if __name__ == "__main__":
    unittest.main()
