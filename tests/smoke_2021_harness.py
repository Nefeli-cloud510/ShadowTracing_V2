"""Real-data harness smoke checks for the shared-calendar execution layer.

Uses only data/2021_public_data.xlsx (196 public days). Both arms of an
experiment are built from one union sample plan, so every comparison happens
on the same issue/target dates. run_shap is disabled here to keep the smoke
run fast; the production harness still emits SHAP artifacts when enabled.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.harness_unified import UnifiedExperimentHarness
from core.protocol_bridge import ElasticNetDataSourceConfig
from core.unified_schema import ExperimentProtocol, ModelSpec

EXCEL = ROOT / "data" / "2021_public_data.xlsx"
TARGET = "SW Plasma Speed, km/s"
SHADOW = "p2_y中心值"
BY = "BY, nT (GSE)"
BZ = "BZ, nT (GSE)"


def _data_source() -> ElasticNetDataSourceConfig:
    return ElasticNetDataSourceConfig(
        omni_file_path=EXCEL,
        lhaaso_file_path=EXCEL,
        time_column="dataset_id",
        target_column=TARGET,
        test_split_ratio=0.2,
        random_state=42,
    )


def _protocol(
    experiment_id: str,
    control: list[str],
    treatment: list[str],
) -> ExperimentProtocol:
    return ExperimentProtocol(
        experiment_id=experiment_id,
        round_id=90,
        task_id="ST_2021_196D_SMOKE",
        target=TARGET,
        forecast_horizon_days=3,
        features={
            "target": TARGET,
            "control": control,
            "treatment": treatment,
            "lags": {},
            "forecast_horizon_days": 3,
            "past_lag_days": 3,
        },
        model=ModelSpec(
            name="Ridge",
            parameters={
                "model_family": "ridge",
                "window_size": 3,
                "past_lag_days": 3,
                "forecast_horizon_days": 3,
                "test_split_ratio": 0.2,
                "random_state": 42,
                "ridge_solver": "svd",
                "alpha_grid": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            },
        ),
        notes=["smoke:2021_public_196d", "smoke:shared_calendar_common_sample"],
    )


def _print_experiment(name: str, protocol: ExperimentProtocol, result) -> None:
    baseline = result.runs[0]
    treatment = result.runs[1]
    print(f"[{name}] {protocol.experiment_id} status={result.status}")
    print(
        f"[{name}] baseline rmse={baseline.metrics.rmse:.3f} "
        f"r={baseline.metrics.pearson_r:.3f}"
    )
    print(
        f"[{name}] treatment rmse={treatment.metrics.rmse:.3f} "
        f"r={treatment.metrics.pearson_r:.3f}"
    )
    print(f"[{name}] gain_rmse={result.comparison.rmse:.3f} gain_r={result.comparison.pearson_r:.3f}")
    robustness = result.robustness
    print(
        f"[{name}] robustness stable={robustness.overall.stable if robustness else None} "
        f"recommendation={robustness.recommendation if robustness else None}"
    )


def main() -> None:
    harness = UnifiedExperimentHarness(data_source=_data_source(), run_shap=False)
    experiments = [
        (
            "AR vs AR+shadow",
            "E_SMK_A01",
            _protocol(
                "E_SMK_A01",
                control=[TARGET],
                treatment=[TARGET, SHADOW],
            ),
        ),
        (
            "AR+IMF vs AR+IMF+shadow",
            "E_SMK_B01",
            _protocol(
                "E_SMK_B01",
                control=[TARGET, BY, BZ],
                treatment=[TARGET, BY, BZ, SHADOW],
            ),
        ),
    ]
    for name, _, protocol in experiments:
        result = harness.run(protocol)
        _print_experiment(name, protocol, result)
        baseline_ids = pd.read_csv(ROOT / result.runs[0].prediction_artifact)["sample_id"].astype(str).tolist()
        treatment_ids = pd.read_csv(ROOT / result.runs[1].prediction_artifact)["sample_id"].astype(str).tolist()
        assert baseline_ids == treatment_ids, f"{name}: 双臂 sample_id 不一致"
        print(f"[{name}] shared_test_samples={len(baseline_ids)}")
        coverage_sources = {item.source for item in result.data_coverage}
        print(f"[{name}] coverage_sources={sorted(coverage_sources)}")


if __name__ == "__main__":
    main()
