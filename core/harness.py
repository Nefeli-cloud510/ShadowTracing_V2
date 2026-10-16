from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from core.json_safe import json_dumps
from core.schema import ExperimentConfig


class ExperimentHarness:
    """
    轻量级实验执行器。

    当前仓库先打通 AI 闭环架构，不依赖外部 astro 数据目录。
    因此这里使用确定性的本地 mock evaluator，后续可以替换成真实 runner。
    """

    def __init__(
        self,
        json_template_path: Path,
        json_output_path: Path,
        runner_script_path: Path,
        metrics_output_path: Path,
    ) -> None:
        self.project_root = Path(__file__).resolve().parent.parent
        self.json_template_path = self.project_root / json_template_path
        self.json_output_path = self.project_root / json_output_path
        self.runner_script_path = self.project_root / runner_script_path
        self.metrics_output_path = self.project_root / metrics_output_path
        self.results_dir = self.project_root / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
        self.json_output_path.parent.mkdir(parents=True, exist_ok=True)

    def run_round(self, round_index: int, config: ExperimentConfig) -> dict:
        round_dir = self.results_dir / f"round_{round_index:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        config_dict = config.model_dump()
        metrics = self._evaluate_config(config)
        result = {
            "round": round_index,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "mode": "local_mock",
            "runner_script": str(self.runner_script_path),
            "config": config_dict,
            "metrics": metrics,
            "artifacts": {
                "round_dir": str(round_dir),
                "config_snapshot": str(round_dir / "config.json"),
                "metrics_snapshot": str(round_dir / "metrics.json"),
                "latest_config": str(self.json_output_path),
                "latest_metrics": str(self.metrics_output_path),
            },
        }

        self._write_json(round_dir / "config.json", config_dict)
        self._write_json(round_dir / "metrics.json", metrics)
        self._write_json(round_dir / "result.json", result)
        self._write_json(self.json_output_path, config_dict)
        self._write_json(self.metrics_output_path, metrics)

        print(f"\n[Round {round_index}] 本地实验已完成")
        print(f"  配置快照: {round_dir / 'config.json'}")
        print(f"  指标快照: {round_dir / 'metrics.json'}")
        print(f"  test_r2: {metrics['test_r2']:.3f}")
        print(f"  test_rmse: {metrics['test_rmse']:.3f}")
        return result

    def _evaluate_config(self, config: ExperimentConfig) -> dict:
        window_penalty = abs(config.window_size - 21) / 40.0
        lag_penalty = abs(config.max_lag_day - 2) / 8.0
        alpha_penalty = abs(config.alpha - 0.35) / 1.5
        ratio_penalty = abs(config.l1_ratio - 0.65) / 1.2
        lag_bonus = 0.12 if config.use_lag_feature else 0.0

        test_r2 = 0.52 - window_penalty - lag_penalty - alpha_penalty - ratio_penalty + lag_bonus
        test_r2 = max(-0.25, min(0.82, test_r2))
        train_r2 = max(test_r2 + 0.08, min(0.9, test_r2 + 0.16))

        test_rmse = max(18.0, 135.0 - 55.0 * max(test_r2, -0.1))
        train_rmse = max(15.0, test_rmse - 7.5)
        test_mae = test_rmse * 0.78
        train_mae = train_rmse * 0.76
        pearson_r = max(-0.2, min(0.95, 0.18 + 0.92 * max(test_r2, 0.0)))
        n_features = config.window_size * (5 + (config.max_lag_day if config.use_lag_feature else 0))

        return {
            "train_rmse": round(train_rmse, 3),
            "test_rmse": round(test_rmse, 3),
            "train_mae": round(train_mae, 3),
            "test_mae": round(test_mae, 3),
            "train_r2": round(train_r2, 3),
            "test_r2": round(test_r2, 3),
            "test_pearson_r": round(pearson_r, 3),
            "baseline_test_rmse": 135.0,
            "n_features": int(n_features),
        }

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.write_text(json_dumps(payload, indent=2), encoding="utf-8")
