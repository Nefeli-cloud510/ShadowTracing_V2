from __future__ import annotations

from pathlib import Path

from core.json_safe import json_dumps


class RoundManager:
    def __init__(self) -> None:
        self.project_root = Path(__file__).resolve().parent.parent
        self.results_dir = self.project_root / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.summary_path = self.results_dir / "summary.json"
        self.rounds: list[dict] = []

    def save_round(self, result: dict) -> None:
        self.rounds.append(result)
        self.summary_path.write_text(
            json_dumps({"rounds": self.rounds}, indent=2),
            encoding="utf-8",
        )

    def get_summary(self) -> str:
        if not self.rounds:
            return "暂无实验结果。"

        lines = ["", "当前实验摘要:"]
        for item in self.rounds:
            metrics = item.get("metrics", {})
            config = item.get("config", {})
            lines.append(
                "  - Round {round_id}: R2={r2:.3f}, RMSE={rmse:.3f}, window={window}, lag={lag}, alpha={alpha:.3f}, l1_ratio={l1:.3f}".format(
                    round_id=item.get("round", "?"),
                    r2=metrics.get("test_r2", 0.0),
                    rmse=metrics.get("test_rmse", 0.0),
                    window=config.get("window_size", "?"),
                    lag=config.get("use_lag_feature", "?"),
                    alpha=config.get("alpha", 0.0),
                    l1=config.get("l1_ratio", 0.0),
                )
            )

        best_round = max(self.rounds, key=lambda item: item.get("metrics", {}).get("test_r2", float("-inf")))
        best_metrics = best_round.get("metrics", {})
        lines.append(
            "最优轮次: Round {round_id} (test_r2={r2:.3f}, test_rmse={rmse:.3f})".format(
                round_id=best_round.get("round", "?"),
                r2=best_metrics.get("test_r2", 0.0),
                rmse=best_metrics.get("test_rmse", 0.0),
            )
        )
        return "\n".join(lines)
