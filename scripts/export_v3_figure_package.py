"""Export a versioned, auditable PNG package for the v3 presentation figures."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import presentation_plots


RENDER_SCRIPT = ROOT / "scripts" / "render_v3_presentation_figures.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_reconstructed_predictions(csv_path: Path) -> list[dict]:
    frame = pd.read_csv(csv_path, encoding="utf-8-sig")
    return [
        {"run_id": run_id, "frame": sub_frame.reset_index(drop=True)}
        for run_id, sub_frame in frame.groupby("run_id", sort=False)
    ]


def _write_daily_error_delta(package_dir: Path, predictions: list[dict]) -> Path:
    models = presentation_plots._build_v3_test_models(predictions)
    rows = []
    for family, frame in models.items():
        base_error = (frame["y_true"] - frame["y_base"]).abs()
        treatment_error = (frame["y_true"] - frame["y_treat"]).abs()
        delta = base_error - treatment_error
        for index, value in frame.iterrows():
            rows.append(
                {
                    "issue_date": value["issue_date"].strftime("%Y-%m-%d"),
                    "model_family": family,
                    "y_true": float(value["y_true"]),
                    "y_pred_baseline": float(value["y_base"]),
                    "y_pred_shadow": float(value["y_treat"]),
                    "abs_error_baseline": float(base_error.loc[index]),
                    "abs_error_shadow": float(treatment_error.loc[index]),
                    "abs_error_delta": float(delta.loc[index]),
                }
            )
    output = package_dir / "v3_daily_abs_error_delta.csv"
    pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8-sig")
    return output


def _read_model_parameters(csv_path: Path) -> dict:
    frame = pd.read_csv(csv_path, encoding="utf-8-sig")
    result = {}
    for _, row in frame.iterrows():
        run_id = str(row["run_id"])
        l1_ratio = row["l1_ratio"]
        if pd.isna(l1_ratio) or str(l1_ratio).strip() in ("", "—", "-"):
            l1_value = None
        else:
            l1_value = float(l1_ratio)
        result[run_id] = {
            "model_family": str(row["模型家族"]),
            "alpha": float(row["alpha"]),
            "l1_ratio": l1_value,
            "param_search_mode": str(row["选参模式"]),
            "search_completed": str(row.get("搜索完整", "未记录")),
            "boundary_hit": str(row.get("边界命中", "未记录")),
            "solver_converged": str(row.get("求解器收敛", "未记录")),
            "n_train": int(row["n_train"]),
            "n_test": int(row["n_test"]),
            "feature_columns": [str(item) for item in str(row["特征列"]).split("、")],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package-version",
        default="1.0.0",
        help="version of this figure package",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="output package directory",
    )
    parser.add_argument(
        "--session",
        default=(
            ROOT
            / "runtime"
            / "live_session"
            / "sessions"
            / "session_20261015_132955_705440"
        ),
        help="live session root",
    )
    parser.add_argument(
        "--round",
        default="round_02",
        help="round directory",
    )
    parser.add_argument(
        "--experiment",
        default="E_R02_03",
        help="experiment id",
    )
    args = parser.parse_args()

    package_dir = (
        Path(args.output)
        if args.output
        else ROOT / "artifacts" / f"v3_figure_package_v{args.package_version}"
    )
    figures_dir = package_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    session = Path(args.session)
    archive_dir = (
        session
        / "results"
        / args.round
        / "archive"
        / f"{args.experiment}_r1"
    )
    metrics = json.loads(
        (archive_dir / "metrics.json").read_text(encoding="utf-8")
    )

    subprocess.run(
        [
            sys.executable,
            str(RENDER_SCRIPT),
            "--session",
            str(session),
            "--round",
            str(args.round),
            "--experiment",
            str(args.experiment),
            "--output",
            str(figures_dir),
        ],
        check=True,
    )

    predictions = _load_reconstructed_predictions(
        figures_dir / "v3_prediction_reconstruction.csv"
    )
    test_start = metrics.get("test_start")
    models = presentation_plots._build_v3_test_models(predictions)
    test_end = (
        models["ridge"]["issue_date"].max().strftime("%Y-%m-%d")
        if test_start
        else None
    )
    if test_start is None:
        test_start = (
            models["ridge"]["issue_date"].min().strftime("%Y-%m-%d")
        )

    combined_path = presentation_plots.compose_v3_combined_overview(
        [
            ("A. 迭代评估段 RMSE 增益与 95% CI", figures_dir / "v3_gain_ci.png"),
            ("B. 逐日绝对误差变化", figures_dir / "v3_daily_abs_error_delta.png"),
            ("C. 迭代评估段预测对照", figures_dir / "v3_test_prediction_comparison.png"),
            ("D. 完整日历概览", figures_dir / "v3_appendix_full_calendar_overview.png"),
        ],
        figures_dir / "v3_combined_overview.png",
        run_id=f"{args.experiment}_r1",
        test_start=test_start,
        test_end=test_end,
    )
    daily_csv_path = _write_daily_error_delta(package_dir, predictions)
    model_parameters_path = package_dir / "v3_model_parameters.csv"
    (figures_dir / "v3_appendix_parameters.csv").replace(model_parameters_path)

    png_files = sorted(figures_dir.glob("*.png"))
    figures = []
    roles = {
        "v3_gain_ci.png": "主图：迭代评估段 RMSE 增益与 95% CI",
        "v3_daily_abs_error_delta.png": "新增：逐日绝对误差变化",
        "v3_test_prediction_comparison.png": "主图：迭代评估段预测与真实值",
        "v3_combined_overview.png": "组合版：2x2 核心图总览",
        "v3_appendix_full_calendar_overview.png": "附录：完整日历经览",
        "v3_appendix_training_quality.png": "附录：训练拟合与质量审计",
        "v3_appendix_parameters.png": "附录：参数与诊断",
    }
    for path in png_files:
        figures.append(
            {
                "file": f"figures/{path.name}",
                "sha256": _sha256(path),
                "role": roles.get(path.name, "附属图"),
            }
        )

    manifest = {
        "package_version": args.package_version,
        "figure_format_version": "v3-fig/1.0",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "experiment": {
            "session_root": str(session),
            "round_id": args.round,
            "experiment_id": args.experiment,
            "run_id": f"{args.experiment}_r1",
            "test_start": test_start,
            "test_end": test_end,
            "split_label": "dev_backtest",
        },
        "figure_format_spec": {
            "dpi": 200,
            "background": "white",
            "font_family": ["Microsoft YaHei", "SimHei", "Microsoft JhengHei"],
            "date_axis": "真实日历日期，缺失日期不压缩",
            "no_smoothing": True,
            "no_interpolation": True,
            "no_uncomputed_prediction_error_bars": True,
            "series_style": {
                "truth": "深灰实线/圆点 (#374151)",
                "baseline": "蓝色虚线 (#1d4ed8)",
                "shadow_treatment": "橙色实线 (#ea580c)",
            },
            "positive_delta_meaning": "加入日影后指标改善",
        },
        "figures": figures,
        "daily_abs_error_delta": {
            "file": "v3_daily_abs_error_delta.csv",
            "formula": "|y_true - y_pred_baseline| - |y_true - y_pred_shadow|",
            "positive_meaning": "当日加入日影后绝对误差更小",
            "sha256": _sha256(daily_csv_path),
        },
        "model_parameters": _read_model_parameters(model_parameters_path),
    }
    (package_dir / "version.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    summary_lines = []
    for _, row in pd.read_csv(
        model_parameters_path, encoding="utf-8-sig"
    ).iterrows():
        l1_text = ""
        if not (
            pd.isna(row["l1_ratio"])
            or str(row["l1_ratio"]).strip() in ("", "—", "-")
        ):
            l1_text = f", l1_ratio={row['l1_ratio']}"
        summary_lines.append(
            f"- {row['run_id']}: {row['模型家族']} alpha={row['alpha']}"
            f"{l1_text}, 选参={row['选参模式']}, "
            f"搜索完整={row['搜索完整']}, "
            f"边界命中={row['边界命中']}, "
            f"求解器收敛={row['求解器收敛']}"
        )
    model_summary = "\n".join(summary_lines)
    readme = f"""# 逐影 v3 图包 v{args.package_version}

生成时间：{manifest['generated_at']}

来源运行：{args.experiment}_r1，轮次 round_02，迭代评估段 {test_start} 至 {test_end}，
迭代评估共同样本 n={len(models['ridge'])}（split_label=dev_backtest），
预测提前量由归档配置决定。

## 图片输出格式版本

- figure_format_version：v3-fig/1.0
- 画布：PNG，dpi=200，白底
- 字体：Microsoft YaHei / SimHei / Microsoft JhengHei
- 日期轴：真实日历日期，缺失日期不压缩、不插值
- 预测只加误差棒：只有实际计算过预测区间时才会绘制
- 主图配色：真实值深灰，基线与加入日影分别蓝、橙
- 逐日绝对误差变化：|实际-基线| − |实际-日影|，正值表示当日日影误差更小

## 当前 v3 闭环模型参数

{model_summary}

## 文件清单

- figures/v3_gain_ci.png
- figures/v3_test_prediction_comparison.png
- figures/v3_daily_abs_error_delta.png（新增）
- figures/v3_combined_overview.png（新增组合版）
- figures/v3_appendix_full_calendar_overview.png
- figures/v3_appendix_training_quality.png
- figures/v3_appendix_parameters.png / .csv
- v3_daily_abs_error_delta.csv
- v3_model_parameters.csv
- version.json

所有图均从同一运行 E_R02_03_r1 的归档预测与指标计算，指标不来自 LLM 文本。
"""
    (package_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"figure package -> {package_dir}")
    print(f"combined overview -> {combined_path}")
    print(f"model parameters -> {model_parameters_path}")


if __name__ == "__main__":
    main()
