"""Freeze the reproducibility conclusion and main cross-window figure.

This script is intentionally read-only with respect to the closed loop:
it never retrains, never expands the diagnostic grid, and never writes to
hypothesis state.  It only turns the frozen outer-backtest artifacts into the
single comparison the defense page will focus on (`mag -> mag_both`,
horizon=3, auto) plus a reproducible copy of the "not reproduced" conclusion.

Outputs written under ``artifacts/legacy_v3_migration/outer_backtest/``:

- ``outer_backtest_reproducibility_conclusion.json``
- ``outer_backtest_reproducibility_conclusion.md``
- ``outer_backtest_cross_window_main_gain_ci.png``
- ``outer_backtest_cross_window_main_gain_ci.csv``
- ``outer_backtest_cross_window_main_gain_ci.json``
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ART_DIR = ROOT / "artifacts" / "legacy_v3_migration"
OUT_DIR = ART_DIR / "outer_backtest"

plt.switch_backend("Agg")
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "WenQuanYi Micro Hei",
    "Arial Unicode MS",
]
plt.rcParams["axes.unicode_minus"] = False

DEV_GAIN_PATH = ART_DIR / "legacy_v3_migration_gain_comparison.csv"
DEV_SUMMARY_PATH = ART_DIR / "legacy_v3_migration_summary.csv"
OUTER_GAIN_PATH = OUT_DIR / "outer_backtest_gain_comparison.csv"
GAP_SUMMARY_PATH = OUT_DIR / "outer_backtest_window_gap_summary.csv"

COMPARISON = {
    "horizon_days": 3,
    "mode": "auto",
    "baseline_variant": "mag",
    "treatment_variant": "mag_both",
}


def _read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    return frame


def dev_reference() -> dict[str, Any]:
    gain = _read_csv(DEV_GAIN_PATH)
    summary = _read_csv(DEV_SUMMARY_PATH)
    row = gain[
        (gain["horizon_days"] == COMPARISON["horizon_days"])
        & (gain["mode"] == COMPARISON["mode"])
        & (gain["baseline_variant"] == COMPARISON["baseline_variant"])
        & (gain["treatment_variant"] == COMPARISON["treatment_variant"])
    ]
    if len(row) != 1:
        raise RuntimeError(f"dev comparison row not unique: {len(row)}")
    row = row.iloc[0]
    baseline = summary[summary["run_id"] == row["baseline_run_id"]].iloc[0]
    treatment = summary[summary["run_id"] == row["treatment_run_id"]].iloc[0]
    return {
        "evaluation_window": "2021-08-26..2021-09-29",
        "horizon_days": int(row["horizon_days"]),
        "mode": str(row["mode"]),
        "comparison": f"{row['baseline_variant']} -> {row['treatment_variant']}",
        "run_pair": f"{row['baseline_run_id']} -> {row['treatment_run_id']}",
        "n_paired_samples": int(row["n_paired_samples"]),
        "baseline_rmse": round(float(baseline["eval_rmse"]), 6),
        "treatment_rmse": round(float(treatment["eval_rmse"]), 6),
        "point_gain_rmse": round(float(row["point_gain_rmse"]), 6),
        "rmse_ci_95": [
            round(float(row["rmse_ci_low"]), 6),
            round(float(row["rmse_ci_high"]), 6),
        ],
        "stable_rmse": bool(row["stable_rmse"]),
        "point_gain_pearson_r": round(float(row["point_gain_pearson_r"]), 6),
        "pearson_r_ci_95": [
            round(float(row["pearson_r_ci_low"]), 6),
            round(float(row["pearson_r_ci_high"]), 6),
        ],
        "note": "这是迭代评估段中的单组结果，不是独立盲测。",
    }


def _missing_reason(missing_fields: dict[str, int]) -> str:
    names = [field.split("@", 1)[0] for field in missing_fields]
    ordered = ["p2_y中心值", "p1_x中心值", "p1_y中心值", "p2_x中心值"]
    display = {
        "p2_y中心值": "p2_y 中心值",
        "p1_x中心值": "p1_x 中心值",
        "p1_y中心值": "p1_y 中心值",
        "p2_x中心值": "p2_x 中心值",
    }
    names = sorted(
        set(names),
        key=lambda name: (
            ordered.index(name) if name in ordered else len(ordered),
            name,
        ),
    )
    label = "、".join(display.get(name, name) for name in names)
    return f"{label}部分缺失，被真实日历共同样本剔除"


def outer_followup() -> list[dict[str, Any]]:
    gain = _read_csv(OUTER_GAIN_PATH)
    gaps = _read_csv(GAP_SUMMARY_PATH)
    rows = []
    for fold in ("f1", "f2"):
        row = gain[
            (gain["fold_id"] == fold)
            & (gain["horizon_days"] == COMPARISON["horizon_days"])
            & (gain["mode"] == COMPARISON["mode"])
            & (gain["baseline_variant"] == COMPARISON["baseline_variant"])
            & (gain["treatment_variant"] == COMPARISON["treatment_variant"])
        ]
        if len(row) != 1:
            raise RuntimeError(f"{fold} target comparison row not unique")
        row = row.iloc[0]
        gap = gaps[
            (gaps["fold_id"] == fold)
            & (gaps["horizon_days"] == COMPARISON["horizon_days"])
        ].iloc[0]
        missing_fields = json.loads(gap["missing_fields_json"])
        rows.append(
            {
                "fold_id": fold,
                "planned_window": (
                    f"{gap['planned_start']}..{gap['planned_end']}"
                ),
                "actual_common_window": (
                    f"{gap['actual_start']}..{gap['actual_end']}"
                ),
                "n_paired_samples": int(row["n_paired_samples"]),
                "horizon_days": int(row["horizon_days"]),
                "mode": str(row["mode"]),
                "point_gain_rmse": round(float(row["point_gain_rmse"]), 6),
                "rmse_ci_95": [
                    round(float(row["rmse_ci_low"]), 6),
                    round(float(row["rmse_ci_high"]), 6),
                ],
                "stable_rmse": bool(row["stable_rmse"]),
                "missing_reason": _missing_reason(missing_fields),
                "reproduces_dev_signal": False,
            }
        )
    return rows


def other_noncrossing_results() -> list[dict[str, Any]]:
    gain = _read_csv(OUTER_GAIN_PATH)
    rows = []
    for row in gain[gain["stable_rmse"] == True].itertuples():  # noqa: E712
        rows.append(
            {
                "fold_id": str(row.fold_id),
                "horizon_days": int(row.horizon_days),
                "mode": str(row.mode),
                "comparison": (
                    f"{row.baseline_variant} -> {row.treatment_variant}"
                ),
                "point_gain_rmse": round(float(row.point_gain_rmse), 6),
                "rmse_ci_95": [
                    round(float(row.rmse_ci_low), 6),
                    round(float(row.rmse_ci_high), 6),
                ],
                "direction": "negative" if row.point_gain_rmse < 0 else "positive",
                "note": str(row.interpretation),
            }
        )
    return rows


def build_conclusion() -> dict[str, Any]:
    return {
        "schema_version": "legacy-v3-outer-backtest-conclusion-v1",
        "label": "dev mag_both reproducibility conclusion",
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "question": (
            "dev 段 35 天中唯一不跨零的 mag_both RMSE 改善，"
            "能否在未参与 dev 探索的早段窗口复现？"
        ),
        "verdict": "not_reproduced",
        "dev_reference": dev_reference(),
        "outer_followup": outer_followup(),
        "other_noncrossing_results": other_noncrossing_results(),
        "boundaries": [
            (
                "这是补充时间回测，不是独立盲测：特征定义、网格、基线、"
                "提前期与分类规则仍来自 dev 段探索。"
            ),
            (
                "mag_both 仅表示未复现 dev 中的不跨零改善，"
                "不能宣称跨时段稳定，也不能宣称已证伪日影物理作用。"
            ),
            (
                "f1/f2 各有真实日历缺失，共同样本分别降为 17 天与 23 天；"
                "缺口已保留在 window_coverage 中。"
            ),
            (
                "24 组比较未做多重比较调整；两个额外不跨零结果方向相反，"
                "应视为探索性信息。"
            ),
            (
                "stable_rmse=true 只表示该窗口内的块 bootstrap 区间未跨零，"
                "不等同于科学意义上的稳定性验证通过。"
            ),
            "未测量的 Pearson 不填零，仍遵循未计算/不可计算规则。",
        ],
        "suggested_defense_wording": (
            "在冻结的迭代评估段中，双方向日影方案出现一次小幅 RMSE 改善；"
            "预登记的早段补充回测中，同一设置的两个窗口均未复现该改善，"
            "且两个窗口各出现了一个方向相反、未做多重比较调整的不跨零结果。"
            "因此该发现当前应表述为局部、待重复，"
            "不能表述为稳定的跨时段增量。"
        ),
        "source_files": {
            "summary": "outer_backtest_summary.csv",
            "daily_predictions": "outer_backtest_daily.csv",
            "gain_comparison": "outer_backtest_gain_comparison.csv",
            "window_coverage": "outer_backtest_window_coverage.csv",
            "audit": "outer_backtest_audit.json",
            "manifest": "outer_backtest_manifest.json",
            "gain_ci_plot": "outer_backtest_gain_ci_by_fold.png",
            "cross_window_main_plot": (
                "outer_backtest_cross_window_main_gain_ci.png"
            ),
            "cross_window_main_csv": (
                "outer_backtest_cross_window_main_gain_ci.csv"
            ),
            "cross_window_main_json": (
                "outer_backtest_cross_window_main_gain_ci.json"
            ),
        },
    }


def write_markdown(conclusion: dict[str, Any]) -> str:
    dev = conclusion["dev_reference"]
    followups = conclusion["outer_followup"]
    extra = conclusion["other_noncrossing_results"]
    lines = [
        "# 外层补充回测结论：dev 的 mag_both 改善未复现",
        "",
        f'生成时间：{datetime.now().strftime("%Y-%m-%d")}',
        "",
        "## 要核验的问题",
        "",
        "冻结 35 天迭代评估段后，dev 中唯一 RMSE CI 不跨零的设置是：",
        "",
        "| 项目 | 值 |",
        "| --- | --- |",
        "| 比较 | mag -> mag_both |",
        "| 提前期 / 模式 | 提前 3 天 / 训练内自动选参 |",
        f"| 共同样本 | n={dev['n_paired_samples']} |",
        f"| 基线 RMSE | {dev['baseline_rmse']:.3f} km/s |",
        f"| 加入双方向日影 RMSE | {dev['treatment_rmse']:.3f} km/s |",
        f"| RMSE 点增益 | {dev['point_gain_rmse']:.3f} km/s |",
        f"| 95% 配对块 bootstrap CI | [{dev['rmse_ci_95'][0]:.3f}, {dev['rmse_ci_95'][1]:.3f}] |",
        f"| Pearson r 点增益 | {dev['point_gain_pearson_r']:.3f} |",
        f"| Pearson r 95% CI | [{dev['pearson_r_ci_95'][0]:.3f}, {dev['pearson_r_ci_95'][1]:.3f}] |",
        "",
        "该结果本身只是迭代评估段内的一处局部证据，需要到未参与的早段窗口核验能否重复。",
        "",
        "## 预登记外层窗口与共同样本",
        "",
        "| 窗口 | 计划区间 | 实际共同区间 | 样本数 | 缺失原因 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in followups:
        lines.append(
            f"| {row['fold_id']} | {row['planned_window']} | "
            f"{row['actual_common_window']} | {row['n_paired_samples']} | "
            f"{row['missing_reason']} |"
        )
    lines += [
        "",
        "dev 段 2021-08-26 起完全排除，未进入训练或选参。",
        "",
        "## mag_both 的重复结果",
        "",
        "| 窗口 | 提前期 | RMSE 点增益 | RMSE 95% CI | 是否复现 dev 信号 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in followups:
        low, high = row["rmse_ci_95"]
        lines.append(
            f"| {row['fold_id']} | {row['horizon_days']} 天 | "
            f"{row['point_gain_rmse']:.3f} | [{low:.3f}, {high:.3f}] | 否 |"
        )
    lines += [
        "",
        "两个窗口的 RMSE CI 均跨零，`stable_rmse=False`。结论："
        "dev 中唯一不跨零的 mag_both 改善未在早段复现。",
        "",
        "## 额外发现（探索性）",
        "",
        "24 组比较中另有 2 组 RMSE CI 不跨零，方向相反：",
        "",
        "| 窗口 | 比较 | RMSE 点增益 | 95% CI | 方向 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in extra:
        low, high = row["rmse_ci_95"]
        lines.append(
            f"| {row['fold_id']}, 提前 {row['horizon_days']} 天, " 
            f"{'自动选参' if row['mode'] == 'auto' else '固定参数'} | "
            f"{row['comparison']} | {row['point_gain_rmse']:.3f} | "
            f"[{low:.3f}, {high:.3f}] | "
            f"{'负向' if row['direction'] == 'negative' else '正向'} |"
        )
    lines += [
        "",
        "未做多重比较调整，不能据此宣称存在稳定规律。",
        "",
        "## 可表述的答辩口径",
        "",
        f"> {conclusion['suggested_defense_wording']}",
        "",
        "## 边界",
        "",
    ]
    for boundary in conclusion["boundaries"]:
        lines.append(f"- {boundary}")
    lines += [
        "",
        "对应机器可读文件：`outer_backtest_reproducibility_conclusion.json`",
        "",
    ]
    return "\n".join(lines)


def write_cross_window_plot(
    conclusion: dict[str, Any], plot_path: Path, csv_path: Path, info_path: Path
) -> None:
    dev = conclusion["dev_reference"]
    rows = [
        (
            "dev",
            "dev（迭代评估段）",
            dev["evaluation_window"].replace("..", " ~ "),
            dev["n_paired_samples"],
            dev["point_gain_rmse"],
            dev["rmse_ci_95"],
            dev["stable_rmse"],
        )
    ]
    for row in conclusion["outer_followup"]:
        rows.append(
            (
                row["fold_id"],
                row["fold_id"].upper() + "（补充回测）",
                row["actual_common_window"].replace("..", " ~ "),
                row["n_paired_samples"],
                row["point_gain_rmse"],
                row["rmse_ci_95"],
                row["stable_rmse"],
            )
        )

    labels = [
        f"{name}\n{window} · n={n}"
        for _, name, window, n, _, _, _ in rows
    ]
    centers = np.array([row[4] for row in rows])
    lows = np.array([row[5][0] for row in rows])
    highs = np.array([row[5][1] for row in rows])
    stable = [row[6] for row in rows]
    y_pos = np.arange(len(rows))[::-1]

    fig, ax = plt.subplots(figsize=(9.0, 4.4), constrained_layout=True)
    for yi, row in zip(y_pos, rows):
        low, high = row[5]
        color = "#b91c1c" if row[0] == "dev" else "#1f2937"
        line_style = "-" if row[6] else ":"
        ax.plot(
            [low, high],
            [yi, yi],
            color=color,
            lw=2.4,
            ls=line_style,
            solid_capstyle="round",
        )
        ax.plot(
            row[4],
            yi,
            marker="o",
            color=color,
            ms=8,
            mec="white",
            mew=1.0,
            zorder=3,
        )
        note = "CI 未跨零" if row[6] else "CI 跨零 · 未复现 dev"
        ax.annotate(
            f"{row[4]:+.3f}  [{low:.3f}, {high:.3f}]\n{note}",
            xy=(high, yi),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=color,
        )
    ax.axvline(0, color="#9ca3af", lw=1.4, ls="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("RMSE 增益（km/s；正值 = 加入双方向日影后改善）", fontsize=11)
    ax.set_title(
        "同一设置在不同时间窗的 RMSE 增益：mag → mag_both，提前 3 天，auto",
        fontsize=12,
    )
    ax.grid(alpha=0.25, axis="x")
    ax.tick_params(axis="x", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(
        min(0.0, float(np.min(lows))) - 0.45,
        max(0.0, float(np.max(highs))) + 1.05,
    )
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    main_rows = [
        {
            "window": row[0],
            "window_label": row[1],
            "calendar_window": row[2],
            "n_paired_samples": row[3],
            "point_gain_rmse": row[4],
            "rmse_ci_low": row[5][0],
            "rmse_ci_high": row[5][1],
            "stable_rmse": row[6],
        }
        for row in rows
    ]
    pd.DataFrame(main_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    info = {
        "schema_version": "legacy-v3-cross-window-main-plot-v1",
        "figure": str(plot_path.name),
        "comparison": "mag -> mag_both",
        "horizon_days": 3,
        "mode": "auto",
        "metric": "RMSE paired gain, km/s",
        "positive_meaning": "加入双方向日影后误差改善",
        "ci_method": "配对日历块 bootstrap 95% CI",
        "bootstrap": {"block_size": 7, "n_draws": 2000, "seed": 20261017},
        "rows": main_rows,
        "note": (
            "主图仅保留 mag->mag_both、h=3、auto；"
            "其余 24 组比较放在 outer_backtest_gain_ci_by_fold.png 备份页。"
        ),
        "source_files": [
            "outer_backtest_gain_comparison.csv",
            "outer_backtest_reproducibility_conclusion.json",
        ],
    }
    from core.json_safe import json_dumps

    info_path.write_text(json_dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    conclusion = build_conclusion()
    conclusion_path = OUT_DIR / "outer_backtest_reproducibility_conclusion.json"
    md_path = OUT_DIR / "outer_backtest_reproducibility_conclusion.md"
    plot_path = OUT_DIR / "outer_backtest_cross_window_main_gain_ci.png"
    csv_path = OUT_DIR / "outer_backtest_cross_window_main_gain_ci.csv"
    info_path = OUT_DIR / "outer_backtest_cross_window_main_gain_ci.json"

    from core.json_safe import json_dumps

    conclusion_path.write_text(
        json_dumps(conclusion, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(write_markdown(conclusion) + "\n", encoding="utf-8")
    write_cross_window_plot(conclusion, plot_path, csv_path, info_path)
    print(
        json_dumps(
            {
                "status": "frozen",
                "verdict": conclusion["verdict"],
                "target": COMPARISON,
                "dev": conclusion["dev_reference"]["rmse_ci_95"],
                "followups": [
                    {
                        "fold": row["fold_id"],
                        "ci": row["rmse_ci_95"],
                        "reproduces_dev_signal": row["reproduces_dev_signal"],
                    }
                    for row in conclusion["outer_followup"]
                ],
                "outputs": [
                    str(path)
                    for path in (
                        conclusion_path,
                        md_path,
                        plot_path,
                        csv_path,
                        info_path,
                    )
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
