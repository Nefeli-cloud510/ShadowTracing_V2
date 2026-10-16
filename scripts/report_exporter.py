"""Export closed-loop round reports as styled HTML or PDF.

The exporter is deliberately read-only: it only reads persisted state under
``runtime/live_session/current`` and renders an HTML document with embedded
images.  PDF generation uses a local headless Chromium/Edge when available and
falls back to an HTML attachment otherwise, so the feature never blocks the
live workflow loop.
"""

from __future__ import annotations

import base64
import html
import json
import mimetypes
import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
LIVE_ROOT = REPO_ROOT / "runtime" / "live_session" / "current"
STATE_DIR = LIVE_ROOT / "state"

STATUS_LABELS = {
    "active": "活跃",
    "observing": "待观察",
    "draft": "草稿",
    "converged": "收敛",
    "pruned": "剪枝",
    "pending": "待定",
    "weakened": "已削弱",
    "newly_split": "本轮新增",
    "completed": "已完成",
    "running": "运行中",
    "pending_approval": "待审批",
    "failed": "失败",
}

STATUS_COLORS = {
    "active": "#1f5f9e",
    "observing": "#a87a1e",
    "draft": "#687080",
    "converged": "#177245",
    "pruned": "#a63a2b",
    "weakened": "#a87a1e",
    "newly_split": "#275b94",
    "completed": "#177245",
    "running": "#1f5f9e",
}

EXPERIMENT_TYPE_LABELS = {
    "incremental_prediction": "增量预测",
    "model_sensitivity": "模型敏感性",
    "baseline_benchmark": "单臂基准",
    "robustness_diagnostic": "稳健性诊断",
    "temporal_sensitivity": "时间轴敏感性",
}

MODEL_SENSITIVITY_LABELS = {
    "stable_positive": "稳定正向增益",
    "stable_negative": "稳定负向增益",
    "no_stable_gain": "未检出稳定增益（CI 跨零）",
    "identical_predictions": "两组预测完全一致（无预测差异）",
    "insufficient_blocks": "CI 不可用 / 有效时间块不足",
    "point_positive": "点估计正向，无稳定证据",
    "point_negative": "点估计负向，无稳定证据",
    "point_zero": "点估计接近零",
    "not_evaluated": "未评估",
}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                return value
    except Exception:
        pass
    return {}


def _resolve_live_root() -> Path:
    """Resolve the active persisted session when the legacy ``current`` root is absent."""
    if LIVE_ROOT.is_dir():
        return LIVE_ROOT
    pointer = _load_json(REPO_ROOT / "runtime" / "live_session" / "sessions" / "active_session.json")
    session_name = str(pointer.get("session") or "").strip()
    if session_name:
        candidate = REPO_ROOT / "runtime" / "live_session" / "sessions" / session_name
        if candidate.is_dir():
            return candidate
    candidates = sorted(
        (REPO_ROOT / "runtime" / "live_session" / "sessions").glob("session_*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else LIVE_ROOT


LIVE_ROOT = _resolve_live_root()
STATE_DIR = LIVE_ROOT / "state"


def _code_version() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return "local-workspace"
        version = result.stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if status.returncode == 0 and status.stdout.strip():
            version = f"{version}-dirty"
        return version
    except Exception:
        return "local-workspace"


def _data_manifest_versions() -> list[str]:
    versions: list[str] = []
    seen: set[tuple[str, str]] = set()
    if not (LIVE_ROOT / "results").is_dir():
        return versions
    for manifest_path in (LIVE_ROOT / "results").rglob("data_manifest.json"):
        manifest = _load_json(manifest_path)
        version = str(manifest.get("data_version") or "unknown")
        target = str(manifest.get("target_column") or "")
        key = (version, target)
        if key in seen:
            continue
        seen.add(key)
        label = version
        if target:
            label = f"{label}（目标：{target}）"
        versions.append(label)
    return versions


def _invalidation_experiment_ids(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    ids: list[str] = []
    for item in value:
        if isinstance(item, dict):
            sid = item.get("experiment_id")
        else:
            sid = item
        if sid:
            ids.append(str(sid))
    return ids


def _invalidation_banner(invalidation: dict[str, Any]) -> str:
    """Render a compact evidence-invalidation banner shared by report sections."""
    if not isinstance(invalidation, dict):
        return ""
    if invalidation.get("status") != "evidence_invalidated":
        return ""
    exp_ids = "、".join(_invalidation_experiment_ids(invalidation.get("invalidated_experiments"))) or "--"
    return (
        '<div class="warning-banner"><strong>本节能科学结论已阻断</strong>：'
        f"{_esc(invalidation.get('root_cause') or '数据列错位导致目标变量错误')}"
        f"受影响实验：{_esc(exp_ids)}。以下仅作审计复现，不再进入正式结论与下一轮规划。</div>"
    )


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "--"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:
        return "--"
    if number == 0:
        return "0"
    if abs(number) >= 1000 or abs(number) < 0.0001:
        return f"{number:.4e}"
    return f"{number:.{digits}f}"


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _match_text(value: Any) -> str:
    if value is None:
        return "未判断"
    if value is True:
        return "匹配"
    if value is False:
        return "偏离"
    if isinstance(value, str) and value.lower() == "partial":
        return "部分匹配"
    return _to_text(value)


def _to_text(value: Any, max_len: int = 0) -> str:
    if value is None:
        return "—"
    if isinstance(value, str):
        text = value.strip()
    elif isinstance(value, bool):
        text = "是" if value else "否"
    elif isinstance(value, (int, float)):
        text = _fmt(value)
    elif isinstance(value, list):
        parts = [_to_text(item, max_len) for item in value if item not in (None, "", [], {})]
        text = "；".join(parts)
    elif isinstance(value, dict):
        parts = []
        for key, item in value.items():
            if item in (None, "", [], {}):
                continue
            parts.append(f"{key}: {_to_text(item, max_len)}")
        text = "；".join(parts)
    else:
        text = str(value)
    if max_len and len(text) > max_len:
        return text[:max_len] + "…"
    return text


def _esc(value: Any) -> str:
    return html.escape(_to_text(value), quote=False)


def _resolve_within(root: Path, relative_path: str) -> Path:
    base = root.resolve()
    candidate = (root / str(relative_path).lstrip("/\\")).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise PermissionError("非法路径访问。") from exc
    return candidate


def _embed_image(relative_path: str) -> str:
    try:
        target = _resolve_within(LIVE_ROOT, relative_path)
        if not target.is_file():
            return ""
        mime_type, _ = mimetypes.guess_type(target.name)
        mime_type = mime_type or "image/png"
        data_uri = "data:%s;base64,%s" % (
            mime_type,
            base64.b64encode(target.read_bytes()).decode("ascii"),
        )
        return data_uri
    except Exception:
        return ""


def current_round() -> int:
    process = _load_json(STATE_DIR / "process.json")
    value = process.get("current_round")
    if isinstance(value, int) and value > 0:
        return value
    snapshot_dir = STATE_DIR / "round_snapshots"
    rounds = []
    if snapshot_dir.is_dir():
        for path in snapshot_dir.glob("round_*.json"):
            try:
                rounds.append(int(path.stem.rsplit("_", 1)[-1]))
            except (ValueError, IndexError):
                continue
    return max(rounds, default=1)


def load_round_bundle(round_number: int) -> dict[str, Any]:
    round_number = int(round_number)
    snapshot_path = STATE_DIR / "round_snapshots" / f"round_{round_number:02d}.json"
    snapshot = _load_json(snapshot_path)
    artifacts = snapshot.get("artifacts", {}) if isinstance(snapshot.get("artifacts"), dict) else {}
    round_dir = LIVE_ROOT / "results" / f"round_{round_number:02d}"

    def _artifact(name: str) -> dict[str, Any]:
        if isinstance(artifacts.get(name), dict):
            return artifacts[name]
        return _load_json(STATE_DIR / f"{name}.json")

    def _round_file(name: str) -> dict[str, Any]:
        if isinstance(snapshot.get(name), dict) and snapshot[name]:
            return snapshot[name]
        return _load_json(round_dir / f"{name}.json")

    bundle: dict[str, Any] = {
        "round": round_number,
        "snapshot": snapshot,
        "task": _artifact("task"),
        "data_dictionary": _artifact("data_dictionary"),
        "hypothesis_tree": _artifact("hypothesis_tree"),
        "uncertainties": _artifact("uncertainties"),
        "decision_log": _artifact("decision_log"),
        "candidate_experiments": _artifact("candidate_experiments"),
        "round_history": _artifact("round_history"),
        "experiment_memory": _artifact("experiment_memory"),
        "protocol": _round_file("protocol"),
        "evaluation": _round_file("evaluation_unified"),
        "result": _round_file("result_unified"),
    }
    has_any = bool(
        snapshot
        or bundle["protocol"]
        or bundle["evaluation"]
        or bundle["result"]
        or bundle["hypothesis_tree"]
    )
    bundle["available"] = has_any
    return bundle


def _decisions(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    return bundle.get("decision_log", {}).get("decisions", []) or []


def _last_decision(
    bundle: dict[str, Any],
    decision_type: str,
    *,
    round_id: int | None = None,
) -> dict[str, Any] | None:
    matches = [item for item in _decisions(bundle) if item.get("decision_type") == decision_type]
    if round_id is not None:
        exact = [item for item in matches if item.get("round_id") == round_id]
        if exact:
            return exact[-1]
    return matches[-1] if matches else None


_COMMIT_DECISION_TYPES = ("continue_next_round", "round_adjusted", "terminate_workflow")


def _latest_commit_decision(
    bundle: dict[str, Any],
    *,
    round_id: int | None = None,
) -> dict[str, Any] | None:
    """Return the newest actual human PI decision, preferring commit decisions over review requests."""
    matches = [
        item
        for item in _decisions(bundle)
        if item.get("decision_type") in _COMMIT_DECISION_TYPES
    ]
    if round_id is not None:
        matches = [item for item in matches if item.get("round_id") == round_id]
    if not matches:
        return None
    latest = matches[0]
    for item in matches[1:]:
        if str(item.get("timestamp") or "") >= str(latest.get("timestamp") or ""):
            latest = item
    return latest


def _commit_decision_label(decision: dict[str, Any] | None) -> str:
    if not decision:
        return "--"
    decision_type = decision.get("decision_type")
    if decision_type == "terminate_workflow":
        return "停止工作流"
    if decision_type == "round_adjusted":
        return "调整方向并进入下一轮"
    if decision_type == "continue_next_round":
        return "进入下一轮"
    return "--"


def _interpretation_source_tag(layer: dict[str, Any]) -> str:
    source = str(layer.get("interpretation_source") or "").strip().lower()
    if source in ("llm_interpreter", "real_llm", "llm", "llm_original"):
        return "真实 LLM 原文"
    return "程序规则回退"


def _human_touchpoints(bundle: dict[str, Any], round_number: int) -> list[str]:
    points: list[str] = []
    approved = _last_decision(bundle, "experiment_approved", round_id=round_number)
    if approved:
        when = str(approved.get("timestamp") or approved.get("details", {}).get("approved_at") or "").replace("T", " ")
        candidate = approved.get("details", {}).get("candidate_id")
        points.append(f"已批准候选实验 {candidate or '--'}（{when[:19]}）")
    confirmed = _last_decision(bundle, "hypothesis_tree_confirmed", round_id=round_number)
    if confirmed:
        points.append("人工 PI 已确认假设空间")
    reviewed = _last_decision(bundle, "round_review_requested", round_id=round_number)
    if reviewed:
        summary = reviewed.get("summary") or "轮次报告已生成"
        points.append(str(summary))
    return points


def _status_pill(status: Any) -> str:
    key = str(status or "").lower()
    label = STATUS_LABELS.get(key, key or "待定")
    color = STATUS_COLORS.get(key, "#687080")
    return (
        f'<span class="pill" style="border-color:{color};color:{color};'
        f'background:{color}14;">{html.escape(label, quote=False)}</span>'
    )


def _support_before_after(
    node: dict[str, Any], round_number: int, current_score: Any
) -> tuple[Any, Any]:
    entries = [
        entry
        for entry in (node.get("support_history") or [])
        if isinstance(entry, dict) and entry.get("round") is not None
    ]
    before: Any = None
    after: Any = None
    for entry in entries:
        try:
            item_round = int(entry["round"])
        except (TypeError, ValueError):
            continue
        score = entry.get("score")
        if score is None:
            continue
        if item_round < round_number:
            before = score
        elif item_round == round_number:
            after = score
    if after is None:
        after = current_score
    if before is None:
        before = entries[0].get("score") if entries else current_score
    return before, after


def _hypothesis_display_map(bundle: dict[str, Any]) -> dict[str, str]:
    """Map internal hypothesis ids to short display ids from the persisted tree."""
    mapping: dict[str, str] = {}
    for node in bundle.get("hypothesis_tree", {}).get("nodes") or []:
        if not isinstance(node, dict):
            continue
        internal_id = node.get("hypothesis_id")
        if not internal_id:
            continue
        display_id = node.get("display_hypothesis_id") or node.get("display_id")
        mapping[str(internal_id)] = str(display_id or internal_id)
    return mapping


def _short_hypothesis_label(
    bundle: dict[str, Any],
    raw_mapping: dict[str, str],
    hypothesis_id: str,
) -> str:
    if hypothesis_id in raw_mapping:
        return raw_mapping[hypothesis_id]
    for node in bundle.get("hypothesis_tree", {}).get("nodes") or []:
        if not isinstance(node, dict):
            continue
        if str(node.get("hypothesis_id")) == hypothesis_id:
            return str(node.get("display_hypothesis_id") or hypothesis_id)
    return hypothesis_id


def _related_hypothesis_text(
    bundle: dict[str, Any],
    raw_mapping: dict[str, str],
    value: Any,
) -> str:
    items: list[str] = []
    if isinstance(value, str):
        items = [part.strip() for part in value.replace("，", "、").split("、") if part.strip()]
    elif isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
    seen: set[str] = set()
    labels: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        labels.append(_short_hypothesis_label(bundle, raw_mapping, item))
    return "、".join(labels) or str(value if value is not None else "")


def _kv_table(rows: list[tuple[str, Any]]) -> str:
    body = "".join(
        f"<tr><th>{_esc(key)}</th><td>{_esc(value)}</td></tr>"
        for key, value in rows
        if value not in (None, "", [], {})
    )
    return f'<table class="kv-table">{body}</table>'


def _bullets(items: Iterable[Any], max_items: int | None = None) -> str:
    if items is None:
        return ""
    values = [item for item in items if item not in (None, "", [], {})]
    if max_items is not None:
        values = values[:max_items]
    if not values:
        return ""
    body = "".join(f"<li>{_esc(value)}</li>" for value in values)
    return f"<ul>{body}</ul>"


def _ordered_run_ids(run_ids: list[str]) -> list[str]:
    """Keep baseline/treatment first, then preserve the remaining run order."""
    ordered = [rid for rid in ("baseline", "treatment") if rid in run_ids]
    ordered += [rid for rid in run_ids if rid not in ordered]
    return ordered


def _quote(label: str, text: Any) -> str:
    return (
        f'<blockquote class="llm-quote"><span class="quote-label">{_esc(label)}</span>'
        f"{_esc(text)}</blockquote>"
    )


def _figure_caption(name: str) -> str:
    text = str(name).replace("_", " ")
    for suffix, label in (
        ("coefficients", "特征系数图"),
        ("scatter", "线性散点图"),
        ("timeseries", "时间轴预测对照图"),
        ("residual scatter", "残差散点图"),
        ("residual hist", "残差分布图"),
        ("shap summary", "SHAP 特征影响图"),
        ("shap bar", "SHAP 平均特征重要性图"),
        ("shap waterfall 0", "SHAP 单样本瀑布图"),
    ):
        marker = f" {suffix}"
        if text.endswith(marker):
            text = f"{text[:-len(marker)].strip()} · {label}"
            break
    return (
        text.replace("elasticnet ", "ElasticNet ")
        .replace("ridge ", "Ridge ")
        .replace("E_R", "实验 E_R")
    )


def _figure_grid(bundle: dict[str, Any], round_number: int) -> str:
    evaluation = bundle.get("evaluation", {})
    result = bundle.get("result", {})
    visualizations = evaluation.get("visualizations") or result.get("visualizations") or []
    runs = evaluation.get("runs") or result.get("runs") or []
    run_ids: list[str] = []
    run_label_by_id: dict[str, str] = {}
    for run in runs:
        if not isinstance(run, dict):
            continue
        rid = str(run.get("run_id") or run.get("name") or "")
        if not rid:
            continue
        run_ids.append(rid)
        run_label_by_id[rid] = str(run.get("name") or rid)
    figures: list[tuple[str, str, str | None]] = []
    seen: set[str] = set()
    if visualizations:
        for item in visualizations:
            path = item.get("path") if isinstance(item, dict) else None
            name = item.get("name") or (Path(path or "").stem) if isinstance(item, dict) else ""
            run_id = None
            lowered_name = str(name).lower()
            for rid in run_ids:
                if lowered_name == rid.lower() or lowered_name.startswith(rid.lower() + "_"):
                    run_id = rid
                    break
            if run_id is None and path:
                segments = [segment.lower() for segment in str(path).replace("\\", "/").split("/")]
                for rid in run_ids:
                    if rid.lower() in segments:
                        run_id = rid
                        break
                if run_id is None and "baseline" in segments:
                    run_id = "baseline"
                elif run_id is None and "treatment" in segments:
                    run_id = "treatment"
            if path and path not in seen:
                seen.add(str(path))
                figures.append((str(name), str(path), run_id))
    if not figures:
        run_dirs = list((LIVE_ROOT / "results" / f"round_{round_number:02d}").glob("E_*/"))
        for run_dir in run_dirs:
            if not run_dir.is_dir():
                continue
            for arm_dir in sorted(item for item in run_dir.iterdir() if item.is_dir()):
                for image in arm_dir.glob("*.png"):
                    if str(image) not in seen:
                        seen.add(str(image))
                        figures.append((image.stem, str(image.relative_to(LIVE_ROOT)), arm_dir.name))
    if not figures:
        return '<p class="muted">本轮未收集到可导出的结果图。</p>'

    def group_label(run_id: str | None) -> str:
        if run_id is None:
            return "其他结果图"
        if run_id == "baseline" or run_label_by_id.get(run_id) == "baseline":
            return "实验结果图 · 对照组"
        if run_id == "treatment" or run_label_by_id.get(run_id) == "treatment":
            return "实验结果图 · 实验组"
        return f"实验结果图 · {run_label_by_id.get(run_id, run_id)}"

    grouped: dict[str, list[tuple[str, str]]] = {}
    order: list[str] = []
    for name, path, run_id in figures:
        label = group_label(run_id)
        if label not in grouped:
            order.append(label)
            grouped[label] = []
        grouped[label].append((name, path))

    parts: list[str] = []
    for label in order:
        items = grouped[label]
        cells: list[str] = []
        for name, path in items:
            data_uri = _embed_image(path)
            if not data_uri:
                continue
            caption = _figure_caption(name)
            cells.append(
                f'<figure class="figure"><img src="{data_uri}" alt="{_esc(caption)}"/>'
                f"<figcaption>{_esc(caption)}</figcaption></figure>"
            )
        if cells:
            parts.append(f'<h4>{_esc(label)}</h4><div class="figure-grid">{"".join(cells)}</div>')
    return "".join(parts)


PRESENTATION_FIGURE_DEFS = (
    (
        "v3_gain_ci.png",
        "主图：迭代评估段 RMSE 增益与 95% CI（正值表示加入日影后改善）",
    ),
    (
        "v3_test_prediction_comparison.png",
        "主图：迭代评估段真实值与基线/加入日影预测对照",
    ),
    (
        "v3_daily_abs_error_delta.png",
        "附图：逐日绝对误差改善量（|基线误差| - |日影误差|）",
    ),
)


def _presentation_figures_html(
    round_number: int,
    experiment_id: str | None = None,
    *,
    include: Iterable[str] | None = None,
    compact: bool = False,
) -> str:
    """Return embedded core result charts for one round, generating if needed."""

    figure_dir = LIVE_ROOT / "exports" / "figures" / f"round_{round_number:02d}"
    names = tuple(include) if include is not None else tuple(
        name for name, _ in PRESENTATION_FIGURE_DEFS
    )
    expected_source: dict[str, Any] | None = None
    try:
        from scripts.report_figures import _resolve_archive

        archive_dir, metrics, resolved_experiment_id = _resolve_archive(
            LIVE_ROOT / "results" / f"round_{round_number:02d}",
            experiment_id,
        )
        expected_source = {
            "experiment_id": resolved_experiment_id,
            "run_id": str(
                metrics.get("run_id") or f"{resolved_experiment_id}_r1"
            ),
            "prediction_source": (
                f"results/round_{round_number:02d}/{resolved_experiment_id}/predictions"
            ),
            "forecast_horizon_days": int(
                metrics["runs"][0]["parameters"]["forecast_horizon_days"]
            ),
            "test_start": metrics.get("test_start"),
        }
    except Exception:
        expected_source = None

    manifest: dict[str, Any] = {}
    manifest_path = figure_dir / "report_chart_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            manifest = {}

    stale = bool(expected_source) and any(
        str(manifest.get(key)) != str(expected_source[key])
        for key in (
            "experiment_id",
            "run_id",
            "prediction_source",
            "forecast_horizon_days",
            "test_start",
        )
    )
    if stale or any(not (figure_dir / name).is_file() for name in names):
        try:
            from scripts.report_figures import render_round_core_charts

            render_round_core_charts(
                session_root=LIVE_ROOT,
                round_number=round_number,
                experiment_id=experiment_id,
                output_dir=figure_dir,
            )
        except Exception:
            return ""

    cells: list[str] = []
    for name, caption in PRESENTATION_FIGURE_DEFS:
        if name not in names:
            continue
        path = figure_dir / name
        if not path.is_file():
            continue
        relative = str(path.relative_to(LIVE_ROOT))
        data_uri = _embed_image(relative)
        if not data_uri:
            continue
        cells.append(
            f'<figure class="figure"><img src="{data_uri}" alt="{_esc(caption)}"/>'
            f"<figcaption>{_esc(caption)}</figcaption></figure>"
        )
    if not cells:
        return ""
    grid_class = "figure-grid figure-grid--compact" if compact else "figure-grid"
    return f'<div class="{grid_class}">{"".join(cells)}</div>'


def _main_question(bundle: dict[str, Any]) -> str:
    task = bundle.get("task", {})
    payload = task.get("payload", {})
    question = payload.get("research_question", {}).get("text")
    if not question:
        question = (
            bundle.get("hypothesis_tree", {}).get("root_question")
            or bundle.get("evaluation", {})
            .get("scientific", {})
            .get("three_layer_conclusion", {})
            .get("scientific_layer", {})
            .get("main_question")
        )
    return str(question or task.get("task_id") or "宇宙线日影物理机制研究")


def _model_name(bundle: dict[str, Any]) -> str:
    protocol_model = bundle.get("protocol", {}).get("model", {})
    if isinstance(protocol_model, dict) and protocol_model.get("name"):
        return str(protocol_model["name"])
    decisions = _decisions(bundle)
    for item in reversed(decisions):
        details = item.get("details", {})
        if isinstance(details, dict) and details.get("model"):
            return str(details["model"])
    return "qwen3.8-flash"


def _round_stage(bundle: dict[str, Any], round_number: int) -> str:
    snapshot = bundle.get("snapshot")
    if snapshot:
        return str(snapshot.get("closure", "snapshot") or "snapshot")
    process = _load_json(STATE_DIR / "process.json")
    stage = process.get("current_stage") or process.get("current_phase") or ""
    return str(stage)


def _cover_section(bundle: dict[str, Any]) -> str:
    round_number = bundle["round"]
    question = _main_question(bundle)
    touchpoints = _human_touchpoints(bundle, round_number)
    experiment = ""
    protocol = bundle.get("protocol", {})
    if protocol.get("experiment_id"):
        experiment = str(protocol["experiment_id"])
    else:
        mem = bundle.get("experiment_memory", {})
        entries = [e for e in (mem.get("entries") or []) if e.get("round_id") == round_number]
        if entries:
            experiment = str(
                entries[0].get("source_experiment_id")
                or entries[0].get("approved_candidate_id")
                or entries[0].get("result_path")
                or ""
            )
    rows = [
        ("科学问题", question),
        ("当前轮次", f"第 {round_number} 轮"),
        ("运行模型", _model_name(bundle)),
        ("本轮阶段", _round_stage(bundle, round_number)),
        ("本轮回合实验", experiment or "尚未审批实验"),
        ("人工触点", "；".join(touchpoints) if touchpoints else "暂无人工操作记录"),
        ("数据版本", "；".join(_data_manifest_versions()) or "--"),
        ("会话 ID", LIVE_ROOT.name),
        ("代码版本", _code_version()),
        ("报告生成时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    ]
    invalidation = _load_json(STATE_DIR / "data_issue_invalidation.json")
    invalidation_html = ""
    if invalidation.get("status") == "evidence_invalidated":
        exp_ids = "、".join(_invalidation_experiment_ids(invalidation.get("invalidated_experiments"))) or "--"
        invalidation_html = (
            '<div class="warning-banner"><strong>该会话数据证据已失效</strong>：'
            f"{_esc(invalidation.get('root_cause') or '数据列错位导致目标变量错误')}"
            f"受影响实验：{_esc(exp_ids)}。旧结果仅保留作审计，不构成正式科学结论，"
            "下一轮需在 v3 数据会话重新建立实验状态。</div>"
        )
    return (
        f'<div class="report-cover">'
        f'<p class="kicker">Shadow Tracing · Closed-Loop Report</p>'
        f'<h1>第 {round_number} 轮实验结果报告</h1>'
        f"<p class=\"question-line\">{_esc(question)}</p>"
        f"{invalidation_html}"
        f"{_kv_table(rows)}</div>"
    )


def _research_goal_section(bundle: dict[str, Any]) -> str:
    task = bundle.get("task", {})
    payload = task.get("payload", {})
    research = payload.get("research_question", {})
    variables = research.get("variables", {})
    constraints = payload.get("constraints", {}) or {}
    dictionary = bundle.get("data_dictionary", {})

    target_parts = []
    target_parts.append(("研究目标", _main_question(bundle)))
    target_parts.append(("核心解释变量", variables.get("x")))
    target_parts.append(("目标变量", variables.get("y")))
    target_parts.append(("候选特征 / 中介变量", variables.get("m_candidates")))
    target_parts.append(("评价指标", payload.get("evaluation", {})))

    constraint_rows = [
        ("未来信息泄露限制", "严格禁用" if constraints.get("no_future_information") else "允许"),
        ("是否允许基于验证结果迭代", "是" if constraints.get("validation_feedback_allowed") else "否"),
        ("最终测试是否盲测", "是" if constraints.get("final_test_blind") else "否"),
        ("每层假设数量上限", constraints.get("max_hypotheses_per_level")),
        ("最低活跃假设数", constraints.get("min_active_hypotheses")),
        ("最大闭环轮数", constraints.get("max_rounds")),
        ("每轮候选实验上限", constraints.get("max_experiments_per_round")),
        ("人工备注", constraints.get("notes")),
    ]

    data_files = []
    for source_name, source in (payload.get("data_sources") or {}).items():
        if isinstance(source, dict):
            data_files.append(f"{source_name}: {source.get('path')}")
    knowledge_dir = LIVE_ROOT / "uploads" / "knowledge"
    knowledge_files = []
    if knowledge_dir.is_dir():
        knowledge_files = [str(path.relative_to(LIVE_ROOT)) for path in sorted(knowledge_dir.rglob("*")) if path.is_file()]

    body = [_h2("1. 研究目标与约束理解")]
    body.append(_h3("1.1 目标与变量"))
    body.append(_kv_table(target_parts))
    body.append(_h3("1.2 约束条件"))
    body.append(_kv_table(constraint_rows))
    body.append(_h3("1.3 输入材料"))
    body.append(
        "<p>"
        f"数据文件：{'；'.join(data_files) if data_files else dictionary.get('dataset_name') or '--'}<br/>"
        f"知识材料：{'；'.join(knowledge_files) if knowledge_files else '尚未接入独立知识文件'}<br/>"
        f"数据字典：{dictionary.get('dataset_name') or dictionary.get('dictionary_id') or '--'} · "
        f"时间范围 {dictionary.get('time_range') or '--'}"
        "</p>"
    )
    body.append(_h3("1.4 任务定义确认"))
    body.append(
        _quote(
            "中央进程控制者 · 任务定义",
            f"目标变量已确认为 {_to_text(variables.get('y') or '未定义')}，"
            f"核心解释变量为 {_to_text(variables.get('x') or '未定义')}，"
            f"候选中介变量 {_to_text(variables.get('m_candidates') or [])}；"
            f"问题类型：{_to_text(research.get('question_type') or 'forecasting')}。",
        )
    )
    return "<section>" + "".join(body) + "</section>"


def _hypothesis_section(bundle: dict[str, Any]) -> str:
    round_number = bundle["round"]
    tree = bundle.get("hypothesis_tree", {})
    nodes = tree.get("nodes") or []
    raw_mapping = _hypothesis_display_map(bundle)
    question_dec = _last_decision(bundle, "scientific_questioning_completed", round_id=round_number)
    questioning_audit = []
    if question_dec:
        details = question_dec.get("details", {})
        if isinstance(details, dict):
            questioning_audit = details.get("node_audit") or []
    audit_by_id = {
        str(item.get("hypothesis_id")): item
        for item in questioning_audit
        if isinstance(item, dict) and item.get("hypothesis_id")
    }

    body = [_h2("2. 假设空间状态与科学假设生成")]
    if not nodes:
        body.append('<p class="muted">本轮未保存假设空间快照。</p>')
        return "<section>" + "".join(body) + "</section>"

    body.append(_h3("2.1 假设空间摘要"))
    header = (
        "<tr><th>假设 ID</th><th>层级</th><th>状态</th>"
        "<th>支持度（前 → 后）</th><th>本轮动作</th></tr>"
    )
    rows = []
    for node in nodes:
        node_id = node.get("display_hypothesis_id") or node.get("hypothesis_id")
        level = node.get("level")
        level_label = {0: "根", 1: "父", 2: "子"}.get(int(level or 0), str(level or "--"))
        current_score = node.get("support_score")
        before, after = _support_before_after(node, round_number, current_score)
        action = "继承"
        if node.get("created_at_round") == round_number:
            action = "本轮新增"
        elif round_number > 1 and before is not None and after is not None and before != after:
            action = "本轮更新"
        audit = audit_by_id.get(str(node.get("hypothesis_id")))
        if audit and audit.get("status"):
            action = f"{action} / 质询后 {STATUS_LABELS.get(str(audit.get('status')).lower(), audit.get('status'))}"
        rows.append(
            "<tr>"
            f"<td>{_esc(node_id)}</td><td>{_esc(level_label)}</td>"
            f"<td>{_status_pill(node.get('status'))}</td>"
            f"<td>{_fmt(before)} → {_fmt(after)}</td><td>{_esc(action)}</td>"
            "</tr>"
        )
    body.append(f"<table>{header}{''.join(rows)}</table>")

    body.append(_h3("2.2 假设详情"))
    for node in nodes:
        node_id = node.get("display_hypothesis_id") or node.get("hypothesis_id")
        current_score = node.get("support_score")
        before, after = _support_before_after(node, round_number, current_score)
        body.append(
            f'<h4>假设 {_esc(node_id)} · {STATUS_LABELS.get(str(node.get("status")).lower(), _to_text(node.get("status")))}</h4>'
        )
        body.append(_kv_table([
            ("假设陈述", node.get("statement") or node.get("display_statement")),
            ("内部标识", node.get("hypothesis_id")),
            ("支持度", f"{_fmt(before)} → {_fmt(after)}（当前 {_fmt(current_score)}）"),
            ("激活条件", node.get("activation_condition")),
            ("预测", node.get("predictions")),
            ("证据项", node.get("evidence_items")),
            ("反证项", node.get("evidence_against")),
            ("可证否条件", node.get("falsification_conditions")),
            ("替代解释", node.get("alternative_explanations")),
        ]))
        rationale = None
        if isinstance(node.get("generation_rationale"), dict):
            rationale = node["generation_rationale"].get("summary")
        if rationale:
            body.append(_quote("假设提出者（LLM 原文 · 生成依据）", rationale))

        records = [
            record
            for record in (node.get("questioning_records") or [])
            if isinstance(record, dict) and record.get("round") == round_number
        ]
        if not records:
            audit = audit_by_id.get(str(node.get("hypothesis_id")))
            if audit:
                records = [audit]
        for record in records:
            direction_map = {
                "supports": "支持",
                "clarifies": "澄清",
                "weakens": "削弱",
            }
            impact = str(record.get("impact_direction") or "")
            impact_label = direction_map.get(impact.lower(), impact)
            body.append(
                _quote(
                    f"科学质询者（LLM 原文 · R{round_number} · {impact_label}）",
                    (
                        record.get("rationale")
                        or f"{node_id} 本轮质询未生成可落盘语段；"
                        f"规则判定 {impact_label}。"
                    )
                    + (
                        f"\n\n可证否依据：{record.get('falsification_basis')}"
                        if record.get("falsification_basis")
                        else ""
                    ),
                )
            )
    if raw_mapping:
        body.append("<p class=\"muted\">假设对照："
                    + "；".join(f"{_esc(value)} = {_esc(key)}" for key, value in sorted(raw_mapping.items()))
                    + "。正文使用短标识，完整内部 ID 仅在本节给出。</p>")
    return "<section>" + "".join(body) + "</section>"


def _uncertainty_candidate_section(bundle: dict[str, Any]) -> str:
    round_number = bundle["round"]
    body = [_h2("3. 不确定性驱动的候选实验")]
    raw_mapping = _hypothesis_display_map(bundle)

    records = bundle.get("uncertainties", {}).get("records") or []
    round_records = []
    for record in records:
        created_round = record.get("created_at_round")
        if isinstance(created_round, int) and created_round <= round_number:
            round_records.append(record)
        elif record.get("created_at_round") is None:
            round_records.append(record)
    round_records = sorted(
        round_records,
        key=lambda item: (
            0 if str(item.get("status")).lower() == "resolved" else 1,
            0 if str(item.get("priority")).lower() == "high" else 1,
        ),
    )
    body.append(_h3("3.1 不确定性队列"))
    if round_records:
        header = (
            "<tr><th>ID</th><th>状态</th><th>优先级</th><th>残留</th>"
            "<th>科学问题</th><th>关联假设</th></tr>"
        )
        rows = []
        for record in round_records[:18]:
            rows.append(
                "<tr>"
                f"<td>{_esc(record.get('uncertainty_id'))}</td>"
                f"<td>{_status_pill(record.get('status'))}</td>"
                f"<td>{_esc(record.get('priority'))}</td>"
                f"<td>{_esc(record.get('resolution_status') or '保留')}</td>"
                f"<td>{_esc(record.get('question'))}</td>"
                f"<td>{_esc(_related_hypothesis_text(bundle, raw_mapping, record.get('related_hypotheses')))}</td>"
                "</tr>"
            )
        body.append(f"<table>{header}{''.join(rows)}</table>")
    else:
        body.append('<p class="muted">本轮不确定性队列未落盘。</p>')

    body.append(_h3("3.2 候选实验生成报告"))
    candidates = [
        item
        for item in (bundle.get("candidate_experiments", {}).get("candidates") or [])
        if isinstance(item, dict) and f"_R{round_number:02d}_" in str(item.get("experiment_id") or "")
    ]
    if candidates:
        approved = _last_decision(bundle, "experiment_approved", round_id=round_number)
        sorted_candidates = sorted(
            candidates, key=lambda item: float(item.get("utility_score") or 0), reverse=True
        )
        rank_rows = []
        for index, candidate in enumerate(sorted_candidates, start=1):
            candidate_experiment_type = str(
                candidate.get("experiment_type") or ""
            ).strip()
            fallback_type = str(candidate.get("type") or "").strip()
            candidate_type_label = EXPERIMENT_TYPE_LABELS.get(
                candidate_experiment_type,
                candidate_experiment_type
                or EXPERIMENT_TYPE_LABELS.get(fallback_type, "未标注"),
            )
            rank_rows.append(
                "<tr>"
                f"<td>{index}</td><td>{_esc(candidate.get('experiment_id'))}</td>"
                f"<td>{_fmt(candidate.get('utility_score'))}</td>"
                f"<td>{_esc(candidate_type_label)}</td></tr>"
            )
        body.append(
            "<table><tr><th>排序</th><th>实验 ID</th><th>U(E)</th><th>类型</th></tr>"
            + "".join(rank_rows)
            + "</table>"
        )
        for candidate in sorted_candidates:
            experiment_id = candidate.get("experiment_id")
            body.append(
                f'<h4>候选实验 {_esc(experiment_id)} · U(E) {_fmt(candidate.get("utility_score"))}</h4>'
            )
            body.append(_quote("实验目的（LLM 原文）", candidate.get("purpose")))
            body.append(_quote("对应不确定性（LLM 原文）", candidate.get("scientific_question")))
            body.append(
                "<p><strong>可验证假设：</strong>"
                f"{_esc(_related_hypothesis_text(bundle, raw_mapping, candidate.get('tested_hypotheses')))} · "
                f"关联不确定性：{_esc(candidate.get('related_uncertainties'))}</p>"
            )
            body.append(_quote("区分性思路（LLM 原文）", candidate.get("distinguishing_insight")))
            design = candidate.get("design") or {}
            body.append(_kv_table([
                ("因变量", design.get("target") or candidate.get("design", {}).get("target")),
                ("自变量 · 对照组", design.get("control")),
                ("自变量 · 实验组", design.get("treatment")),
                ("滞后构造", design.get("lags")),
                ("预测视野（天）", design.get("forecast_horizon_days")),
                ("历史滞后（天）", design.get("past_lag_days")),
                ("时间窗口（天）", design.get("window_size")),
                ("控制滞后（天）", design.get("control_lag_days")),
                ("区分焦点", bundle.get("protocol", {}).get("display_design_focus") or design.get("focus")),
                ("模型", bundle.get("protocol", {}).get("model", {}).get("name") or "ElasticNet"),
            ]))
            body.append(_kv_table([
                ("所回应的上一轮发现", candidate.get("responded_prior_finding")),
                ("要解决的不确定性", candidate.get("target_uncertainty_focus")),
                ("相对上一轮改变的因素", candidate.get("changed_factors")),
                ("保持固定的条件", candidate.get("held_fixed_factors")),
                ("探索性实验", candidate.get("exploratory")),
            ]))
            branches = candidate.get("expected_outcome_branches") or {}
            if isinstance(branches, dict) and branches:
                branch_rows = []
                for branch_label, branch_action in branches.items():
                    if branch_action in (None, "", [], {}):
                        continue
                    branch_rows.append(
                        f"<tr><th>{_esc(branch_label)}</th><td>{_esc(branch_action)}</td></tr>"
                    )
                body.append(
                    "<p><strong>预登记结果分支与下一步行动</strong></p>"
                    f"<table>{''.join(branch_rows)}</table>"
                )
            body.append(_quote("实验综合价值分析（LLM 原文）", candidate.get("value_analysis")))
            for label, key in (
                ("信息增益", "estimated_information_gain"),
                ("性能增益", "estimated_performance_gain"),
                ("风险", "estimated_risk"),
                ("成本", "estimated_cost"),
            ):
                estimate = candidate.get(key) or {}
                if isinstance(estimate, dict):
                    body.append(
                        f'<p><strong>{label}：</strong>'
                        f"{_fmt(estimate.get('value'))} · {_esc(estimate.get('rationale'))}</p>"
                    )
        if approved:
            details = approved.get("details", {})
            body.append(_h3("3.3 推荐选择与审批"))
            body.append(
                _quote(
                    "实验选择（决策层）",
                    details.get("plan_summary")
                    or f"推荐 {details.get('candidate_id')}，综合价值 {_fmt(details.get('utility_score'))}。",
                )
            )
    else:
        body.append('<p class="muted">本轮候选实验尚未落盘。</p>')
    return "<section>" + "".join(body) + "</section>"


def _execution_section(bundle: dict[str, Any]) -> str:
    round_number = bundle["round"]
    body = [_h2("4. 实验执行与结果")]
    protocol = bundle.get("protocol", {})
    result = bundle.get("result", {})
    evaluation = bundle.get("evaluation", {})
    metrics = evaluation.get("metrics") or {}
    runs = result.get("runs") or []
    experiment_type = str(
        result.get("experiment_type")
        or protocol.get("experiment_type")
        or ""
    ).lower()

    if not (protocol or result):
        body.append('<p class="muted">本轮尚未进入实验执行或结果未落盘。</p>')
        return "<section>" + "".join(body) + "</section>"

    body.append(_h3("4.1 执行协议"))
    approval = _last_decision(bundle, "experiment_approved", round_id=round_number)
    body.append(_kv_table([
        ("实验 ID", protocol.get("experiment_id") or result.get("experiment_id")),
        ("实验类型", EXPERIMENT_TYPE_LABELS.get(experiment_type, experiment_type or "--")),
        ("执行状态", result.get("status") or "进行中"),
        ("审批记录", approval.get("timestamp") if approval else "--"),
        ("目标变量", protocol.get("target") or protocol.get("display_target")),
        ("对照组特征", protocol.get("display_control") or protocol.get("features", {}).get("control")),
        ("实验组特征", protocol.get("display_treatment") or protocol.get("features", {}).get("treatment")),
        ("区分焦点", protocol.get("display_design_focus") or protocol.get("design_focus")),
        ("科学目标", protocol.get("scientific_objective")),
    ]))

    body.append(_h3("4.2 指标对照"))
    result_runs = result.get("runs") or []
    metrics_runs = metrics.get("runs") if isinstance(metrics, dict) else []
    all_runs = result_runs or metrics_runs or runs
    run_by_id = {str(run.get("run_id") or run.get("name")): run for run in all_runs if isinstance(run, dict)}
    run_ids = _ordered_run_ids(list(run_by_id.keys()))
    delta = result.get("comparison") or metrics.get("delta") or {}
    metric_names = [("Pearson r", "pearson_r"), ("RMSE", "rmse"), ("MAE", "mae"), ("R²", "r2")]
    if run_ids:
        run_headers = []
        for rid in run_ids:
            run_label = str(run_by_id[rid].get("name") or rid)
            run_headers.append(
                f"<th>{_esc(run_label)}（{_esc(rid)}）</th>"
                if run_label != rid
                else f"<th>{_esc(rid)}</th>"
            )
        header = "<tr><th>指标</th>" + "".join(run_headers) + "</tr>"
        metric_rows = []
        for label, key in metric_names:
            cells = []
            for rid in run_ids:
                run = run_by_id[rid]
                run_metrics = run.get("metrics") or {}
                cells.append(_fmt(metrics.get(f"{rid}_{key}", run_metrics.get(key))))
            metric_rows.append(f"<tr><td>{label}</td>{''.join(f'<td>{cell}</td>' for cell in cells)}</tr>")
        body.append(f"<table>{header}{''.join(metric_rows)}</table>")
    else:
        baseline = metrics.get("baseline") or {}
        treatment = metrics.get("treatment") or {}
        header = "<tr><th>指标</th><th>对照组</th><th>实验组</th><th>增量</th></tr>"
        metric_rows = []
        for label, key in metric_names:
            metric_rows.append(
                "<tr>"
                f"<td>{label}</td><td>{_fmt(metrics.get(f'baseline_{key}', baseline.get(key)))}</td>"
                f"<td>{_fmt(metrics.get(f'treatment_{key}', treatment.get(key)))}</td>"
                f"<td>{_fmt(delta.get(key))}</td></tr>"
            )
        body.append(f"<table>{header}{''.join(metric_rows)}</table>")

    contrasts = result.get("contrasts") or (metrics.get("contrasts") if isinstance(metrics, dict) else []) or []
    if contrasts:
        body.append(_h3("4.2.1 对比检验与置信区间"))
        contrast_header = "<tr><th>对比</th><th>ΔPearson r</th><th>95% CI</th><th>ΔRMSE</th><th>95% CI</th><th>说明</th></tr>"
        contrast_rows = []
        for contrast in contrasts:
            if not isinstance(contrast, dict):
                continue
            contrast_delta = contrast.get("metrics_delta") or {}
            ci = contrast.get("bootstrap_ci") or {}
            pearson_ci = ci.get("pearson_r_ci")
            rmse_ci = ci.get("rmse_ci")
            contrast_rows.append(
                "<tr>"
                f"<td>{_esc(contrast.get('contrast_id') or contrast.get('name'))}</td>"
                f"<td>{_fmt(contrast_delta.get('pearson_r'))}</td>"
                f"<td>{_fmt(pearson_ci[0])} ~ {_fmt(pearson_ci[1]) if isinstance(pearson_ci, list) and len(pearson_ci) == 2 else '--'}</td>"
                f"<td>{_fmt(contrast_delta.get('rmse'))}</td>"
                f"<td>{_fmt(rmse_ci[0])} ~ {_fmt(rmse_ci[1]) if isinstance(rmse_ci, list) and len(rmse_ci) == 2 else '--'}</td>"
                f"<td>{_esc(contrast.get('note'))}</td></tr>"
            )
        body.append(f"<table>{contrast_header}{''.join(contrast_rows)}</table>")

    diagnostics = (
        result.get("diagnostics")
        or (metrics.get("diagnostics") if isinstance(metrics, dict) else [])
        or []
    )
    if diagnostics:
        body.append(_h3("4.2.2 诊断与稳健性"))
        diagnostic_rows = []
        for item in diagnostics:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("diagnostic") or item.get("type") or "--"
            status = item.get("status") or "completed"
            detail = {
                key: value
                for key, value in item.items()
                if key not in ("name", "diagnostic", "type", "status")
            }
            diagnostic_rows.append(
                f"<tr><td>{_esc(name)}</td><td>{_esc(status)}</td><td>{_esc(detail)}</td></tr>"
            )
        body.append(
            "<table><tr><th>诊断项</th><th>状态</th><th>配置与结果</th></tr>"
            + "".join(diagnostic_rows)
            + "</table>"
        )

    model = protocol.get("model", {}) or {}
    parameters = model.get("parameters") if isinstance(model, dict) else {}
    if isinstance(parameters, dict):
        body.append(_h3("4.3 模型参数与调参缘由"))
        rationale = None
        features = protocol.get("features", {}) or {}
        notes = features.get("notes") or []
        for note in notes:
            if "llm_candidate_design_rationale" in str(note):
                rationale = str(note).split(":", 1)[-1]
                break
        if rationale:
            body.append(_quote("实验规划者（LLM 原文 · 调参缘由）", rationale))
        elif protocol.get("display_design_focus"):
            body.append(
                _quote(
                    "实验规划者（LLM 原文 · 调参缘由）",
                    f"围绕区分焦点 {_to_text(protocol.get('display_design_focus'))} "
                    f"构建对照设计，固定常规物理基线并仅加入待验证变量，"
                    f"在相同窗口、滞后与超前条件下比较预测增量。",
                )
            )
        body.append(_kv_table([
            (key, value)
            for key, value in parameters.items()
            if not isinstance(value, (dict, list))
        ]))
    body.append(_h3("4.4 核心结果图"))
    experiment_id = bundle.get("result", {}).get("experiment_id")
    core_charts = _presentation_figures_html(
        round_number, experiment_id
    )
    body.append(
        core_charts
        or '<p class="muted">本轮尚未生成可嵌入的核心结果图。</p>'
    )
    body.append(_h3("4.5 运行级诊断图"))
    body.append(_figure_grid(bundle, round_number))
    return "<section>" + "".join(body) + "</section>"


def _classify_family_gain_for_export(
    point: Any,
    rmse_ci: Any,
) -> str:
    ci = rmse_ci if isinstance(rmse_ci, list) and len(rmse_ci) >= 2 else None
    lower = upper = None
    if ci is not None:
        try:
            lower = float(ci[0])
            upper = float(ci[1])
        except (TypeError, ValueError):
            lower = upper = None
    if lower is not None and upper is not None and lower == lower and upper == upper:
        if lower == 0.0 and upper == 0.0:
            return "identical_predictions" if point is None or abs(float(point)) < 1e-9 else "no_stable_gain"
        if lower < 0 < upper or lower == 0 or upper == 0:
            return "no_stable_gain"
        if lower > 0:
            return "stable_positive"
        if upper < 0:
            return "stable_negative"
        return "insufficient_blocks"
    if point is not None:
        try:
            number = float(point)
        except (TypeError, ValueError):
            number = None
        if number is not None and number == number:
            return "point_positive" if number > 0 else "point_negative" if number < 0 else "point_zero"
    return "not_evaluated"


def _derive_family_rows_from_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    runs = result.get("runs") or []
    by_run_id = {
        str(run.get("run_id") or run.get("name")): run
        for run in runs
        if isinstance(run, dict)
    }
    seen: set[str] = set()
    for contrast in result.get("contrasts") or []:
        if not isinstance(contrast, dict):
            continue
        family = None
        for run_id in contrast.get("run_ids") or []:
            run = by_run_id.get(str(run_id))
            if run and run.get("model_family"):
                family = str(run["model_family"])
                break
        family = family or "unknown"
        if family in seen:
            continue
        seen.add(family)
        delta = contrast.get("metrics_delta") or {}
        bootstrap = contrast.get("bootstrap_ci") or {}
        rmse_ci = bootstrap.get("rmse_ci")
        point = delta.get("rmse")
        evaluation = _classify_family_gain_for_export(point, rmse_ci)
        rows.append(
            {
                "family": family,
                "contrast_id": contrast.get("contrast_id"),
                "point_gain_rmse": point,
                "point_gain_pearson_r": delta.get("pearson_r"),
                "rmse_ci": rmse_ci,
                "pearson_ci": bootstrap.get("pearson_r_ci"),
                "crosses_zero": (
                    (
                        float(rmse_ci[0]) < 0 < float(rmse_ci[1])
                        or (float(rmse_ci[0]) == 0 and float(rmse_ci[1]) > 0)
                        or (float(rmse_ci[1]) == 0 and float(rmse_ci[0]) < 0)
                    )
                    if isinstance(rmse_ci, list) and len(rmse_ci) >= 2
                    else None
                )
                ,
                "evaluation": evaluation,
                "evaluation_label": MODEL_SENSITIVITY_LABELS.get(evaluation, evaluation),
            }
        )
    return rows


def _family_rows_from_conclusion(
    conclusion: dict[str, Any],
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    experiment_layer = conclusion.get("experiment_layer") or {}
    structured = experiment_layer.get("model_sensitivity_families") or []
    if structured:
        return [item for item in structured if isinstance(item, dict)]
    return _derive_family_rows_from_result(result)


def _family_rows_verdict(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "未评估"
    statuses = [str(row.get("evaluation") or "") for row in rows]
    if all(status == "identical_predictions" for status in statuses):
        return "两组预测完全一致（无预测差异）"
    if any(status in {"no_stable_gain", "insufficient_blocks", "not_evaluated"} for status in statuses):
        return "未检出稳定增益"
    stable_positive = sum(status == "stable_positive" for status in statuses)
    stable_negative = sum(status == "stable_negative" for status in statuses)
    if stable_positive and stable_negative:
        return "混合方向（需先核查数据与执行）"
    if stable_positive == len(rows):
        return "稳定正向增益"
    if stable_negative == len(rows):
        return "稳定负向增益"
    return "未检出稳定增益"


def _ci_text(ci: Any) -> str:
    if isinstance(ci, list) and len(ci) >= 2:
        return f"{_fmt(ci[0])} ~ {_fmt(ci[1])}"
    return "--"


def _model_sensitivity_family_section(
    conclusion: dict[str, Any],
    result: dict[str, Any],
) -> str:
    rows = _family_rows_from_conclusion(conclusion, result)
    if not rows:
        return ""
    header = (
        "<tr><th>模型家族</th><th>ΔRMSE</th><th>95% CI（RMSE）</th>"
        "<th>ΔPearson r</th><th>95% CI（Pearson r）</th><th>判定</th></tr>"
    )
    table_rows = []
    for row in rows:
        table_rows.append(
            "<tr>"
            f"<td>{_esc(row.get('family'))}</td>"
            f"<td>{_fmt(row.get('point_gain_rmse'))}</td>"
            f"<td>{_ci_text(row.get('rmse_ci'))}</td>"
            f"<td>{_fmt(row.get('point_gain_pearson_r'))}</td>"
            f"<td>{_ci_text(row.get('pearson_ci'))}</td>"
            f"<td>{_esc(row.get('evaluation_label') or row.get('evaluation'))}</td>"
            "</tr>"
        )
    verdict = _family_rows_verdict(rows)
    return (
        '<h4>5.1.1 各模型家族日影增量与 CI</h4>'
        f'<p class="detail-card__meta">总体判定（程序保守规则）：{_esc(verdict)}</p>'
        f"<table>{header}{''.join(table_rows)}</table>"
        '<p class="muted">点估计方向与稳定证据分开判断；任一家族 CI 跨零、缺失或有效时间块不足时，'
        '整体结论固定为“未检出稳定增益”，本轮不自动改变假设支持度。</p>'
    )


def _four_layer_section(bundle: dict[str, Any]) -> str:
    round_number = bundle["round"]
    body = [_h2(f"5. 第 {round_number} 轮实验结果报告")]
    raw_mapping = _hypothesis_display_map(bundle)
    evaluation = bundle.get("evaluation", {})
    scientific = evaluation.get("scientific", {}) or {}
    conclusion = scientific.get("three_layer_conclusion") or {}
    metrics = evaluation.get("metrics") or {}
    result = bundle.get("result", {})
    assessments = scientific.get("hypothesis_assessments") or conclusion.get("hypothesis_layer") or []
    data_layer = conclusion.get("data_layer") or {}
    scientific_layer = conclusion.get("scientific_layer") or {}
    tracking_layer = conclusion.get("tracking_layer") or {}

    if not conclusion:
        body.append('<p class="muted">本轮评估结果尚未落盘，四层分析将在评估完成后补齐。</p>')
        return "<section>" + "".join(body) + "</section>"

    experiment_type = str(result.get("experiment_type") or "").lower()
    type_label = EXPERIMENT_TYPE_LABELS.get(experiment_type, experiment_type or "未标注")
    body.append(f'<p class="detail-card__meta">实验类型 · {_esc(type_label)}</p>')
    body.append(_h3("5.1 指标快照"))
    delta = metrics.get("delta") or {}
    result_runs = result.get("runs") or (metrics.get("runs") if isinstance(metrics, dict) else []) or []
    run_by_id = {
        str(run.get("run_id") or run.get("name")): run
        for run in result_runs
        if isinstance(run, dict)
    }
    run_ids = _ordered_run_ids(list(run_by_id.keys()))
    if run_ids:
        run_headers = []
        for rid in run_ids:
            run_label = str(run_by_id[rid].get("name") or rid)
            run_headers.append(
                f"<th>{_esc(run_label)}（{_esc(rid)}）</th>"
                if run_label != rid
                else f"<th>{_esc(rid)}</th>"
            )
        header = "<tr><th>指标</th>" + "".join(run_headers) + "</tr>"
        table_rows = []
        for label, key in [("Pearson r", "pearson_r"), ("RMSE", "rmse"), ("MAE", "mae"), ("R²", "r2")]:
            cells = []
            for rid in run_ids:
                run_metrics = run_by_id[rid].get("metrics") or {}
                cells.append(_fmt(metrics.get(f"{rid}_{key}", run_metrics.get(key))))
            table_rows.append(f"<tr><td>{label}</td>{''.join(f'<td>{cell}</td>' for cell in cells)}</tr>")
        body.append(f"<table>{header}{''.join(table_rows)}</table>")
        contrasts = result.get("contrasts") or (metrics.get("contrasts") if isinstance(metrics, dict) else []) or []
        if contrasts:
            contrast_header = "<tr><th>对比</th><th>ΔPearson r</th><th>95% CI</th><th>ΔRMSE</th><th>95% CI</th></tr>"
            contrast_rows = []
            for contrast in contrasts:
                if not isinstance(contrast, dict):
                    continue
                contrast_delta = contrast.get("metrics_delta") or {}
                ci = contrast.get("bootstrap_ci") or {}
                pearson_ci = ci.get("pearson_r_ci")
                rmse_ci = ci.get("rmse_ci")
                contrast_rows.append(
                    "<tr>"
                    f"<td>{_esc(contrast.get('contrast_id') or contrast.get('name'))}</td>"
                    f"<td>{_fmt(contrast_delta.get('pearson_r'))}</td>"
                    f"<td>{_fmt(pearson_ci[0])} ~ {_fmt(pearson_ci[1]) if isinstance(pearson_ci, list) and len(pearson_ci) == 2 else '--'}</td>"
                    f"<td>{_fmt(contrast_delta.get('rmse'))}</td>"
                    f"<td>{_fmt(rmse_ci[0])} ~ {_fmt(rmse_ci[1]) if isinstance(rmse_ci, list) and len(rmse_ci) == 2 else '--'}</td></tr>"
                )
            body.append(f"<table>{contrast_header}{''.join(contrast_rows)}</table>")
    else:
        header = "<tr><th>指标</th><th>对照组</th><th>实验组</th><th>变化</th></tr>"
        rows = [
            ("RMSE", "baseline_rmse", "treatment_rmse", "rmse"),
            ("Pearson r", "baseline_pearson_r", "treatment_pearson_r", "pearson_r"),
            ("MAE", "baseline_mae", "treatment_mae", "mae"),
        ]
        table_rows = []
        for label, base_key, treat_key, delta_key in rows:
            table_rows.append(
                "<tr>"
                f"<td>{label}</td><td>{_fmt(metrics.get(base_key))}</td>"
                f"<td>{_fmt(metrics.get(treat_key))}</td><td>{_fmt(delta.get(delta_key))}</td></tr>"
            )
        body.append(f"<table>{header}{''.join(table_rows)}</table>")

    family_section = _model_sensitivity_family_section(conclusion, result)
    if family_section:
        body.append(family_section)

    body.append(_h3("5.2 第一层 · 数据层"))
    data_source_tag = _interpretation_source_tag(data_layer)
    body.append(_quote(f"RMSE 变化归因（{data_source_tag}）", data_layer.get("rmse_attribution")))
    body.append(_quote(f"Pearson-r 变化归因（{data_source_tag}）", data_layer.get("pearson_attribution")))
    body.append(_quote("ΔSkill 含义（程序计算）", data_layer.get("skill_delta_meaning")))
    body.append(_quote("异常 / 不一致识别", data_layer.get("anomalies")))
    body.append(_quote("推荐后续关注方向", data_layer.get("next_focus")))
    chart_analyses = data_layer.get("chart_analyses")
    if chart_analyses:
        body.append(_quote(f"图表逐图分析（{data_source_tag}）", chart_analyses))
    else:
        body.append('<p class="muted">本轮图表逐图分析未生成，结果图已在 4.4 节以原始图像落盘。</p>')

    body.append(_h3("5.3 第二层 · 假设层"))
    header = (
        "<tr><th>假设</th><th>预期方向</th><th>预期范围</th><th>实际增量</th>"
        "<th>方向匹配</th><th>幅度匹配</th><th>结论</th><th>支持度</th></tr>"
    )
    table_rows = []
    for row in assessments:
        hypothesis_id = row.get("display_hypothesis_id") or row.get("hypothesis_id")
        hypothesis_label = _short_hypothesis_label(bundle, raw_mapping, str(hypothesis_id))
        direction = row.get("predicted_direction")
        predicted_range = row.get("predicted_range") or row.get("expected_range")
        support_after = row.get("support_after")
        support_before = row.get("support_before")
        if support_before is None and support_after is not None:
            support_before = "--"
        support_text = (
            f"{_fmt(support_before)} → {_fmt(support_after)}"
            if support_before not in (None, "--")
            else f"{_fmt(support_after)}（本轮后）"
        )
        table_rows.append(
            "<tr>"
            f"<td>{_esc(hypothesis_label)}</td>"
            f"<td>{_esc(direction)}</td><td>{_esc(predicted_range)}</td>"
            f"<td>{_fmt(_first_not_none(row.get('observed_metric_delta'), row.get('actual_delta'), row.get('delta')))}</td>"
            f"<td>{_match_text(row.get('direction_matched'))}</td>"
            f"<td>{_match_text(row.get('magnitude_matched'))}</td>"
            f"<td>{_esc(row.get('conclusion') or row.get('reason'))}</td>"
            f"<td>{_esc(support_text)}</td>"
            "</tr>"
        )
    body.append(f"<table>{header}{''.join(table_rows)}</table>")

    body.append(_h3("5.4 第三层 · 科学问题层"))
    body.append(_kv_table([
        ("主科学问题", scientific_layer.get("main_question") or _main_question(bundle)),
        ("证据摘要（程序计算）", scientific_layer.get("evidence_text")),
    ]))
    scientific_source_tag = _interpretation_source_tag(scientific_layer)
    body.append(_quote(f"科学问题回答（{scientific_source_tag}）", scientific_layer.get("answer")))
    body.append(_quote(f"传递路径回答（{scientific_source_tag}）", scientific_layer.get("path_answer")))

    body.append(_h3("5.5 第四层 · 实验追踪层"))
    body.append(_bullets(tracking_layer.get("audit_items")))
    body.append("<p>")
    body.append(f"来源模块：{_esc(tracking_layer.get('sources'))}<br/>")
    body.append(f"快照引用：{_esc(tracking_layer.get('snapshot_refs'))}")
    body.append("</p>")
    return "<section>" + "".join(body) + "</section>"


def _next_advice_section(bundle: dict[str, Any]) -> str:
    round_number = bundle["round"]
    body = [_h2("6. 给下一轮的建议")]
    evaluation = bundle.get("evaluation", {})
    scientific = evaluation.get("scientific", {}) or {}
    conclusion = scientific.get("three_layer_conclusion") or {}
    suggestion = None
    if isinstance(conclusion.get("next_round_suggestion"), dict):
        suggestion = conclusion["next_round_suggestion"]
    if not suggestion:
        history = [
            entry
            for entry in (bundle.get("round_history", {}).get("entries") or [])
            if entry.get("round_id") == round_number
        ]
        if history:
            three_layer = history[-1].get("three_layer_conclusion") or {}
            if isinstance(three_layer.get("next_round_suggestion"), dict):
                suggestion = three_layer["next_round_suggestion"]
    if suggestion and any(
        suggestion.get(key)
        for key in (
            "evidence_summary",
            "remaining_uncertainty_analysis",
            "experiment_design_advice",
            "hypothesis_space_advice",
            "pi_decision_advice",
        )
    ):
        for label, key in (
            ("1. 当前证据摘要", "evidence_summary"),
            ("2. 剩余不确定性定位", "remaining_uncertainty_analysis"),
            ("3. 具体实验设计建议", "experiment_design_advice"),
            ("3.2 假设空间建议", "hypothesis_space_advice"),
            ("4. 对人类 PI 的决策建议", "pi_decision_advice"),
        ):
            if suggestion.get(key):
                body.append(_quote(f"给下一轮的建议 · {label}（LLM 原文）", suggestion[key]))
        if suggestion.get("resolved_uncertainties_this_round"):
            body.append("<p><strong>本轮已解决的不确定性</strong></p>")
            body.append(_bullets(suggestion["resolved_uncertainties_this_round"]))
        if suggestion.get("unresolved_uncertainties_todo"):
            body.append("<p><strong>仍需解决的不确定性</strong></p>")
            body.append(_bullets(suggestion["unresolved_uncertainties_todo"]))
        if suggestion.get("notes"):
            body.append("<p><strong>注意事项</strong></p>")
            body.append(_bullets(suggestion["notes"]))
    else:
        review = _last_decision(bundle, "round_review_requested", round_id=round_number)
        review_detail = {}
        if review and isinstance(review.get("details"), dict):
            review_detail = review["details"]
        evaluation_summary = review_detail.get("evaluation_summary") or {}
        findings = evaluation_summary.get("key_findings") or []
        body.append(_quote("本轮实验反馈（程序汇总）", findings))
        remaining = evaluation_summary.get("remaining_uncertainties") or []
        if remaining:
            body.append("<p><strong>仍需解决的不确定性</strong></p>")
            body.append(_bullets([f"{item.get('uncertainty_id')}：{item.get('question')}" for item in remaining]))
        if evaluation_summary.get("robustness_recommendation"):
            body.append(_quote("稳健性建议（程序汇总）", evaluation_summary.get("robustness_recommendation")))
        disagreement_updates = review_detail.get("disagreement_updates") or []
        if disagreement_updates:
            body.append("<p><strong>本轮分歧收窄</strong></p>")
            body.append(_bullets([item.get("summary") for item in disagreement_updates]))
        traces = evaluation_summary.get("reasoning_traces") or []
        if traces:
            body.append(_quote("科学解释者（LLM 原文 · 推理痕迹）", [item.get("summary") for item in traces]))
        if not (findings or remaining or review_detail or disagreement_updates or traces):
            body.append('<p class="muted">本轮未生成 LLM 下一轮建议，以下按程序证据状态组织下一步。</p>')
        body.append("<p><strong>程序侧下一步建议（回退）</strong></p>")
        fallback_notes = [
            "下一轮应从当前数据版本与验证目标出发设计实验；未获得稳定证据前，不得把点估计视为科学结论。",
            "若本轮两组预测完全一致，先核验特征是否真正进入模型、日影系数是否被正则化压零，再决定是否重跑或做区分性实验。",
        ]
        invalidation = _load_json(STATE_DIR / "data_issue_invalidation.json")
        if invalidation.get("status") == "evidence_invalidated":
            exp_ids = "、".join(_invalidation_experiment_ids(invalidation.get("invalidated_experiments"))) or "--"
            fallback_notes.insert(
                0,
                f"当前会话数据证据已失效（{_esc(invalidation.get('root_cause') or '目标变量错误')}）；"
                f"受影响实验：{_esc(exp_ids)}，本轮依据不进入下一轮规划。",
            )
        body.append(_bullets(fallback_notes))
    return "<section>" + "".join(body) + "</section>"


def _conclusion_section(bundle: dict[str, Any]) -> str:
    round_number = bundle["round"]
    body = [_h2("7. 科学结论")]
    evaluation = bundle.get("evaluation", {})
    scientific = evaluation.get("scientific", {}) or {}
    conclusion = scientific.get("three_layer_conclusion") or {}
    scientific_layer = conclusion.get("scientific_layer") or {}
    data_layer = conclusion.get("data_layer") or {}
    invalidation = _load_json(STATE_DIR / "data_issue_invalidation.json")
    evidence_invalidated = invalidation.get("status") == "evidence_invalidated"

    body.append(_h3("7.1 数据指标分析"))
    if evidence_invalidated:
        body.append(_invalidation_banner(invalidation))
    metrics = evaluation.get("metrics") or {}
    result = bundle.get("result", {})
    result_runs = result.get("runs") or (metrics.get("runs") if isinstance(metrics, dict) else []) or []
    task = bundle.get("task", {})
    research_variables = (task.get("payload", {}).get("research_question", {}).get("variables") or {})
    target_name = (
        str(research_variables.get("y") or "").strip()
        or str(task.get("payload", {}).get("target_variable") or "").strip()
        or "目标变量"
    )
    metric_summary: list[str] = []
    for run in result_runs:
        if not isinstance(run, dict):
            continue
        rid = str(run.get("run_id") or run.get("name") or "运行")
        name = str(run.get("name") or rid)
        run_metrics = run.get("metrics") or {}
        pearson = run_metrics.get("pearson_r", metrics.get(f"{rid}_pearson_r"))
        rmse = run_metrics.get("rmse", metrics.get(f"{rid}_rmse"))
        metric_summary.append(f"{name}（{rid}） Pearson r={_fmt(pearson)}，RMSE={_fmt(rmse)}")
    if metric_summary:
        metric_analysis = "；".join(metric_summary)
    else:
        metric_analysis = (
            f"对照组 Pearson r={_fmt(metrics.get('baseline_pearson_r'))}，"
            f"实验组 Pearson r={_fmt(metrics.get('treatment_pearson_r'))}"
        )
    body.append(
        _quote(
            "科学结论 · 数据指标分析",
            (
                f"对{target_name}的预测："
                f"{metric_analysis}；"
                f"{_to_text(data_layer.get('rmse_attribution'))}；"
                f"{_to_text(data_layer.get('pearson_attribution'))}。"
            ),
        )
    )
    if evidence_invalidated:
        body.append(
            '<p class="muted">数据失效状态下，上述自然语言由旧解释生成，程序不保证其与失效指标一致；'
            "任何数值不得被引用为支持度更新依据。</p>"
        )
    body.append(_h3("7.2 科学假设分析"))
    body.append(_quote("科学结论 · 假设分析", scientific_layer.get("answer")))
    body.append(_h3("7.3 给下一轮的建议简述"))
    evidence_summary = scientific.get("evidence_summary") or {}
    if evidence_summary:
        body.append(_quote("科学结论 · 证据摘要", {
            "新增支持证据": evidence_summary.get("new_evidence_for"),
            "新增反对证据": evidence_summary.get("new_evidence_against"),
        }))
    else:
        body.append(_quote("科学结论 · 建议简述", data_layer.get("next_focus")))
    body.append(_h3("7.4 剩余不确定性"))
    remaining = (
        evidence_summary.get("remaining_uncertainties")
        or evaluation.get("metrics", {}).get("remaining_uncertainties")
        or []
    )
    if remaining:
        header = "<tr><th>ID</th><th>问题</th><th>优先级</th></tr>"
        rows = []
        for item in remaining:
            rows.append(
                "<tr>"
                f"<td>{_esc(item.get('uncertainty_id'))}</td>"
                f"<td>{_esc(item.get('question'))}</td>"
                f"<td>{_esc(item.get('priority'))}</td></tr>"
            )
        body.append(f"<table>{header}{''.join(rows)}</table>")
    else:
        body.append('<p class="muted">剩余不确定性未落盘。</p>')
    body.append(_h3("7.5 人工 PI 最终决策"))
    review = _last_decision(bundle, "round_review_requested", round_id=round_number)
    commit_decision = _latest_commit_decision(bundle, round_id=round_number)
    if commit_decision:
        decision_details = commit_decision.get("details") or {}
        decision_reason = (
            decision_details.get("human_feedback")
            or decision_details.get("human_notes")
            or commit_decision.get("summary")
            or "--"
        )
        raw_decision_time = commit_decision.get("timestamp")
        if not raw_decision_time and review:
            raw_decision_time = review.get("timestamp")
        decision_time = str(raw_decision_time or "")[:19]
    else:
        decision_reason = (
            (review.get("details") or {}).get("human_feedback")
            or review.get("summary")
            if review
            else None
        )
        decision_time = str(review.get("timestamp") or "")[:19] if review else None
    decision_label = _commit_decision_label(commit_decision)
    if not commit_decision and review:
        decision_label = "待人工决策"
    body.append(_kv_table([
        ("决策", decision_label),
        ("决策理由", decision_reason or "--"),
        ("决策时间", decision_time or "--"),
    ]))
    return "<section>" + "".join(body) + "</section>"


def _audit_section(bundle: dict[str, Any]) -> str:
    round_number = bundle["round"]
    body = [_h2("8. 审计链与人工确认")]
    decisions = [
        item
        for item in _decisions(bundle)
        if item.get("round_id") == round_number or item.get("decision_type") == "hypothesis_generated"
    ]
    if not decisions:
        body.append('<p class="muted">本轮无审计记录。</p>')
        return "<section>" + "".join(body) + "</section>"
    header = "<tr><th>时间</th><th>事件</th><th>模块</th><th>审计 ID</th><th>人工操作</th></tr>"
    rows = []
    for item in decisions:
        details = item.get("details") or {}
        human = ""
        if item.get("made_by") == "human_pi":
            human = "人工确认"
        elif details.get("human_notes"):
            human = "人工批注"
        rows.append(
            "<tr>"
            f"<td>{_esc(str(item.get('timestamp') or '')[:19])}</td>"
            f"<td>{_esc(item.get('summary') or item.get('decision_type'))}</td>"
            f"<td>{_esc(item.get('step') or item.get('phase'))}</td>"
            f"<td>{_esc(item.get('decision_id'))}</td>"
            f"<td>{_esc(human or '--')}</td>"
            "</tr>"
        )
    body.append(f"<table>{header}{''.join(rows)}</table>")
    body.append(_bullets([
        "数值计算与指标判定由程序完成。",
        "LLM 负责科学文本、假设类型判断、不确定性识别与参数建议。",
        "人工 PI 拥有最终批准权；支持度与状态改动保留审计链，可撤销质询造成的衰减。",
    ]))
    return "<section>" + "".join(body) + "</section>"


def _h2(text: str) -> str:
    return f'<h2 class="section-title">{_esc(text)}</h2>'


def _h3(text: str) -> str:
    return f'<h3 class="sub-title">{_esc(text)}</h3>'


def _build_round_section(round_number: int) -> str:
    bundle = load_round_bundle(round_number)
    if not bundle.get("available"):
        return (
            f'<section class="round-section"><h2 class="section-title">第 {round_number} 轮</h2>'
            '<p class="muted">该轮尚无可用报告数据。</p></section>'
        )
    return (
        '<section class="round-section">'
        + "".join(
            [
                _cover_section(bundle),
                _research_goal_section(bundle),
                _hypothesis_section(bundle),
                _uncertainty_candidate_section(bundle),
                _execution_section(bundle),
                _four_layer_section(bundle),
                _next_advice_section(bundle),
                _conclusion_section(bundle),
                _audit_section(bundle),
            ]
        )
        + "</section>"
    )


def build_round_report_html(round_number: int) -> str:
    body = _build_round_section(int(round_number))
    return _page_wrap(f"Shadow Tracing · 第 {round_number} 轮实验结果报告", body)


def build_all_rounds_report_html() -> str:
    latest = current_round()
    sections: list[str] = []
    for round_number in range(1, latest + 1):
        section = _build_round_section(round_number)
        if round_number > 1:
            section = f'<div class="page-break"></div>{section}'
        sections.append(section)
    return _page_wrap("Shadow Tracing · 全部轮次实验结果报告", "".join(sections))


def _round_history_entry(bundle: dict[str, Any], round_number: int) -> dict[str, Any]:
    entries = bundle.get("round_history", {}).get("entries") or []
    matches = [item for item in entries if item.get("round_id") == round_number]
    return matches[-1] if matches else {}


def _summary_text(value: Any, limit: int = 220) -> str:
    return _to_text(value, max_len=limit) if value not in (None, "", [], {}) else "--"


def _cross_round_overview_html() -> str:
    latest = current_round()
    rows: list[str] = []
    for round_number in range(1, latest):
        current_bundle = load_round_bundle(round_number)
        next_bundle = load_round_bundle(round_number + 1)
        if not current_bundle.get("available") or not next_bundle.get("available"):
            continue
        current_entry = _round_history_entry(current_bundle, round_number)
        next_entry = _round_history_entry(next_bundle, round_number + 1)
        findings = current_entry.get("scientific_findings") or []
        rationale = next_entry.get("selection_rationale")
        changed = (
            next_entry.get("planned_changed_factors")
            or next_entry.get("actual_changed_factors")
            or []
        )
        delta_notes = next_entry.get("protocol_delta_notes") or []
        new_findings = next_entry.get("scientific_findings") or []

        changed_parts = []
        if rationale:
            changed_parts.append(_to_text(rationale, 220))
        if changed:
            changed_parts.append("改变：{}".format("；".join(str(item) for item in changed[:4])))
        if delta_notes:
            changed_parts.append("协议核对：{}".format("；".join(str(item) for item in delta_notes[:2])))
        if not changed_parts:
            changed_parts.append("尚未记录明确的决策差异")

        rows.append(
            "<tr>"
            f"<td>{_esc(_summary_text('；'.join(str(item) for item in findings[:2])))}</td>"
            f"<td>{_esc('；'.join(changed_parts))}</td>"
            f"<td>{_esc(_summary_text('；'.join(str(item) for item in new_findings[:2])))}</td>"
            "</tr>"
        )
    if not rows:
        return '<p class="muted">尚无足够的相邻轮次用于跨轮概览。</p>'
    header = "<tr><th>第一轮发现</th><th>因此改变什么</th><th>新一轮新证据</th></tr>"
    return f"<table>{header}{''.join(rows)}</table>"


def build_defense_summary_html() -> str:
    """Build the compact six-item defense summary for all completed rounds."""

    latest = current_round()
    body = [_h2("答辩摘要 · 跨轮实验闭环")]
    body.append(
        '<p class="muted">本页只表达：检验谁、疑问来源、为什么选、实际怎么做、'
        "得到什么、改变什么。完整指标、图与审计链保留在长报告中。</p>"
    )
    body.append(_h3("0. 跨轮概览"))
    body.append(_cross_round_overview_html())

    for round_number in range(1, latest + 1):
        bundle = load_round_bundle(round_number)
        if not bundle.get("available"):
            continue
        entry = _round_history_entry(bundle, round_number)
        protocol = bundle.get("protocol", {})
        evaluation = bundle.get("evaluation", {})
        raw_mapping = _hypothesis_display_map(bundle)
        tested = list(
            dict.fromkeys(
                entry.get("tested_hypotheses")
                or protocol.get("tested_hypotheses")
                or []
            )
        )
        tested_labels = [_short_hypothesis_label(bundle, raw_mapping, str(item)) for item in tested]

        source_parts = []
        if entry.get("source_round_id") is not None:
            source_parts.append(f"上一轮：第 {entry['source_round_id']} 轮")
        source_findings = entry.get("source_findings")
        if source_findings:
            source_parts.append("发现：" + _to_text("；".join(str(item) for item in source_findings[:2]), 260))
        uncertainties = entry.get("target_uncertainty_ids") or entry.get("target_uncertainties") or protocol.get("target_uncertainties") or []
        if uncertainties:
            source_parts.append("目标不确定性：" + "、".join(str(item) for item in uncertainties[:5]))
        if not source_parts:
            source_parts.append("首轮：问题与不确定性初始化，不依赖上一轮实验")

        why_parts = []
        rationale = entry.get("selection_rationale") or protocol.get("responded_prior_finding")
        if rationale:
            why_parts.append(_to_text(rationale, 260))
        planned = entry.get("planned_changed_factors") or entry.get("actual_changed_factors") or []
        if planned:
            why_parts.append("改变因素：" + "；".join(str(item) for item in planned[:5]))
        if entry.get("candidate_id"):
            why_parts.append(f"候选：{entry['candidate_id']}")
        if not why_parts:
            why_parts.append("本轮无独立选择记录，需人工补充")

        snapshot = entry.get("actual_config_snapshot") or {}
        actual_parts = []
        experiment_type = snapshot.get("experiment_type") or protocol.get("experiment_type") or entry.get("executed_experiment_id") or "--"
        actual_parts.append(f"实验类型：{experiment_type}")
        target = snapshot.get("target") or protocol.get("target")
        if target:
            actual_parts.append(f"目标：{target}")
        horizon = snapshot.get("forecast_horizon_days")
        if horizon is not None:
            actual_parts.append(f"预测超前期：{int(horizon)} 天")
        past_lag = snapshot.get("past_lag_days")
        if past_lag is not None:
            actual_parts.append(f"历史滞后：{int(past_lag)} 天")
        window = snapshot.get("window_size")
        if window is not None:
            actual_parts.append(f"时间窗口：{int(window)} 天")
        families = snapshot.get("model_families") or []
        if families:
            actual_parts.append("模型：" + "/".join(str(item) for item in families))
        run_count = snapshot.get("run_count") or len(protocol.get("runs_spec") or [])
        contrast_count = snapshot.get("contrast_count") or len(protocol.get("contrasts_spec") or [])
        actual_parts.append(f"运行组数：{run_count or '--'}，对比数：{contrast_count or '--'}")
        if entry.get("actual_changed_factors"):
            actual_parts.append("实际执行：" + "；".join(str(item) for item in entry["actual_changed_factors"][:5]))
        delta_notes = entry.get("protocol_delta_notes") or []
        if delta_notes:
            actual_parts.append("协议核对：" + "；".join(str(item) for item in delta_notes[:3]))

        findings = entry.get("scientific_findings") or []
        result_parts = [_to_text(str(item), 320) for item in findings[:3]] or ["本轮结果未落盘"]
        assessments = evaluation.get("hypothesis_assessments") or []
        if assessments:
            result_parts.append(
                "结构化判定：" + _to_text(
                    [
                        {
                            "假设": item.get("hypothesis_id"),
                            "分类": item.get("evidence_classification"),
                            "标签": item.get("evidence_labels"),
                            "更新": item.get("updated"),
                        }
                        for item in assessments[:3]
                    ],
                    320,
                )
            )

        commit_decision = _latest_commit_decision(bundle, round_id=round_number)
        decision_label = _commit_decision_label(commit_decision)
        change_parts = []
        if commit_decision and commit_decision.get("decision_type") == "terminate_workflow":
            change_parts.append("停止工作流（人工或预算决定，不视为科学问题已解决）")
        elif decision_label not in ("--",):
            change_parts.append(decision_label)
        elif entry.get("status") == "stopped":
            change_parts.append("停止（人工或预算原因，不视为科学收敛）")
        else:
            change_parts.append("状态写入完成；依据证据分类决定是否调整支持度")
        if entry.get("protocol_delta_notes"):
            change_parts.append("计划与执行差异已记录，后续结论以编译后的实际参数为准")
        if not change_parts:
            change_parts.append("待人工决策")

        body.append(_h3(f"第 {round_number} 轮"))
        body.append(
            '<table class="kv-table summary-table">'
            f"<tr><th>检验谁</th><td>{_esc('、'.join(tested_labels) or '--')}</td></tr>"
            f"<tr><th>疑问来源</th><td>{_esc('；'.join(source_parts))}</td></tr>"
            f"<tr><th>为什么选</th><td>{_esc('；'.join(why_parts))}</td></tr>"
            f"<tr><th>实际怎么做</th><td>{_esc('；'.join(actual_parts))}</td></tr>"
            f"<tr><th>得到什么</th><td>{_esc('；'.join(result_parts))}</td></tr>"
            f"<tr><th>改变什么</th><td>{_esc('；'.join(change_parts))}</td></tr>"
            "</table>"
        )
        body.append("<h4>本轮关键图</h4>")
        experiment_id = bundle.get("result", {}).get("experiment_id")
        defense_charts = _presentation_figures_html(
            round_number,
            experiment_id,
            include=("v3_gain_ci.png", "v3_test_prediction_comparison.png"),
            compact=True,
        )
        body.append(
            defense_charts
            or '<p class="muted">本轮核心结果图尚未生成。</p>'
        )
    return _page_wrap("Shadow Tracing · 答辩摘要", "".join(body))


def _chrome_candidates() -> list[Path]:
    candidates: list[Path] = []
    for env_var in ("CHROME_PATH", "CHROMIUM_PATH", "EDGE_PATH", "BROWSER_PATH"):
        value = os.environ.get(env_var)
        if value:
            candidates.append(Path(value))
    common_paths = [
        "C:/Program Files/Google/Chrome/Application/chrome.exe",
        "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
        "C:/Program Files/Microsoft/Edge/Application/msedge.exe",
        "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
        "/usr/bin/google-chrome",
        "/usr/bin/chromium-browser",
        "/usr/bin/chromium",
        "/snap/bin/chromium",
    ]
    candidates.extend(Path(path) for path in common_paths)
    for name in ("google-chrome", "chromium", "chromium-browser", "chrome", "msedge", "microsoft-edge"):
        found = shutil.which(name)
        if found:
            candidates.append(Path(found))
    seen: set[str] = set()
    result: list[Path] = []
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=False)
        except Exception:
            continue
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        if resolved.is_file():
            result.append(resolved)
    return result


def _pdf_stem(file_name_stem: str) -> str:
    stem = str(file_name_stem)
    if stem.lower().endswith(".pdf"):
        stem = stem[:-4]
    return stem


def render_report_pdf(html_text: str, file_name_stem: str) -> tuple[bytes | None, bool]:
    """Render HTML to PDF via headless Chromium.

    Returns ``(pdf_bytes, used_html_fallback)``.  When no browser is available
    the caller falls back to serving the HTML document itself.
    """
    file_name_stem = _pdf_stem(file_name_stem)
    export_dir = LIVE_ROOT / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    html_path = export_dir / f"{file_name_stem}.html"
    pdf_path = export_dir / f"{file_name_stem}.pdf"
    html_path.write_text(html_text, encoding="utf-8")
    if pdf_path.exists():
        try:
            pdf_path.unlink()
        except OSError:
            pass

    browsers = _chrome_candidates()
    headless_modes = (["--headless=new", "--no-pdf-header-footer"], ["--headless", "--print-to-pdf-no-header"])
    for browser in browsers:
        for extra_flags in headless_modes:
            profile_dir: str | None = None
            try:
                profile_dir = tempfile.mkdtemp(prefix="shadow_chrome_", dir=str(export_dir))
                command = [
                    str(browser),
                    "--disable-gpu",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    f"--user-data-dir={profile_dir}",
                    *extra_flags,
                    f"--print-to-pdf={pdf_path}",
                    html_path.as_uri(),
                ]
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    timeout=120,
                    check=False,
                )
                if pdf_path.is_file() and pdf_path.stat().st_size > 0:
                    return pdf_path.read_bytes(), False
            except Exception:
                continue
            finally:
                if profile_dir:
                    shutil.rmtree(profile_dir, ignore_errors=True)
    return None, True


def report_export_filename(scope: str, round_number: int | None = None) -> str:
    if scope == "all":
        return "ShadowTracing_All_Rounds_Report.pdf"
    if scope == "summary":
        return "ShadowTracing_Defense_Summary.pdf"
    return f"ShadowTracing_Round_{int(round_number or current_round()):02d}_Report.pdf"


def _page_wrap(title: str, body: str) -> str:
    css = """
* { box-sizing: border-box; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
@page { size: A4; margin: 12mm 10mm; }
body {
  margin: 0; padding: 26px 30px 50px; background: #FAFAF7; color: #0F2142;
  font-family: "Microsoft YaHei", "PingFang SC", "Segoe UI", "Noto Sans CJK SC", Arial, sans-serif;
  font-size: 13px; line-height: 1.65;
}
h1 { font-size: 27px; margin: 0 0 12px; color: #0F2142; line-height: 1.25; }
.section-title { font-size: 19px; margin: 26px 0 12px; padding: 3px 0 3px 10px; border-left: 4px solid #275B94; color: #0F2142; }
.sub-title { font-size: 15px; margin: 18px 0 8px; color: #15375F; }
h4 { font-size: 13.5px; margin: 15px 0 6px; color: #15375F; }
p { margin: 6px 0; }
.report-cover { padding: 16px 0 14px; border-bottom: 2px solid #275B94; margin-bottom: 16px; }
.kicker { color: #275B94; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; margin: 0 0 8px; font-size: 12px; }
.question-line { color: #36567e; font-size: 14px; margin: 0 0 12px; }
.warning-banner { background: #FBEAE5; border: 1px solid #C4552F; color: #7D2D14; border-radius: 8px; padding: 10px 12px; margin: 0 0 12px; font-size: 12px; line-height: 1.55; }
table { width: 100%; border-collapse: collapse; margin: 10px 0 14px; font-size: 11.5px; table-layout: fixed; }
th, td { border: 1px solid #E3E1D8; padding: 6px 8px; vertical-align: top; text-align: left; overflow-wrap: anywhere; word-break: break-word; }
th { background: #F1EFE8; color: #0F2142; font-weight: 700; }
tr:nth-child(even) td { background: #FCFBF8; }
table.kv-table { table-layout: auto; }
table.kv-table th { width: 34%; background: transparent; color: #36567E; }
table.kv-table th, table.kv-table td { border: none; border-bottom: 1px solid #E8E6DE; padding: 7px 8px; }
table.kv-table tr:nth-child(even) td { background: transparent; }
table.summary-table th { width: 96px; background: transparent; color: #36567E; font-size: 11.5px; }
.llm-quote { background: #F4F3EE; border-left: 3px solid #C7A46A; padding: 10px 12px; margin: 10px 0; white-space: pre-wrap; border-radius: 0 8px 8px 0; }
.quote-label { display: block; font-size: 11px; color: #8A764F; font-weight: 700; margin-bottom: 4px; }
.pill { display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 11px; font-weight: 600; border: 1px solid currentColor; }
.figure-grid { display: grid; grid-template-columns: 1fr; gap: 14px; margin: 12px 0; }
.figure-grid--compact { grid-template-columns: 1fr 1fr; gap: 12px; }
.figure-grid--compact .figure { padding: 7px; }
.figure-grid--compact .figure img { max-height: 430px; }
.figure { margin: 0; border: 1px solid #E3E1D8; background: #fff; padding: 10px; border-radius: 8px; break-inside: avoid; text-align: center; }
.figure img { display: block; max-width: 100%; width: auto; height: auto; max-height: 640px; margin: 0 auto; object-fit: contain; }
figcaption { font-size: 11px; color: #66708A; margin-top: 5px; }
.muted { color: #6B7990; }
.page-break { break-before: page; height: 0; }
ul { margin: 6px 0 12px; padding-left: 20px; }
li { margin: 2px 0; }
@media print { .figure-grid { break-inside: auto; } }
"""
    return (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"/>'
        f"<title>{html.escape(title)}</title><style>{css}</style></head><body>{body}</body></html>"
    )
