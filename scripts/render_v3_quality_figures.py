"""Render calendar-faithful v3 figures from the active live session archive.

Archived prediction CSVs only contain the test segment, so this script also
reconstructs the training-segment predictions by refitting each arm with the
archived best parameters. It validates the regenerated test predictions against
the archived CSVs before drawing and never changes experiment evaluation state.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.quality_plots import (
    TARGET_COL,
    compute_quality_stats,
    load_calendar_frame,
    load_sample_manifest,
    plot_prediction_scatter_quality,
    plot_shadow_observation_series,
)
from models import elasticnet_runner as runner


RUN_STYLES = [
    ("ridge_baseline", "Ridge 对照组", "#c0392b"),
    ("ridge_treatment", "Ridge 实验组", "#e67e22"),
    ("elasticnet_baseline", "ElasticNet 对照组", "#2471a3"),
    ("elasticnet_treatment", "ElasticNet 实验组", "#148f77"),
]


def _read_arm_config(
    archive_dir: Path,
    run_id: str,
) -> runner.ExperimentConfig:
    path = archive_dir / run_id / "config.json"
    if not path.exists():
        raise FileNotFoundError(f"缺少归档 run 配置: {path}")
    return runner.ExperimentConfig.model_validate(
        json.loads(path.read_text(encoding="utf-8"))
    )


def _archived_best_params(
    archive_dir: Path,
    run_id: str,
) -> tuple[float, float | None]:
    path = archive_dir / run_id / "param_search.csv"
    if not path.exists():
        raise FileNotFoundError(f"缺少归档参数搜索记录: {path}")
    search = pd.read_csv(path)
    best = search.loc[search["rank"] == 1]
    if len(best) != 1:
        raise ValueError(f"{run_id} 档案中 rank=1 的记录数异常: {len(best)}")
    row = best.iloc[0]
    alpha = float(row["param_alpha"])
    l1_raw = row.get("param_l1_ratio")
    l1 = None if l1_raw is None or pd.isna(l1_raw) else float(l1_raw)
    return alpha, l1


def reconstruct_v3_prediction_frames(
    *,
    experiment_id: str,
    archive_dir: Path,
    predictions_dir: Path,
    output_dir: Path,
) -> list[dict]:
    """Refit archived arms and merge reconstructed train predictions with tests."""

    manifest = load_sample_manifest(archive_dir)
    complete = manifest.loc[manifest["drop_reason"].isna()].copy()
    complete["target_date"] = pd.to_datetime(
        complete["target_date"], errors="coerce"
    ).dt.normalize()
    if complete.empty:
        raise ValueError("归档 sample_manifest 中没有任何保留样本")

    shared_keys = {
        (
            row.issue_date.date().isoformat(),
            row.target_date.date().isoformat(),
            int(row.horizon_days),
        )
        for row in complete.itertuples()
    }
    arm_configs = {
        run_id: _read_arm_config(archive_dir, run_id)
        for run_id, _, _ in RUN_STYLES
    }
    union_features = list(
        dict.fromkeys(
            column
            for config in arm_configs.values()
            for column in config.file_config.feature_columns
        )
    )
    first = arm_configs[RUN_STYLES[0][0]]
    shared_config = first.model_copy(
        update={
            "file_config": first.file_config.model_copy(
                update={"feature_columns": union_features}
            )
        }
    )
    merged = runner.load_and_merge_data(shared_config)

    predictions: list[dict] = []
    audit_rows: list[dict] = []
    for run_id, label, color in RUN_STYLES:
        config = arm_configs[run_id]
        alpha, l1_ratio = _archived_best_params(archive_dir, run_id)
        window = runner.build_sliding_window(
            merged.X,
            merged.y,
            merged.time,
            shared_config.time_window_config,
            only_keys=shared_keys,
            feature_columns=list(config.file_config.feature_columns),
        )
        records = window.records_all
        split_by_sample = dict(zip(complete["sample_id"], complete["split"]))
        splits = [
            split_by_sample.get(record.sample_id)
            for record in records
        ]
        unmatched = sum(
            1
            for value in splits
            if value not in ("train", "test", "boundary_excluded")
        )
        if unmatched:
            raise ValueError(
                f"{run_id} 有 {unmatched} 个重建样本未匹配归档 manifest"
            )
        train_mask = np.asarray([value == "train" for value in splits])
        test_mask = np.asarray([value == "test" for value in splits])
        boundary_mask = np.asarray(
            [value == "boundary_excluded" for value in splits]
        )
        x_train = pd.DataFrame(
            window.X_all[train_mask],
            columns=window.feature_names,
        )
        x_test = pd.DataFrame(
            window.X_all[test_mask],
            columns=window.feature_names,
        )
        fixed_config = config.elasticnet_config.model_copy(
            update={
                "model_family": config.elasticnet_config.model_family,
                "alpha": alpha,
                "l1_ratio": (
                    l1_ratio
                    if l1_ratio is not None
                    else float(config.elasticnet_config.l1_ratio)
                ),
                "param_search_mode": "fixed",
            }
        )
        model, _ = runner.train_elasticnet_model_with_search(
            x_train,
            window.y_all[train_mask],
            fixed_config,
        )
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)

        archived_test = pd.read_csv(
            predictions_dir / f"{experiment_id}_{run_id}.csv"
        )
        boundary_records = [
            record
            for record, keep in zip(records, boundary_mask)
            if keep
        ]
        boundary_frame = pd.DataFrame(
            {
                "sample_id": [
                    record.sample_id for record in boundary_records
                ],
                "issue_date": [
                    record.issue_date for record in boundary_records
                ],
                "target_date": [
                    record.target_date for record in boundary_records
                ],
                "y_true": window.y_all[boundary_mask],
                "y_pred": np.nan,
                "split": "boundary_excluded",
                "source": "no_prediction_boundary",
            }
        )
        test_records = [
            record
            for record, keep in zip(records, test_mask)
            if keep
        ]
        test_index = {
            record.sample_id: index
            for index, record in enumerate(test_records)
        }
        order = [
            test_index[sample_id]
            for sample_id in archived_test["sample_id"]
        ]
        regenerated_test = y_test_pred[np.asarray(order, dtype=int)]
        max_abs_diff = float(
            np.max(
                np.abs(
                    regenerated_test
                    - archived_test["y_pred"].to_numpy(dtype=float)
                )
            )
        )
        if max_abs_diff > 1e-6:
            raise ValueError(
                f"{run_id} 重建测试预测与归档不一致 (max|diff|={max_abs_diff})"
            )

        train_records = [
            record
            for record, keep in zip(records, train_mask)
            if keep
        ]
        train_frame = pd.DataFrame(
            {
                "sample_id": [record.sample_id for record in train_records],
                "issue_date": [record.issue_date for record in train_records],
                "target_date": [record.target_date for record in train_records],
                "y_true": window.y_all[train_mask],
                "y_pred": y_train_pred,
                "split": "train",
                "source": "reconstructed_train",
            }
        )
        test_frame = archived_test.copy()
        test_frame["split"] = "test"
        test_frame["source"] = "archived_test"
        test_frame["issue_date"] = pd.to_datetime(
            test_frame["issue_date"], errors="coerce"
        ).dt.normalize()
        test_frame["target_date"] = pd.to_datetime(
            test_frame["target_date"], errors="coerce"
        ).dt.normalize()
        test_frame["y_true"] = pd.to_numeric(
            test_frame["y_true"], errors="coerce"
        )
        test_frame["y_pred"] = pd.to_numeric(
            test_frame["y_pred"], errors="coerce"
        )
        combined = pd.concat(
            [train_frame, boundary_frame, test_frame],
            ignore_index=True,
        )
        combined["_date"] = pd.to_datetime(combined["issue_date"])
        combined = combined.sort_values(
            ["_date", "split", "sample_id"]
        ).reset_index(drop=True)
        combined = combined.assign(
            run_id=run_id,
            label=label,
            color=color,
        )
        predictions.append(
            {
                "run_id": run_id,
                "label": label,
                "color": color,
                "frame": combined,
            }
        )
        audit_rows.append(
            {
                "run_id": run_id,
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
                "archived_test_max_abs_diff": max_abs_diff,
                "validated": True,
            }
        )

    audit = pd.DataFrame(audit_rows)
    audit_path = output_dir / "v3_prediction_reconstruction_audit.csv"
    audit_path.write_text(
        audit.to_csv(index=False),
        encoding="utf-8-sig",
    )
    reconstruction_path = output_dir / "v3_prediction_reconstruction.csv"
    reconstruction_path.write_text(
        pd.concat(
            [item["frame"] for item in predictions],
            ignore_index=True,
        ).to_csv(index=False),
        encoding="utf-8-sig",
    )
    print(f"reconstruction audit -> {audit_path}")
    print(f"reconstructed rows -> {reconstruction_path}")
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--session",
        default=ROOT / "runtime" / "live_session" / "sessions" / "session_20261015_132955_705440",
        help="live session root",
    )
    parser.add_argument(
        "--round",
        default="round_02",
        help="round directory to read predictions from",
    )
    parser.add_argument(
        "--experiment",
        default="E_R02_03",
        help="experiment id whose prediction CSVs are used",
    )
    parser.add_argument(
        "--output",
        default=ROOT / "artifacts" / "v3_quality_figures",
        help="output directory for figures and audit json",
    )
    args = parser.parse_args()

    session = Path(args.session)
    raw = session / "data" / "raw"
    omni_path = raw / "omni_2021_full_calendar_v3.csv"
    lhaaso_path = raw / "lhaaso_shadow_pfss_2021_full_calendar_v3.csv"
    archive_dir = session / "results" / args.round / "archive" / f"{args.experiment}_r1"
    predictions_dir = session / "results" / args.round / args.experiment / "predictions"

    if not omni_path.exists() or not lhaaso_path.exists():
        raise FileNotFoundError(f"v3 raw data not found under {raw}")
    if not predictions_dir.exists():
        raise FileNotFoundError(f"predictions not found: {predictions_dir}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    calendar = load_calendar_frame(omni_path, lhaaso_path)
    manifest = load_sample_manifest(archive_dir) if archive_dir.exists() else None
    stats = compute_quality_stats(calendar, manifest)

    predictions = reconstruct_v3_prediction_frames(
        experiment_id=args.experiment,
        archive_dir=archive_dir,
        predictions_dir=predictions_dir,
        output_dir=output_dir,
    )

    audit_path = output_dir / "quality_audit.json"
    audit_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"audit -> {audit_path}")

    shadow_fig = plot_shadow_observation_series(
        calendar,
        output_dir / "v3_shadow_observation_series.png",
        stats=stats,
    )
    pred_fig = plot_prediction_scatter_quality(
        calendar,
        predictions,
        output_dir / "v3_prediction_vs_truth.png",
        stats=stats,
        manifest=manifest,
        target_col=TARGET_COL,
    )
    print(f"shadow figure -> {shadow_fig}")
    print(f"prediction figure -> {pred_fig}")


if __name__ == "__main__":
    main()
