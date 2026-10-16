import json
import pathlib
import sys


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "candidates"
    root = pathlib.Path(
        r"D:\ShadowTracing\runtime\live_session\sessions"
        r"\session_20261015_132955_705440"
    )
    if mode == "result":
        result_path = root / "results" / "round_01" / "result_unified.json"
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        print("experiment_id:", payload.get("experiment_id"))
        print("experiment_type:", payload.get("experiment_type"))
        print("status:", payload.get("status"), "validity:", payload.get("validity_status"))
        print("round:", payload.get("round_id"))
        print("duration_seconds:", payload.get("duration_seconds"))
        print()
        for run in payload.get("runs", []):
            metrics = run.get("metrics", {})
            print(
                "RUN",
                run.get("run_id"),
                "| model",
                run.get("model_family"),
                "| label",
                run.get("name"),
                "| train",
                run.get("n_train"),
                "| test",
                run.get("n_test"),
                "| mode",
                run.get("training_mode"),
            )
            print("   features:", run.get("feature_columns"))
            print(
                "   rmse",
                metrics.get("test_rmse"),
                "| mae",
                metrics.get("test_mae"),
                "| pearson",
                metrics.get("test_pearson_r"),
                "| r2",
                metrics.get("test_r2"),
            )
            print("   pred artifact:", run.get("prediction_artifact"))
        print()
        for contrast in payload.get("contrasts", []):
            print(
                "CONTRAST",
                contrast.get("contrast_id"),
                "| runs",
                contrast.get("run_ids"),
                "| kind",
                contrast.get("kind"),
            )
            delta = contrast.get("metrics_delta", {})
            print("   delta:", json.dumps(delta, ensure_ascii=False))
            print("   bootstrap:", json.dumps(contrast.get("bootstrap_ci"), ensure_ascii=False))
        print()
        ro = payload.get("robustness")
        if ro is not None:
            print("robustness:", json.dumps(ro, ensure_ascii=False, indent=2)[:6000])
        print()
        print("diagnostics:", json.dumps(payload.get("diagnostics"), ensure_ascii=False, indent=2)[:5000])
        print()
        print("warnings:", json.dumps(payload.get("warnings"), ensure_ascii=False, indent=2))
        print("visualizations count:", len(payload.get("visualizations") or []))
        return

    payload = json.loads(
        (root / "state" / "candidate_experiments.json").read_text(encoding="utf-8")
    )
    for cand in payload["candidates"]:
        print("=" * 120)
        print(
            cand.get("experiment_id"),
            "|",
            cand.get("experiment_type"),
            "| utility",
            cand.get("utility_score"),
        )
        design = cand.get("design", {})
        print("  control:", json.dumps(design.get("control"), ensure_ascii=False))
        print("  treatment:", json.dumps(design.get("treatment"), ensure_ascii=False))
        print(
            "  windows:",
            {
                key: design.get(key)
                for key in (
                    "forecast_horizon_days",
                    "past_lag_days",
                    "window_size",
                    "control_lag_days",
                    "treatment_lag_days",
                )
            },
        )
        print("  model_families:", design.get("model_families"))
        hp = cand.get("hypothesis_predictions", {})
        print("  predictions:")
        for hid, pred in hp.items():
            print("   ", hid, json.dumps(pred, ensure_ascii=False))


if __name__ == "__main__":
    main()
