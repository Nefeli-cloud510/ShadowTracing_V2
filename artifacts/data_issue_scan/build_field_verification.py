"""Verify execution CSV columns against authoritative daily sources.

Read-only against reference files outside ShadowTracing. Writes audit artifacts
under the repository artifacts/ folder.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


OUT_DIR = Path(r"D:\ShadowTracing\artifacts\data_issue_scan")
AUTHORITATIVE_OMNI = Path(
    r"D:\sunshadow\calender_internal_2020_2024.xlsx"
)
AUTHORITATIVE_OMNI_ALT = Path(
    r"D:\helioprior\data\out_of_ICME_raw_OMNI_Daily_DATA(1).xlsx"
)
AUTHORITATIVE_SHADOW = Path(
    r"D:\sunshadow\calender_internal_2020_2024.xlsx"
)
AUTHORITATIVE_SHADOW_ALT = Path(
    r"D:\helioprior\data\raw_out_of_ICME_sunshadow2.xlsx"
)
LIVE_OMNI = Path(
    r"D:\ShadowTracing\runtime\live_run_input\omni_2021_full_calendar_v2.csv"
)
LIVE_SHADOW = Path(
    r"D:\ShadowTracing\runtime\live_run_input\lhaaso_shadow_pfss_2021_full_calendar_v2.csv"
)

OMNI_FIELDS = [
    "Scalar B, nT",
    "BX, nT (GSE, GSM)",
    "BY, nT (GSE)",
    "BZ, nT (GSE)",
    "BY, nT (GSM)",
    "BZ, nT (GSM)",
    "SW Plasma Temperature, K",
    "SW Proton Density, N/cm^3",
    "SW Plasma Speed, km/s",
    "Kp index",
    "Dst-index, nT",
    "ap_index, nT",
]

SHADOW_FIELDS = [
    "p2_y中心值",
    "p2_y中心值误差",
    "p1_x中心值",
    "p1_x中心值误差",
    "p3 sigma_宽度",
    "_llf 最优卡方值",
    "l0_04utc_deg",
    "photo_br_mean",
    "photo_br_absmean",
    "photo_n_lat",
    "photo_n_lon",
    "ss_br_mean",
    "ss_br_absmean",
    "ss_n_lat",
    "ss_n_lon",
]


def read_authoritative_omni() -> pd.DataFrame:
    path = AUTHORITATIVE_OMNI if AUTHORITATIVE_OMNI.exists() else AUTHORITATIVE_OMNI_ALT
    df = pd.read_excel(path, sheet_name=0)
    if "data_utc" in df.columns:
        df["DATE_UTC"] = pd.to_datetime(df["data_utc"].astype(str), format="%Y%m%d", errors="coerce")
    else:
        df["DATE_UTC"] = pd.to_datetime(df["TIME"].astype(str), format="%Y%m%d", errors="coerce")
    df["DATE_STR"] = df["DATE_UTC"].dt.strftime("%Y%m%d")
    return df


def read_authoritative_shadow() -> pd.DataFrame:
    df = pd.read_excel(AUTHORITATIVE_SHADOW, sheet_name=0)
    df["DATE_UTC"] = pd.to_datetime(df["data_utc"].astype(str), format="%Y%m%d", errors="coerce")
    df["DATE_STR"] = df["DATE_UTC"].dt.strftime("%Y%m%d")
    return df


def merge_by_date(live: pd.DataFrame, ref: pd.DataFrame, suffix: str = "") -> pd.DataFrame:
    live = live.copy()
    live["DATE_STR"] = live["TIME"].astype(str).str.replace(r"\.0$", "", regex=True)
    ref_c = ref.copy()
    ref_c = ref_c.rename(columns={c: f"{c}{suffix}" for c in ref_c.columns if c != "DATE_STR"})
    return live.merge(ref_c[["DATE_STR"] + [c for c in ref_c.columns if c != "DATE_STR"]], on="DATE_STR", how="left")


def pearson_corr(a: pd.Series, b: pd.Series) -> float:
    mask = a.notna() & b.notna()
    if mask.sum() < 2:
        return float("nan")
    return float(a[mask].corr(b[mask]))


def numeric_report(df: pd.DataFrame, fields: list[str], suffix: str = "") -> dict:
    out: dict[str, dict] = {}
    for col in fields:
        live_col = col
        ref_col = f"{col}{suffix}"
        if live_col not in df.columns or ref_col not in df.columns:
            continue
        live = pd.to_numeric(df[live_col], errors="coerce")
        ref = pd.to_numeric(df[ref_col], errors="coerce")
        same = (live == ref)
        mismatch_mask = live.notna() & ref.notna() & ~same
        out[col] = {
            "n_live": int(live.notna().sum()),
            "n_ref": int(ref.notna().sum()),
            "n_match": int((live == ref).sum()),
            "n_mismatch": int(mismatch_mask.sum()),
            "corr": pearson_corr(live, ref),
            "mean_abs_diff": float((live - ref).abs().mean()) if mismatch_mask.any() else 0.0,
            "example_20210321": {
                "live": None if pd.isna(live.iloc[0]) else float(live.iloc[0]),
                "ref": None if pd.isna(ref.iloc[0]) else float(ref.iloc[0]),
            },
        }
    return out


def identify_shifted_identity(df: pd.DataFrame, fields: list[str], suffix: str = "") -> list[dict]:
    """Report which authoritative field each live column actually resembles."""
    rows = []
    for live_col in fields:
        if live_col not in df.columns:
            continue
        live = pd.to_numeric(df[live_col], errors="coerce")
        best = []
        for ref_col in [c for c in df.columns if c.endswith(suffix) and c != f"DATE_STR{suffix}"]:
            ref = pd.to_numeric(df[ref_col], errors="coerce")
            r = pearson_corr(live, ref)
            if abs(r) >= 0.5 and r == r:
                best.append((r, ref_col))
        best.sort(key=lambda x: -abs(x[0]))
        rows.append(
            {
                "live_column": live_col,
                "top_resemblances": [
                    {"authoritative_field": c, "corr": float(r)} for r, c in best[:3]
                ],
            }
        )
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    live_omni = pd.read_csv(LIVE_OMNI)
    live_shadow = pd.read_csv(LIVE_SHADOW)

    omni_ref = read_authoritative_omni()
    shadow_ref = read_authoritative_shadow()

    omni_merged = merge_by_date(live_omni, omni_ref, suffix="__ref")
    shadow_merged = merge_by_date(live_shadow, shadow_ref, suffix="__ref")

    date_both = set(omni_merged["DATE_STR"]) & set(shadow_merged["DATE_STR"])
    omni_report = numeric_report(omni_merged, OMNI_FIELDS, "__ref")
    shadow_report = numeric_report(shadow_merged, SHADOW_FIELDS, "__ref")
    omni_identity = identify_shifted_identity(omni_merged, OMNI_FIELDS, "__ref")
    shadow_identity = identify_shifted_identity(shadow_merged, SHADOW_FIELDS, "__ref")

    # Print the direct comparison table for 2021-03-21 in a readable form.
    first = omni_merged[omni_merged["DATE_STR"] == "20210321"]
    print("=== OMNI merged rows ===")
    print(f"live days={live_omni.shape[0]} rows, merged days={len(omni_merged)}, ref non-null={len(omni_merged[omni_merged['Scalar B, nT__ref'].notna()])}")
    if not first.empty:
        print("2021-03-21 line:")
        print(first[["TIME", "DATE_STR"] + [f"{c}__ref" for c in OMNI_FIELDS[:6]]].to_string(index=False))
    print("\n=== Field verification (same-name match) ===")
    print(json.dumps(omni_report, ensure_ascii=False, indent=2, default=str))
    print("\n=== Identity cross-correlation ===")
    print(json.dumps(omni_identity, ensure_ascii=False, indent=2, default=str))
    print("\n=== Shadow merged ===")
    print(f"shadow live days={live_shadow.shape[0]} rows, merged={len(shadow_merged)}, ref non-null={len(shadow_merged[shadow_merged['p2_y中心值__ref'].notna()])}")
    print(json.dumps(shadow_report, ensure_ascii=False, indent=2, default=str))
    print(json.dumps(shadow_identity, ensure_ascii=False, indent=2, default=str))

    consolidated = {
        "sources": {
            "live_omni": str(LIVE_OMNI),
            "live_shadow": str(LIVE_SHADOW),
            "authoritative_omni": str(AUTHORITATIVE_OMNI),
            "authoritative_shadow": str(AUTHORITATIVE_SHADOW),
        },
        "date_overlap": len(date_both),
        "omni": omni_report,
        "shadow": shadow_report,
        "identity_resemblances_omni": omni_identity,
        "identity_resemblances_shadow": shadow_identity,
    }
    (OUT_DIR / "field_verification_report.json").write_text(
        json.dumps(consolidated, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    omni_merged.to_csv(OUT_DIR / "omni_live_vs_authoritative_merged.csv", index=False)
    shadow_merged.to_csv(OUT_DIR / "shadow_live_vs_authoritative_merged.csv", index=False)


if __name__ == "__main__":
    main()
