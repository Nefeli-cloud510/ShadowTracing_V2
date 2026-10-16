"""Independent acceptance check for v3 files against raw source files."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


OMNI_V3 = Path(r"D:\ShadowTracing\runtime\live_run_input\omni_2021_full_calendar_v3.csv")
SHADOW_V3 = Path(r"D:\ShadowTracing\runtime\live_run_input\lhaaso_shadow_pfss_2021_full_calendar_v3.csv")
OMNI_RAW = Path(r"D:\helioprior\data\out_of_ICME_raw_OMNI_Daily_DATA(1).xlsx")
SHADOW_RAW = Path(r"D:\helioprior\data\raw_out_of_ICME_sunshadow2.xlsx")
PFSS_V1 = Path(r"D:\astro_rearrange\data\analysis\pfss_basic_features_daily_v1.csv")
CAL = Path(r"D:\sunshadow\calender_internal_2020_2024.xlsx")
OUT = Path(r"D:\ShadowTracing\artifacts\data_issue_scan\v3_acceptance_report.json")


def dnorm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()


def exact_report(merged: pd.DataFrame, fields: list[str], suffix: str) -> list[dict]:
    reports = []
    for f in fields:
        lv = pd.to_numeric(merged[f], errors="coerce")
        rv = pd.to_numeric(merged[f"{f}{suffix}"], errors="coerce")
        cond = lv == rv
        reports.append(
            {
                "field": f,
                "n": int(lv.notna().sum()),
                "n_match_exact": int((lv.notna() & rv.notna() & cond).sum()),
                "max_abs_diff": float((lv - rv).abs().max()) if lv.notna().any() else None,
            }
        )
    return reports


def main() -> None:
    omni = pd.read_csv(OMNI_V3)
    shadow = pd.read_csv(SHADOW_V3)
    omni["TIME"] = dnorm(omni["TIME"])
    shadow["TIME"] = dnorm(shadow["TIME"])
    cal = pd.read_excel(CAL, sheet_name=0)
    cal["d"] = dnorm(cal["data_utc"])
    cal = cal.drop_duplicates("d", keep="last")
    omni_raw = pd.read_excel(OMNI_RAW, sheet_name=0)
    omni_raw["d"] = dnorm(omni_raw["TIME"])
    omni_raw = omni_raw.drop_duplicates("d", keep="last")
    shadow_raw = pd.read_excel(SHADOW_RAW, sheet_name=0)
    shadow_raw["d"] = dnorm(shadow_raw["TIME"])
    shadow_raw = shadow_raw.drop_duplicates("d", keep="last")
    pfss = pd.read_csv(PFSS_V1)
    pfss["d"] = dnorm(pfss["date_str"])

    omni_cal = omni.merge(cal[["d"] + omni.columns.drop("TIME").tolist()], left_on="TIME", right_on="d", suffixes=("", "_cal"))
    omni_raw_g = omni.merge(omni_raw[["d", "Scalar B, nT", "SW Plasma Temperature, K", "SW Proton Density, N/cm^3", "SW Plasma Speed, km/s"]],
                            left_on="TIME", right_on="d", suffixes=("", "_raw"))

    shadow_cal = shadow.merge(cal[["d", "p2_y中心值", "p2_y中心值误差", "p1_x中心值", "p1_x中心值误差", "p3 sigma_宽度", "_llf 最优卡方值"]],
                              left_on="TIME", right_on="d", suffixes=("", "_cal"))
    shadow_raw_g = shadow.merge(shadow_raw[["d", "p2_y中心值", "p2_y中心值误差", "p1_x中心值", "p1_x中心值误差", "p3 sigma_宽度", "_llf 最优卡方值"]],
                                left_on="TIME", right_on="d", suffixes=("", "_raw"))

    pfss_cols = ["l0_04utc_deg", "photo_br_mean", "photo_br_absmean", "photo_n_lat", "photo_n_lon", "ss_br_mean", "ss_br_absmean", "ss_n_lat", "ss_n_lon"]
    shadow_pfss = shadow.merge(pfss[["d"] + pfss_cols], left_on="TIME", right_on="d", suffixes=("", "_pfss"))

    report = {
        "omni_v3_vs_calendar": exact_report(omni_cal, omni.columns.drop("TIME").tolist(), "_cal"),
        "omni_v3_core_vs_raw_xlsx": exact_report(omni_raw_g, ["Scalar B, nT", "SW Plasma Temperature, K", "SW Proton Density, N/cm^3", "SW Plasma Speed, km/s"], "_raw"),
        "shadow_v3_vs_calendar": exact_report(shadow_cal, ["p2_y中心值", "p2_y中心值误差", "p1_x中心值", "p1_x中心值误差", "p3 sigma_宽度", "_llf 最优卡方值"], "_cal"),
        "shadow_v3_vs_raw_xlsx": exact_report(shadow_raw_g, ["p2_y中心值", "p2_y中心值误差", "p1_x中心值", "p1_x中心值误差", "p3 sigma_宽度", "_llf 最优卡方值"], "_raw"),
        "shadow_v3_pfss_vs_pfss_v1": exact_report(shadow_pfss, pfss_cols, "_pfss"),
    }
    (OUT).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
