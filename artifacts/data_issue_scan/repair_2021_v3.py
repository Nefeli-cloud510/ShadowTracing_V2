"""Repair the 2021 live execution CSV pair from authoritative sources.

Reads:
  - runtime/live_run_input/omni_2021_full_calendar_v2.csv          (keep date list; values are wrong)
  - runtime/live_run_input/lhaaso_shadow_pfss_2021_full_calendar_v2.csv
  - D:/sunshadow/calender_internal_2020_2024.xlsx                 (OMNI + LHAASO columns)
  - D:/astro_rearrange/data/analysis/pfss_basic_features_daily_v1.csv (photo/ss columns)

Writes versioned v3 files plus this audit file. Reference directories are only
read, never modified.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


RUNTIME = Path(r"D:\ShadowTracing\runtime\live_run_input")
ARTIFACTS = Path(r"D:\ShadowTracing\artifacts\data_issue_scan")
CALENDAR = Path(r"D:\sunshadow\calender_internal_2020_2024.xlsx")
PFSS = Path(r"D:\astro_rearrange\data\analysis\pfss_basic_features_daily_v1.csv")
LIVE_OMNI = RUNTIME / "omni_2021_full_calendar_v2.csv"
LIVE_SHADOW = RUNTIME / "lhaaso_shadow_pfss_2021_full_calendar_v2.csv"

OMNI_OUT = RUNTIME / "omni_2021_full_calendar_v3.csv"
SHADOW_OUT = RUNTIME / "lhaaso_shadow_pfss_2021_full_calendar_v3.csv"
RELATION_CHECK_OUT = ARTIFACTS / "v3_physical_relation_check.json"
PROVENANCE_OUT = ARTIFACTS / "v3_data_provenance.json"


def normalize_date(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()


def corr(a: pd.Series, b: pd.Series) -> float:
    m = a.notna() & b.notna()
    if m.sum() < 2:
        return float("nan")
    return float(a[m].corr(b[m]))


def load_calendar() -> tuple[pd.DataFrame, pd.DataFrame]:
    cal = pd.read_excel(CALENDAR, sheet_name=0)
    cal["d"] = normalize_date(cal["data_utc"])
    cal = cal[cal["d"].str.match(r"^\d{8}$")]

    omni_cols = [
        "d",
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
    shadow_cols = [
        "d",
        "_TS拟合优度",
        "p0振幅",
        "p0误差",
        "p1_x中心值",
        "p1_x中心值误差",
        "p2_y中心值",
        "p2_y中心值误差",
        "p3 sigma_宽度",
        "p3 sigma_宽度误差",
        "_llf 最优卡方值",
    ]
    omni = cal[omni_cols].copy()
    shadow = cal[shadow_cols].copy()
    return omni, shadow


def build_omni_v3(live_omni: pd.DataFrame, omni_ref: pd.DataFrame) -> pd.DataFrame:
    live_dates = normalize_date(live_omni["TIME"])
    omni_ref = omni_ref[omni_ref["d"].isin(set(live_dates))].copy()
    omni_ref = omni_ref.drop_duplicates("d", keep="last")

    out = pd.DataFrame({"TIME": live_dates})
    for col in omni_ref.columns:
        if col == "d":
            continue
        out[col] = out["TIME"].map(dict(zip(omni_ref["d"], omni_ref[col])))
    for col in ["Scalar B, nT", "BX, nT (GSE, GSM)"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["Scalar B, nT"] = out["Scalar B, nT"].where(out["Scalar B, nT"].notna(), out["BX, nT (GSE, GSM)"])
    out["BX, nT (GSE, GSM)"] = out["BX, nT (GSE, GSM)"].where(
        out["BX, nT (GSE, GSM)"].notna(),
        pd.to_numeric(out["Scalar B, nT"], errors="coerce"),
    )

    wanted = [
        "TIME",
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
    return out[wanted]


def build_shadow_v3(
    live_shadow: pd.DataFrame,
    shadow_ref: pd.DataFrame,
    pfss: pd.DataFrame,
) -> pd.DataFrame:
    live_dates = normalize_date(live_shadow["TIME"])
    shadow_ref = shadow_ref[shadow_ref["d"].isin(set(live_dates))].copy()
    shadow_ref = shadow_ref.drop_duplicates("d", keep="last")
    pfss = pfss.copy()
    pfss["d"] = normalize_date(pfss["date_str"])

    out = pd.DataFrame({"TIME": live_dates})
    for col in shadow_ref.columns:
        if col == "d":
            continue
        out[col] = out["TIME"].map(dict(zip(shadow_ref["d"], shadow_ref[col])))

    for col in [
        "l0_04utc_deg",
        "photo_br_mean",
        "photo_br_absmean",
        "photo_n_lat",
        "photo_n_lon",
        "ss_br_mean",
        "ss_br_absmean",
        "ss_n_lat",
        "ss_n_lon",
    ]:
        out[col] = out["TIME"].map(dict(zip(pfss["d"], pfss[col])))

    wanted = [
        "TIME",
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
    return out[wanted]


def relation_checks(omni: pd.DataFrame) -> list[dict]:
    checks: list[dict] = []
    omni = omni.copy()
    for col in omni.columns:
        if col != "TIME":
            omni[col] = pd.to_numeric(omni[col], errors="coerce")
    b = omni["Scalar B, nT"]
    bx, byg, bzg = omni["BX, nT (GSE, GSM)"], omni["BY, nT (GSE)"], omni["BZ, nT (GSE)"]
    bx_abs = bx.abs()
    checks.append(
        {
            "check": "|Bx| <= Scalar B",
            "fail_count": int((bx_abs > b + 1e-6).sum()),
            "bad_days": sorted(omni.loc[bx_abs > b + 1e-6, "TIME"].tolist())[:8],
        }
    )
    # Approximate embedded component check: full magnitude > |component| is expected.
    checks.append(
        {
            "check": "|By_GSE| <= Scalar B",
            "fail_count": int((byg.abs() > b + 1e-6).sum()),
            "bad_days": sorted(omni.loc[byg.abs() > b + 1e-6, "TIME"].tolist())[:8],
        }
    )
    checks.append(
        {
            "check": "|Bz_GSE| <= Scalar B",
            "fail_count": int((bzg.abs() > b + 1e-6).sum()),
            "bad_days": sorted(omni.loc[bzg.abs() > b + 1e-6, "TIME"].tolist())[:8],
        }
    )
    checks.append(
        {
            "check": "speed in [150, 1500] km/s",
            "fail_count": int(((omni["SW Plasma Speed, km/s"] < 150) | (omni["SW Plasma Speed, km/s"] > 1500)).sum()),
        }
    )
    checks.append(
        {
            "check": "density in [0.01, 100] cm^-3",
            "fail_count": int(
                ((omni["SW Proton Density, N/cm^3"] < 0.01) | (omni["SW Proton Density, N/cm^3"] > 100)).sum()
            ),
        }
    )
    checks.append(
        {
            "check": "temperature in [1e4, 5e6] K",
            "fail_count": int(
                ((omni["SW Plasma Temperature, K"] < 1e4) | (omni["SW Plasma Temperature, K"] > 5e6)).sum()
            ),
        }
    )
    return checks


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    live_omni = pd.read_csv(LIVE_OMNI)
    live_shadow = pd.read_csv(LIVE_SHADOW)
    omni_ref, shadow_ref = load_calendar()
    pfss = pd.read_csv(PFSS)

    omni_v3 = build_omni_v3(live_omni, omni_ref)
    shadow_v3 = build_shadow_v3(live_shadow, shadow_ref, pfss)

    omni_v3.to_csv(OMNI_OUT, index=False)
    shadow_v3.to_csv(SHADOW_OUT, index=False)

    checks = relation_checks(omni_v3)
    (RELATION_CHECK_OUT).write_text(
        json.dumps(checks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    provenance = {
        "created": "repair_2021_v3.py",
        "v2_issues": {
            "omni": "all columns shifted one field left; Scalar B zero-filled; SW Plasma Speed contained density; Kp/Dst/ap shifted",
            "shadow": "six LHAASO columns shifted; p2_y center values placed one column right with abs(); p1_x/p3/_llf mismatched; photo/ss copied verbatim from wrong source positions",
        },
        "v3_sources": {
            "omni": str(CALENDAR),
            "shadow": str(CALENDAR),
            "pfss": str(PFSS),
        },
        "outputs": {
            "omni": str(OMNI_OUT),
            "shadow": str(SHADOW_OUT),
        },
        "row_counts": {
            "omni_v3": len(omni_v3),
            "shadow_v3": len(shadow_v3),
        },
        "relation_checks": checks,
    }
    (PROVENANCE_OUT).write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OMNI v3: {OMNI_OUT} rows={len(omni_v3)} cols={len(omni_v3.columns)}")
    print(f"SHADOW v3: {SHADOW_OUT} rows={len(shadow_v3)} cols={len(shadow_v3.columns)}")
    print("relation checks", json.dumps(checks, ensure_ascii=False, indent=2))
    print("\nOMNI v3 2021-03-21:")
    print(omni_v3[omni_v3["TIME"] == "20210321"].to_string(index=False))
    print("\nSHADOW v3 2021-03-21:")
    print(shadow_v3[shadow_v3["TIME"] == "20210321"].to_string(index=False))


if __name__ == "__main__":
    main()
