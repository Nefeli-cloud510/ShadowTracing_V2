"""Fine-grained identity check for LHAASO/PFSS live columns."""

from pathlib import Path

import pandas as pd


LIVE = Path(r"D:\ShadowTracing\runtime\live_run_input\lhaaso_shadow_pfss_2021_full_calendar_v2.csv")
SHADOW_SRC = Path(r"D:\ShadowTracing\data\raw_out_of_ICME_sunshadow2.xlsx")
SHADOW_SRC_ALT = Path(r"D:\helioprior\data\raw_out_of_ICME_sunshadow2.xlsx")
CAL = Path(r"D:\sunshadow\calender_internal_2020_2024.xlsx")
PFSS = Path(r"D:\ShadowTracing\data\PFSS_data.xlsx")


def to_date(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(r"\.0$", "", regex=True)
    return s


def corr(a: pd.Series, b: pd.Series) -> float:
    m = a.notna() & b.notna()
    return float(a[m].corr(b[m])) if m.sum() > 2 else float("nan")


def main() -> None:
    live = pd.read_csv(LIVE)
    cal = pd.read_excel(CAL, sheet_name=0)
    pfss = pd.read_excel(PFSS, sheet_name=0)
    shadow = pd.read_excel(SHADOW_SRC if SHADOW_SRC.exists() else SHADOW_SRC_ALT, sheet_name=0)

    live["d"] = to_date(live["TIME"])
    cal["d"] = to_date(cal["data_utc"])
    pfss["d"] = to_date(pfss["TIME"])
    shadow["d"] = to_date(shadow["TIME"])

    print("cal rows", cal.shape, "pfss rows", pfss.shape, "shadow rows", shadow.shape)
    print("overlap live/sunshadow:", len(set(live.d) & set(shadow.d)))
    print("\n=== PFSS sample ===")
    print(pfss.head().to_string(index=False))
    print("\n=== shadow source sample (first 6 rows) ===")
    print(shadow.head(6).to_string(index=False))
    print("\n=== calendar columns ===")
    print([str(c) for c in cal.columns])

    for live_col in [
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
    ]:
        lv = pd.to_numeric(live[live_col], errors="coerce")
        print(f"\nLIVE {live_col}")
        cands = []
        for df, tag in [(shadow, "shadow"), (cal, "cal"), (pfss, "pfss")]:
            for c in df.columns:
                if str(c) in ("d", "TIME", "YEAR", "DOY", "DATE", "HR"):
                    continue
                rv = pd.to_numeric(df[c], errors="coerce")
                joined = live[["d", live_col]].merge(df[["d", c]], on="d", how="inner")
                if joined.empty:
                    continue
                r = corr(
                    pd.to_numeric(joined.iloc[:, 1], errors="coerce"),
                    pd.to_numeric(joined.iloc[:, 2], errors="coerce"),
                )
                if r == r:
                    cands.append((abs(r), r, tag, str(c)))
        for item in sorted(cands, reverse=True)[:5]:
            print(f"  {item[2]:<7} r={item[1]:+.4f}  {item[3]}")


if __name__ == "__main__":
    main()
