"""Scan authoritative data sources for the 2021 daily OMNI / LHAASO comparison."""

from pathlib import Path

import pandas as pd


SOURCES = [
    r"D:\ShadowTracing\data\2021_public_data.xlsx",
    r"D:\ShadowTracing\data\out_of_ICME_raw_OMNI_Daily_DATA(1).xlsx",
    r"D:\ShadowTracing\data\PFSS_data.xlsx",
    r"D:\ShadowTracing\data\raw_out_of_ICME_sunshadow2.xlsx",
    r"D:\sunshadow\calender_internal_2020_2024.xlsx",
    r"D:\helioprior\data\out_of_ICME_raw_OMNI_Daily_DATA(1).xlsx",
    r"D:\helioprior\data\raw_out_of_ICME_sunshadow2.xlsx",
]


def main() -> None:
    for path in SOURCES:
        p = Path(path)
        print(f"\n### {path} exists={p.exists()}")
        if not p.exists():
            continue
        try:
            xl = pd.ExcelFile(p)
            for sheet in xl.sheet_names:
                df = pd.read_excel(xl, sheet_name=sheet, nrows=5)
                print(f"  sheet={sheet!r} shape_sample={df.shape}")
                cols = [str(c)[:50] for c in df.columns]
                print("  cols=", cols)
        except Exception as exc:  # keep scanning every source
            print("  ERROR", repr(exc))


if __name__ == "__main__":
    main()
