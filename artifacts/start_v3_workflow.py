import json
from pathlib import Path

import requests


BASE = "http://127.0.0.1:8765"
ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "runtime" / "live_run_input"


def main() -> None:
    data = {
        "question": "LHAASO观测到的宇宙线日影南北偏移是否包含能够提前改善未来太阳风速度预测的有效信息？",
        "model": "qwen3.8-flash",
        "rounds": "2",
        "x_variable": "p2_y中心值",
        "y_variable": "SW Plasma Speed, km/s",
        "question_type": "forecasting",
        "m_candidates": json.dumps(
            [
                "SW Plasma Speed, km/s",
                "ss_br_mean",
                "BX, nT (GSE, GSM)",
                "p2_y中心值误差",
            ],
            ensure_ascii=False,
        ),
    }
    data_files = [
        INPUT_DIR / "omni_2021_full_calendar_v3.csv",
        INPUT_DIR / "lhaaso_shadow_pfss_2021_full_calendar_v3.csv",
    ]
    with requests.Session() as session:
        with open(INPUT_DIR / "omni_2021_full_calendar_v3.csv", "rb") as omni_file, open(
            INPUT_DIR / "lhaaso_shadow_pfss_2021_full_calendar_v3.csv", "rb"
        ) as shadow_file:
            files = [
                ("data_files", ("omni_2021_full_calendar_v3.csv", omni_file)),
                ("data_files", ("lhaaso_shadow_pfss_2021_full_calendar_v3.csv", shadow_file)),
            ]
            response = session.post(f"{BASE}/api/workflow/start", data=data, files=files)
    print(response.status_code)
    print(response.text[:2000])


if __name__ == "__main__":
    main()
