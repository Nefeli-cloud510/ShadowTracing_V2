from pathlib import Path

import requests


BASE = "http://127.0.0.1:8765"
ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "runtime" / "live_run_input"
OUTPUTS_DIR = ROOT / "outputs"

KNOWLEDGE_FILES = [
    ("lhaaso_imf_2021.txt", OUTPUTS_DIR / "_tmp_lhaaso_imf.txt"),
    ("nanyun_2021.txt", OUTPUTS_DIR / "_tmp_nanyun.txt"),
    ("lizhe_2021.txt", OUTPUTS_DIR / "_tmp_lizhe.txt"),
    ("thesis_2021.txt", OUTPUTS_DIR / "_tmp_thesis.txt"),
]


def main() -> None:
    data = {
        "question": "LHAASO观测到的宇宙线日影南北偏移是否包含能够提前改善未来太阳风速度预测的有效信息？",
        "model": "qwen3.8-flash",
        "rounds": "2",
        "x_variable": "p2_y中心值",
        "y_variable": "SW Plasma Speed, km/s",
        "question_type": "forecasting",
        "m_candidates": (
            '["SW Plasma Speed, km/s","ss_br_mean","BX, nT (GSE, GSM)","p2_y中心值误差"]'
        ),
    }
    with requests.Session() as session:
        files = []
        for name, path in KNOWLEDGE_FILES:
            if not path.exists():
                raise FileNotFoundError(path)
            files.append(("knowledge_files", (name, path.open("rb"))))
        for name in ("omni_2021_full_calendar_v3.csv", "lhaaso_shadow_pfss_2021_full_calendar_v3.csv"):
            path = INPUT_DIR / name
            if not path.exists():
                raise FileNotFoundError(path)
            files.append(("data_files", (name, path.open("rb"))))
        response = session.post(f"{BASE}/api/workflow/start", data=data, files=files)
    print(response.status_code)
    print(response.text[:4000])
    (ROOT / "artifacts" / "start_v3_rag_response.json").write_text(
        response.text, encoding="utf-8"
    )


if __name__ == "__main__":
    main()
