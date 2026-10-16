import requests


BASE = "http://127.0.0.1:8765"

REJECTION = (
    "拒审：候选把 OMNI 总磁场强度 Scalar B, nT 作为日影特征设计焦点，"
    "与本轮科学问题不符。日影南北偏移特征必须使用 p2_y中心值；"
    "本轮不执行以 Scalar B 为焦点的实验。"
)


def main() -> None:
    for candidate_id in ("E_R01_01", "E_R01_02", "E_R01_04"):
        response = requests.post(
            f"{BASE}/api/workflow/approval",
            json={"action": "reject", "candidateId": candidate_id, "humanNotes": REJECTION},
            timeout=30,
        )
        print(candidate_id, response.status_code, response.text[:300])


if __name__ == "__main__":
    main()
