import json
import sys
from pathlib import Path

import requests


def main() -> int:
    url = sys.argv[1]
    body_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    body = json.loads(body_path.read_text(encoding="utf-8")) if body_path else {}
    response = requests.post(url, json=body, timeout=30)
    print(response.status_code)
    print(response.text[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
