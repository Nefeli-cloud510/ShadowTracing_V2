from __future__ import annotations

import json
import sys
from pathlib import Path


SESSION = Path(
    r"D:\ShadowTracing\runtime\live_session\sessions\session_20261016_235451_805278"
)


def inspect_raw(path: Path) -> None:
    print(f"==== {path.name} ====")
    raw = path.read_bytes()
    print("bytes", len(raw))
    text = raw.decode("utf-8", errors="replace")
    bad = [
        (i, ch)
        for i, ch in enumerate(text)
        if ord(ch) < 32 and ch not in "\r\n\t"
    ]
    if not bad:
        print("no control chars outside CR/LF/TAB")
    for idx, ch in bad[:10]:
        print("bad", idx, repr(ch), repr(text[max(0, idx - 60) : idx + 60]))
    try:
        json.loads(text)
        print("json OK")
    except json.JSONDecodeError as exc:
        print("json broken:", exc)
        line = text.splitlines()[exc.lineno - 1]
        print("line", exc.lineno, "col", exc.colno)
        print(repr(line[max(0, exc.colno - 80) : exc.colno + 80]))


def main() -> None:
    for name in sys.argv[1:]:
        inspect_raw(SESSION / name)


if __name__ == "__main__":
    main()
