import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pattern", help="case-sensitive substring to match")
    parser.add_argument("path", help="file to scan")
    parser.add_argument("--encoding", default="utf-8")
    args = parser.parse_args()

    text = Path(args.path).read_text(encoding=args.encoding)
    for index, line in enumerate(text.splitlines(), start=1):
        if args.pattern in line:
            print(f"{index}: {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
