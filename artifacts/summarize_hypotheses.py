import json
import sys
from pathlib import Path


def main() -> int:
    path = Path(sys.argv[1])
    payload = json.loads(path.read_text(encoding="utf-8"))
    nodes = payload.get("nodes", [])
    print(f"tree_id: {payload.get('tree_id')}")
    print(f"round: {payload.get('current_round')}  active={payload.get('active_hypotheses')}  pending={payload.get('pending_hypotheses')}")
    for node in nodes:
        statement = node.get("display_statement") or node.get("statement") or ""
        print(
            f"- {node.get('display_hypothesis_id')} [{node.get('level')}] "
            f"status={node.get('status')} support={node.get('support_score')} "
            f"children={node.get('children_ids')}"
        )
        print(f"    {statement[:120]}")
        predictions = node.get("predictions") or []
        if predictions:
            shown = "; ".join(
                f"{item.get('experiment_id')}: {item.get('metric')} "
                f"{item.get('expected_direction')} {item.get('expected_range')}"
                for item in predictions[:3]
            )
            print(f"    predictions: {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
