"""Re-extract PDF text for legacy live-session uploads.

Before this migration, uploaded PDFs were stored as placeholder text only. This
script reads ``state/upload_manifest.json`` for a session, extracts each PDF
into ``outputs/pdf_texts`` with the same helper used by new uploads, and writes
an updated manifest while preserving a timestamped backup.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.json_safe import json_dumps
from core.pdf_extraction import extract_pdf_text


def re_extract_session(root: Path, *, force: bool = False) -> dict[str, object]:
    root = Path(root)
    manifest_path = root / "state" / "upload_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"upload manifest not found: {manifest_path}")

    original = manifest_path.read_text(encoding="utf-8")
    backup = manifest_path.with_name(
        f"upload_manifest.pre_extract.{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    shutil.copy2(manifest_path, backup)

    records = json_loads(original)
    updated: list[dict[str, object]] = []
    pdf_record_count = 0
    for index, record in enumerate(records):
        if str(record.get("file_type", "")).lower() != ".pdf":
            updated.append(record)
            continue
        pdf_record_count += 1
        current_status = str(record.get("extraction_status", ""))
        if current_status == "extracted" and not force:
            updated.append(record)
            continue

        raw_path = root / str(record["storage_path"])
        text_path = root / str(record["extracted_text_path"])
        extraction = extract_pdf_text(raw_path)
        if extraction["error"] is None:
            text_path.parent.mkdir(parents=True, exist_ok=True)
            text_path.write_text(str(extraction["text"]), encoding="utf-8")
            record["extraction_status"] = "extracted"
            record["extraction_engine"] = extraction["engine"]
            record["page_count"] = extraction["page_count"]
            record["character_count"] = extraction["character_count"]
            record["extraction_note"] = None
        else:
            record["extraction_status"] = "placeholder_only"
            record["extraction_engine"] = None
            record["page_count"] = extraction["page_count"]
            record["character_count"] = extraction["character_count"]
            record["extraction_note"] = f"PDF文本提取失败：{extraction['error']}"
        updated.append(record)

    manifest_path.write_text(
        json_dumps({"records": updated}, indent=2),
        encoding="utf-8",
    )
    return {
        "root": str(root),
        "backup": str(backup),
        "records": updated,
        "pdf_records": pdf_record_count,
    }


def json_loads(raw: str) -> list[dict[str, object]]:
    import json

    parsed = json.loads(raw)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("records"), list):
        raise ValueError("invalid upload manifest structure")
    return parsed["records"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=str(Path("runtime/live_session/sessions")),
        help="Session root containing state/upload_manifest.json",
    )
    parser.add_argument("--force", action="store_true", help="Re-extract already extracted PDFs")
    args = parser.parse_args()
    result = re_extract_session(Path(args.root), force=args.force)
    for record in result["records"]:
        if str(record.get("file_type", "")).lower() == ".pdf":
            print(
                f"{record['file_name']}: "
                f"{record.get('extraction_status')} | "
                f"{record.get('page_count')} pages | "
                f"{record.get('character_count')} chars | "
                f"{record.get('extraction_engine')}"
            )
    print(f"backup: {result['backup']}")
