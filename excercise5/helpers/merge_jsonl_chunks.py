#!/usr/bin/env python3
"""
merge_jsonl_chunks.py

Merge consecutive chunks from the same document (same source_path by default)
until reaching a target max word count, then start a new merged chunk.

- No overlap.
- Preserves key metadata and recomputes num_words.
- Assumes input JSONL items represent chunks with fields like:
  doc_id, source_path, chunk_index, start_char, end_char, text, num_words, doc_date_iso, doc_timestamp, etc.

Usage:
  python merge_jsonl_chunks.py \
    --in  input.jsonl \
    --out merged.jsonl \
    --max-words 350 \
    --group-by source_path
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Optional

WORD_RE = re.compile(r"\S+")

def count_words(text: str) -> int:
    return len(WORD_RE.findall(text or ""))

def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"Bad JSON on line {line_no}: {e}") from e

def write_jsonl(path: Path, items: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def sort_key(obj: Dict[str, Any]) -> Tuple:
    # Prefer numeric ordering by chunk_index; fall back to start_char.
    ci = obj.get("chunk_index")
    sc = obj.get("start_char")
    try:
        ci_val = int(ci)
    except Exception:
        ci_val = 10**18
    try:
        sc_val = int(sc)
    except Exception:
        sc_val = 10**18
    return (ci_val, sc_val)

def make_chunk_uid(
    group_value: str,
    doc_id: str,
    chunking_method: str,
    merged_idx: int
) -> str:
    # Keep it readable and stable-ish.
    # Example: "MERGED:british_parliament_debates:debates2023-06-28:semantic:0"
    safe_group = group_value.replace("\\", "/")
    return f"MERGED:{safe_group}:{doc_id}:{chunking_method}:{merged_idx}"

def merge_group(
    chunks: List[Dict[str, Any]],
    max_words: int,
    group_by_field: str,
) -> List[Dict[str, Any]]:
    if not chunks:
        return []

    chunks_sorted = sorted(chunks, key=sort_key)
    out: List[Dict[str, Any]] = []

    current_text_parts: List[str] = []
    current_word_count = 0
    current_first: Optional[Dict[str, Any]] = None
    current_last: Optional[Dict[str, Any]] = None
    merged_idx = 0

    def flush():
        nonlocal current_text_parts, current_word_count, current_first, current_last, merged_idx
        if not current_first or not current_last:
            return

        merged_text = "\n".join(part.rstrip() for part in current_text_parts if part is not None).strip()

        # Build merged object
        group_value = str(current_first.get(group_by_field, ""))
        doc_id = str(current_first.get("doc_id", ""))
        chunking_method = str(current_first.get("chunking_method", ""))
        merged_obj = dict(current_first)

        merged_obj["chunk_uid"] = make_chunk_uid(group_value, doc_id, chunking_method, merged_idx)
        merged_obj["chunk_index"] = merged_idx
        merged_obj["start_char"] = current_first.get("start_char")
        merged_obj["end_char"] = current_last.get("end_char")
        merged_obj["text"] = merged_text
        merged_obj["num_words"] = count_words(merged_text)

        out.append(merged_obj)

        merged_idx += 1
        current_text_parts = []
        current_word_count = 0
        current_first = None
        current_last = None

    for ch in chunks_sorted:
        text = ch.get("text", "")
        w = ch.get("num_words")
        if not isinstance(w, int):
            w = count_words(text)

        if current_first is None:
            current_first = ch
            current_last = ch
            current_text_parts = [text]
            current_word_count = w
            continue

        if current_word_count > 0 and (current_word_count + w) > max_words:
            flush()
            current_first = ch
            current_last = ch
            current_text_parts = [text]
            current_word_count = w
        else:
            current_text_parts.append(text)
            current_word_count += w
            current_last = ch

    flush()
    return out

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input JSONL path")
    ap.add_argument("--out", dest="out", required=True, help="Output JSONL path")
    ap.add_argument("--max-words", type=int, required=True, help="Max words per merged chunk (approx)")
    ap.add_argument(
        "--group-by",
        default="source_path",
        choices=["source_path", "doc_id"],
        help="What defines 'same file'. Default: source_path",
    )
    args = ap.parse_args()

    inp = Path(args.inp)
    outp = Path(args.out)
    max_words = args.max_words
    group_by = args.group_by

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for obj in read_jsonl(inp):
        key = str(obj.get(group_by, ""))
        groups.setdefault(key, []).append(obj)

    merged_all: List[Dict[str, Any]] = []
    for _, group_chunks in groups.items():
        merged_all.extend(merge_group(group_chunks, max_words=max_words, group_by_field=group_by))

    merged_all.sort(key=lambda o: (str(o.get(group_by, "")), int(o.get("chunk_index", 0))))

    write_jsonl(outp, merged_all)
    print(f"Wrote {len(merged_all)} merged chunks to: {outp}")

if __name__ == "__main__":
    main()
