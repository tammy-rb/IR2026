"""
s_01_time_histogram.py

Exercise 4 — Stage 2 (Temporal Indexing): Time distribution report.

Purpose:
- Produce an artifact proving temporal metadata (doc_timestamp / doc_date_iso)
  was extracted and attached to chunks correctly.
- Generates a time distribution of chunks by year.

Inputs:
- outputs/chunks/chunks_fixed.jsonl
- outputs/chunks/chunks_semantic.jsonl

Outputs:
- outputs/reports/time_histograms/fixed_chunks_by_year.png
- outputs/reports/time_histograms/fixed_chunks_by_year.json
- outputs/reports/time_histograms/semantic_chunks_by_year.png
- outputs/reports/time_histograms/semantic_chunks_by_year.json

Run:
  python s_01_time_histogram.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, List

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.chunk import Chunk
from paths import (
    CHUNKS_DIR,
    BBC_NEWS_CHUNKS_JSONL,
    NBC_NEWS_CHUNKS_JSONL,
    REPORTS_DIR,
    ensure_dirs,
)


TIME_HIST_DIR = REPORTS_DIR / "time_histograms"


@dataclass(frozen=True)
class TimeHistogramResult:
    png_path: Path
    json_path: Path
    total_with_year: int
    total_chunks_seen: int


class TimeHistogramBuilder:
    """
    Builds and saves a time-distribution histogram (by year) from chunk JSONL files.

    Expected JSONL fields per chunk:
    - doc_date_iso: "YYYY-MM-DD" (optional)
    - doc_timestamp: Unix seconds UTC (optional)
    """

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir

    # ---------- Reading ----------

    @staticmethod
    def read_chunks_stream(path: Path) -> Iterable[Chunk]:
        """Stream-read validated `Chunk` objects from a JSONL file."""
        if not path.is_file():
            raise FileNotFoundError(str(path))

        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield Chunk.from_dict(json.loads(line))
                except Exception as e:
                    raise ValueError(f"Invalid chunk in {path.name} at line {line_no}: {e}") from e

    # ---------- Year extraction ----------

    @staticmethod
    def _year_from_chunk(c: Chunk) -> Optional[int]:
        """Extract year from chunk metadata.

        Prefer doc_date_iso, fallback to doc_timestamp.
        """
        if c.doc_date_iso:
            try:
                return int(str(c.doc_date_iso)[:4])
            except Exception:
                pass

        if c.doc_timestamp is not None:
            try:
                return datetime.fromtimestamp(int(c.doc_timestamp), tz=timezone.utc).year
            except Exception:
                pass

        return None

    def counts_by_year_from_jsonl(self, jsonl_path: Path) -> Tuple[Counter, int]:
        """
        Returns:
            (Counter(year->count), total_chunks_seen)
        """
        counts: Counter = Counter()
        total = 0

        for c in self.read_chunks_stream(jsonl_path):
            total += 1
            y = self._year_from_chunk(c)
            if y is not None:
                counts[y] += 1

        return counts, total

    # ---------- Save outputs ----------

    def save(self, counts: Counter, total_seen: int, name: str, title: str) -> TimeHistogramResult:
        """
        Save JSON counts + PNG bar chart.

        Args:
            counts: Counter(year->count)
            total_seen: number of chunks read from file(s)
            name: base output name (e.g. "fixed", "semantic", "all")
            title: plot title
        """
        self.out_dir.mkdir(parents=True, exist_ok=True)

        years_sorted = sorted(counts.keys())
        values = [counts[y] for y in years_sorted]

        png_path = self.out_dir / f"{name}_chunks_by_year.png"
        json_path = self.out_dir / f"{name}_chunks_by_year.json"

        # JSON (also includes coverage info)
        payload = {
            "name": name,
            "total_chunks_seen": total_seen,
            "total_with_year": int(sum(values)),
            "counts_by_year": {str(y): int(counts[y]) for y in years_sorted},
        }
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        # Plot (no explicit colors)
        plt.figure()
        plt.bar(years_sorted, values)
        plt.title(title)
        plt.xlabel("Year")
        plt.ylabel("Number of chunks")
        if len(years_sorted) > 0:
            plt.xticks(years_sorted, rotation=45)
        plt.tight_layout()
        plt.savefig(png_path, dpi=200)
        plt.close()

        return TimeHistogramResult(
            png_path=png_path,
            json_path=json_path,
            total_with_year=int(sum(values)),
            total_chunks_seen=total_seen,
        )

    # ---------- High-level API ----------

    def build_from_jsonl(self, jsonl_path: Path, name: str, title: str) -> TimeHistogramResult:
        counts, total = self.counts_by_year_from_jsonl(jsonl_path)
        return self.save(counts, total_seen=total, name=name, title=title)

    def build_combined_from_jsonls(self, jsonl_paths: list[Path], name: str, title: str) -> TimeHistogramResult:
        combined: Counter = Counter()
        total_seen = 0
        for p in jsonl_paths:
            counts, total = self.counts_by_year_from_jsonl(p)
            combined.update(counts)
            total_seen += total
        return self.save(combined, total_seen=total_seen, name=name, title=title)


def _default_targets() -> Dict[str, Tuple[Path, str]]:
    return {
        "fixed": (CHUNKS_DIR / "chunks_fixed.jsonl", "Fixed chunks: time distribution by year"),
        "semantic": (CHUNKS_DIR / "chunks_semantic.jsonl", "Semantic chunks: time distribution by year"),
        "bbc_news": (BBC_NEWS_CHUNKS_JSONL, "BBC news chunks: time distribution by year"),
        "nbc_news": (NBC_NEWS_CHUNKS_JSONL, "NBC news chunks: time distribution by year"),
    }


def main(args: Optional[List[str]] = None) -> None:
    ensure_dirs()

    builder = TimeHistogramBuilder(out_dir=TIME_HIST_DIR)

    targets = _default_targets()

    selected = args[1:] if args is not None and len(args) > 1 else sys.argv[1:]
    if selected:
        missing = [name for name in selected if name not in targets]
        if missing:
            raise ValueError(f"Unknown histogram targets: {', '.join(missing)}")
        worklist = {name: targets[name] for name in selected}
    else:
        worklist = targets

    for name, (jsonl_path, title) in worklist.items():
        if not jsonl_path.is_file():
            print(f"⚠️ skipping {name} histogram: missing {jsonl_path}")
            continue

        result = builder.build_from_jsonl(
            jsonl_path,
            name=name,
            title=title,
        )
        print(
            f"✅ {name} histogram: {result.png_path} | {result.json_path} "
            f"| covered={result.total_with_year}/{result.total_chunks_seen}"
        )


if __name__ == "__main__":
    main()
