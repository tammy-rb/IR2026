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
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from paths import CHUNKS_DIR, TIME_HIST_DIR, ensure_dirs


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
    def read_jsonl_stream(path: Path) -> Iterable[Dict[str, Any]]:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    # ---------- Year extraction ----------

    @staticmethod
    def _year_from_chunk_dict(c: Dict[str, Any]) -> Optional[int]:
        """
        Extract year from chunk metadata.
        Prefer doc_date_iso, fallback to doc_timestamp.
        """
        iso = c.get("doc_date_iso")
        if iso:
            # Expect "YYYY-MM-DD"
            try:
                return int(str(iso)[:4])
            except Exception:
                pass

        ts = c.get("doc_timestamp")
        if ts is not None:
            try:
                return datetime.fromtimestamp(int(ts), tz=timezone.utc).year
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

        for c in self.read_jsonl_stream(jsonl_path):
            total += 1
            y = self._year_from_chunk_dict(c)
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


def main() -> None:
    ensure_dirs()

    fixed_jsonl = CHUNKS_DIR / "chunks_fixed.jsonl"
    semantic_jsonl = CHUNKS_DIR / "chunks_semantic.jsonl"

    if not fixed_jsonl.is_file():
        raise FileNotFoundError(f"Missing chunks file: {fixed_jsonl}")
    if not semantic_jsonl.is_file():
        raise FileNotFoundError(f"Missing chunks file: {semantic_jsonl}")

    builder = TimeHistogramBuilder(out_dir=TIME_HIST_DIR)

    r_fixed = builder.build_from_jsonl(
        fixed_jsonl,
        name="fixed",
        title="Fixed chunks: time distribution by year",
    )
    print(f"✅ fixed histogram: {r_fixed.png_path} | {r_fixed.json_path} "
          f"| covered={r_fixed.total_with_year}/{r_fixed.total_chunks_seen}")

    r_sem = builder.build_from_jsonl(
        semantic_jsonl,
        name="semantic",
        title="Semantic chunks: time distribution by year",
    )
    print(f"✅ semantic histogram: {r_sem.png_path} | {r_sem.json_path} "
          f"| covered={r_sem.total_with_year}/{r_sem.total_chunks_seen}")


if __name__ == "__main__":
    main()
