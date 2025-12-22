"""
s_04_analyze_results.py

Comprehensive analysis of RAG experiment results.

This script:
1. Loads all experiment results from JSON files
2. Computes evaluation metrics (precision, recall, accuracy)
3. Compares pipelines, K values, and query types
4. Generates detailed reports with specific examples
5. Saves analysis results for visualization and README

Output:
- outputs/analysis/metrics.json - All computed metrics
- outputs/analysis/examples.json - Specific retrieval examples
- outputs/analysis/comparison_tables.json - Comparison tables
- outputs/analysis/summary.txt - Human-readable summary
"""

from __future__ import annotations

import os
import json
from typing import Dict, List, Any, Tuple
from collections import defaultdict
from dataclasses import dataclass, asdict
import numpy as np

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "outputs", "rag_runs")
ANALYSIS_DIR = os.path.join(BASE_DIR, "outputs", "analysis")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ============================================================
# Data Structures
# ============================================================

@dataclass
class RetrievalMetrics:
    """Metrics for a single query result."""
    query_id: str
    query_type: str  # "factual" or "conceptual"
    query_text: str
    
    pipeline: str  # e.g., "fixed_bm25"
    chunking: str  # "fixed" or "semantic"
    representation: str  # "bm25" or "dense"
    k: int
    
    # File-level metrics
    expected_files: List[str]  # From query metadata
    retrieved_files: List[str]  # Actual retrieved
    correct_files: int  # How many expected files were retrieved
    file_precision: float  # correct / retrieved
    file_recall: float  # correct / expected
    
    # Chunk-level metrics (manual annotation needed)
    relevant_chunks: int  # How many chunks were actually relevant (to be filled manually)
    total_chunks: int  # k
    
    # Answer quality
    answer_length: int  # Character count
    has_citation: bool  # Does answer cite sources?
    
    # Examples for case studies
    top_3_sources: List[Dict[str, Any]]  # Top 3 retrieved chunks with metadata


@dataclass
class ComparisonResult:
    """Comparison between two conditions."""
    condition_a: str
    condition_b: str
    metric: str
    value_a: float
    value_b: float
    difference: float
    winner: str


# ============================================================
# Load Results
# ============================================================

def load_all_results() -> List[Dict[str, Any]]:
    """Load all JSON result files from outputs/rag_runs/."""
    results = []
    
    if not os.path.exists(RESULTS_DIR):
        raise FileNotFoundError(f"Results directory not found: {RESULTS_DIR}")
    
    for filename in os.listdir(RESULTS_DIR):
        if filename.endswith(".json"):
            filepath = os.path.join(RESULTS_DIR, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    results.extend(data)
                else:
                    results.append(data)
    
    print(f"Loaded {len(results)} experiment results from {RESULTS_DIR}")
    return results


# ============================================================
# Metric Computation
# ============================================================

def extract_filename(source_path: str) -> str:
    """Extract just the filename from a full path."""
    return os.path.basename(source_path) if source_path else ""


def compute_file_metrics(expected: List[str], retrieved_refs: List[Dict]) -> Tuple[int, float, float]:
    """
    Compute file-level precision and recall.
    
    Args:
        expected: List of expected filenames (from query metadata)
        retrieved_refs: List of reference dicts from retrieval
    
    Returns:
        (correct_count, precision, recall)
    """
    if not expected:
        # No ground truth available
        return 0, 0.0, 0.0
    
    expected_set = {extract_filename(f) for f in expected}
    retrieved_set = {extract_filename(ref.get("source_path", "")) for ref in retrieved_refs}
    
    correct = len(expected_set & retrieved_set)
    
    precision = correct / len(retrieved_set) if retrieved_set else 0.0
    recall = correct / len(expected_set) if expected_set else 0.0
    
    return correct, precision, recall


def analyze_result(result: Dict[str, Any], idx: int) -> RetrievalMetrics:
    """Convert a single result dict into structured metrics."""
    
    query_type = result["query_type"]
    query_text = result["query"]
    expected_files = result.get("expected_source", [])
    
    pipeline = result["pipeline"]
    chunking = pipeline["chunking"]
    representation = pipeline["representation"]
    k = result["k"]
    
    refs = result.get("references", [])
    answer = result.get("answer", "")
    
    # File-level metrics
    correct_files, file_prec, file_rec = compute_file_metrics(expected_files, refs)
    retrieved_files = [extract_filename(r.get("source_path", "")) for r in refs]
    
    # Answer analysis
    has_citation = "[" in answer and "]" in answer
    
    # Top 3 sources for examples
    top_3 = []
    for ref in refs[:3]:
        top_3.append({
            "corpus": ref.get("corpus", "unknown"),
            "file": extract_filename(ref.get("source_path", "")),
            "chunk_index": ref.get("chunk_index", -1),
            "offsets": f"{ref.get('start_char', 0)}-{ref.get('end_char', 0)}"
        })
    
    return RetrievalMetrics(
        query_id=f"q{idx}",
        query_type=query_type,
        query_text=query_text[:100],  # Truncate for readability
        
        pipeline=f"{chunking}_{representation}",
        chunking=chunking,
        representation=representation,
        k=k,
        
        expected_files=expected_files,
        retrieved_files=retrieved_files,
        correct_files=correct_files,
        file_precision=file_prec,
        file_recall=file_rec,
        
        relevant_chunks=0,  # To be filled manually if needed
        total_chunks=k,
        
        answer_length=len(answer),
        has_citation=has_citation,
        
        top_3_sources=top_3
    )


# ============================================================
# Aggregation & Comparison
# ============================================================

def aggregate_metrics(metrics: List[RetrievalMetrics]) -> Dict[str, Any]:
    """Compute aggregate statistics across all results."""
    
    if not metrics:
        return {}
    
    # Group by different dimensions
    by_pipeline = defaultdict(list)
    by_chunking = defaultdict(list)
    by_representation = defaultdict(list)
    by_k = defaultdict(list)
    by_query_type = defaultdict(list)
    
    for m in metrics:
        by_pipeline[m.pipeline].append(m)
        by_chunking[m.chunking].append(m)
        by_representation[m.representation].append(m)
        by_k[m.k].append(m)
        by_query_type[m.query_type].append(m)
    
    def avg(values):
        return np.mean(values) if values else 0.0
    
    def compute_stats(group: List[RetrievalMetrics]) -> Dict:
        return {
            "count": len(group),
            "avg_file_precision": avg([m.file_precision for m in group]),
            "avg_file_recall": avg([m.file_recall for m in group]),
            "avg_correct_files": avg([m.correct_files for m in group]),
            "citation_rate": sum([m.has_citation for m in group]) / len(group) if group else 0,
            "avg_answer_length": avg([m.answer_length for m in group]),
        }
    
    return {
        "overall": compute_stats(metrics),
        "by_pipeline": {k: compute_stats(v) for k, v in by_pipeline.items()},
        "by_chunking": {k: compute_stats(v) for k, v in by_chunking.items()},
        "by_representation": {k: compute_stats(v) for k, v in by_representation.items()},
        "by_k": {k: compute_stats(v) for k, v in by_k.items()},
        "by_query_type": {k: compute_stats(v) for k, v in by_query_type.items()},
    }


def find_best_worst_cases(metrics: List[RetrievalMetrics]) -> Dict[str, Any]:
    """Identify best and worst performing cases for case studies."""
    
    # Best: highest file recall
    best_recall = sorted(metrics, key=lambda m: m.file_recall, reverse=True)[:5]
    
    # Worst: lowest file recall (where expected files exist)
    worst_recall = sorted(
        [m for m in metrics if m.expected_files],
        key=lambda m: m.file_recall
    )[:5]
    
    # Perfect retrievals
    perfect = [m for m in metrics if m.file_recall == 1.0 and m.expected_files]
    
    # Complete failures
    failures = [m for m in metrics if m.file_recall == 0.0 and m.expected_files]
    
    return {
        "best_recall": [asdict(m) for m in best_recall],
        "worst_recall": [asdict(m) for m in worst_recall],
        "perfect_retrievals": len(perfect),
        "complete_failures": len(failures),
        "failure_examples": [asdict(m) for m in failures[:3]],
    }


def compare_conditions(metrics: List[RetrievalMetrics]) -> List[ComparisonResult]:
    """Generate pairwise comparisons between different conditions."""
    
    comparisons = []
    
    # 1. Fixed vs Semantic chunking
    fixed = [m for m in metrics if m.chunking == "fixed"]
    semantic = [m for m in metrics if m.chunking == "semantic"]
    
    if fixed and semantic:
        fixed_prec = np.mean([m.file_precision for m in fixed])
        sem_prec = np.mean([m.file_precision for m in semantic])
        
        comparisons.append(ComparisonResult(
            condition_a="fixed_chunking",
            condition_b="semantic_chunking",
            metric="file_precision",
            value_a=fixed_prec,
            value_b=sem_prec,
            difference=sem_prec - fixed_prec,
            winner="semantic" if sem_prec > fixed_prec else "fixed"
        ))
    
    # 2. BM25 vs Dense
    bm25 = [m for m in metrics if m.representation == "bm25"]
    dense = [m for m in metrics if m.representation == "dense"]
    
    if bm25 and dense:
        bm25_recall = np.mean([m.file_recall for m in bm25])
        dense_recall = np.mean([m.file_recall for m in dense])
        
        comparisons.append(ComparisonResult(
            condition_a="bm25",
            condition_b="dense_embeddings",
            metric="file_recall",
            value_a=bm25_recall,
            value_b=dense_recall,
            difference=dense_recall - bm25_recall,
            winner="dense" if dense_recall > bm25_recall else "bm25"
        ))
    
    # 3. Factual vs Conceptual (by representation)
    for repr_type in ["bm25", "dense"]:
        factual = [m for m in metrics if m.query_type == "factual" and m.representation == repr_type]
        conceptual = [m for m in metrics if m.query_type == "conceptual" and m.representation == repr_type]
        
        if factual and conceptual:
            fact_prec = np.mean([m.file_precision for m in factual])
            conc_prec = np.mean([m.file_precision for m in conceptual])
            
            comparisons.append(ComparisonResult(
                condition_a=f"factual_{repr_type}",
                condition_b=f"conceptual_{repr_type}",
                metric="file_precision",
                value_a=fact_prec,
                value_b=conc_prec,
                difference=conc_prec - fact_prec,
                winner=f"factual_{repr_type}" if fact_prec > conc_prec else f"conceptual_{repr_type}"
            ))
    
    return comparisons


# ============================================================
# Report Generation
# ============================================================

def generate_summary_report(
    metrics: List[RetrievalMetrics],
    aggregates: Dict[str, Any],
    comparisons: List[ComparisonResult],
    examples: Dict[str, Any]
) -> str:
    """Generate a human-readable summary report."""
    
    lines = []
    lines.append("=" * 80)
    lines.append("RAG EXPERIMENT ANALYSIS SUMMARY")
    lines.append("=" * 80)
    lines.append("")
    
    # Overall stats
    lines.append("OVERALL STATISTICS")
    lines.append("-" * 80)
    overall = aggregates["overall"]
    lines.append(f"Total experiments run: {overall['count']}")
    lines.append(f"Average file precision: {overall['avg_file_precision']:.3f}")
    lines.append(f"Average file recall: {overall['avg_file_recall']:.3f}")
    lines.append(f"Citation rate: {overall['citation_rate']:.1%}")
    lines.append("")
    
    # By pipeline
    lines.append("PERFORMANCE BY PIPELINE")
    lines.append("-" * 80)
    for pipeline, stats in aggregates["by_pipeline"].items():
        lines.append(f"{pipeline:20s} | Prec: {stats['avg_file_precision']:.3f} | "
                    f"Rec: {stats['avg_file_recall']:.3f} | "
                    f"Correct files: {stats['avg_correct_files']:.2f}")
    lines.append("")
    
    # By K value
    lines.append("PERFORMANCE BY K VALUE")
    lines.append("-" * 80)
    for k, stats in sorted(aggregates["by_k"].items()):
        lines.append(f"k={k:2d} | Prec: {stats['avg_file_precision']:.3f} | "
                    f"Rec: {stats['avg_file_recall']:.3f}")
    lines.append("")
    
    # Comparisons
    lines.append("KEY COMPARISONS")
    lines.append("-" * 80)
    for comp in comparisons:
        lines.append(f"{comp.condition_a} vs {comp.condition_b} ({comp.metric}):")
        lines.append(f"  {comp.condition_a}: {comp.value_a:.3f}")
        lines.append(f"  {comp.condition_b}: {comp.value_b:.3f}")
        lines.append(f"  Difference: {comp.difference:+.3f} | Winner: {comp.winner}")
        lines.append("")
    
    # Best/Worst cases
    lines.append("NOTABLE CASES")
    lines.append("-" * 80)
    lines.append(f"Perfect retrievals: {examples['perfect_retrievals']}")
    lines.append(f"Complete failures: {examples['complete_failures']}")
    lines.append("")
    
    if examples.get("failure_examples"):
        lines.append("Example failures:")
        for ex in examples["failure_examples"][:2]:
            lines.append(f"  - {ex['query_text'][:60]}...")
            lines.append(f"    Expected: {ex['expected_files']}")
            lines.append(f"    Retrieved: {ex['retrieved_files'][:3]}")
            lines.append("")
    
    return "\n".join(lines)


# ============================================================
# Main
# ============================================================

def main():
    ensure_dir(ANALYSIS_DIR)
    
    print("Loading results...")
    results = load_all_results()
    
    print("Computing metrics...")
    metrics = [analyze_result(r, i) for i, r in enumerate(results)]
    
    print("Aggregating statistics...")
    aggregates = aggregate_metrics(metrics)
    
    print("Finding best/worst cases...")
    examples = find_best_worst_cases(metrics)
    
    print("Generating comparisons...")
    comparisons = compare_conditions(metrics)
    
    print("Generating summary report...")
    summary = generate_summary_report(metrics, aggregates, comparisons, examples)
    
    # Save all outputs
    print("\nSaving analysis results...")
    
    with open(os.path.join(ANALYSIS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump([asdict(m) for m in metrics], f, indent=2, ensure_ascii=False)
    
    with open(os.path.join(ANALYSIS_DIR, "aggregates.json"), "w", encoding="utf-8") as f:
        json.dump(aggregates, f, indent=2, ensure_ascii=False)
    
    with open(os.path.join(ANALYSIS_DIR, "comparisons.json"), "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in comparisons], f, indent=2, ensure_ascii=False)
    
    with open(os.path.join(ANALYSIS_DIR, "examples.json"), "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    
    with open(os.path.join(ANALYSIS_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary)
    
    print(summary)
    
    print(f"\n✅ Analysis complete! Results saved to: {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()