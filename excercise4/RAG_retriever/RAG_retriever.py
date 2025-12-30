"""
s_03_RAG_retriever.py

Central retrieval engine for a Retrieval-Augmented Generation (RAG) system.

This module encapsulates all retrieval logic and resources, including:
- Sparse lexical retrieval using BM25
- Dense semantic retrieval using FAISS with OpenAI embeddings
- Support for multiple chunking strategies (fixed / semantic)

All indices are loaded once and cached in memory, allowing efficient
repeated top-K retrieval calls from an external LLM orchestration layer.

This file intentionally contains NO LLM logic.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from paths import BM25_DIR, CHUNKS_DIR, OPENAI_DIR
from .retrievers.bm25 import BM25Retriever
from .retrievers.dense_faiss import DenseFAISSRetriever
from .utils import build_context_block, detect_corpus_label

from .duckling_time_analysis import analyze_query_time
from .temporal_policy import RetrievalPlan, build_retrieval_plan

load_dotenv()

# ============================================================
# Configuration
# ============================================================

BM25_FIXED_DIR = BM25_DIR / "fixed"
BM25_SEM_DIR = BM25_DIR / "semantic"

FAISS_FIXED_DIR = OPENAI_DIR / "fixed_faiss"
FAISS_SEM_DIR = OPENAI_DIR / "semantic_faiss"

CHUNKS_FIXED_JSONL = CHUNKS_DIR / "chunks_fixed.jsonl"
CHUNKS_SEM_JSONL = CHUNKS_DIR / "chunks_semantic.jsonl"

OPENAI_EMBED_MODEL = "text-embedding-3-large"
STOP_WORDS = "english"


# ============================================================
# Pipeline model
# ============================================================

@dataclass(frozen=True)
class Pipeline:
    chunking: str         # "fixed" or "semantic"
    representation: str   # "bm25" or "dense"
    bm25: Optional[BM25Retriever]
    dense: Optional[DenseFAISSRetriever]


# ============================================================
# Helpers (ranking + time)
# ============================================================

def _minmax_normalize(vals: List[float], eps: float = 1e-12) -> List[float]:
    if not vals:
        return []
    vmin = min(vals)
    vmax = max(vals)
    if abs(vmax - vmin) < eps:
        return [0.0 for _ in vals]
    return [(v - vmin) / (vmax - vmin + eps) for v in vals]


def _ts_from_chunk(c) -> Optional[int]:
    ts = getattr(c, "doc_timestamp", None)
    if ts is None:
        return None
    try:
        return int(ts)
    except Exception:
        return None


def _hard_filter(
    items: List[Tuple[Any, float]],
    start_ts: Optional[int],
    end_ts: Optional[int],
) -> List[Tuple[Any, float]]:
    """
    Keep only chunks whose doc_timestamp is within the (inclusive) bounds.
    None means open bound.
    """
    out: List[Tuple[Any, float]] = []
    for chunk, score in items:
        ts = _ts_from_chunk(chunk)
        if ts is None:
            continue
        if start_ts is not None and ts < start_ts:
            continue
        if end_ts is not None and ts > end_ts:
            continue
        out.append((chunk, score))
    return out


def _soft_rerank(
    items: List[Tuple[Any, float]],
    *,
    ref_ts: int,
    alpha: float,
    lam: float,
) -> List[Tuple[Any, float, float, float]]:
    """
    Apply Soft Decay / Recency Weighting re-ranking:

        Score = (1-α) * Sim + α * 1/(1 + Δt_days * λ)

    Assumptions:
      - Incoming "score" is semantic similarity where higher is better.
        (DenseFAISSRetriever must convert distance -> similarity.)
      - Δt is measured in DAYS.
      - Future documents are not boosted (Δt clamped to 0).

    Returns:
      List[(chunk, final_score, sim_norm, time_score)] sorted by final_score desc.
    """
    if not items:
        return []

    sims = [float(s) for _c, s in items]
    sims_norm = _minmax_normalize(sims)

    ranked: List[Tuple[Any, float, float, float]] = []
    for (chunk, _sim), sim_n in zip(items, sims_norm):
        ts = _ts_from_chunk(chunk)
        if ts is None:
            # Missing timestamps: treat as very old (max penalty)
            dt_days = 10_000.0
        else:
            dt_sec = float(ref_ts - ts)
            # Do not boost future docs
            if dt_sec < 0:
                dt_sec = 0.0
            dt_days = dt_sec / 86400.0

        time_score = 1.0 / (1.0 + dt_days * float(lam))
        final = (1.0 - float(alpha)) * float(sim_n) + float(alpha) * float(time_score)
        ranked.append((chunk, float(final), float(sim_n), float(time_score)))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


# ============================================================
# Main Retriever
# ============================================================

class RAGRetriever:
    """
    Unified retrieval engine for a RAG system.

    Exposes:
      - get_topk(): time-blind baseline retrieval (Stage 1)
      - get_topk_timeaware(): time-aware retrieval (Stage 3: Hard Filtering / Soft Decay)
    """

    def __init__(self) -> None:
        self._pipelines: Dict[Tuple[str, str], Pipeline] = {}
        self._load_all()

    def _load_all(self) -> None:
        for chunking in ("fixed", "semantic"):
            for repr_ in ("bm25", "dense"):
                self._pipelines[(chunking, repr_)] = self._load_pipeline(chunking, repr_)

    def _load_pipeline(self, chunking: str, repr_: str) -> Pipeline:
        bm25 = dense = None

        if repr_ == "bm25":
            bm25 = BM25Retriever(
                index_dir=BM25_FIXED_DIR if chunking == "fixed" else BM25_SEM_DIR,
                chunks_jsonl=CHUNKS_FIXED_JSONL if chunking == "fixed" else CHUNKS_SEM_JSONL,
                stop_words=STOP_WORDS,
            )

        if repr_ == "dense":
            dense = DenseFAISSRetriever(
                index_dir=FAISS_FIXED_DIR if chunking == "fixed" else FAISS_SEM_DIR,
                embed_model=OPENAI_EMBED_MODEL,
            )

        return Pipeline(chunking=chunking, representation=repr_, bm25=bm25, dense=dense)

    def _get_pipe(self, chunking: str, representation: str) -> Pipeline:
        if (chunking, representation) not in self._pipelines:
            raise KeyError(f"Unsupported pipeline: chunking={chunking!r}, representation={representation!r}")
        return self._pipelines[(chunking, representation)]

    # -------------------------
    # Stage 1: baseline
    # -------------------------
    def get_topk(self, query: str, chunking: str, representation: str, k: int) -> Dict[str, Any]:
        """
        Time-blind baseline retrieval (Stage 1).
        """
        pipe = self._get_pipe(chunking, representation)

        if representation == "bm25":
            if pipe.bm25 is None:
                raise RuntimeError("BM25 pipeline not initialized.")
            retrieved = pipe.bm25.search(query, k)
        else:
            if pipe.dense is None:
                raise RuntimeError("Dense pipeline not initialized.")
            retrieved = pipe.dense.search(query, k)

        return {
            "retrieved": [
                {"chunk": asdict(c), "score": float(s), "text": c.text}
                for (c, s) in retrieved
            ],
            "context": build_context_block(retrieved),
            "refs": [
                {
                    "corpus": (c.corpus or detect_corpus_label(c.source_path)),
                    "file_name": os.path.basename(c.source_path or ""),
                    "source_path": c.source_path,
                    "start_char": c.start_char,
                    "end_char": c.end_char,
                    "chunk_index": c.chunk_index,
                    "doc_date_iso": c.doc_date_iso,
                    "doc_timestamp": c.doc_timestamp,
                }
                for (c, _s) in retrieved
            ],
        }

    # -------------------------
    # Stage 3: time-aware
    # -------------------------
    def get_topk_timeaware(
        self,
        query: str,
        chunking: str,
        representation: str,
        k: int,
        *,
        duckling_url: str = "http://localhost:8000/parse",
        locale: str = "en_US",
        plan_override: Optional[Dict[str, Any]] = None,
        max_hard_oversample: int = 4000,
    ) -> Dict[str, Any]:
        """
        Stage 3: Time-aware retrieval.

        Flow:
          1) Analyze query time via Duckling (deterministic extraction)
          2) Build a RetrievalPlan (hard vs soft + params)
          3) Retrieve candidates with oversampling (k + oversample)
          4) Apply:
               - Hard filtering (explicit ranges)
               - Soft decay reranking (current/recent/none)
        """
        pipe = self._get_pipe(chunking, representation)

        # 1) time analysis
        time_info = analyze_query_time(query, duckling_url=duckling_url, locale=locale)

        # 2) build plan (defaults)
        plan = build_retrieval_plan(time_info, k=k)

        # Optional override for experiments (alpha/lam/strategy/oversample)
        if plan_override:
            plan = RetrievalPlan(
                strategy=str(plan_override.get("strategy", plan.strategy)),
                start_ts=plan_override.get("start_ts", plan.start_ts),
                end_ts=plan_override.get("end_ts", plan.end_ts),
                ref_ts=int(plan_override.get("ref_ts", plan.ref_ts)),
                alpha=float(plan_override.get("alpha", plan.alpha)),
                lam=float(plan_override.get("lam", plan.lam)),
                oversample=int(plan_override.get("oversample", plan.oversample)),
            )

        # 3) retrieve candidates (k + oversample)
        def _retrieve_candidates(oversample: int) -> List[Tuple[Any, float]]:
            if representation == "bm25":
                if pipe.bm25 is None:
                    raise RuntimeError("BM25 pipeline not initialized.")
                return pipe.bm25.search_candidates(query, k, oversample=oversample)
            else:
                if pipe.dense is None:
                    raise RuntimeError("Dense pipeline not initialized.")
                return pipe.dense.search_candidates(query, k, oversample=oversample)

        cands = _retrieve_candidates(plan.oversample)

        # 4) apply plan
        final_items: List[Tuple[Any, float]] = []
        debug_rows: List[Dict[str, Any]] = []

        if plan.strategy == "hard":
            # Hard filter might leave <k results. We'll try to expand oversample up to a cap.
            overs = int(plan.oversample)
            filtered = _hard_filter(cands, plan.start_ts, plan.end_ts)

            while len(filtered) < k and overs < max_hard_oversample:
                overs = min(max_hard_oversample, max(overs * 2, overs + 200))
                cands = _retrieve_candidates(overs)
                filtered = _hard_filter(cands, plan.start_ts, plan.end_ts)

            filtered.sort(key=lambda x: x[1], reverse=True)
            final_items = filtered[:k]

        elif plan.strategy == "soft":
            ranked = _soft_rerank(cands, ref_ts=plan.ref_ts, alpha=plan.alpha, lam=plan.lam)
            final_items = [(c, s_final) for (c, s_final, _sn, _ts) in ranked[:k]]

            # Keep debug info for analysis (top ~50 or k*5)
            for (c, s_final, sim_n, time_s) in ranked[: min(len(ranked), max(50, k * 5))]:
                debug_rows.append(
                    {
                        "doc_date_iso": getattr(c, "doc_date_iso", None),
                        "doc_timestamp": getattr(c, "doc_timestamp", None),
                        "final_score": float(s_final),
                        "sim_norm": float(sim_n),
                        "time_score": float(time_s),
                        "source_path": getattr(c, "source_path", None),
                        "chunk_index": getattr(c, "chunk_index", None),
                    }
                )

        else:
            # "none" -> behave like baseline
            final_items = cands[:k]

        return {
            "time_info": time_info,
            "plan": {
                "strategy": plan.strategy,
                "start_ts": plan.start_ts,
                "end_ts": plan.end_ts,
                "ref_ts": plan.ref_ts,
                "alpha": plan.alpha,
                "lam": plan.lam,
                "oversample": plan.oversample,
            },
            "retrieved": [
                {"chunk": asdict(c), "score": float(s), "text": c.text}
                for (c, s) in final_items
            ],
            "context": build_context_block(final_items),
            "refs": [
                {
                    "corpus": (c.corpus or detect_corpus_label(c.source_path)),
                    "file_name": os.path.basename(c.source_path or ""),
                    "source_path": c.source_path,
                    "start_char": c.start_char,
                    "end_char": c.end_char,
                    "chunk_index": c.chunk_index,
                    "doc_date_iso": c.doc_date_iso,
                    "doc_timestamp": c.doc_timestamp,
                }
                for (c, _s) in final_items
            ],
            "debug": {
                "candidate_count": len(cands),
                "rerank_rows": debug_rows,
            },
        }
