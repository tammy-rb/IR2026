from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from paths import (
    BM25_FIXED_DIR,
    BM25_SEM_DIR,
    CHUNKS_FIXED_JSONL,
    CHUNKS_SEM_JSONL,
    OPENAI_QDRANT_FIXED_DIR,
    OPENAI_QDRANT_SEMANTIC_DIR,
)
from config import (
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION_FIXED,
    QDRANT_COLLECTION_SEMANTIC,
    OPENAI_EMBED_MODEL,
)
from .retrievers.bm25 import BM25Retriever
from .retrievers.dense_qdrant import QdrantDenseRetriever
from .utils import build_context_block, detect_corpus_label
from .prefilter.chuncks_selector import ChunkFilter

from .duckling_time_analysis import analyze_query_time
from .temporal_policy import RetrievalPlan, build_retrieval_plan

from .temporal.soft_decay import soft_decay_rerank

# Import temporal utility functions
from .temporal_utils import (
    compute_corpus_bounds,
    months_to_seconds,
    ts_to_iso,
    format_evolution_context,
)

load_dotenv()

# ============================================================
# Configuration
# ============================================================

STOP_WORDS = "english"


# ============================================================
# Pipeline model
# ============================================================

@dataclass(frozen=True)
class Pipeline:
    chunking: str         # "fixed" or "semantic"
    representation: str   # "bm25" or "dense"
    bm25: Optional[BM25Retriever]
    dense: Optional[QdrantDenseRetriever]


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
        self._corpus_bounds: Dict[str, Tuple[int, int]] = {}  # chunking -> (min_ts, max_ts)
        self._load_all()

    def _load_all(self) -> None:
        # Compute corpus bounds per chunking ONCE (independent of representation)
        self._corpus_bounds["fixed"] = compute_corpus_bounds(CHUNKS_FIXED_JSONL)
        self._corpus_bounds["semantic"] = compute_corpus_bounds(CHUNKS_SEM_JSONL)
        
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
            dense = QdrantDenseRetriever(
                qdrant_host=QDRANT_HOST,
                qdrant_port=QDRANT_PORT,
                collection_name=QDRANT_COLLECTION_FIXED if chunking == "fixed" else QDRANT_COLLECTION_SEMANTIC,
                model=OPENAI_EMBED_MODEL,
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

    def get_topk_filtered(
        self,
        query: str,
        chunking: str,
        representation: str,
        k: int,
        *,
        flt: ChunkFilter,
        oversample: int = 0,
    ) -> Dict[str, Any]:
        """
        Metadata-prefiltered retrieval.

        Retrieves top-k chunks after applying metadata-based prefiltering
        (e.g., time range, corpus, doc_ids) BEFORE scoring.

        Args:
            query: Search query
            chunking: "fixed" or "semantic"
            representation: "bm25" or "dense" (only bm25 supports prefiltering currently)
            k: Number of results to return
            flt: ChunkFilter specifying metadata constraints
            oversample: Additional candidates to retrieve for post-processing

        Returns:
            Dictionary with retrieved chunks, context, and refs
        """
        pipe = self._get_pipe(chunking, representation)

        if representation == "bm25":
            if pipe.bm25 is None:
                raise RuntimeError("BM25 pipeline not initialized.")
            retrieved = pipe.bm25.search_candidates_prefiltered(
                query, k, flt=flt, oversample=oversample
            )
        else:
            # Dense (Qdrant) retriever
            if pipe.dense is None:
                raise RuntimeError("Dense pipeline not initialized.")
            retrieved = pipe.dense.search_candidates_prefiltered(
                query, k, flt=flt, oversample=oversample
            )

        return {
            "filter": {
                "time_min_ts": flt.time_min_ts,
                "time_max_ts": flt.time_max_ts,
                "require_timestamp": flt.require_timestamp,
                "corpora": list(flt.corpora) if flt.corpora else None,
                "chunking_methods": list(flt.chunking_methods) if flt.chunking_methods else None,
                "doc_ids": list(flt.doc_ids) if flt.doc_ids else None,
            },
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
    ) -> Dict[str, Any]:
        """
        Stage 3: Time-aware retrieval.

        Flow:
          1) Analyze query time via Duckling
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

        # 3) retrieve candidates based on strategy
        final_items: List[Tuple[Any, float]] = []
        debug_rows: List[Dict[str, Any]] = []
        candidate_count = 0

        if plan.strategy == "hard":
            # Use prefiltered retrieval instead of oversampling + post-filtering
            flt = ChunkFilter(
                time_min_ts=plan.start_ts,
                time_max_ts=plan.end_ts,
                require_timestamp=True,
            )
            
            if representation == "bm25":
                if pipe.bm25 is None:
                    raise RuntimeError("BM25 pipeline not initialized.")
                filtered = pipe.bm25.search_candidates_prefiltered(query, k, flt=flt, oversample=0)
            else:
                if pipe.dense is None:
                    raise RuntimeError("Dense pipeline not initialized.")
                filtered = pipe.dense.search_candidates_prefiltered(query, k, flt=flt, oversample=0)
            
            candidate_count = len(filtered)
            final_items = filtered[:k]

        elif plan.strategy == "soft":
            # Retrieve candidates with oversampling for soft decay
            if representation == "bm25":
                if pipe.bm25 is None:
                    raise RuntimeError("BM25 pipeline not initialized.")
                cands = pipe.bm25.search_candidates(query, k, oversample=plan.oversample)
            else:
                if pipe.dense is None:
                    raise RuntimeError("Dense pipeline not initialized.")
                cands = pipe.dense.search_candidates(query, k, oversample=plan.oversample)
            
            candidate_count = len(cands)
            
            # Normalize sims for BM25 only; dense already in a comparable [0,1] similarity space
            normalize = (representation == "bm25")

            ranked = soft_decay_rerank(
                cands,
                ref_ts=plan.ref_ts,
                alpha=plan.alpha,
                h=plan.h,
                normalize_sims=normalize,
            )

            final_items = [(c, s_final) for (c, s_final, _su, _ts) in ranked[:k]]

            # Debug rows (top ~50 or k*5)
            for (c, s_final, sim_used, time_s) in ranked[: min(len(ranked), max(50, k * 5))]:
                debug_rows.append(
                    {
                        "doc_date_iso": getattr(c, "doc_date_iso", None),
                        "doc_timestamp": getattr(c, "doc_timestamp", None),
                        "final_score": float(s_final),
                        "sim_used": float(sim_used),
                        "time_score": float(time_s),
                        "source_path": getattr(c, "source_path", None),
                        "chunk_index": getattr(c, "chunk_index", None),
                    }
                )

        else:
            # "none" -> behave like baseline
            if representation == "bm25":
                if pipe.bm25 is None:
                    raise RuntimeError("BM25 pipeline not initialized.")
                cands = pipe.bm25.search_candidates(query, k, oversample=0)
            else:
                if pipe.dense is None:
                    raise RuntimeError("Dense pipeline not initialized.")
                cands = pipe.dense.search_candidates(query, k, oversample=0)
            candidate_count = len(cands)
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
                "candidate_count": candidate_count,
                "rerank_rows": debug_rows,
            },
        }

    # -------------------------
    # Stage 4: Evolution retrieval
    # -------------------------
    # Note: Helper functions (compute_corpus_bounds, months_to_seconds, ts_to_iso,
    #       format_evolution_context) are in temporal_utils.py
    
    def get_topk_evolution(
        self,
        query: str,
        chunking: str,
        representation: str,
        k: int,
        *,
        window_months: int = 8,
        oversample: int = 200,
        max_oversample: int = 800,
    ) -> Dict[str, Any]:
        """
        Stage 4: Evolution retrieval (double retrieval).

        Retrieves:
          - Top-K relevant chunks from the EARLIEST window (first window_months)
          - Top-K relevant chunks from the LATEST window   (last  window_months)

        Returns a formatted context string with both time periods, sorted by
        distance to window boundary (closest to farthest).

        Args:
            query: Search query
            chunking: "fixed" or "semantic"
            representation: "bm25" or "dense"
            k: Number of chunks to retrieve from each window
            window_months: Size of early/late windows in months (default 8)
            oversample: Initial oversample factor (default 200)
            max_oversample: Maximum oversample limit (default 800)

        Returns:
            Dictionary with retrieved chunks, formatted context, and metadata

        """
        pipe = self._get_pipe(chunking, representation)

        # Corpus bounds for this chunking method
        if chunking not in self._corpus_bounds:
            raise RuntimeError(f"Missing corpus bounds for chunking={chunking!r}")

        min_ts, max_ts = self._corpus_bounds[chunking]
        w_sec = months_to_seconds(window_months)

        early_range = (min_ts, min(min_ts + w_sec, max_ts))
        late_range = (max(max_ts - w_sec, min_ts), max_ts)

        early_start_ts, early_end_ts = early_range
        late_start_ts, late_end_ts = late_range

        # Use prefiltering for evolution windows
        early_flt = ChunkFilter(
            time_min_ts=early_start_ts,
            time_max_ts=early_end_ts,
            require_timestamp=True,
        )
        
        late_flt = ChunkFilter(
            time_min_ts=late_start_ts,
            time_max_ts=late_end_ts,
            require_timestamp=True,
        )

        # Retrieve from each window using prefiltering
        if representation == "bm25":
            if pipe.bm25 is None:
                raise RuntimeError("BM25 pipeline not initialized.")
            early_items = pipe.bm25.search_candidates_prefiltered(query, k, flt=early_flt, oversample=0)
            late_items = pipe.bm25.search_candidates_prefiltered(query, k, flt=late_flt, oversample=0)
        else:
            if pipe.dense is None:
                raise RuntimeError("Dense pipeline not initialized.")
            early_items = pipe.dense.search_candidates_prefiltered(query, k, flt=early_flt, oversample=0)
            late_items = pipe.dense.search_candidates_prefiltered(query, k, flt=late_flt, oversample=0)

        context = format_evolution_context(
            early_items=early_items,
            late_items=late_items,
            early_range=early_range,
            late_range=late_range,
        )

        def _refs_for(items: List[Tuple[Any, float]]) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            for (c, _s) in items:
                out.append(
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
                )
            return out

        return {
            "query": query,
            "chunking": chunking,
            "representation": representation,
            "k": k,
            "window_months": window_months,
            "ranges": {
                "early": {
                    "start_ts": early_start_ts,
                    "end_ts": early_end_ts,
                    "start_iso": ts_to_iso(early_start_ts),
                    "end_iso": ts_to_iso(early_end_ts),
                },
                "late": {
                    "start_ts": late_start_ts,
                    "end_ts": late_end_ts,
                    "start_iso": ts_to_iso(late_start_ts),
                    "end_iso": ts_to_iso(late_end_ts),
                },
            },
            "retrieved": {
                "early": [{"chunk": asdict(c), "score": float(s), "text": c.text} for (c, s) in early_items],
                "late": [{"chunk": asdict(c), "score": float(s), "text": c.text} for (c, s) in late_items],
            },
            "context": context,
            "refs": {
                "early": _refs_for(early_items),
                "late": _refs_for(late_items),
            },
            "debug": {
                "early_found": len(early_items),
                "late_found": len(late_items),
                "corpus_bounds": {"min_ts": min_ts, "max_ts": max_ts},
            },
        }
