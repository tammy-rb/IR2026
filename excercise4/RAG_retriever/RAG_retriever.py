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
    FAISS_FIXED_DIR,
    FAISS_SEM_DIR,
)
from .retrievers.bm25 import BM25Retriever
from .retrievers.dense_faiss import DenseFAISSRetriever
from .utils import build_context_block, detect_corpus_label

from .duckling_time_analysis import analyze_query_time
from .temporal_policy import RetrievalPlan, build_retrieval_plan

from .temporal.hard_filter import hard_filter
from .temporal.soft_decay import soft_decay_rerank

load_dotenv()

# ============================================================
# Configuration
# ============================================================

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
            overs = int(plan.oversample)
            filtered = hard_filter(cands, start_ts=plan.start_ts, end_ts=plan.end_ts)

            # Hard filter might leave <k results -> expand oversampling
            while len(filtered) < k and overs < max_hard_oversample:
                overs = min(max_hard_oversample, max(overs * 2, overs + 200))
                cands = _retrieve_candidates(overs)
                filtered = hard_filter(cands, start_ts=plan.start_ts, end_ts=plan.end_ts)

            filtered.sort(key=lambda x: x[1], reverse=True)
            final_items = filtered[:k]

        elif plan.strategy == "soft":
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
