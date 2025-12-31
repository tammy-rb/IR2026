from __future__ import annotations

from typing import Any, List, Tuple

from .utils import delta_days, log_normalize, ts_from_chunk


def soft_decay_rerank(
    items: List[Tuple[Any, float]],
    *,
    ref_ts: int,
    alpha: float,
    h: float,
    normalize_sims: bool,
) -> List[Tuple[Any, float, float, float]]:
    """
    Recency Prior Reranking based on the paper formula:
    
        Score(q,d,t) = α * similarity + (1-α) * 0.5^(age_days / h)

    Notes:
      - If normalize_sims=True -> log normalize incoming sim scores (BM25).
      - If normalize_sims=False -> use incoming scores as-is (dense similarity).
      - Future docs are not boosted (Δt clamped to 0).
      - Missing timestamps get max time penalty.

    Returns:
      List[(chunk, final_score, norm_sim, time_score)] sorted by final_score desc.
    """
    if not items:
        return []

    sims = [float(score) for _chunk, score in items]
    norm_sims = log_normalize(sims) if normalize_sims else sims

    reranked: List[Tuple[Any, float, float, float]] = []
    for (chunk, _), sim in zip(items, norm_sims):
        doc_ts = ts_from_chunk(chunk)
        dt_days = delta_days(ref_ts, doc_ts, clamp_future=True)

        # Time score calculation: 0.5 ^ (days / half-life)
        time_score = 0.5 ** (float(dt_days) / float(h))
        
        # Final score calculation using alpha as the semantic weight
        final = (float(alpha) * float(sim)) + ((1.0 - float(alpha)) * float(time_score))

        reranked.append((chunk, float(final), float(sim), float(time_score)))

    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked