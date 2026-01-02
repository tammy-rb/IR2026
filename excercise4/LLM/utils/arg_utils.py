# utils/arg_utils.py
from __future__ import annotations

from typing import List, Tuple


def parse_ks(values: List[int]) -> List[int]:
    """
    argparse will give list[int] from --k 5 10 etc.
    We also sanitize: unique, positive, sorted.
    """
    ks = sorted({int(v) for v in values if int(v) > 0})
    if not ks:
        raise ValueError("At least one positive K is required.")
    return ks


def default_pipelines() -> List[Tuple[str, str]]:
    return [
        ("fixed", "bm25"),
        ("semantic", "bm25"),
        ("fixed", "dense"),
        ("semantic", "dense"),
    ]


def parse_pipelines(pipes: List[str]) -> List[Tuple[str, str]]:
    """
    Input strings like: fixed/bm25  semantic/dense
    """
    out: List[Tuple[str, str]] = []
    for p in pipes:
        if "/" not in p:
            raise ValueError(f"Invalid pipeline '{p}'. Use chunking/repr e.g. fixed/bm25")
        chunking, repr_ = p.split("/", 1)
        chunking = chunking.strip()
        repr_ = repr_.strip()
        if chunking not in {"fixed", "semantic"}:
            raise ValueError(f"Invalid chunking '{chunking}' in pipeline '{p}'")
        if repr_ not in {"bm25", "dense"}:
            raise ValueError(f"Invalid representation '{repr_}' in pipeline '{p}'")
        out.append((chunking, repr_))
    return out
