from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Pattern


@dataclass(frozen=True)
class EvolutionDetectionResult:
    is_evolution: bool
    matched_pattern: str | None = None
    matched_text: str | None = None


def _compile_patterns(patterns: List[str]) -> List[Pattern[str]]:
    return [re.compile(p, flags=re.IGNORECASE) for p in patterns]


# High-precision English-only patterns for "evolution-like" queries
# Captures change, development, comparison across time, early vs late, etc.
DEFAULT_EVOLUTION_PATTERNS: List[str] = [
    # Explicit "how did/has X change over time"
    r"\bhow\s+did\s+.*\b(change|evolve|develop|shift|transform|progress)\b",
    r"\bhow\s+has\s+.*\b(changed|evolved|developed|shifted|transformed|progressed)\b",
    r"\bhow\s+have\s+.*\b(changed|evolved|developed|shifted|transformed|progressed)\b",

    # Stance / rhetoric / policy change
    r"\b(rhetoric|stance|policy|position|approach|attitude|view|views)\b.*\b(changed|evolved|developed|shifted)\b",
    r"\b(changed|evolved|developed|shifted)\b.*\b(rhetoric|stance|policy|position|approach|attitude|view|views)\b",

    # Over-time indicators
    r"\b(over\s+time|throughout\s+time|across\s+time)\b",
    r"\b(across|over)\s+(years|months|decades|time)\b",

    # Comparison across periods
    r"\bbetween\b.+\band\b.+",
    r"\bfrom\b.+\bto\b.+",

    # Early vs late / first vs last
    r"\b(first|earliest)\b.*\b(last|latest)\b",
    r"\b(early)\b.*\b(late)\b",

    # Speech-based evolution
    r"\b(first)\s+speech\b.*\b(last)\s+speech\b",
]

_COMPILED_DEFAULT_PATTERNS = _compile_patterns(DEFAULT_EVOLUTION_PATTERNS)


def is_evolution_query(
    query: str,
    patterns: List[Pattern[str]] | None = None,
) -> EvolutionDetectionResult:
    """
    Detect whether a query asks for a change-over-time / evolution analysis.

    Args:
        query: user query string (English)
        patterns: optional list of precompiled regex patterns

    Returns:
        EvolutionDetectionResult
    """
    q = (query or "").strip()
    if not q:
        return EvolutionDetectionResult(is_evolution=False)

    pats = patterns or _COMPILED_DEFAULT_PATTERNS

    for p in pats:
        match = p.search(q)
        if match:
            return EvolutionDetectionResult(
                is_evolution=True,
                matched_pattern=p.pattern,
                matched_text=match.group(0),
            )

    return EvolutionDetectionResult(is_evolution=False)


def detect_evolution_bool(query: str) -> bool:
    """
    Convenience wrapper: return True iff query is evolution-like.
    """
    return is_evolution_query(query).is_evolution
