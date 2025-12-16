"""Citation extraction utilities for query audit logging."""
from __future__ import annotations

import re
from typing import List

# Matches BM25 and vector handles like B1 or V2 regardless of surrounding brackets/spacing.
_CITATION_RE = re.compile(r"[BV]\d+")


def extract_citations(answer: str) -> List[str]:
    """Return sorted unique citation handles found in ``answer``.

    Args:
        answer: Final LLM answer text that should contain [B#] or [V#] markers.

    Returns:
        Sorted, de-duplicated citation handles (e.g., ["B1", "V2"]).
    """

    if not answer:
        return []

    return sorted(set(_CITATION_RE.findall(answer)))


__all__ = ["extract_citations"]
