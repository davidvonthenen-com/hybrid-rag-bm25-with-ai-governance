from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class RetrievalHit:
    channel: str  # bm25_doc | bm25_chunk | bm25_neighbor | vector
    handle: str   # B1..Bn or V1..Vn
    index: str
    os_id: str
    score: float

    path: str
    category: str
    chunk_index: Optional[int] = None
    chunk_count: Optional[int] = None

    text: str = ""

    explicit_terms: Optional[List[str]] = None
    entity_overlap: Optional[int] = None

    def to_jsonable(self) -> Dict[str, Any]:
        return asdict(self)


__all__ = ["RetrievalHit"]
