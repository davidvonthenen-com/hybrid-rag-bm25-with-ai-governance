"""BM25-based re-ranking utilities for OpenSearch hits."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import bm25s
from opensearchpy.exceptions import NotFoundError

from .opensearch_client import MyOpenSearch

try:
    import Stemmer  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    Stemmer = None  # type: ignore

from .models import RetrievalHit


_BM25_STOPWORDS = "en"
_STEMMER = Stemmer.Stemmer("english") if Stemmer else None  # type: ignore[attr-defined]


def rerank_hits_with_bm25(
    question: str,
    res_long: Dict[str, Any],
    res_hot: Dict[str, Any],
    top_k: int = 10,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Re-rank hits from LONG and HOT stores using BM25.

    Parameters
    ----------
    question:
        Natural-language query posed by the user.
    res_long:
        Raw OpenSearch response for the LONG index.
    res_hot:
        Raw OpenSearch response for the HOT index.
    top_k:
        Maximum number of combined hits to return.

    Returns
    -------
    (keep_long, keep_hot, combined)
        keep_long:
            Re-ranked hits belonging to the LONG store.
        keep_hot:
            Re-ranked hits belonging to the HOT store.
        combined:
            Cross-store ranking limited to ``top_k`` documents.
    """
    if top_k <= 0:
        return [], [], []

    hits: List[Dict[str, Any]] = []
    corpus: List[str] = []

    # Flatten hits from both stores into a single list, tracking origin.
    for res in (res_long, res_hot):
        store_label = res.get("_store_label", "?")
        index_used = res.get("_index_used", "?")
        for hit in res.get("hits", {}).get("hits", []) or []:
            if not isinstance(hit, dict):
                continue
            hit["_store_label"] = store_label
            hit["_index_used"] = index_used
            hits.append(hit)
            text = (hit.get("_source", {}).get("content") or "").strip()
            corpus.append(text)

    if not hits:
        return [], [], []

    corpus_tokens = bm25s.tokenize(corpus, stopwords=_BM25_STOPWORDS, stemmer=_STEMMER)
    has_tokens = any(len(doc_tokens) > 0 for doc_tokens in corpus_tokens)
    query_tokens = bm25s.tokenize(question, stemmer=_STEMMER)

    if not has_tokens or not query_tokens:
        # Fallback: respect the original OpenSearch scores.
        sorted_hits = sorted(
            hits,
            key=lambda h: h.get("_score", float("-inf")),
            reverse=True,
        )
        top_hits = sorted_hits[: min(top_k, len(sorted_hits))]
        for hit in top_hits:
            hit.setdefault("_bm25_score", hit.get("_score"))
    else:
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        k = min(top_k, len(hits))
        results, scores = retriever.retrieve(query_tokens, k=k)

        # bm25s returns arrays shaped (n_queries, k). We only issue one query.
        doc_ids = list(results[0])
        doc_scores = list(scores[0])

        top_hits: List[Dict[str, Any]] = []
        for doc_id, score in zip(doc_ids, doc_scores):
            doc_index = int(doc_id)
            if doc_index < 0 or doc_index >= len(hits):
                continue
            hit = hits[doc_index]
            if "_original_score" not in hit and "_score" in hit:
                hit["_original_score"] = hit["_score"]
            hit["_score"] = float(score)
            hit["_bm25_score"] = float(score)
            top_hits.append(hit)

        if not top_hits:
            sorted_hits = sorted(
                hits,
                key=lambda h: h.get("_score", float("-inf")),
                reverse=True,
            )
            top_hits = sorted_hits[: min(top_k, len(sorted_hits))]

    combined = top_hits[: min(top_k, len(top_hits))]
    for hit in combined:
        hit.setdefault("_bm25_score", hit.get("_score"))

    long_label = res_long.get("_store_label", "LONG")
    hot_label = res_hot.get("_store_label", "HOT")

    keep_long = [h for h in combined if h.get("_store_label") == long_label]
    keep_hot = [h for h in combined if h.get("_store_label") == hot_label]

    return keep_long, keep_hot, combined


def _entity_should_clauses(entities: List[str]) -> List[Dict[str, Any]]:
    should: List[Dict[str, Any]] = []
    for ent in entities:
        ent_l = ent.strip().lower()
        if not ent_l:
            continue
        should.append({"term": {"explicit_terms": {"value": ent_l, "boost": 10.0}}})
        should.append({"match_phrase": {"explicit_terms_text": {"query": ent_l, "boost": 8.0}}})
        should.append({"match_phrase": {"content": {"query": ent, "boost": 3.0}}})
    return should


def _build_bm25_doc_query(question: str, entities: List[str], *, k: int) -> Dict[str, Any]:
    should = _entity_should_clauses(entities)
    return {
        "size": k,
        "_source": ["filepath", "category", "explicit_terms", "explicit_terms_text", "content"],
        "query": {
            "bool": {
                "must": [{
                    "multi_match": {
                        "query": question,
                        "fields": ["explicit_terms_text^6", "content^2", "category^0.5"],
                        "type": "best_fields",
                        "operator": "or",
                    }
                }],
                "should": should,
                "minimum_should_match": 1 if should else 0,
            }
        },
    }


def _build_bm25_chunk_query(
    question: str,
    entities: List[str],
    *,
    k: int,
    anchor_paths: Optional[List[str]] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    should = _entity_should_clauses(entities)

    operator = "and" if strict else "or"
    msm = None if strict else "30%"

    filters: List[Dict[str, Any]] = []
    if anchor_paths:
        filters.append({
            "bool": {
                "should": [
                    {"terms": {"parent_filepath": anchor_paths}},
                    {"terms": {"filepath": anchor_paths}},
                ],
                "minimum_should_match": 1,
            }
        })

    mm: Dict[str, Any] = {
        "query": question,
        "fields": ["explicit_terms_text^8", "content^2", "category^0.5"],
        "type": "best_fields",
        "operator": operator,
    }
    if msm:
        mm["minimum_should_match"] = msm

    return {
        "size": k,
        "_source": [
            "filepath", "parent_filepath", "chunk_index", "chunk_count",
            "category", "explicit_terms", "explicit_terms_text", "content",
        ],
        "query": {
            "bool": {
                "filter": filters,
                "must": [{"multi_match": mm}],
                "should": should,
                "minimum_should_match": 1 if should else 0,
            }
        },
    }


def bm25_retrieve_doc_anchors(
    bm25_client: MyOpenSearch,
    index: str,
    *,
    question: str,
    entities: List[str],
    k: int,
    observability: bool,
) -> Tuple[List[str], Dict[str, Any], List[Dict[str, Any]]]:
    q = _build_bm25_doc_query(question, entities, k=k)
    if observability:
        print("\n[BM25_DOC_QUERY]\n" + json.dumps(q, ensure_ascii=False, indent=2, sort_keys=True, default=str))

    res = bm25_client.search(index=index, body=q)
    hits = res.get("hits", {}).get("hits", []) or []

    paths: List[str] = []
    for h in hits:
        src = h.get("_source", {}) or {}
        fp = src.get("filepath")
        if fp:
            paths.append(fp)

    # unique preserve order
    seen = set()
    out: List[str] = []
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out, q, hits


def bm25_retrieve_chunks(
    bm25_client: MyOpenSearch,
    index: str,
    *,
    question: str,
    entities: List[str],
    k: int,
    anchor_paths: Optional[List[str]],
    neighbor_window: int,
    observability: bool,
) -> Tuple[List[RetrievalHit], Dict[str, Any], List[Dict[str, Any]]]:
    attempts: List[Tuple[Optional[List[str]], bool]] = []
    if anchor_paths:
        attempts.extend([
            (anchor_paths, True),
            (anchor_paths, False),
            (None, True),
            (None, False),
        ])
    else:
        attempts.extend([(None, True), (None, False)])

    q: Dict[str, Any] = {}
    raw_hits: List[Dict[str, Any]] = []
    for ap, strict in attempts:
        q = _build_bm25_chunk_query(question, entities, k=k, anchor_paths=ap, strict=strict)
        if observability:
            print("\n[BM25_CHUNK_QUERY]\n" + json.dumps({**q, "_note": f"anchor={bool(ap)} strict={strict}"}, ensure_ascii=False, indent=2, sort_keys=True, default=str))
        res = bm25_client.search(index=index, body=q)
        raw_hits = res.get("hits", {}).get("hits", []) or []
        if raw_hits:
            break

    ent_set = {e.strip().lower() for e in entities if e.strip()}
    hits: List[RetrievalHit] = []
    for i, h in enumerate(raw_hits, start=1):
        src = h.get("_source", {}) or {}
        explicit_terms = src.get("explicit_terms") or []
        overlap = None
        if ent_set and explicit_terms:
            overlap = len({t.strip().lower() for t in explicit_terms} & ent_set)

        path = src.get("parent_filepath") or src.get("filepath") or ""
        hits.append(
            RetrievalHit(
                channel="bm25_chunk",
                handle=f"B{i}",
                index=index,
                os_id=h.get("_id", ""),
                score=float(h.get("_score") or 0.0),
                path=path,
                category=src.get("category") or "",
                chunk_index=src.get("chunk_index"),
                chunk_count=src.get("chunk_count"),
                text=(src.get("content") or "").strip(),
                explicit_terms=explicit_terms,
                entity_overlap=overlap,
            )
        )

    if neighbor_window > 0 and hits:
        expanded = _expand_bm25_neighbors(
            bm25_client,
            index=index,
            seed_hits=hits,
            window=neighbor_window,
            observability=observability,
        )
        # Re-handle
        hits = [
            RetrievalHit(**{**h.to_jsonable(), "handle": f"B{i+1}"}) for i, h in enumerate(expanded)
        ]

    return hits, q, raw_hits


def _expand_bm25_neighbors(
    bm25_client: Any,
    *,
    index: str,
    seed_hits: List[RetrievalHit],
    window: int,
    observability: bool,
) -> List[RetrievalHit]:
    # group seed positions per doc path
    doc_positions: Dict[str, List[int]] = {}
    seed_score: Dict[str, float] = {}
    for h in seed_hits:
        seed_score[h.os_id] = h.score
        if h.chunk_index is None:
            continue
        doc_positions.setdefault(h.path, []).append(int(h.chunk_index))

    expanded: List[RetrievalHit] = []
    for path, positions in doc_positions.items():
        idxs: set[int] = set()
        for pos in positions:
            for off in range(-window, window + 1):
                if pos + off >= 0:
                    idxs.add(pos + off)
        for ci in sorted(idxs):
            chunk_id = f"{path}::chunk-{ci:03d}"
            try:
                doc = bm25_client.get(index=index, id=chunk_id)
            except NotFoundError:
                continue
            src = doc.get("_source", {}) or {}
            expanded.append(
                RetrievalHit(
                    channel="bm25_chunk" if chunk_id in seed_score else "bm25_neighbor",
                    handle="B0",  # temporary, reassigned later
                    index=index,
                    os_id=chunk_id,
                    score=float(seed_score.get(chunk_id, 0.0)),
                    path=src.get("parent_filepath") or src.get("filepath") or path,
                    category=src.get("category") or "",
                    chunk_index=src.get("chunk_index"),
                    chunk_count=src.get("chunk_count"),
                    text=(src.get("content") or "").strip(),
                    explicit_terms=src.get("explicit_terms") or [],
                    entity_overlap=None,
                )
            )

    # de-dup (path, chunk_index) preserve order
    seen: set[Tuple[str, Optional[int]]] = set()
    uniq: List[RetrievalHit] = []
    for h in expanded:
        key = (h.path, h.chunk_index)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(h)

    if observability:
        print(f"\n[BM25_NEIGHBOR_EXPANSION] window={window} expanded_chunks={len(uniq)}")

    return uniq


__all__ = [
    "rerank_hits_with_bm25",
    "_entity_should_clauses",
    "_build_bm25_doc_query",
    "_build_bm25_chunk_query",
    "bm25_retrieve_doc_anchors",
    "bm25_retrieve_chunks",
    "_expand_bm25_neighbors",
]
