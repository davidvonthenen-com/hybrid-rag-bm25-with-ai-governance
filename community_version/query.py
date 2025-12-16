#!/usr/bin/env python3
"""
Hybrid RAG query runner (no LangChain, no LLM-driven policy).

Separation of concerns (explicit + auditable):
- BM25 = grounding/evidence channel (entity-biased lexical search)
- Vector kNN = semantic/support channel (phrasing/terminology), typically filtered
  to BM25-anchored documents to prevent semantic drift.

Generation:
- Default: 2-pass LLM
  1) grounded draft from BM25-only evidence (citations [B#])
  2) optional rewrite for clarity using vector context (citations [V#] allowed
     only for non-factual clarifications; factual claims must stay grounded)
- Optional: single-pass with both contexts in separate blocks.

All OpenSearch queries and retrieved hits can be printed (--observability) and/or
saved as JSONL (--save-results) for auditability.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from opensearchpy.exceptions import NotFoundError

from common.embeddings import EmbeddingModel, to_list
from common.llm import load_llm
from common.logging import get_logger
from common.named_entity import normalize_entities, post_ner
from common.opensearch_client import create_long_client, create_vector_client

LOGGER = get_logger(__name__)

_CITATION_RE = re.compile(r"\[(B\d+|V\d+)\]")


# --------------------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------------------

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


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hybrid RAG query (BM25 grounding + vector semantic support) with full auditability."
    )
    p.add_argument("--question", help="User question to answer.")

    p.add_argument("--observability", action="store_true", default=False)
    p.add_argument("--save-results", type=str, default=None, help="Append JSONL records to this path.")

    # Retrieval knobs
    p.add_argument("--top-k", type=int, default=10, help="Total evidence chunks budget.")
    p.add_argument("--bm25-k", type=int, default=None, help="BM25 chunk budget (default ~60% of top-k).")
    p.add_argument("--vec-k", type=int, default=None, help="Vector chunk budget (default remainder).")
    p.add_argument("--bm25-doc-k", type=int, default=20, help="Doc-level BM25 anchors to fetch.")
    p.add_argument("--neighbor-window", type=int, default=0, help="Add ±N adjacent chunks around BM25 hits.")
    p.add_argument("--vec-filter", choices=["anchor", "none"], default="anchor",
                   help="Filter vector search to BM25-anchored docs when possible.")

    # LLM knobs
    p.add_argument("--model", type=str, default=None, help="Optional override model name (if supported).")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=700)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--single-pass", action="store_true", default=False)

    # Index override hooks (optional)
    p.add_argument("--bm25-full-index", type=str, default=None)
    p.add_argument("--bm25-chunk-index", type=str, default=None)
    p.add_argument("--vec-index", type=str, default=None)

    return p.parse_args(argv)


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True, default=str)


def _extract_citations(answer: str) -> List[str]:
    return sorted(set(m.group(1) for m in _CITATION_RE.finditer(answer)))


def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    """Fallback formatting for non-chat completion interfaces."""
    parts: List[str] = []
    for m in messages:
        role = (m.get("role") or "user").upper()
        content = m.get("content") or ""
        parts.append(f"{role}:\n{content}".strip())
    return "\n\n".join(parts).strip()


# --------------------------------------------------------------------------------------
# Entity extraction (no LLM policy)
# --------------------------------------------------------------------------------------

def extract_entities(question: str) -> List[str]:
    try:
        ner_payload = post_ner(question)
        return normalize_entities(ner_payload)
    except Exception as exc:
        LOGGER.warning("NER failed (%s). Proceeding without entities.", exc)
        return []


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


# --------------------------------------------------------------------------------------
# OpenSearch queries
# --------------------------------------------------------------------------------------

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


def _build_vector_query(
    query_vector: List[float],
    *,
    k: int,
    candidate_k: int,
    anchor_paths: Optional[List[str]] = None,
    vector_field: str = "embedding",
) -> Dict[str, Any]:
    knn_clause = {"knn": {vector_field: {"vector": query_vector, "k": candidate_k}}}
    return {
        "size": k,
        "_source": ["path", "category", "chunk_index", "chunk_count", "text"],
        "query": {
            "bool": {
                "must": [knn_clause],
                "filter": [{"terms": {"path": anchor_paths}}] if anchor_paths else [],
            }
        },
    }


# --------------------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------------------

def bm25_retrieve_doc_anchors(
    bm25_client: Any,
    index: str,
    *,
    question: str,
    entities: List[str],
    k: int,
    observability: bool,
) -> Tuple[List[str], Dict[str, Any], List[Dict[str, Any]]]:
    q = _build_bm25_doc_query(question, entities, k=k)
    if observability:
        print("\n[BM25_DOC_QUERY]\n" + _json(q))

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
    bm25_client: Any,
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
            print("\n[BM25_CHUNK_QUERY]\n" + _json({**q, "_note": f"anchor={bool(ap)} strict={strict}"}))
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


def vector_retrieve_chunks(
    vec_client: Any,
    index: str,
    *,
    question: str,
    anchor_paths: Optional[List[str]],
    k: int,
    candidate_k: int,
    observability: bool,
    vector_field: str = "embedding",
) -> Tuple[List[RetrievalHit], Dict[str, Any], List[Dict[str, Any]]]:
    embedder = EmbeddingModel()
    qvec = to_list(embedder.encode([question])[0])

    q = _build_vector_query(qvec, k=k, candidate_k=candidate_k, anchor_paths=anchor_paths, vector_field=vector_field)

    if observability:
        dbg = json.loads(json.dumps(q))
        try:
            dbg["query"]["bool"]["must"][0]["knn"][vector_field]["vector"] = "<omitted>"
        except Exception:
            pass
        print("\n[VECTOR_QUERY]\n" + _json(dbg))

    res = vec_client.search(index=index, body=q)
    raw_hits = res.get("hits", {}).get("hits", []) or []

    hits: List[RetrievalHit] = []
    for i, h in enumerate(raw_hits, start=1):
        src = h.get("_source", {}) or {}
        hits.append(
            RetrievalHit(
                channel="vector",
                handle=f"V{i}",
                index=index,
                os_id=h.get("_id", ""),
                score=float(h.get("_score") or 0.0),
                path=src.get("path") or "",
                category=src.get("category") or "",
                chunk_index=src.get("chunk_index"),
                chunk_count=src.get("chunk_count"),
                text=(src.get("text") or "").strip(),
            )
        )

    return hits, q, raw_hits


# --------------------------------------------------------------------------------------
# LLM call + response normalization
# --------------------------------------------------------------------------------------

def _extract_llm_text(resp: Any) -> str:
    """Normalize many possible completion response shapes into a plain string."""
    if resp is None:
        return ""

    if isinstance(resp, str):
        return resp.strip()

    # llama_cpp and many other clients return dicts
    if isinstance(resp, dict):
        choices = resp.get("choices")
        if isinstance(choices, list) and choices:
            c0 = choices[0]
            if isinstance(c0, dict):
                # chat completion style
                msg = c0.get("message")
                if isinstance(msg, dict) and "content" in msg:
                    return str(msg.get("content") or "").strip()
                # text completion style
                if "text" in c0:
                    return str(c0.get("text") or "").strip()
                # streaming delta style
                delta = c0.get("delta")
                if isinstance(delta, dict) and "content" in delta:
                    return str(delta.get("content") or "").strip()
        # some wrappers return {"content": "..."}
        if "content" in resp and isinstance(resp["content"], str):
            return resp["content"].strip()
        return str(resp).strip()

    # OpenAI v1 python client response objects have .choices
    choices = getattr(resp, "choices", None)
    if choices and isinstance(choices, list):
        c0 = choices[0]
        # chat
        msg = getattr(c0, "message", None)
        if msg is not None:
            content = getattr(msg, "content", None)
            if isinstance(content, str):
                return content.strip()
        # completion
        text = getattr(c0, "text", None)
        if isinstance(text, str):
            return text.strip()

    # Fallback: stringify
    return str(resp).strip()


def call_llm_chat(
    llm: Any,
    *,
    messages: List[Dict[str, str]],
    model: Optional[str],
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> str:
    """Chat-style completion, with explicit support for llama_cpp.Llama.

    This fixes the failure mode where calling a llama_cpp model directly
    (llm(prompt)) returns a raw dict completion object rather than the text.
    """
    # 1) llama_cpp: use create_chat_completion (best behavior for role-based prompts)
    create_chat = getattr(llm, "create_chat_completion", None)
    if callable(create_chat):
        resp = create_chat(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        return _extract_llm_text(resp)

    # 2) OpenAI-like client: llm.chat.completions.create(...)
    chat = getattr(llm, "chat", None)
    if chat is not None:
        completions = getattr(chat, "completions", None)
        create = getattr(completions, "create", None) if completions is not None else None
        if callable(create):
            kwargs: Dict[str, Any] = {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if model:
                kwargs["model"] = model
            resp = create(**kwargs)
            return _extract_llm_text(resp)

    # 3) Some wrappers provide invoke(messages) -> str/dict
    invoke = getattr(llm, "invoke", None)
    if callable(invoke):
        resp = invoke(messages)
        return _extract_llm_text(resp)

    # 4) Text completion fallback: convert messages to prompt and try create_completion / __call__
    prompt = _messages_to_prompt(messages)

    create_completion = getattr(llm, "create_completion", None)
    if callable(create_completion):
        resp = create_completion(prompt=prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        return _extract_llm_text(resp)

    if callable(llm):
        # llama_cpp.__call__ uses a low default max_tokens; pass explicit args if supported
        try:
            resp = llm(prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        except TypeError:
            resp = llm(prompt)
        return _extract_llm_text(resp)

    raise TypeError("Unsupported LLM client type returned by load_llm().")


# --------------------------------------------------------------------------------------
# Prompt building (explicit separation)
# --------------------------------------------------------------------------------------

def _format_hits(hits: List[RetrievalHit], *, title: str) -> str:
    blocks = [f"## {title}"]
    for h in hits:
        meta = [f"handle={h.handle}", f"score={h.score:.4f}", f"path={h.path}"]
        if h.chunk_index is not None and h.chunk_count is not None:
            meta.append(f"chunk={h.chunk_index}/{h.chunk_count}")
        if h.category:
            meta.append(f"category={h.category}")
        blocks.append(f"[{h.handle}] ({', '.join(meta)})\n{h.text}".strip())
    return "\n\n".join(blocks).strip()


def build_grounding_prompt(question: str, bm25_hits: List[RetrievalHit]) -> List[Dict[str, str]]:
    print("***** Building grounding prompt...")

    context = _format_hits(bm25_hits, title="BM25 Grounding Evidence (authoritative facts)")
    system = (
        "Answer using ONLY the BM25 Grounding Evidence.\n"
        "Rules:\n"
        "- Every factual claim must cite at least one [B#].\n"
        "- If evidence does not support the answer, respond: I don't know based on the provided evidence.\n"
        "- Do not quote the evidence headers/metadata. Use them only for citations.\n"
        "- If there is conflicting evidence, disclose the conflict and data.\n"
    )
    user = f"QUESTION:\n{question}\n\nGROUNDING_EVIDENCE:\n{context}\n"

    prompt_details = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    print("=====================================================")
    print(_messages_to_prompt(prompt_details))
    print("=====================================================")

    return prompt_details


def build_vector_only_prompt(question: str, vec_hits: List[RetrievalHit]) -> List[Dict[str, str]]:
    print("***** Building vector-only prompt...")
    context = _format_hits(vec_hits, title="Vector Evidence (semantic fallback)")
    system = (
        "Answer using ONLY the Vector Evidence.\n"
        "Rules:\n"
        "- Every factual claim must cite at least one [V#].\n"
        "- If evidence does not support the answer, respond: I don't know based on the provided evidence.\n"
        "- Do not quote the evidence headers/metadata.\n"
    )
    user = f"QUESTION:\n{question}\n\nEVIDENCE:\n{context}\n"
    prompt_details = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    print("=====================================================")
    print(_messages_to_prompt(prompt_details))
    print("=====================================================")

    return prompt_details


def build_refine_prompt(question: str, grounded_draft: str, vec_hits: List[RetrievalHit]) -> List[Dict[str, str]]:
    print("***** Building refine prompt...")

    vec_context = _format_hits(vec_hits, title="Vector Semantic Context (phrasing/terminology support)")
    system = (
        "Rewrite the grounded draft for clarity.\n"
        "Rules:\n"
        "- Do NOT add new factual claims beyond what appears in the draft.\n"
        "- Preserve [B#] citations exactly.\n"
        "- You may add brief non-factual clarifications supported by vector context and cite [V#].\n"
        "- Output ONLY the rewritten answer.\n"
        "- If there is conflicting evidence, disclose the conflict and data.\n"
    )
    user = (
        f"QUESTION:\n{question}\n\n"
        f"GROUNDED_DRAFT:\n{grounded_draft}\n\n"
        f"{vec_context}\n"
    )
    prompt_details = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    print("=====================================================")
    print(_messages_to_prompt(prompt_details))
    print("=====================================================")

    return prompt_details


def build_single_pass_prompt(question: str, bm25_hits: List[RetrievalHit], vec_hits: List[RetrievalHit]) -> List[Dict[str, str]]:
    print("***** Building single-pass prompt...")

    bm25_context = _format_hits(bm25_hits, title="BM25 Grounding Evidence (authoritative facts)")
    vec_context = _format_hits(vec_hits, title="Vector Semantic Context (phrasing/terminology support)")
    system = (
        "Answer using the provided contexts.\n"
        "Separation of concerns:\n"
        "- Use BM25 Grounding Evidence for factual claims.\n"
        "- Use Vector Semantic Context only for wording/terminology, not new facts.\n"
        "Rules:\n"
        "- Every factual claim must cite [B#].\n"
        "- If BM25 evidence does not support the answer, respond: I don't know based on the provided evidence.\n"
    )
    user = f"QUESTION:\n{question}\n\n{bm25_context}\n\n{vec_context}\n"
    prompt_details = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    print("=====================================================")
    print(_messages_to_prompt(prompt_details))
    print("=====================================================")

    return prompt_details


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------

def run_one(
    question: str,
    *,
    bm25_client: Any,
    vec_client: Any,
    llm: Any,
    args: argparse.Namespace,
) -> Tuple[str, Dict[str, Any]]:
    entities = extract_entities(question)

    # Indices
    bm25_full_index = args.bm25_full_index or getattr(bm25_client.settings, "opensearch_full_index", None) or getattr(bm25_client.settings, "opensearch_hot_index", None)
    bm25_chunk_index = args.bm25_chunk_index or getattr(bm25_client.settings, "opensearch_long_index", None)
    vec_index = args.vec_index or getattr(vec_client.settings, "opensearch_vector_index", None)
    if not bm25_full_index or not bm25_chunk_index or not vec_index:
        raise RuntimeError("Could not resolve required index names (full, chunk, vector). Check settings/CLI overrides.")

    # Budget split
    top_k = int(args.top_k)
    bm25_k = int(args.bm25_k) if args.bm25_k is not None else max(1, int(round(top_k * 0.6)))
    vec_k = int(args.vec_k) if args.vec_k is not None else max(0, top_k - bm25_k)
    if bm25_k + vec_k != top_k:
        vec_k = max(0, top_k - bm25_k)

    # 1) doc anchors
    anchor_paths, bm25_doc_query, bm25_doc_raw = bm25_retrieve_doc_anchors(
        bm25_client,
        bm25_full_index,
        question=question,
        entities=entities,
        k=int(args.bm25_doc_k),
        observability=args.observability,
    )

    # 2) bm25 chunks
    bm25_hits, bm25_chunk_query, bm25_chunk_raw = bm25_retrieve_chunks(
        bm25_client,
        bm25_chunk_index,
        question=question,
        entities=entities,
        k=bm25_k,
        anchor_paths=anchor_paths if anchor_paths else None,
        neighbor_window=int(args.neighbor_window),
        observability=args.observability,
    )

    # 3) vector chunks
    vec_anchor_paths: Optional[List[str]] = None
    if args.vec_filter == "anchor" and len(anchor_paths) >= 2:
        vec_anchor_paths = anchor_paths

    vec_hits: List[RetrievalHit] = []
    vec_query: Dict[str, Any] = {}
    vec_raw: List[Dict[str, Any]] = []
    if vec_k > 0:
        vec_hits, vec_query, vec_raw = vector_retrieve_chunks(
            vec_client,
            vec_index,
            question=question,
            anchor_paths=vec_anchor_paths,
            k=vec_k,
            candidate_k=max(vec_k * 5, 50),
            observability=args.observability,
            vector_field="embedding",
        )
        # deterministic top-up if anchor filter starves results
        if vec_anchor_paths and len(vec_hits) < vec_k:
            topup, _, _ = vector_retrieve_chunks(
                vec_client,
                vec_index,
                question=question,
                anchor_paths=None,
                k=vec_k * 2,
                candidate_k=max(vec_k * 10, 100),
                observability=args.observability,
                vector_field="embedding",
            )
            seen = {(h.path, h.chunk_index) for h in vec_hits}
            for h in topup:
                key = (h.path, h.chunk_index)
                if key in seen:
                    continue
                vec_hits.append(h)
                seen.add(key)
                if len(vec_hits) >= vec_k:
                    break
            vec_hits = [RetrievalHit(**{**h.to_jsonable(), "handle": f"V{i+1}"}) for i, h in enumerate(vec_hits)]

    if args.observability:
        print("\n[ENTITIES]", entities)
        print(f"\n[ANCHORS] {len(anchor_paths)}")
        for pth in anchor_paths[:10]:
            print("  -", pth)
        print(f"\n[BM25_HITS] {len(bm25_hits)}")
        for h in bm25_hits[: min(10, len(bm25_hits))]:
            print(f"  {h.handle} score={h.score:.3f} chunk={h.chunk_index} path={h.path}")
        print(f"\n[VEC_HITS] {len(vec_hits)} filter={'ON' if vec_anchor_paths else 'OFF'}")
        for h in vec_hits[: min(10, len(vec_hits))]:
            print(f"  {h.handle} score={h.score:.3f} chunk={h.chunk_index} path={h.path}")

    # Generation
    model = args.model
    temperature = float(args.temperature)
    top_p = float(args.top_p)
    max_tokens = int(args.max_tokens)

    grounded_draft: Optional[str] = None
    if args.single_pass:
        msgs = build_single_pass_prompt(question, bm25_hits=bm25_hits, vec_hits=vec_hits)
        answer = call_llm_chat(llm, messages=msgs, model=model, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    else:
        if bm25_hits:
            msgs_a = build_grounding_prompt(question, bm25_hits=bm25_hits)
        else:
            # no BM25 evidence, use vector-only evidence (still citation-restricted)
            msgs_a = build_vector_only_prompt(question, vec_hits=vec_hits)
        grounded_draft = call_llm_chat(llm, messages=msgs_a, model=model, temperature=temperature, top_p=top_p, max_tokens=max_tokens)

        if bm25_hits and vec_hits:
            msgs_b = build_refine_prompt(question, grounded_draft=grounded_draft, vec_hits=vec_hits)
            answer = call_llm_chat(llm, messages=msgs_b, model=model, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        else:
            answer = grounded_draft

    citations = _extract_citations(answer)

    audit: Dict[str, Any] = {
        "question": question,
        "entities": entities,
        "indices": {
            "bm25_full": bm25_full_index,
            "bm25_chunks": bm25_chunk_index,
            "vector_chunks": vec_index,
        },
        "retrieval": {
            "anchor_paths": anchor_paths,
            "bm25_doc_query": bm25_doc_query,
            "bm25_chunk_query": bm25_chunk_query,
            "vector_query": vec_query,
            "bm25_doc_hits": [
                {
                    "filepath": (h.get("_source", {}) or {}).get("filepath"),
                    "category": (h.get("_source", {}) or {}).get("category"),
                    "score": float(h.get("_score") or 0.0),
                }
                for h in bm25_doc_raw
            ],
            "bm25_hits": [h.to_jsonable() for h in bm25_hits],
            "vector_hits": [h.to_jsonable() for h in vec_hits],
        },
        "generation": {
            "single_pass": bool(args.single_pass),
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "grounded_draft": grounded_draft,
            "final_answer": answer,
            "citations_in_answer": citations,
        },
    }

    return answer, audit


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def run_queries(
    questions: List[str],
    *,
    args: argparse.Namespace,
) -> None:
    bm25_client, _ = create_long_client()
    vec_client, _ = create_vector_client()
    llm = load_llm()

    for question in questions:
        print("\n" + "=" * 100)
        print(f"QUESTION: {question}")
        print("=" * 100)

        start = time.time()
        answer, audit = run_one(
            question,
            bm25_client=bm25_client,
            vec_client=vec_client,
            llm=llm,
            args=args,
        )
        elapsed = time.time() - start

        print("\n" + "=" * 100)
        print("ANSWER:")
        print(answer)
        print("\n" + "=" * 100)
        print(f"Query time: {elapsed:.2f}s")

        cites = audit.get("generation", {}).get("citations_in_answer", []) or []
        print("\nCitations used in answer:", ", ".join(cites) if cites else "(none)")

        if args.save_results:
            audit["timing_s"] = elapsed
            audit["created_at_ms"] = int(time.time() * 1000)
            append_jsonl(args.save_results, audit)
            print(f"\nSaved JSONL record to: {args.save_results}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    questions: List[str]
    if args.question:
        questions = [args.question]
    else:
        # Keep default examples neutral to avoid injecting unrelated entities.
        questions = [
            "How much did Google purchase Windsurf for?",
            "How much did OpenAI purchase Windsurf for?",
        ]

    run_queries(questions, args=args)


if __name__ == "__main__":
    main()
