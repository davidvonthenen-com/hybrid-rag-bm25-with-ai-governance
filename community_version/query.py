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
from typing import Any, Dict, List, Optional, Sequence, Tuple

from common.bm25 import bm25_retrieve_chunks, bm25_retrieve_doc_anchors
from common.embeddings import vector_retrieve_chunks
from common.llm import (
    build_grounding_prompt,
    build_refine_prompt,
    build_single_pass_prompt,
    build_vector_only_prompt,
    call_llm_chat,
    load_llm,
)
from common.logging import get_logger
from common.models import RetrievalHit
from common.named_entity import extract_entities
from common.opensearch_client import create_long_client, create_vector_client

LOGGER = get_logger(__name__)

_CITATION_RE = re.compile(r"\[(B\d+|V\d+)\]")


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


def _extract_citations(answer: str) -> List[str]:
    return sorted(set(m.group(1) for m in _CITATION_RE.finditer(answer)))


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
    bm25_full_index = args.bm25_full_index or getattr(bm25_client.settings, "opensearch_full_index", None)
    # TODO: need to deal with HOT INDEX or getattr(bm25_client.settings, "opensearch_hot_index", None)
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

    # 2A) bm25 LONG INDEX chunks
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

    # 2B) bm25 HOT INDEX which contains users' personal data
    # TODO: implement later. if we implement this, we should obtain the bm25 LONG INDEX, bm25 HOT INDEX, and vector chunks in parallel.

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

        # ground the initial draft in BM25 (or vector-only) evidence
        grounded_draft = call_llm_chat(llm, messages=msgs_a, model=model, temperature=temperature, top_p=top_p, max_tokens=max_tokens)

        # refine with vector context if available
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
