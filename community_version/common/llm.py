"""LLM loading and hybrid RAG orchestration utilities."""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from .bm25 import rerank_hits_with_bm25
from .config import Settings, load_settings
from .embeddings import embed_question, normalize_vector_hits
from .logging import get_logger
from .models import RetrievalHit
from .named_entity import normalize_entities, post_ner
from .opensearch_client import (
    build_query_external_ranking,
    build_query_opensearch_ranking,
    combine_hits,
    create_hot_client,
    create_long_client,
    create_vector_client,
    knn_search_one,
    rank_hits,
    render_matches,
    render_observability_summary,
    search_one,
)

LOGGER = get_logger(__name__)


def _load_fireworks_client(settings: Settings) -> OpenAI:
    """Create an OpenAI-compatible client configured for Fireworks AI.

    Args:
        settings: Runtime settings containing Fireworks configuration.
    Returns:
        An OpenAI client pointed at the Fireworks AI endpoint.
    Raises:
        RuntimeError: If the FIREWORKS_API_KEY environment variable is missing.
    """

    api_key = os.getenv("FIREWORKS_API_KEY")
    if not api_key:
        raise RuntimeError("FIREWORKS_API_KEY must be set when using Fireworks AI.")

    LOGGER.info("Connecting to Fireworks AI at %s", settings.fireworks_base_url)
    client = OpenAI(
        base_url=settings.fireworks_base_url,
        api_key=api_key,
    )
    setattr(client, "default_model", settings.fireworks_model)
    setattr(client, "use_fireworks_completions", True)
    return client


@lru_cache(maxsize=1)
def load_llm(settings: Optional[Settings] = None) -> OpenAI:
    """Construct and cache an OpenAI-compatible client for the LLM server."""

    if settings is None:
        settings = load_settings()

    if settings.fireworksai:
        return _load_fireworks_client(settings)

    LOGGER.info("Connecting to LLM server at %s", settings.llm_server_url)
    client = OpenAI(
        base_url=settings.llm_server_url,
        api_key=settings.llm_server_api_key,
    )
    setattr(client, "default_model", settings.llm_server_model)
    return client


def _save_results(path: str, payload: Dict[str, Any]) -> None:
    """Append a JSON line to ``path`` for auditability."""

    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _build_context(hits: List[Dict[str, Any]], max_chars_per_doc: int = 0) -> str:
    """Construct a text block to feed into the LLM from OpenSearch hits."""

    if not hits:
        return ""

    formatted: List[str] = []
    for hit in hits:
        source = hit.get("_source", {}) or {}
        filepath = source.get("filepath", "<unknown>")
        store_label = hit.get("_store_label", "?")
        content = (source.get("content") or "").strip()
        if max_chars_per_doc > 0:
            content = content[:max_chars_per_doc]
        formatted.append(f"---\nStore: {store_label}\nDoc: {filepath}\n{content}\n")

    return "\n".join(formatted)


def _hit_key(hit: Dict[str, Any]) -> str:
    """Generate a stable key to deduplicate hits across modalities."""

    source = hit.get("_source", {}) or {}
    path = source.get("filepath") or source.get("path") or ""
    chunk = source.get("chunk_index")
    if chunk is None:
        return str(path)
    try:
        return f"{path}::chunk-{int(chunk):03d}"
    except Exception:
        return f"{path}::{chunk}"


def _merge_hybrid_ranked(
    bm25_hits: List[Dict[str, Any]],
    vector_hits: List[Dict[str, Any]],
    *,
    top_k: int,
    vector_fraction: float,
) -> List[Dict[str, Any]]:
    """Merge BM25 and vector hits while preserving per-list order.

    The function enforces a minimum share of results from each modality (when
    available) so the final list truly represents a hybrid of lexical and
    semantic signals. A small helper handles duplicate suppression to keep the
    main control flow easy to read.
    """

    if top_k <= 0:
        return []

    vector_fraction = max(0.0, min(1.0, float(vector_fraction)))
    vec_budget = max(0, min(top_k, int(round(top_k * vector_fraction))))
    bm_budget = top_k - vec_budget

    seen: set[str] = set()
    merged: List[Dict[str, Any]] = []

    def add_unique_hit(hit: Dict[str, Any]) -> None:
        key = _hit_key(hit)
        if key in seen:
            return
        seen.add(key)
        merged.append(hit)

    for hit in bm25_hits[:bm_budget] + vector_hits[:vec_budget]:
        add_unique_hit(hit)

    if len(merged) < top_k:
        for hit in bm25_hits[bm_budget:] + vector_hits[vec_budget:]:
            if len(merged) >= top_k:
                break
            add_unique_hit(hit)

    return merged


def generate_answer(
    llm: Any,
    question: str,
    context: str,
    *,
    observability: bool = False,
    max_tokens: int = 32768,
    temperature: float = 0.2,
    top_p: float = 0.8,
) -> str:
    """Run a chat completion against the LLM using the provided context."""

    if not context.strip():
        return "No supporting documents found."

    system_msg = "Answer using ONLY the provided context below."
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\n"

    if observability:
        LOGGER.info("LLM prompt context length=%d chars", len(context))

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_prompt},
    ]
    return call_llm_chat(
        llm,
        messages=messages,
        model=getattr(llm, "default_model", None),
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )


def _build_bm25_query(
    question: str,
    entities: List[str],
    *,
    external_ranker: bool,
    observability: bool,
) -> Dict[str, Any]:
    """Choose the appropriate BM25 query strategy based on configuration."""

    if external_ranker:
        if observability:
            print("\n\nUsing EXTERNAL ranking with BM25 re-ranking after retrieval.\n\n")
        return build_query_external_ranking(question, entities)

    if observability:
        print("\n\nUsing INTERNAL OpenSearch ranking only.\n\n")
    return build_query_opensearch_ranking(question, entities)


def _search_stores(
    query: Dict[str, Any],
    question_vector: List[float],
    *,
    top_k: int,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Settings]:
    """Execute LONG, HOT, and VECTOR searches in parallel."""

    long_client, long_index = create_long_client()
    hot_client, hot_index = create_hot_client()
    vector_client, vector_index = create_vector_client()

    vec_k = max(1, min(int(top_k), 200))
    vec_candidates = max(vector_client.settings.search_size, vec_k)

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_long = executor.submit(
            search_one, "LONG", long_client, long_index, query, long_client.settings
        )
        future_hot = executor.submit(
            search_one, "HOT", hot_client, hot_index, query, hot_client.settings
        )
        future_vec = executor.submit(
            knn_search_one,
            "VECTOR",
            vector_client,
            vector_index,
            question_vector,
            k=vec_k,
            num_candidates=vec_candidates,
        )

        res_long = future_long.result()
        res_hot = future_hot.result()
        res_vec = future_vec.result()

    return res_long, res_hot, res_vec, vector_client.settings


def ask(
    llm: Llama,
    question: str,
    *,
    observability: bool = True,
    external_ranker: bool = True,
    top_k: int = 10,
    save_path: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Answer a question using hybrid BM25 + vector retrieval and an LLM."""

    settings = load_settings()
    ner_payload = post_ner(question)
    entities = normalize_entities(ner_payload)

    query = _build_bm25_query(question, entities, external_ranker=external_ranker, observability=observability)

    if observability:
        if entities:
            print(f"[NER] entities: {entities}")
            print("\n[QUERY] dis_max (entity path):")
        else:
            print("[NER] No entities detected; using full-question match only.")
            print("\n[QUERY] dis_max (no-entity path):")
        print(json.dumps(query, indent=2))

    question_vector = embed_question(question)
    res_long, res_hot, res_vec, vector_settings = _search_stores(
        query, question_vector, top_k=top_k
    )

    if external_ranker:
        keep_long, keep_hot, bm25_combined = rerank_hits_with_bm25(
            question, res_long, res_hot, top_k=top_k
        )
    else:
        alpha = settings.ranking_alpha
        keep_long = rank_hits(res_long, alpha=alpha)
        keep_hot = rank_hits(res_hot, alpha=alpha)
        bm25_combined = combine_hits(keep_long, keep_hot, top_k=top_k)

    vector_hits = normalize_vector_hits(res_vec)

    vector_alpha = vector_settings.ranking_alpha
    hybrid_hits = _merge_hybrid_ranked(
        bm25_combined,
        vector_hits,
        top_k=top_k,
        vector_fraction=vector_alpha,
    )

    if observability:
        print(render_observability_summary(res_long))
        print(render_observability_summary(res_hot))
        print(render_observability_summary(res_vec))

        print(f"\n[RESULTS] LONG kept={len(keep_long)} of {len(res_long.get('hits',{}).get('hits',[]))}")
        print(f"[RESULTS] HOT  kept={len(keep_hot)} of {len(res_hot.get('hits',{}).get('hits',[]))}")
        print(f"[RESULTS] VEC  kept={len(vector_hits)} of {len(res_vec.get('hits',{}).get('hits',[]))}")

        print("\n[HYBRID MATCHES]")
        print(render_matches(hybrid_hits))

    if save_path:
        payload = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "question": question,
            "entities": entities,
            "alpha": settings.ranking_alpha,
            "size": settings.search_size,
            "preference": settings.search_preference,
            "vector": {
                "index": res_vec.get("_index_used"),
                "total": res_vec.get("hits", {}).get("total", {}).get("value", 0),
                "error": res_vec.get("_error"),
                "kept_filepaths": [h.get("_source", {}).get("filepath") for h in vector_hits[:top_k]],
            },
            "long": {
                "index": res_long.get("_index_used"),
                "total": res_long.get("hits", {}).get("total", {}).get("value", 0),
                "error": res_long.get("_error"),
                "kept_filepaths": [h.get("_source", {}).get("filepath") for h in keep_long],
            },
            "hot": {
                "index": res_hot.get("_index_used"),
                "total": res_hot.get("hits", {}).get("total", {}).get("value", 0),
                "error": res_hot.get("_error"),
                "kept_filepaths": [h.get("_source", {}).get("filepath") for h in keep_hot],
            },
            "hybrid_filepaths": [h.get("_source", {}).get("filepath") for h in hybrid_hits],
        }
        _save_results(save_path, payload)

    context_block = _build_context(hybrid_hits)
    answer = generate_answer(llm, question, context_block, observability=observability)

    return answer, hybrid_hits


def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    """Fallback formatting for non-chat completion interfaces."""
    parts: List[str] = []
    for m in messages:
        role = (m.get("role") or "user").upper()
        content = m.get("content") or ""
        parts.append(f"{role}:\n{content}".strip())
    return "\n\n".join(parts).strip()


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

    if getattr(llm, "use_fireworks_completions", False):
        completions = getattr(llm, "completions", None)
        create = getattr(completions, "create", None) if completions is not None else None
        if callable(create):
            prompt = _messages_to_prompt(messages)
            resolved_model = model or getattr(llm, "default_model", None)
            kwargs: Dict[str, Any] = {
                "prompt": prompt,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
            }
            if resolved_model:
                kwargs["model"] = resolved_model
            resp = create(**kwargs)
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
            resolved_model = model or getattr(llm, "default_model", None)
            if resolved_model:
                kwargs["model"] = resolved_model
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


def _format_hits(hits: List[RetrievalHit], *, title: str) -> str:
    """Render retrieval hits into a tag-delimited context block.

    Each hit is wrapped in an explicit open/close tag so multi-line chunks
    remain unambiguous:

        [B1]
        ...text...
        [/B1]

    This makes it easier for smaller instruction models to reliably reference
    chunks and for downstream parsing/auditing to remain robust.
    """

    if not hits:
        return f"=== {title} ===\n(none)"

    blocks: List[str] = [f"=== {title} ==="]
    for h in hits:
        open_tag = f"[{h.handle}]"
        close_tag = f"[/{h.handle}]"
        # Keep metadata minimal to reduce the chance the model quotes it.
        meta: List[str] = []
        if h.path:
            meta.append(f"path={h.path}")
        if h.chunk_index is not None and h.chunk_count is not None:
            meta.append(f"chunk={h.chunk_index}/{h.chunk_count}")
        if h.category:
            meta.append(f"category={h.category}")
        meta_line = f"META: {', '.join(meta)}" if meta else ""

        text = (h.text or "").strip()
        if meta_line:
            blocks.append(f"{open_tag}\n{meta_line}\n{text}\n{close_tag}".strip())
        else:
            blocks.append(f"{open_tag}\n{text}\n{close_tag}".strip())

    return "\n\n".join(blocks).strip()


def _allowed_citation_tags(hits: List[RetrievalHit]) -> List[str]:
    """Return citation tags (opening tags only) for a set of hits."""

    out: List[str] = []
    for h in hits:
        handle = (h.handle or "").strip()
        if not handle:
            continue
        out.append(f"[{handle}]")
    return out


def build_grounding_prompt(question: str, bm25_hits: List[RetrievalHit]) -> List[Dict[str, str]]:
    # print("***** Building grounding prompt...")
    context = _format_hits(bm25_hits, title="BM25 Grounding Evidence (authoritative facts)")
    allowed = " ".join(_allowed_citation_tags(bm25_hits)) if bm25_hits else "(none)"
    system = (
        "You are a grounded QA assistant. Answer using ONLY the BM25 Grounding Evidence.\n"
        "Evidence chunks are delimited as [B#] ... [/B#].\n"
        "\n"
        "CITATION RULES (mandatory):\n"
        f"- Allowed citation tags: {allowed}\n"
        "- After EVERY sentence that contains a factual claim, append one or more citation tags.\n"
        "- Only cite BM25 context as [B#]. Never use closing tags like [/B1] in your answer.\n"
        # "- Use the opening tag only (e.g., [B1]); never use closing tags like [/B1] in your answer.\n"
        # "- If multiple citations are needed, write them back-to-back like [B1][B2]. Do NOT write [B1, B2].\n"
        "- Never invent citation numbers or use tags not listed above.\n"
        "\n"
        "If the evidence does not support the answer, write exactly: I don't know based on the provided evidence.\n"
        "Do not quote the evidence headers/metadata.\n"
        "If evidence conflicts, disclose the conflict.\n"
        "Output ONLY the answer text."
    )
    user = f"QUESTION:\n{question}\n\nGROUNDING_EVIDENCE:\n{context}\n"

    prompt_details = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    print("=====================================================")
    print(_messages_to_prompt(prompt_details))
    print("=====================================================")

    return prompt_details


def build_vector_only_prompt(question: str, vec_hits: List[RetrievalHit]) -> List[Dict[str, str]]:
    # print("***** Building vector-only prompt...")
    context = _format_hits(vec_hits, title="Vector Evidence (semantic fallback)")
    allowed = " ".join(_allowed_citation_tags(vec_hits)) if vec_hits else "(none)"
    system = (
        "Answer using ONLY the Vector Evidence.\n"
        "Evidence chunks are delimited as [V#] ... [/V#].\n"
        "\n"
        "CITATION RULES (mandatory):\n"
        f"- Allowed citation tags: {allowed}\n"
        "- After EVERY sentence that contains a factual claim, append one or more citation tags.\n"
        "- Only cite vector context as [V#]. Never use closing tags like [/V1] in your answer.\n"
        # "- Use the opening tag only (e.g., [V1]); never use closing tags like [/V1] in your answer.\n"
        # "- If multiple citations are needed, write them back-to-back like [V1][V2]. Do NOT write [V1, V2].\n"
        "- Never invent citation numbers or use tags not listed above.\n"
        "\n"
        "If evidence does not support the answer, write exactly: I don't know based on the provided evidence.\n"
        "Do not quote the evidence headers/metadata.\n"
        "Output ONLY the answer text."
    )
    user = f"QUESTION:\n{question}\n\nEVIDENCE:\n{context}\n"
    prompt_details = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    print("=====================================================")
    print(_messages_to_prompt(prompt_details))
    print("=====================================================")

    return prompt_details


def build_refine_prompt(question: str, grounded_draft: str, vec_hits: List[RetrievalHit]) -> List[Dict[str, str]]:
    # print("***** Building refine prompt...")
    vec_context = _format_hits(vec_hits, title="Vector Semantic Context (phrasing/terminology support)")
    allowed_v = " ".join(_allowed_citation_tags(vec_hits)) if vec_hits else "(none)"
    system = (
        "Rewrite the grounded draft for clarity and readability.\n"
        "\n"
        "CRITICAL RULES:\n"
        "- Do NOT add any new factual claims beyond what appears in the grounded draft.\n"
        "- Preserve all existing [B#] citations EXACTLY (do not delete, renumber, merge, or move them).\n"
        "- You MAY add brief non-factual clarifications (definitions, paraphrases) supported by the vector context.\n"
        "\n"
        "VECTOR CITATIONS (optional):\n"
        f"- Allowed vector citation tags: {allowed_v}\n"
        "- Only cite vector context as [V#]. Never use closing tags like [/V1] in your answer.\n"
        # "- If multiple vector citations are needed, write [V1][V2] (do NOT write [V1, V2]).\n"
        "\n"
        "Output ONLY the rewritten answer text.\n"
        "If there is conflicting evidence, disclose the conflict and data.\n"
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
    # print("***** Building single-pass prompt...")
    bm25_context = _format_hits(bm25_hits, title="BM25 Grounding Evidence (authoritative facts)")
    vec_context = _format_hits(vec_hits, title="Vector Semantic Context (phrasing/terminology support)")
    allowed_b = " ".join(_allowed_citation_tags(bm25_hits)) if bm25_hits else "(none)"
    allowed_v = " ".join(_allowed_citation_tags(vec_hits)) if vec_hits else "(none)"
    system = (
        "Answer using the provided contexts.\n"
        "Separation of concerns:\n"
        "- Use BM25 Grounding Evidence for factual claims.\n"
        "- Use Vector Semantic Context only for wording/terminology, not new facts.\n"
        "\n"
        "BM25 CITATIONS (mandatory for facts):\n"
        f"- Allowed BM25 citation tags: {allowed_b}\n"
        "- Only cite BM25 context as [B#]. Never use closing tags like [/B1] in your answer.\n"
        # "- Only cite BM25 context as [B#], and only for definitions/paraphrases.\n"
        # "- If multiple BM25 citations are needed, write [B1][B2] (do NOT write [B1, B2]).\n"
        "- Never use closing tags like [/B1] in your answer.\n"
        "\n"
        "VECTOR CITATIONS (optional, non-factual clarifications only):\n"
        f"- Allowed vector citation tags: {allowed_v}\n"
        "- Only cite vector context as [V#]. Never use closing tags like [/V1] in your answer.\n"
        # "- Only cite vector context as [V#], and only for definitions/paraphrases.\n"
        # "- If multiple vector citations are needed, write [V1][V2] (do NOT write [V1, V2]).\n"
        "\n"
        "If BM25 evidence does not support the answer, write exactly: I don't know based on the provided evidence.\n"
        "Output ONLY the answer text."
    )
    user = f"QUESTION:\n{question}\n\n{bm25_context}\n\n{vec_context}\n"
    prompt_details = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    print("=====================================================")
    print(_messages_to_prompt(prompt_details))
    print("=====================================================")

    return prompt_details


def build_citation_repair_prompt(
    question: str,
    *,
    draft: str,
    bm25_hits: List[RetrievalHit],
    vec_hits: List[RetrievalHit],
    stage: str = "final",
) -> List[Dict[str, str]]:
    """Build a prompt that makes the LLM add/repair inline citations.

    This is intentionally strict and example-driven to increase compliance on
    smaller instruction models.
    """

    bm25_context = _format_hits(bm25_hits, title="BM25 Grounding Evidence (authoritative facts)")
    vec_context = _format_hits(vec_hits, title="Vector Semantic Context (phrasing/terminology support)")

    allowed_b = " ".join(_allowed_citation_tags(bm25_hits)) if bm25_hits else "(none)"
    allowed_v = " ".join(_allowed_citation_tags(vec_hits)) if vec_hits else "(none)"

    if bm25_hits and not vec_hits:
        modality_rule = (
            "Use ONLY BM25 citations [B#] for factual claims. Do NOT use [V#]."
        )
    elif vec_hits and not bm25_hits:
        modality_rule = (
            "Use ONLY vector citations [V#] for factual claims (no BM25 evidence is available)."
        )
    else:
        modality_rule = (
            "Prefer BM25 citations [B#] for factual claims. Use [V#] only for non-factual clarifications."
        )

    system = (
        "You are a citation-repair tool.\n"
        f"Stage: {stage}\n"
        "\n"
        "TASK:\n"
        "Rewrite the DRAFT ANSWER so that it includes REQUIRED inline citations.\n"
        "Evidence chunks are delimited as [B#] ... [/B#] and [V#] ... [/V#].\n"
        "Cite using ONLY the OPENING tags: [B1] or [V2].\n"
        "\n"
        "CITATION FORMAT (MANDATORY):\n"
        f"- Allowed BM25 tags: {allowed_b}\n"
        f"- Allowed vector tags: {allowed_v}\n"
        # "- After EVERY sentence that contains a factual claim, append one or more citation tags.\n"
        # "- If multiple citations are needed, write them back-to-back: [B1][B2] (NOT [B1, B2]).\n"
        "- Never invent citation numbers or tags not listed above.\n"
        "- Never output closing tags like [/B1] or [/V1] in the answer.\n"
        "\n"
        "CONTENT RULES:\n"
        f"- {modality_rule}\n"
        "- Do NOT add new facts beyond the evidence.\n"
        "- Keep wording as close as possible to the draft; only add citations where needed.\n"
        "- If a claim in the draft is not supported by the evidence, replace it with: I don't know based on the provided evidence.\n"
        "\n"
        "Output ONLY the revised answer text."
    )

    user = (
        f"QUESTION:\n{question}\n\n"
        f"DRAFT_ANSWER:\n{draft}\n\n"
        f"EVIDENCE_BM25:\n{bm25_context}\n\n"
        f"EVIDENCE_VECTOR:\n{vec_context}\n"
    )

    prompt_details = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    print("=====================================================")
    print(_messages_to_prompt(prompt_details))
    print("=====================================================")

    return prompt_details


__all__ = [
    "load_llm",
    "generate_answer",
    "ask",
    "call_llm_chat",
    "build_grounding_prompt",
    "build_vector_only_prompt",
    "build_refine_prompt",
    "build_single_pass_prompt",
    "build_citation_repair_prompt",
]
