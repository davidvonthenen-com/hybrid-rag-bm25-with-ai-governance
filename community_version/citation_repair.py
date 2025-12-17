"""Citation repair utilities.

These helpers attempt to ensure final answers include citation handles
when the initial LLM output omits them.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

from citations import extract_citations
from common.models import RetrievalHit


def ensure_answer_has_citations(
    question: str,
    answer: str,
    *,
    bm25_hits: List[RetrievalHit],
    vec_hits: List[RetrievalHit],
    llm: Any,
    model: Optional[str],
    temperature: float,
    top_p: float,
    max_tokens: int,
    repair_llm: Optional[Any] = None,
    llm_call: Optional[Callable[..., str]] = None,
) -> Tuple[str, List[str], bool]:
    """Ensure the final answer carries citation handles.

    Args:
        question: Original user question for context.
        answer: LLM-produced answer (may be missing citations).
        bm25_hits: Evidence hits with [B#] handles.
        vec_hits: Evidence hits with [V#] handles.
        llm: Primary LLM client for generation.
        model: Optional model override passed to the LLM client.
        temperature: Temperature used for generation.
        top_p: Top-p value used for generation.
        max_tokens: Max tokens used for generation.
        repair_llm: Optional override used only for tests to stub LLM calls.
        llm_call: Optional call helper used to avoid importing heavy LLM modules in tests.

    Returns:
        A tuple of (answer_with_citations, citations, repair_attempted).
    """

    citations = extract_citations(answer)
    if citations or (not bm25_hits and not vec_hits):
        return answer, citations, False

    repair_client = repair_llm or llm
    evidence_blocks: List[str] = []
    for hit in bm25_hits + vec_hits:
        snippet = hit.text.strip().replace("\n", " ")
        evidence_blocks.append(f"[{hit.handle}] {snippet}")
    evidence_text = "\n".join(evidence_blocks)

    system = (
        "Add citation handles to the provided answer without changing its meaning.\n"
        "Rules:\n"
        "- Use the existing evidence handles (e.g., [B#], [V#]) when citing.\n"
        "- Preserve the answer wording as much as possible while adding inline citations.\n"
        "- Return only the updated answer with citations.\n"
    )

    user = (
        f"QUESTION:\n{question}\n\n"
        f"ANSWER_WITHOUT_CITATIONS:\n{answer}\n\n"
        f"EVIDENCE_SNIPPETS:\n{evidence_text}\n"
    )

    llm_invoker = llm_call
    if llm_invoker is None:
        from common.llm import call_llm_chat as default_call_llm_chat

        llm_invoker = default_call_llm_chat

    repaired = llm_invoker(
        repair_client,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    repaired_citations = extract_citations(repaired)
    if repaired_citations:
        return repaired, repaired_citations, True

    return answer, citations, True
