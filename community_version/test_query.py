import sys
import unittest
from pathlib import Path

TEST_ROOT = Path(__file__).resolve().parent
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

import citations
from common.models import RetrievalHit
from citation_repair import ensure_answer_has_citations


class ExtractCitationsTests(unittest.TestCase):
    def test_single_citation(self) -> None:
        self.assertEqual(citations.extract_citations("Answer [B1]"), ["B1"])

    def test_multiple_in_single_bracket(self) -> None:
        self.assertEqual(citations.extract_citations("See [B1, B2] for details"), ["B1", "B2"])

    def test_mixed_and_repeated_citations(self) -> None:
        text = "Combined [V3] and [B2, B2] markers with [V3] duplicates"
        self.assertEqual(citations.extract_citations(text), ["B2", "V3"])


class EnsureAnswerHasCitationsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.stub_hits = [
            RetrievalHit(
                channel="bm25_chunk",
                handle="B1",
                index="bm25",
                os_id="1",
                score=1.0,
                path="p",
                category="news",
                text="Fact about Windsurf.",
            ),
            RetrievalHit(
                channel="vector",
                handle="V1",
                index="vec",
                os_id="2",
                score=0.9,
                path="p",
                category="news",
                text="Clarification about deal structure.",
            ),
        ]

    def test_no_repair_when_citations_present(self) -> None:
        answer, citations_found, attempted = ensure_answer_has_citations(
            "What happened?",
            "Already cited [B1]",
            bm25_hits=self.stub_hits[:1],
            vec_hits=self.stub_hits[1:],
            llm=None,
            model=None,
            temperature=0.0,
            top_p=1.0,
            max_tokens=50,
        )

        self.assertEqual(answer, "Already cited [B1]")
        self.assertEqual(citations_found, ["B1"])
        self.assertFalse(attempted)

    def test_skips_repair_when_no_evidence(self) -> None:
        answer, citations_found, attempted = ensure_answer_has_citations(
            "What happened?",
            "Answer without citations",
            bm25_hits=[],
            vec_hits=[],
            llm=None,
            model=None,
            temperature=0.0,
            top_p=1.0,
            max_tokens=50,
        )

        self.assertEqual(answer, "Answer without citations")
        self.assertEqual(citations_found, [])
        self.assertFalse(attempted)

    def test_repairs_missing_citations_with_stub_llm(self) -> None:
        class StubLLM:
            def __call__(self, messages=None, temperature=None, top_p=None, max_tokens=None, **_: object):
                return "Answer with handles [B1] and [V1]"

        answer, citations_found, attempted = ensure_answer_has_citations(
            "What happened?",
            "Answer without citations",
            bm25_hits=self.stub_hits[:1],
            vec_hits=self.stub_hits[1:],
            llm=None,
            model=None,
            temperature=0.0,
            top_p=1.0,
            max_tokens=50,
            repair_llm=StubLLM(),
            llm_call=lambda llm, messages=None, **kwargs: llm(messages=messages, **kwargs),
        )

        self.assertEqual(answer, "Answer with handles [B1] and [V1]")
        self.assertEqual(citations_found, ["B1", "V1"])
        self.assertTrue(attempted)


if __name__ == "__main__":
    unittest.main()
