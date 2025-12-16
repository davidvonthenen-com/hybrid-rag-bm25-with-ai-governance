import sys
import unittest
from pathlib import Path

TEST_ROOT = Path(__file__).resolve().parent
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

import citations


class ExtractCitationsTests(unittest.TestCase):
    def test_single_citation(self) -> None:
        self.assertEqual(citations.extract_citations("Answer [B1]"), ["B1"])

    def test_multiple_in_single_bracket(self) -> None:
        self.assertEqual(citations.extract_citations("See [B1, B2] for details"), ["B1", "B2"])

    def test_mixed_and_repeated_citations(self) -> None:
        text = "Combined [V3] and [B2, B2] markers with [V3] duplicates"
        self.assertEqual(citations.extract_citations(text), ["B2", "V3"])


if __name__ == "__main__":
    unittest.main()
