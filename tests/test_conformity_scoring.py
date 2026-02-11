from __future__ import annotations

import os
import sys
import unittest


# Ensure local imports work when running `python -m unittest` from repo root.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)


from aam.experiments.olmo_conformity.scoring import (  # noqa: E402
    evaluate_correctness,
    normalize_text_for_matching,
    parse_answer_text,
)


class ConformityScoringTests(unittest.TestCase):
    def test_normalize_collapses_dotted_abbreviations(self) -> None:
        self.assertEqual(normalize_text_for_matching("Washington, D.C."), "washington dc")
        self.assertEqual(normalize_text_for_matching("U.S."), "us")

    def test_correctness_matches_without_dc_punctuation(self) -> None:
        self.assertTrue(
            evaluate_correctness(
                parsed_answer_text="The capital is Washington DC.",
                ground_truth_text="Washington, D.C.",
            )
        )

    def test_correctness_normalizes_unicode_subscripts(self) -> None:
        self.assertTrue(
            evaluate_correctness(
                parsed_answer_text="The chemical formula is **H₂O**.",
                ground_truth_text="H2O",
            )
        )

    def test_parse_strips_post_think_answer_region(self) -> None:
        raw = "I think it might be Paris.\n</think>\n\nThe capital of France is Lyon."
        parsed = parse_answer_text(raw)
        self.assertTrue(parsed.startswith("The capital of France is Lyon"))
        self.assertFalse(evaluate_correctness(parsed_answer_text=parsed, ground_truth_text="Paris"))

    def test_parse_truncates_hallucinated_conversation_tail(self) -> None:
        raw = "Paris.\n\nUSER:\nWhat is the significance of the number"
        parsed = parse_answer_text(raw)
        self.assertEqual(parsed, "Paris.")


if __name__ == "__main__":
    unittest.main()

