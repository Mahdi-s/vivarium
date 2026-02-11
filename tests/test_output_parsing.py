from __future__ import annotations

import os
import sys
import unittest


# Ensure local imports work when running `python -m unittest` from repo root.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)


from aam.output_parsing import OutputParsingConfig, OutputQualityLabel, classify_output  # noqa: E402


class OutputParsingTests(unittest.TestCase):
    def test_classifies_empty_after_unicode_and_zero_width_cleanup(self) -> None:
        res = classify_output(raw_text=" \n\t\u200b\u200d  \n", cfg=OutputParsingConfig())
        self.assertEqual(res.label, OutputQualityLabel.EMPTY)
        self.assertGreater(res.metadata["raw_length"], 0)
        self.assertEqual(res.metadata["normalized_length"], 0)

    def test_classifies_degenerate_repetition_by_consecutive_tokens(self) -> None:
        res = classify_output(raw_text=("the " * 12).strip(), cfg=OutputParsingConfig())
        self.assertEqual(res.label, OutputQualityLabel.DEGENERATE_REPETITION)
        self.assertGreaterEqual(res.metadata["max_consecutive_identical_tokens"], 11)
        self.assertLess(res.metadata["repetition_score"], 0.3)

    def test_classifies_prompt_leakage_with_fuzzy_ngram_coverage(self) -> None:
        system_prompt = (
            "Output instructions:\n"
            "Respond in $functional form. $functional form is a plain english description of the function "
            "without using mathematical notations. do not output any other characters except plain text.\n"
        )
        user_prompt = "What is the capital of France?"
        raw = (
            "tag, in $functional form. $functional form is a plain english\n"
            "description of the function without using mathematical\n"
            "notations. do not output any other characters except\n"
            "plain text. ... if the question is in English or another\n"
            "language, answer that question in English or the original\n"
        )
        res = classify_output(
            raw_text=raw,
            cfg=OutputParsingConfig(),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        self.assertEqual(res.label, OutputQualityLabel.PROMPT_LEAKAGE)
        self.assertGreaterEqual(res.metadata["prompt_similarity_score"], 0.40)

    def test_classifies_structural_garbage_when_mostly_special_tokens_and_markup(self) -> None:
        raw = (
            "<|begin_of_text|><|start_header_id|>assistant<|end_header_id|>\n\n"
            "<p>Hello</p>\n\n"
            "<|eot_id|>"
        )
        res = classify_output(raw_text=raw, cfg=OutputParsingConfig())
        self.assertEqual(res.label, OutputQualityLabel.STRUCTURAL_GARBAGE)
        self.assertIn("begin_of_text", res.metadata["special_tokens_found"])
        self.assertGreater(res.metadata["structural_stripped_ratio"], 0.50)

    def test_classifies_incoherent_by_heuristics_when_logprobs_unavailable(self) -> None:
        raw = (
            "florptang glibberwock nargle sporgle blart quzzled snorfle "
            "plimdor frastle crundit blenchor smarvle grintor plovery "
            "snargit frimble drozzit clarnok brastin glonver trazzle "
            "splonkit dravven plortin grazzle"
        )
        res = classify_output(raw_text=raw, cfg=OutputParsingConfig())
        self.assertEqual(res.label, OutputQualityLabel.INCOHERENT)
        self.assertFalse(res.metadata["logprobs_used"])
        self.assertFalse(res.metadata["coherence_heuristic"]["skipped"])

    def test_classifies_partial_valid_when_answer_present_but_noise_high(self) -> None:
        raw = "The capital of France is Paris.\n\n<|endoftext|><|endoftext|>"
        res = classify_output(raw_text=raw, cfg=OutputParsingConfig(), expected_answer_texts=["Paris"])
        self.assertEqual(res.label, OutputQualityLabel.PARTIAL_VALID)
        self.assertTrue(res.metadata["valid_answer_found"])
        self.assertGreaterEqual(res.metadata["structural_stripped_ratio"], 0.30)
        self.assertLessEqual(res.metadata["structural_stripped_ratio"], 0.50)

    def test_classifies_valid_clean_output(self) -> None:
        res = classify_output(raw_text="Paris.", cfg=OutputParsingConfig(), expected_answer_texts=["Paris"])
        self.assertEqual(res.label, OutputQualityLabel.VALID)
        self.assertFalse(res.metadata["encoding_issues_detected"])

    def test_detects_mojibake_as_encoding_issue(self) -> None:
        # Two common mojibake markers to clear the heuristic threshold.
        raw = "It isnâ€™t working; it isnâ€™t correct."
        res = classify_output(raw_text=raw, cfg=OutputParsingConfig())
        self.assertTrue(res.metadata["encoding_issues_detected"])


if __name__ == "__main__":
    unittest.main()

