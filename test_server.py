#!/usr/bin/env python3
"""Deterministic tests for the next-token wheel's API transformation."""

import math
import unittest

from server import WheelError, candidates_from_response, display_token


class CandidateTests(unittest.TestCase):
    def test_probabilities_are_renormalized_across_displayed_candidates(self) -> None:
        payload = {
            "response": " ignored",
            "logprobs": [
                {
                    "token": " ignored",
                    "logprob": -0.2,
                    "top_logprobs": [
                        {"token": " one", "logprob": math.log(0.5)},
                        {"token": " two", "logprob": math.log(0.3)},
                        {"token": " three", "logprob": math.log(0.1)},
                    ],
                }
            ],
        }
        candidates = candidates_from_response(payload, 3)
        probabilities = [item["p_displayed"] for item in candidates]

        self.assertAlmostEqual(sum(probabilities), 1.0)
        self.assertAlmostEqual(probabilities[0], 5 / 9)
        self.assertAlmostEqual(probabilities[1], 3 / 9)
        self.assertAlmostEqual(probabilities[2], 1 / 9)
        self.assertEqual([item["raw"] for item in candidates], [" one", " two", " three"])

    def test_display_labels_do_not_change_raw_whitespace(self) -> None:
        self.assertEqual(display_token(" hello\n"), "␠hello↵")
        self.assertEqual(display_token("\t"), "⇥")
        self.assertEqual(display_token(""), "[END]")

    def test_missing_logprobs_is_a_clear_error(self) -> None:
        with self.assertRaisesRegex(WheelError, "did not return log probabilities"):
            candidates_from_response({}, 5)

    def test_replacement_character_fails_closed(self) -> None:
        payload = {
            "logprobs": [
                {
                    "top_logprobs": [
                        {"token": " good", "logprob": -0.1},
                        {"token": "\ufffd", "logprob": -1.2},
                    ]
                }
            ]
        }
        with self.assertRaisesRegex(WheelError, "decoded safely"):
            candidates_from_response(payload, 2)


if __name__ == "__main__":
    unittest.main()
