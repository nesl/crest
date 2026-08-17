# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for LLM candidate envelope and search-space validation."""

import math
import sys
import unittest
from pathlib import Path

from pydantic import ValidationError

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crest.optimizers.llm.schemas import (  # noqa: E402
    CandidateBatch,
    validate_candidate_batch,
)
from crest.optimizers.llm.search_space import SearchParam, SearchSpaceDescriptor  # noqa: E402


DESCRIPTOR = SearchSpaceDescriptor(
    (
        SearchParam("width", "int", low=2, high=8),
        SearchParam("dropout", "float", low=0.0, high=0.5),
        SearchParam("enabled", "categorical", choices=(True, False)),
        SearchParam("mode", "categorical", choices=("small", "large")),
    )
)
VALID = {"width": 4, "dropout": 0.25, "enabled": True, "mode": "small"}


def validate(candidates, **kwargs):
    """Validate candidate dictionaries against the shared test descriptor."""
    return validate_candidate_batch(
        CandidateBatch(candidates=candidates),
        DESCRIPTOR,
        batch_size=kwargs.pop("batch_size", len(candidates)),
        **kwargs,
    )


class CandidateValidationTests(unittest.TestCase):
    """Cover exact shape, values, and duplicate rejection behavior."""

    def test_valid_candidate_is_accepted_without_coercion(self) -> None:
        """A valid exact raw parameter dictionary passes unchanged."""
        result = validate([VALID])

        self.assertEqual(result.accepted, (VALID,))
        self.assertEqual(result.rejected, ())

    def test_candidate_batch_rejects_invalid_envelopes(self) -> None:
        """Pydantic owns malformed JSON and envelope shape failures."""
        with self.assertRaises(ValidationError):
            CandidateBatch.model_validate_json("not-json")
        with self.assertRaises(ValidationError):
            CandidateBatch.model_validate({"candidates": []})
        with self.assertRaises(ValidationError):
            CandidateBatch.model_validate({"candidates": [VALID], "extra": True})

    def test_missing_and_unknown_keys_are_rejected(self) -> None:
        """Candidates must contain exactly the descriptor's raw key set."""
        missing = dict(VALID)
        missing.pop("mode")
        unknown = {**VALID, "decoded_width": 4}

        missing_result = validate([missing])
        unknown_result = validate([unknown])

        self.assertEqual(missing_result.rejected[0].code, "missing_keys")
        self.assertIn("mode", missing_result.rejected[0].message)
        self.assertEqual(unknown_result.rejected[0].code, "unknown_keys")
        self.assertIn("decoded_width", unknown_result.rejected[0].message)

    def test_types_ranges_categories_and_finite_floats_are_enforced(self) -> None:
        """Invalid values receive stable machine-readable rejection codes."""
        cases = [
            ({**VALID, "width": True}, "type_error"),
            ({**VALID, "width": 9}, "out_of_range"),
            ({**VALID, "dropout": math.inf}, "non_finite_float"),
            ({**VALID, "enabled": 1}, "invalid_categorical"),
            ({**VALID, "mode": "medium"}, "invalid_categorical"),
        ]
        for candidate, expected_code in cases:
            with self.subTest(candidate=candidate):
                result = validate([candidate])
                self.assertEqual(result.rejected[0].code, expected_code)

    def test_completed_queued_and_in_batch_duplicates_are_distinguished(self) -> None:
        """Duplicate sources are reported separately before enqueueing."""
        alternate = {**VALID, "width": 6}
        third = {**VALID, "width": 8}
        result = validate(
            [VALID, alternate, third, third],
            completed_candidates=[VALID],
            queued_candidates=[alternate],
        )

        self.assertEqual(result.accepted, (third,))
        self.assertEqual(
            [rejection.code for rejection in result.rejected],
            ["duplicate_completed", "duplicate_queued", "duplicate_batch"],
        )

    def test_candidates_beyond_requested_batch_size_are_rejected(self) -> None:
        """A provider cannot silently enlarge the requested evaluation batch."""
        result = validate([VALID, {**VALID, "width": 6}], batch_size=1)

        self.assertEqual(result.accepted, (VALID,))
        self.assertEqual(result.rejected[0].code, "batch_size_exceeded")


if __name__ == "__main__":
    unittest.main()
