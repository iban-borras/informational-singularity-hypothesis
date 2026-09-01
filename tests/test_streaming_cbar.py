"""Equivalence tests for the exact packed-v33 streaming cbar counter."""

from __future__ import annotations

import gzip
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from level1_nesting_tree import build_tree_from_structural, collect_all_metrics_iterative
from hsi_v2_d0154_f27_cbar_streaming import (
    AGGREGATE_TOLERANCE,
    E_EXPECTED,
    PREREG_SHA256,
    SEGMENT_TOLERANCE,
    SYNTHETIC_FIXTURES,
    classify_outcome,
    validate_equivalence_payload,
)
from v2.signature.streaming_cbar import (
    DyckMeanChildrenCounter,
    SegmentSpec,
    scan_v33_gzip_segments,
)


def pack_structural(structural: str) -> bytes:
    code = {"0": 0, "1": 1, "(": 2, ")": 3}
    output = bytearray()
    for start in range(0, len(structural), 4):
        values = [code[value] for value in structural[start : start + 4]]
        values.extend([0] * (4 - len(values)))
        output.append((values[0] << 6) | (values[1] << 4) | (values[2] << 2) | values[3])
    return bytes(output)


def canonical_counts(structural: str) -> dict[str, int]:
    root, metadata = build_tree_from_structural(
        structural, max_nodes=1_000_000, timeout_seconds=60
    )
    _, _, _, children_counts, complete = collect_all_metrics_iterative(
        root, timeout_seconds=60
    )
    assert complete
    return {
        "total_nodes": int(metadata["nodes_created"]),
        "total_edges": int(metadata["nodes_created"]) - 1,
        "total_branching_nodes": len(children_counts),
    }


@pytest.mark.parametrize(
    "structural",
    ["()", "()()", "(()())", "))()", "((()", "0(1)0", "1((0)(1(0)))1"],
)
def test_counter_matches_canonical_parser(structural: str) -> None:
    counter = DyckMeanChildrenCounter()
    counter.scan(pack_structural(structural), 0, len(structural))
    observed = counter.result()
    expected = canonical_counts(structural)
    assert {key: observed[key] for key in expected} == expected


def test_random_structures_match_canonical_parser() -> None:
    rng = random.Random(170101)
    alphabet = "01()"
    for length in range(1, 80):
        structural = "".join(rng.choice(alphabet) for _ in range(length))
        counter = DyckMeanChildrenCounter(initial_stack_capacity=2)
        packed = pack_structural(structural)
        split = min(len(structural), 3)
        counter.scan(packed, 0, split)
        counter.scan(packed, split, len(structural))
        observed = counter.result()
        expected = canonical_counts(structural)
        assert {key: observed[key] for key in expected} == expected


def test_unaligned_multi_segment_one_pass(tmp_path: Path) -> None:
    structural = "01()0(()())1))()((()001(1)0"
    packed = pack_structural(structural)
    source = tmp_path / "fixture.struct.gz"
    with gzip.open(source, "wb") as handle:
        handle.write(packed)

    specs = [SegmentSpec(1, 3, 7), SegmentSpec(2, 13, 9), SegmentSpec(3, 23, 3)]
    result = scan_v33_gzip_segments(
        source,
        len(structural),
        specs,
        source_sha256="A" * 64,
        chunk_bytes=2,
        checkpoint_interval_bytes=3,
    )
    for spec, observed in zip(specs, result["segments"], strict=True):
        expected = canonical_counts(structural[spec.start : spec.end])
        assert {key: observed[key] for key in expected} == expected


def test_checkpoint_resume_is_scientifically_identical(tmp_path: Path) -> None:
    structural = ("0(()())1))((()" * 20) + "001"
    source = tmp_path / "fixture.struct.gz"
    with gzip.open(source, "wb") as handle:
        handle.write(pack_structural(structural))
    specs = [SegmentSpec(1, 1, 91), SegmentSpec(2, 113, 127)]

    direct = scan_v33_gzip_segments(
        source,
        len(structural),
        specs,
        source_sha256="B" * 64,
        chunk_bytes=5,
        checkpoint_interval_bytes=10,
    )

    captured = []

    class StopAfterCheckpoint(RuntimeError):
        pass

    def interrupt(state):
        captured.append(state)
        raise StopAfterCheckpoint

    with pytest.raises(StopAfterCheckpoint):
        scan_v33_gzip_segments(
            source,
            len(structural),
            specs,
            source_sha256="B" * 64,
            chunk_bytes=5,
            checkpoint_interval_bytes=10,
            checkpoint_callback=interrupt,
        )

    resumed = scan_v33_gzip_segments(
        source,
        len(structural),
        specs,
        source_sha256="B" * 64,
        chunk_bytes=5,
        checkpoint_interval_bytes=10,
        resume_state=captured[-1],
    )
    scientific_keys = (
        "segments",
        "aggregate_total_edges",
        "aggregate_total_nodes",
        "aggregate_branching_nodes",
        "cbar_aggregate",
    )
    assert {key: resumed[key] for key in scientific_keys} == {
        key: direct[key] for key in scientific_keys
    }


def test_numba_failure_restores_state_before_python_fallback(monkeypatch) -> None:
    import v2.signature.streaming_cbar as module

    structural = "(()())()"
    counter = DyckMeanChildrenCounter()
    original = module._scan_codes_numba

    def fail_after_mutation(values, start, end, stack, *state):
        stack[0] = 999
        raise RuntimeError("synthetic compiler failure")

    monkeypatch.setattr(module, "_scan_codes_numba", fail_after_mutation)
    try:
        counter.scan(pack_structural(structural), 0, len(structural))
    finally:
        monkeypatch.setattr(module, "_scan_codes_numba", original)
    observed = counter.result()
    expected = canonical_counts(structural)
    assert {key: observed[key] for key in expected} == expected


def passing_equivalence() -> dict:
    return {
        "status": "passed",
        "preregistration_sha256": PREREG_SHA256,
        "synthetic": {
            "passed": True,
            "fixtures": [
                {"fixture": name, "exact_match": True}
                for name in SYNTHETIC_FIXTURES
            ],
        },
        "e24_calibration": {
            "source_path": "E.struct.gz",
            "metadata_path": "E.json",
            "source_sha256": "8D3BE0CE2159B699D424103C7BD4C02517D2A020ABE2B671EFFFA15CDCAC2DAD",
            "passed": True,
            "integer_match": True,
            "observed": E_EXPECTED,
        },
    }


def test_equivalence_payload_requires_exact_frozen_counts() -> None:
    payload = passing_equivalence()
    validate_equivalence_payload(payload)
    payload["e24_calibration"]["observed"] = {**E_EXPECTED, "total_edges": 1}
    with pytest.raises(RuntimeError, match="total_edges"):
        validate_equivalence_payload(payload)


def test_outcome_hierarchy_uses_frozen_thresholds() -> None:
    equivalence = passing_equivalence()
    assert classify_outcome(equivalence, None, None, "stage0") is None
    assert (
        classify_outcome(equivalence, None, "failure", "all")
        == "inputs-or-equivalence-not-derivable"
    )
    assert (
        classify_outcome(
            equivalence,
            {
                "phi_plus_one_distance": AGGREGATE_TOLERANCE,
                "maximum_segment_distance": SEGMENT_TOLERANCE,
            },
            None,
            "all",
        )
        == "f-cbar-preserved-at-27"
    )
    assert (
        classify_outcome(
            equivalence,
            {
                "phi_plus_one_distance": AGGREGATE_TOLERANCE,
                "maximum_segment_distance": SEGMENT_TOLERANCE * 1.01,
            },
            None,
            "all",
        )
        == "f-cbar-aggregate-only-at-27"
    )
    assert (
        classify_outcome(
            equivalence,
            {
                "phi_plus_one_distance": AGGREGATE_TOLERANCE * 1.01,
                "maximum_segment_distance": 0.0,
            },
            None,
            "all",
        )
        == "f-cbar-not-preserved-at-27"
    )
