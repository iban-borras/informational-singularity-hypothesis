from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from v2.phase1.tower import build_pattern_spaces, build_projection_bundle
from v2.postreview.item4_multiseed_monte_carlo import (
    BASELINE_DISPLAY,
    CLEAN_BASELINE_FULL,
    NULL_FAMILIES,
    SEEDS,
    SOURCE_SPECS,
    compact_profile_from_bundle,
    frozen_protocol,
    tail_vector_from_compact,
    tail_vector_from_full_run,
    valid_fit_metadata,
    write_full_phase1_artifacts,
)
from v2.postreview.item4_statistics import (
    HISTORICAL_SEEDS,
    clopper_pearson_interval,
    classify_outcome,
    coordinate_dispersion,
    holm_stepdown,
    pareto_exceeds,
    summarize_source_family,
)


def test_frozen_ledger_and_protocol_are_complete() -> None:
    assert tuple(SOURCE_SPECS) == ("B", "I", "D", "G")
    assert NULL_FAMILIES == ("markov1", "matched-lz", "phase-matched-lz")
    assert SEEDS == tuple(range(1, 512))
    assert set(HISTORICAL_SEEDS).issubset(SEEDS)
    assert len(SOURCE_SPECS) * len(NULL_FAMILIES) * len(SEEDS) == 6_132
    assert frozen_protocol()["tail_edges"] == 2


def test_compact_profile_is_exactly_equivalent_to_full_canonical_run(
    tmp_path: Path,
) -> None:
    motif = np.asarray([0, 0, 1, 0, 1, 1, 0, 1], dtype=np.uint8)
    bits = np.tile(motif, 1_500)
    segments = [np.ascontiguousarray(part) for part in np.split(bits, 3)]
    scales = (8, 12, 16, 20, 24, 28, 32)
    pattern_spaces = build_pattern_spaces(
        segments,
        scales,
        min_count_floor=16,
        min_count_rate=1e-6,
        min_segment_support=2,
        cv_max=1.5,
    )
    bundle = build_projection_bundle(pattern_spaces, list(scales), ["prefix", "suffix"])
    compact = compact_profile_from_bundle(pattern_spaces, bundle)
    run_dir = tmp_path / "full"
    write_full_phase1_artifacts(
        run_dir,
        spec={
            "source": "B",
            "iteration": 20,
            "cache_path": str(tmp_path / "fixture.npy"),
            "cache_sha256": "A" * 64,
        },
        pattern_spaces=pattern_spaces,
        bundle=bundle,
    )
    assert tail_vector_from_compact(compact) == tail_vector_from_full_run(run_dir)
    assert len(compact["scale_summary"]) == 7
    assert len(compact["fiber_summary"]["prefix"]) == 6
    assert len(compact["transport"]["prefix"]["steps"]) == 5


def test_pareto_exceedance_counts_ties() -> None:
    observed = {"R": 0.8, "S": 0.9, "A": 0.4}
    assert pareto_exceeds(observed, observed)
    assert pareto_exceeds({"R": 0.9, "S": 0.9, "A": 0.5}, observed)
    assert not pareto_exceeds({"R": 0.9, "S": 0.89, "A": 0.5}, observed)


def test_clopper_pearson_boundaries_and_dispersion() -> None:
    low_zero, high_zero = clopper_pearson_interval(0, 511)
    low_full, high_full = clopper_pearson_interval(511, 511)
    assert low_zero == 0.0
    assert 0.0 < high_zero < 0.01
    assert 0.99 < low_full < 1.0
    assert high_full == 1.0
    stats = coordinate_dispersion([1.0, 2.0, 3.0])
    assert stats["mean"] == 2.0
    assert stats["sample_variance"] == 1.0
    assert stats["median"] == 2.0
    assert stats["mad"] == 1.0


def test_source_family_summary_uses_frozen_seed_ledger() -> None:
    observed = {"R": 0.8, "S": 0.8, "A": 0.8}
    nulls = [
        {"seed": seed, "R": 0.9 if seed == 1 else 0.1, "S": 0.9, "A": 0.9}
        for seed in range(1, 512)
    ]
    row = summarize_source_family("B", "markov1", observed, nulls)
    assert row["pareto_exceedance_count"] == 1
    assert row["p_value"] == 2 / 512
    assert row["rank_resolution"] == 1 / 512
    assert row["historical_six_seed_subset"]["pareto_exceedance_count"] == 0
    with pytest.raises(ValueError):
        summarize_source_family("B", "markov1", observed, nulls[:-1])


def test_holm_stepdown_is_deterministic_and_stops_after_first_retention() -> None:
    rows = [
        {"source": "B", "family": "z", "p_value": 0.001},
        {"source": "I", "family": "a", "p_value": 0.004},
        {"source": "D", "family": "a", "p_value": 0.010},
    ]
    trace = holm_stepdown(rows, alpha=0.05)
    assert [(row["source"], row["reject"]) for row in trace] == [
        ("B", True),
        ("I", True),
        ("D", True),
    ]
    rows[1]["p_value"] = 0.04
    rows[2]["p_value"] = 0.03
    trace = holm_stepdown(rows, alpha=0.05)
    assert [row["reject"] for row in trace] == [True, False, False]
    assert [row["holm_adjusted_p_value"] for row in trace] == sorted(
        row["holm_adjusted_p_value"] for row in trace
    )


def _trace(rejected: set[tuple[str, str]]) -> list[dict[str, object]]:
    return [
        {"source": source, "family": family, "reject": (source, family) in rejected}
        for source in SOURCE_SPECS
        for family in NULL_FAMILIES
    ]


def test_outcome_hierarchy_all_branches() -> None:
    empty: list[dict[str, object]] = []
    assert (
        classify_outcome(
            derivable=False,
            degenerate=False,
            baseline_reproduced=True,
            holm_trace=empty,
            sources=SOURCE_SPECS,
            families=NULL_FAMILIES,
        )["outcome"]
        == "inputs-or-run-not-derivable"
    )
    assert (
        classify_outcome(
            derivable=True,
            degenerate=True,
            baseline_reproduced=True,
            holm_trace=empty,
            sources=SOURCE_SPECS,
            families=NULL_FAMILIES,
        )["outcome"]
        == "gate1-tail-readout-degenerate"
    )
    assert (
        classify_outcome(
            derivable=True,
            degenerate=False,
            baseline_reproduced=False,
            holm_trace=empty,
            sources=SOURCE_SPECS,
            families=NULL_FAMILIES,
        )["outcome"]
        == "canonical-tail-baseline-not-reproduced"
    )

    all_tests = {
        (source, family) for source in SOURCE_SPECS for family in NULL_FAMILIES
    }
    assert (
        classify_outcome(
            derivable=True,
            degenerate=False,
            baseline_reproduced=True,
            holm_trace=_trace(all_tests),
            sources=SOURCE_SPECS,
            families=NULL_FAMILIES,
        )["outcome"]
        == "gate1-tail-multiseed-reinforced-across-tested-sources"
    )

    one_source = {("B", family) for family in NULL_FAMILIES}
    assert (
        classify_outcome(
            derivable=True,
            degenerate=False,
            baseline_reproduced=True,
            holm_trace=_trace(one_source),
            sources=SOURCE_SPECS,
            families=NULL_FAMILIES,
        )["outcome"]
        == "gate1-tail-multiseed-source-limited"
    )
    assert (
        classify_outcome(
            derivable=True,
            degenerate=False,
            baseline_reproduced=True,
            holm_trace=_trace({("B", "markov1")}),
            sources=SOURCE_SPECS,
            families=NULL_FAMILIES,
        )["outcome"]
        == "gate1-tail-multiseed-null-family-sensitive"
    )
    assert (
        classify_outcome(
            derivable=True,
            degenerate=False,
            baseline_reproduced=True,
            holm_trace=_trace(set()),
            sources=SOURCE_SPECS,
            families=NULL_FAMILIES,
        )["outcome"]
        == "gate1-tail-multiseed-not-reinforced"
    )


def test_baseline_display_uses_frozen_four_decimal_strings() -> None:
    assert BASELINE_DISPLAY["B"] == {"R": "0.8622", "S": "0.9978", "A": "0.3539"}
    assert format(0.86225, ".4f") == "0.8622"
    for source, vector in CLEAN_BASELINE_FULL.items():
        assert {
            key: format(value, ".4f") for key, value in vector.items()
        } == BASELINE_DISPLAY[source]


def test_all_null_fit_metadata_schemas_are_distinct() -> None:
    common = {
        "candidate_block_bits": [
            1,
            2,
            4,
            8,
            16,
            32,
            64,
            128,
            256,
            512,
            1024,
            2048,
            4096,
        ],
        "window_bits": 32_768,
        "window_starts": [0, 10, 20],
        "lz_abs_error": 0.0,
    }
    assert valid_fit_metadata(
        {
            "fit_metadata": {
                "method": "maximum_likelihood_first_order_binary_markov",
                "transition_counts": [[1, 2], [3, 4]],
                "transition_probabilities": [[0.3, 0.7], [0.4, 0.6]],
            }
        },
        "markov1",
    )
    assert valid_fit_metadata(
        {"fit_metadata": {**common, "method": "block_shuffle_lz_match"}},
        "matched-lz",
    )
    assert valid_fit_metadata(
        {
            "fit_metadata": {
                **common,
                "method": "phase_class_block_shuffle_lz_match",
                "candidate_phase_period_blocks": [2, 3, 5, 7],
            }
        },
        "phase-matched-lz",
    )


def test_full_artifact_manifest_is_json_serializable(tmp_path: Path) -> None:
    payload = {"baseline": BASELINE_DISPLAY, "protocol": frozen_protocol()}
    path = tmp_path / "payload.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8")) == payload
