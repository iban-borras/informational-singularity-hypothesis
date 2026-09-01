from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from v2.postreview import item3_rule_local_discriminants as item3


def test_site_geometry_is_relative_disjoint_and_blind() -> None:
    specs = item3.derive_site_specs("B", 100_000_000)
    assert [spec.start for spec in specs] == [9_700_000, 48_500_000, 87_300_000]
    assert all(spec.start >= item3.WINDOW_BITS for spec in specs)
    assert all(left.end <= right.start for left, right in zip(specs, specs[1:]))


def test_site_geometry_rejects_observed_prefix_overlap() -> None:
    with pytest.raises(ValueError, match="previously observed prefix"):
        item3.derive_site_specs("B", 10_000_000)


def test_v33_counter_handles_partial_final_byte(tmp_path: Path) -> None:
    # 0,1,(,),0 encodes three observable bits and one partial final byte.
    codes = [0, 1, 2, 3, 0]
    packed = bytes([(codes[0] << 6) | (codes[1] << 4) | (codes[2] << 2) | codes[3], 0])
    path = tmp_path / "fixture.struct.gz"
    with gzip.open(path, "wb") as handle:
        handle.write(packed)
    path.with_suffix("").with_suffix(".json").write_text(
        json.dumps({"sequence_length": len(codes)}), encoding="utf-8"
    )
    result = item3.count_observable_bits(path, raw_chunk_bytes=1)
    assert result["observable_length"] == 3
    assert result["packed_bytes_read"] == 2
    assert result["padding_symbols_excluded"] == 3


def test_extension_words_follow_frozen_orientation() -> None:
    assert item3.extension_word("1010", "10101100", "prefix") == "1100"
    assert item3.extension_word("1010", "00111010", "suffix") == "0011"
    with pytest.raises(ValueError):
        item3.extension_word("1010", "11111111", "prefix")


def test_edge_profile_uses_segment_specific_parent_and_child_mass() -> None:
    edge = {
        "target_m": 4,
        "source_m": 8,
        "parents": [
            {
                "pattern": "1010",
                "segment_counts": [10, 0, 0],
                "children": [
                    {"pattern": "10100000", "segment_counts": [8, 0, 0]},
                    {"pattern": "10101111", "segment_counts": [2, 0, 0]},
                ],
            },
            {
                "pattern": "1111",
                "segment_counts": [5, 0, 0],
                "children": [
                    {"pattern": "11110001", "segment_counts": [5, 0, 0]},
                ],
            },
        ],
    }
    profile = item3.build_edge_profile(edge, 0, "prefix")
    assert profile["degree_distribution"][1] == pytest.approx(1 / 3)
    assert profile["degree_distribution"][2] == pytest.approx(2 / 3)
    assert profile["extension_distribution"][0] == pytest.approx(8 / 15)
    assert profile["extension_distribution"][15] == pytest.approx(2 / 15)
    assert profile["extension_distribution"][1] == pytest.approx(5 / 15)
    assert sum(profile["sibling_entropy_weights"]) == pytest.approx(1.0)


def test_frozen_null_fit_metadata_guards() -> None:
    markov = {
        "fit_metadata": {
            "method": "maximum_likelihood_first_order_binary_markov",
            "transition_counts": [[1, 2], [3, 4]],
            "transition_probabilities": [[1 / 3, 2 / 3], [3 / 7, 4 / 7]],
        }
    }
    assert item3.valid_fit_metadata(markov, "markov1")
    phase = {
        "fit_metadata": {
            "method": "phase_class_block_shuffle_lz_match",
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
            "candidate_phase_period_blocks": [2, 3, 5, 7],
            "window_bits": 32_768,
            "window_starts": [0, 10, 20],
            "lz_abs_error": 0.001,
        }
    }
    assert item3.valid_fit_metadata(phase, "phase-matched-lz")


def test_sqrt_js_is_symmetric_and_bounded() -> None:
    p = [1.0, 0.0]
    q = [0.0, 1.0]
    assert item3.sqrt_js_distance(p, q) == pytest.approx(1.0)
    assert item3.sqrt_js_distance(p, q) == item3.sqrt_js_distance(q, p)
    assert item3.sqrt_js_distance(p, p) == pytest.approx(0.0)


def test_weighted_wasserstein_matches_point_masses() -> None:
    assert item3.weighted_wasserstein_1([0.0], [1.0], [1.0], [1.0]) == pytest.approx(
        1.0
    )
    assert item3.weighted_wasserstein_1(
        [0.0, 1.0], [0.5, 0.5], [0.25, 0.75], [0.5, 0.5]
    ) == pytest.approx(0.25)


def test_motif_reverse_is_an_involution() -> None:
    distribution = np.arange(1, 17, dtype=np.float64)
    distribution /= distribution.sum()
    reversed_once = item3.reverse_motif_distribution(distribution.tolist())
    assert item3.reverse_motif_distribution(reversed_once) == pytest.approx(
        distribution.tolist()
    )


def test_bootstrap_is_deterministic_and_requires_positive_segments() -> None:
    first = item3.bootstrap_median_ci([0.01, 0.02, 0.03], seed=20260928)
    second = item3.bootstrap_median_ci([0.01, 0.02, 0.03], seed=20260928)
    assert first == second
    assert first[0] > 0
    mixed = item3.bootstrap_median_ci([-0.01, 0.02, 0.03], seed=20260928)
    assert mixed[0] < 0


def test_outcome_hierarchy_all_non_failure_branches() -> None:
    base = [
        {
            "probe1": {
                "raw_positive": True,
                "hard_null_positive": True,
                "G_median": 0.1,
            },
            "probe2": {
                "raw_positive": True,
                "hard_null_positive": True,
                "G_median": 0.1,
            },
        }
        for _ in range(3)
    ]
    assert item3.classify_item3([], base, {"probe1": True, "probe2": True}) == (
        "rule-local-discriminant-two-probe-consensus"
    )
    assert item3.classify_item3([], base, {"probe1": True, "probe2": False}) == (
        "mixed-rule-local-signal"
    )
    assert item3.classify_item3([], base, {"probe1": False, "probe2": False}) == (
        "source-statistics-or-null-mimicry"
    )
    none = [
        {
            "probe1": {
                "raw_positive": False,
                "hard_null_positive": False,
                "G_median": 0.0,
            },
            "probe2": {
                "raw_positive": False,
                "hard_null_positive": False,
                "G_median": 0.0,
            },
        }
        for _ in range(3)
    ]
    assert item3.classify_item3([], none, {"probe1": False, "probe2": False}) == (
        "no-tested-rule-local-separation"
    )
    assert item3.classify_item3(
        [{"bad": True}], none, {"probe1": False, "probe2": False}
    ) == ("projective-readout-degenerate")


def test_segment_comparison_is_label_symmetric() -> None:
    left = synthetic_segment_profile(degree=1, motif=1, entropy=0.1, active=0.2)
    right = synthetic_segment_profile(degree=2, motif=2, entropy=0.8, active=0.7)
    forward = item3.compare_segment_profiles(left, right)
    reverse = item3.compare_segment_profiles(right, left)
    assert forward["D1"] == pytest.approx(reverse["D1"])
    assert forward["D2"] == pytest.approx(reverse["D2"])


def test_full_readout_is_invariant_to_source_label_swap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profiles = synthetic_profile_matrix()
    monkeypatch.setattr(item3, "load_job_profiles", lambda records: profiles)
    forward = item3.build_item3_readout([])["summary"]
    swapped = {
        (("BXOR" if key[0] == "B" else "B"), *key[1:]): value
        for key, value in profiles.items()
    }
    monkeypatch.setattr(item3, "load_job_profiles", lambda records: swapped)
    reverse = item3.build_item3_readout([])["summary"]
    assert forward["outcome"] == reverse["outcome"]
    assert forward["region_positive"] == reverse["region_positive"]
    for left, right in zip(forward["per_site"], reverse["per_site"]):
        assert left["probe1"]["G_median"] == pytest.approx(right["probe1"]["G_median"])
        assert left["probe2"]["G_median"] == pytest.approx(right["probe2"]["G_median"])
        assert left["D3_median"] == pytest.approx(right["D3_median"])


def test_profile_loader_uses_the_frozen_five_field_identity(tmp_path: Path) -> None:
    records = []
    for source in ("B", "BXOR"):
        for q in ("1/10", "1/2", "9/10"):
            rows = [("observed", None, None)]
            rows.extend(
                ("null", family, seed)
                for family in item3.NULL_FAMILIES
                for seed in item3.SEEDS
            )
            for kind, family, seed in rows:
                path = (
                    tmp_path
                    / f"{source}_{q.replace('/', '-')}_{kind}_{family}_{seed}.json"
                )
                path.write_text('{"segments": []}', encoding="utf-8")
                records.append(
                    {
                        "source": source,
                        "q": q,
                        "kind": kind,
                        "family": family,
                        "seed": seed,
                        "profile_path": str(path),
                    }
                )

    profiles = item3.load_job_profiles(records)

    assert len(profiles) == 78
    assert profiles[("B", "1/10", "observed", None, None)] == {"segments": []}
    assert profiles[("BXOR", "9/10", "null", "phase-matched-lz", 503)] == {
        "segments": []
    }


def test_profile_job_builds_the_complete_canonical_bundle(tmp_path: Path) -> None:
    cache_path = tmp_path / "fixture.npy"
    bits = np.zeros(item3.WINDOW_BITS, dtype=np.uint8)
    sidecar = item3.freeze_cache(cache_path, bits, metadata={"cache_kind": "fixture"})
    identity = {
        "job_id": "fixture",
        "source": "B",
        "q": "1/10",
        "kind": "observed",
        "family": None,
        "seed": None,
        "cache_path": str(cache_path),
        "cache_sha256": sidecar["cache_sha256"],
        "stage0_manifest_sha256": "0" * 64,
        "preregistration_sha256": item3.PREREG_SHA256,
        "protocol": item3.frozen_protocol(),
        "code_hashes": {"fixture": "1" * 64},
    }
    spec = {
        **{
            key: identity[key]
            for key in ("job_id", "source", "q", "kind", "family", "seed")
        },
        "cache_path": str(cache_path),
        "cache_sha256": sidecar["cache_sha256"],
        "identity": identity,
        "job_dir": str(tmp_path / "job"),
    }

    record = item3.build_profile_job(spec)
    profile = json.loads(Path(record["profile_path"]).read_text(encoding="utf-8"))

    assert record["status"] == "completed"
    assert len(profile["segments"]) == 3
    for segment in profile["segments"]:
        for policy in item3.POLICIES:
            assert len(segment["policies"][policy]["edges"]) == 6
            assert len(segment["policies"][policy]["transport_steps"]) == 5
    assert item3.build_profile_job(spec)["reused"] is True


def synthetic_profile_matrix() -> dict[
    tuple[str, str, str, str | None, int | None], dict
]:
    profiles = {}
    for q, _, _ in item3.SITE_FRACTIONS:
        profiles[("B", q, "observed", None, None)] = synthetic_job_profile(
            degree=1, motif=1, entropy=0.1, active=0.2
        )
        profiles[("BXOR", q, "observed", None, None)] = synthetic_job_profile(
            degree=2, motif=2, entropy=0.8, active=0.7
        )
        for family in item3.NULL_FAMILIES:
            for seed in item3.SEEDS:
                null = synthetic_job_profile(degree=1, motif=1, entropy=0.2, active=0.3)
                profiles[("B", q, "null", family, seed)] = null
                profiles[("BXOR", q, "null", family, seed)] = null
    return profiles


def synthetic_job_profile(**kwargs: float | int) -> dict:
    return {"segments": [synthetic_segment_profile(**kwargs) for _ in range(3)]}


def synthetic_segment_profile(
    *, degree: int, motif: int, entropy: float, active: float
) -> dict:
    policies = {}
    for policy in item3.POLICIES:
        edges = []
        for target_m, source_m in zip(item3.SCALES, item3.SCALES[1:]):
            degree_distribution = [0.0] * 17
            degree_distribution[degree] = 1.0
            extension_distribution = [0.0] * 16
            extension_distribution[motif] = 1.0
            edges.append(
                {
                    "target_m": target_m,
                    "source_m": source_m,
                    "parent_mass_total": 1.0,
                    "degree_distribution": degree_distribution,
                    "sibling_entropy_values": [entropy],
                    "sibling_entropy_weights": [1.0],
                    "extension_distribution": extension_distribution,
                    "extension_mass_total": 1.0,
                }
            )
        steps = [
            {
                "target_m": target_m,
                "source_m": source_m,
                "transport_active_mass": active,
                "transport_entropy_over_four": entropy / 4.0,
                "parent_mass_total": 1.0,
            }
            for target_m, source_m in zip(item3.SCALES[:-2], item3.SCALES[1:-1])
        ]
        policies[policy] = {"edges": edges, "transport_steps": steps}
    return {"segment_index": 0, "policies": policies}
