from __future__ import annotations

import math
from typing import Iterable

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*args, **kwargs):  # type: ignore
        def decorator(func):
            return func
        return decorator


VALID_POLICIES = {"prefix", "suffix", "canonical"}
PRIMARY_Q_VALUES = (0, 1, 2, 3)


@njit(cache=True)
def rolling_codes_uint64_max64(bits: np.ndarray, m: int) -> np.ndarray:
    n = bits.size
    window_count = n - m + 1
    codes = np.empty(window_count, dtype=np.uint64)

    code = np.uint64(0)
    for index in range(m):
        code = (code << np.uint64(1)) | np.uint64(bits[index])
    codes[0] = code

    mask = np.uint64(0xFFFFFFFFFFFFFFFF) if m == 64 else (np.uint64(1) << np.uint64(m)) - np.uint64(1)
    for index in range(1, window_count):
        code = ((code << np.uint64(1)) & mask) | np.uint64(bits[index + m - 1])
        codes[index] = code

    return codes


def parse_scales(raw: str) -> list[int]:
    values = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    if len(values) < 2:
        raise ValueError("At least two distinct scales are required.")
    for value in values:
        if value <= 0:
            raise ValueError("All scales must be positive.")
        if value > 64:
            raise ValueError("This first Phase 1 implementation supports scales <= 64.")
    return values


def parse_policies(raw: str) -> list[str]:
    policies = []
    for item in raw.split(","):
        policy = item.strip().lower()
        if not policy:
            continue
        if policy not in VALID_POLICIES:
            raise ValueError(f"Unsupported projection policy: {policy}")
        if policy not in policies:
            policies.append(policy)
    if not policies:
        raise ValueError("At least one projection policy is required.")
    return policies


def build_pattern_spaces(
    segments: list[np.ndarray],
    scales: Iterable[int],
    *,
    min_count_floor: int = 16,
    min_count_rate: float = 1e-6,
    min_segment_support: int | None = None,
    cv_max: float = 1.5,
    progress=None,
    progress_base: int = 0,
) -> dict[int, dict]:
    segment_count = len(segments)
    if segment_count == 0:
        raise ValueError("At least one segment is required.")

    scales = list(scales)
    effective_support = min_segment_support or max(2, math.ceil(segment_count / 2))
    pattern_spaces: dict[int, dict] = {}

    for offset, m in enumerate(scales, start=1):
        artifact = _build_pattern_space_for_scale(
            segments,
            m,
            min_count_floor=min_count_floor,
            min_count_rate=min_count_rate,
            min_segment_support=effective_support,
            cv_max=cv_max,
        )
        pattern_spaces[m] = artifact
        if progress is not None:
            progress.update(progress_base + offset, message=f"m={m} | kept={artifact['kept_pattern_count']}")

    return pattern_spaces


def build_projection_bundle(
    pattern_spaces: dict[int, dict],
    scales: list[int],
    policies: list[str],
    *,
    q_values: tuple[int, ...] = PRIMARY_Q_VALUES,
    progress=None,
    progress_base: int = 0,
) -> dict:
    edges = [(scales[index], scales[index + 1]) for index in range(len(scales) - 1)]
    projection_artifacts: dict[str, list[dict]] = {policy: [] for policy in policies}
    fiber_artifacts: dict[str, list[dict]] = {policy: [] for policy in policies}
    moment_artifacts: dict[str, list[dict]] = {policy: [] for policy in policies}

    step = 0
    for policy in policies:
        previous_global_moments: dict[int, float] | None = None
        for target_m, source_m in edges:
            projection, fibers, moments = _build_projection_edge(
                target_space=pattern_spaces[target_m],
                source_space=pattern_spaces[source_m],
                target_m=target_m,
                source_m=source_m,
                policy=policy,
                q_values=q_values,
            )
            moments["global_ratios_to_previous_edge"] = (
                {str(q): _safe_ratio(moments["global_moments"][str(q)], previous_global_moments[q]) for q in q_values}
                if previous_global_moments is not None
                else {str(q): None for q in q_values}
            )
            previous_global_moments = {q: moments["global_moments"][str(q)] for q in q_values}
            projection_artifacts[policy].append(projection)
            fiber_artifacts[policy].append(fibers)
            moment_artifacts[policy].append(moments)

            step += 1
            if progress is not None:
                progress.update(progress_base + step, message=f"{policy} | {source_m}->{target_m}")

    return {
        "projection_towers": projection_artifacts,
        "fibers": fiber_artifacts,
        "moments": moment_artifacts,
    }


def serialize_pattern_spaces(pattern_spaces: dict[int, dict]) -> dict[int, dict]:
    return {m: _strip_internal_fields(payload) for m, payload in pattern_spaces.items()}


def summarize_phase1(pattern_spaces: dict[int, dict], projection_bundle: dict) -> dict:
    scale_summary = []
    for m, payload in sorted(pattern_spaces.items()):
        scale_summary.append(
            {
                "m": m,
                "kept_pattern_count": payload["kept_pattern_count"],
                "candidate_pattern_count": payload["candidate_pattern_count"],
                "min_count_threshold": payload["min_count_threshold"],
                "min_segment_support": payload["min_segment_support"],
                "cv_max": payload["cv_max"],
            }
        )

    fiber_summary = {}
    for policy, artifacts in projection_bundle["fibers"].items():
        fiber_summary[policy] = [
            {
                "source_m": item["source_m"],
                "target_m": item["target_m"],
                "survival_fraction": item["aggregate"]["survival_fraction"],
                "mean_fiber_size": item["aggregate"]["mean_fiber_size"],
                "singleton_fraction": item["aggregate"]["singleton_fraction"],
                "mean_child_entropy": item["aggregate"]["mean_child_entropy"],
                "orphan_child_count": item["aggregate"]["orphan_child_count"],
            }
            for item in artifacts
        ]

    return {"scale_summary": scale_summary, "fiber_summary": fiber_summary}


def code_to_pattern(code: int, m: int) -> str:
    return format(int(code), f"0{m}b")


def _build_pattern_space_for_scale(
    segments: list[np.ndarray],
    m: int,
    *,
    min_count_floor: int,
    min_count_rate: float,
    min_segment_support: int,
    cv_max: float,
) -> dict:
    segment_snapshots = []
    summary: dict[int, list[float]] = {}
    total_windows = 0

    for segment in segments:
        if m > int(segment.size):
            raise ValueError(f"Scale m={m} exceeds segment length {segment.size}.")

        codes = rolling_codes_uint64_max64(segment, m)
        unique_codes, counts = np.unique(codes, return_counts=True)
        window_count = int(codes.size)
        total_windows += window_count
        segment_snapshots.append((unique_codes, counts, window_count))

        for code, count in zip(unique_codes.tolist(), counts.tolist()):
            freq = float(count / window_count)
            if code not in summary:
                summary[code] = [float(count), 1.0, freq, freq * freq]
            else:
                stats = summary[code]
                stats[0] += float(count)
                stats[1] += 1.0
                stats[2] += freq
                stats[3] += freq * freq

    count_threshold = max(min_count_floor, math.ceil(min_count_rate * total_windows))
    kept: dict[int, dict] = {}
    for code, stats in summary.items():
        total_count = int(stats[0])
        support_count = int(stats[1])
        if total_count < count_threshold or support_count < min_segment_support:
            continue

        mean_supporting_freq = stats[2] / support_count
        variance = max(0.0, (stats[3] / support_count) - (mean_supporting_freq * mean_supporting_freq))
        cv_supporting = math.sqrt(variance) / mean_supporting_freq if mean_supporting_freq > 0 else 0.0
        if cv_supporting > cv_max:
            continue

        kept[code] = {
            "raw_count": total_count,
            "segment_support": support_count,
            "cv_supporting": cv_supporting,
        }

    segment_count = len(segments)
    segment_windows = [window_count for _, _, window_count in segment_snapshots]
    kept_codes = set(kept)
    detailed_segment_counts: dict[int, list[int]] = {code: [0] * segment_count for code in kept_codes}

    for segment_index, (unique_codes, counts, _) in enumerate(segment_snapshots):
        for code, count in zip(unique_codes.tolist(), counts.tolist()):
            if code in kept_codes:
                detailed_segment_counts[code][segment_index] = int(count)

    patterns = []
    for code, stats in kept.items():
        segment_counts = detailed_segment_counts[code]
        segment_frequencies = [
            float(count / window_count) if window_count else 0.0
            for count, window_count in zip(segment_counts, segment_windows)
        ]
        patterns.append(
            {
                "pattern": code_to_pattern(code, m),
                "raw_count": stats["raw_count"],
                "normalized_frequency": float(stats["raw_count"] / total_windows),
                "segment_counts": segment_counts,
                "segment_frequencies": segment_frequencies,
                "segment_support": stats["segment_support"],
                "cv_supporting": stats["cv_supporting"],
                "inclusion_reason": "count_support_cv",
                "_code": code,
            }
        )

    patterns.sort(key=lambda item: (-item["raw_count"], item["pattern"]))
    pattern_index = {entry["_code"]: entry for entry in patterns}

    return {
        "m": m,
        "segment_count": segment_count,
        "segment_window_counts": segment_windows,
        "total_window_count": total_windows,
        "candidate_pattern_count": len(summary),
        "kept_pattern_count": len(patterns),
        "min_count_threshold": count_threshold,
        "min_segment_support": min_segment_support,
        "cv_max": cv_max,
        "pattern_space_mode": "locally_stable_patterns",
        "patterns": patterns,
        "_pattern_index": pattern_index,
        "_included_codes": kept_codes,
    }


def _build_projection_edge(
    *,
    target_space: dict,
    source_space: dict,
    target_m: int,
    source_m: int,
    policy: str,
    q_values: tuple[int, ...],
) -> tuple[dict, dict, dict]:
    if source_m <= target_m:
        raise ValueError(f"Projection requires source_m > target_m, got {source_m} and {target_m}.")

    target_codes = target_space["_included_codes"]
    target_index = target_space["_pattern_index"]
    source_patterns = source_space["patterns"]
    segment_count = target_space["segment_count"]

    fiber_records = {}
    for code, parent_entry in target_index.items():
        fiber_records[code] = {
            "pattern": parent_entry["pattern"],
            "raw_count": parent_entry["raw_count"],
            "normalized_frequency": parent_entry["normalized_frequency"],
            "segment_support": parent_entry["segment_support"],
            "segment_counts": parent_entry["segment_counts"],
            "_segment_fiber_sizes": [0] * segment_count,
            "children": [],
        }

    mapped_child_count = 0
    orphan_child_count = 0

    for child_entry in source_patterns:
        parent_code = _project_code(
            child_entry["_code"],
            source_m=source_m,
            target_m=target_m,
            policy=policy,
            target_index=target_index,
        )
        if parent_code is None:
            orphan_child_count += 1
            continue

        mapped_child_count += 1
        parent_record = fiber_records[parent_code]
        parent_record["children"].append(
            {
                "pattern": child_entry["pattern"],
                "raw_count": child_entry["raw_count"],
                "normalized_frequency": child_entry["normalized_frequency"],
                "segment_support": child_entry["segment_support"],
                "segment_counts": child_entry["segment_counts"],
            }
        )
        for index, count in enumerate(child_entry["segment_counts"]):
            if count > 0:
                parent_record["_segment_fiber_sizes"][index] += 1

    parent_payloads = []
    fiber_sizes = []
    segment_fiber_size_matrix = []
    child_entropies = []

    for record in fiber_records.values():
        child_masses = [child["normalized_frequency"] for child in record["children"]]
        child_total_mass = float(sum(child_masses))
        child_entropy = _mass_entropy(child_masses)
        fiber_size = len(record["children"])
        segment_fiber_sizes = record["_segment_fiber_sizes"]

        parent_payloads.append(
            {
                "pattern": record["pattern"],
                "raw_count": record["raw_count"],
                "normalized_frequency": record["normalized_frequency"],
                "segment_support": record["segment_support"],
                "segment_counts": record["segment_counts"],
                "fiber_size": fiber_size,
                "segment_fiber_sizes": segment_fiber_sizes,
                "child_total_mass": child_total_mass,
                "child_entropy": child_entropy,
                "survival_flag": fiber_size > 0,
                "children": record["children"],
            }
        )
        fiber_sizes.append(fiber_size)
        segment_fiber_size_matrix.append(segment_fiber_sizes)
        child_entropies.append(child_entropy)

    parent_payloads.sort(key=lambda item: (-item["raw_count"], item["pattern"]))

    parent_count = len(parent_payloads)
    surviving_parent_count = sum(1 for item in parent_payloads if item["fiber_size"] > 0)
    singleton_count = sum(1 for item in parent_payloads if item["fiber_size"] == 1)

    projection_artifact = {
        "policy": policy,
        "target_m": target_m,
        "source_m": source_m,
        "delta_bits": source_m - target_m,
        "target_pattern_count": parent_count,
        "source_pattern_count": len(source_patterns),
        "mapped_child_count": mapped_child_count,
        "orphan_child_count": orphan_child_count,
        "parents_hit_count": surviving_parent_count,
        "parents_hit_fraction": float(surviving_parent_count / parent_count) if parent_count else 0.0,
    }

    fibers_artifact = {
        "policy": policy,
        "target_m": target_m,
        "source_m": source_m,
        "aggregate": {
            "parent_count": parent_count,
            "surviving_parent_count": surviving_parent_count,
            "parentless_count": parent_count - surviving_parent_count,
            "survival_fraction": float(surviving_parent_count / parent_count) if parent_count else 0.0,
            "mean_fiber_size": float(np.mean(fiber_sizes)) if fiber_sizes else 0.0,
            "median_fiber_size": float(np.median(fiber_sizes)) if fiber_sizes else 0.0,
            "singleton_fraction": float(singleton_count / parent_count) if parent_count else 0.0,
            "mean_child_entropy": float(np.mean(child_entropies)) if child_entropies else 0.0,
            "orphan_child_count": orphan_child_count,
        },
        "parents": parent_payloads,
    }

    global_moments = {}
    segment_moments = {}
    for q in q_values:
        global_moments[str(q)] = _moment_from_sizes(fiber_sizes, q)
        segment_moments[str(q)] = [
            _moment_from_sizes([row[index] for row in segment_fiber_size_matrix], q)
            for index in range(segment_count)
        ]

    moments_artifact = {
        "policy": policy,
        "target_m": target_m,
        "source_m": source_m,
        "q0_definition": "non_empty_fiber_count",
        "global_moments": global_moments,
        "segment_moments": segment_moments,
        "segment_mean_moments": {q: float(np.mean(values)) if values else 0.0 for q, values in segment_moments.items()},
        "segment_std_moments": {q: float(np.std(values)) if values else 0.0 for q, values in segment_moments.items()},
    }

    return projection_artifact, fibers_artifact, moments_artifact


def _project_code(
    code: int,
    *,
    source_m: int,
    target_m: int,
    policy: str,
    target_index: dict[int, dict],
) -> int | None:
    delta = source_m - target_m
    prefix_code = int(code >> delta)
    suffix_code = int(code & _mask_for_width(target_m))

    if policy == "prefix":
        return prefix_code if prefix_code in target_index else None
    if policy == "suffix":
        return suffix_code if suffix_code in target_index else None
    if policy == "canonical":
        prefix_entry = target_index.get(prefix_code)
        suffix_entry = target_index.get(suffix_code)
        if prefix_entry is None and suffix_entry is None:
            return None
        if suffix_entry is None:
            return prefix_code
        if prefix_entry is None:
            return suffix_code
        if prefix_entry["raw_count"] > suffix_entry["raw_count"]:
            return prefix_code
        if suffix_entry["raw_count"] > prefix_entry["raw_count"]:
            return suffix_code
        return min(prefix_code, suffix_code)
    raise ValueError(f"Unsupported policy: {policy}")


def _mask_for_width(width: int) -> int:
    return 0xFFFFFFFFFFFFFFFF if width >= 64 else (1 << width) - 1


def _moment_from_sizes(sizes: list[int], q: int) -> float:
    if q == 0:
        return float(sum(1 for size in sizes if size > 0))
    return float(sum(size ** q for size in sizes))


def _mass_entropy(masses: list[float]) -> float:
    if not masses:
        return 0.0
    total = sum(masses)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for mass in masses:
        p = mass / total
        if p > 0:
            entropy -= p * math.log2(p)
    return float(entropy)


def _safe_ratio(current: float, previous: float) -> float | None:
    if previous == 0:
        return None
    return float(current / previous)


def _strip_internal_fields(payload: dict) -> dict:
    clean = {}
    for key, value in payload.items():
        if key.startswith("_"):
            continue
        if key == "patterns":
            clean["patterns"] = [{k: v for k, v in entry.items() if not k.startswith("_")} for entry in value]
        else:
            clean[key] = value
    return clean
