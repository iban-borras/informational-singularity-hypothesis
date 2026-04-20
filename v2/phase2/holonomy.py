from __future__ import annotations

from .defects import (
    _infer_segment_count,
    _mean,
    _pairwise_jaccard_mean,
    _pick_best,
    _require_edge,
    _safe_ratio,
    _weight_entropy,
    classify_variant_role,
    load_policy_fibers,
)


def build_holonomy_rows(
    runs: list[dict],
    *,
    window_scales: tuple[int, int, int],
    policies: list[str],
) -> list[dict]:
    if len(policies) != 2:
        raise ValueError("Loop holonomy pilot requires exactly two policies.")

    left_policy, right_policy = policies
    rows = []
    for run in runs:
        config = run["dataset"]["config"]
        left_fibers = load_policy_fibers(_run_dir(run), left_policy)
        right_fibers = load_policy_fibers(_run_dir(run), right_policy)
        holonomy_result = compute_loop_holonomy_result(
            left_fibers,
            right_fibers,
            window_scales=window_scales,
            left_policy=left_policy,
            right_policy=right_policy,
        )
        rows.append(
            {
                "variant": config["variant"],
                "variant_role": classify_variant_role(config["variant"]),
                "iteration": config.get("iteration"),
                "segment_bits": config.get("segment_bits"),
                "num_segments": config.get("num_segments"),
                "segment_offset_bits": config.get("segment_offset_bits", 0),
                "scales": list(config.get("scales", [])),
                "phase1_policies": list(config.get("policies", [])),
                "window_scales": list(window_scales),
                "policies": list(policies),
                "run_dir": str(_run_dir(run)),
                "dataset_path": run["_dataset_path"],
                "summary_path": run["_summary_path"],
                "generated_at": run["generated_at"],
                "holonomy_result": holonomy_result,
                "loopable_parent_count": holonomy_result["loopable_parent_count"],
                "partial_loop_parent_count": holonomy_result["partial_loop_parent_count"],
                "closed_parent_count": holonomy_result["closed_parent_count"],
                "loopable_parent_fraction": holonomy_result["loopable_parent_fraction"],
                "holonomy_rate": holonomy_result["holonomy_rate"],
                "drift_rate": holonomy_result["drift_rate"],
                "flat_loop_rate": holonomy_result["flat_loop_rate"],
                "mean_cross_return_distance": holonomy_result["mean_cross_return_distance"],
                "mean_left_return_drift": holonomy_result["mean_left_return_drift"],
                "mean_right_return_drift": holonomy_result["mean_right_return_drift"],
                "holonomy_family_count": holonomy_result["holonomy_family_count"],
                "holonomy_entropy": holonomy_result["holonomy_entropy"],
                "segment_consistency": holonomy_result["segment_consistency"],
            }
        )
    return rows


def compute_loop_holonomy_result(
    left_fibers: list[dict],
    right_fibers: list[dict],
    *,
    window_scales: tuple[int, int, int],
    left_policy: str,
    right_policy: str,
) -> dict:
    low_scale, mid_scale, high_scale = window_scales
    left_low = _require_edge(left_fibers, target_m=low_scale, source_m=mid_scale)
    left_high = _require_edge(left_fibers, target_m=mid_scale, source_m=high_scale)
    right_low = _require_edge(right_fibers, target_m=low_scale, source_m=mid_scale)
    right_high = _require_edge(right_fibers, target_m=mid_scale, source_m=high_scale)

    left_low_index = {parent["pattern"]: parent for parent in left_low.get("parents", [])}
    right_low_index = {parent["pattern"]: parent for parent in right_low.get("parents", [])}
    common_patterns = sorted(set(left_low_index) & set(right_low_index))
    left_high_index = {parent["pattern"]: parent for parent in left_high.get("parents", [])}
    right_high_index = {parent["pattern"]: parent for parent in right_high.get("parents", [])}

    parent_results = []
    family_weights: dict[str, float] = {}
    family_counts: dict[str, int] = {}
    loopable_weight = 0.0
    weighted_cross_distance = 0.0
    weighted_drift = 0.0

    for pattern in common_patterns:
        detail = _analyze_loop_parent(
            pattern,
            left_low_index[pattern],
            left_high_index,
            right_low_index[pattern],
            right_high_index,
            low_scale=low_scale,
            left_policy=left_policy,
            right_policy=right_policy,
            segment_index=None,
        )
        if detail is None:
            continue
        parent_results.append(detail)
        if detail["loopable"]:
            loopable_weight += detail["parent_weight"]
            weighted_cross_distance += detail["parent_weight"] * detail["cross_return_distance"]
            weighted_drift += detail["parent_weight"] * (
                detail["left_return_drift"] + detail["right_return_drift"]
            ) / 2.0
        if detail["holonomy_family"] is not None:
            family = detail["holonomy_family"]
            family_weights[family] = family_weights.get(family, 0.0) + detail["parent_weight"]
            family_counts[family] = family_counts.get(family, 0) + 1

    segment_count = min(_infer_segment_count(left_low), _infer_segment_count(right_low))
    segment_results = []
    segment_family_sets = []
    for segment_index in range(segment_count):
        segment_result = _analyze_loop_segment(
            common_patterns,
            left_low_index,
            left_high_index,
            right_low_index,
            right_high_index,
            low_scale=low_scale,
            left_policy=left_policy,
            right_policy=right_policy,
            segment_index=segment_index,
        )
        segment_results.append(segment_result)
        if segment_result["families"]:
            segment_family_sets.append(set(segment_result["families"]))

    examined_parent_count = len(parent_results)
    loopable_parent_count = sum(1 for item in parent_results if item["loopable"])
    partial_loop_parent_count = sum(1 for item in parent_results if item["partial_loop"])
    closed_parent_count = sum(1 for item in parent_results if item["closed_loop"])
    holonomy_parent_count = sum(1 for item in parent_results if item["holonomy_flag"])
    drift_parent_count = sum(1 for item in parent_results if item["drift_flag"])
    flat_parent_count = sum(1 for item in parent_results if item["flat_flag"])

    top_families = [
        {"family": family, "weight": family_weights[family], "count": family_counts[family]}
        for family in sorted(family_weights, key=lambda item: (-family_weights[item], -family_counts[item], item))
    ]
    cross_distances = [item["cross_return_distance"] for item in parent_results if item["loopable"]]
    left_drifts = [item["left_return_drift"] for item in parent_results if item["loopable"]]
    right_drifts = [item["right_return_drift"] for item in parent_results if item["loopable"]]

    return {
        "window_scales": [low_scale, mid_scale, high_scale],
        "left_policy": left_policy,
        "right_policy": right_policy,
        "loop_definition": {
            "left_path": f"{left_policy}->{left_policy}->{right_policy}-return",
            "right_path": f"{right_policy}->{right_policy}->{left_policy}-return",
        },
        "examined_parent_count": examined_parent_count,
        "loopable_parent_count": loopable_parent_count,
        "partial_loop_parent_count": partial_loop_parent_count,
        "closed_parent_count": closed_parent_count,
        "loopable_parent_fraction": _safe_ratio(loopable_parent_count, examined_parent_count),
        "holonomy_parent_count": holonomy_parent_count,
        "holonomy_rate": _safe_ratio(holonomy_parent_count, loopable_parent_count),
        "drift_parent_count": drift_parent_count,
        "drift_rate": _safe_ratio(drift_parent_count, loopable_parent_count),
        "flat_parent_count": flat_parent_count,
        "flat_loop_rate": _safe_ratio(flat_parent_count, loopable_parent_count),
        "mean_cross_return_distance": _mean(cross_distances),
        "mean_left_return_drift": _mean(left_drifts),
        "mean_right_return_drift": _mean(right_drifts),
        "weighted_cross_return_distance": _safe_ratio(weighted_cross_distance, loopable_weight),
        "weighted_mean_return_drift": _safe_ratio(weighted_drift, loopable_weight),
        "holonomy_family_count": len(top_families),
        "holonomy_entropy": _weight_entropy(list(family_weights.values())),
        "segment_consistency": _pairwise_jaccard_mean(segment_family_sets),
        "top_holonomy_families": top_families[:10],
        "segment_results": segment_results,
        "parent_results": parent_results,
        "notes": [
            "left path = dominant continuation under the left policy across both steps, then return to the low scale through the right projection",
            "right path = dominant continuation under the right policy across both steps, then return to the low scale through the left projection",
            "holonomy = disagreement between the two returned low-scale patterns",
            "drift = either returned low-scale pattern differs from the original low-scale parent",
            "flat loop = both returned patterns match each other and the original parent pattern",
        ],
    }


def render_console_summary(rows: list[dict]) -> str:
    lines = [
        "Phase 2 cross-policy loop holonomy pilot",
        "-" * 118,
        f"{'variant':<14}{'iter':>6}{'loop':>10}{'holo':>10}{'drift':>10}{'flat':>10}{'xdist':>10}{'ldrift':>10}{'rdrift':>10}{'entropy':>10}",
    ]
    for row in rows:
        lines.append(
            f"{row['variant']:<14}"
            f"{_fmt_int(row['iteration']):>6}"
            f"{_fmt(row.get('loopable_parent_fraction')):>10}"
            f"{_fmt(row.get('holonomy_rate')):>10}"
            f"{_fmt(row.get('drift_rate')):>10}"
            f"{_fmt(row.get('flat_loop_rate')):>10}"
            f"{_fmt(row.get('mean_cross_return_distance')):>10}"
            f"{_fmt(row.get('mean_left_return_drift')):>10}"
            f"{_fmt(row.get('mean_right_return_drift')):>10}"
            f"{_fmt(row.get('holonomy_entropy')):>10}"
        )
    return "\n".join(lines)


def render_markdown_report(rows: list[dict], selection: dict) -> str:
    lines = [
        "# Phase 2 Cross-Policy Loop Holonomy Pilot",
        "",
        "## Selection",
        "",
        f"- Variants: {', '.join(selection['variants'])}",
        f"- Segment protocol: {selection['num_segments']} x {selection['segment_bits']}",
        f"- Segment offset bits: {selection['segment_offset_bits']}",
        f"- Phase 1 scales: {', '.join(str(value) for value in selection['phase1_scales'])}",
        f"- Loop window: {', '.join(str(value) for value in selection['window_scales'])}",
        f"- Policies: {', '.join(selection['policies'])}",
        "",
        "## Compact Readout",
        "",
        "| Variant | Iter | Loopable frac | Holonomy rate | Drift rate | Flat-loop rate | Mean cross distance | Mean left drift | Mean right drift | Holonomy entropy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['iteration']} | "
            f"{_fmt(row.get('loopable_parent_fraction'))} | { _fmt(row.get('holonomy_rate'))} | "
            f"{_fmt(row.get('drift_rate'))} | { _fmt(row.get('flat_loop_rate'))} | "
            f"{_fmt(row.get('mean_cross_return_distance'))} | { _fmt(row.get('mean_left_return_drift'))} | "
            f"{_fmt(row.get('mean_right_return_drift'))} | { _fmt(row.get('holonomy_entropy'))} |"
        )

    for row in rows:
        result = row["holonomy_result"]
        lines.extend(
            [
                "",
                f"## {row['variant']}",
                "",
                f"- Loop definition: `{result['loop_definition']['left_path']}` versus `{result['loop_definition']['right_path']}`",
                f"- Examined parents: {result['examined_parent_count']}",
                f"- Loopable parents: {result['loopable_parent_count']}",
                f"- Partial-loop parents: {result['partial_loop_parent_count']}",
                f"- Closed parents: {result['closed_parent_count']}",
                f"- Loopable fraction: {_fmt(result['loopable_parent_fraction'])}",
                f"- Holonomy rate: {_fmt(result['holonomy_rate'])}",
                f"- Drift rate: {_fmt(result['drift_rate'])}",
                f"- Flat-loop rate: {_fmt(result['flat_loop_rate'])}",
                f"- Mean cross return distance: {_fmt(result['mean_cross_return_distance'])}",
                f"- Mean left return drift: {_fmt(result['mean_left_return_drift'])}",
                f"- Mean right return drift: {_fmt(result['mean_right_return_drift'])}",
                f"- Holonomy entropy: {_fmt(result['holonomy_entropy'])}",
                f"- Segment consistency: {_fmt(result['segment_consistency'])}",
                "",
                "Top holonomy families:",
            ]
        )
        if result["top_holonomy_families"]:
            for family in result["top_holonomy_families"][:5]:
                lines.append(f"- `{family['family']}` | weight={_fmt(family['weight'])} | count={family['count']}")
        else:
            lines.append("- none")

    return "\n".join(lines)


def _analyze_loop_segment(
    common_patterns: list[str],
    left_low_index: dict[str, dict],
    left_high_index: dict[str, dict],
    right_low_index: dict[str, dict],
    right_high_index: dict[str, dict],
    *,
    low_scale: int,
    left_policy: str,
    right_policy: str,
    segment_index: int,
) -> dict:
    parent_results = []
    families = []
    for pattern in common_patterns:
        detail = _analyze_loop_parent(
            pattern,
            left_low_index[pattern],
            left_high_index,
            right_low_index[pattern],
            right_high_index,
            low_scale=low_scale,
            left_policy=left_policy,
            right_policy=right_policy,
            segment_index=segment_index,
        )
        if detail is None:
            continue
        parent_results.append(detail)
        if detail["holonomy_family"] is not None:
            families.append(detail["holonomy_family"])

    examined_parent_count = len(parent_results)
    loopable_parent_count = sum(1 for item in parent_results if item["loopable"])
    holonomy_parent_count = sum(1 for item in parent_results if item["holonomy_flag"])
    drift_parent_count = sum(1 for item in parent_results if item["drift_flag"])
    flat_parent_count = sum(1 for item in parent_results if item["flat_flag"])
    cross_distances = [item["cross_return_distance"] for item in parent_results if item["loopable"]]
    left_drifts = [item["left_return_drift"] for item in parent_results if item["loopable"]]
    right_drifts = [item["right_return_drift"] for item in parent_results if item["loopable"]]

    return {
        "segment_index": segment_index,
        "examined_parent_count": examined_parent_count,
        "loopable_parent_count": loopable_parent_count,
        "loopable_parent_fraction": _safe_ratio(loopable_parent_count, examined_parent_count),
        "holonomy_parent_count": holonomy_parent_count,
        "holonomy_rate": _safe_ratio(holonomy_parent_count, loopable_parent_count),
        "drift_parent_count": drift_parent_count,
        "drift_rate": _safe_ratio(drift_parent_count, loopable_parent_count),
        "flat_parent_count": flat_parent_count,
        "flat_loop_rate": _safe_ratio(flat_parent_count, loopable_parent_count),
        "mean_cross_return_distance": _mean(cross_distances),
        "mean_left_return_drift": _mean(left_drifts),
        "mean_right_return_drift": _mean(right_drifts),
        "families": sorted(set(families)),
    }


def _analyze_loop_parent(
    pattern: str,
    left_parent: dict,
    left_high_index: dict[str, dict],
    right_parent: dict,
    right_high_index: dict[str, dict],
    *,
    low_scale: int,
    left_policy: str,
    right_policy: str,
    segment_index: int | None,
) -> dict | None:
    left_path = _dominant_two_step_path(left_parent, left_high_index, segment_index=segment_index)
    right_path = _dominant_two_step_path(right_parent, right_high_index, segment_index=segment_index)
    parent_weight = _parent_weight(left_parent, segment_index)

    if left_path is None and right_path is None:
        return _empty_detail(pattern, parent_weight, partial_loop=False, closed_loop=True)
    if left_path is None or right_path is None:
        return _empty_detail(pattern, parent_weight, partial_loop=True, closed_loop=False)

    left_return = _project_pattern(left_path["high_pattern"], target_m=low_scale, policy=right_policy)
    right_return = _project_pattern(right_path["high_pattern"], target_m=low_scale, policy=left_policy)
    cross_return_distance = _normalized_hamming(left_return, right_return)
    left_return_drift = _normalized_hamming(left_return, pattern)
    right_return_drift = _normalized_hamming(right_return, pattern)
    holonomy_flag = cross_return_distance > 0.0
    drift_flag = left_return_drift > 0.0 or right_return_drift > 0.0
    flat_flag = not holonomy_flag and not drift_flag
    holonomy_family = None if flat_flag else f"{pattern}|L:{left_return}|R:{right_return}"

    return {
        "pattern": pattern,
        "parent_weight": parent_weight,
        "loopable": True,
        "partial_loop": False,
        "closed_loop": False,
        "left_return_pattern": left_return,
        "right_return_pattern": right_return,
        "cross_return_distance": cross_return_distance,
        "left_return_drift": left_return_drift,
        "right_return_drift": right_return_drift,
        "holonomy_flag": holonomy_flag,
        "drift_flag": drift_flag,
        "flat_flag": flat_flag,
        "holonomy_family": holonomy_family,
        "left_path": left_path,
        "right_path": right_path,
    }


def _dominant_two_step_path(parent: dict, high_parent_index: dict[str, dict], *, segment_index: int | None) -> dict | None:
    children = parent.get("children", [])
    if not children:
        return None

    child_entries = []
    for child in children:
        local_score = _child_weight(child, segment_index)
        high_parent = high_parent_index.get(child["pattern"])
        best_grandchild = _pick_best_grandchild(high_parent, segment_index=segment_index)
        child_entries.append(
            {
                "pattern": child["pattern"],
                "local_score": local_score,
                "raw_count": int(child.get("raw_count", 0)),
                "segment_support": int(child.get("segment_support", 0)),
                "grandchild_count": 0 if high_parent is None else len(high_parent.get("children", [])),
                "grandchild_pattern": None if best_grandchild is None else best_grandchild["pattern"],
                "grandchild_score": None if best_grandchild is None else best_grandchild["local_score"],
            }
        )

    best_child = _pick_best(child_entries, score_key="local_score", require_positive=True)
    if best_child is None or best_child["grandchild_pattern"] is None:
        return None
    return {
        "mid_pattern": best_child["pattern"],
        "high_pattern": best_child["grandchild_pattern"],
        "step1_score": best_child["local_score"],
        "step2_score": best_child["grandchild_score"],
    }


def _pick_best_grandchild(high_parent: dict | None, *, segment_index: int | None) -> dict | None:
    if high_parent is None:
        return None
    entries = []
    for child in high_parent.get("children", []):
        entries.append(
            {
                "pattern": child["pattern"],
                "local_score": _child_weight(child, segment_index),
                "raw_count": int(child.get("raw_count", 0)),
                "segment_support": int(child.get("segment_support", 0)),
                "grandchild_count": 0,
            }
        )
    return _pick_best(entries, score_key="local_score", require_positive=True)


def _project_pattern(pattern: str, *, target_m: int, policy: str) -> str:
    if len(pattern) < target_m:
        raise ValueError(f"Cannot project pattern of length {len(pattern)} to {target_m}.")
    if policy == "prefix":
        return pattern[:target_m]
    if policy == "suffix":
        return pattern[-target_m:]
    raise ValueError(f"Unsupported projection policy for loop holonomy: {policy}")


def _normalized_hamming(left: str, right: str) -> float:
    if len(left) != len(right):
        raise ValueError("Normalized Hamming distance requires equal-length patterns.")
    if not left:
        return 0.0
    return float(sum(1 for a, b in zip(left, right) if a != b) / len(left))


def _child_weight(child: dict, segment_index: int | None) -> float:
    return (
        float(child.get("normalized_frequency", 0.0))
        if segment_index is None
        else float(child.get("segment_counts", [])[segment_index])
    )


def _parent_weight(parent: dict, segment_index: int | None) -> float:
    return (
        float(parent.get("normalized_frequency", 0.0))
        if segment_index is None
        else float(parent.get("segment_counts", [])[segment_index])
    )


def _empty_detail(pattern: str, parent_weight: float, *, partial_loop: bool, closed_loop: bool) -> dict:
    return {
        "pattern": pattern,
        "parent_weight": parent_weight,
        "loopable": False,
        "partial_loop": partial_loop,
        "closed_loop": closed_loop,
        "left_return_pattern": None,
        "right_return_pattern": None,
        "cross_return_distance": None,
        "left_return_drift": None,
        "right_return_drift": None,
        "holonomy_flag": False,
        "drift_flag": False,
        "flat_flag": False,
        "holonomy_family": None,
        "left_path": None,
        "right_path": None,
    }


def _run_dir(run: dict):
    from pathlib import Path

    return Path(run["_run_dir"])


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def _fmt_int(value: int | None) -> str:
    if value is None:
        return "-"
    return str(value)
