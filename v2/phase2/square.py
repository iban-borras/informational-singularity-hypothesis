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


def build_square_rows(
    runs: list[dict],
    *,
    window_scales: tuple[int, int, int, int],
    left_policy: str,
    right_policy: str,
) -> list[dict]:
    rows = []
    for run in runs:
        config = run["dataset"]["config"]
        run_dir = _run_dir(run)
        left_fibers = load_policy_fibers(run_dir, left_policy)
        right_fibers = load_policy_fibers(run_dir, right_policy)
        square_result = compute_square_result(
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
                "left_policy": left_policy,
                "right_policy": right_policy,
                "run_dir": str(run_dir),
                "dataset_path": run["_dataset_path"],
                "summary_path": run["_summary_path"],
                "generated_at": run["generated_at"],
                "square_result": square_result,
                "bridge_count": square_result["bridge_count"],
                "future_bridge_fraction": square_result["future_bridge_fraction"],
                "squareable_bridge_fraction": square_result["squareable_bridge_fraction"],
                "anchored_square_rate": square_result["anchored_square_rate"],
                "square_defect_rate": square_result["square_defect_rate"],
                "bridge_disagreement_rate": square_result["bridge_disagreement_rate"],
                "mean_square_distance": square_result["mean_square_distance"],
                "square_entropy": square_result["square_entropy"],
                "segment_consistency": square_result["segment_consistency"],
            }
        )
    return rows


def compute_square_result(
    left_fibers: list[dict],
    right_fibers: list[dict],
    *,
    window_scales: tuple[int, int, int, int],
    left_policy: str,
    right_policy: str,
) -> dict:
    base_scale, parent_scale, bridge_scale, high_scale = window_scales
    left_low = _require_edge(left_fibers, target_m=parent_scale, source_m=bridge_scale)
    right_low = _require_edge(right_fibers, target_m=parent_scale, source_m=bridge_scale)
    left_high = _require_edge(left_fibers, target_m=bridge_scale, source_m=high_scale)
    right_high = _require_edge(right_fibers, target_m=bridge_scale, source_m=high_scale)

    bridge_items = _build_bridge_items(left_low, right_low, base_scale=base_scale)
    left_parent_index = {item["pattern"]: item for item in left_low.get("parents", [])}
    right_parent_index = {item["pattern"]: item for item in right_low.get("parents", [])}
    left_high_index = {item["pattern"]: item for item in left_high.get("parents", [])}
    right_high_index = {item["pattern"]: item for item in right_high.get("parents", [])}

    bridge_results = []
    defect_weight = 0.0
    future_weight = 0.0
    family_weights: dict[str, float] = {}
    family_counts: dict[str, int] = {}
    for bridge in bridge_items:
        detail = _analyze_bridge(
            bridge,
            left_parent_index,
            right_parent_index,
            left_high_index,
            right_high_index,
            segment_index=None,
        )
        if detail is None:
            continue
        bridge_results.append(detail)
        if detail["future_bridge"]:
            future_weight += detail["bridge_weight"]
            if detail["square_defect_flag"]:
                defect_weight += detail["bridge_weight"]
                family = detail["square_family"]
                if family is not None:
                    family_weights[family] = family_weights.get(family, 0.0) + detail["bridge_weight"]
                    family_counts[family] = family_counts.get(family, 0) + 1

    segment_results = []
    segment_family_sets = []
    segment_count = min(_infer_segment_count(left_low), _infer_segment_count(right_low))
    for segment_index in range(segment_count):
        segment_result = _analyze_square_segment(
            bridge_items,
            left_parent_index,
            right_parent_index,
            left_high_index,
            right_high_index,
            segment_index=segment_index,
        )
        segment_results.append(segment_result)
        if segment_result["families"]:
            segment_family_sets.append(set(segment_result["families"]))

    bridge_count = len(bridge_results)
    future_bridge_count = sum(1 for item in bridge_results if item["future_bridge"])
    squareable_bridge_count = sum(1 for item in bridge_results if item["squareable"])
    anchored_square_count = sum(1 for item in bridge_results if item["anchored_square"])
    square_defect_count = sum(1 for item in bridge_results if item["square_defect_flag"])
    disagreement_count = sum(1 for item in bridge_results if item["bridge_disagreement_flag"])
    distances = [item["square_distance"] for item in bridge_results if item["future_bridge"]]
    top_families = [
        {"family": family, "weight": family_weights[family], "count": family_counts[family]}
        for family in sorted(family_weights, key=lambda item: (-family_weights[item], -family_counts[item], item))
    ]

    return {
        "window_scales": [base_scale, parent_scale, bridge_scale, high_scale],
        "left_policy": left_policy,
        "right_policy": right_policy,
        "bridge_count": bridge_count,
        "future_bridge_count": future_bridge_count,
        "future_bridge_fraction": _safe_ratio(future_bridge_count, bridge_count),
        "squareable_bridge_count": squareable_bridge_count,
        "squareable_bridge_fraction": _safe_ratio(squareable_bridge_count, bridge_count),
        "anchored_square_count": anchored_square_count,
        "anchored_square_rate": _safe_ratio(anchored_square_count, future_bridge_count),
        "square_defect_count": square_defect_count,
        "square_defect_rate": _safe_ratio(square_defect_count, future_bridge_count),
        "square_defect_mass_rate": _safe_ratio(defect_weight, future_weight),
        "bridge_disagreement_count": disagreement_count,
        "bridge_disagreement_rate": _safe_ratio(disagreement_count, future_bridge_count),
        "mean_square_distance": _mean(distances),
        "square_family_count": len(top_families),
        "square_entropy": _weight_entropy(list(family_weights.values())),
        "segment_consistency": _pairwise_jaccard_mean(segment_family_sets),
        "top_square_families": top_families[:10],
        "segment_results": segment_results,
        "bridge_results": bridge_results,
        "notes": [
            "bridge = a bridge-scale pattern present as a left-policy child of its left parent and a right-policy child of its right parent",
            "squareable = both parents recover a dominant bridge candidate and the actual bridge remains future-bearing at the high scale under both policies",
            "anchored square = both corner-dominant bridge candidates coincide with the actual shared bridge",
            "square defect = a future-bearing bridge whose corner-dominant bridge candidates fail to anchor the shared bridge",
        ],
    }


def render_console_summary(rows: list[dict]) -> str:
    lines = [
        "Phase 2 commuting-square cocycle pilot",
        "-" * 118,
        f"{'variant':<14}{'iter':>6}{'future':>10}{'sqbl':>10}{'anchor':>10}{'def':>10}{'disagr':>10}{'dist':>10}{'entropy':>10}{'seg':>10}",
    ]
    for row in rows:
        lines.append(
            f"{row['variant']:<14}"
            f"{_fmt_int(row['iteration']):>6}"
            f"{_fmt(row.get('future_bridge_fraction')):>10}"
            f"{_fmt(row.get('squareable_bridge_fraction')):>10}"
            f"{_fmt(row.get('anchored_square_rate')):>10}"
            f"{_fmt(row.get('square_defect_rate')):>10}"
            f"{_fmt(row.get('bridge_disagreement_rate')):>10}"
            f"{_fmt(row.get('mean_square_distance')):>10}"
            f"{_fmt(row.get('square_entropy')):>10}"
            f"{_fmt(row.get('segment_consistency')):>10}"
        )
    return "\n".join(lines)


def render_markdown_report(rows: list[dict], selection: dict) -> str:
    lines = [
        "# Phase 2 Commuting-Square Cocycle Pilot",
        "",
        "## Selection",
        "",
        f"- Variants: {', '.join(selection['variants'])}",
        f"- Segment protocol: {selection['num_segments']} x {selection['segment_bits']}",
        f"- Segment offset bits: {selection['segment_offset_bits']}",
        f"- Phase 1 scales: {', '.join(str(value) for value in selection['phase1_scales'])}",
        f"- Square window: {', '.join(str(value) for value in selection['window_scales'])}",
        f"- Left policy: {selection['left_policy']}",
        f"- Right policy: {selection['right_policy']}",
        "",
        "## Compact Readout",
        "",
        "| Variant | Iter | Future bridge frac | Squareable frac | Anchored rate | Square defect rate | Bridge disagreement rate | Mean square distance |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['iteration']} | "
            f"{_fmt(row.get('future_bridge_fraction'))} | { _fmt(row.get('squareable_bridge_fraction'))} | "
            f"{_fmt(row.get('anchored_square_rate'))} | { _fmt(row.get('square_defect_rate'))} | "
            f"{_fmt(row.get('bridge_disagreement_rate'))} | { _fmt(row.get('mean_square_distance'))} |"
        )

    for row in rows:
        result = row["square_result"]
        lines.extend(
            [
                "",
                f"## {row['variant']}",
                "",
                f"- Bridge count: {result['bridge_count']}",
                f"- Future bridge fraction: {_fmt(result['future_bridge_fraction'])}",
                f"- Squareable fraction: {_fmt(result['squareable_bridge_fraction'])}",
                f"- Anchored square rate: {_fmt(result['anchored_square_rate'])}",
                f"- Square defect rate: {_fmt(result['square_defect_rate'])}",
                f"- Square defect mass rate: {_fmt(result['square_defect_mass_rate'])}",
                f"- Bridge disagreement rate: {_fmt(result['bridge_disagreement_rate'])}",
                f"- Mean square distance: {_fmt(result['mean_square_distance'])}",
                f"- Square entropy: {_fmt(result['square_entropy'])}",
                f"- Segment consistency: {_fmt(result['segment_consistency'])}",
                "",
                "Top square families:",
            ]
        )
        if result["top_square_families"]:
            for family in result["top_square_families"][:5]:
                lines.append(
                    f"- `{family['family']}` | weight={_fmt(family['weight'])} | count={family['count']}"
                )
        else:
            lines.append("- none")
    return "\n".join(lines)


def _analyze_square_segment(
    bridge_items: list[dict],
    left_parent_index: dict[str, dict],
    right_parent_index: dict[str, dict],
    left_high_index: dict[str, dict],
    right_high_index: dict[str, dict],
    *,
    segment_index: int,
) -> dict:
    bridge_results = []
    families = []
    for bridge in bridge_items:
        detail = _analyze_bridge(
            bridge,
            left_parent_index,
            right_parent_index,
            left_high_index,
            right_high_index,
            segment_index=segment_index,
        )
        if detail is None:
            continue
        bridge_results.append(detail)
        if detail["square_defect_flag"] and detail["square_family"] is not None:
            families.append(detail["square_family"])
    bridge_count = len(bridge_results)
    future_bridge_count = sum(1 for item in bridge_results if item["future_bridge"])
    squareable_bridge_count = sum(1 for item in bridge_results if item["squareable"])
    anchored_square_count = sum(1 for item in bridge_results if item["anchored_square"])
    square_defect_count = sum(1 for item in bridge_results if item["square_defect_flag"])
    disagreement_count = sum(1 for item in bridge_results if item["bridge_disagreement_flag"])
    distances = [item["square_distance"] for item in bridge_results if item["future_bridge"]]
    return {
        "segment_index": segment_index,
        "bridge_count": bridge_count,
        "future_bridge_fraction": _safe_ratio(future_bridge_count, bridge_count),
        "squareable_bridge_fraction": _safe_ratio(squareable_bridge_count, bridge_count),
        "anchored_square_rate": _safe_ratio(anchored_square_count, future_bridge_count),
        "square_defect_rate": _safe_ratio(square_defect_count, future_bridge_count),
        "bridge_disagreement_rate": _safe_ratio(disagreement_count, future_bridge_count),
        "mean_square_distance": _mean(distances),
        "families": sorted(set(families)),
    }


def _analyze_bridge(
    bridge: dict,
    left_parent_index: dict[str, dict],
    right_parent_index: dict[str, dict],
    left_high_index: dict[str, dict],
    right_high_index: dict[str, dict],
    *,
    segment_index: int | None,
) -> dict | None:
    left_parent = left_parent_index.get(bridge["left_parent"])
    right_parent = right_parent_index.get(bridge["right_parent"])
    if left_parent is None or right_parent is None:
        return None

    actual_bridge = bridge["bridge_pattern"]
    left_candidate = _dominant_bridge_candidate(left_parent, segment_index=segment_index)
    right_candidate = _dominant_bridge_candidate(right_parent, segment_index=segment_index)
    bridge_weight = _bridge_weight(bridge, segment_index)

    actual_future = _bridge_future_bearing(actual_bridge, left_high_index, right_high_index, segment_index=segment_index)
    squareable = actual_future and left_candidate is not None and right_candidate is not None
    if not squareable:
        return {
            "bridge_pattern": actual_bridge,
            "left_parent": bridge["left_parent"],
            "right_parent": bridge["right_parent"],
            "bridge_weight": bridge_weight,
            "future_bridge": actual_future,
            "squareable": False,
            "anchored_square": False,
            "square_defect_flag": False,
            "bridge_disagreement_flag": False,
            "square_distance": None,
            "square_family": None,
            "left_candidate": None if left_candidate is None else left_candidate["pattern"],
            "right_candidate": None if right_candidate is None else right_candidate["pattern"],
        }

    left_pattern = left_candidate["pattern"]
    right_pattern = right_candidate["pattern"]
    anchored_square = left_pattern == actual_bridge and right_pattern == actual_bridge
    bridge_disagreement_flag = left_pattern != right_pattern
    square_defect_flag = not anchored_square
    square_distance = (
        _normalized_hamming(actual_bridge, left_pattern)
        + _normalized_hamming(actual_bridge, right_pattern)
        + _normalized_hamming(left_pattern, right_pattern)
    ) / 3.0
    square_family = None
    if square_defect_flag:
        square_family = f"{bridge['left_parent']}|{actual_bridge}|{bridge['right_parent']}=>{left_pattern}|{right_pattern}"
    return {
        "bridge_pattern": actual_bridge,
        "left_parent": bridge["left_parent"],
        "right_parent": bridge["right_parent"],
        "bridge_weight": bridge_weight,
        "future_bridge": actual_future,
        "squareable": True,
        "anchored_square": anchored_square,
        "square_defect_flag": square_defect_flag,
        "bridge_disagreement_flag": bridge_disagreement_flag,
        "square_distance": square_distance,
        "square_family": square_family,
        "left_candidate": left_pattern,
        "right_candidate": right_pattern,
    }


def _build_bridge_items(left_low: dict, right_low: dict, *, base_scale: int) -> list[dict]:
    left_child_to_parent = {}
    for parent in left_low.get("parents", []):
        for child in parent.get("children", []):
            left_child_to_parent[child["pattern"]] = parent
    right_child_to_parent = {}
    for parent in right_low.get("parents", []):
        for child in parent.get("children", []):
            right_child_to_parent[child["pattern"]] = parent

    bridge_patterns = sorted(set(left_child_to_parent) & set(right_child_to_parent))
    items = []
    for bridge_pattern in bridge_patterns:
        left_parent = left_child_to_parent[bridge_pattern]
        right_parent = right_child_to_parent[bridge_pattern]
        if left_parent["pattern"][-base_scale:] != right_parent["pattern"][:base_scale]:
            continue
        child_meta = _child_meta(left_parent, bridge_pattern)
        items.append(
            {
                "bridge_pattern": bridge_pattern,
                "left_parent": left_parent["pattern"],
                "right_parent": right_parent["pattern"],
                "raw_count": child_meta.get("raw_count", 0),
                "normalized_frequency": child_meta.get("normalized_frequency", 0.0),
                "segment_counts": child_meta.get("segment_counts", []),
            }
        )
    return items


def _dominant_bridge_candidate(parent: dict, *, segment_index: int | None) -> dict | None:
    children = parent.get("children", [])
    if not children:
        return None
    entries = []
    for child in children:
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


def _bridge_future_bearing(
    bridge_pattern: str,
    left_high_index: dict[str, dict],
    right_high_index: dict[str, dict],
    *,
    segment_index: int | None,
) -> bool:
    return _has_positive_child(left_high_index.get(bridge_pattern), segment_index) and _has_positive_child(
        right_high_index.get(bridge_pattern), segment_index
    )


def _has_positive_child(parent: dict | None, segment_index: int | None) -> bool:
    if parent is None:
        return False
    for child in parent.get("children", []):
        if _child_weight(child, segment_index) > 0:
            return True
    return False


def _child_meta(parent: dict, bridge_pattern: str) -> dict:
    for child in parent.get("children", []):
        if child["pattern"] == bridge_pattern:
            return child
    raise KeyError(bridge_pattern)


def _child_weight(child: dict, segment_index: int | None) -> float:
    return (
        float(child.get("normalized_frequency", 0.0))
        if segment_index is None
        else float(child.get("segment_counts", [])[segment_index])
    )


def _bridge_weight(bridge: dict, segment_index: int | None) -> float:
    return (
        float(bridge.get("normalized_frequency", 0.0))
        if segment_index is None
        else float(bridge.get("segment_counts", [])[segment_index])
    )


def _run_dir(run: dict):
    from pathlib import Path

    return Path(run["_run_dir"])


def _normalized_hamming(left: str, right: str) -> float:
    if len(left) != len(right):
        raise ValueError("Normalized Hamming distance requires equal-length strings.")
    if not left:
        return 0.0
    return float(sum(1 for a, b in zip(left, right) if a != b) / len(left))


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def _fmt_int(value: int | None) -> str:
    if value is None:
        return "-"
    return str(value)
