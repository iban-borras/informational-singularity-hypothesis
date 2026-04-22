from __future__ import annotations

from pathlib import Path

import numpy as np

from .defects import _require_edge, _safe_ratio, classify_variant_role, load_policy_fibers
from .transport_defect import _build_transport_kernel, _js_divergence, _weighted_jaccard
from v2.phase1.tower import rolling_codes_uint64_max64


def compute_common_support_kernel_result(
    *,
    anchor_run_dir: Path | str,
    candidate_run_dir: Path | str,
    low_scale: int,
    high_scale: int,
    pattern_selection: str,
    low_selected_patterns: list[dict],
    high_selected_patterns: list[dict],
) -> dict:
    if high_scale <= low_scale:
        raise ValueError(
            f"Expected high scale > low scale, got {high_scale} <= {low_scale}."
        )

    anchor_path = Path(anchor_run_dir).resolve()
    candidate_path = Path(candidate_run_dir).resolve()

    anchor_prefix_fibers = load_policy_fibers(anchor_path, "prefix")
    anchor_suffix_fibers = load_policy_fibers(anchor_path, "suffix")
    candidate_prefix_fibers = load_policy_fibers(candidate_path, "prefix")
    candidate_suffix_fibers = load_policy_fibers(candidate_path, "suffix")

    anchor_prefix_edge = _require_edge(
        anchor_prefix_fibers,
        target_m=low_scale,
        source_m=high_scale,
    )
    anchor_suffix_edge = _require_edge(
        anchor_suffix_fibers,
        target_m=low_scale,
        source_m=high_scale,
    )
    candidate_prefix_edge = _require_edge(
        candidate_prefix_fibers,
        target_m=low_scale,
        source_m=high_scale,
    )
    candidate_suffix_edge = _require_edge(
        candidate_suffix_fibers,
        target_m=low_scale,
        source_m=high_scale,
    )

    anchor_prefix_index = _index_edge_parents(anchor_prefix_edge)
    anchor_suffix_index = _index_edge_parents(anchor_suffix_edge)
    candidate_prefix_index = _index_edge_parents(candidate_prefix_edge)
    candidate_suffix_index = _index_edge_parents(candidate_suffix_edge)

    parent_items = [dict(item) for item in low_selected_patterns]
    child_pattern_set = {
        str(item["pattern"])
        for item in high_selected_patterns
    }
    if not parent_items:
        raise ValueError("No selected low-scale parent patterns were provided.")
    if not child_pattern_set:
        raise ValueError("No selected high-scale child patterns were provided.")

    per_pattern = []
    selected_parent_weight = 0.0
    active_anchor_weight = 0.0
    comparable_weight = 0.0

    anchor_terminal_count = 0
    candidate_terminal_count = 0
    empty_common_support_count = 0

    support_coverage_sum = 0.0
    anchor_dropout_sum = 0.0
    candidate_common_sum = 0.0
    candidate_spillover_sum = 0.0
    support_jaccard_sum = 0.0
    kernel_js_sum = 0.0
    kernel_wj_sum = 0.0
    top_child_full_match_weight = 0.0
    top_child_common_match_weight = 0.0

    for parent_item in parent_items:
        pattern = str(parent_item["pattern"])
        weight = _selection_weight(parent_item)
        selected_parent_weight += weight

        anchor_kernel = _build_transport_kernel(
            pattern=pattern,
            high_pattern_set=child_pattern_set,
            prefix_parent=anchor_prefix_index.get(pattern),
            suffix_parent=anchor_suffix_index.get(pattern),
        )
        candidate_kernel = _build_transport_kernel(
            pattern=pattern,
            high_pattern_set=child_pattern_set,
            prefix_parent=candidate_prefix_index.get(pattern),
            suffix_parent=candidate_suffix_index.get(pattern),
        )

        anchor_terminal = bool(anchor_kernel["terminal_flag"])
        candidate_terminal = bool(candidate_kernel["terminal_flag"])
        if anchor_terminal:
            anchor_terminal_count += 1
        if candidate_terminal:
            candidate_terminal_count += 1

        anchor_weights = _kernel_to_weight_map(anchor_kernel.get("kernel", []))
        candidate_weights = _kernel_to_weight_map(candidate_kernel.get("kernel", []))
        anchor_top_child = _top_child(anchor_weights)
        candidate_top_child = _top_child(candidate_weights)

        common_patterns = sorted(set(anchor_weights) & set(candidate_weights))
        union_patterns = sorted(set(anchor_weights) | set(candidate_weights))

        anchor_common_mass = sum(anchor_weights.get(item, 0.0) for item in common_patterns)
        candidate_common_mass = sum(candidate_weights.get(item, 0.0) for item in common_patterns)
        anchor_dropout_mass = max(0.0, 1.0 - anchor_common_mass) if anchor_weights else 0.0
        candidate_spillover_mass = max(0.0, 1.0 - candidate_common_mass) if candidate_weights else 0.0
        support_jaccard = _safe_ratio(len(common_patterns), len(union_patterns))

        common_anchor_weights = _restrict_and_normalize(anchor_weights, common_patterns)
        common_candidate_weights = _restrict_and_normalize(candidate_weights, common_patterns)
        common_js = None
        common_wj = None
        top_child_common_anchor = _top_child(common_anchor_weights)
        top_child_common_candidate = _top_child(common_candidate_weights)
        top_child_common_match = None
        if common_anchor_weights and common_candidate_weights:
            common_js = _js_divergence(common_anchor_weights, common_candidate_weights)
            common_wj = 1.0 - _weighted_jaccard(
                common_anchor_weights,
                common_candidate_weights,
            )
            top_child_common_match = (
                top_child_common_anchor == top_child_common_candidate
            )
        elif anchor_weights:
            empty_common_support_count += 1

        top_child_full_match = (
            anchor_top_child is not None
            and candidate_top_child is not None
            and anchor_top_child == candidate_top_child
        )

        if anchor_weights:
            active_anchor_weight += weight
            support_coverage_sum += weight * anchor_common_mass
            anchor_dropout_sum += weight * anchor_dropout_mass
            candidate_common_sum += weight * candidate_common_mass
            candidate_spillover_sum += weight * candidate_spillover_mass
            support_jaccard_sum += weight * support_jaccard
            if top_child_full_match:
                top_child_full_match_weight += weight
            if common_js is not None and common_wj is not None:
                comparable_weight += weight
                kernel_js_sum += weight * common_js
                kernel_wj_sum += weight * common_wj
                if top_child_common_match:
                    top_child_common_match_weight += weight

        per_pattern.append(
            {
                "pattern": pattern,
                "weight": weight,
                "selection_raw_count": int(parent_item.get("raw_count", 0)),
                "selection_normalized_frequency": float(
                    parent_item.get("normalized_frequency", 0.0)
                ),
                "anchor_terminal_flag": anchor_terminal,
                "candidate_terminal_flag": candidate_terminal,
                "anchor_child_count": len(anchor_weights),
                "candidate_child_count": len(candidate_weights),
                "common_child_count": len(common_patterns),
                "union_child_count": len(union_patterns),
                "anchor_common_mass": anchor_common_mass,
                "candidate_common_mass": candidate_common_mass,
                "anchor_dropout_mass": anchor_dropout_mass,
                "candidate_spillover_mass": candidate_spillover_mass,
                "support_jaccard": support_jaccard,
                "common_js": common_js,
                "common_wj": common_wj,
                "top_child_full_match": top_child_full_match,
                "top_child_common_match": top_child_common_match,
                "anchor_top_child": anchor_top_child,
                "candidate_top_child": candidate_top_child,
                "anchor_common_top_child": top_child_common_anchor,
                "candidate_common_top_child": top_child_common_candidate,
                "anchor_kernel_preview": _kernel_preview(anchor_kernel.get("kernel", [])),
                "candidate_kernel_preview": _kernel_preview(
                    candidate_kernel.get("kernel", [])
                ),
            }
        )

    ranked_patterns = sorted(
        per_pattern,
        key=lambda item: (
            -(item["common_js"] if item["common_js"] is not None else -1.0),
            -(item["candidate_spillover_mass"]),
            -(item["weight"]),
            item["pattern"],
        ),
    )
    top_divergent_patterns = [
        {
            "pattern": item["pattern"],
            "weight": item["weight"],
            "anchor_common_mass": item["anchor_common_mass"],
            "candidate_spillover_mass": item["candidate_spillover_mass"],
            "support_jaccard": item["support_jaccard"],
            "common_js": item["common_js"],
            "common_wj": item["common_wj"],
            "top_child_full_match": item["top_child_full_match"],
            "top_child_common_match": item["top_child_common_match"],
        }
        for item in ranked_patterns
        if item["anchor_child_count"] > 0
    ][:10]

    return {
        "variant": _variant_from_run_dir(candidate_path),
        "variant_role": classify_variant_role(_variant_from_run_dir(candidate_path)),
        "anchor_run_dir": str(anchor_path),
        "candidate_run_dir": str(candidate_path),
        "low_scale": low_scale,
        "high_scale": high_scale,
        "pattern_selection": pattern_selection,
        "selected_parent_count": len(parent_items),
        "selected_child_count": len(child_pattern_set),
        "anchor_terminal_count": anchor_terminal_count,
        "anchor_terminal_fraction": _safe_ratio(anchor_terminal_count, len(parent_items)),
        "candidate_terminal_count": candidate_terminal_count,
        "candidate_terminal_fraction": _safe_ratio(candidate_terminal_count, len(parent_items)),
        "empty_common_support_count": empty_common_support_count,
        "empty_common_support_fraction": _safe_ratio(
            empty_common_support_count,
            max(len(parent_items) - anchor_terminal_count, 0),
        ),
        "active_anchor_parent_fraction": _safe_ratio(
            len(parent_items) - anchor_terminal_count,
            len(parent_items),
        ),
        "comparable_weight_fraction": _safe_ratio(comparable_weight, active_anchor_weight),
        "support_coverage_anchor_mean": _safe_ratio(
            support_coverage_sum,
            active_anchor_weight,
        ),
        "anchor_dropout_mean": _safe_ratio(anchor_dropout_sum, active_anchor_weight),
        "candidate_common_mass_mean": _safe_ratio(candidate_common_sum, active_anchor_weight),
        "candidate_spillover_mean": _safe_ratio(
            candidate_spillover_sum,
            active_anchor_weight,
        ),
        "support_jaccard_mean": _safe_ratio(support_jaccard_sum, active_anchor_weight),
        "kernel_js_mean": _ratio_or_none(kernel_js_sum, comparable_weight),
        "kernel_wj_mean": _ratio_or_none(kernel_wj_sum, comparable_weight),
        "top_child_full_match_mass": _safe_ratio(
            top_child_full_match_weight,
            active_anchor_weight,
        ),
        "top_child_common_match_mass": _ratio_or_none(
            top_child_common_match_weight,
            comparable_weight,
        ),
        "selected_parent_weight_total": selected_parent_weight,
        "active_anchor_weight_total": active_anchor_weight,
        "comparable_weight_total": comparable_weight,
        "top_divergent_patterns": top_divergent_patterns,
        "per_pattern": ranked_patterns,
        "order_axis_defined": False,
        "notes": [
            "The object compares the anchor and candidate transport kernels directly on their shared child support.",
            "support_coverage_anchor_mean = weighted anchor mass that survives on shared support.",
            "candidate_spillover_mean = weighted candidate mass that falls outside the anchor support.",
            "kernel_js_mean = Jensen-Shannon divergence on renormalized shared support only.",
            "kernel_wj_mean = 1 - weighted Jaccard on renormalized shared support only.",
            "No flow-like metric is reported yet because a canonical ordering of child patterns has not been fixed.",
        ],
    }


def compute_window_local_common_support_kernel_result(
    *,
    anchor_run_dir: Path | str,
    candidate_run_dir: Path | str,
    anchor_window_bits: np.ndarray,
    candidate_window_bits: np.ndarray,
    low_scale: int,
    high_scale: int,
    pattern_selection: str,
    low_selected_patterns: list[dict],
    high_selected_patterns: list[dict],
) -> dict:
    if high_scale <= low_scale:
        raise ValueError(
            f"Expected high scale > low scale, got {high_scale} <= {low_scale}."
        )
    if int(anchor_window_bits.size) < high_scale or int(candidate_window_bits.size) < high_scale:
        raise ValueError("Window bits are shorter than the requested high scale.")

    anchor_path = Path(anchor_run_dir).resolve()
    candidate_path = Path(candidate_run_dir).resolve()

    anchor_prefix_fibers = load_policy_fibers(anchor_path, "prefix")
    anchor_suffix_fibers = load_policy_fibers(anchor_path, "suffix")
    candidate_prefix_fibers = load_policy_fibers(candidate_path, "prefix")
    candidate_suffix_fibers = load_policy_fibers(candidate_path, "suffix")

    anchor_prefix_edge = _require_edge(
        anchor_prefix_fibers,
        target_m=low_scale,
        source_m=high_scale,
    )
    anchor_suffix_edge = _require_edge(
        anchor_suffix_fibers,
        target_m=low_scale,
        source_m=high_scale,
    )
    candidate_prefix_edge = _require_edge(
        candidate_prefix_fibers,
        target_m=low_scale,
        source_m=high_scale,
    )
    candidate_suffix_edge = _require_edge(
        candidate_suffix_fibers,
        target_m=low_scale,
        source_m=high_scale,
    )

    anchor_prefix_index = _index_edge_parents(anchor_prefix_edge)
    anchor_suffix_index = _index_edge_parents(anchor_suffix_edge)
    candidate_prefix_index = _index_edge_parents(candidate_prefix_edge)
    candidate_suffix_index = _index_edge_parents(candidate_suffix_edge)

    parent_items = [dict(item) for item in low_selected_patterns]
    child_pattern_set = {
        str(item["pattern"])
        for item in high_selected_patterns
    }
    if not parent_items:
        raise ValueError("No selected low-scale parent patterns were provided.")
    if not child_pattern_set:
        raise ValueError("No selected high-scale child patterns were provided.")

    anchor_parent_structure = _build_window_parent_structure(
        parent_items=parent_items,
        high_pattern_set=child_pattern_set,
        prefix_index=anchor_prefix_index,
        suffix_index=anchor_suffix_index,
    )
    candidate_parent_structure = _build_window_parent_structure(
        parent_items=parent_items,
        high_pattern_set=child_pattern_set,
        prefix_index=candidate_prefix_index,
        suffix_index=candidate_suffix_index,
    )

    anchor_low_counts = _window_pattern_count_map(
        anchor_window_bits,
        low_selected_patterns,
        low_scale,
    )
    candidate_low_counts = _window_pattern_count_map(
        candidate_window_bits,
        low_selected_patterns,
        low_scale,
    )
    anchor_high_counts = _window_pattern_count_map(
        anchor_window_bits,
        high_selected_patterns,
        high_scale,
    )
    candidate_high_counts = _window_pattern_count_map(
        candidate_window_bits,
        high_selected_patterns,
        high_scale,
    )

    anchor_kernels = _build_window_local_event_kernels(
        parent_items=parent_items,
        parent_structure=anchor_parent_structure,
        local_high_counts=anchor_high_counts,
    )
    candidate_kernels = _build_window_local_event_kernels(
        parent_items=parent_items,
        parent_structure=candidate_parent_structure,
        local_high_counts=candidate_high_counts,
    )

    per_pattern = []
    selected_parent_weight = 0.0
    active_anchor_weight = 0.0
    comparable_weight = 0.0

    anchor_terminal_count = 0
    candidate_terminal_count = 0
    anchor_local_parent_active_count = 0
    candidate_local_parent_active_count = 0
    anchor_local_kernel_nonempty_count = 0
    candidate_local_kernel_nonempty_count = 0
    empty_common_support_count = 0

    support_coverage_sum = 0.0
    anchor_dropout_sum = 0.0
    candidate_common_sum = 0.0
    candidate_spillover_sum = 0.0
    support_jaccard_sum = 0.0
    kernel_js_sum = 0.0
    kernel_wj_sum = 0.0
    top_child_full_match_weight = 0.0
    top_child_common_match_weight = 0.0

    for parent_item in parent_items:
        pattern = str(parent_item["pattern"])
        anchor_local_parent_count = int(anchor_low_counts.get(pattern, 0))
        candidate_local_parent_count = int(candidate_low_counts.get(pattern, 0))
        anchor_kernel = anchor_kernels[pattern]
        candidate_kernel = candidate_kernels[pattern]
        anchor_local_event_total = int(anchor_kernel["event_total"])
        candidate_local_event_total = int(candidate_kernel["event_total"])
        weight = float(anchor_local_event_total)
        selected_parent_weight += weight

        if anchor_kernel["structural_terminal_flag"]:
            anchor_terminal_count += 1
        if candidate_kernel["structural_terminal_flag"]:
            candidate_terminal_count += 1
        if anchor_local_parent_count > 0:
            anchor_local_parent_active_count += 1
        if candidate_local_parent_count > 0:
            candidate_local_parent_active_count += 1
        if anchor_local_event_total > 0:
            anchor_local_kernel_nonempty_count += 1
        if candidate_local_event_total > 0:
            candidate_local_kernel_nonempty_count += 1

        anchor_weights = anchor_kernel["weights"]
        candidate_weights = (
            candidate_kernel["weights"]
            if candidate_local_event_total > 0
            else {}
        )
        anchor_top_child = _top_child(anchor_weights)
        candidate_top_child = _top_child(candidate_weights)

        common_patterns = sorted(set(anchor_weights) & set(candidate_weights))
        union_patterns = sorted(set(anchor_weights) | set(candidate_weights))

        anchor_common_mass = sum(anchor_weights.get(item, 0.0) for item in common_patterns)
        candidate_common_mass = sum(candidate_weights.get(item, 0.0) for item in common_patterns)
        anchor_dropout_mass = max(0.0, 1.0 - anchor_common_mass) if anchor_weights else 0.0
        candidate_spillover_mass = max(0.0, 1.0 - candidate_common_mass) if candidate_weights else 0.0
        support_jaccard = _safe_ratio(len(common_patterns), len(union_patterns))

        common_anchor_weights = _restrict_and_normalize(anchor_weights, common_patterns)
        common_candidate_weights = _restrict_and_normalize(candidate_weights, common_patterns)
        common_js = None
        common_wj = None
        top_child_common_anchor = _top_child(common_anchor_weights)
        top_child_common_candidate = _top_child(common_candidate_weights)
        top_child_common_match = None
        if common_anchor_weights and common_candidate_weights:
            common_js = _js_divergence(common_anchor_weights, common_candidate_weights)
            common_wj = 1.0 - _weighted_jaccard(
                common_anchor_weights,
                common_candidate_weights,
            )
            top_child_common_match = (
                top_child_common_anchor == top_child_common_candidate
            )
        elif anchor_weights:
            empty_common_support_count += 1

        top_child_full_match = (
            anchor_top_child is not None
            and candidate_top_child is not None
            and anchor_top_child == candidate_top_child
        )

        if weight > 0.0:
            active_anchor_weight += weight
            support_coverage_sum += weight * anchor_common_mass
            anchor_dropout_sum += weight * anchor_dropout_mass
            candidate_common_sum += weight * candidate_common_mass
            candidate_spillover_sum += weight * candidate_spillover_mass
            support_jaccard_sum += weight * support_jaccard
            if top_child_full_match:
                top_child_full_match_weight += weight
            if common_js is not None and common_wj is not None:
                comparable_weight += weight
                kernel_js_sum += weight * common_js
                kernel_wj_sum += weight * common_wj
                if top_child_common_match:
                    top_child_common_match_weight += weight

        per_pattern.append(
            {
                "pattern": pattern,
                "weight": weight,
                "selection_raw_count": int(parent_item.get("raw_count", 0)),
                "selection_normalized_frequency": float(
                    parent_item.get("normalized_frequency", 0.0)
                ),
                "anchor_local_parent_count": anchor_local_parent_count,
                "candidate_local_parent_count": candidate_local_parent_count,
                "anchor_local_event_total": anchor_local_event_total,
                "candidate_local_event_total": candidate_local_event_total,
                "anchor_terminal_flag": anchor_kernel["structural_terminal_flag"],
                "candidate_terminal_flag": candidate_kernel["structural_terminal_flag"],
                "anchor_local_empty_flag": anchor_kernel["local_empty_flag"],
                "candidate_local_empty_flag": candidate_kernel["local_empty_flag"],
                "anchor_child_count": anchor_kernel["structural_child_count"],
                "candidate_child_count": candidate_kernel["structural_child_count"],
                "anchor_local_child_count": len(anchor_weights),
                "candidate_local_child_count": len(candidate_weights),
                "common_child_count": len(common_patterns),
                "union_child_count": len(union_patterns),
                "anchor_common_mass": anchor_common_mass,
                "candidate_common_mass": candidate_common_mass,
                "anchor_dropout_mass": anchor_dropout_mass,
                "candidate_spillover_mass": candidate_spillover_mass,
                "support_jaccard": support_jaccard,
                "common_js": common_js,
                "common_wj": common_wj,
                "top_child_full_match": top_child_full_match,
                "top_child_common_match": top_child_common_match,
                "anchor_top_child": anchor_top_child,
                "candidate_top_child": candidate_top_child,
                "anchor_common_top_child": top_child_common_anchor,
                "candidate_common_top_child": top_child_common_candidate,
                "anchor_kernel_preview": anchor_kernel["preview"],
                "candidate_kernel_preview": candidate_kernel["preview"],
            }
        )

    ranked_patterns = sorted(
        per_pattern,
        key=lambda item: (
            -(item["common_js"] if item["common_js"] is not None else -1.0),
            -(item["candidate_spillover_mass"]),
            -(item["weight"]),
            item["pattern"],
        ),
    )
    top_divergent_patterns = [
        {
            "pattern": item["pattern"],
            "weight": item["weight"],
            "anchor_local_parent_count": item["anchor_local_parent_count"],
            "candidate_local_parent_count": item["candidate_local_parent_count"],
            "anchor_local_event_total": item["anchor_local_event_total"],
            "candidate_local_event_total": item["candidate_local_event_total"],
            "anchor_common_mass": item["anchor_common_mass"],
            "candidate_spillover_mass": item["candidate_spillover_mass"],
            "support_jaccard": item["support_jaccard"],
            "common_js": item["common_js"],
            "common_wj": item["common_wj"],
            "top_child_full_match": item["top_child_full_match"],
            "top_child_common_match": item["top_child_common_match"],
        }
        for item in ranked_patterns
        if item["anchor_local_event_total"] > 0
    ][:10]

    return {
        "variant": _variant_from_run_dir(candidate_path),
        "variant_role": classify_variant_role(_variant_from_run_dir(candidate_path)),
        "anchor_run_dir": str(anchor_path),
        "candidate_run_dir": str(candidate_path),
        "low_scale": low_scale,
        "high_scale": high_scale,
        "pattern_selection": pattern_selection,
        "selected_parent_count": len(parent_items),
        "selected_child_count": len(child_pattern_set),
        "anchor_terminal_count": anchor_terminal_count,
        "anchor_terminal_fraction": _safe_ratio(anchor_terminal_count, len(parent_items)),
        "candidate_terminal_count": candidate_terminal_count,
        "candidate_terminal_fraction": _safe_ratio(candidate_terminal_count, len(parent_items)),
        "anchor_local_parent_active_count": anchor_local_parent_active_count,
        "anchor_local_parent_fraction": _safe_ratio(
            anchor_local_parent_active_count,
            len(parent_items),
        ),
        "candidate_local_parent_active_count": candidate_local_parent_active_count,
        "candidate_local_parent_fraction": _safe_ratio(
            candidate_local_parent_active_count,
            len(parent_items),
        ),
        "anchor_local_kernel_nonempty_count": anchor_local_kernel_nonempty_count,
        "anchor_local_kernel_nonempty_fraction": _safe_ratio(
            anchor_local_kernel_nonempty_count,
            max(anchor_local_parent_active_count, 1),
        ),
        "candidate_local_kernel_nonempty_count": candidate_local_kernel_nonempty_count,
        "candidate_local_kernel_nonempty_fraction": _safe_ratio(
            candidate_local_kernel_nonempty_count,
            max(candidate_local_parent_active_count, 1),
        ),
        "empty_common_support_count": empty_common_support_count,
        "empty_common_support_fraction": _safe_ratio(
            empty_common_support_count,
            max(anchor_local_parent_active_count, 0),
        ),
        "active_anchor_parent_fraction": _safe_ratio(
            anchor_local_parent_active_count,
            len(parent_items),
        ),
        "comparable_weight_fraction": _safe_ratio(comparable_weight, active_anchor_weight),
        "support_coverage_anchor_mean": _safe_ratio(
            support_coverage_sum,
            active_anchor_weight,
        ),
        "anchor_dropout_mean": _safe_ratio(anchor_dropout_sum, active_anchor_weight),
        "candidate_common_mass_mean": _safe_ratio(candidate_common_sum, active_anchor_weight),
        "candidate_spillover_mean": _safe_ratio(
            candidate_spillover_sum,
            active_anchor_weight,
        ),
        "support_jaccard_mean": _safe_ratio(support_jaccard_sum, active_anchor_weight),
        "kernel_js_mean": _ratio_or_none(kernel_js_sum, comparable_weight),
        "kernel_wj_mean": _ratio_or_none(kernel_wj_sum, comparable_weight),
        "top_child_full_match_mass": _safe_ratio(
            top_child_full_match_weight,
            active_anchor_weight,
        ),
        "top_child_common_match_mass": _ratio_or_none(
            top_child_common_match_weight,
            comparable_weight,
        ),
        "selected_parent_weight_total": selected_parent_weight,
        "active_anchor_weight_total": active_anchor_weight,
        "comparable_weight_total": comparable_weight,
        "window_bit_count": int(anchor_window_bits.size),
        "window_local_flag": True,
        "top_divergent_patterns": top_divergent_patterns,
        "per_pattern": ranked_patterns,
        "order_axis_defined": False,
        "notes": [
            "The object compares locally rebuilt anchor and candidate transport kernels on shared child support inside a fixed late window.",
            "Kernel weights are reconstructed from realized child-anchored events inside the window, not from global Phase 1 fiber masses.",
            "Each child occurrence contributes at most once per parent-child pair, even if the same pair is admissible through both prefix and suffix.",
            "Parent weighting uses the anchor local realized event mass on the frozen child universe.",
            "support_coverage_anchor_mean = weighted anchor local mass that survives on shared support.",
            "candidate_spillover_mean = weighted candidate local mass that falls outside the anchor support.",
            "kernel_js_mean = Jensen-Shannon divergence on renormalized shared support only.",
            "kernel_wj_mean = 1 - weighted Jaccard on renormalized shared support only.",
            "No flow-like metric is reported yet because a canonical ordering of child patterns has not been fixed.",
        ],
    }


def _index_edge_parents(edge: dict) -> dict[str, dict]:
    return {
        str(parent["pattern"]): parent
        for parent in edge.get("parents", [])
    }


def _kernel_to_weight_map(kernel: list[dict]) -> dict[str, float]:
    return {
        str(item["pattern"]): float(item["weight"])
        for item in kernel
        if float(item.get("weight", 0.0)) > 0.0
    }


def _window_pattern_count_map(
    window_bits: np.ndarray,
    selected_patterns: list[dict],
    pattern_scale: int,
) -> dict[str, int]:
    if int(window_bits.size) < pattern_scale:
        return {
            str(item["pattern"]): 0
            for item in selected_patterns
        }
    codes = rolling_codes_uint64_max64(window_bits, pattern_scale)
    unique_codes, unique_counts = np.unique(codes, return_counts=True)
    code_count_map = {
        int(code): int(count)
        for code, count in zip(unique_codes.tolist(), unique_counts.tolist())
    }
    return {
        str(item["pattern"]): int(code_count_map.get(int(item["code"]), 0))
        for item in selected_patterns
    }


def _build_window_parent_structure(
    *,
    parent_items: list[dict],
    high_pattern_set: set[str],
    prefix_index: dict[str, dict],
    suffix_index: dict[str, dict],
) -> dict[str, dict]:
    parent_structure = {
        str(item["pattern"]): {
            "structural_child_patterns": set(),
            "child_policies": {},
        }
        for item in parent_items
    }
    for policy, parent_index in (("prefix", prefix_index), ("suffix", suffix_index)):
        for parent_pattern in parent_structure:
            parent = parent_index.get(parent_pattern)
            if parent is None:
                continue
            for child in parent.get("children", []):
                child_pattern = str(child["pattern"])
                if child_pattern not in high_pattern_set:
                    continue
                parent_entry = parent_structure[parent_pattern]
                parent_entry["structural_child_patterns"].add(child_pattern)
                parent_entry["child_policies"].setdefault(child_pattern, set()).add(policy)
    return parent_structure


def _build_window_local_event_kernels(
    *,
    parent_items: list[dict],
    parent_structure: dict[str, dict],
    local_high_counts: dict[str, int],
) -> dict[str, dict]:
    child_to_parents: dict[str, dict[str, set[str]]] = {}
    for parent_pattern, parent_entry in parent_structure.items():
        for child_pattern, policies in parent_entry["child_policies"].items():
            child_to_parents.setdefault(child_pattern, {})[parent_pattern] = set(policies)

    parent_scores = {
        str(item["pattern"]): {}
        for item in parent_items
    }
    for child_pattern, child_count in local_high_counts.items():
        if int(child_count) <= 0:
            continue
        for parent_pattern, policies in child_to_parents.get(child_pattern, {}).items():
            entry = parent_scores[parent_pattern].setdefault(
                child_pattern,
                {
                    "pattern": child_pattern,
                    "score": 0.0,
                    "event_count": 0,
                    "policies": set(),
                },
            )
            entry["score"] += float(child_count)
            entry["event_count"] += int(child_count)
            entry["policies"].update(policies)

    kernels = {}
    for parent_pattern, parent_entry in parent_structure.items():
        structural_child_patterns = parent_entry["structural_child_patterns"]
        candidate_scores = parent_scores[parent_pattern]
        if not structural_child_patterns:
            kernels[parent_pattern] = {
                "structural_terminal_flag": True,
                "local_empty_flag": True,
                "structural_child_count": 0,
                "event_total": 0,
                "weights": {},
                "preview": [],
            }
            continue

        if not candidate_scores:
            kernels[parent_pattern] = {
                "structural_terminal_flag": False,
                "local_empty_flag": True,
                "structural_child_count": len(structural_child_patterns),
                "event_total": 0,
                "weights": {},
                "preview": [],
            }
            continue

        total = sum(max(item["score"], 0.0) for item in candidate_scores.values())
        if total <= 0.0:
            kernels[parent_pattern] = {
                "structural_terminal_flag": False,
                "local_empty_flag": True,
                "structural_child_count": len(structural_child_patterns),
                "event_total": 0,
                "weights": {},
                "preview": [],
            }
            continue

        preview = [
            {
                "pattern": item["pattern"],
                "weight": item["score"] / total,
                "event_count": int(item["event_count"]),
                "policies": sorted(item["policies"]),
            }
            for item in sorted(
                candidate_scores.values(),
                key=lambda item: (-item["score"], -len(item["policies"]), item["pattern"]),
            )
        ]
        kernels[parent_pattern] = {
            "structural_terminal_flag": False,
            "local_empty_flag": False,
            "structural_child_count": len(structural_child_patterns),
            "event_total": int(total),
            "weights": {
                str(item["pattern"]): float(item["weight"])
                for item in preview
            },
            "preview": _kernel_preview(preview),
        }
    return kernels


def _restrict_and_normalize(
    weight_map: dict[str, float],
    patterns: list[str],
) -> dict[str, float]:
    restricted = {
        pattern: float(weight_map[pattern])
        for pattern in patterns
        if float(weight_map.get(pattern, 0.0)) > 0.0
    }
    total = sum(restricted.values())
    if total <= 0.0:
        return {}
    return {
        pattern: value / total
        for pattern, value in restricted.items()
    }


def _selection_weight(pattern_item: dict) -> float:
    weight = float(pattern_item.get("normalized_frequency", 0.0))
    if weight > 0.0:
        return weight
    raw_count = float(pattern_item.get("raw_count", 0.0))
    if raw_count > 0.0:
        return raw_count
    return 1.0


def _top_child(weight_map: dict[str, float]) -> str | None:
    if not weight_map:
        return None
    return min(
        (
            (-float(weight), str(pattern))
            for pattern, weight in weight_map.items()
        )
    )[1]


def _kernel_preview(kernel: list[dict], *, limit: int = 8) -> list[dict]:
    return [
        {
            "pattern": str(item["pattern"]),
            "weight": float(item["weight"]),
            "policies": list(item.get("policies", [])),
        }
        for item in kernel[:limit]
    ]


def _ratio_or_none(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    return float(numerator / denominator)


def _variant_from_run_dir(run_dir: Path) -> str:
    dataset_path = run_dir / "dataset.json"
    if not dataset_path.exists():
        return run_dir.name
    import json

    with open(dataset_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    config = payload.get("dataset", {}).get("config")
    if config is None:
        config = payload.get("config", {})
    return str(config.get("variant", run_dir.name))
