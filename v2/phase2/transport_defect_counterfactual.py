from __future__ import annotations

from pathlib import Path

from .defects import load_policy_fibers
from .transport_defect import (
    _build_transport_kernel,
    _js_divergence,
    _normalize_bin_counts,
    _ordered_flow_distance,
    _pattern_bin_counts,
    _relative_gap,
    _require_edge,
    _safe_ratio,
    _segment_defect_stability,
    _top_bin_label,
    _transport_profile,
    _weighted_jaccard,
)


def compute_counterfactual_transport_defect_result(
    low_row: dict,
    high_row: dict,
    *,
    anchor_run_dir: Path | str,
) -> dict:
    low_scale = int(low_row["pattern_scale"])
    high_scale = int(high_row["pattern_scale"])
    if high_scale <= low_scale:
        raise ValueError(
            f"Expected high scale > low scale, got {high_scale} <= {low_scale}."
        )
    if low_row["pattern_selection"] != high_row["pattern_selection"]:
        raise ValueError("Low/high return-lag rows must use the same pattern selection.")

    low_run_dir = Path(low_row["run_dir"]).resolve()
    high_run_dir = Path(high_row["run_dir"]).resolve()
    if low_run_dir != high_run_dir:
        raise ValueError(
            "Low/high return-lag rows do not point to the same evaluated run directory."
        )

    anchor_path = Path(anchor_run_dir).resolve()
    prefix_fibers = load_policy_fibers(anchor_path, "prefix")
    suffix_fibers = load_policy_fibers(anchor_path, "suffix")
    prefix_edge = _require_edge(prefix_fibers, target_m=low_scale, source_m=high_scale)
    suffix_edge = _require_edge(suffix_fibers, target_m=low_scale, source_m=high_scale)
    prefix_index = {parent["pattern"]: parent for parent in prefix_edge.get("parents", [])}
    suffix_index = {parent["pattern"]: parent for parent in suffix_edge.get("parents", [])}

    low_patterns = _index_patterns(low_row)
    high_patterns = _index_patterns(high_row)
    low_pattern_set = set(low_patterns)
    high_pattern_set = set(high_patterns)
    if not low_pattern_set:
        raise ValueError("Low-scale row does not contain any selected pattern results.")
    if not high_pattern_set:
        raise ValueError("High-scale row does not contain any selected pattern results.")

    per_pattern = []
    weight_total = 0.0
    js_sum = 0.0
    wj_sum = 0.0
    flow_sum = 0.0
    mean_gap_sum = 0.0
    top_mismatch_weight = 0.0
    coverage_sum = 0.0
    terminal_count = 0
    connected_count = 0
    zero_coverage_count = 0

    for pattern, low_item in sorted(low_patterns.items()):
        transport = _build_transport_kernel(
            pattern=pattern,
            high_pattern_set=high_pattern_set,
            prefix_parent=prefix_index.get(pattern),
            suffix_parent=suffix_index.get(pattern),
        )
        weight = max(float(low_item.get("occurrence_count", 0)), 1.0)
        low_bins = _normalize_bin_counts(_pattern_bin_counts(low_item))

        if transport["terminal_flag"]:
            terminal_count += 1
            per_pattern.append(
                {
                    "pattern": pattern,
                    "weight": weight,
                    "occurrence_count": int(low_item.get("occurrence_count", 0)),
                    "segment_occurrences": list(low_item.get("segment_occurrences", [])),
                    "kernel_candidate_count": 0,
                    "covered_candidate_count": 0,
                    "terminal_flag": True,
                    "support_coverage_mass": None,
                    "kernel": [],
                    "covered_kernel": [],
                    "observed_top_bin": _top_bin_label(low_item),
                    "transported_top_bin": None,
                    "delta_js": None,
                    "delta_wj": None,
                    "delta_flow": None,
                    "delta_top": None,
                    "delta_mean": None,
                }
            )
            continue

        connected_count += 1
        covered_kernel, support_coverage_mass = _restrict_kernel_to_covered_profiles(
            transport["kernel"],
            high_patterns,
        )
        if support_coverage_mass <= 0.0:
            zero_coverage_count += 1

        transported_profile = (
            _transport_profile(covered_kernel, high_patterns)
            if covered_kernel
            else {"bin_probs": {}, "mean_return_lag": 0.0, "top_bin": None}
        )
        delta_js = _js_divergence(low_bins, transported_profile["bin_probs"])
        delta_wj = 1.0 - _weighted_jaccard(low_bins, transported_profile["bin_probs"])
        delta_flow = _ordered_flow_distance(low_bins, transported_profile["bin_probs"])
        delta_top = _top_bin_label(low_item) != transported_profile["top_bin"]
        delta_mean = _relative_gap(
            float(low_item.get("mean_return_lag", 0.0)),
            transported_profile["mean_return_lag"],
        )

        weight_total += weight
        js_sum += weight * delta_js
        wj_sum += weight * delta_wj
        flow_sum += weight * delta_flow
        mean_gap_sum += weight * delta_mean
        coverage_sum += weight * support_coverage_mass
        if delta_top:
            top_mismatch_weight += weight

        per_pattern.append(
            {
                "pattern": pattern,
                "weight": weight,
                "occurrence_count": int(low_item.get("occurrence_count", 0)),
                "segment_occurrences": list(low_item.get("segment_occurrences", [])),
                "kernel_candidate_count": len(transport["kernel"]),
                "covered_candidate_count": len(covered_kernel),
                "terminal_flag": False,
                "support_coverage_mass": support_coverage_mass,
                "kernel": _annotate_kernel(transport["kernel"], high_patterns),
                "covered_kernel": covered_kernel,
                "observed_top_bin": _top_bin_label(low_item),
                "transported_top_bin": transported_profile["top_bin"],
                "delta_js": delta_js,
                "delta_wj": delta_wj,
                "delta_flow": delta_flow,
                "delta_top": delta_top,
                "delta_mean": delta_mean,
            }
        )

    per_pattern.sort(
        key=lambda item: (
            item["terminal_flag"],
            -(item["delta_js"] if item["delta_js"] is not None else -1.0),
            -(item["weight"]),
            item["pattern"],
        )
    )

    top_patterns = [
        {
            "pattern": item["pattern"],
            "delta_js": item["delta_js"],
            "delta_wj": item["delta_wj"],
            "delta_flow": item["delta_flow"],
            "delta_top": item["delta_top"],
            "delta_mean": item["delta_mean"],
            "support_coverage_mass": item["support_coverage_mass"],
            "weight": item["weight"],
        }
        for item in per_pattern
        if not item["terminal_flag"]
    ][:10]

    return {
        "variant": low_row["variant"],
        "iteration": low_row.get("iteration"),
        "low_scale": low_scale,
        "high_scale": high_scale,
        "pattern_selection": low_row["pattern_selection"],
        "selected_low_pattern_count": len(low_pattern_set),
        "selected_high_pattern_count": len(high_pattern_set),
        "connected_pattern_count": connected_count,
        "terminal_pattern_count": terminal_count,
        "terminal_fraction": _safe_ratio(terminal_count, len(low_pattern_set)),
        "support_coverage_mean": _safe_ratio(coverage_sum, weight_total),
        "zero_coverage_fraction": _safe_ratio(zero_coverage_count, connected_count),
        "defect_js_mean": _safe_ratio(js_sum, weight_total),
        "defect_wj_mean": _safe_ratio(wj_sum, weight_total),
        "defect_flow_mean": _safe_ratio(flow_sum, weight_total),
        "defect_top_mass": _safe_ratio(top_mismatch_weight, weight_total),
        "defect_mean_gap": _safe_ratio(mean_gap_sum, weight_total),
        "segment_defect_stability": _segment_defect_stability(per_pattern),
        "run_dir": str(low_run_dir),
        "anchor_run_dir": str(anchor_path),
        "source_kind": low_row.get("source_kind"),
        "null_model": low_row.get("null_model"),
        "null_seed": low_row.get("null_seed"),
        "phase1_policies": list(low_row.get("phase1_policies", [])),
        "low_dataset_path": low_row.get("dataset_path"),
        "high_dataset_path": high_row.get("dataset_path"),
        "top_defect_patterns": top_patterns,
        "per_pattern": per_pattern,
        "notes": [
            "transport kernel = observed anchor child support restricted to the frozen selected high-scale patterns",
            "counterfactual transported profile = weighted mixture of candidate high-scale lag-bin profiles under the observed anchor kernel",
            "support_coverage_mean = weighted mean fraction of observed kernel mass that is populated by candidate high-scale lag profiles",
            "the transported profile is renormalized over covered kernel mass; uncovered mass is reported separately through support_coverage_mean",
            "delta_js = Jensen-Shannon divergence between candidate low-scale and counterfactual transported lag-bin profiles",
            "delta_wj = 1 - weighted Jaccard between candidate low-scale and counterfactual transported lag-bin profiles",
            "delta_flow = ordered earth-mover distance on the log-lag-bin axis",
            "delta_top = dominant lag-bin mismatch indicator",
            "delta_mean = relative mean-return-lag gap",
        ],
    }


def _index_patterns(row: dict) -> dict[str, dict]:
    return {
        str(item["pattern"]): item
        for item in row["lag_result"].get("pattern_results", [])
    }


def _restrict_kernel_to_covered_profiles(
    kernel: list[dict],
    high_patterns: dict[str, dict],
) -> tuple[list[dict], float]:
    covered = []
    covered_mass = 0.0
    for entry in kernel:
        high_item = high_patterns.get(entry["pattern"])
        if high_item is None or not _has_pattern_profile(high_item):
            continue
        covered_mass += float(entry["weight"])
        covered.append(entry)

    if covered_mass <= 0.0:
        return [], 0.0

    return (
        [
            {
                **entry,
                "weight": float(entry["weight"]) / covered_mass,
            }
            for entry in covered
        ],
        covered_mass,
    )


def _annotate_kernel(kernel: list[dict], high_patterns: dict[str, dict]) -> list[dict]:
    annotated = []
    for entry in kernel:
        high_item = high_patterns.get(entry["pattern"])
        annotated.append(
            {
                **entry,
                "covered_profile": bool(high_item and _has_pattern_profile(high_item)),
                "occurrence_count": int(high_item.get("occurrence_count", 0)) if high_item else 0,
                "return_count": int(high_item.get("return_count", 0)) if high_item else 0,
            }
        )
    return annotated


def _has_pattern_profile(pattern_item: dict) -> bool:
    return bool(_pattern_bin_counts(pattern_item)) and int(pattern_item.get("return_count", 0)) > 0
