from __future__ import annotations

import json
import math
from pathlib import Path


def load_return_lag_row(dataset_path: Path, *, variant: str) -> dict:
    with open(dataset_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("rows", [])
    for row in rows:
        if str(row.get("variant")) == variant:
            return row
    raise ValueError(f"Variant {variant!r} not found in {dataset_path}")


def compare_return_profiles(observed_row: dict, null_row: dict) -> dict:
    observed = _index_patterns(observed_row)
    null = _index_patterns(null_row)

    observed_patterns = set(observed)
    null_patterns = set(null)
    shared_patterns = sorted(observed_patterns & null_patterns)
    union_patterns = sorted(observed_patterns | null_patterns)

    per_pattern = []
    total_weight = 0.0
    weighted_jaccard_sum = 0.0
    js_divergence_sum = 0.0
    relative_mean_gap_sum = 0.0
    relative_median_gap_sum = 0.0
    top_bin_match_weight = 0.0
    long_lag_match_weight = 0.0

    for pattern in shared_patterns:
        observed_item = observed[pattern]
        null_item = null[pattern]
        weight = float(min(observed_item["occurrence_count"], null_item["occurrence_count"]))
        weight = max(weight, 1.0)
        observed_bins = _bin_counts_from_pattern(observed_item)
        null_bins = _bin_counts_from_pattern(null_item)
        profile_jaccard = _weighted_jaccard(observed_bins, null_bins)
        js_divergence = _js_divergence(observed_bins, null_bins)
        mean_gap = _relative_gap(
            float(observed_item.get("mean_return_lag", 0.0)),
            float(null_item.get("mean_return_lag", 0.0)),
        )
        median_gap = _relative_gap(
            float(observed_item.get("median_return_lag", 0.0)),
            float(null_item.get("median_return_lag", 0.0)),
        )
        observed_top = _top_bin_label(observed_item)
        null_top = _top_bin_label(null_item)
        top_bin_match = observed_top == null_top
        long_lag_match = math.isclose(
            float(observed_item.get("long_lag_fraction", 0.0)),
            float(null_item.get("long_lag_fraction", 0.0)),
            rel_tol=0.0,
            abs_tol=1e-12,
        )

        total_weight += weight
        weighted_jaccard_sum += weight * profile_jaccard
        js_divergence_sum += weight * js_divergence
        relative_mean_gap_sum += weight * mean_gap
        relative_median_gap_sum += weight * median_gap
        if top_bin_match:
            top_bin_match_weight += weight
        if long_lag_match:
            long_lag_match_weight += weight

        per_pattern.append(
            {
                "pattern": pattern,
                "weight": weight,
                "observed_occurrence_count": observed_item["occurrence_count"],
                "null_occurrence_count": null_item["occurrence_count"],
                "profile_weighted_jaccard": profile_jaccard,
                "profile_js_divergence": js_divergence,
                "relative_mean_lag_gap": mean_gap,
                "relative_median_lag_gap": median_gap,
                "observed_top_bin": observed_top,
                "null_top_bin": null_top,
                "top_bin_match": top_bin_match,
                "long_lag_fraction_match": long_lag_match,
            }
        )

    per_pattern.sort(
        key=lambda item: (
            -item["profile_js_divergence"],
            item["profile_weighted_jaccard"],
            -item["relative_mean_lag_gap"],
            item["pattern"],
        )
    )

    return {
        "observed_variant": observed_row["variant"],
        "null_variant": null_row["variant"],
        "pattern_scale": observed_row["pattern_scale"],
        "pattern_selection": observed_row["pattern_selection"],
        "shared_pattern_count": len(shared_patterns),
        "union_pattern_count": len(union_patterns),
        "selection_jaccard": _safe_ratio(len(shared_patterns), len(union_patterns)),
        "weighted_profile_jaccard": _safe_ratio(weighted_jaccard_sum, total_weight),
        "weighted_profile_js_divergence": _safe_ratio(js_divergence_sum, total_weight),
        "weighted_relative_mean_lag_gap": _safe_ratio(relative_mean_gap_sum, total_weight),
        "weighted_relative_median_lag_gap": _safe_ratio(relative_median_gap_sum, total_weight),
        "weighted_top_bin_match_fraction": _safe_ratio(top_bin_match_weight, total_weight),
        "weighted_long_lag_match_fraction": _safe_ratio(long_lag_match_weight, total_weight),
        "observed_only_patterns": sorted(observed_patterns - null_patterns),
        "null_only_patterns": sorted(null_patterns - observed_patterns),
        "per_pattern": per_pattern,
        "notes": [
            "profile_weighted_jaccard compares the full lag-bin mass profile pattern-by-pattern",
            "profile_js_divergence is Jensen-Shannon divergence on normalized lag-bin counts",
            "relative mean/median lag gaps are normalized by the larger of the two pattern means/medians",
            "weighted fractions use min(observed_occurrence_count, null_occurrence_count) as the pattern weight",
        ],
    }


def render_profile_comparison_report(comparison: dict) -> str:
    lines = [
        "# Phase 2 Return-Profile Divergence",
        "",
        f"- Observed variant: {comparison['observed_variant']}",
        f"- Null variant: {comparison['null_variant']}",
        f"- Pattern scale: {comparison['pattern_scale']}",
        f"- Pattern selection: {comparison['pattern_selection']}",
        f"- Shared patterns: {comparison['shared_pattern_count']}/{comparison['union_pattern_count']}",
        f"- Selection Jaccard: {_fmt(comparison['selection_jaccard'])}",
        f"- Weighted profile Jaccard: {_fmt(comparison['weighted_profile_jaccard'])}",
        f"- Weighted profile JS divergence: {_fmt(comparison['weighted_profile_js_divergence'])}",
        f"- Weighted relative mean-lag gap: {_fmt(comparison['weighted_relative_mean_lag_gap'])}",
        f"- Weighted relative median-lag gap: {_fmt(comparison['weighted_relative_median_lag_gap'])}",
        f"- Weighted top-bin match fraction: {_fmt(comparison['weighted_top_bin_match_fraction'])}",
        f"- Weighted long-lag match fraction: {_fmt(comparison['weighted_long_lag_match_fraction'])}",
        "",
        "Observed-only patterns:",
        f"- {', '.join(comparison['observed_only_patterns']) if comparison['observed_only_patterns'] else 'none'}",
        "Null-only patterns:",
        f"- {', '.join(comparison['null_only_patterns']) if comparison['null_only_patterns'] else 'none'}",
        "",
        "Top divergent shared patterns:",
    ]
    for item in comparison["per_pattern"][:10]:
        lines.append(
            f"- `{item['pattern']}` | JS={_fmt(item['profile_js_divergence'])} | "
            f"wJ={_fmt(item['profile_weighted_jaccard'])} | "
            f"d_mean={_fmt(item['relative_mean_lag_gap'])} | "
            f"top={item['observed_top_bin']} vs {item['null_top_bin']}"
        )
    return "\n".join(lines)


def _index_patterns(row: dict) -> dict[str, dict]:
    return {
        str(item["pattern"]): item
        for item in row["lag_result"].get("pattern_results", [])
    }


def _bin_counts_from_pattern(pattern_item: dict) -> dict[str, int]:
    lag_bin_counts = pattern_item.get("lag_bin_counts")
    if isinstance(lag_bin_counts, dict) and lag_bin_counts:
        return {str(key): int(value) for key, value in lag_bin_counts.items()}

    bins = {}
    for item in pattern_item.get("top_return_bins", []):
        bins[str(item["bin"])] = int(item["count"])
    return bins


def _weighted_jaccard(left: dict[str, int], right: dict[str, int]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 1.0
    shared = sum(min(left.get(key, 0), right.get(key, 0)) for key in keys)
    union = sum(max(left.get(key, 0), right.get(key, 0)) for key in keys)
    return _safe_ratio(shared, union)


def _js_divergence(left: dict[str, int], right: dict[str, int]) -> float:
    keys = sorted(set(left) | set(right))
    if not keys:
        return 0.0
    left_total = float(sum(left.values()))
    right_total = float(sum(right.values()))
    if left_total <= 0.0 or right_total <= 0.0:
        return 0.0
    left_probs = [left.get(key, 0) / left_total for key in keys]
    right_probs = [right.get(key, 0) / right_total for key in keys]
    midpoint = [(lp + rp) / 2.0 for lp, rp in zip(left_probs, right_probs)]
    return (_kl_divergence(left_probs, midpoint) + _kl_divergence(right_probs, midpoint)) / 2.0


def _kl_divergence(left_probs: list[float], right_probs: list[float]) -> float:
    total = 0.0
    for left_value, right_value in zip(left_probs, right_probs):
        if left_value <= 0.0 or right_value <= 0.0:
            continue
        total += left_value * math.log(left_value / right_value, 2)
    return total


def _relative_gap(left: float, right: float) -> float:
    scale = max(abs(left), abs(right), 1.0)
    return abs(left - right) / scale


def _top_bin_label(pattern_item: dict) -> str | None:
    top_bins = pattern_item.get("top_return_bins", [])
    if not top_bins:
        return None
    return str(top_bins[0]["bin"])


def _safe_ratio(numerator: float | int, denominator: float | int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"
