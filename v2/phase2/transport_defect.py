from __future__ import annotations

import json
import math
from pathlib import Path

from .defects import _require_edge, classify_variant_role, load_policy_fibers
from .return_profile_compare import load_return_lag_row


def build_transport_defect_rows(
    *,
    low_dataset_path: Path,
    high_dataset_path: Path,
    variants: list[str],
) -> list[dict]:
    rows = []
    for variant in variants:
        low_row = _load_return_lag_row_casefold(low_dataset_path, variant=variant)
        high_row = _load_return_lag_row_casefold(high_dataset_path, variant=variant)
        result = compute_transport_defect_result(low_row, high_row)
        row = {
            "variant": low_row["variant"],
            "variant_role": classify_variant_role(str(low_row["variant"])),
            "low_scale": result["low_scale"],
            "high_scale": result["high_scale"],
            "pattern_selection": result["pattern_selection"],
            "run_dir": result["run_dir"],
            "source_kind": result["source_kind"],
            "null_model": result["null_model"],
            "null_seed": result["null_seed"],
            "phase1_policies": result["phase1_policies"],
            "terminal_fraction": result["terminal_fraction"],
            "defect_js_mean": result["defect_js_mean"],
            "defect_wj_mean": result["defect_wj_mean"],
            "defect_flow_mean": result["defect_flow_mean"],
            "defect_top_mass": result["defect_top_mass"],
            "defect_mean_gap": result["defect_mean_gap"],
            "segment_defect_stability": result["segment_defect_stability"],
            "connected_pattern_count": result["connected_pattern_count"],
            "selected_low_pattern_count": result["selected_low_pattern_count"],
            "selected_high_pattern_count": result["selected_high_pattern_count"],
            "transport_result": result,
        }
        rows.append(row)
    return rows


def compute_transport_defect_result(low_row: dict, high_row: dict) -> dict:
    low_scale = int(low_row["pattern_scale"])
    high_scale = int(high_row["pattern_scale"])
    if high_scale <= low_scale:
        raise ValueError(
            f"Expected high scale > low scale, got {high_scale} <= {low_scale}."
        )
    if low_row["pattern_selection"] != high_row["pattern_selection"]:
        raise ValueError("Low/high return-lag runs must use the same pattern selection.")

    low_run_dir = Path(low_row["run_dir"]).resolve()
    high_run_dir = Path(high_row["run_dir"]).resolve()
    if low_run_dir != high_run_dir:
        raise ValueError(
            "Low/high return-lag rows do not point to the same Phase 1 run directory."
        )

    prefix_fibers = load_policy_fibers(low_run_dir, "prefix")
    suffix_fibers = load_policy_fibers(low_run_dir, "suffix")
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
    defect_weight = 0.0
    weight_total = 0.0
    top_mismatch_weight = 0.0
    js_sum = 0.0
    wj_sum = 0.0
    flow_sum = 0.0
    mean_gap_sum = 0.0
    terminal_count = 0
    nonterminal_count = 0

    for pattern, low_item in sorted(low_patterns.items()):
        prefix_parent = prefix_index.get(pattern)
        suffix_parent = suffix_index.get(pattern)
        transport = _build_transport_kernel(
            pattern=pattern,
            high_pattern_set=high_pattern_set,
            prefix_parent=prefix_parent,
            suffix_parent=suffix_parent,
        )
        weight = float(low_item.get("occurrence_count", 0))
        weight = max(weight, 1.0)
        low_bins = _normalize_bin_counts(_pattern_bin_counts(low_item))

        if transport["terminal_flag"]:
            terminal_count += 1
            per_pattern.append(
                {
                    "pattern": pattern,
                    "weight": weight,
                    "occurrence_count": int(low_item.get("occurrence_count", 0)),
                    "segment_occurrences": list(low_item.get("segment_occurrences", [])),
                    "candidate_count": 0,
                    "terminal_flag": True,
                    "kernel": [],
                    "observed_top_bin": _top_bin_label(low_item),
                    "transported_top_bin": None,
                    "delta_js": None,
                    "delta_wj": None,
                    "delta_top": None,
                    "delta_mean": None,
                }
            )
            continue

        nonterminal_count += 1
        transported_profile = _transport_profile(transport["kernel"], high_patterns)
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
        if delta_top:
            top_mismatch_weight += weight
        defect_weight += weight

        per_pattern.append(
            {
                "pattern": pattern,
                "weight": weight,
                "occurrence_count": int(low_item.get("occurrence_count", 0)),
                "segment_occurrences": list(low_item.get("segment_occurrences", [])),
                "candidate_count": len(transport["kernel"]),
                "terminal_flag": False,
                "kernel": transport["kernel"],
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

    segment_defect_stability = _segment_defect_stability(per_pattern)
    top_patterns = [
        {
            "pattern": item["pattern"],
            "delta_js": item["delta_js"],
            "delta_wj": item["delta_wj"],
            "delta_flow": item["delta_flow"],
            "delta_top": item["delta_top"],
            "delta_mean": item["delta_mean"],
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
        "connected_pattern_count": nonterminal_count,
        "terminal_pattern_count": terminal_count,
        "terminal_fraction": _safe_ratio(terminal_count, len(low_pattern_set)),
        "defect_js_mean": _safe_ratio(js_sum, weight_total),
        "defect_wj_mean": _safe_ratio(wj_sum, weight_total),
        "defect_flow_mean": _safe_ratio(flow_sum, weight_total),
        "defect_top_mass": _safe_ratio(top_mismatch_weight, weight_total),
        "defect_mean_gap": _safe_ratio(mean_gap_sum, weight_total),
        "segment_defect_stability": segment_defect_stability,
        "run_dir": str(low_run_dir),
        "source_kind": low_row.get("source_kind"),
        "null_model": low_row.get("null_model"),
        "null_seed": low_row.get("null_seed"),
        "phase1_policies": list(low_row.get("phase1_policies", [])),
        "low_dataset_path": low_row.get("dataset_path"),
        "high_dataset_path": high_row.get("dataset_path"),
        "top_defect_patterns": top_patterns,
        "per_pattern": per_pattern,
        "notes": [
            "transport kernel = normalized child support restricted to selected high-scale patterns and aggregated across prefix/suffix edges",
            "transported profile = weighted mixture of selected high-scale lag-bin profiles",
            "delta_js = Jensen-Shannon divergence between observed and transported lag-bin profiles",
            "delta_wj = 1 - weighted Jaccard between observed and transported lag-bin profiles",
            "delta_flow = ordered earth-mover distance on the log-lag-bin axis",
            "delta_top = dominant lag-bin mismatch indicator",
            "delta_mean = relative mean-return-lag gap",
        ],
    }


def render_console_summary(rows: list[dict]) -> str:
    lines = [
        "Phase 2 N2-01 transport-defect pilot",
        "-" * 116,
        f"{'variant':<18}{'kind':<14}{'term':>8}{'d_JS':>10}{'d_wJ':>10}{'d_flow':>10}{'d_top':>10}{'d_mean':>10}{'seg_stab':>10}{'conn':>8}",
    ]
    for row in rows:
        lines.append(
            f"{row['variant']:<18}"
            f"{str(row.get('null_model') or row.get('source_kind')):<14}"
            f"{row['terminal_fraction']:>8.4f}"
            f"{row['defect_js_mean']:>10.4f}"
            f"{row['defect_wj_mean']:>10.4f}"
            f"{row['defect_flow_mean']:>10.4f}"
            f"{row['defect_top_mass']:>10.4f}"
            f"{row['defect_mean_gap']:>10.4f}"
            f"{row['segment_defect_stability']:>10.4f}"
            f"{row['connected_pattern_count']:>8}"
        )
    return "\n".join(lines)


def render_markdown_report(rows: list[dict], selection: dict) -> str:
    lines = [
        "# Phase 2 N2-01 Transport-Defect Pilot",
        "",
        "## Selection",
        "",
        f"- Variants: {', '.join(selection['variants'])}",
        f"- Low/high scales: {selection['low_scale']} -> {selection['high_scale']}",
        f"- Pattern selection: {selection['pattern_selection']}",
        "",
        "## Compact Readout",
        "",
        "| Variant | Source | Terminal frac | Defect JS | Defect wJ | Defect flow | Defect top mass | Defect mean gap | Segment stability | Connected |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        source_label = row.get("null_model") or row.get("source_kind") or "-"
        lines.append(
            f"| {row['variant']} | {source_label} | "
            f"{_fmt(row['terminal_fraction'])} | "
            f"{_fmt(row['defect_js_mean'])} | "
            f"{_fmt(row['defect_wj_mean'])} | "
            f"{_fmt(row['defect_flow_mean'])} | "
            f"{_fmt(row['defect_top_mass'])} | "
            f"{_fmt(row['defect_mean_gap'])} | "
            f"{_fmt(row['segment_defect_stability'])} | "
            f"{row['connected_pattern_count']} |"
        )

    for row in rows:
        lines.extend(["", f"## {row['variant']}"])
        lines.extend(
            [
                "",
                f"- Run dir: `{row['run_dir']}`",
                f"- Source kind: `{row['source_kind']}`",
                f"- Null model: `{row['null_model']}`" if row.get("null_model") else "- Null model: observed",
                f"- Terminal fraction: {_fmt(row['terminal_fraction'])}",
                f"- Defect JS mean: {_fmt(row['defect_js_mean'])}",
                f"- Defect wJ mean: {_fmt(row['defect_wj_mean'])}",
                f"- Defect flow mean: {_fmt(row['defect_flow_mean'])}",
                f"- Defect top mass: {_fmt(row['defect_top_mass'])}",
                f"- Defect mean gap: {_fmt(row['defect_mean_gap'])}",
                f"- Segment defect stability: {_fmt(row['segment_defect_stability'])}",
                "",
                "Top defect patterns:",
            ]
        )
        if row["transport_result"]["top_defect_patterns"]:
            for item in row["transport_result"]["top_defect_patterns"][:5]:
                lines.append(
                    f"- `{item['pattern']}` | JS={_fmt(item['delta_js'])} | "
                    f"wJ={_fmt(item['delta_wj'])} | flow={_fmt(item['delta_flow'])} | top={item['delta_top']} | "
                    f"d_mean={_fmt(item['delta_mean'])}"
                )
        else:
            lines.append("- none")
    return "\n".join(lines)


def _index_patterns(row: dict) -> dict[str, dict]:
    return {
        str(item["pattern"]): item
        for item in row["lag_result"].get("pattern_results", [])
    }


def _load_return_lag_row_casefold(dataset_path: Path, *, variant: str) -> dict:
    try:
        return load_return_lag_row(dataset_path, variant=variant)
    except ValueError:
        with open(dataset_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        target = str(variant).casefold()
        for row in payload.get("rows", []):
            if str(row.get("variant")).casefold() == target:
                return row
        raise


def _build_transport_kernel(
    *,
    pattern: str,
    high_pattern_set: set[str],
    prefix_parent: dict | None,
    suffix_parent: dict | None,
) -> dict:
    candidate_scores: dict[str, dict] = {}
    for policy, parent in (("prefix", prefix_parent), ("suffix", suffix_parent)):
        if parent is None:
            continue
        for child in parent.get("children", []):
            child_pattern = str(child["pattern"])
            if child_pattern not in high_pattern_set:
                continue
            entry = candidate_scores.setdefault(
                child_pattern,
                {
                    "pattern": child_pattern,
                    "score": 0.0,
                    "policies": set(),
                    "policy_scores": {},
                },
            )
            support = float(child.get("normalized_frequency", 0.0))
            if support <= 0.0:
                support = float(child.get("raw_count", 0))
            entry["score"] += support
            entry["policies"].add(policy)
            entry["policy_scores"][policy] = support

    if not candidate_scores:
        return {"terminal_flag": True, "kernel": []}

    total = sum(max(item["score"], 0.0) for item in candidate_scores.values())
    if total <= 0.0:
        return {"terminal_flag": True, "kernel": []}

    kernel = [
        {
            "pattern": item["pattern"],
            "weight": item["score"] / total,
            "policies": sorted(item["policies"]),
            "policy_scores": item["policy_scores"],
        }
        for item in sorted(
            candidate_scores.values(),
            key=lambda item: (-item["score"], -len(item["policies"]), item["pattern"]),
        )
    ]
    return {"terminal_flag": False, "kernel": kernel}


def _transport_profile(kernel: list[dict], high_patterns: dict[str, dict]) -> dict:
    bin_probs: dict[str, float] = {}
    mean_return_lag = 0.0
    for entry in kernel:
        high_item = high_patterns[entry["pattern"]]
        weight = float(entry["weight"])
        high_probs = _normalize_bin_counts(_pattern_bin_counts(high_item))
        for bin_label, prob in high_probs.items():
            bin_probs[bin_label] = bin_probs.get(bin_label, 0.0) + (weight * prob)
        mean_return_lag += weight * float(high_item.get("mean_return_lag", 0.0))
    return {
        "bin_probs": bin_probs,
        "mean_return_lag": mean_return_lag,
        "top_bin": _top_bin_from_probs(bin_probs),
    }


def _segment_defect_stability(per_pattern: list[dict]) -> float:
    active = [item for item in per_pattern if not item["terminal_flag"]]
    if not active:
        return 0.0
    ranked = sorted(
        active,
        key=lambda item: (
            -(item["delta_js"] if item["delta_js"] is not None else -1.0),
            -(item["weight"]),
            item["pattern"],
        ),
    )
    top_patterns = ranked[: min(5, len(ranked))]
    segment_sets = []
    segment_count = max((len(item.get("segment_occurrences", [])) for item in top_patterns), default=0)
    for segment_index in range(segment_count):
        present = {
            item["pattern"]
            for item in top_patterns
            if segment_index < len(item.get("segment_occurrences", []))
            and int(item["segment_occurrences"][segment_index]) > 0
        }
        if present:
            segment_sets.append(present)
    if len(segment_sets) < 2:
        return 0.0
    overlaps = []
    for left_index in range(len(segment_sets)):
        for right_index in range(left_index + 1, len(segment_sets)):
            left = segment_sets[left_index]
            right = segment_sets[right_index]
            overlaps.append(_safe_ratio(len(left & right), len(left | right)))
    return _mean(overlaps)


def _pattern_bin_counts(pattern_item: dict) -> dict[str, float]:
    lag_bin_counts = pattern_item.get("lag_bin_counts")
    if isinstance(lag_bin_counts, dict) and lag_bin_counts:
        return {str(key): float(value) for key, value in lag_bin_counts.items()}
    top_bins = pattern_item.get("top_return_bins", [])
    if top_bins:
        return {str(item["bin"]): float(item["count"]) for item in top_bins}
    return {}


def _normalize_bin_counts(counts: dict[str, float]) -> dict[str, float]:
    total = sum(float(value) for value in counts.values())
    if total <= 0.0:
        return {}
    return {key: float(value) / total for key, value in counts.items()}


def _weighted_jaccard(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 1.0
    shared = sum(min(left.get(key, 0.0), right.get(key, 0.0)) for key in keys)
    union = sum(max(left.get(key, 0.0), right.get(key, 0.0)) for key in keys)
    return _safe_ratio(shared, union)


def _js_divergence(left: dict[str, float], right: dict[str, float]) -> float:
    keys = sorted(set(left) | set(right))
    if not keys:
        return 0.0
    left_probs = [left.get(key, 0.0) for key in keys]
    right_probs = [right.get(key, 0.0) for key in keys]
    midpoint = [(lp + rp) / 2.0 for lp, rp in zip(left_probs, right_probs)]
    return (_kl_divergence(left_probs, midpoint) + _kl_divergence(right_probs, midpoint)) / 2.0


def _ordered_flow_distance(left: dict[str, float], right: dict[str, float]) -> float:
    keys = sorted(set(left) | set(right), key=_lag_bin_index)
    if not keys:
        return 0.0
    imbalance = 0.0
    distance = 0.0
    previous_index = None
    for key in keys:
        index = _lag_bin_index(key)
        if previous_index is not None:
            distance += abs(imbalance) * float(index - previous_index)
        imbalance += left.get(key, 0.0) - right.get(key, 0.0)
        previous_index = index
    return distance


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
    if top_bins:
        return str(top_bins[0]["bin"])
    lag_bin_counts = pattern_item.get("lag_bin_counts")
    if isinstance(lag_bin_counts, dict) and lag_bin_counts:
        return _top_bin_from_probs(_normalize_bin_counts(_pattern_bin_counts(pattern_item)))
    return None


def _top_bin_from_probs(probs: dict[str, float]) -> str | None:
    if not probs:
        return None
    ranked = sorted(probs.items(), key=lambda item: (-item[1], item[0]))
    return ranked[0][0]


def _lag_bin_index(label: str) -> int:
    low_text, _, _ = str(label).partition("-")
    low = max(int(low_text), 1)
    return int(math.log2(low))


def _safe_ratio(numerator: float | int, denominator: float | int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"
