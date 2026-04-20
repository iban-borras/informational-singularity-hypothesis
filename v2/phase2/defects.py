from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path


def load_policy_fibers(run_dir: Path, policy: str) -> list[dict]:
    path = run_dir / "fibers" / f"fibers_{policy}.json"
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list of fiber artifacts in {path}")
    return payload


def build_defect_rows(
    runs: list[dict],
    *,
    window_scales: tuple[int, int, int],
    policies: list[str],
) -> list[dict]:
    rows = []
    for run in runs:
        config = run["dataset"]["config"]
        run_dir = Path(run["_run_dir"])
        policy_results: dict[str, dict] = {}
        for policy in policies:
            fibers = load_policy_fibers(run_dir, policy)
            policy_results[policy] = compute_policy_defect_result(
                fibers,
                window_scales=window_scales,
            )

        row = {
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
            "run_dir": str(run_dir),
            "dataset_path": run["_dataset_path"],
            "summary_path": run["_summary_path"],
            "generated_at": run["generated_at"],
            "policy_results": policy_results,
        }
        for policy, result in policy_results.items():
            row[f"{policy}_examined_parent_count"] = result["examined_parent_count"]
            row[f"{policy}_tested_parent_count"] = result["tested_parent_count"]
            row[f"{policy}_untested_parent_count"] = result["untested_parent_count"]
            row[f"{policy}_future_parent_fraction"] = result["future_parent_fraction"]
            row[f"{policy}_terminal_parent_count"] = result["terminal_parent_count"]
            row[f"{policy}_terminal_parent_fraction"] = result["terminal_parent_fraction"]
            row[f"{policy}_defect_parent_count"] = result["defect_parent_count"]
            row[f"{policy}_defect_rate"] = result["defect_rate"]
            row[f"{policy}_defect_mass_rate"] = result["defect_mass_rate"]
            row[f"{policy}_defect_family_count"] = result["defect_family_count"]
            row[f"{policy}_defect_entropy"] = result["defect_entropy"]
            row[f"{policy}_segment_consistency"] = result["segment_consistency"]
            row[f"{policy}_mean_defect_strength"] = result["mean_defect_strength"]

        if len(policies) >= 2:
            left = policy_results[policies[0]]
            right = policy_results[policies[1]]
            row["defect_rate_gap_signed"] = left["defect_rate"] - right["defect_rate"]
            row["defect_rate_gap_abs"] = abs(row["defect_rate_gap_signed"])
            row["defect_mass_gap_signed"] = left["defect_mass_rate"] - right["defect_mass_rate"]
            row["defect_mass_gap_abs"] = abs(row["defect_mass_gap_signed"])
            row["defect_entropy_gap_signed"] = left["defect_entropy"] - right["defect_entropy"]
            row["defect_entropy_gap_abs"] = abs(row["defect_entropy_gap_signed"])
        else:
            row["defect_rate_gap_signed"] = None
            row["defect_rate_gap_abs"] = None
            row["defect_mass_gap_signed"] = None
            row["defect_mass_gap_abs"] = None
            row["defect_entropy_gap_signed"] = None
            row["defect_entropy_gap_abs"] = None
        rows.append(row)

    return rows


def compute_policy_defect_result(
    fibers: list[dict],
    *,
    window_scales: tuple[int, int, int],
) -> dict:
    low_scale, mid_scale, high_scale = window_scales
    low_edge = _require_edge(fibers, target_m=low_scale, source_m=mid_scale)
    high_edge = _require_edge(fibers, target_m=mid_scale, source_m=high_scale)
    high_parent_index = {parent["pattern"]: parent for parent in high_edge.get("parents", [])}

    parent_results = []
    defect_weight = 0.0
    tested_weight = 0.0
    family_weights: dict[str, float] = {}
    family_counts: dict[str, int] = {}

    for parent in low_edge.get("parents", []):
        detail = _analyze_parent(parent, high_parent_index)
        if detail is None:
            continue
        parent_results.append(detail)
        if detail["future_bearing"]:
            tested_weight += detail["parent_weight"]
        if detail["defect_flag"]:
            defect_weight += detail["parent_weight"]
            family = detail["defect_family"]
            if family is not None:
                family_weights[family] = family_weights.get(family, 0.0) + detail["parent_weight"]
                family_counts[family] = family_counts.get(family, 0) + 1

    segment_results = []
    segment_family_sets = []
    segment_count = _infer_segment_count(low_edge)
    for segment_index in range(segment_count):
        segment_result = _analyze_segment(low_edge, high_parent_index, segment_index=segment_index)
        segment_results.append(segment_result)
        if segment_result["families"]:
            segment_family_sets.append(set(segment_result["families"]))

    defect_strength_values = [
        item["defect_strength"]
        for item in parent_results
        if item["defect_flag"] and item["defect_strength"] is not None
    ]
    top_families = [
        {
            "family": family,
            "weight": family_weights[family],
            "count": family_counts[family],
        }
        for family in sorted(
            family_weights,
            key=lambda item: (-family_weights[item], -family_counts[item], item),
        )
    ]

    examined_parent_count = len(parent_results)
    tested_parent_count = sum(1 for item in parent_results if item["future_bearing"])
    terminal_parent_count = sum(1 for item in parent_results if item["terminal_flag"])
    defect_parent_count = sum(1 for item in parent_results if item["defect_flag"])
    result = {
        "window_scales": [low_scale, mid_scale, high_scale],
        "examined_parent_count": examined_parent_count,
        "tested_parent_count": tested_parent_count,
        "untested_parent_count": max(0, len(low_edge.get("parents", [])) - examined_parent_count),
        "future_parent_fraction": _safe_ratio(tested_parent_count, examined_parent_count),
        "terminal_parent_count": terminal_parent_count,
        "terminal_parent_fraction": _safe_ratio(terminal_parent_count, examined_parent_count),
        "defect_parent_count": defect_parent_count,
        "defect_rate": _safe_ratio(defect_parent_count, tested_parent_count),
        "defect_mass_rate": _safe_ratio(defect_weight, tested_weight),
        "defect_family_count": len(top_families),
        "defect_entropy": _weight_entropy(list(family_weights.values())),
        "segment_consistency": _pairwise_jaccard_mean(segment_family_sets),
        "mean_defect_strength": _mean(defect_strength_values),
        "top_defect_families": top_families[:10],
        "segment_results": segment_results,
        "parent_results": parent_results,
        "notes": [
            "local_best_child = top child by immediate fiber mass at the lower edge",
            "deep_best_child = child with highest one-step-ahead branching richness at the next higher edge",
            "terminal closure = no child has positive one-step-ahead branching richness",
            "defect = mismatch between local_best_child and deep_best_child, conditioned on future-bearing parents only",
        ],
    }
    return result


def render_console_summary(rows: list[dict]) -> str:
    lines = [
        "Phase 2 minimum commutation-defect pilot",
        "-" * 116,
        f"{'variant':<14}{'iter':>6}{'p_def':>10}{'s_def':>10}{'p_fut':>10}{'s_fut':>10}{'p_term':>10}{'s_term':>10}{'gap_d':>10}{'gap_m':>10}",
    ]
    for row in rows:
        lines.append(
            f"{row['variant']:<14}"
            f"{_fmt_int(row['iteration']):>6}"
            f"{_fmt(row.get('prefix_defect_rate')):>10}"
            f"{_fmt(row.get('suffix_defect_rate')):>10}"
            f"{_fmt(row.get('prefix_future_parent_fraction')):>10}"
            f"{_fmt(row.get('suffix_future_parent_fraction')):>10}"
            f"{_fmt(row.get('prefix_terminal_parent_fraction')):>10}"
            f"{_fmt(row.get('suffix_terminal_parent_fraction')):>10}"
            f"{_fmt(row.get('defect_rate_gap_abs')):>10}"
            f"{_fmt(row.get('defect_mass_gap_abs')):>10}"
        )
    return "\n".join(lines)


def render_markdown_report(rows: list[dict], selection: dict) -> str:
    lines = [
        "# Phase 2 Minimum Commutation-Defect Pilot",
        "",
        "## Selection",
        "",
        f"- Variants: {', '.join(selection['variants'])}",
        f"- Segment protocol: {selection['num_segments']} x {selection['segment_bits']}",
        f"- Segment offset bits: {selection['segment_offset_bits']}",
        f"- Phase 1 scales: {', '.join(str(value) for value in selection['phase1_scales'])}",
        f"- Defect window: {', '.join(str(value) for value in selection['window_scales'])}",
        f"- Policies: {', '.join(selection['policies'])}",
        "",
        "## Compact Readout",
        "",
        "| Variant | Iter | Prefix defect rate | Suffix defect rate | Prefix future frac | Suffix future frac | Prefix terminal frac | Suffix terminal frac | |gap defect| | |gap mass| |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['iteration']} | "
            f"{_fmt(row.get('prefix_defect_rate'))} | { _fmt(row.get('suffix_defect_rate'))} | "
            f"{_fmt(row.get('prefix_future_parent_fraction'))} | { _fmt(row.get('suffix_future_parent_fraction'))} | "
            f"{_fmt(row.get('prefix_terminal_parent_fraction'))} | { _fmt(row.get('suffix_terminal_parent_fraction'))} | "
            f"{_fmt(row.get('defect_rate_gap_abs'))} | { _fmt(row.get('defect_mass_gap_abs'))} |"
        )

    for row in rows:
        lines.extend(["", f"## {row['variant']}"])
        for policy in row["policies"]:
            result = row["policy_results"][policy]
            lines.extend(
                [
                    "",
                    f"### Policy `{policy}`",
                    "",
                    f"- Examined parents: {result['examined_parent_count']}",
                    f"- Tested parents: {result['tested_parent_count']}",
                    f"- Untested parents: {result['untested_parent_count']}",
                    f"- Future-bearing fraction: {_fmt(result['future_parent_fraction'])}",
                    f"- Terminal parents: {result['terminal_parent_count']}",
                    f"- Terminal fraction: {_fmt(result['terminal_parent_fraction'])}",
                    f"- Defect rate: {_fmt(result['defect_rate'])}",
                    f"- Defect mass rate: {_fmt(result['defect_mass_rate'])}",
                    f"- Defect family count: {result['defect_family_count']}",
                    f"- Defect entropy: {_fmt(result['defect_entropy'])}",
                    f"- Segment consistency: {_fmt(result['segment_consistency'])}",
                    f"- Mean defect strength: {_fmt(result['mean_defect_strength'])}",
                    "",
                    "Top defect families:",
                ]
            )
            if result["top_defect_families"]:
                for family in result["top_defect_families"][:5]:
                    lines.append(
                        f"- `{family['family']}` | weight={_fmt(family['weight'])} | count={family['count']}"
                    )
            else:
                lines.append("- none")

    return "\n".join(lines)


def classify_variant_role(variant: str) -> str:
    normalized = variant.upper()
    if normalized in {"B", "F", "A", "M", "N"}:
        return "primary"
    if normalized in {"E", "I", "J", "K", "L", "D", "G", "H"}:
        return "supporting"
    return "other"


def _require_edge(fibers: list[dict], *, target_m: int, source_m: int) -> dict:
    for item in fibers:
        if item.get("target_m") == target_m and item.get("source_m") == source_m:
            return item
    raise ValueError(f"Missing fiber edge {source_m}->{target_m} in requested Phase 1 run.")


def _analyze_segment(low_edge: dict, high_parent_index: dict[str, dict], *, segment_index: int) -> dict:
    parent_results = []
    tested_weight = 0.0
    defect_weight = 0.0
    families = []
    for parent in low_edge.get("parents", []):
        detail = _analyze_parent(parent, high_parent_index, segment_index=segment_index)
        if detail is None:
            continue
        parent_results.append(detail)
        if detail["future_bearing"]:
            tested_weight += detail["parent_weight"]
        if detail["defect_flag"]:
            defect_weight += detail["parent_weight"]
            if detail["defect_family"] is not None:
                families.append(detail["defect_family"])

    examined_parent_count = len(parent_results)
    tested_parent_count = sum(1 for item in parent_results if item["future_bearing"])
    terminal_parent_count = sum(1 for item in parent_results if item["terminal_flag"])
    defect_parent_count = sum(1 for item in parent_results if item["defect_flag"])
    return {
        "segment_index": segment_index,
        "examined_parent_count": examined_parent_count,
        "tested_parent_count": tested_parent_count,
        "future_parent_fraction": _safe_ratio(tested_parent_count, examined_parent_count),
        "terminal_parent_count": terminal_parent_count,
        "terminal_parent_fraction": _safe_ratio(terminal_parent_count, examined_parent_count),
        "defect_parent_count": defect_parent_count,
        "defect_rate": _safe_ratio(defect_parent_count, tested_parent_count),
        "defect_mass_rate": _safe_ratio(defect_weight, tested_weight),
        "families": sorted(set(families)),
    }


def _analyze_parent(parent: dict, high_parent_index: dict[str, dict], *, segment_index: int | None = None) -> dict | None:
    children = parent.get("children", [])
    if not children:
        return None

    child_entries = []
    for child in children:
        high_parent = high_parent_index.get(child["pattern"])
        local_score = (
            float(child.get("normalized_frequency", 0.0))
            if segment_index is None
            else float(child.get("segment_counts", [])[segment_index])
        )
        deep_score = _descendant_score(high_parent, segment_index=segment_index)
        child_entries.append(
            {
                "pattern": child["pattern"],
                "local_score": local_score,
                "deep_score": deep_score,
                "raw_count": int(child.get("raw_count", 0)),
                "segment_support": int(child.get("segment_support", 0)),
                "grandchild_count": len(high_parent.get("children", [])) if high_parent else 0,
            }
        )

    local_best = _pick_best(child_entries, score_key="local_score", require_positive=True)
    if local_best is None:
        return None
    deep_best = _pick_best(child_entries, score_key="deep_score", require_positive=True)
    deep_missing = deep_best is None
    future_bearing = not deep_missing
    terminal_flag = deep_missing
    defect_flag = future_bearing and local_best["pattern"] != deep_best["pattern"]
    defect_family = None
    if defect_flag:
        defect_family = f"{local_best['pattern']}->{deep_best['pattern']}"

    total_deep_score = sum(max(item["deep_score"], 0.0) for item in child_entries)
    local_deep_score = next(
        item["deep_score"] for item in child_entries if item["pattern"] == local_best["pattern"]
    )
    defect_strength = None
    if defect_flag and total_deep_score > 0:
        defect_strength = max(0.0, deep_best["deep_score"] - local_deep_score) / total_deep_score

    parent_weight = (
        float(parent.get("normalized_frequency", 0.0))
        if segment_index is None
        else float(parent.get("segment_counts", [])[segment_index])
    )
    return {
        "pattern": parent["pattern"],
        "parent_weight": parent_weight,
        "fiber_size": int(parent.get("fiber_size", 0)),
        "local_best_child": local_best["pattern"],
        "local_best_score": local_best["local_score"],
        "local_best_deep_score": local_deep_score,
        "deep_best_child": None if deep_best is None else deep_best["pattern"],
        "deep_best_score": None if deep_best is None else deep_best["deep_score"],
        "deep_best_missing": deep_missing,
        "future_bearing": future_bearing,
        "terminal_flag": terminal_flag,
        "defect_flag": defect_flag,
        "defect_family": defect_family,
        "defect_strength": defect_strength,
    }


def _descendant_score(high_parent: dict | None, *, segment_index: int | None) -> float:
    if high_parent is None:
        return 0.0
    if segment_index is None:
        return float(high_parent.get("child_entropy", 0.0))
    masses = [
        float(int(grandchild.get("segment_counts", [])[segment_index]))
        for grandchild in high_parent.get("children", [])
    ]
    return _weight_entropy(masses)


def _pick_best(entries: list[dict], *, score_key: str, require_positive: bool) -> dict | None:
    if not entries:
        return None
    if require_positive:
        pool = [item for item in entries if item[score_key] > 0]
        if not pool:
            return None
    else:
        pool = list(entries)
    ranked = sorted(
        pool,
        key=lambda item: (
            -float(item[score_key]),
            -int(item["grandchild_count"]),
            -int(item["segment_support"]),
            -int(item["raw_count"]),
            item["pattern"],
        ),
    )
    return ranked[0]


def _infer_segment_count(low_edge: dict) -> int:
    for parent in low_edge.get("parents", []):
        segment_counts = parent.get("segment_counts", [])
        if segment_counts:
            return len(segment_counts)
    return 0


def _pairwise_jaccard_mean(sets: list[set[str]]) -> float | None:
    if len(sets) < 2:
        return None
    values = []
    for left, right in combinations(sets, 2):
        union = left | right
        if not union:
            continue
        values.append(len(left & right) / len(union))
    if not values:
        return None
    return float(sum(values) / len(values))


def _weight_entropy(weights: list[float]) -> float:
    positive = [float(value) for value in weights if value > 0]
    total = sum(positive)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for value in positive:
        p = value / total
        entropy -= p * math.log2(p)
    return float(entropy)


def _safe_ratio(numerator: float | int, denominator: float | int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def _fmt_int(value: int | None) -> str:
    if value is None:
        return "-"
    return str(value)
