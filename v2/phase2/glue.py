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


def build_glue_rows(
    runs: list[dict],
    *,
    window_scales: tuple[int, int, int, int],
    policies: list[str],
) -> list[dict]:
    rows = []
    for run in runs:
        config = run["dataset"]["config"]
        run_dir = _run_dir(run)
        policy_results = {}
        for policy in policies:
            fibers = load_policy_fibers(run_dir, policy)
            policy_results[policy] = compute_policy_glue_result(
                fibers,
                window_scales=window_scales,
                policy=policy,
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
            row[f"{policy}_candidate_pair_count"] = result["candidate_pair_count"]
            row[f"{policy}_glueable_pair_count"] = result["glueable_pair_count"]
            row[f"{policy}_glueable_pair_fraction"] = result["glueable_pair_fraction"]
            row[f"{policy}_glue_defect_pair_count"] = result["glue_defect_pair_count"]
            row[f"{policy}_glue_defect_rate"] = result["glue_defect_rate"]
            row[f"{policy}_glue_defect_mass_rate"] = result["glue_defect_mass_rate"]
            row[f"{policy}_mean_glue_distance"] = result["mean_glue_distance"]
            row[f"{policy}_glue_family_count"] = result["glue_family_count"]
            row[f"{policy}_glue_entropy"] = result["glue_entropy"]
            row[f"{policy}_segment_consistency"] = result["segment_consistency"]
        if len(policies) >= 2:
            left = policy_results[policies[0]]
            right = policy_results[policies[1]]
            row["glue_defect_rate_gap_signed"] = left["glue_defect_rate"] - right["glue_defect_rate"]
            row["glue_defect_rate_gap_abs"] = abs(row["glue_defect_rate_gap_signed"])
            row["glue_distance_gap_signed"] = left["mean_glue_distance"] - right["mean_glue_distance"]
            row["glue_distance_gap_abs"] = abs(row["glue_distance_gap_signed"])
        else:
            row["glue_defect_rate_gap_signed"] = None
            row["glue_defect_rate_gap_abs"] = None
            row["glue_distance_gap_signed"] = None
            row["glue_distance_gap_abs"] = None
        rows.append(row)
    return rows


def compute_policy_glue_result(
    fibers: list[dict],
    *,
    window_scales: tuple[int, int, int, int],
    policy: str,
) -> dict:
    base_scale, parent_scale, mid_scale, high_scale = window_scales
    if not (base_scale < parent_scale < mid_scale < high_scale):
        raise ValueError("Glue pilot requires four strictly ascending scales.")
    parent_edge = _require_edge(fibers, target_m=parent_scale, source_m=mid_scale)
    high_edge = _require_edge(fibers, target_m=mid_scale, source_m=high_scale)
    high_index = {parent["pattern"]: parent for parent in high_edge.get("parents", [])}
    high_overlap_len = high_scale - (parent_scale - base_scale)

    candidate_pairs = _build_overlap_pairs(parent_edge, overlap_len=base_scale)
    pair_results = []
    glueable_weight = 0.0
    defect_weight = 0.0
    family_weights: dict[str, float] = {}
    family_counts: dict[str, int] = {}
    for left_pattern, right_pattern in candidate_pairs:
        left_parent = _parent_by_pattern(parent_edge, left_pattern)
        right_parent = _parent_by_pattern(parent_edge, right_pattern)
        detail = _analyze_pair(
            left_parent,
            right_parent,
            high_index,
            overlap_len=high_overlap_len,
            segment_index=None,
            policy=policy,
        )
        if detail is None:
            continue
        pair_results.append(detail)
        if detail["glueable"]:
            glueable_weight += detail["pair_weight"]
            if detail["glue_defect_flag"]:
                defect_weight += detail["pair_weight"]
                family = detail["glue_family"]
                if family is not None:
                    family_weights[family] = family_weights.get(family, 0.0) + detail["pair_weight"]
                    family_counts[family] = family_counts.get(family, 0) + 1

    segment_results = []
    segment_family_sets = []
    segment_count = _infer_segment_count(parent_edge)
    for segment_index in range(segment_count):
        segment_result = _analyze_segment(
            candidate_pairs,
            parent_edge,
            high_index,
            overlap_len=high_overlap_len,
            policy=policy,
            segment_index=segment_index,
        )
        segment_results.append(segment_result)
        if segment_result["families"]:
            segment_family_sets.append(set(segment_result["families"]))

    candidate_pair_count = len(pair_results)
    glueable_pair_count = sum(1 for item in pair_results if item["glueable"])
    glue_defect_pair_count = sum(1 for item in pair_results if item["glue_defect_flag"])
    distances = [item["glue_distance"] for item in pair_results if item["glueable"]]
    top_families = [
        {"family": family, "weight": family_weights[family], "count": family_counts[family]}
        for family in sorted(family_weights, key=lambda item: (-family_weights[item], -family_counts[item], item))
    ]
    return {
        "window_scales": [base_scale, parent_scale, mid_scale, high_scale],
        "policy": policy,
        "candidate_pair_count": candidate_pair_count,
        "glueable_pair_count": glueable_pair_count,
        "glueable_pair_fraction": _safe_ratio(glueable_pair_count, candidate_pair_count),
        "glue_defect_pair_count": glue_defect_pair_count,
        "glue_defect_rate": _safe_ratio(glue_defect_pair_count, glueable_pair_count),
        "glue_defect_mass_rate": _safe_ratio(defect_weight, glueable_weight),
        "mean_glue_distance": _mean(distances),
        "glue_family_count": len(top_families),
        "glue_entropy": _weight_entropy(list(family_weights.values())),
        "segment_consistency": _pairwise_jaccard_mean(segment_family_sets),
        "top_glue_families": top_families[:10],
        "segment_results": segment_results,
        "pair_results": pair_results,
        "notes": [
            f"candidate pair = ordered parent-scale patterns with suffix{base_scale}(left) = prefix{base_scale}(right)",
            f"dominant descendant = top local continuation path under policy `{policy}` from {parent_scale} to {high_scale}",
            f"glue defect = suffix{high_overlap_len}(left_high) fails to match prefix{high_overlap_len}(right_high)",
        ],
    }


def render_console_summary(rows: list[dict]) -> str:
    lines = [
        "Phase 2 overlap-glue cocycle pilot",
        "-" * 120,
        f"{'variant':<14}{'iter':>6}{'p_glu':>10}{'s_glu':>10}{'p_def':>10}{'s_def':>10}{'p_dist':>10}{'s_dist':>10}{'gap_d':>10}{'gap_x':>10}",
    ]
    for row in rows:
        lines.append(
            f"{row['variant']:<14}"
            f"{_fmt_int(row['iteration']):>6}"
            f"{_fmt(row.get('prefix_glueable_pair_fraction')):>10}"
            f"{_fmt(row.get('suffix_glueable_pair_fraction')):>10}"
            f"{_fmt(row.get('prefix_glue_defect_rate')):>10}"
            f"{_fmt(row.get('suffix_glue_defect_rate')):>10}"
            f"{_fmt(row.get('prefix_mean_glue_distance')):>10}"
            f"{_fmt(row.get('suffix_mean_glue_distance')):>10}"
            f"{_fmt(row.get('glue_defect_rate_gap_abs')):>10}"
            f"{_fmt(row.get('glue_distance_gap_abs')):>10}"
        )
    return "\n".join(lines)


def render_markdown_report(rows: list[dict], selection: dict) -> str:
    lines = [
        "# Phase 2 Overlap-Glue Cocycle Pilot",
        "",
        "## Selection",
        "",
        f"- Variants: {', '.join(selection['variants'])}",
        f"- Segment protocol: {selection['num_segments']} x {selection['segment_bits']}",
        f"- Segment offset bits: {selection['segment_offset_bits']}",
        f"- Phase 1 scales: {', '.join(str(value) for value in selection['phase1_scales'])}",
        f"- Glue window: {', '.join(str(value) for value in selection['window_scales'])}",
        f"- Policies: {', '.join(selection['policies'])}",
        "",
        "## Compact Readout",
        "",
        "| Variant | Iter | Prefix glueable frac | Suffix glueable frac | Prefix defect rate | Suffix defect rate | Prefix mean distance | Suffix mean distance |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['iteration']} | "
            f"{_fmt(row.get('prefix_glueable_pair_fraction'))} | { _fmt(row.get('suffix_glueable_pair_fraction'))} | "
            f"{_fmt(row.get('prefix_glue_defect_rate'))} | { _fmt(row.get('suffix_glue_defect_rate'))} | "
            f"{_fmt(row.get('prefix_mean_glue_distance'))} | { _fmt(row.get('suffix_mean_glue_distance'))} |"
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
                    f"- Candidate pairs: {result['candidate_pair_count']}",
                    f"- Glueable pairs: {result['glueable_pair_count']}",
                    f"- Glueable fraction: {_fmt(result['glueable_pair_fraction'])}",
                    f"- Glue defect rate: {_fmt(result['glue_defect_rate'])}",
                    f"- Glue defect mass rate: {_fmt(result['glue_defect_mass_rate'])}",
                    f"- Mean glue distance: {_fmt(result['mean_glue_distance'])}",
                    f"- Glue family count: {result['glue_family_count']}",
                    f"- Glue entropy: {_fmt(result['glue_entropy'])}",
                    f"- Segment consistency: {_fmt(result['segment_consistency'])}",
                    "",
                    "Top glue families:",
                ]
            )
            if result["top_glue_families"]:
                for family in result["top_glue_families"][:5]:
                    lines.append(
                        f"- `{family['family']}` | weight={_fmt(family['weight'])} | count={family['count']}"
                    )
            else:
                lines.append("- none")
    return "\n".join(lines)


def _analyze_segment(
    candidate_pairs: list[tuple[str, str]],
    parent_edge: dict,
    high_index: dict[str, dict],
    *,
    overlap_len: int,
    policy: str,
    segment_index: int,
) -> dict:
    parent_index = {parent["pattern"]: parent for parent in parent_edge.get("parents", [])}
    pair_results = []
    families = []
    for left_pattern, right_pattern in candidate_pairs:
        detail = _analyze_pair(
            parent_index[left_pattern],
            parent_index[right_pattern],
            high_index,
            overlap_len=overlap_len,
            segment_index=segment_index,
            policy=policy,
        )
        if detail is None:
            continue
        pair_results.append(detail)
        if detail["glue_defect_flag"] and detail["glue_family"] is not None:
            families.append(detail["glue_family"])
    candidate_pair_count = len(pair_results)
    glueable_pair_count = sum(1 for item in pair_results if item["glueable"])
    glue_defect_pair_count = sum(1 for item in pair_results if item["glue_defect_flag"])
    distances = [item["glue_distance"] for item in pair_results if item["glueable"]]
    return {
        "segment_index": segment_index,
        "candidate_pair_count": candidate_pair_count,
        "glueable_pair_count": glueable_pair_count,
        "glueable_pair_fraction": _safe_ratio(glueable_pair_count, candidate_pair_count),
        "glue_defect_pair_count": glue_defect_pair_count,
        "glue_defect_rate": _safe_ratio(glue_defect_pair_count, glueable_pair_count),
        "mean_glue_distance": _mean(distances),
        "families": sorted(set(families)),
    }


def _analyze_pair(
    left_parent: dict,
    right_parent: dict,
    high_index: dict[str, dict],
    *,
    overlap_len: int,
    segment_index: int | None,
    policy: str,
) -> dict | None:
    left_desc = _dominant_two_step_descendant(left_parent, high_index, segment_index=segment_index)
    right_desc = _dominant_two_step_descendant(right_parent, high_index, segment_index=segment_index)
    pair_weight = min(_parent_weight(left_parent, segment_index), _parent_weight(right_parent, segment_index))
    if left_desc is None or right_desc is None:
        return {
            "left_pattern": left_parent["pattern"],
            "right_pattern": right_parent["pattern"],
            "pair_weight": pair_weight,
            "glueable": False,
            "glue_defect_flag": False,
            "glue_distance": None,
            "glue_family": None,
            "policy": policy,
        }
    left_overlap = left_desc["high_pattern"][-overlap_len:]
    right_overlap = right_desc["high_pattern"][:overlap_len]
    glue_distance = _normalized_hamming(left_overlap, right_overlap)
    glue_defect_flag = glue_distance > 0.0
    glue_family = None
    if glue_defect_flag:
        glue_family = f"{left_parent['pattern']}|{right_parent['pattern']}=>{left_overlap}!={right_overlap}"
    return {
        "left_pattern": left_parent["pattern"],
        "right_pattern": right_parent["pattern"],
        "left_high_pattern": left_desc["high_pattern"],
        "right_high_pattern": right_desc["high_pattern"],
        "pair_weight": pair_weight,
        "glueable": True,
        "glue_defect_flag": glue_defect_flag,
        "glue_distance": glue_distance,
        "glue_family": glue_family,
        "policy": policy,
    }


def _build_overlap_pairs(parent_edge: dict, *, overlap_len: int) -> list[tuple[str, str]]:
    prefix_index: dict[str, list[str]] = {}
    for parent in parent_edge.get("parents", []):
        prefix_key = parent["pattern"][:overlap_len]
        prefix_index.setdefault(prefix_key, []).append(parent["pattern"])

    pairs = []
    for parent in parent_edge.get("parents", []):
        suffix_key = parent["pattern"][-overlap_len:]
        for right_pattern in prefix_index.get(suffix_key, []):
            if right_pattern == parent["pattern"]:
                continue
            pairs.append((parent["pattern"], right_pattern))
    return pairs


def _dominant_two_step_descendant(parent: dict, high_index: dict[str, dict], *, segment_index: int | None) -> dict | None:
    children = parent.get("children", [])
    if not children:
        return None
    child_entries = []
    for child in children:
        local_score = _child_weight(child, segment_index)
        high_parent = high_index.get(child["pattern"])
        grandchild = _pick_best_grandchild(high_parent, segment_index=segment_index)
        child_entries.append(
            {
                "pattern": child["pattern"],
                "local_score": local_score,
                "raw_count": int(child.get("raw_count", 0)),
                "segment_support": int(child.get("segment_support", 0)),
                "grandchild_count": 0 if high_parent is None else len(high_parent.get("children", [])),
                "grandchild_pattern": None if grandchild is None else grandchild["pattern"],
            }
        )
    best_child = _pick_best(child_entries, score_key="local_score", require_positive=True)
    if best_child is None or best_child["grandchild_pattern"] is None:
        return None
    return {"mid_pattern": best_child["pattern"], "high_pattern": best_child["grandchild_pattern"]}


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


def _parent_by_pattern(parent_edge: dict, pattern: str) -> dict:
    for parent in parent_edge.get("parents", []):
        if parent["pattern"] == pattern:
            return parent
    raise KeyError(pattern)


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


def _normalized_hamming(left: str, right: str) -> float:
    if len(left) != len(right):
        raise ValueError("Normalized Hamming distance requires equal-length strings.")
    if not left:
        return 0.0
    return float(sum(1 for a, b in zip(left, right) if a != b) / len(left))


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
