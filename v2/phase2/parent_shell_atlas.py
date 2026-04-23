from __future__ import annotations

import csv
import math
from pathlib import Path

from .defects import _require_edge, load_policy_fibers
from .kernel_common_support import _build_window_parent_structure, _index_edge_parents
from .return_lag import prepare_pattern_selection


_SUBTYPE_ORDER = {
    "prefix-only": 0,
    "suffix-only": 1,
    "bridge-both": 2,
    "unlinked": 3,
    "other": 9,
}

_SUBTYPE_LABELS = {
    "prefix-only": "Prefix-only",
    "suffix-only": "Suffix-only",
    "bridge-both": "Bridge-both",
    "unlinked": "Unlinked",
    "other": "Other",
}

_STATE_ORDER = {
    "dead": 0,
    "dark": 1,
    "disjoint": 2,
    "mixed": 3,
    "loss": 4,
    "spill": 5,
    "exact": 6,
    "other": 9,
}


def build_shell_structural_metadata(
    anchor_run: dict,
    *,
    low_scale: int,
    high_scale: int,
    top_patterns: int,
    pattern_selection: str,
    anchor_shell_patterns: set[str],
) -> dict:
    anchor_run_dir = Path(anchor_run["_run_dir"]).resolve()
    low_selection = prepare_pattern_selection(
        anchor_run,
        pattern_scale=low_scale,
        top_patterns=top_patterns,
        pattern_selection=pattern_selection,
    )
    high_selection = prepare_pattern_selection(
        anchor_run,
        pattern_scale=high_scale,
        top_patterns=top_patterns,
        pattern_selection=pattern_selection,
    )

    prefix_edge = _require_edge(
        load_policy_fibers(anchor_run_dir, "prefix"),
        target_m=low_scale,
        source_m=high_scale,
    )
    suffix_edge = _require_edge(
        load_policy_fibers(anchor_run_dir, "suffix"),
        target_m=low_scale,
        source_m=high_scale,
    )
    parent_structure = _build_window_parent_structure(
        parent_items=low_selection["selected_patterns"],
        high_pattern_set={str(item["pattern"]) for item in high_selection["selected_patterns"]},
        prefix_index=_index_edge_parents(prefix_edge),
        suffix_index=_index_edge_parents(suffix_edge),
    )

    pattern_metadata = {}
    for pattern in sorted(anchor_shell_patterns):
        structure = parent_structure.get(
            pattern,
            {"structural_child_patterns": set(), "child_policies": {}},
        )
        child_policies = structure.get("child_policies", {})
        policy_union = set()
        prefix_child_count = 0
        suffix_child_count = 0
        bridge_child_count = 0
        for policies in child_policies.values():
            policies = set(policies)
            policy_union.update(policies)
            if "prefix" in policies:
                prefix_child_count += 1
            if "suffix" in policies:
                suffix_child_count += 1
            if policies == {"prefix", "suffix"}:
                bridge_child_count += 1

        subtype = classify_shell_subtype(child_policies)
        structural_child_count = len(structure.get("structural_child_patterns", set()))
        pattern_metadata[pattern] = {
            "pattern": pattern,
            "shell_subtype": subtype,
            "shell_subtype_label": _SUBTYPE_LABELS[subtype],
            "structural_child_count": structural_child_count,
            "prefix_structural_child_count": prefix_child_count,
            "suffix_structural_child_count": suffix_child_count,
            "bridge_structural_child_count": bridge_child_count,
            "policy_union": sorted(policy_union),
            "monitored_flag": structural_child_count > 0,
        }

    structural_summary_rows = []
    total_shell_patterns = len(anchor_shell_patterns)
    for subtype in _ordered_subtypes(pattern_metadata.values()):
        subtype_items = [
            item for item in pattern_metadata.values() if item["shell_subtype"] == subtype
        ]
        monitored_count = sum(1 for item in subtype_items if item["monitored_flag"])
        structural_summary_rows.append(
            {
                "shell_subtype": subtype,
                "shell_subtype_label": _SUBTYPE_LABELS[subtype],
                "shell_pattern_count": len(subtype_items),
                "monitored_shell_pattern_count": monitored_count,
                "unmonitored_shell_pattern_count": len(subtype_items) - monitored_count,
                "shell_pattern_fraction": _ratio_or_none(len(subtype_items), total_shell_patterns),
            }
        )

    return {
        "anchor_run_dir": str(anchor_run_dir),
        "pattern_metadata": pattern_metadata,
        "structural_summary_rows": structural_summary_rows,
        "shell_pattern_count": total_shell_patterns,
        "monitored_shell_pattern_count": sum(
            1 for item in pattern_metadata.values() if item["monitored_flag"]
        ),
        "low_selection_count": len(low_selection["selected_patterns"]),
        "high_selection_count": len(high_selection["selected_patterns"]),
    }


def build_parent_shell_atlas_rows(
    lagaware_payload: dict,
    *,
    shell_metadata: dict,
) -> list[dict]:
    rows = []
    selection = lagaware_payload["selection"]
    pattern_metadata = shell_metadata["pattern_metadata"]
    dataset_path = lagaware_payload["_dataset_path"]
    for item in lagaware_payload.get("pattern_rows", []):
        pattern = str(item["pattern"])
        if pattern not in pattern_metadata:
            continue
        anchor_events = int(item.get("anchor_local_event_total", 0))
        if anchor_events <= 0:
            continue

        candidate_events = int(item.get("candidate_local_event_total", 0))
        retention = _ratio_or_none(candidate_events, anchor_events)
        bias = None
        deformation = None
        if retention is not None and retention > 0.0:
            bias = math.log2(retention)
            deformation = abs(bias)

        meta = pattern_metadata[pattern]
        rows.append(
            {
                "variant": str(item["variant"]),
                "source_label": str(item["source_label"]),
                "source_kind": str(item["source_kind"]),
                "null_model": item.get("null_model"),
                "null_seed": item.get("null_seed"),
                "anchor_variant": str(item["anchor_variant"]),
                "anchor_offset_bits": int(item["anchor_offset_bits"]),
                "window_bits": int(item["window_bits"]),
                "low_scale": int(item["low_scale"]),
                "high_scale": int(item["high_scale"]),
                "pattern_selection": str(item["pattern_selection"]),
                "kernel_mode": str(item["kernel_mode"]),
                "candidate_lag_bits": int(selection.get("candidate_lag_bits", 0)),
                "pattern": pattern,
                "shell_subtype": meta["shell_subtype"],
                "shell_subtype_label": meta["shell_subtype_label"],
                "structural_child_count": int(meta["structural_child_count"]),
                "prefix_structural_child_count": int(meta["prefix_structural_child_count"]),
                "suffix_structural_child_count": int(meta["suffix_structural_child_count"]),
                "bridge_structural_child_count": int(meta["bridge_structural_child_count"]),
                "monitored_flag": bool(meta["monitored_flag"]),
                "policy_union": list(meta["policy_union"]),
                "support_state": str(item.get("support_state", "other")),
                "anchor_local_parent_count": int(item.get("anchor_local_parent_count", 0)),
                "candidate_local_parent_count": int(item.get("candidate_local_parent_count", 0)),
                "anchor_local_event_total": anchor_events,
                "candidate_local_event_total": candidate_events,
                "anchor_local_child_count": int(item.get("anchor_local_child_count", 0)),
                "candidate_local_child_count": int(item.get("candidate_local_child_count", 0)),
                "common_child_count": int(item.get("common_child_count", 0)),
                "common_wj": item.get("common_wj"),
                "top_child_full_match": item.get("top_child_full_match"),
                "event_retention_ratio": retention,
                "event_bias_log2": bias,
                "event_deformation_log2": deformation,
                "dataset_path": dataset_path,
            }
        )

    rows.sort(
        key=lambda row: (
            row["variant"],
            _source_kind_sort_key(row["source_kind"], row.get("null_model")),
            row.get("null_seed") if row.get("null_seed") is not None else -1,
            _SUBTYPE_ORDER.get(row["shell_subtype"], 99),
            row["anchor_offset_bits"],
            row["pattern"],
        )
    )
    return rows


def summarize_parent_shell_patterns(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (
            row["variant"],
            row["source_label"],
            row["source_kind"],
            row.get("null_model"),
            row.get("null_seed"),
            row["shell_subtype"],
            row["pattern"],
        )
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for group in grouped.values():
        first = group[0]
        survivor_rows = [
            row for row in group if int(row["candidate_local_event_total"]) > 0
        ]
        anchor_sum = sum(int(row["anchor_local_event_total"]) for row in group)
        candidate_sum = sum(int(row["candidate_local_event_total"]) for row in group)
        gross_loss_pressure_sum = sum(
            max(
                int(row["anchor_local_event_total"])
                - int(row["candidate_local_event_total"]),
                0,
            )
            for row in group
        )
        net_deficit_mass = max(anchor_sum - candidate_sum, 0)
        survivor_anchor_sum = sum(int(row["anchor_local_event_total"]) for row in survivor_rows)
        survivor_candidate_sum = sum(
            int(row["candidate_local_event_total"]) for row in survivor_rows
        )
        bias_weighted_sum = sum(
            float(row["event_bias_log2"]) * int(row["anchor_local_event_total"])
            for row in survivor_rows
            if row["event_bias_log2"] is not None
        )
        deformation_weighted_sum = sum(
            float(row["event_deformation_log2"]) * int(row["anchor_local_event_total"])
            for row in survivor_rows
            if row["event_deformation_log2"] is not None
        )
        summary_rows.append(
            {
                "variant": first["variant"],
                "source_label": first["source_label"],
                "source_kind": first["source_kind"],
                "null_model": first.get("null_model"),
                "null_seed": first.get("null_seed"),
                "shell_subtype": first["shell_subtype"],
                "shell_subtype_label": first["shell_subtype_label"],
                "pattern": first["pattern"],
                "structural_child_count": first["structural_child_count"],
                "prefix_structural_child_count": first["prefix_structural_child_count"],
                "suffix_structural_child_count": first["suffix_structural_child_count"],
                "bridge_structural_child_count": first["bridge_structural_child_count"],
                "active_offset_count": len(group),
                "exact_offset_count": sum(
                    1 for row in group if row["support_state"] == "exact"
                ),
                "dead_offset_count": sum(
                    1 for row in group if row["support_state"] == "dead"
                ),
                "dark_offset_count": sum(
                    1 for row in group if row["support_state"] == "dark"
                ),
                "anchor_event_total_sum": anchor_sum,
                "candidate_event_total_sum": candidate_sum,
                "partition_mass_retention_pooled": _ratio_or_none(candidate_sum, anchor_sum),
                "survivor_internal_retention_pooled": _ratio_or_none(
                    survivor_candidate_sum,
                    survivor_anchor_sum,
                ),
                "survivor_internal_bias_log2_mean_pooled": _ratio_or_none(
                    bias_weighted_sum,
                    survivor_anchor_sum,
                ),
                "survivor_internal_deformation_log2_mean_pooled": _ratio_or_none(
                    deformation_weighted_sum,
                    survivor_anchor_sum,
                ),
                "gross_loss_pressure_sum": gross_loss_pressure_sum,
                "net_deficit_mass": net_deficit_mass,
                "jitter_gap_mass": max(
                    gross_loss_pressure_sum - net_deficit_mass,
                    0,
                ),
                "dead_anchor_mass_fraction": _ratio_or_none(
                    sum(
                        int(row["anchor_local_event_total"])
                        for row in group
                        if row["support_state"] == "dead"
                    ),
                    anchor_sum,
                ),
                "dark_anchor_mass_fraction": _ratio_or_none(
                    sum(
                        int(row["anchor_local_event_total"])
                        for row in group
                        if row["support_state"] == "dark"
                    ),
                    anchor_sum,
                ),
            }
        )

    summary_rows.sort(
        key=lambda row: (
            row["variant"],
            _source_kind_sort_key(row["source_kind"], row.get("null_model")),
            row.get("null_seed") if row.get("null_seed") is not None else -1,
            _SUBTYPE_ORDER.get(row["shell_subtype"], 99),
            -int(row["net_deficit_mass"]),
            row["pattern"],
        )
    )
    return summary_rows


def summarize_parent_shell_subtypes(
    rows: list[dict],
    *,
    structural_summary_rows: list[dict],
) -> list[dict]:
    structural_counts = {
        row["shell_subtype"]: {
            "shell_pattern_count": int(row["shell_pattern_count"]),
            "monitored_shell_pattern_count": int(row["monitored_shell_pattern_count"]),
            "unmonitored_shell_pattern_count": int(row["unmonitored_shell_pattern_count"]),
        }
        for row in structural_summary_rows
    }

    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (
            row["variant"],
            row["source_label"],
            row["source_kind"],
            row.get("null_model"),
            row.get("null_seed"),
            row["shell_subtype"],
        )
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for group in grouped.values():
        first = group[0]
        subtype_counts = structural_counts.get(first["shell_subtype"], {})
        anchor_sum = sum(int(row["anchor_local_event_total"]) for row in group)
        candidate_sum = sum(int(row["candidate_local_event_total"]) for row in group)
        gross_loss_pressure_sum = sum(
            max(
                int(row["anchor_local_event_total"])
                - int(row["candidate_local_event_total"]),
                0,
            )
            for row in group
        )
        net_deficit_mass = max(anchor_sum - candidate_sum, 0)
        survivor_rows = [row for row in group if int(row["candidate_local_event_total"]) > 0]
        survivor_anchor_sum = sum(int(row["anchor_local_event_total"]) for row in survivor_rows)
        survivor_candidate_sum = sum(
            int(row["candidate_local_event_total"]) for row in survivor_rows
        )
        bias_weighted_sum = 0.0
        deformation_weighted_sum = 0.0
        for row in survivor_rows:
            anchor_events = int(row["anchor_local_event_total"])
            if row["event_bias_log2"] is not None:
                bias_weighted_sum += float(row["event_bias_log2"]) * anchor_events
            if row["event_deformation_log2"] is not None:
                deformation_weighted_sum += float(row["event_deformation_log2"]) * anchor_events

        summary_rows.append(
            {
                "variant": first["variant"],
                "source_label": first["source_label"],
                "source_kind": first["source_kind"],
                "null_model": first.get("null_model"),
                "null_seed": first.get("null_seed"),
                "shell_subtype": first["shell_subtype"],
                "shell_subtype_label": first["shell_subtype_label"],
                "shell_pattern_count": subtype_counts.get("shell_pattern_count", 0),
                "monitored_shell_pattern_count": subtype_counts.get(
                    "monitored_shell_pattern_count",
                    0,
                ),
                "unmonitored_shell_pattern_count": subtype_counts.get(
                    "unmonitored_shell_pattern_count",
                    0,
                ),
                "active_pattern_count": len({str(row["pattern"]) for row in group}),
                "anchor_event_total_sum": anchor_sum,
                "candidate_event_total_sum": candidate_sum,
                "partition_mass_retention_pooled": _ratio_or_none(candidate_sum, anchor_sum),
                "survivor_internal_retention_pooled": _ratio_or_none(
                    survivor_candidate_sum,
                    survivor_anchor_sum,
                ),
                "survivor_internal_bias_log2_mean_pooled": _ratio_or_none(
                    bias_weighted_sum,
                    survivor_anchor_sum,
                ),
                "survivor_internal_deformation_log2_mean_pooled": _ratio_or_none(
                    deformation_weighted_sum,
                    survivor_anchor_sum,
                ),
                "gross_loss_pressure_sum": gross_loss_pressure_sum,
                "net_deficit_mass": net_deficit_mass,
                "jitter_gap_mass": max(
                    gross_loss_pressure_sum - net_deficit_mass,
                    0,
                ),
                "dead_anchor_mass_fraction": _ratio_or_none(
                    sum(
                        int(row["anchor_local_event_total"])
                        for row in group
                        if row["support_state"] == "dead"
                    ),
                    anchor_sum,
                ),
                "dark_anchor_mass_fraction": _ratio_or_none(
                    sum(
                        int(row["anchor_local_event_total"])
                        for row in group
                        if row["support_state"] == "dark"
                    ),
                    anchor_sum,
                ),
            }
        )

    summary_rows.sort(
        key=lambda row: (
            row["variant"],
            _source_kind_sort_key(row["source_kind"], row.get("null_model")),
            row.get("null_seed") if row.get("null_seed") is not None else -1,
            _SUBTYPE_ORDER.get(row["shell_subtype"], 99),
        )
    )
    return summary_rows


def summarize_shell_deficit_concentration(
    pattern_summary_rows: list[dict],
) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in pattern_summary_rows:
        key = (
            row["variant"],
            row["source_label"],
            row["source_kind"],
            row.get("null_model"),
            row.get("null_seed"),
        )
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for group in grouped.values():
        first = group[0]
        ranked = sorted(
            group,
            key=lambda row: (
                -int(row["net_deficit_mass"]),
                _SUBTYPE_ORDER.get(row["shell_subtype"], 99),
                row["pattern"],
            ),
        )
        total_net_deficit = sum(int(row["net_deficit_mass"]) for row in ranked)
        total_gross_loss_pressure = sum(
            int(row["gross_loss_pressure_sum"]) for row in ranked
        )
        subtype_totals: dict[str, int] = {}
        for row in ranked:
            subtype = row["shell_subtype"]
            subtype_totals[subtype] = subtype_totals.get(subtype, 0) + int(
                row["net_deficit_mass"]
            )
        dominant_subtype = None
        dominant_subtype_share = None
        if subtype_totals and total_net_deficit > 0:
            dominant_subtype = max(
                subtype_totals,
                key=lambda item: (subtype_totals[item], -_SUBTYPE_ORDER.get(item, 99), item),
            )
            dominant_subtype_share = _ratio_or_none(
                subtype_totals[dominant_subtype],
                total_net_deficit,
            )

        summary_rows.append(
            {
                "variant": first["variant"],
                "source_label": first["source_label"],
                "source_kind": first["source_kind"],
                "null_model": first.get("null_model"),
                "null_seed": first.get("null_seed"),
                "pattern_count": len(ranked),
                "net_deficit_pattern_count": sum(
                    1 for row in ranked if int(row["net_deficit_mass"]) > 0
                ),
                "net_deficit_mass_sum": total_net_deficit,
                "gross_loss_pressure_sum": total_gross_loss_pressure,
                "jitter_gap_mass": max(
                    total_gross_loss_pressure - total_net_deficit,
                    0,
                ),
                "top1_net_deficit_share": _topk_share(
                    ranked,
                    total_net_deficit,
                    1,
                    "net_deficit_mass",
                ),
                "top3_net_deficit_share": _topk_share(
                    ranked,
                    total_net_deficit,
                    3,
                    "net_deficit_mass",
                ),
                "top5_net_deficit_share": _topk_share(
                    ranked,
                    total_net_deficit,
                    5,
                    "net_deficit_mass",
                ),
                "dominant_subtype": dominant_subtype,
                "dominant_subtype_label": None
                if dominant_subtype is None
                else _SUBTYPE_LABELS[dominant_subtype],
                "dominant_subtype_share": dominant_subtype_share,
                "top_deficit_patterns": [
                    {
                        "pattern": row["pattern"],
                        "shell_subtype": row["shell_subtype"],
                        "shell_subtype_label": row["shell_subtype_label"],
                        "net_deficit_mass": int(row["net_deficit_mass"]),
                        "net_deficit_share": _ratio_or_none(
                            int(row["net_deficit_mass"]),
                            total_net_deficit,
                        ),
                        "gross_loss_pressure_sum": int(row["gross_loss_pressure_sum"]),
                        "jitter_gap_mass": int(row["jitter_gap_mass"]),
                        "partition_mass_retention_pooled": row["partition_mass_retention_pooled"],
                        "dead_offset_count": row["dead_offset_count"],
                        "dark_offset_count": row["dark_offset_count"],
                    }
                    for row in ranked[:10]
                ],
            }
        )

    summary_rows.sort(
        key=lambda row: (
            row["variant"],
            _source_kind_sort_key(row["source_kind"], row.get("null_model")),
            row.get("null_seed") if row.get("null_seed") is not None else -1,
        )
    )
    return summary_rows


def render_parent_shell_atlas_report(
    selection: dict,
    structural_summary_rows: list[dict],
    subtype_summary_rows: list[dict],
    concentration_rows: list[dict],
    pattern_summary_rows: list[dict],
) -> str:
    lines = [
        "# Phase 2 Parent Shell Atlas",
        "",
        "## Selection",
        "",
        f"- Anchor variant: {selection['anchor_variant']}",
        f"- Candidate variant: {selection['candidate_variant']}",
        f"- Low/high scales: {selection['low_scale']} -> {selection['high_scale']}",
        f"- Pattern selection: {selection['pattern_selection']}",
        f"- Top patterns: {selection['top_patterns']}",
        f"- Candidate lag bits: {selection['candidate_lag_bits']}",
        f"- Revalidation summary: `{selection['revalidation_summary_path']}`",
        f"- Lag-aware dataset: `{selection['lagaware_dataset_path']}`",
        f"- Anchor run: `{selection['anchor_run_dir']}`",
        "",
        "## Structural Shell Taxonomy",
        "",
        "| Subtype | Shell patterns | Monitored | Unmonitored | Fraction |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in structural_summary_rows:
        lines.append(
            f"| {row['shell_subtype_label']} | {row['shell_pattern_count']} | "
            f"{row['monitored_shell_pattern_count']} | {row['unmonitored_shell_pattern_count']} | "
            f"{_fmt(row['shell_pattern_fraction'])} |"
        )

    lines.extend(
        [
            "",
            "## Subtype Readout",
            "",
            "| Variant | Source | Subtype | Active / monitored | Part ret | Surv ret | Surv def | Surv bias | Dead m | Dark m | Net def | Gross loss | Jitter gap |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in subtype_summary_rows:
        lines.append(
            f"| {row['variant']} | {row['source_label']} | {row['shell_subtype_label']} | "
            f"{row['active_pattern_count']} / {row['monitored_shell_pattern_count']} | "
            f"{_fmt(row['partition_mass_retention_pooled'])} | "
            f"{_fmt(row['survivor_internal_retention_pooled'])} | "
            f"{_fmt(row['survivor_internal_deformation_log2_mean_pooled'])} | "
            f"{_fmt(row['survivor_internal_bias_log2_mean_pooled'])} | "
            f"{_fmt(row['dead_anchor_mass_fraction'])} | "
            f"{_fmt(row['dark_anchor_mass_fraction'])} | "
            f"{row['net_deficit_mass']} | "
            f"{row['gross_loss_pressure_sum']} | "
            f"{row['jitter_gap_mass']} |"
        )

    lines.extend(
        [
            "",
            "## Net Deficit Concentration",
            "",
            "| Variant | Source | Net def | Gross loss | Jitter gap | Top1 | Top3 | Top5 | Dominant subtype | Dom share |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
        ]
    )
    for row in concentration_rows:
        lines.append(
            f"| {row['variant']} | {row['source_label']} | {row['net_deficit_mass_sum']} | "
            f"{row['gross_loss_pressure_sum']} | {row['jitter_gap_mass']} | "
            f"{_fmt(row['top1_net_deficit_share'])} | {_fmt(row['top3_net_deficit_share'])} | "
            f"{_fmt(row['top5_net_deficit_share'])} | "
            f"{row['dominant_subtype_label'] or '-'} | {_fmt(row['dominant_subtype_share'])} |"
        )

    lines.extend(
        [
            "",
            "## Top Net Deficit Patterns",
            "",
        ]
    )
    for case in concentration_rows:
        heading = f"{case['variant']} | {case['source_label']}"
        if case.get("null_model"):
            heading += f" | {case['null_model']}"
        if case.get("null_seed") is not None:
            heading += f" | seed {case['null_seed']}"
        lines.append(f"### {heading}")
        lines.append("")
        if not case["top_deficit_patterns"]:
            lines.append("No positive shell deficit under the current monitored atlas.")
            lines.append("")
            continue
        lines.append("| Pattern | Subtype | Net def | Share | Gross loss | Jitter gap | Part ret | Dead wins | Dark wins |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for item in case["top_deficit_patterns"][:5]:
            lines.append(
                f"| `{item['pattern']}` | {item['shell_subtype_label']} | "
                f"{item['net_deficit_mass']} | {_fmt(item['net_deficit_share'])} | "
                f"{item['gross_loss_pressure_sum']} | {item['jitter_gap_mass']} | "
                f"{_fmt(item['partition_mass_retention_pooled'])} | "
                f"{item['dead_offset_count']} | {item['dark_offset_count']} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Notes",
            "",
            "- The shell taxonomy is defined from the anchor Phase 1 structure alone, but relative to the monitored high-scale child universe selected by the current bridge-linked top-k contract.",
            "- `Prefix-only` and `Suffix-only` are therefore observation-conditional shell subtypes, not absolute global topological classes.",
            "- `Unlinked` shell patterns belong to the anchor shell but have no monitored child in the selected high-scale bridge-linked universe, so they do not appear in the lag-aware kernel rows.",
            "- `Net def` is computed after pooling all windows pattern by pattern: `max(sum(anchor) - sum(candidate), 0)`.",
            "- `Gross loss` is the sum of positive per-window shell loss `sum(max(anchor - candidate, 0))`; it captures local loss pressure without allowing gain windows to cancel it.",
            "- `Jitter gap = gross loss - net def` captures the extra non-cancelling temporal dispersion beyond the pooled net deficit.",
            "- The atlas is diagnostic. It localizes shell loss inside the current monitored child universe; it is not yet a final law object.",
        ]
    )
    return "\n".join(lines)


def render_parent_shell_atlas_console_summary(
    structural_summary_rows: list[dict],
    subtype_summary_rows: list[dict],
    concentration_rows: list[dict],
) -> str:
    variant_width = 10
    source_width = 28
    subtype_width = 13
    lines = [
        "Phase 2 parent shell atlas",
        "-" * 154,
        "Shell structure: "
        + ", ".join(
            f"{row['shell_subtype_label']}={row['shell_pattern_count']}"
            for row in structural_summary_rows
        ),
        "",
        f"{'variant':<{variant_width}} {'source':<{source_width}} {'subtype':<{subtype_width}} {'act/mon':>8}{'partRet':>10}{'survDef':>10}{'deadM':>10}{'darkM':>10}{'netDef':>10}{'jGap':>10}",
    ]
    for row in subtype_summary_rows:
        active_label = f"{row['active_pattern_count']}/{row['monitored_shell_pattern_count']}"
        lines.append(
            f"{_truncate(row['variant'], variant_width):<{variant_width}} "
            f"{_truncate(row['source_label'], source_width):<{source_width}} "
            f"{_truncate(row['shell_subtype_label'], subtype_width):<{subtype_width}} "
            f"{active_label:>8}"
            f"{_fmt(row['partition_mass_retention_pooled']):>10}"
            f"{_fmt(row['survivor_internal_deformation_log2_mean_pooled']):>10}"
            f"{_fmt(row['dead_anchor_mass_fraction']):>10}"
            f"{_fmt(row['dark_anchor_mass_fraction']):>10}"
            f"{int(row['net_deficit_mass']):>10}"
            f"{int(row['jitter_gap_mass']):>10}"
        )

    lines.extend(
        [
            "",
            "Deficit concentration",
            "-" * 110,
            f"{'variant':<{variant_width}} {'source':<{source_width}} {'netDef':>10}{'gross':>10}{'jGap':>10}{'top1':>10}{'top3':>10}{'top5':>10}{'dom subtype':>16}",
        ]
    )
    for row in concentration_rows:
        lines.append(
            f"{_truncate(row['variant'], variant_width):<{variant_width}} "
            f"{_truncate(row['source_label'], source_width):<{source_width}} "
            f"{int(row['net_deficit_mass_sum']):>10}"
            f"{int(row['gross_loss_pressure_sum']):>10}"
            f"{int(row['jitter_gap_mass']):>10}"
            f"{_fmt(row['top1_net_deficit_share']):>10}"
            f"{_fmt(row['top3_net_deficit_share']):>10}"
            f"{_fmt(row['top5_net_deficit_share']):>10}"
            f"{_truncate(row['dominant_subtype_label'] or '-', 16):>16}"
        )
    return "\n".join(lines)


def write_parent_shell_atlas_csv(
    rows: list[dict],
    csv_path: Path,
) -> None:
    fieldnames = [
        "variant",
        "source_label",
        "source_kind",
        "null_model",
        "null_seed",
        "anchor_variant",
        "anchor_offset_bits",
        "window_bits",
        "low_scale",
        "high_scale",
        "pattern_selection",
        "kernel_mode",
        "candidate_lag_bits",
        "pattern",
        "shell_subtype",
        "shell_subtype_label",
        "structural_child_count",
        "prefix_structural_child_count",
        "suffix_structural_child_count",
        "bridge_structural_child_count",
        "support_state",
        "anchor_local_parent_count",
        "candidate_local_parent_count",
        "anchor_local_event_total",
        "candidate_local_event_total",
        "anchor_local_child_count",
        "candidate_local_child_count",
        "common_child_count",
        "common_wj",
        "top_child_full_match",
        "event_retention_ratio",
        "event_bias_log2",
        "event_deformation_log2",
        "dataset_path",
    ]
    _write_csv(rows, fieldnames, csv_path)


def write_parent_shell_structural_csv(rows: list[dict], csv_path: Path) -> None:
    fieldnames = [
        "shell_subtype",
        "shell_subtype_label",
        "shell_pattern_count",
        "monitored_shell_pattern_count",
        "unmonitored_shell_pattern_count",
        "shell_pattern_fraction",
    ]
    _write_csv(rows, fieldnames, csv_path)


def write_parent_shell_subtype_csv(rows: list[dict], csv_path: Path) -> None:
    fieldnames = [
        "variant",
        "source_label",
        "source_kind",
        "null_model",
        "null_seed",
        "shell_subtype",
        "shell_subtype_label",
        "shell_pattern_count",
        "monitored_shell_pattern_count",
        "unmonitored_shell_pattern_count",
        "active_pattern_count",
        "anchor_event_total_sum",
        "candidate_event_total_sum",
        "partition_mass_retention_pooled",
        "survivor_internal_retention_pooled",
        "survivor_internal_bias_log2_mean_pooled",
        "survivor_internal_deformation_log2_mean_pooled",
        "net_deficit_mass",
        "gross_loss_pressure_sum",
        "jitter_gap_mass",
        "dead_anchor_mass_fraction",
        "dark_anchor_mass_fraction",
    ]
    _write_csv(rows, fieldnames, csv_path)


def write_parent_shell_pattern_csv(rows: list[dict], csv_path: Path) -> None:
    fieldnames = [
        "variant",
        "source_label",
        "source_kind",
        "null_model",
        "null_seed",
        "shell_subtype",
        "shell_subtype_label",
        "pattern",
        "structural_child_count",
        "prefix_structural_child_count",
        "suffix_structural_child_count",
        "bridge_structural_child_count",
        "active_offset_count",
        "exact_offset_count",
        "dead_offset_count",
        "dark_offset_count",
        "anchor_event_total_sum",
        "candidate_event_total_sum",
        "partition_mass_retention_pooled",
        "survivor_internal_retention_pooled",
        "survivor_internal_bias_log2_mean_pooled",
        "survivor_internal_deformation_log2_mean_pooled",
        "net_deficit_mass",
        "gross_loss_pressure_sum",
        "jitter_gap_mass",
        "dead_anchor_mass_fraction",
        "dark_anchor_mass_fraction",
    ]
    _write_csv(rows, fieldnames, csv_path)


def write_parent_shell_concentration_csv(rows: list[dict], csv_path: Path) -> None:
    fieldnames = [
        "variant",
        "source_label",
        "source_kind",
        "null_model",
        "null_seed",
        "pattern_count",
        "net_deficit_pattern_count",
        "net_deficit_mass_sum",
        "gross_loss_pressure_sum",
        "jitter_gap_mass",
        "top1_net_deficit_share",
        "top3_net_deficit_share",
        "top5_net_deficit_share",
        "dominant_subtype",
        "dominant_subtype_label",
        "dominant_subtype_share",
    ]
    _write_csv(rows, fieldnames, csv_path)


def classify_shell_subtype(child_policies: dict[str, set[str]]) -> str:
    if not child_policies:
        return "unlinked"
    policy_union = set()
    for policies in child_policies.values():
        policy_union.update(set(policies))
    if policy_union == {"prefix"}:
        return "prefix-only"
    if policy_union == {"suffix"}:
        return "suffix-only"
    if policy_union == {"prefix", "suffix"}:
        return "bridge-both"
    return "other"


def _ordered_subtypes(items) -> list[str]:
    seen = {
        str(item["shell_subtype"])
        for item in items
    }
    return sorted(seen, key=lambda item: (_SUBTYPE_ORDER.get(item, 99), item))


def _topk_share(
    rows: list[dict],
    total_mass: int,
    k: int,
    field_name: str,
) -> float | None:
    if total_mass <= 0:
        return None
    return float(
        sum(int(row[field_name]) for row in rows[:k])
        / total_mass
    )


def _source_kind_sort_key(source_kind: str, null_model: str | None) -> tuple[int, str]:
    if source_kind == "observed":
        return (0, "observed")
    if null_model == "markov1":
        return (1, "markov1")
    if null_model == "matched-lz":
        return (2, "matched-lz")
    return (9, source_kind)


def _write_csv(rows: list[dict], fieldnames: list[str], csv_path: Path) -> None:
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _ratio_or_none(numerator: int | float, denominator: int | float) -> float | None:
    if float(denominator) <= 0.0:
        return None
    return float(numerator) / float(denominator)


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def _truncate(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    if width <= 3:
        return value[:width]
    return value[: width - 3] + "..."
