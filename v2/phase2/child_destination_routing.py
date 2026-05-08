from __future__ import annotations

import csv
import json
from pathlib import Path

from .kernel_support_dropout import classify_support_state


def build_child_destination_routing_readout(
    atlas_payload: dict,
    *,
    allow_preview_truncated: bool = False,
) -> dict:
    """Compare anchor and candidate child destinations for monitored shell parents."""
    selection = dict(atlas_payload.get("selection", {}))
    atlas_rows = atlas_payload.get("atlas_rows", [])
    if not atlas_rows:
        raise ValueError("Atlas payload does not contain atlas_rows.")

    lagaware_path = Path(selection.get("lagaware_dataset_path", ""))
    if not lagaware_path.is_file():
        lagaware_path = _recover_lagaware_path_from_rows(atlas_rows)
    if lagaware_path is None or not lagaware_path.is_file():
        raise ValueError("Could not resolve lag-aware dataset path from atlas payload.")

    lagaware_payload = _load_json(lagaware_path)
    shell_metadata = build_shell_metadata(atlas_rows)
    if not shell_metadata:
        raise ValueError("No monitored shell patterns found in atlas payload.")

    routing_rows = build_routing_rows(
        lagaware_payload,
        shell_metadata=shell_metadata,
    )
    if not allow_preview_truncated:
        _raise_on_preview_truncation(routing_rows)
    pattern_rows = summarize_pattern_routing(routing_rows)
    source_rows = summarize_source_routing(pattern_rows)

    return {
        "selection": {
            **selection,
            "lagaware_dataset_path": str(lagaware_path),
        },
        "notes": [
            "N2-08 compares child destinations inside monitored shell parents.",
            "The primary child-retention metric is jitter-compensated: child masses are pooled by destination across windows before overlap is computed.",
            "Window-synchronous overlap is retained as a diagnostic control, not as the primary routing claim.",
            "Monitored reroute only means reroute into the tracked child universe; off-universe destination changes remain counted as deficit.",
            "Preview truncation is rejected by default because N2-08 requires complete monitored child mass maps.",
        ],
        "source_summary": source_rows,
        "pattern_summary": pattern_rows,
        "routing_rows": routing_rows,
    }


def build_shell_metadata(atlas_rows: list[dict]) -> dict[str, dict]:
    metadata: dict[str, dict] = {}
    for row in atlas_rows:
        if not bool(row.get("monitored_flag", False)):
            continue
        pattern = str(row["pattern"])
        current = metadata.get(pattern)
        item = {
            "pattern": pattern,
            "shell_subtype": str(row.get("shell_subtype", "")),
            "shell_subtype_label": str(row.get("shell_subtype_label", "")),
            "structural_child_count": int(row.get("structural_child_count", 0)),
            "prefix_structural_child_count": int(
                row.get("prefix_structural_child_count", 0)
            ),
            "suffix_structural_child_count": int(
                row.get("suffix_structural_child_count", 0)
            ),
            "bridge_structural_child_count": int(
                row.get("bridge_structural_child_count", 0)
            ),
        }
        if current is None:
            metadata[pattern] = item
    return metadata


def build_routing_rows(
    lagaware_payload: dict,
    *,
    shell_metadata: dict[str, dict],
) -> list[dict]:
    rows = []
    for source_row in lagaware_payload.get("kernel_rows", []):
        kernel_result = source_row.get("kernel_result", {})
        for item in kernel_result.get("per_pattern", []):
            pattern = str(item["pattern"])
            meta = shell_metadata.get(pattern)
            if meta is None:
                continue
            anchor_total = int(item.get("anchor_local_event_total", 0))
            if anchor_total <= 0:
                continue
            candidate_total = int(item.get("candidate_local_event_total", 0))
            anchor_preview = list(item.get("anchor_kernel_preview", []))
            candidate_preview = list(item.get("candidate_kernel_preview", []))
            anchor_child_count = int(item.get("anchor_local_child_count", 0))
            candidate_child_count = int(item.get("candidate_local_child_count", 0))
            preview_truncated = (
                len(anchor_preview) < anchor_child_count
                or len(candidate_preview) < candidate_child_count
            )

            anchor_child_mass = _child_mass_map(anchor_preview, anchor_total)
            candidate_child_mass = _child_mass_map(candidate_preview, candidate_total)
            common_children = sorted(set(anchor_child_mass) & set(candidate_child_mass))
            anchor_children = set(anchor_child_mass)

            window_child_overlap_mass = sum(
                min(anchor_child_mass[child], candidate_child_mass[child])
                for child in common_children
            )
            candidate_on_anchor_child_mass = sum(
                candidate_child_mass.get(child, 0.0)
                for child in anchor_children
            )
            candidate_reroute_mass = sum(
                mass
                for child, mass in candidate_child_mass.items()
                if child not in anchor_children
            )
            same_child_excess_mass = sum(
                max(candidate_child_mass.get(child, 0.0) - anchor_child_mass.get(child, 0.0), 0.0)
                for child in common_children
            )
            window_anchor_child_deficit_mass = max(
                float(anchor_total) - window_child_overlap_mass,
                0.0,
            )

            row = {
                "variant": source_row["variant"],
                "source_label": source_row["source_label"],
                "source_kind": source_row["source_kind"],
                "null_model": source_row.get("null_model"),
                "null_seed": source_row.get("null_seed"),
                "anchor_variant": source_row["anchor_variant"],
                "anchor_offset_bits": int(source_row["anchor_offset_bits"]),
                "candidate_offset_bits": int(source_row.get("candidate_offset_bits", 0)),
                "candidate_lag_bits": int(source_row.get("candidate_lag_bits", 0)),
                "window_bits": int(source_row.get("window_bits", 0)),
                "low_scale": int(source_row.get("low_scale", 0)),
                "high_scale": int(source_row.get("high_scale", 0)),
                "pattern_selection": str(source_row.get("pattern_selection", "")),
                "kernel_mode": str(source_row.get("kernel_mode", "")),
                "pattern": pattern,
                "shell_subtype": meta["shell_subtype"],
                "shell_subtype_label": meta["shell_subtype_label"],
                "structural_child_count": meta["structural_child_count"],
                "prefix_structural_child_count": meta["prefix_structural_child_count"],
                "suffix_structural_child_count": meta["suffix_structural_child_count"],
                "bridge_structural_child_count": meta["bridge_structural_child_count"],
                "support_state": classify_support_state(item),
                "anchor_local_event_total": anchor_total,
                "candidate_local_event_total": candidate_total,
                "anchor_local_child_count": anchor_child_count,
                "candidate_local_child_count": candidate_child_count,
                "common_child_count": int(item.get("common_child_count", 0)),
                "anchor_child_mass_map": _sorted_mass_map(anchor_child_mass),
                "candidate_child_mass_map": _sorted_mass_map(candidate_child_mass),
                "window_child_overlap_mass": window_child_overlap_mass,
                "window_anchor_child_deficit_mass": window_anchor_child_deficit_mass,
                "child_overlap_mass": window_child_overlap_mass,
                "anchor_child_deficit_mass": window_anchor_child_deficit_mass,
                "candidate_on_anchor_child_mass": candidate_on_anchor_child_mass,
                "candidate_reroute_mass": candidate_reroute_mass,
                "same_child_excess_mass": same_child_excess_mass,
                "child_destination_retention": _ratio_or_none(
                    window_child_overlap_mass,
                    anchor_total,
                ),
                "candidate_reroute_share": _ratio_or_none(
                    candidate_reroute_mass,
                    candidate_total,
                ),
                "candidate_on_anchor_child_share": _ratio_or_none(
                    candidate_on_anchor_child_mass,
                    candidate_total,
                ),
                "support_jaccard": item.get("support_jaccard"),
                "common_wj": item.get("common_wj"),
                "top_child_full_match": item.get("top_child_full_match"),
                "top_child_common_match": item.get("top_child_common_match"),
                "anchor_top_child": item.get("anchor_top_child"),
                "candidate_top_child": item.get("candidate_top_child"),
                "anchor_child_patterns": sorted(anchor_child_mass),
                "candidate_child_patterns": sorted(candidate_child_mass),
                "common_child_patterns": common_children,
                "preview_truncated": preview_truncated,
            }
            rows.append(row)

    rows.sort(
        key=lambda row: (
            row["variant"],
            _source_kind_sort_key(row["source_kind"], row.get("null_model")),
            row.get("null_seed") if row.get("null_seed") is not None else -1,
            row["shell_subtype"],
            row["pattern"],
            row["anchor_offset_bits"],
        )
    )
    return rows


def summarize_pattern_routing(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (
            row["variant"],
            row["source_label"],
            row["source_kind"],
            row.get("null_model"),
            row.get("null_seed"),
            row["shell_subtype"],
            row["shell_subtype_label"],
            row["pattern"],
        )
        grouped.setdefault(key, []).append(row)

    summaries = []
    for group in grouped.values():
        first = group[0]
        anchor_total = sum(float(row["anchor_local_event_total"]) for row in group)
        candidate_total = sum(float(row["candidate_local_event_total"]) for row in group)
        anchor_child_mass = _sum_child_mass_maps(group, "anchor_child_mass_map")
        candidate_child_mass = _sum_child_mass_maps(group, "candidate_child_mass_map")
        common_children = sorted(set(anchor_child_mass) & set(candidate_child_mass))
        anchor_children = set(anchor_child_mass)
        child_overlap = sum(
            min(anchor_child_mass[child], candidate_child_mass[child])
            for child in common_children
        )
        window_child_overlap = sum(
            float(row["window_child_overlap_mass"]) for row in group
        )
        child_deficit = max(anchor_total - child_overlap, 0.0)
        window_child_deficit = sum(
            float(row["window_anchor_child_deficit_mass"]) for row in group
        )
        candidate_on_anchor_child = sum(
            candidate_child_mass.get(child, 0.0)
            for child in anchor_children
        )
        candidate_reroute = sum(
            mass
            for child, mass in candidate_child_mass.items()
            if child not in anchor_children
        )
        same_child_excess = sum(
            max(candidate_child_mass.get(child, 0.0) - anchor_child_mass.get(child, 0.0), 0.0)
            for child in common_children
        )
        summaries.append(
            {
                "variant": first["variant"],
                "source_label": first["source_label"],
                "source_kind": first["source_kind"],
                "null_model": first.get("null_model"),
                "null_seed": first.get("null_seed"),
                "anchor_variant": first["anchor_variant"],
                "candidate_lag_bits": first["candidate_lag_bits"],
                "low_scale": first["low_scale"],
                "high_scale": first["high_scale"],
                "pattern_selection": first["pattern_selection"],
                "kernel_mode": first["kernel_mode"],
                "shell_subtype": first["shell_subtype"],
                "shell_subtype_label": first["shell_subtype_label"],
                "pattern": first["pattern"],
                "structural_child_count": first["structural_child_count"],
                "window_count": len(group),
                "anchor_event_total_sum": anchor_total,
                "candidate_event_total_sum": candidate_total,
                "anchor_child_mass_map": _sorted_mass_map(anchor_child_mass),
                "candidate_child_mass_map": _sorted_mass_map(candidate_child_mass),
                "jitter_compensated_child_overlap_mass_sum": child_overlap,
                "window_synchronous_child_overlap_mass_sum": window_child_overlap,
                "child_overlap_mass_sum": child_overlap,
                "anchor_child_deficit_mass_sum": child_deficit,
                "window_synchronous_anchor_child_deficit_mass_sum": window_child_deficit,
                "candidate_on_anchor_child_mass_sum": candidate_on_anchor_child,
                "monitored_candidate_reroute_mass_sum": candidate_reroute,
                "candidate_reroute_mass_sum": candidate_reroute,
                "same_child_excess_mass_sum": same_child_excess,
                "child_destination_retention_pooled": _ratio_or_none(
                    child_overlap,
                    anchor_total,
                ),
                "window_synchronous_child_destination_retention_pooled": _ratio_or_none(
                    window_child_overlap,
                    anchor_total,
                ),
                "candidate_reroute_share_pooled": _ratio_or_none(
                    candidate_reroute,
                    candidate_total,
                ),
                "monitored_candidate_reroute_share_pooled": _ratio_or_none(
                    candidate_reroute,
                    candidate_total,
                ),
                "candidate_on_anchor_child_share_pooled": _ratio_or_none(
                    candidate_on_anchor_child,
                    candidate_total,
                ),
                "top_child_full_match_mass_fraction": _mass_fraction(
                    group,
                    flag="top_child_full_match",
                ),
                "preview_truncated": any(bool(row["preview_truncated"]) for row in group),
                "routing_class": classify_routing(
                    child_retention=_ratio_or_none(child_overlap, anchor_total),
                    reroute_share=_ratio_or_none(candidate_reroute, candidate_total),
                    candidate_total=candidate_total,
                ),
            }
        )

    summaries.sort(
        key=lambda row: (
            row["variant"],
            _source_kind_sort_key(row["source_kind"], row.get("null_model")),
            row.get("null_seed") if row.get("null_seed") is not None else -1,
            -float(row["anchor_child_deficit_mass_sum"]),
            -float(row["candidate_reroute_mass_sum"]),
            row["pattern"],
        )
    )
    return summaries


def summarize_source_routing(pattern_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in pattern_rows:
        key = (
            row["variant"],
            row["source_label"],
            row["source_kind"],
            row.get("null_model"),
            row.get("null_seed"),
        )
        grouped.setdefault(key, []).append(row)

    summaries = []
    for group in grouped.values():
        first = group[0]
        anchor_total = sum(float(row["anchor_event_total_sum"]) for row in group)
        candidate_total = sum(float(row["candidate_event_total_sum"]) for row in group)
        child_overlap = sum(float(row["child_overlap_mass_sum"]) for row in group)
        window_child_overlap = sum(
            float(row["window_synchronous_child_overlap_mass_sum"]) for row in group
        )
        child_deficit = sum(float(row["anchor_child_deficit_mass_sum"]) for row in group)
        window_child_deficit = sum(
            float(row["window_synchronous_anchor_child_deficit_mass_sum"])
            for row in group
        )
        candidate_on_anchor_child = sum(
            float(row["candidate_on_anchor_child_mass_sum"]) for row in group
        )
        candidate_reroute = sum(float(row["candidate_reroute_mass_sum"]) for row in group)
        same_child_excess = sum(float(row["same_child_excess_mass_sum"]) for row in group)
        child_retention = _ratio_or_none(child_overlap, anchor_total)
        window_child_retention = _ratio_or_none(window_child_overlap, anchor_total)
        reroute_share = _ratio_or_none(candidate_reroute, candidate_total)
        class_counts = _count_classes(group)
        summaries.append(
            {
                "variant": first["variant"],
                "source_label": first["source_label"],
                "source_kind": first["source_kind"],
                "null_model": first.get("null_model"),
                "null_seed": first.get("null_seed"),
                "anchor_variant": first["anchor_variant"],
                "candidate_lag_bits": first["candidate_lag_bits"],
                "low_scale": first["low_scale"],
                "high_scale": first["high_scale"],
                "pattern_selection": first["pattern_selection"],
                "kernel_mode": first["kernel_mode"],
                "pattern_count": len(group),
                "anchor_event_total_sum": anchor_total,
                "candidate_event_total_sum": candidate_total,
                "jitter_compensated_child_overlap_mass_sum": child_overlap,
                "window_synchronous_child_overlap_mass_sum": window_child_overlap,
                "child_overlap_mass_sum": child_overlap,
                "anchor_child_deficit_mass_sum": child_deficit,
                "window_synchronous_anchor_child_deficit_mass_sum": window_child_deficit,
                "candidate_on_anchor_child_mass_sum": candidate_on_anchor_child,
                "monitored_candidate_reroute_mass_sum": candidate_reroute,
                "candidate_reroute_mass_sum": candidate_reroute,
                "same_child_excess_mass_sum": same_child_excess,
                "child_destination_retention_pooled": child_retention,
                "window_synchronous_child_destination_retention_pooled": window_child_retention,
                "candidate_reroute_share_pooled": reroute_share,
                "monitored_candidate_reroute_share_pooled": reroute_share,
                "candidate_on_anchor_child_share_pooled": _ratio_or_none(
                    candidate_on_anchor_child,
                    candidate_total,
                ),
                "top_child_full_match_mass_fraction": _weighted_mean(
                    group,
                    "top_child_full_match_mass_fraction",
                    "anchor_event_total_sum",
                ),
                "preview_truncated": any(bool(row["preview_truncated"]) for row in group),
                "dominant_routing_class": classify_routing(
                    child_retention=child_retention,
                    reroute_share=reroute_share,
                    candidate_total=candidate_total,
                ),
                "pattern_routing_class_counts": class_counts,
                "top_deficit_patterns": _top_patterns(
                    group,
                    field="anchor_child_deficit_mass_sum",
                    limit=10,
                ),
                "top_reroute_patterns": _top_patterns(
                    group,
                    field="candidate_reroute_mass_sum",
                    limit=10,
                ),
            }
        )

    summaries.sort(
        key=lambda row: (
            row["variant"],
            _source_kind_sort_key(row["source_kind"], row.get("null_model")),
            row.get("null_seed") if row.get("null_seed") is not None else -1,
        )
    )
    return summaries


def classify_routing(
    *,
    child_retention: float | None,
    reroute_share: float | None,
    candidate_total: float,
) -> str:
    if candidate_total <= 0:
        return "destination-collapse"
    retention = child_retention if child_retention is not None else 0.0
    reroute = reroute_share if reroute_share is not None else 0.0
    if reroute >= 0.25:
        return "rerouted"
    if reroute <= 0.10:
        if retention >= 0.75:
            return "same-destination-conserved"
        if retention >= 0.50:
            return "same-destination-partial"
        if retention > 0.0:
            return "same-destination-deficit"
        return "destination-collapse"
    return "mixed-destination"


def write_source_routing_csv(rows: list[dict], csv_path: Path) -> None:
    fieldnames = [
        "variant",
        "source_label",
        "source_kind",
        "null_model",
        "null_seed",
        "anchor_variant",
        "candidate_lag_bits",
        "low_scale",
        "high_scale",
        "pattern_selection",
        "kernel_mode",
        "pattern_count",
        "anchor_event_total_sum",
        "candidate_event_total_sum",
        "jitter_compensated_child_overlap_mass_sum",
        "window_synchronous_child_overlap_mass_sum",
        "child_overlap_mass_sum",
        "anchor_child_deficit_mass_sum",
        "window_synchronous_anchor_child_deficit_mass_sum",
        "candidate_on_anchor_child_mass_sum",
        "monitored_candidate_reroute_mass_sum",
        "candidate_reroute_mass_sum",
        "same_child_excess_mass_sum",
        "child_destination_retention_pooled",
        "window_synchronous_child_destination_retention_pooled",
        "candidate_reroute_share_pooled",
        "monitored_candidate_reroute_share_pooled",
        "candidate_on_anchor_child_share_pooled",
        "top_child_full_match_mass_fraction",
        "preview_truncated",
        "dominant_routing_class",
        "pattern_routing_class_counts",
    ]
    _write_csv(rows, csv_path, fieldnames)


def write_pattern_routing_csv(rows: list[dict], csv_path: Path) -> None:
    fieldnames = [
        "variant",
        "source_label",
        "source_kind",
        "null_model",
        "null_seed",
        "anchor_variant",
        "candidate_lag_bits",
        "low_scale",
        "high_scale",
        "pattern_selection",
        "kernel_mode",
        "shell_subtype",
        "shell_subtype_label",
        "pattern",
        "structural_child_count",
        "window_count",
        "anchor_event_total_sum",
        "candidate_event_total_sum",
        "jitter_compensated_child_overlap_mass_sum",
        "window_synchronous_child_overlap_mass_sum",
        "child_overlap_mass_sum",
        "anchor_child_deficit_mass_sum",
        "window_synchronous_anchor_child_deficit_mass_sum",
        "candidate_on_anchor_child_mass_sum",
        "monitored_candidate_reroute_mass_sum",
        "candidate_reroute_mass_sum",
        "same_child_excess_mass_sum",
        "child_destination_retention_pooled",
        "window_synchronous_child_destination_retention_pooled",
        "candidate_reroute_share_pooled",
        "monitored_candidate_reroute_share_pooled",
        "candidate_on_anchor_child_share_pooled",
        "top_child_full_match_mass_fraction",
        "preview_truncated",
        "routing_class",
    ]
    _write_csv(rows, csv_path, fieldnames)


def write_routing_rows_csv(rows: list[dict], csv_path: Path) -> None:
    fieldnames = [
        "variant",
        "source_label",
        "source_kind",
        "null_model",
        "null_seed",
        "anchor_variant",
        "anchor_offset_bits",
        "candidate_offset_bits",
        "candidate_lag_bits",
        "window_bits",
        "low_scale",
        "high_scale",
        "pattern_selection",
        "kernel_mode",
        "pattern",
        "shell_subtype",
        "shell_subtype_label",
        "structural_child_count",
        "support_state",
        "anchor_local_event_total",
        "candidate_local_event_total",
        "anchor_local_child_count",
        "candidate_local_child_count",
        "common_child_count",
        "anchor_child_mass_map",
        "candidate_child_mass_map",
        "window_child_overlap_mass",
        "window_anchor_child_deficit_mass",
        "child_overlap_mass",
        "anchor_child_deficit_mass",
        "candidate_on_anchor_child_mass",
        "candidate_reroute_mass",
        "same_child_excess_mass",
        "child_destination_retention",
        "candidate_reroute_share",
        "candidate_on_anchor_child_share",
        "support_jaccard",
        "common_wj",
        "top_child_full_match",
        "top_child_common_match",
        "anchor_top_child",
        "candidate_top_child",
        "anchor_child_patterns",
        "candidate_child_patterns",
        "common_child_patterns",
        "preview_truncated",
    ]
    _write_csv(rows, csv_path, fieldnames)


def render_child_destination_console_summary(source_rows: list[dict]) -> str:
    lines = [
        "Phase 2 child-destination routing readout",
        "-" * 148,
        (
            f"{'variant':<12} {'source':<30} {'patterns':>8} {'childRet':>9} "
            f"{'syncRet':>9} {'monRer':>9} {'topMatch':>9} {'deficit':>10} {'class':>28}"
        ),
    ]
    for row in source_rows:
        lines.append(
            f"{_clip(row['variant'], 12):<12} "
            f"{_clip(row['source_label'], 30):<30} "
            f"{int(row['pattern_count']):>8} "
            f"{_fmt_float(row['child_destination_retention_pooled']):>9} "
            f"{_fmt_float(row['window_synchronous_child_destination_retention_pooled']):>9} "
            f"{_fmt_float(row['monitored_candidate_reroute_share_pooled']):>9} "
            f"{_fmt_float(row['top_child_full_match_mass_fraction']):>9} "
            f"{_fmt_num(row['anchor_child_deficit_mass_sum']):>10} "
            f"{_clip(row['dominant_routing_class'], 28):>28}"
        )
    return "\n".join(lines)


def render_child_destination_report(readout: dict) -> str:
    selection = readout.get("selection", {})
    source_rows = readout["source_summary"]
    lines = [
        "# Phase 2 Child-Destination Routing",
        "",
        "## Selection",
        "",
        f"- Anchor variant: {selection.get('anchor_variant', '-')}",
        f"- Candidate variant: {selection.get('candidate_variant', '-')}",
        f"- Low/high scales: {selection.get('low_scale', '-')} -> {selection.get('high_scale', '-')}",
        f"- Pattern selection: {selection.get('pattern_selection', '-')}",
        f"- Top patterns: {selection.get('top_patterns', '-')}",
        f"- Candidate lag bits: {selection.get('candidate_lag_bits', '-')}",
        f"- Lag-aware dataset: `{selection.get('lagaware_dataset_path', '-')}`",
        "",
        "## Method Boundary",
        "",
        "This readout compares child destinations within monitored shell parents.",
        "The primary overlap pools mass by child destination across all windows before applying `min(anchor, candidate)`.",
        "The window-synchronous overlap is retained only as a diagnostic control for temporal jitter.",
        "`Monitored reroute` only measures candidate mass landing on tracked children outside the anchor child set.",
        "Candidate mass outside the tracked child universe cannot be classified as reroute here; it remains deficit/off-support.",
        "It does not infer cross-parent routing.",
        "",
        "## Source Summary",
        "",
        "| Variant | Source | Patterns | Child retention | Sync retention | Monitored reroute | Top-match mass | Child deficit | Class |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in source_rows:
        lines.append(
            f"| {row['variant']} | {row['source_label']} | {row['pattern_count']} | "
            f"{_fmt_float(row['child_destination_retention_pooled'])} | "
            f"{_fmt_float(row['window_synchronous_child_destination_retention_pooled'])} | "
            f"{_fmt_float(row['monitored_candidate_reroute_share_pooled'])} | "
            f"{_fmt_float(row['top_child_full_match_mass_fraction'])} | "
            f"{_fmt_num(row['anchor_child_deficit_mass_sum'])} | "
            f"{row['dominant_routing_class']} |"
        )

    lines.extend(["", "## Top Child-Deficit Patterns", ""])
    for row in source_rows:
        lines.append(f"### {row['variant']} | {row['source_label']}")
        lines.append("")
        patterns = row.get("top_deficit_patterns", [])
        if not patterns:
            lines.append("No child-destination deficit patterns.")
            lines.append("")
            continue
        lines.append("| Pattern | Subtype | Retention | Sync retention | Monitored reroute | Deficit | Class |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | --- |")
        for item in patterns[:5]:
            lines.append(
                f"| `{item['pattern']}` | {item['shell_subtype_label']} | "
                f"{_fmt_float(item['child_destination_retention_pooled'])} | "
                f"{_fmt_float(item['window_synchronous_child_destination_retention_pooled'])} | "
                f"{_fmt_float(item['candidate_reroute_share_pooled'])} | "
                f"{_fmt_num(item['anchor_child_deficit_mass_sum'])} | "
                f"{item['routing_class']} |"
            )
        lines.append("")

    return "\n".join(lines)


def _recover_lagaware_path_from_rows(atlas_rows: list[dict]) -> Path | None:
    paths = {
        str(row.get("dataset_path", ""))
        for row in atlas_rows
        if str(row.get("dataset_path", "")).strip()
    }
    for raw in sorted(paths):
        path = Path(raw)
        if path.is_file():
            return path
    return None


def _child_mass_map(preview: list[dict], event_total: int) -> dict[str, float]:
    if event_total <= 0:
        return {}
    return {
        str(item["pattern"]): float(item.get("weight", 0.0)) * float(event_total)
        for item in preview
        if float(item.get("weight", 0.0)) > 0.0
    }


def _mass_fraction(group: list[dict], *, flag: str) -> float | None:
    total = sum(float(row["anchor_local_event_total"]) for row in group)
    if total <= 0:
        return None
    matched = sum(
        float(row["anchor_local_event_total"])
        for row in group
        if bool(row.get(flag))
    )
    return matched / total


def _weighted_mean(group: list[dict], value_key: str, weight_key: str) -> float | None:
    total_weight = sum(float(row.get(weight_key, 0.0)) for row in group)
    if total_weight <= 0:
        return None
    return sum(
        float(row.get(value_key, 0.0) or 0.0) * float(row.get(weight_key, 0.0))
        for row in group
    ) / total_weight


def _top_patterns(group: list[dict], *, field: str, limit: int) -> list[dict]:
    ranked = sorted(
        [row for row in group if float(row.get(field, 0.0)) > 0.0],
        key=lambda row: (-float(row[field]), row["pattern"]),
    )
    keys = [
        "pattern",
        "shell_subtype",
        "shell_subtype_label",
        "child_destination_retention_pooled",
        "window_synchronous_child_destination_retention_pooled",
        "candidate_reroute_share_pooled",
        "anchor_child_deficit_mass_sum",
        "window_synchronous_anchor_child_deficit_mass_sum",
        "candidate_reroute_mass_sum",
        "routing_class",
    ]
    return [
        {key: row.get(key) for key in keys}
        for row in ranked[:limit]
    ]


def _count_classes(group: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in group:
        key = str(row["routing_class"])
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _raise_on_preview_truncation(rows: list[dict]) -> None:
    bad_rows = [row for row in rows if bool(row.get("preview_truncated"))]
    if not bad_rows:
        return
    sample = bad_rows[0]
    raise ValueError(
        "N2-08 requires complete monitored child mass maps, but at least one "
        "kernel preview is truncated. Re-run the upstream kernel with full child "
        "mass export, or pass --allow-preview-truncated for exploratory output "
        f"only. First truncated row: variant={sample.get('variant')} "
        f"source={sample.get('source_label')} pattern={sample.get('pattern')} "
        f"offset={sample.get('anchor_offset_bits')}"
    )


def _sum_child_mass_maps(rows: list[dict], key: str) -> dict[str, float]:
    total: dict[str, float] = {}
    for row in rows:
        for child, mass in dict(row.get(key, {})).items():
            value = float(mass)
            if value <= 0.0:
                continue
            total[str(child)] = total.get(str(child), 0.0) + value
    return _sorted_mass_map(total)


def _sorted_mass_map(mass_map: dict[str, float]) -> dict[str, float]:
    return {
        str(child): float(mass)
        for child, mass in sorted(mass_map.items())
        if float(mass) > 0.0
    }


def _ratio_or_none(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_csv(rows: list[dict], csv_path: Path, fieldnames: list[str]) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: _csv_value(row.get(field))
                    for field in fieldnames
                }
            )


def _csv_value(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def _source_kind_sort_key(source_kind: str, null_model: str | None) -> int:
    if source_kind == "observed":
        return 0
    if null_model == "markov1":
        return 1
    if null_model == "matched-lz":
        return 2
    if null_model == "phase-matched-lz":
        return 3
    return 9


def _fmt_float(value) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def _fmt_num(value) -> str:
    if value is None:
        return "-"
    return f"{float(value):.0f}"


def _clip(value, width: int) -> str:
    text = str(value)
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."
