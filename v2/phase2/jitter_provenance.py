from __future__ import annotations

import csv
import json
from pathlib import Path


def build_jitter_provenance_readout(atlas_payload: dict) -> dict:
    """Derive a same-parent temporal provenance readout from shell-atlas rows."""
    atlas_rows = atlas_payload.get("atlas_rows", [])
    if not atlas_rows:
        raise ValueError("Atlas payload does not contain atlas_rows.")

    window_rows = build_window_delta_rows(atlas_rows)
    pattern_rows = summarize_pattern_provenance(window_rows)
    source_rows = summarize_source_provenance(pattern_rows)

    return {
        "selection": dict(atlas_payload.get("selection", {})),
        "notes": [
            "N2-07 is a same-parent temporal provenance readout derived from a parent-shell atlas artifact.",
            "It does not yet infer cross-parent or child-destination routing.",
            "A local loss is counted when candidate events are below anchor events in a window.",
            "A local surplus is counted when candidate events exceed anchor events in a window.",
            "Compensated loss is the portion of gross loss balanced by surplus across the same parent pattern over the band.",
            "The provenance shift is the surplus centroid minus the loss centroid in bits.",
        ],
        "source_summary": source_rows,
        "pattern_summary": pattern_rows,
        "window_deltas": window_rows,
    }


def build_window_delta_rows(atlas_rows: list[dict]) -> list[dict]:
    rows = []
    for row in atlas_rows:
        anchor_events = int(row.get("anchor_local_event_total", 0))
        candidate_events = int(row.get("candidate_local_event_total", 0))
        delta = candidate_events - anchor_events
        loss = max(anchor_events - candidate_events, 0)
        surplus = max(candidate_events - anchor_events, 0)
        rows.append(
            {
                "variant": str(row["variant"]),
                "source_label": str(row["source_label"]),
                "source_kind": str(row["source_kind"]),
                "null_model": row.get("null_model"),
                "null_seed": row.get("null_seed"),
                "anchor_variant": str(row.get("anchor_variant", "")),
                "anchor_offset_bits": int(row["anchor_offset_bits"]),
                "candidate_lag_bits": int(row.get("candidate_lag_bits", 0)),
                "candidate_offset_bits": int(row["anchor_offset_bits"])
                + int(row.get("candidate_lag_bits", 0)),
                "window_bits": int(row.get("window_bits", 0)),
                "low_scale": int(row.get("low_scale", 0)),
                "high_scale": int(row.get("high_scale", 0)),
                "pattern_selection": str(row.get("pattern_selection", "")),
                "kernel_mode": str(row.get("kernel_mode", "")),
                "pattern": str(row["pattern"]),
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
                "support_state": str(row.get("support_state", "other")),
                "anchor_local_event_total": anchor_events,
                "candidate_local_event_total": candidate_events,
                "delta_event_total": delta,
                "loss_mass": loss,
                "surplus_mass": surplus,
                "loss_flag": loss > 0,
                "surplus_flag": surplus > 0,
                "exact_balance_flag": loss == 0 and surplus == 0,
                "common_wj": row.get("common_wj"),
                "top_child_full_match": row.get("top_child_full_match"),
                "event_retention_ratio": row.get("event_retention_ratio"),
            }
        )

    rows.sort(
        key=lambda item: (
            item["variant"],
            _source_kind_sort_key(item["source_kind"], item.get("null_model")),
            item.get("null_seed") if item.get("null_seed") is not None else -1,
            item["shell_subtype"],
            item["pattern"],
            item["anchor_offset_bits"],
        )
    )
    return rows


def summarize_pattern_provenance(window_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in window_rows:
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
        anchor_sum = sum(int(row["anchor_local_event_total"]) for row in group)
        candidate_sum = sum(int(row["candidate_local_event_total"]) for row in group)
        loss_sum = sum(int(row["loss_mass"]) for row in group)
        surplus_sum = sum(int(row["surplus_mass"]) for row in group)
        compensated_loss = min(loss_sum, surplus_sum)
        net_deficit = max(loss_sum - surplus_sum, 0)
        net_surplus = max(surplus_sum - loss_sum, 0)
        loss_centroid = _weighted_centroid(group, "loss_mass")
        surplus_centroid = _weighted_centroid(group, "surplus_mass")
        shift = None
        if loss_centroid is not None and surplus_centroid is not None:
            shift = surplus_centroid - loss_centroid

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
                "prefix_structural_child_count": first[
                    "prefix_structural_child_count"
                ],
                "suffix_structural_child_count": first[
                    "suffix_structural_child_count"
                ],
                "bridge_structural_child_count": first[
                    "bridge_structural_child_count"
                ],
                "window_count": len(group),
                "loss_window_count": sum(1 for row in group if row["loss_flag"]),
                "surplus_window_count": sum(1 for row in group if row["surplus_flag"]),
                "exact_balance_window_count": sum(
                    1 for row in group if row["exact_balance_flag"]
                ),
                "anchor_event_total_sum": anchor_sum,
                "candidate_event_total_sum": candidate_sum,
                "partition_mass_retention_pooled": _ratio_or_none(
                    candidate_sum,
                    anchor_sum,
                ),
                "gross_loss_mass_sum": loss_sum,
                "gross_surplus_mass_sum": surplus_sum,
                "compensated_loss_mass": compensated_loss,
                "net_deficit_mass": net_deficit,
                "net_surplus_mass": net_surplus,
                "compensated_loss_share_of_gross": _ratio_or_none(
                    compensated_loss,
                    loss_sum,
                ),
                "net_deficit_share_of_gross": _ratio_or_none(net_deficit, loss_sum),
                "loss_centroid_bits": loss_centroid,
                "surplus_centroid_bits": surplus_centroid,
                "provenance_shift_bits": shift,
                "provenance_class": classify_pattern_provenance(
                    loss_sum=loss_sum,
                    surplus_sum=surplus_sum,
                    compensated_loss=compensated_loss,
                    net_deficit=net_deficit,
                    net_surplus=net_surplus,
                ),
            }
        )

    summaries.sort(
        key=lambda item: (
            item["variant"],
            _source_kind_sort_key(item["source_kind"], item.get("null_model")),
            item.get("null_seed") if item.get("null_seed") is not None else -1,
            -int(item["compensated_loss_mass"]),
            -int(item["net_deficit_mass"]),
            item["pattern"],
        )
    )
    return summaries


def summarize_source_provenance(pattern_rows: list[dict]) -> list[dict]:
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
        anchor_sum = sum(int(row["anchor_event_total_sum"]) for row in group)
        candidate_sum = sum(int(row["candidate_event_total_sum"]) for row in group)
        loss_sum = sum(int(row["gross_loss_mass_sum"]) for row in group)
        surplus_sum = sum(int(row["gross_surplus_mass_sum"]) for row in group)
        compensated_loss = sum(int(row["compensated_loss_mass"]) for row in group)
        net_deficit = sum(int(row["net_deficit_mass"]) for row in group)
        net_surplus = sum(int(row["net_surplus_mass"]) for row in group)
        loss_centroid = _weighted_source_centroid(group, "loss_centroid_bits", "gross_loss_mass_sum")
        surplus_centroid = _weighted_source_centroid(
            group,
            "surplus_centroid_bits",
            "gross_surplus_mass_sum",
        )
        shift = None
        if loss_centroid is not None and surplus_centroid is not None:
            shift = surplus_centroid - loss_centroid
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
                "anchor_event_total_sum": anchor_sum,
                "candidate_event_total_sum": candidate_sum,
                "partition_mass_retention_pooled": _ratio_or_none(
                    candidate_sum,
                    anchor_sum,
                ),
                "gross_loss_mass_sum": loss_sum,
                "gross_surplus_mass_sum": surplus_sum,
                "compensated_loss_mass": compensated_loss,
                "net_deficit_mass": net_deficit,
                "net_surplus_mass": net_surplus,
                "compensated_loss_share_of_gross": _ratio_or_none(
                    compensated_loss,
                    loss_sum,
                ),
                "net_deficit_share_of_gross": _ratio_or_none(net_deficit, loss_sum),
                "net_surplus_share_of_surplus": _ratio_or_none(net_surplus, surplus_sum),
                "loss_centroid_bits": loss_centroid,
                "surplus_centroid_bits": surplus_centroid,
                "provenance_shift_bits": shift,
                "dominant_provenance_class": _dominant_class(class_counts),
                "provenance_class_counts": class_counts,
                "top_compensated_patterns": _top_patterns(
                    group,
                    field="compensated_loss_mass",
                    limit=10,
                ),
                "top_net_deficit_patterns": _top_patterns(
                    group,
                    field="net_deficit_mass",
                    limit=10,
                ),
            }
        )

    summaries.sort(
        key=lambda item: (
            item["variant"],
            _source_kind_sort_key(item["source_kind"], item.get("null_model")),
            item.get("null_seed") if item.get("null_seed") is not None else -1,
        )
    )
    return summaries


def classify_pattern_provenance(
    *,
    loss_sum: int,
    surplus_sum: int,
    compensated_loss: int,
    net_deficit: int,
    net_surplus: int,
) -> str:
    if loss_sum <= 0 and surplus_sum <= 0:
        return "flat"
    if loss_sum <= 0:
        return "surplus-only"
    if surplus_sum <= 0:
        return "uncompensated-deficit"
    compensated_share = compensated_loss / loss_sum
    net_deficit_share = net_deficit / loss_sum
    if compensated_share >= 0.75 and net_deficit_share <= 0.25:
        return "jitter-compensated"
    if net_deficit_share >= 0.75:
        return "net-deficit"
    if net_surplus > net_deficit:
        return "surplus-dominant-mixed"
    return "mixed"


def write_source_provenance_csv(rows: list[dict], csv_path: Path) -> None:
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
        "partition_mass_retention_pooled",
        "gross_loss_mass_sum",
        "gross_surplus_mass_sum",
        "compensated_loss_mass",
        "net_deficit_mass",
        "net_surplus_mass",
        "compensated_loss_share_of_gross",
        "net_deficit_share_of_gross",
        "net_surplus_share_of_surplus",
        "loss_centroid_bits",
        "surplus_centroid_bits",
        "provenance_shift_bits",
        "dominant_provenance_class",
        "provenance_class_counts",
    ]
    _write_csv(rows, csv_path, fieldnames)


def write_pattern_provenance_csv(rows: list[dict], csv_path: Path) -> None:
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
        "prefix_structural_child_count",
        "suffix_structural_child_count",
        "bridge_structural_child_count",
        "window_count",
        "loss_window_count",
        "surplus_window_count",
        "exact_balance_window_count",
        "anchor_event_total_sum",
        "candidate_event_total_sum",
        "partition_mass_retention_pooled",
        "gross_loss_mass_sum",
        "gross_surplus_mass_sum",
        "compensated_loss_mass",
        "net_deficit_mass",
        "net_surplus_mass",
        "compensated_loss_share_of_gross",
        "net_deficit_share_of_gross",
        "loss_centroid_bits",
        "surplus_centroid_bits",
        "provenance_shift_bits",
        "provenance_class",
    ]
    _write_csv(rows, csv_path, fieldnames)


def write_window_deltas_csv(rows: list[dict], csv_path: Path) -> None:
    fieldnames = [
        "variant",
        "source_label",
        "source_kind",
        "null_model",
        "null_seed",
        "anchor_variant",
        "anchor_offset_bits",
        "candidate_lag_bits",
        "candidate_offset_bits",
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
        "delta_event_total",
        "loss_mass",
        "surplus_mass",
        "loss_flag",
        "surplus_flag",
        "exact_balance_flag",
        "common_wj",
        "top_child_full_match",
        "event_retention_ratio",
    ]
    _write_csv(rows, csv_path, fieldnames)


def render_jitter_provenance_console_summary(source_rows: list[dict]) -> str:
    lines = [
        "Phase 2 jitter provenance readout",
        "-" * 132,
        (
            f"{'variant':<12} {'source':<30} {'patterns':>8} {'loss':>8} "
            f"{'surp':>8} {'comp':>8} {'netDef':>8} {'jitShare':>9} "
            f"{'shift':>12} {'class':>24}"
        ),
    ]
    for row in source_rows:
        lines.append(
            f"{_clip(row['variant'], 12):<12} "
            f"{_clip(row['source_label'], 30):<30} "
            f"{int(row['pattern_count']):>8} "
            f"{int(row['gross_loss_mass_sum']):>8} "
            f"{int(row['gross_surplus_mass_sum']):>8} "
            f"{int(row['compensated_loss_mass']):>8} "
            f"{int(row['net_deficit_mass']):>8} "
            f"{_fmt_float(row['compensated_loss_share_of_gross']):>9} "
            f"{_fmt_shift(row['provenance_shift_bits']):>12} "
            f"{_clip(row['dominant_provenance_class'] or '-', 24):>24}"
        )
    return "\n".join(lines)


def render_jitter_provenance_report(readout: dict) -> str:
    selection = readout.get("selection", {})
    source_rows = readout["source_summary"]
    lines = [
        "# Phase 2 Jitter Provenance Readout",
        "",
        "## Selection",
        "",
        f"- Anchor variant: {selection.get('anchor_variant', '-')}",
        f"- Candidate variant: {selection.get('candidate_variant', '-')}",
        f"- Low/high scales: {selection.get('low_scale', '-')} -> {selection.get('high_scale', '-')}",
        f"- Pattern selection: {selection.get('pattern_selection', '-')}",
        f"- Top patterns: {selection.get('top_patterns', '-')}",
        f"- Candidate lag bits: {selection.get('candidate_lag_bits', '-')}",
        "",
        "## Method Boundary",
        "",
        "This readout tracks same-parent temporal compensation inside an existing shell-atlas artifact.",
        "It can say whether local shell losses are compensated by surplus appearances elsewhere in the analyzed band.",
        "It does not yet prove cross-parent routing or child-destination flow.",
        "",
        "## Source Summary",
        "",
        "| Variant | Source | Patterns | Loss | Surplus | Compensated | Net deficit | Comp/Loss | Shift | Dominant class |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in source_rows:
        lines.append(
            f"| {row['variant']} | {row['source_label']} | {row['pattern_count']} | "
            f"{row['gross_loss_mass_sum']} | {row['gross_surplus_mass_sum']} | "
            f"{row['compensated_loss_mass']} | {row['net_deficit_mass']} | "
            f"{_fmt_float(row['compensated_loss_share_of_gross'])} | "
            f"{_fmt_shift(row['provenance_shift_bits'])} | "
            f"{row['dominant_provenance_class'] or '-'} |"
        )

    lines.extend(
        [
            "",
            "## Top Compensated Patterns",
            "",
        ]
    )
    for row in source_rows:
        lines.append(f"### {row['variant']} | {row['source_label']}")
        lines.append("")
        top_patterns = row.get("top_compensated_patterns", [])
        if not top_patterns:
            lines.append("No compensated-loss patterns.")
            lines.append("")
            continue
        lines.append("| Pattern | Subtype | Comp | NetDef | Loss | Surplus | Shift | Class |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
        for item in top_patterns[:5]:
            lines.append(
                f"| `{item['pattern']}` | {item['shell_subtype_label']} | "
                f"{item['compensated_loss_mass']} | {item['net_deficit_mass']} | "
                f"{item['gross_loss_mass_sum']} | {item['gross_surplus_mass_sum']} | "
                f"{_fmt_shift(item['provenance_shift_bits'])} | {item['provenance_class']} |"
            )
        lines.append("")

    return "\n".join(lines)


def source_summary_for_json(rows: list[dict]) -> list[dict]:
    clean_rows = []
    for row in rows:
        item = dict(row)
        item["provenance_class_counts"] = dict(item.get("provenance_class_counts", {}))
        clean_rows.append(item)
    return clean_rows


def _weighted_centroid(group: list[dict], mass_field: str) -> float | None:
    total = sum(float(row[mass_field]) for row in group)
    if total <= 0.0:
        return None
    return sum(
        float(row[mass_field]) * float(row["anchor_offset_bits"])
        for row in group
    ) / total


def _weighted_source_centroid(
    group: list[dict],
    centroid_field: str,
    mass_field: str,
) -> float | None:
    weighted = [
        (float(row[centroid_field]), float(row[mass_field]))
        for row in group
        if row.get(centroid_field) is not None and float(row.get(mass_field, 0)) > 0.0
    ]
    total = sum(mass for _, mass in weighted)
    if total <= 0.0:
        return None
    return sum(value * mass for value, mass in weighted) / total


def _top_patterns(group: list[dict], *, field: str, limit: int) -> list[dict]:
    ranked = sorted(
        [row for row in group if int(row.get(field, 0)) > 0],
        key=lambda row: (-int(row[field]), row["pattern"]),
    )
    return [
        {
            "pattern": row["pattern"],
            "shell_subtype": row["shell_subtype"],
            "shell_subtype_label": row["shell_subtype_label"],
            "gross_loss_mass_sum": row["gross_loss_mass_sum"],
            "gross_surplus_mass_sum": row["gross_surplus_mass_sum"],
            "compensated_loss_mass": row["compensated_loss_mass"],
            "net_deficit_mass": row["net_deficit_mass"],
            "net_surplus_mass": row["net_surplus_mass"],
            "provenance_shift_bits": row["provenance_shift_bits"],
            "provenance_class": row["provenance_class"],
        }
        for row in ranked[:limit]
    ]


def _count_classes(group: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in group:
        key = str(row["provenance_class"])
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _dominant_class(counts: dict[str, int]) -> str | None:
    if not counts:
        return None
    return max(counts, key=lambda key: (counts[key], key))


def _ratio_or_none(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def _source_kind_sort_key(source_kind: str, null_model: str | None) -> int:
    if source_kind == "observed":
        return 0
    if null_model == "markov1":
        return 1
    if null_model == "matched-lz":
        return 2
    return 9


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


def _fmt_float(value) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def _fmt_shift(value) -> str:
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
