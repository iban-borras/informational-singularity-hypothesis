from __future__ import annotations

import csv
from pathlib import Path


def summarize_parent_survival_band_rows(
    rows: list[dict],
    pattern_rows: list[dict],
) -> list[dict]:
    grouped_rows: dict[tuple[str, str, str | None, int | None], list[dict]] = {}
    for row in rows:
        key = (
            row["variant"],
            row["source_kind"],
            row.get("null_model"),
            row.get("null_seed"),
        )
        grouped_rows.setdefault(key, []).append(row)

    grouped_patterns: dict[
        tuple[str, str, str | None, int | None],
        dict[int, list[dict]],
    ] = {}
    for row in pattern_rows:
        key = (
            row["variant"],
            row["source_kind"],
            row.get("null_model"),
            row.get("null_seed"),
        )
        offset_map = grouped_patterns.setdefault(key, {})
        offset_map.setdefault(int(row["anchor_offset_bits"]), []).append(row)

    summaries = []
    for key, group in grouped_rows.items():
        first = group[0]
        group_sorted = sorted(group, key=lambda item: int(item["anchor_offset_bits"]))
        offsets = [int(row["anchor_offset_bits"]) for row in group_sorted]

        exact_counts = [int(row["exact_parent_count"]) for row in group_sorted]
        dead_counts = [int(row["dead_parent_count"]) for row in group_sorted]
        dark_counts = [int(row["dark_parent_count"]) for row in group_sorted]
        partial_counts = [int(row["partial_parent_count"]) for row in group_sorted]
        exact_masses = [float(row["exact_mass_fraction"] or 0.0) for row in group_sorted]
        dead_masses = [float(row["dead_mass_fraction"] or 0.0) for row in group_sorted]
        dark_masses = [float(row["dark_mass_fraction"] or 0.0) for row in group_sorted]
        signatures = [
            (
                int(row["exact_parent_count"]),
                int(row["dead_parent_count"]),
                int(row["dark_parent_count"]),
                int(row["partial_parent_count"]),
            )
            for row in group_sorted
        ]

        offset_patterns = grouped_patterns.get(key, {})
        exact_sets: list[set[str]] = []
        dead_sets: list[set[str]] = []
        dark_sets: list[set[str]] = []
        exact_arity_values: set[int] = set()
        dead_arity_values: set[int] = set()
        dark_arity_values: set[int] = set()

        for offset in offsets:
            items = offset_patterns.get(offset, [])
            exact_current = {
                str(item["pattern"])
                for item in items
                if item["support_state"] == "exact"
            }
            dead_current = {
                str(item["pattern"])
                for item in items
                if item["support_state"] == "dead"
            }
            dark_current = {
                str(item["pattern"])
                for item in items
                if item["support_state"] == "dark"
            }
            exact_sets.append(exact_current)
            dead_sets.append(dead_current)
            dark_sets.append(dark_current)

            for item in items:
                child_count = int(item.get("anchor_local_child_count", 0))
                state = str(item["support_state"])
                if state == "exact":
                    exact_arity_values.add(child_count)
                elif state == "dead":
                    dead_arity_values.add(child_count)
                elif state == "dark":
                    dark_arity_values.add(child_count)

        exact_intersection = _set_intersection(exact_sets)
        exact_union = _set_union(exact_sets)
        dead_intersection = _set_intersection(dead_sets)
        dead_union = _set_union(dead_sets)
        dark_intersection = _set_intersection(dark_sets)
        dark_union = _set_union(dark_sets)

        exact_identity_stable = bool(exact_sets) and exact_intersection == exact_union
        dead_identity_stable = bool(dead_sets) and dead_intersection == dead_union
        dark_identity_stable = bool(dark_sets) and dark_intersection == dark_union

        summaries.append(
            {
                "variant": first["variant"],
                "source_label": first["source_label"],
                "source_kind": first["source_kind"],
                "null_model": first.get("null_model"),
                "null_seed": first.get("null_seed"),
                "offset_count": len(group_sorted),
                "offsets": offsets,
                "active_parent_count_min": min(int(row["active_parent_count"]) for row in group_sorted),
                "active_parent_count_max": max(int(row["active_parent_count"]) for row in group_sorted),
                "exact_parent_count_min": min(exact_counts),
                "exact_parent_count_max": max(exact_counts),
                "dead_parent_count_min": min(dead_counts),
                "dead_parent_count_max": max(dead_counts),
                "dark_parent_count_min": min(dark_counts),
                "dark_parent_count_max": max(dark_counts),
                "partial_parent_count_min": min(partial_counts),
                "partial_parent_count_max": max(partial_counts),
                "exact_mass_fraction_min": min(exact_masses),
                "exact_mass_fraction_max": max(exact_masses),
                "dead_mass_fraction_min": min(dead_masses),
                "dead_mass_fraction_max": max(dead_masses),
                "dark_mass_fraction_min": min(dark_masses),
                "dark_mass_fraction_max": max(dark_masses),
                "frontier_signature_stable": len(set(signatures)) == 1,
                "exact_pattern_union_count": len(exact_union),
                "exact_pattern_intersection_count": len(exact_intersection),
                "exact_pattern_identity_stable": exact_identity_stable,
                "dead_pattern_union_count": len(dead_union),
                "dead_pattern_intersection_count": len(dead_intersection),
                "dead_pattern_identity_stable": dead_identity_stable,
                "dark_pattern_union_count": len(dark_union),
                "dark_pattern_intersection_count": len(dark_intersection),
                "dark_pattern_identity_stable": dark_identity_stable,
                "exact_anchor_child_arities": sorted(exact_arity_values),
                "dead_anchor_child_arities": sorted(dead_arity_values),
                "dark_anchor_child_arities": sorted(dark_arity_values),
                "exact_two_child_only": bool(exact_arity_values) and exact_arity_values == {2},
                "dead_one_child_only": bool(dead_arity_values) and dead_arity_values == {1},
                "dark_present": max(dark_counts) > 0,
                "zero_partial_band": max(partial_counts) == 0,
            }
        )

    return sorted(
        summaries,
        key=lambda item: (
            item["variant"],
            _source_kind_sort_key(item["source_kind"], item.get("null_model")),
            item.get("null_seed") if item.get("null_seed") is not None else -1,
        ),
    )


def render_parent_survival_band_report(
    selection: dict,
    grouped_summary: list[dict],
    rows: list[dict],
) -> str:
    lines = [
        "# Phase 2 Parent Survival Band Stability",
        "",
        "## Selection",
        "",
        f"- Anchor variant: {selection['anchor_variant']}",
        f"- Variants: {', '.join(selection['variants'])}",
        f"- Segment protocol: {selection['num_segments']} x {selection['segment_bits']}",
        f"- Offsets: {', '.join(str(value) for value in selection['offsets'])}",
        f"- Source freeze mode: {selection['source_freeze_mode']}",
        f"- Frozen total bits: {selection['frozen_total_bits']}",
        f"- Low/high scales: {selection['low_scale']} -> {selection['high_scale']}",
        f"- Pattern selection: {selection['pattern_selection']}",
        f"- Top patterns: {selection['top_patterns']}",
        f"- Window bits: {selection['window_bits']}",
        "",
        "## Band Summary",
        "",
        "| Variant | Source | Windows | Exact rng | Dead rng | Dark max | Part max | Exact m rng | Dead m rng | Ex arity | Dead arity | Ex IDs | Dead IDs | Sig |",
        "| --- | --- | ---: | --- | --- | ---: | ---: | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in grouped_summary:
        lines.append(
            f"| {item['variant']} | {item['source_label']} | {item['offset_count']} | "
            f"{_fmt_range(item['exact_parent_count_min'], item['exact_parent_count_max'])} | "
            f"{_fmt_range(item['dead_parent_count_min'], item['dead_parent_count_max'])} | "
            f"{item['dark_parent_count_max']} | "
            f"{item['partial_parent_count_max']} | "
            f"{_fmt_float_range(item['exact_mass_fraction_min'], item['exact_mass_fraction_max'])} | "
            f"{_fmt_float_range(item['dead_mass_fraction_min'], item['dead_mass_fraction_max'])} | "
            f"{_fmt_int_list(item['exact_anchor_child_arities'])} | "
            f"{_fmt_int_list(item['dead_anchor_child_arities'])} | "
            f"{_fmt_bool(item['exact_pattern_identity_stable'])} | "
            f"{_fmt_bool(item['dead_pattern_identity_stable'])} | "
            f"{_fmt_bool(item['frontier_signature_stable'])} |"
        )

    lines.extend(
        [
            "",
            "## Offset Readout",
            "",
            "| Variant | Source | Offset | Act_p | Exact_p | Dead_p | Dark_p | Part_p | Exact_m | Dead_m | Dark_m | WJ_cs |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(
        rows,
        key=lambda item: (
            item["variant"],
            _source_kind_sort_key(item["source_kind"], item.get("null_model")),
            item.get("null_seed") if item.get("null_seed") is not None else -1,
            int(item["anchor_offset_bits"]),
        ),
    ):
        lines.append(
            f"| {row['variant']} | {row['source_label']} | {row['anchor_offset_bits']} | "
            f"{row['active_parent_count']} | {row['exact_parent_count']} | {row['dead_parent_count']} | "
            f"{row['dark_parent_count']} | {row['partial_parent_count']} | "
            f"{_fmt_float(row['exact_mass_fraction'])} | {_fmt_float(row['dead_mass_fraction'])} | "
            f"{_fmt_float(row['dark_mass_fraction'])} | {_fmt_float(row['kernel_wj_mean'])} |"
        )

    lines.extend(
        [
            "",
            "## Stability Reading",
            "",
        ]
    )
    for item in grouped_summary:
        lines.append(f"### {item['variant']} | {item['source_label']}")
        lines.append("")
        lines.append(
            f"- Offsets: {', '.join(str(value) for value in item['offsets'])}"
        )
        lines.append(
            f"- Frontier signature stable: {_fmt_bool(item['frontier_signature_stable'])}"
        )
        lines.append(
            f"- Exact-pattern identity stable: {_fmt_bool(item['exact_pattern_identity_stable'])} "
            f"({item['exact_pattern_intersection_count']} / {item['exact_pattern_union_count']})"
        )
        lines.append(
            f"- Exact stable core across band: {item['exact_pattern_intersection_count']} patterns"
        )
        lines.append(
            f"- Dead-pattern identity stable: {_fmt_bool(item['dead_pattern_identity_stable'])} "
            f"({item['dead_pattern_intersection_count']} / {item['dead_pattern_union_count']})"
        )
        lines.append(
            f"- Dead emergent shell across band: {item['dead_pattern_union_count']} patterns"
        )
        lines.append(
            f"- Exact arity support: {_fmt_int_list(item['exact_anchor_child_arities'])}"
        )
        lines.append(
            f"- Dead arity support: {_fmt_int_list(item['dead_anchor_child_arities'])}"
        )
        lines.append(
            f"- Exact two-child only: {_fmt_bool(item['exact_two_child_only'])}"
        )
        lines.append(
            f"- Dead one-child only: {_fmt_bool(item['dead_one_child_only'])}"
        )
        lines.append("")

    return "\n".join(lines)


def write_parent_survival_band_summary_csv(
    rows: list[dict],
    csv_path: Path,
) -> None:
    fieldnames = [
        "variant",
        "source_label",
        "source_kind",
        "null_model",
        "null_seed",
        "offset_count",
        "active_parent_count_min",
        "active_parent_count_max",
        "exact_parent_count_min",
        "exact_parent_count_max",
        "dead_parent_count_min",
        "dead_parent_count_max",
        "dark_parent_count_min",
        "dark_parent_count_max",
        "partial_parent_count_min",
        "partial_parent_count_max",
        "exact_mass_fraction_min",
        "exact_mass_fraction_max",
        "dead_mass_fraction_min",
        "dead_mass_fraction_max",
        "dark_mass_fraction_min",
        "dark_mass_fraction_max",
        "frontier_signature_stable",
        "exact_pattern_union_count",
        "exact_pattern_intersection_count",
        "exact_pattern_identity_stable",
        "dead_pattern_union_count",
        "dead_pattern_intersection_count",
        "dead_pattern_identity_stable",
        "dark_pattern_union_count",
        "dark_pattern_intersection_count",
        "dark_pattern_identity_stable",
        "exact_anchor_child_arities",
        "dead_anchor_child_arities",
        "dark_anchor_child_arities",
        "exact_two_child_only",
        "dead_one_child_only",
        "dark_present",
        "zero_partial_band",
        "offsets",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        ",".join(str(value) for value in row[key])
                        if key in {
                            "exact_anchor_child_arities",
                            "dead_anchor_child_arities",
                            "dark_anchor_child_arities",
                            "offsets",
                        }
                        else row.get(key)
                    )
                    for key in fieldnames
                }
            )


def render_parent_survival_band_console_summary(rows: list[dict]) -> str:
    lines = [
        "Phase 2 parent survival band stability",
        "-" * 164,
        f"{'variant':<12}{'source':<28}{'wins':>6}{'exact':>10}{'dead':>10}{'dark':>8}{'part':>8}{'exact_m':>16}{'dead_m':>16}{'ex_arity':>12}{'dead_arity':>12}{'sig':>6}",
    ]
    for row in rows:
        lines.append(
            f"{_truncate(row['variant'], 12):<12}"
            f"{_truncate(row['source_label'], 28):<28}"
            f"{row['offset_count']:>6}"
            f"{_fmt_range(row['exact_parent_count_min'], row['exact_parent_count_max']):>10}"
            f"{_fmt_range(row['dead_parent_count_min'], row['dead_parent_count_max']):>10}"
            f"{row['dark_parent_count_max']:>8}"
            f"{row['partial_parent_count_max']:>8}"
            f"{_fmt_float_range(row['exact_mass_fraction_min'], row['exact_mass_fraction_max']):>16}"
            f"{_fmt_float_range(row['dead_mass_fraction_min'], row['dead_mass_fraction_max']):>16}"
            f"{_fmt_int_list(row['exact_anchor_child_arities']):>12}"
            f"{_fmt_int_list(row['dead_anchor_child_arities']):>12}"
            f"{_fmt_bool_short(row['frontier_signature_stable']):>6}"
        )
    return "\n".join(lines)


def _set_intersection(sets: list[set[str]]) -> set[str]:
    if not sets:
        return set()
    result = set(sets[0])
    for item in sets[1:]:
        result &= item
    return result


def _set_union(sets: list[set[str]]) -> set[str]:
    result: set[str] = set()
    for item in sets:
        result |= item
    return result


def _fmt_float(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def _fmt_range(minimum: int, maximum: int) -> str:
    return f"{minimum}-{maximum}"


def _fmt_float_range(minimum: float | None, maximum: float | None) -> str:
    if minimum is None or maximum is None:
        return "-"
    return f"{float(minimum):.4f}-{float(maximum):.4f}"


def _fmt_int_list(values: list[int]) -> str:
    if not values:
        return "-"
    return ",".join(str(value) for value in values)


def _fmt_bool(value: bool) -> str:
    return "yes" if bool(value) else "no"


def _fmt_bool_short(value: bool) -> str:
    return "Y" if bool(value) else "N"


def _truncate(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    if width <= 3:
        return value[:width]
    return value[: width - 3] + "..."


def _source_kind_sort_key(source_kind: str, null_model: str | None) -> tuple[int, str]:
    if source_kind == "observed":
        return (0, "")
    if null_model is None:
        return (1, "")
    return (2, str(null_model))
