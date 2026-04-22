from __future__ import annotations

import csv
from pathlib import Path

from .kernel_throughput import load_kernel_dataset


_STATE_ORDER = {
    "dead": 0,
    "dark": 1,
    "disjoint": 2,
    "mixed": 3,
    "loss": 4,
    "spill": 5,
    "exact": 6,
    "other": 7,
}


def build_support_dropout_rows(payload: dict) -> list[dict]:
    rows = []
    selection = payload["selection"]
    for source_row in payload["rows"]:
        kernel_result = source_row.get("kernel_result", {})
        per_pattern = kernel_result.get("per_pattern", [])
        active_patterns = [
            item
            for item in per_pattern
            if int(item.get("anchor_local_event_total", 0)) > 0
        ]

        active_parent_count = len(active_patterns)
        anchor_event_total = sum(
            int(item.get("anchor_local_event_total", 0))
            for item in active_patterns
        )

        state_counts = {name: 0 for name in _STATE_ORDER}
        state_masses = {name: 0.0 for name in _STATE_ORDER}
        child_retention_weighted_sum = 0.0
        child_precision_weighted_sum = 0.0
        frontier_patterns = []

        for item in active_patterns:
            anchor_child_count = int(item.get("anchor_local_child_count", 0))
            candidate_child_count = int(item.get("candidate_local_child_count", 0))
            common_child_count = int(item.get("common_child_count", 0))
            anchor_local_event_total = int(item.get("anchor_local_event_total", 0))
            candidate_local_event_total = int(item.get("candidate_local_event_total", 0))

            support_state = classify_support_state(item)
            state_counts[support_state] += 1
            state_masses[support_state] += float(anchor_local_event_total)

            child_retention_ratio = _ratio_or_none(
                common_child_count,
                anchor_child_count,
            )
            child_precision_ratio = _ratio_or_none(
                common_child_count,
                candidate_child_count,
            )
            event_retention_ratio = _ratio_or_none(
                candidate_local_event_total,
                anchor_local_event_total,
            )

            if child_retention_ratio is not None:
                child_retention_weighted_sum += (
                    float(anchor_local_event_total) * child_retention_ratio
                )
            if child_precision_ratio is not None:
                child_precision_weighted_sum += (
                    float(anchor_local_event_total) * child_precision_ratio
                )

            frontier_patterns.append(
                {
                    "pattern": str(item["pattern"]),
                    "support_state": support_state,
                    "weight": float(item.get("weight", 0.0)),
                    "anchor_local_parent_count": int(item.get("anchor_local_parent_count", 0)),
                    "candidate_local_parent_count": int(
                        item.get("candidate_local_parent_count", 0)
                    ),
                    "anchor_local_event_total": anchor_local_event_total,
                    "candidate_local_event_total": candidate_local_event_total,
                    "anchor_local_child_count": anchor_child_count,
                    "candidate_local_child_count": candidate_child_count,
                    "common_child_count": common_child_count,
                    "child_retention_ratio": child_retention_ratio,
                    "child_precision_ratio": child_precision_ratio,
                    "event_retention_ratio": event_retention_ratio,
                    "common_wj": item.get("common_wj"),
                    "top_child_full_match": item.get("top_child_full_match"),
                }
            )

        frontier_patterns.sort(
            key=lambda item: (
                _STATE_ORDER[item["support_state"]],
                _sort_ratio(item["child_retention_ratio"]),
                _sort_ratio(item["event_retention_ratio"]),
                -float(item["anchor_local_event_total"]),
                item["pattern"],
            )
        )

        rows.append(
            {
                "variant": source_row["variant"],
                "source_label": source_row["source_label"],
                "source_kind": source_row["source_kind"],
                "null_model": source_row.get("null_model"),
                "null_seed": source_row.get("null_seed"),
                "anchor_variant": source_row["anchor_variant"],
                "anchor_offset_bits": source_row["anchor_offset_bits"],
                "window_bits": source_row["window_bits"],
                "low_scale": source_row["low_scale"],
                "high_scale": source_row["high_scale"],
                "pattern_selection": source_row["pattern_selection"],
                "kernel_mode": source_row["kernel_mode"],
                "selected_parent_count": source_row["selected_parent_count"],
                "active_parent_count": active_parent_count,
                "anchor_event_total": int(anchor_event_total),
                "exact_parent_count": state_counts["exact"],
                "dead_parent_count": state_counts["dead"],
                "dark_parent_count": state_counts["dark"],
                "disjoint_parent_count": state_counts["disjoint"],
                "loss_parent_count": state_counts["loss"],
                "spill_parent_count": state_counts["spill"],
                "mixed_parent_count": state_counts["mixed"],
                "exact_parent_fraction": _ratio_or_none(
                    state_counts["exact"],
                    active_parent_count,
                ),
                "dead_parent_fraction": _ratio_or_none(
                    state_counts["dead"],
                    active_parent_count,
                ),
                "dark_parent_fraction": _ratio_or_none(
                    state_counts["dark"],
                    active_parent_count,
                ),
                "disjoint_parent_fraction": _ratio_or_none(
                    state_counts["disjoint"],
                    active_parent_count,
                ),
                "loss_parent_fraction": _ratio_or_none(
                    state_counts["loss"],
                    active_parent_count,
                ),
                "spill_parent_fraction": _ratio_or_none(
                    state_counts["spill"],
                    active_parent_count,
                ),
                "mixed_parent_fraction": _ratio_or_none(
                    state_counts["mixed"],
                    active_parent_count,
                ),
                "support_loss_parent_fraction": _ratio_or_none(
                    state_counts["dead"]
                    + state_counts["dark"]
                    + state_counts["disjoint"]
                    + state_counts["loss"]
                    + state_counts["mixed"],
                    active_parent_count,
                ),
                "exact_mass_fraction": _ratio_or_none(
                    state_masses["exact"],
                    anchor_event_total,
                ),
                "dead_mass_fraction": _ratio_or_none(
                    state_masses["dead"],
                    anchor_event_total,
                ),
                "dark_mass_fraction": _ratio_or_none(
                    state_masses["dark"],
                    anchor_event_total,
                ),
                "disjoint_mass_fraction": _ratio_or_none(
                    state_masses["disjoint"],
                    anchor_event_total,
                ),
                "loss_mass_fraction": _ratio_or_none(
                    state_masses["loss"],
                    anchor_event_total,
                ),
                "spill_mass_fraction": _ratio_or_none(
                    state_masses["spill"],
                    anchor_event_total,
                ),
                "mixed_mass_fraction": _ratio_or_none(
                    state_masses["mixed"],
                    anchor_event_total,
                ),
                "support_loss_mass_fraction": _ratio_or_none(
                    state_masses["dead"]
                    + state_masses["dark"]
                    + state_masses["disjoint"]
                    + state_masses["loss"]
                    + state_masses["mixed"],
                    anchor_event_total,
                ),
                "child_retention_weighted_mean": _ratio_or_none(
                    child_retention_weighted_sum,
                    anchor_event_total,
                ),
                "child_precision_weighted_mean": _ratio_or_none(
                    child_precision_weighted_sum,
                    anchor_event_total,
                ),
                "support_coverage_anchor_mean": source_row["support_coverage_anchor_mean"],
                "candidate_spillover_mean": source_row["candidate_spillover_mean"],
                "support_jaccard_mean": source_row["support_jaccard_mean"],
                "kernel_wj_mean": source_row["kernel_wj_mean"],
                "top_child_full_match_mass": source_row["top_child_full_match_mass"],
                "monitored_survivor_parent_fraction": _ratio_or_none(
                    state_counts["exact"]
                    + state_counts["disjoint"]
                    + state_counts["loss"]
                    + state_counts["spill"]
                    + state_counts["mixed"],
                    active_parent_count,
                ),
                "monitored_survivor_mass_fraction": _ratio_or_none(
                    state_masses["exact"]
                    + state_masses["disjoint"]
                    + state_masses["loss"]
                    + state_masses["spill"]
                    + state_masses["mixed"],
                    anchor_event_total,
                ),
                "selection_source_variant": selection.get("selection_source_variant"),
                "selection_source_run_dir": selection.get("selection_source_run_dir"),
                "dataset_path": payload["_dataset_path"],
                "frontier_patterns": frontier_patterns[:10],
            }
        )

    rows.sort(
        key=lambda row: (
            row["variant"],
            _source_kind_sort_key(row["source_kind"], row.get("null_model")),
            row.get("null_seed") if row.get("null_seed") is not None else -1,
        )
    )
    return rows


def summarize_support_dropout_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str | None, int | None], list[dict]] = {}
    for row in rows:
        key = (
            row["variant"],
            row["source_kind"],
            row.get("null_model"),
            row.get("null_seed"),
        )
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for group in grouped.values():
        first = group[0]
        summary_rows.append(
            {
                "variant": first["variant"],
                "source_label": first["source_label"],
                "source_kind": first["source_kind"],
                "null_model": first.get("null_model"),
                "null_seed": first.get("null_seed"),
                "row_count": len(group),
                "active_parent_count_avg": _mean_present(group, "active_parent_count"),
                "anchor_event_total_avg": _mean_present(group, "anchor_event_total"),
                "exact_parent_fraction_avg": _mean_present(group, "exact_parent_fraction"),
                "dead_parent_fraction_avg": _mean_present(
                    group,
                    "dead_parent_fraction",
                ),
                "dark_parent_fraction_avg": _mean_present(
                    group,
                    "dark_parent_fraction",
                ),
                "disjoint_parent_fraction_avg": _mean_present(
                    group,
                    "disjoint_parent_fraction",
                ),
                "loss_parent_fraction_avg": _mean_present(group, "loss_parent_fraction"),
                "spill_parent_fraction_avg": _mean_present(group, "spill_parent_fraction"),
                "mixed_parent_fraction_avg": _mean_present(group, "mixed_parent_fraction"),
                "support_loss_mass_fraction_avg": _mean_present(
                    group,
                    "support_loss_mass_fraction",
                ),
                "exact_mass_fraction_avg": _mean_present(group, "exact_mass_fraction"),
                "dead_mass_fraction_avg": _mean_present(group, "dead_mass_fraction"),
                "dark_mass_fraction_avg": _mean_present(group, "dark_mass_fraction"),
                "child_retention_weighted_mean_avg": _mean_present(
                    group,
                    "child_retention_weighted_mean",
                ),
                "child_precision_weighted_mean_avg": _mean_present(
                    group,
                    "child_precision_weighted_mean",
                ),
                "monitored_survivor_parent_fraction_avg": _mean_present(
                    group,
                    "monitored_survivor_parent_fraction",
                ),
                "monitored_survivor_mass_fraction_avg": _mean_present(
                    group,
                    "monitored_survivor_mass_fraction",
                ),
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


def render_support_dropout_report(
    selection: dict,
    summary_rows: list[dict],
    rows: list[dict],
) -> str:
    lines = [
        "# Phase 2 Kernel Support-Dropout Readout",
        "",
        "## Selection",
        "",
        f"- Anchor variant: {selection['anchor_variant']}",
        f"- Variants: {', '.join(selection['variants'])}",
        f"- Low/high scales: {selection['low_scale']} -> {selection['high_scale']}",
        f"- Pattern selection: {selection['pattern_selection']}",
        f"- Kernel mode: {selection['kernel_mode']}",
        f"- Top patterns: {selection['top_patterns']}",
        f"- Late anchor offset: {selection['anchor_offset_bits']}",
        f"- Window bits: {selection['window_bits']}",
        f"- Source dataset: `{selection['source_dataset_path']}`",
        "",
        "## Grouped Summary",
        "",
        "| Variant | Source | Rows | Act_p | Surv_p | Exact_p | Dead_p | Dark_p | Loss_p | Spill_p | Mixed_p | Surv_m | Dead_m | Dark_m |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summary_rows:
        lines.append(
            f"| {item['variant']} | {item['source_label']} | {item['row_count']} | "
            f"{_fmt(item['active_parent_count_avg'])} | {_fmt(item['monitored_survivor_parent_fraction_avg'])} | {_fmt(item['exact_parent_fraction_avg'])} | "
            f"{_fmt(item['dead_parent_fraction_avg'])} | {_fmt(item['dark_parent_fraction_avg'])} | {_fmt(item['loss_parent_fraction_avg'])} | "
            f"{_fmt(item['spill_parent_fraction_avg'])} | {_fmt(item['mixed_parent_fraction_avg'])} | "
            f"{_fmt(item['monitored_survivor_mass_fraction_avg'])} | {_fmt(item['dead_mass_fraction_avg'])} | "
            f"{_fmt(item['dark_mass_fraction_avg'])} |"
        )

    lines.extend(
        [
            "",
            "## Readout",
            "",
            "| Variant | Source | Act_p | Ev_a | Surv_p | Exact_m | Dead_m | Dark_m | Loss_m | Spill_m | Mixed_m | Ret_child | Prec_child | WJ_cs | Top_full |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['source_label']} | "
            f"{row['active_parent_count']} | {row['anchor_event_total']} | {_fmt(row['monitored_survivor_parent_fraction'])} | "
            f"{_fmt(row['exact_mass_fraction'])} | {_fmt(row['dead_mass_fraction'])} | {_fmt(row['dark_mass_fraction'])} | "
            f"{_fmt(row['loss_mass_fraction'])} | {_fmt(row['spill_mass_fraction'])} | {_fmt(row['mixed_mass_fraction'])} | "
            f"{_fmt(row['child_retention_weighted_mean'])} | {_fmt(row['child_precision_weighted_mean'])} | "
            f"{_fmt(row['kernel_wj_mean'])} | {_fmt(row['top_child_full_match_mass'])} |"
        )

    lines.extend(
        [
            "",
            "## Frontier Patterns",
            "",
        ]
    )
    for row in rows:
        heading = f"{row['variant']} | {row['source_label']}"
        if row.get("null_model"):
            heading += f" | {row['null_model']}"
        if row.get("null_seed") is not None:
            heading += f" | seed {row['null_seed']}"
        lines.append(f"### {heading}")
        lines.append("")
        if not row["frontier_patterns"]:
            lines.append("No active anchor patterns.")
            lines.append("")
            continue
        lines.append(
            "| Pattern | State | Pa_a | Pa_x | Ev_a | Ev_x | Ch_a | Ch_x | Ch_cap | Ret_child | Ret_event | Top_full |"
        )
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for item in row["frontier_patterns"][:5]:
            lines.append(
                f"| `{item['pattern']}` | {item['support_state']} | "
                f"{item['anchor_local_parent_count']} | {item['candidate_local_parent_count']} | "
                f"{item['anchor_local_event_total']} | {item['candidate_local_event_total']} | "
                f"{item['anchor_local_child_count']} | {item['candidate_local_child_count']} | {item['common_child_count']} | "
                f"{_fmt(item['child_retention_ratio'])} | {_fmt(item['event_retention_ratio'])} | "
                f"{str(bool(item['top_child_full_match']))} |"
            )
        lines.append("")

    return "\n".join(lines)


def write_support_dropout_csv(rows: list[dict], csv_path: Path) -> None:
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
        "selected_parent_count",
        "active_parent_count",
        "anchor_event_total",
        "exact_parent_count",
        "dead_parent_count",
        "dark_parent_count",
        "disjoint_parent_count",
        "loss_parent_count",
        "spill_parent_count",
        "mixed_parent_count",
        "monitored_survivor_parent_fraction",
        "exact_parent_fraction",
        "dead_parent_fraction",
        "dark_parent_fraction",
        "disjoint_parent_fraction",
        "loss_parent_fraction",
        "spill_parent_fraction",
        "mixed_parent_fraction",
        "support_loss_parent_fraction",
        "monitored_survivor_mass_fraction",
        "exact_mass_fraction",
        "dead_mass_fraction",
        "dark_mass_fraction",
        "disjoint_mass_fraction",
        "loss_mass_fraction",
        "spill_mass_fraction",
        "mixed_mass_fraction",
        "support_loss_mass_fraction",
        "child_retention_weighted_mean",
        "child_precision_weighted_mean",
        "support_coverage_anchor_mean",
        "candidate_spillover_mean",
        "support_jaccard_mean",
        "kernel_wj_mean",
        "top_child_full_match_mass",
        "dataset_path",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def classify_support_state(item: dict) -> str:
    candidate_local_parent_count = int(item.get("candidate_local_parent_count", 0))
    anchor_child_count = int(item.get("anchor_local_child_count", 0))
    candidate_child_count = int(item.get("candidate_local_child_count", 0))
    common_child_count = int(item.get("common_child_count", 0))
    candidate_local_event_total = int(item.get("candidate_local_event_total", 0))

    if candidate_local_parent_count <= 0:
        return "dead"
    if candidate_local_event_total <= 0 or candidate_child_count <= 0:
        return "dark"
    if (
        common_child_count == anchor_child_count
        and common_child_count == candidate_child_count
    ):
        return "exact"
    if common_child_count <= 0:
        return "disjoint"

    anchor_loss = common_child_count < anchor_child_count
    spill = common_child_count < candidate_child_count
    if anchor_loss and spill:
        return "mixed"
    if anchor_loss:
        return "loss"
    if spill:
        return "spill"
    return "other"


def _mean_present(rows: list[dict], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    if not values:
        return None
    return float(sum(float(value) for value in values) / len(values))


def _ratio_or_none(numerator: float | int, denominator: float | int) -> float | None:
    denominator = float(denominator)
    if denominator <= 0.0:
        return None
    return float(numerator) / denominator


def _sort_ratio(value: float | None) -> float:
    if value is None:
        return -1.0
    return float(value)


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def _source_kind_sort_key(source_kind: str, null_model: str | None) -> tuple[int, str]:
    if source_kind == "observed":
        return (0, "")
    if null_model is None:
        return (1, "")
    return (2, str(null_model))
