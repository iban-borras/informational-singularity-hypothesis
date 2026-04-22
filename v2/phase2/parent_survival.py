from __future__ import annotations

import csv
from pathlib import Path

from .kernel_support_dropout import classify_support_state
from .kernel_throughput import load_kernel_dataset


def build_parent_survival_rows(payload: dict) -> tuple[list[dict], list[dict]]:
    rows = []
    pattern_rows = []
    selection = payload["selection"]
    dataset_path = payload["_dataset_path"]

    for source_row in payload["rows"]:
        kernel_result = source_row.get("kernel_result", {})
        per_pattern = kernel_result.get("per_pattern", [])
        active_patterns = [
            item
            for item in per_pattern
            if int(item.get("anchor_local_event_total", 0)) > 0
        ]

        exact_patterns: list[dict] = []
        dead_patterns: list[dict] = []
        dark_patterns: list[dict] = []
        partial_patterns: list[dict] = []

        active_parent_count = len(active_patterns)
        anchor_event_total = sum(
            int(item.get("anchor_local_event_total", 0))
            for item in active_patterns
        )
        survivor_count = 0
        exact_count = 0
        dead_count = 0
        dark_count = 0
        partial_count = 0
        exact_mass = 0.0
        dead_mass = 0.0
        dark_mass = 0.0
        partial_mass = 0.0

        for item in active_patterns:
            state = classify_support_state(item)
            candidate_event_total = int(item.get("candidate_local_event_total", 0))
            anchor_event = int(item.get("anchor_local_event_total", 0))
            record = {
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
                "pattern": str(item["pattern"]),
                "support_state": state,
                "anchor_local_parent_count": int(item.get("anchor_local_parent_count", 0)),
                "candidate_local_parent_count": int(
                    item.get("candidate_local_parent_count", 0)
                ),
                "anchor_local_event_total": anchor_event,
                "candidate_local_event_total": candidate_event_total,
                "anchor_local_child_count": int(item.get("anchor_local_child_count", 0)),
                "candidate_local_child_count": int(
                    item.get("candidate_local_child_count", 0)
                ),
                "common_child_count": int(item.get("common_child_count", 0)),
                "event_retention_ratio": _ratio_or_none(
                    candidate_event_total,
                    anchor_event,
                ),
                "common_wj": item.get("common_wj"),
                "top_child_full_match": item.get("top_child_full_match"),
                "dataset_path": dataset_path,
            }
            pattern_rows.append(record)

            if state in {"exact", "disjoint", "loss", "spill", "mixed"}:
                survivor_count += 1
            if state == "exact":
                exact_count += 1
                exact_mass += anchor_event
                exact_patterns.append(record)
            elif state == "dead":
                dead_count += 1
                dead_mass += anchor_event
                dead_patterns.append(record)
            elif state == "dark":
                dark_count += 1
                dark_mass += anchor_event
                dark_patterns.append(record)
            else:
                partial_count += 1
                partial_mass += anchor_event
                partial_patterns.append(record)

        exact_patterns.sort(
            key=lambda item: (
                -item["anchor_local_event_total"],
                item["pattern"],
            )
        )
        dead_patterns.sort(
            key=lambda item: (
                -item["anchor_local_event_total"],
                item["pattern"],
            )
        )
        dark_patterns.sort(
            key=lambda item: (
                -item["anchor_local_event_total"],
                item["pattern"],
            )
        )
        partial_patterns.sort(
            key=lambda item: (
                item["support_state"],
                -item["anchor_local_event_total"],
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
                "survivor_parent_count": survivor_count,
                "exact_parent_count": exact_count,
                "dead_parent_count": dead_count,
                "dark_parent_count": dark_count,
                "partial_parent_count": partial_count,
                "survivor_parent_fraction": _ratio_or_none(
                    survivor_count,
                    active_parent_count,
                ),
                "exact_parent_fraction": _ratio_or_none(
                    exact_count,
                    active_parent_count,
                ),
                "dead_parent_fraction": _ratio_or_none(
                    dead_count,
                    active_parent_count,
                ),
                "dark_parent_fraction": _ratio_or_none(
                    dark_count,
                    active_parent_count,
                ),
                "partial_parent_fraction": _ratio_or_none(
                    partial_count,
                    active_parent_count,
                ),
                "survivor_mass_fraction": _ratio_or_none(
                    exact_mass + partial_mass,
                    anchor_event_total,
                ),
                "exact_mass_fraction": _ratio_or_none(
                    exact_mass,
                    anchor_event_total,
                ),
                "dead_mass_fraction": _ratio_or_none(
                    dead_mass,
                    anchor_event_total,
                ),
                "dark_mass_fraction": _ratio_or_none(
                    dark_mass,
                    anchor_event_total,
                ),
                "partial_mass_fraction": _ratio_or_none(
                    partial_mass,
                    anchor_event_total,
                ),
                "support_coverage_anchor_mean": source_row["support_coverage_anchor_mean"],
                "kernel_wj_mean": source_row["kernel_wj_mean"],
                "top_child_full_match_mass": source_row["top_child_full_match_mass"],
                "selection_source_variant": selection.get("selection_source_variant"),
                "selection_source_run_dir": selection.get("selection_source_run_dir"),
                "dataset_path": dataset_path,
                "exact_patterns": exact_patterns[:10],
                "dead_patterns": dead_patterns[:10],
                "dark_patterns": dark_patterns[:10],
                "partial_patterns": partial_patterns[:10],
            }
        )

    rows.sort(
        key=lambda row: (
            row["variant"],
            _source_kind_sort_key(row["source_kind"], row.get("null_model")),
            row.get("null_seed") if row.get("null_seed") is not None else -1,
        )
    )
    pattern_rows.sort(
        key=lambda row: (
            row["variant"],
            _source_kind_sort_key(row["source_kind"], row.get("null_model")),
            row.get("null_seed") if row.get("null_seed") is not None else -1,
            _state_sort_key(row["support_state"]),
            -row["anchor_local_event_total"],
            row["pattern"],
        )
    )
    return rows, pattern_rows


def summarize_parent_survival_rows(rows: list[dict]) -> list[dict]:
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
                "survivor_parent_fraction_avg": _mean_present(
                    group,
                    "survivor_parent_fraction",
                ),
                "exact_parent_fraction_avg": _mean_present(group, "exact_parent_fraction"),
                "dead_parent_fraction_avg": _mean_present(
                    group,
                    "dead_parent_fraction",
                ),
                "dark_parent_fraction_avg": _mean_present(
                    group,
                    "dark_parent_fraction",
                ),
                "partial_parent_fraction_avg": _mean_present(
                    group,
                    "partial_parent_fraction",
                ),
                "survivor_mass_fraction_avg": _mean_present(group, "survivor_mass_fraction"),
                "exact_mass_fraction_avg": _mean_present(group, "exact_mass_fraction"),
                "dead_mass_fraction_avg": _mean_present(
                    group,
                    "dead_mass_fraction",
                ),
                "dark_mass_fraction_avg": _mean_present(
                    group,
                    "dark_mass_fraction",
                ),
                "partial_mass_fraction_avg": _mean_present(
                    group,
                    "partial_mass_fraction",
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


def render_parent_survival_report(
    selection: dict,
    summary_rows: list[dict],
    rows: list[dict],
) -> str:
    lines = [
        "# Phase 2 Parent Survival Selector",
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
        "| Variant | Source | Rows | Act_p | Surv_p | Exact_p | Dead_p | Dark_p | Part_p | Surv_m | Exact_m | Dead_m | Dark_m | Part_m |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summary_rows:
        lines.append(
            f"| {item['variant']} | {item['source_label']} | {item['row_count']} | "
            f"{_fmt(item['active_parent_count_avg'])} | {_fmt(item['survivor_parent_fraction_avg'])} | "
            f"{_fmt(item['exact_parent_fraction_avg'])} | {_fmt(item['dead_parent_fraction_avg'])} | {_fmt(item['dark_parent_fraction_avg'])} | {_fmt(item['partial_parent_fraction_avg'])} | "
            f"{_fmt(item['survivor_mass_fraction_avg'])} | {_fmt(item['exact_mass_fraction_avg'])} | {_fmt(item['dead_mass_fraction_avg'])} | {_fmt(item['dark_mass_fraction_avg'])} | {_fmt(item['partial_mass_fraction_avg'])} |"
        )

    lines.extend(
        [
            "",
            "## Readout",
            "",
            "| Variant | Source | Act_p | Surv_p | Exact_p | Dead_p | Dark_p | Part_p | Exact_m | Dead_m | Dark_m | Part_m | Cov_a | WJ_cs |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['source_label']} | {row['active_parent_count']} | "
            f"{row['survivor_parent_count']} | {row['exact_parent_count']} | {row['dead_parent_count']} | {row['dark_parent_count']} | {row['partial_parent_count']} | "
            f"{_fmt(row['exact_mass_fraction'])} | {_fmt(row['dead_mass_fraction'])} | {_fmt(row['dark_mass_fraction'])} | {_fmt(row['partial_mass_fraction'])} | "
            f"{_fmt(row['support_coverage_anchor_mean'])} | {_fmt(row['kernel_wj_mean'])} |"
        )

    for row in rows:
        heading = f"{row['variant']} | {row['source_label']}"
        if row.get("null_model"):
            heading += f" | {row['null_model']}"
        if row.get("null_seed") is not None:
            heading += f" | seed {row['null_seed']}"
        lines.extend(
            [
                "",
                f"## {heading}",
                "",
                "### Exact Survivors",
                "",
            ]
        )
        if not row["exact_patterns"]:
            lines.append("No exact survivors.")
        else:
            lines.append("| Pattern | Ev_a | Ev_x | Ret_x | Ch_a | Ch_x | Ch_cap |")
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
            for item in row["exact_patterns"][:5]:
                lines.append(
                    f"| `{item['pattern']}` | {item['anchor_local_event_total']} | {item['candidate_local_event_total']} | "
                    f"{_fmt(item['event_retention_ratio'])} | {item['anchor_local_child_count']} | {item['candidate_local_child_count']} | {item['common_child_count']} |"
                )

        lines.extend(
            [
                "",
                "### Dead Frontier",
                "",
            ]
        )
        if not row["dead_patterns"]:
            lines.append("No dead parents.")
        else:
            lines.append("| Pattern | Ev_a | Pa_a | Pa_x | Ch_a | Ch_x |")
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
            for item in row["dead_patterns"][:5]:
                lines.append(
                    f"| `{item['pattern']}` | {item['anchor_local_event_total']} | {item['anchor_local_parent_count']} | "
                    f"{item['candidate_local_parent_count']} | {item['anchor_local_child_count']} | {item['candidate_local_child_count']} |"
                )

        lines.extend(
            [
                "",
                "### Dark Continuation Frontier",
                "",
            ]
        )
        if not row["dark_patterns"]:
            lines.append("No dark-continuation parents.")
        else:
            lines.append("| Pattern | Ev_a | Pa_a | Pa_x | Ch_a | Ch_x |")
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
            for item in row["dark_patterns"][:5]:
                lines.append(
                    f"| `{item['pattern']}` | {item['anchor_local_event_total']} | {item['anchor_local_parent_count']} | "
                    f"{item['candidate_local_parent_count']} | {item['anchor_local_child_count']} | {item['candidate_local_child_count']} |"
                )

        lines.extend(
            [
                "",
                "### Partial Frontier",
                "",
            ]
        )
        if not row["partial_patterns"]:
            lines.append("No partial frontier patterns.")
        else:
            lines.append("| Pattern | State | Ev_a | Ev_x | Ret_x | Ch_a | Ch_x | Ch_cap |")
            lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
            for item in row["partial_patterns"][:5]:
                lines.append(
                    f"| `{item['pattern']}` | {item['support_state']} | {item['anchor_local_event_total']} | "
                    f"{item['candidate_local_event_total']} | {_fmt(item['event_retention_ratio'])} | "
                    f"{item['anchor_local_child_count']} | {item['candidate_local_child_count']} | {item['common_child_count']} |"
                )

    return "\n".join(lines)


def write_parent_survival_csv(rows: list[dict], csv_path: Path) -> None:
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
        "survivor_parent_count",
        "exact_parent_count",
        "dead_parent_count",
        "dark_parent_count",
        "partial_parent_count",
        "survivor_parent_fraction",
        "exact_parent_fraction",
        "dead_parent_fraction",
        "dark_parent_fraction",
        "partial_parent_fraction",
        "survivor_mass_fraction",
        "exact_mass_fraction",
        "dead_mass_fraction",
        "dark_mass_fraction",
        "partial_mass_fraction",
        "support_coverage_anchor_mean",
        "kernel_wj_mean",
        "top_child_full_match_mass",
        "dataset_path",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_parent_patterns_csv(pattern_rows: list[dict], csv_path: Path) -> None:
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
        "pattern",
        "support_state",
        "anchor_local_parent_count",
        "candidate_local_parent_count",
        "anchor_local_event_total",
        "candidate_local_event_total",
        "anchor_local_child_count",
        "candidate_local_child_count",
        "common_child_count",
        "event_retention_ratio",
        "common_wj",
        "top_child_full_match",
        "dataset_path",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in pattern_rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


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


def _state_sort_key(state: str) -> tuple[int, str]:
    order = {
        "dead": 0,
        "dark": 1,
        "disjoint": 2,
        "loss": 3,
        "spill": 4,
        "mixed": 5,
        "exact": 6,
    }
    return (order.get(state, 99), state)
