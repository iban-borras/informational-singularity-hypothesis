from __future__ import annotations

import csv
import json
from pathlib import Path


def load_kernel_dataset(dataset_path: Path) -> dict:
    with open(dataset_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("rows")
    selection = payload.get("selection")
    if not isinstance(rows, list):
        raise ValueError(f"Expected rows list in {dataset_path}")
    if not isinstance(selection, dict):
        raise ValueError(f"Expected selection object in {dataset_path}")
    payload["_dataset_path"] = str(dataset_path)
    return payload


def build_throughput_rows(payload: dict) -> list[dict]:
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

        anchor_event_total = sum(
            int(item.get("anchor_local_event_total", 0))
            for item in active_patterns
        )
        candidate_event_total = sum(
            int(item.get("candidate_local_event_total", 0))
            for item in active_patterns
        )
        throughput_retention = _ratio_or_none(
            candidate_event_total,
            anchor_event_total,
        )
        throughput_gap_signed = (
            float(throughput_retention - 1.0)
            if throughput_retention is not None
            else None
        )

        parent_ratios = [
            _ratio_or_none(
                int(item.get("candidate_local_event_total", 0)),
                int(item.get("anchor_local_event_total", 0)),
            )
            for item in active_patterns
        ]
        parent_ratios = [value for value in parent_ratios if value is not None]

        weakest_patterns = [
            {
                "pattern": str(item["pattern"]),
                "weight": float(item.get("weight", 0.0)),
                "anchor_local_event_total": int(item.get("anchor_local_event_total", 0)),
                "candidate_local_event_total": int(item.get("candidate_local_event_total", 0)),
                "event_retention_ratio": _ratio_or_none(
                    int(item.get("candidate_local_event_total", 0)),
                    int(item.get("anchor_local_event_total", 0)),
                ),
                "common_js": item.get("common_js"),
                "common_wj": item.get("common_wj"),
                "top_child_full_match": item.get("top_child_full_match"),
            }
            for item in sorted(
                active_patterns,
                key=lambda item: (
                    _sort_ratio(
                        _ratio_or_none(
                            int(item.get("candidate_local_event_total", 0)),
                            int(item.get("anchor_local_event_total", 0)),
                        )
                    ),
                    -float(item.get("weight", 0.0)),
                    str(item["pattern"]),
                ),
            )
        ][:10]

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
                "active_parent_count": len(active_patterns),
                "anchor_event_total": int(anchor_event_total),
                "candidate_event_total": int(candidate_event_total),
                "throughput_retention_ratio": throughput_retention,
                "throughput_gap_signed": throughput_gap_signed,
                "parent_retention_min": min(parent_ratios) if parent_ratios else None,
                "parent_retention_max": max(parent_ratios) if parent_ratios else None,
                "support_coverage_anchor_mean": source_row["support_coverage_anchor_mean"],
                "candidate_spillover_mean": source_row["candidate_spillover_mean"],
                "support_jaccard_mean": source_row["support_jaccard_mean"],
                "kernel_js_mean": source_row["kernel_js_mean"],
                "kernel_wj_mean": source_row["kernel_wj_mean"],
                "top_child_full_match_mass": source_row["top_child_full_match_mass"],
                "top_child_common_match_mass": source_row["top_child_common_match_mass"],
                "comparable_weight_fraction": source_row["comparable_weight_fraction"],
                "selection_source_variant": selection.get("selection_source_variant"),
                "selection_source_run_dir": selection.get("selection_source_run_dir"),
                "dataset_path": payload["_dataset_path"],
                "weakest_patterns": weakest_patterns,
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


def summarize_throughput_rows(rows: list[dict]) -> list[dict]:
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
                "anchor_event_total_avg": _mean_present(group, "anchor_event_total"),
                "candidate_event_total_avg": _mean_present(group, "candidate_event_total"),
                "throughput_retention_ratio_avg": _mean_present(
                    group,
                    "throughput_retention_ratio",
                ),
                "throughput_retention_ratio_min": _min_present(
                    group,
                    "throughput_retention_ratio",
                ),
                "throughput_retention_ratio_max": _max_present(
                    group,
                    "throughput_retention_ratio",
                ),
                "kernel_wj_mean_avg": _mean_present(group, "kernel_wj_mean"),
                "top_child_full_match_mass_avg": _mean_present(
                    group,
                    "top_child_full_match_mass",
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


def render_throughput_report(selection: dict, summary_rows: list[dict], rows: list[dict]) -> str:
    lines = [
        "# Phase 2 Kernel Throughput Readout",
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
        "| Variant | Source | Rows | Ev_a avg | Ev_x avg | Ret_x avg | Ret_x min | Ret_x max | WJ_cs avg | Top_full avg |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summary_rows:
        lines.append(
            f"| {item['variant']} | {item['source_label']} | {item['row_count']} | "
            f"{_fmt(item['anchor_event_total_avg'])} | {_fmt(item['candidate_event_total_avg'])} | "
            f"{_fmt(item['throughput_retention_ratio_avg'])} | {_fmt(item['throughput_retention_ratio_min'])} | {_fmt(item['throughput_retention_ratio_max'])} | "
            f"{_fmt(item['kernel_wj_mean_avg'])} | {_fmt(item['top_child_full_match_mass_avg'])} |"
        )

    lines.extend(
        [
            "",
            "## Readout",
            "",
            "| Variant | Source | Ev_a | Ev_x | Ret_x | Gap_x | Ret_min | Ret_max | Cov_a | WJ_cs | Top_full |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['source_label']} | "
            f"{row['anchor_event_total']} | {row['candidate_event_total']} | "
            f"{_fmt(row['throughput_retention_ratio'])} | {_fmt(row['throughput_gap_signed'])} | "
            f"{_fmt(row['parent_retention_min'])} | {_fmt(row['parent_retention_max'])} | "
            f"{_fmt(row['support_coverage_anchor_mean'])} | {_fmt(row['kernel_wj_mean'])} | {_fmt(row['top_child_full_match_mass'])} |"
        )

    lines.extend(
        [
            "",
            "## Weakest Patterns",
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
        if not row["weakest_patterns"]:
            lines.append("No active anchor patterns.")
            lines.append("")
            continue
        lines.append("| Pattern | Ev_a | Ev_x | Ret_x | WJ_cs | Top_full |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for item in row["weakest_patterns"][:5]:
            lines.append(
                f"| `{item['pattern']}` | {item['anchor_local_event_total']} | {item['candidate_local_event_total']} | "
                f"{_fmt(item['event_retention_ratio'])} | {_fmt(item['common_wj'])} | {str(bool(item['top_child_full_match']))} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Notes",
            "",
            "- `Ev_a` = total anchor realized event mass over active parents in the frozen child universe.",
            "- `Ev_x` = candidate realized event mass over the same anchor-defined parent universe.",
            "- `Ret_x = Ev_x / Ev_a` measures transport-throughput retention before kernel renormalization.",
            "- `Gap_x = Ret_x - 1` is the signed throughput gap. Negative values indicate candidate loss relative to the anchor.",
            "- `WJ_cs` and `Top_full` are carried over from the normalized common-support kernel readout to separate shape from throughput.",
        ]
    )
    return "\n".join(lines)


def write_throughput_csv(rows: list[dict], csv_path: Path) -> None:
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
        "candidate_event_total",
        "throughput_retention_ratio",
        "throughput_gap_signed",
        "parent_retention_min",
        "parent_retention_max",
        "support_coverage_anchor_mean",
        "candidate_spillover_mean",
        "support_jaccard_mean",
        "kernel_js_mean",
        "kernel_wj_mean",
        "top_child_full_match_mass",
        "top_child_common_match_mass",
        "comparable_weight_fraction",
        "dataset_path",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _ratio_or_none(numerator: int | float, denominator: int | float) -> float | None:
    if float(denominator) <= 0.0:
        return None
    return float(float(numerator) / float(denominator))


def _source_kind_sort_key(source_kind: str, null_model: str | None) -> tuple[int, str]:
    if source_kind == "observed":
        return (0, "")
    if null_model == "markov1":
        return (1, null_model)
    if null_model == "matched-lz":
        return (2, null_model)
    return (9, null_model or source_kind)


def _sort_ratio(value: float | None) -> float:
    if value is None:
        return 10.0
    return float(value)


def _mean_present(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def _min_present(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return float(min(values))


def _max_present(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return float(max(values))


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.4f}"
