from __future__ import annotations

import csv
import math
from pathlib import Path


_PARTITION_ORDER = {
    "total": 0,
    "core": 1,
    "shell": 2,
    "arity1": 3,
    "arity2": 4,
    "arity_other": 5,
    "unclassified": 6,
}

_PARTITION_LABELS = {
    "total": "Total",
    "core": "Core",
    "shell": "Shell",
    "arity1": "Arity-1",
    "arity2": "Arity-2",
    "arity_other": "Other-arity",
    "unclassified": "Unclassified",
}

_STATE_NAMES = (
    "exact",
    "dead",
    "dark",
    "disjoint",
    "loss",
    "spill",
    "mixed",
    "other",
)


def build_parent_density_residual_rows(
    lagaware_payload: dict,
    *,
    anchor_core_patterns: set[str],
    anchor_shell_patterns: set[str],
) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for item in lagaware_payload.get("pattern_rows", []):
        if int(item.get("anchor_local_event_total", 0)) <= 0:
            continue
        key = (
            str(item["variant"]),
            str(item["source_label"]),
            str(item["source_kind"]),
            item.get("null_model"),
            item.get("null_seed"),
            str(item["anchor_variant"]),
            int(item["anchor_offset_bits"]),
            int(item["window_bits"]),
            int(item["low_scale"]),
            int(item["high_scale"]),
            str(item["pattern_selection"]),
            str(item["kernel_mode"]),
        )
        grouped.setdefault(key, []).append(item)

    rows = []
    selection = lagaware_payload["selection"]
    for key, pattern_rows in grouped.items():
        (
            variant,
            source_label,
            source_kind,
            null_model,
            null_seed,
            anchor_variant,
            anchor_offset_bits,
            window_bits,
            low_scale,
            high_scale,
            pattern_selection,
            kernel_mode,
        ) = key

        partitions = {name: [] for name in _PARTITION_ORDER}
        total_anchor_event_total = sum(
            int(item.get("anchor_local_event_total", 0))
            for item in pattern_rows
        )

        for item in pattern_rows:
            pattern = str(item["pattern"])
            arity = int(item.get("anchor_local_child_count", 0))
            partitions["total"].append(item)

            if pattern in anchor_core_patterns:
                partitions["core"].append(item)
            elif pattern in anchor_shell_patterns:
                partitions["shell"].append(item)
            else:
                partitions["unclassified"].append(item)

            if arity == 1:
                partitions["arity1"].append(item)
            elif arity == 2:
                partitions["arity2"].append(item)
            else:
                partitions["arity_other"].append(item)

        for partition, items in partitions.items():
            row = _build_partition_row(
                items,
                partition=partition,
                partition_label=_PARTITION_LABELS[partition],
                variant=variant,
                source_label=source_label,
                source_kind=source_kind,
                null_model=null_model,
                null_seed=null_seed,
                anchor_variant=anchor_variant,
                anchor_offset_bits=anchor_offset_bits,
                window_bits=window_bits,
                low_scale=low_scale,
                high_scale=high_scale,
                pattern_selection=pattern_selection,
                kernel_mode=kernel_mode,
                total_anchor_event_total=total_anchor_event_total,
                candidate_lag_bits=int(selection.get("candidate_lag_bits", 0)),
                dataset_path=lagaware_payload["_dataset_path"],
            )
            rows.append(row)

    rows.sort(
        key=lambda row: (
            row["variant"],
            _source_kind_sort_key(row["source_kind"], row.get("null_model")),
            row.get("null_seed") if row.get("null_seed") is not None else -1,
            _PARTITION_ORDER[row["partition"]],
            int(row["anchor_offset_bits"]),
        )
    )
    return rows


def summarize_parent_density_residual_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (
            row["variant"],
            row["source_label"],
            row["source_kind"],
            row.get("null_model"),
            row.get("null_seed"),
            row["partition"],
        )
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for group in grouped.values():
        first = group[0]
        active = [row for row in group if int(row["partition_anchor_event_total"]) > 0]
        if not active:
            continue
        survivor = [row for row in active if int(row["survivor_pattern_count"]) > 0]

        summary_rows.append(
            {
                "variant": first["variant"],
                "source_label": first["source_label"],
                "source_kind": first["source_kind"],
                "null_model": first.get("null_model"),
                "null_seed": first.get("null_seed"),
                "partition": first["partition"],
                "partition_label": first["partition_label"],
                "row_count": len(group),
                "active_offset_count": len(active),
                "exact_offset_count": sum(1 for row in active if int(row["exact_pattern_count"]) > 0),
                "survivor_offset_count": len(survivor),
                "partition_pattern_count_min": min(int(row["partition_pattern_count"]) for row in active),
                "partition_pattern_count_max": max(int(row["partition_pattern_count"]) for row in active),
                "exact_pattern_count_min": min(int(row["exact_pattern_count"]) for row in active),
                "exact_pattern_count_max": max(int(row["exact_pattern_count"]) for row in active),
                "survivor_pattern_count_min": min(int(row["survivor_pattern_count"]) for row in active),
                "survivor_pattern_count_max": max(int(row["survivor_pattern_count"]) for row in active),
                "partition_anchor_event_total_min": min(
                    int(row["partition_anchor_event_total"]) for row in active
                ),
                "partition_anchor_event_total_max": max(
                    int(row["partition_anchor_event_total"]) for row in active
                ),
                "partition_anchor_event_total_sum": sum(
                    int(row["partition_anchor_event_total"]) for row in active
                ),
                "partition_candidate_event_total_sum": sum(
                    int(row["partition_candidate_event_total"]) for row in active
                ),
                "survivor_anchor_event_total_sum": sum(
                    int(row["survivor_anchor_event_total"]) for row in active
                ),
                "survivor_candidate_event_total_sum": sum(
                    int(row["survivor_candidate_event_total"]) for row in active
                ),
                "partition_mass_share_of_total_anchor_avg": _mean_present(
                    active,
                    "partition_mass_share_of_total_anchor",
                ),
                "partition_mass_retention_pooled": _ratio_or_none(
                    sum(int(row["partition_candidate_event_total"]) for row in active),
                    sum(int(row["partition_anchor_event_total"]) for row in active),
                ),
                "partition_mass_retention_avg": _mean_present(
                    active,
                    "partition_mass_retention",
                ),
                "partition_mass_retention_min": _min_present(
                    active,
                    "partition_mass_retention",
                ),
                "partition_mass_retention_max": _max_present(
                    active,
                    "partition_mass_retention",
                ),
                "survivor_internal_retention_avg": _mean_present(
                    survivor,
                    "survivor_internal_retention",
                ),
                "survivor_internal_retention_pooled": _ratio_or_none(
                    sum(int(row["survivor_candidate_event_total"]) for row in active),
                    sum(int(row["survivor_anchor_event_total"]) for row in active),
                ),
                "survivor_internal_retention_min": _min_present(
                    survivor,
                    "survivor_internal_retention",
                ),
                "survivor_internal_retention_max": _max_present(
                    survivor,
                    "survivor_internal_retention",
                ),
                "survivor_internal_bias_log2_mean_avg": _mean_present(
                    survivor,
                    "survivor_internal_bias_log2_mean",
                ),
                "survivor_internal_bias_log2_mean_pooled": _ratio_or_none(
                    sum(float(row["survivor_bias_weighted_sum"]) for row in active),
                    sum(int(row["survivor_anchor_event_total"]) for row in active),
                ),
                "survivor_internal_bias_log2_mean_min": _min_present(
                    survivor,
                    "survivor_internal_bias_log2_mean",
                ),
                "survivor_internal_bias_log2_mean_max": _max_present(
                    survivor,
                    "survivor_internal_bias_log2_mean",
                ),
                "survivor_internal_deformation_log2_mean_avg": _mean_present(
                    survivor,
                    "survivor_internal_deformation_log2_mean",
                ),
                "survivor_internal_deformation_log2_mean_pooled": _ratio_or_none(
                    sum(float(row["survivor_deformation_weighted_sum"]) for row in active),
                    sum(int(row["survivor_anchor_event_total"]) for row in active),
                ),
                "survivor_internal_deformation_log2_mean_max": _max_present(
                    survivor,
                    "survivor_internal_deformation_log2_mean",
                ),
                "survivor_pattern_ratio_min": _min_present(
                    survivor,
                    "survivor_pattern_ratio_min",
                ),
                "survivor_pattern_ratio_max": _max_present(
                    survivor,
                    "survivor_pattern_ratio_max",
                ),
                "exact_anchor_mass_fraction_avg": _mean_present(
                    active,
                    "exact_anchor_mass_fraction",
                ),
                "exact_anchor_mass_fraction_pooled": _ratio_or_none(
                    sum(int(row["exact_anchor_event_total"]) for row in active),
                    sum(int(row["partition_anchor_event_total"]) for row in active),
                ),
                "dead_anchor_mass_fraction_avg": _mean_present(
                    active,
                    "dead_anchor_mass_fraction",
                ),
                "dead_anchor_mass_fraction_pooled": _ratio_or_none(
                    sum(int(row["dead_anchor_event_total"]) for row in active),
                    sum(int(row["partition_anchor_event_total"]) for row in active),
                ),
                "dark_anchor_mass_fraction_avg": _mean_present(
                    active,
                    "dark_anchor_mass_fraction",
                ),
                "dark_anchor_mass_fraction_pooled": _ratio_or_none(
                    sum(int(row["dark_anchor_event_total"]) for row in active),
                    sum(int(row["partition_anchor_event_total"]) for row in active),
                ),
                "loss_anchor_mass_fraction_avg": _mean_present(
                    active,
                    "loss_anchor_mass_fraction",
                ),
                "spill_anchor_mass_fraction_avg": _mean_present(
                    active,
                    "spill_anchor_mass_fraction",
                ),
                "mixed_anchor_mass_fraction_avg": _mean_present(
                    active,
                    "mixed_anchor_mass_fraction",
                ),
            }
        )

    summary_rows.sort(
        key=lambda row: (
            row["variant"],
            _source_kind_sort_key(row["source_kind"], row.get("null_model")),
            row.get("null_seed") if row.get("null_seed") is not None else -1,
            _PARTITION_ORDER[row["partition"]],
        )
    )
    return summary_rows


def render_parent_density_residual_report(
    selection: dict,
    summary_rows: list[dict],
    rows: list[dict],
) -> str:
    lines = [
        "# Phase 2 Parent Density Residual Readout",
        "",
        "## Selection",
        "",
        f"- Anchor variant: {selection['anchor_variant']}",
        f"- Candidate variant: {selection['candidate_variant']}",
        f"- Variants: {', '.join(selection['variants'])}",
        f"- Low/high scales: {selection['low_scale']} -> {selection['high_scale']}",
        f"- Pattern selection: {selection['pattern_selection']}",
        f"- Candidate lag bits: {selection.get('candidate_lag_bits')}",
        f"- Offsets: {', '.join(str(value) for value in selection['offsets'])}",
        f"- Source revalidation run: `{selection['revalidation_summary_path']}`",
        "",
        "## Grouped Summary",
        "",
        "| Variant | Source | Partition | Active | Exact | Survivors | Part ret | Part min | Part max | Surv ret | Surv def | Surv bias | Dead m | Dark m |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summary_rows:
        lines.append(
            f"| {item['variant']} | {item['source_label']} | {item['partition_label']} | "
            f"{item['active_offset_count']} / {item['row_count']} | "
            f"{item['exact_offset_count']} / {item['row_count']} | "
            f"{item['survivor_offset_count']} / {item['row_count']} | "
            f"{_fmt(item['partition_mass_retention_pooled'])} | "
            f"{_fmt(item['partition_mass_retention_min'])} | "
            f"{_fmt(item['partition_mass_retention_max'])} | "
            f"{_fmt(item['survivor_internal_retention_pooled'])} | "
            f"{_fmt(item['survivor_internal_deformation_log2_mean_pooled'])} | "
            f"{_fmt(item['survivor_internal_bias_log2_mean_pooled'])} | "
            f"{_fmt(item['dead_anchor_mass_fraction_pooled'])} | "
            f"{_fmt(item['dark_anchor_mass_fraction_pooled'])} |"
        )

    lines.extend(
        [
            "",
            "## Offset Readout",
            "",
            "| Variant | Source | Partition | Offset | Count | Exact | Dead | Dark | Share | Part ret | Surv ret | Surv def | Surv bias |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['source_label']} | {row['partition_label']} | "
            f"{row['anchor_offset_bits']} | {row['partition_pattern_count']} | "
            f"{row['exact_pattern_count']} | {row['dead_pattern_count']} | {row['dark_pattern_count']} | "
            f"{_fmt(row['partition_mass_share_of_total_anchor'])} | "
            f"{_fmt(row['partition_mass_retention'])} | "
            f"{_fmt(row['survivor_internal_retention'])} | "
            f"{_fmt(row['survivor_internal_deformation_log2_mean'])} | "
            f"{_fmt(row['survivor_internal_bias_log2_mean'])} |"
        )

    lines.extend(
        [
            "",
            "## Strongest Survivor Residual Rows",
            "",
        ]
    )
    strongest = sorted(
        [row for row in rows if row["survivor_pattern_count"] > 0],
        key=lambda item: (
            -float(item["survivor_internal_deformation_log2_mean"] or 0.0),
            item["variant"],
            _PARTITION_ORDER[item["partition"]],
            int(item["anchor_offset_bits"]),
        ),
    )[:10]
    for row in strongest:
        lines.append(
            f"### {row['variant']} | {row['source_label']} | {row['partition_label']} | {row['anchor_offset_bits']}"
        )
        lines.append("")
        lines.append(
            f"- Partition mass retention: {_fmt(row['partition_mass_retention'])}"
        )
        lines.append(
            f"- Survivor internal retention: {_fmt(row['survivor_internal_retention'])}"
        )
        lines.append(
            f"- Survivor deformation (abs log2 mean): {_fmt(row['survivor_internal_deformation_log2_mean'])}"
        )
        lines.append(
            f"- Survivor bias (signed log2 mean): {_fmt(row['survivor_internal_bias_log2_mean'])}"
        )
        lines.append(
            f"- Survivor ratio range: {_fmt(row['survivor_pattern_ratio_min'])} -> {_fmt(row['survivor_pattern_ratio_max'])}"
        )
        lines.append("")
        if not row["top_residual_patterns"]:
            lines.append("No surviving patterns in this partition.")
            lines.append("")
            continue
        lines.append("| Pattern | State | Ev_a | Ev_x | Ratio | |log2| |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
        for item in row["top_residual_patterns"][:5]:
            lines.append(
                f"| `{item['pattern']}` | {item['support_state']} | "
                f"{item['anchor_local_event_total']} | {item['candidate_local_event_total']} | "
                f"{_fmt(item['event_retention_ratio'])} | {_fmt(item['abs_log2_ratio'])} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Notes",
            "",
            "- `Part ret` is partition-level mass retention: sum(candidate events) / sum(anchor events) over the full partition, including `dead` and `dark` patterns.",
            "- `Surv ret`, `Surv def`, and `Surv bias` are conditional survivor-internal metrics computed only where `candidate_local_event_total > 0`; this avoids the undefined `log2(0)` case.",
            "- `Dead m` and `Dark m` are anchor-mass fractions inside the partition, so support pruning/zombification remains visible instead of being hidden by survivor-only conditioning.",
            "- `core/shell` come from the independent shell-lag probe; `arity-*` come from the anchor-local child count inside the lag-aware band.",
        ]
    )
    return "\n".join(lines)


def write_parent_density_residual_csv(rows: list[dict], csv_path: Path) -> None:
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
        "partition",
        "partition_label",
        "partition_pattern_count",
        "exact_pattern_count",
        "dead_pattern_count",
        "dark_pattern_count",
        "disjoint_pattern_count",
        "loss_pattern_count",
        "spill_pattern_count",
        "mixed_pattern_count",
        "other_pattern_count",
        "survivor_pattern_count",
        "total_anchor_event_total",
        "partition_anchor_event_total",
        "partition_candidate_event_total",
        "survivor_anchor_event_total",
        "survivor_candidate_event_total",
        "partition_mass_share_of_total_anchor",
        "partition_mass_retention",
        "survivor_internal_retention",
        "survivor_bias_weighted_sum",
        "survivor_deformation_weighted_sum",
        "survivor_internal_bias_log2_mean",
        "survivor_internal_deformation_log2_mean",
        "survivor_pattern_ratio_min",
        "survivor_pattern_ratio_max",
        "exact_anchor_mass_fraction",
        "dead_anchor_mass_fraction",
        "dark_anchor_mass_fraction",
        "disjoint_anchor_mass_fraction",
        "loss_anchor_mass_fraction",
        "spill_anchor_mass_fraction",
        "mixed_anchor_mass_fraction",
        "other_anchor_mass_fraction",
        "dataset_path",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _build_partition_row(
    items: list[dict],
    *,
    partition: str,
    partition_label: str,
    variant: str,
    source_label: str,
    source_kind: str,
    null_model: str | None,
    null_seed: int | None,
    anchor_variant: str,
    anchor_offset_bits: int,
    window_bits: int,
    low_scale: int,
    high_scale: int,
    pattern_selection: str,
    kernel_mode: str,
    total_anchor_event_total: int,
    candidate_lag_bits: int,
    dataset_path: str,
) -> dict:
    state_counts = {name: 0 for name in _STATE_NAMES}
    state_anchor_event_totals = {name: 0 for name in _STATE_NAMES}

    partition_anchor_event_total = 0
    partition_candidate_event_total = 0
    survivor_anchor_event_total = 0
    survivor_candidate_event_total = 0
    ratios = []
    top_residual_patterns = []

    for item in items:
        state = str(item.get("support_state", "other"))
        if state not in state_counts:
            state = "other"
        anchor_events = int(item.get("anchor_local_event_total", 0))
        candidate_events = int(item.get("candidate_local_event_total", 0))

        state_counts[state] += 1
        state_anchor_event_totals[state] += anchor_events
        partition_anchor_event_total += anchor_events
        partition_candidate_event_total += candidate_events

        ratio = _ratio_or_none(candidate_events, anchor_events)
        if ratio is None or ratio <= 0.0:
            continue

        survivor_anchor_event_total += anchor_events
        survivor_candidate_event_total += candidate_events
        log2_ratio = math.log2(ratio)
        ratios.append((float(anchor_events), ratio, log2_ratio))
        top_residual_patterns.append(
            {
                "pattern": str(item["pattern"]),
                "support_state": state,
                "anchor_local_event_total": anchor_events,
                "candidate_local_event_total": candidate_events,
                "event_retention_ratio": ratio,
                "log2_ratio": log2_ratio,
                "abs_log2_ratio": abs(log2_ratio),
            }
        )

    partition_mass_share = _ratio_or_none(partition_anchor_event_total, total_anchor_event_total)
    partition_mass_retention = _ratio_or_none(
        partition_candidate_event_total,
        partition_anchor_event_total,
    )
    survivor_internal_retention = _ratio_or_none(
        survivor_candidate_event_total,
        survivor_anchor_event_total,
    )

    survivor_bias = None
    survivor_deformation = None
    survivor_bias_weighted_sum = 0.0
    survivor_deformation_weighted_sum = 0.0
    if survivor_anchor_event_total > 0 and ratios:
        survivor_bias_weighted_sum = sum(weight * log2_ratio for weight, _, log2_ratio in ratios)
        survivor_deformation_weighted_sum = sum(
            weight * abs(log2_ratio) for weight, _, log2_ratio in ratios
        )
        survivor_bias = survivor_bias_weighted_sum / survivor_anchor_event_total
        survivor_deformation = survivor_deformation_weighted_sum / survivor_anchor_event_total

    top_residual_patterns.sort(
        key=lambda item: (
            -float(item["abs_log2_ratio"]),
            -int(item["anchor_local_event_total"]),
            item["pattern"],
        )
    )

    row = {
        "variant": variant,
        "source_label": source_label,
        "source_kind": source_kind,
        "null_model": null_model,
        "null_seed": null_seed,
        "anchor_variant": anchor_variant,
        "anchor_offset_bits": anchor_offset_bits,
        "window_bits": window_bits,
        "low_scale": low_scale,
        "high_scale": high_scale,
        "pattern_selection": pattern_selection,
        "kernel_mode": kernel_mode,
        "candidate_lag_bits": candidate_lag_bits,
        "partition": partition,
        "partition_label": partition_label,
        "partition_pattern_count": len(items),
        "survivor_pattern_count": len(ratios),
        "total_anchor_event_total": int(total_anchor_event_total),
        "partition_anchor_event_total": int(partition_anchor_event_total),
        "partition_candidate_event_total": int(partition_candidate_event_total),
        "survivor_anchor_event_total": int(survivor_anchor_event_total),
        "survivor_candidate_event_total": int(survivor_candidate_event_total),
        "partition_mass_share_of_total_anchor": partition_mass_share,
        "partition_mass_retention": partition_mass_retention,
        "survivor_internal_retention": survivor_internal_retention,
        "survivor_bias_weighted_sum": survivor_bias_weighted_sum,
        "survivor_deformation_weighted_sum": survivor_deformation_weighted_sum,
        "survivor_internal_bias_log2_mean": survivor_bias,
        "survivor_internal_deformation_log2_mean": survivor_deformation,
        "survivor_pattern_ratio_min": min((ratio for _, ratio, _ in ratios), default=None),
        "survivor_pattern_ratio_max": max((ratio for _, ratio, _ in ratios), default=None),
        "dataset_path": dataset_path,
        "top_residual_patterns": top_residual_patterns[:10],
    }

    for state in _STATE_NAMES:
        row[f"{state}_pattern_count"] = int(state_counts[state])
        row[f"{state}_anchor_event_total"] = int(state_anchor_event_totals[state])
        row[f"{state}_anchor_mass_fraction"] = _ratio_or_none(
            state_anchor_event_totals[state],
            partition_anchor_event_total,
        )
    return row


def _ratio_or_none(numerator: int | float, denominator: int | float) -> float | None:
    if float(denominator) <= 0.0:
        return None
    return float(numerator) / float(denominator)


def _mean_present(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _min_present(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return min(values)


def _max_present(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return max(values)


def _source_kind_sort_key(source_kind: str, null_model: str | None) -> tuple[int, str]:
    if source_kind == "observed":
        return (0, "observed")
    if null_model == "markov1":
        return (1, "markov1")
    if null_model == "matched-lz":
        return (2, "matched-lz")
    return (9, source_kind)


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"
