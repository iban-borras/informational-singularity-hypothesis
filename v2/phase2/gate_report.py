from __future__ import annotations

import csv
import json
from pathlib import Path


def load_profile_comparison(summary_path: Path) -> dict:
    with open(summary_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    comparison = payload.get("comparison")
    if not isinstance(comparison, dict):
        raise ValueError(f"Expected comparison object in {summary_path}")
    comparison["_summary_path"] = str(summary_path)
    return comparison


def build_gate_rows(comparisons: list[dict]) -> list[dict]:
    rows = []
    for item in comparisons:
        shared = int(item["shared_pattern_count"])
        union = int(item["union_pattern_count"])
        rows.append(
            {
                "observed_variant": item["observed_variant"],
                "null_variant": item["null_variant"],
                "pattern_scale": int(item["pattern_scale"]),
                "pattern_selection": item["pattern_selection"],
                "shared_pattern_count": shared,
                "union_pattern_count": union,
                "selection_jaccard": float(item["selection_jaccard"]),
                "weighted_profile_jaccard": float(item["weighted_profile_jaccard"]),
                "weighted_profile_js_divergence": float(item["weighted_profile_js_divergence"]),
                "weighted_relative_mean_lag_gap": float(item["weighted_relative_mean_lag_gap"]),
                "weighted_relative_median_lag_gap": float(item["weighted_relative_median_lag_gap"]),
                "weighted_top_bin_match_fraction": float(item["weighted_top_bin_match_fraction"]),
                "weighted_long_lag_match_fraction": float(item["weighted_long_lag_match_fraction"]),
                "summary_path": item["_summary_path"],
                "readout": _classify_readout(item),
            }
        )

    rows.sort(
        key=lambda row: (
            row["pattern_scale"],
            row["null_variant"],
        )
    )
    return rows


def render_gate_report(rows: list[dict]) -> str:
    if not rows:
        return "# Phase 2 Gate Report\n\nNo rows.\n"

    observed_variant = rows[0]["observed_variant"]
    selection = rows[0]["pattern_selection"]
    scales = sorted({row["pattern_scale"] for row in rows})
    nulls = []
    for row in rows:
        if row["null_variant"] not in nulls:
            nulls.append(row["null_variant"])

    lines = [
        "# Phase 2 Gate Report",
        "",
        f"- Observed variant: {observed_variant}",
        f"- Pattern selection: {selection}",
        f"- Pattern scales: {', '.join(str(value) for value in scales)}",
        f"- Null contrasts: {', '.join(nulls)}",
        "",
        "## Compact Table",
        "",
        "| Null | m | Shared | Union | Selection J | Profile wJ | Profile JS | d_mean | d_median | top_eq | Readout |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['null_variant']} | {row['pattern_scale']} | {row['shared_pattern_count']} | {row['union_pattern_count']} | "
            f"{_fmt(row['selection_jaccard'])} | {_fmt_or_na(row['weighted_profile_jaccard'], na=row['shared_pattern_count'] == 0)} | {_fmt_or_na(row['weighted_profile_js_divergence'], na=row['shared_pattern_count'] == 0)} | "
            f"{_fmt_or_na(row['weighted_relative_mean_lag_gap'], na=row['shared_pattern_count'] == 0)} | {_fmt_or_na(row['weighted_relative_median_lag_gap'], na=row['shared_pattern_count'] == 0)} | "
            f"{_fmt_or_na(row['weighted_top_bin_match_fraction'], na=row['shared_pattern_count'] == 0)} | {row['readout']} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `Selection J` compares which bridge-linked patterns are selected at all.",
            "- `Profile wJ` compares the full lag-bin mass profile inside shared patterns.",
            "- `Profile JS` is Jensen-Shannon divergence on normalized lag-bin counts.",
            "- `top_eq` is the weighted fraction of shared patterns whose dominant lag bin coincides.",
            "- Rows with `Shared = 0` are already open at the level of selected bridge-linked pattern identities.",
            "- For `Shared = 0`, profile fields are shown as `n/a` because no shared-pattern profile remains to compare.",
        ]
    )
    return "\n".join(lines)


def write_gate_csv(rows: list[dict], csv_path: Path) -> None:
    fieldnames = [
        "observed_variant",
        "null_variant",
        "pattern_scale",
        "pattern_selection",
        "shared_pattern_count",
        "union_pattern_count",
        "selection_jaccard",
        "weighted_profile_jaccard",
        "weighted_profile_js_divergence",
        "weighted_relative_mean_lag_gap",
        "weighted_relative_median_lag_gap",
        "weighted_top_bin_match_fraction",
        "weighted_long_lag_match_fraction",
        "readout",
        "summary_path",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _classify_readout(item: dict) -> str:
    shared = int(item["shared_pattern_count"])
    union = int(item["union_pattern_count"])
    if shared == 0 and union > 0:
        return "disjoint bridge-linked selection"
    if float(item["weighted_top_bin_match_fraction"]) == 0.0:
        return "shared patterns but divergent dominant return bins"
    return "partial overlap"


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def _fmt_or_na(value: float | None, *, na: bool = False) -> str:
    if na:
        return "n/a"
    return _fmt(value)
