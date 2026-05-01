from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Any


def build_flow_entropy_readout(component_run: Path) -> dict[str, Any]:
    rows = load_csv(component_run / "source_summary.csv")
    readout_rows = []
    for channel in sorted({row["channel"] for row in rows}):
        channel_rows = [row for row in rows if row["channel"] == channel]
        matched_rows = [row for row in channel_rows if row["variant"] == "B-matched-lz"]
        if not matched_rows:
            raise ValueError(f"No matched-LZ envelope rows found for channel {channel!r}")
        observed_rows = [row for row in channel_rows if row["source_label"] == "observed"]
        markov_rows = [row for row in channel_rows if row["variant"] == "B-markov1"]
        matched_stats = summarize_matched_lz(channel, matched_rows)
        readout_rows.extend(score_observed(channel, observed_rows, matched_stats))
        readout_rows.extend(score_markov(channel, markov_rows))
    readout_rows.sort(key=lambda row: (row["channel"], source_order(row), row["variant"], row["source_label"]))
    return {
        "selection": {
            "component_run": str(component_run),
        },
        "notes": [
            "N3-07 is a direct flow-entropy readout derived from the audited N3-05c component-normalized quotient.",
            "Lower component entropy and fewer effective components indicate more canalized flow.",
            "Higher top-k component share indicates stronger concentration.",
            "Scores are evaluated against the matched-LZ seed envelope per channel.",
            "This is internal and pre-geometric: it is a routing-thermodynamic readout, not a topology or curvature claim.",
        ],
        "flow_entropy_rows": readout_rows,
    }


def summarize_matched_lz(channel: str, rows: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "channel": channel,
        "entropy_min": min_float(rows, "component_weight_entropy_norm"),
        "entropy_avg": avg_float(rows, "component_weight_entropy_norm"),
        "entropy_max": max_float(rows, "component_weight_entropy_norm"),
        "effective_min": min_float(rows, "effective_weight_components"),
        "effective_avg": avg_float(rows, "effective_weight_components"),
        "effective_max": max_float(rows, "effective_weight_components"),
        "top3_min": min_float(rows, "top3_component_weight_share"),
        "top3_avg": avg_float(rows, "top3_component_weight_share"),
        "top3_max": max_float(rows, "top3_component_weight_share"),
        "top5_min": min_float(rows, "top5_component_weight_share"),
        "top5_avg": avg_float(rows, "top5_component_weight_share"),
        "top5_max": max_float(rows, "top5_component_weight_share"),
        "matched_seed_count": len(rows),
    }


def score_observed(
    channel: str,
    rows: list[dict[str, str]],
    matched: dict[str, Any],
) -> list[dict[str, Any]]:
    scored = []
    for row in rows:
        entropy = optional_float(row.get("component_weight_entropy_norm"))
        effective = optional_float(row.get("effective_weight_components"))
        top3 = optional_float(row.get("top3_component_weight_share"))
        top5 = optional_float(row.get("top5_component_weight_share"))
        entropy_gap = None if entropy is None else matched["entropy_min"] - entropy
        effective_gap = None if effective is None else matched["effective_min"] - effective
        top3_gap = None if top3 is None else top3 - matched["top3_max"]
        top5_gap = None if top5 is None else top5 - matched["top5_max"]
        score_terms = [
            entropy_gap is not None and entropy_gap > 0,
            effective_gap is not None and effective_gap > 0,
            top3_gap is not None and top3_gap > 0,
            top5_gap is not None and top5_gap > 0,
        ]
        score = sum(1 for item in score_terms if item)
        scored.append(
            {
                "variant": row["variant"],
                "source_label": row["source_label"],
                "source_class": "observed",
                "channel": channel,
                "component_entropy": entropy,
                "effective_components": effective,
                "top3_share": top3,
                "top5_share": top5,
                "matched_entropy_min": matched["entropy_min"],
                "matched_entropy_avg": matched["entropy_avg"],
                "matched_entropy_max": matched["entropy_max"],
                "matched_effective_min": matched["effective_min"],
                "matched_effective_avg": matched["effective_avg"],
                "matched_effective_max": matched["effective_max"],
                "matched_top3_min": matched["top3_min"],
                "matched_top3_avg": matched["top3_avg"],
                "matched_top3_max": matched["top3_max"],
                "matched_top5_min": matched["top5_min"],
                "matched_top5_avg": matched["top5_avg"],
                "matched_top5_max": matched["top5_max"],
                "entropy_gap_vs_best_matched": entropy_gap,
                "effective_gap_vs_best_matched": effective_gap,
                "top3_gap_vs_best_matched": top3_gap,
                "top5_gap_vs_best_matched": top5_gap,
                "flow_entropy_score": score,
                "verdict": observed_verdict(score),
            }
        )
    return scored


def score_markov(channel: str, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    scored = []
    for row in rows:
        scored.append(
            {
                "variant": row["variant"],
                "source_label": row["source_label"],
                "source_class": "collapse-control",
                "channel": channel,
                "component_entropy": optional_float(row.get("component_weight_entropy_norm")),
                "effective_components": optional_float(row.get("effective_weight_components")),
                "top3_share": optional_float(row.get("top3_component_weight_share")),
                "top5_share": optional_float(row.get("top5_component_weight_share")),
                "matched_entropy_min": None,
                "matched_entropy_avg": None,
                "matched_entropy_max": None,
                "matched_effective_min": None,
                "matched_effective_avg": None,
                "matched_effective_max": None,
                "matched_top3_min": None,
                "matched_top3_avg": None,
                "matched_top3_max": None,
                "matched_top5_min": None,
                "matched_top5_avg": None,
                "matched_top5_max": None,
                "entropy_gap_vs_best_matched": None,
                "effective_gap_vs_best_matched": None,
                "top3_gap_vs_best_matched": None,
                "top5_gap_vs_best_matched": None,
                "flow_entropy_score": 0,
                "verdict": "no-flow-collapse-control",
            }
        )
    return scored


def observed_verdict(score: int) -> str:
    if score >= 4:
        return "strict-low-entropy-flow"
    if score >= 3:
        return "strong-low-entropy-flow"
    if score >= 2:
        return "partial-low-entropy-flow"
    return "not-separated-from-matched-lz"


def source_order(row: dict[str, Any]) -> int:
    order = {"observed": 0, "collapse-control": 1}
    return order.get(row.get("source_class", ""), 9)


def load_csv(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def numeric_values(rows: list[dict[str, str]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = optional_float(row.get(key))
        if value is not None:
            values.append(value)
    return values


def min_float(rows: list[dict[str, str]], key: str) -> float | None:
    values = numeric_values(rows, key)
    return min(values) if values else None


def max_float(rows: list[dict[str, str]], key: str) -> float | None:
    values = numeric_values(rows, key)
    return max(values) if values else None


def avg_float(rows: list[dict[str, str]], key: str) -> float | None:
    values = numeric_values(rows, key)
    return statistics.mean(values) if values else None


def optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)
