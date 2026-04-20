from __future__ import annotations

from datetime import datetime
from typing import Iterable

from v2.phase1.report import (
    classify_variant_role,
    discover_phase1_runs,
    filter_runs,
    infer_family_from_latest_run,
    select_latest_per_variant,
    split_rows_by_role,
)


def build_coherence_rows(
    runs: Iterable[dict], *, tail_edges: int = 2, include_hints: bool = False
) -> list[dict]:
    rows = []
    for run in runs:
        config = run["dataset"]["config"]
        summary = run["summary"]
        prefix_summary = _require_policy_series(summary, "prefix", config["variant"], run["_run_dir"])
        suffix_summary = _require_policy_series(summary, "suffix", config["variant"], run["_run_dir"])
        prefix_moments = _require_payload(
            run.get("moments_prefix") or [],
            "moments_prefix",
            config["variant"],
            run["_run_dir"],
        )

        tail_prefix = prefix_summary[-tail_edges:] if tail_edges > 0 else prefix_summary
        tail_suffix = suffix_summary[-tail_edges:] if tail_edges > 0 else suffix_summary
        tail_moments = prefix_moments[-tail_edges:] if tail_edges > 0 else prefix_moments

        scale_summary = summary.get("scale_summary", [])
        kept_first = scale_summary[0]["kept_pattern_count"] if scale_summary else 0
        kept_last = scale_summary[-1]["kept_pattern_count"] if scale_summary else 0
        breadth_ratio = (kept_last / kept_first) if kept_first else 0.0

        tail_survival_stats = _metric_stats(tail_prefix, "survival_fraction")
        tail_entropy_stats = _metric_stats(tail_prefix, "mean_child_entropy")
        tail_gap_stats = _mean_pair_gap_stats(tail_prefix, tail_suffix, "mean_child_entropy")
        tail_singleton_gap_stats = _mean_pair_gap_stats(tail_prefix, tail_suffix, "singleton_fraction")
        tail_q1 = _mean_ratio(tail_moments, "1")
        tail_q2 = _mean_ratio(tail_moments, "2")
        tail_q3 = _mean_ratio(tail_moments, "3")
        tail_q_spread = None if tail_q1 is None or tail_q3 is None else float(tail_q1 - tail_q3)

        row = {
            "variant": config["variant"],
            "variant_role": classify_variant_role(config["variant"]),
            "iteration": config.get("iteration"),
            "segment_bits": config.get("segment_bits"),
            "num_segments": config.get("num_segments"),
            "segment_offset_bits": config.get("segment_offset_bits", 0),
            "scales": list(config.get("scales", [])),
            "policies": list(config.get("policies", [])),
            "tail_edges": tail_edges,
            "kept_first": kept_first,
            "kept_last": kept_last,
            "breadth_ratio_last_to_first": breadth_ratio,
            "retention_last": (
                float(scale_summary[-1]["kept_pattern_count"]) / float(scale_summary[-1]["candidate_pattern_count"])
                if scale_summary and float(scale_summary[-1]["candidate_pattern_count"]) > 0
                else None
            ),
            "tail_survival": tail_survival_stats["mean"],
            "tail_survival_mean": tail_survival_stats["mean"],
            "tail_survival_min": tail_survival_stats["min"],
            "tail_survival_last": tail_survival_stats["last"],
            "tail_entropy": tail_entropy_stats["mean"],
            "tail_entropy_mean": tail_entropy_stats["mean"],
            "tail_entropy_min": tail_entropy_stats["min"],
            "tail_entropy_last": tail_entropy_stats["last"],
            "tail_q1": tail_q1,
            "tail_q2": tail_q2,
            "tail_q3": tail_q3,
            "tail_q_spread": tail_q_spread,
            "tail_entropy_gap": tail_gap_stats["abs"],
            "tail_entropy_gap_abs": tail_gap_stats["abs"],
            "tail_entropy_gap_signed": tail_gap_stats["signed"],
            "tail_entropy_gap_relative": tail_gap_stats["relative"],
            "tail_singleton_gap": tail_singleton_gap_stats["abs"],
            "tail_singleton_gap_abs": tail_singleton_gap_stats["abs"],
            "tail_singleton_gap_signed": tail_singleton_gap_stats["signed"],
            "tail_singleton_gap_relative": tail_singleton_gap_stats["relative"],
            "summary_path": run["_summary_path"],
            "run_dir": run["_run_dir"],
            "generated_at": run["generated_at"],
        }
        row["selective_persistence_index_simple"] = _safe_product(
            row["retention_last"],
            row["tail_survival_last"],
        )
        row["coherence_hint"] = _coherence_hint(row) if include_hints else None
        rows.append(row)

    return rows


def render_console_summary(rows: list[dict], *, include_hints: bool = False) -> str:
    if not rows:
        return "No matching Phase 1 coherence rows found."

    grouped = split_rows_by_role(rows)
    sections: list[str] = []

    if grouped["primary"]:
        sections.extend(
            [
                "Primary coherence readout (B/F/A/M/N core contrast)",
                "-" * 106,
                _render_header(include_hints=include_hints),
                *_render_rows(grouped["primary"], include_hints=include_hints),
            ]
        )

    if grouped["supporting"]:
        if sections:
            sections.append("")
        sections.extend(
            [
                "Supporting coherence readout",
                "-" * 106,
                _render_header(include_hints=include_hints),
                *_render_rows(grouped["supporting"], include_hints=include_hints),
            ]
        )

    if grouped["other"]:
        if sections:
            sections.append("")
        sections.extend(
            [
                "Other variants",
                "-" * 106,
                _render_header(include_hints=include_hints),
                *_render_rows(grouped["other"], include_hints=include_hints),
            ]
        )

    return "\n".join(sections)


def render_markdown_report(rows: list[dict], selection: dict) -> str:
    grouped = split_rows_by_role(rows)
    include_hints = selection.get("include_hints", False)
    lines = [
        "# HSI v2 Phase 1 Coherence Report",
        "",
        "## Selection",
        "",
        f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Iterations in report: `{', '.join(str(value) for value in selection['iterations'])}`",
        f"- Segment layout: `{selection['num_segments']} x {selection['segment_bits']}` bits",
        f"- Segment offset: `{selection['segment_offset_bits']}` bits",
        f"- Scale ladder: `{', '.join(str(value) for value in selection['scales'])}`",
        f"- Policies: `{', '.join(selection['policies'])}`",
        f"- Tail edges used: `{selection['tail_edges']}`",
        f"- Family inferred: `{selection['family_inferred']}`",
        f"- Interpretive labels enabled: `{include_hints}`",
        f"- Variants in report: `{', '.join(row['variant'] for row in rows)}`",
        "",
        "## Primary Coherence Readout",
        "",
        "This report focuses on tail coherence derived from the last edges of the projective tower.",
        "",
    ]

    if grouped["primary"]:
        lines.extend(_render_markdown_table(grouped["primary"], include_hints=include_hints))
    else:
        lines.append("No primary rows available.")

    if grouped["supporting"]:
        lines.extend(["", "## Supporting Coherence Readout", ""])
        lines.extend(_render_markdown_table(grouped["supporting"], include_hints=include_hints))

    if grouped["other"]:
        lines.extend(["", "## Other Variants", ""])
        lines.extend(_render_markdown_table(grouped["other"], include_hints=include_hints))

    lines.extend(["", "## Run Paths", ""])
    for row in rows:
        lines.append(f"- `{row['variant']}`: `{row['summary_path']}`")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `SPI_simple` is the minimal selective-persistence index defined as `retention@last * surv_last`.",
            "- Tail survival and entropy now expose mean, min, and last summaries to prevent smoothing bias at the top of the tower.",
            "- Tail entropy and singleton gaps now expose signed, relative, and absolute summaries.",
            "- Signed gaps are computed as `prefix - suffix`.",
            "- Relative gaps are computed as `(prefix - suffix) / (|prefix| + |suffix|)` when the denominator is non-zero.",
            "- Absolute gaps are still preserved in the JSON payload for magnitude-only inspection.",
        ]
    )
    if include_hints:
        lines.extend(
            [
                "- Interpretive labels are heuristic reading aids derived from the current batch, not canonical outputs of Phase 1.",
                "- Treat `coherent / rigid / crystalline / collapsed / trivial` as descriptive synthesis, not as mathematically final classes.",
            ]
        )

    return "\n".join(lines)


def build_selection(rows: list[dict], *, family_inferred: bool, tail_edges: int) -> dict:
    iterations = sorted({row["iteration"] for row in rows if row["iteration"] is not None})
    segment_bits = sorted({row["segment_bits"] for row in rows})
    num_segments = sorted({row["num_segments"] for row in rows})
    segment_offset_bits = sorted({row["segment_offset_bits"] for row in rows})
    return {
        "iterations": iterations,
        "segment_bits": segment_bits[0] if len(segment_bits) == 1 else "mixed",
        "num_segments": num_segments[0] if len(num_segments) == 1 else "mixed",
        "segment_offset_bits": segment_offset_bits[0] if len(segment_offset_bits) == 1 else "mixed",
        "scales": rows[0]["scales"],
        "policies": rows[0]["policies"],
        "variants": [row["variant"] for row in rows],
        "tail_edges": tail_edges,
        "family_inferred": family_inferred,
        "include_hints": bool(rows and rows[0].get("coherence_hint") is not None),
    }


def _render_header(*, include_hints: bool) -> str:
    header = (
        f"{'variant':8} {'iter':>4} {'b_ratio':>9} {'ret':>8} {'spi':>8} {'s_mean':>8} {'s_min':>8} {'s_last':>8} "
        f"{'e_mean':>8} {'e_last':>8} {'q_sp':>8} {'gap_s':>8}"
    )
    if include_hints:
        header += f" {'hint':>12}"
    return header


def _render_rows(rows: list[dict], *, include_hints: bool) -> list[str]:
    rendered = []
    for row in rows:
        line = (
            f"{row['variant']:8} "
            f"{row['iteration']:>4} "
            f"{_fmt(row['breadth_ratio_last_to_first']):>9} "
            f"{_fmt(row['retention_last']):>8} "
            f"{_fmt(row['selective_persistence_index_simple']):>8} "
            f"{_fmt(row['tail_survival_mean']):>8} "
            f"{_fmt(row['tail_survival_min']):>8} "
            f"{_fmt(row['tail_survival_last']):>8} "
            f"{_fmt(row['tail_entropy_mean']):>8} "
            f"{_fmt(row['tail_entropy_last']):>8} "
            f"{_fmt(row['tail_q_spread']):>8} "
            f"{_fmt(row['tail_entropy_gap_signed']):>8}"
        )
        if include_hints:
            line += f" {(row.get('coherence_hint') or 'n/a'):>12}"
        rendered.append(line)
    return rendered


def _render_markdown_table(rows: list[dict], *, include_hints: bool) -> list[str]:
    if include_hints:
        lines = [
            "| Variant | Iter | Role | breadth ratio | retention@last | SPI_simple | surv mean | surv min | surv last | ent mean | ent last | q-spread | gap signed | gap rel | Interpretive label |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    else:
        lines = [
            "| Variant | Iter | Role | breadth ratio | retention@last | SPI_simple | surv mean | surv min | surv last | ent mean | ent last | q-spread | gap signed | gap rel |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    for row in rows:
        line = (
            f"| {row['variant']} | {row['iteration']} | {row['variant_role']} | "
            f"{_fmt(row['breadth_ratio_last_to_first'])} | {_fmt(row['retention_last'])} | "
            f"{_fmt(row['selective_persistence_index_simple'])} | {_fmt(row['tail_survival_mean'])} | "
            f"{_fmt(row['tail_survival_min'])} | {_fmt(row['tail_survival_last'])} | "
            f"{_fmt(row['tail_entropy_mean'])} | {_fmt(row['tail_entropy_last'])} | {_fmt(row['tail_q_spread'])} | "
            f"{_fmt(row['tail_entropy_gap_signed'])} | {_fmt(row['tail_entropy_gap_relative'])}"
        )
        if include_hints:
            line += f" | {row.get('coherence_hint') or 'n/a'} |"
        else:
            line += " |"
        lines.append(line)
    return lines


def _coherence_hint(row: dict) -> str:
    if row["kept_last"] == 0 or row["tail_survival_last"] == 0 or row["tail_survival_min"] == 0:
        return "collapsed"
    if row["kept_last"] <= 1 and row["tail_entropy_last"] <= 0.01 and row["breadth_ratio_last_to_first"] <= 1.2:
        return "trivial"
    if (
        row["tail_entropy_gap"] <= 0.005
        and row["breadth_ratio_last_to_first"] <= 5.0
        and (row["tail_q_spread"] or 0.0) <= 0.08
    ):
        return "crystalline"
    if (
        row["tail_entropy_gap"] <= 0.02
        and (row["tail_q_spread"] or 0.0) <= 0.12
        and row["breadth_ratio_last_to_first"] <= 10.0
    ):
        return "rigid"
    if (
        row["tail_entropy_gap"] >= 0.03
        and (row["tail_q_spread"] or 0.0) >= 0.15
        and row["breadth_ratio_last_to_first"] >= 10.0
        and row["tail_entropy_last"] >= 0.15
    ):
        return "coherent"
    return "mixed"


def _mean_metric(items: list[dict], key: str) -> float:
    if not items:
        return 0.0
    return float(sum(item[key] for item in items) / len(items))


def _metric_stats(items: list[dict], key: str) -> dict[str, float]:
    if not items:
        return {"mean": 0.0, "min": 0.0, "last": 0.0}
    values = [float(item[key]) for item in items]
    return {
        "mean": float(sum(values) / len(values)),
        "min": float(min(values)),
        "last": float(values[-1]),
    }


def _mean_ratio(moment_items: list[dict], q_key: str) -> float | None:
    values = []
    for item in moment_items:
        value = item.get("global_ratios_to_previous_edge", {}).get(q_key)
        if value is not None:
            values.append(float(value))
    if not values:
        return None
    return float(sum(values) / len(values))


def _mean_pair_gap_stats(left: list[dict], right: list[dict], key: str) -> dict[str, float]:
    if not left or not right:
        return {"abs": 0.0, "signed": 0.0, "relative": 0.0}
    pair_count = min(len(left), len(right))
    if pair_count == 0:
        return {"abs": 0.0, "signed": 0.0, "relative": 0.0}

    abs_sum = 0.0
    signed_sum = 0.0
    relative_sum = 0.0
    for index in range(pair_count):
        left_value = float(left[index][key])
        right_value = float(right[index][key])
        delta = left_value - right_value
        abs_sum += abs(delta)
        signed_sum += delta
        relative_sum += _relative_gap(left_value, right_value)

    return {
        "abs": float(abs_sum / pair_count),
        "signed": float(signed_sum / pair_count),
        "relative": float(relative_sum / pair_count),
    }


def _relative_gap(left_value: float, right_value: float) -> float:
    denominator = abs(left_value) + abs(right_value)
    if denominator <= 0:
        return 0.0
    return float((left_value - right_value) / denominator)


def _safe_product(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left * right)


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _require_policy_series(summary: dict, policy: str, variant: str, run_dir: str) -> list[dict]:
    payload = summary.get("fiber_summary", {}).get(policy) or []
    if payload:
        return payload
    raise ValueError(
        f"Phase 1 coherence report expects '{policy}' fiber summary for variant {variant}: {run_dir}"
    )


def _require_payload(payload: list[dict], label: str, variant: str, run_dir: str) -> list[dict]:
    if payload:
        return payload
    raise ValueError(
        f"Phase 1 coherence report expects '{label}' artifacts for variant {variant}: {run_dir}"
    )
