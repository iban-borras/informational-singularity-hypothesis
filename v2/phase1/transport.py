from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

from v2.phase1.report import classify_variant_role, split_rows_by_role


def build_transport_rows(
    runs: Iterable[dict],
    *,
    main_policy: str = "prefix",
    control_policy: str = "suffix",
    tail_edges: int = 2,
    include_hints: bool = False,
) -> list[dict]:
    rows = []
    for run in runs:
        config = run["dataset"]["config"]
        scale_summary = run["summary"].get("scale_summary", [])
        kept_first = scale_summary[0]["kept_pattern_count"] if scale_summary else 0
        kept_last = scale_summary[-1]["kept_pattern_count"] if scale_summary else 0
        breadth_ratio = (kept_last / kept_first) if kept_first else 0.0

        main_fibers = _require_fibers_for_policy(run, main_policy)
        control_fibers = _require_fibers_for_policy(run, control_policy)
        main_profile = _build_transport_profile(main_fibers, tail_edges=tail_edges)
        control_profile = _build_transport_profile(control_fibers, tail_edges=tail_edges)

        main_summary = main_profile["summary"]
        control_summary = control_profile["summary"]
        transport_entropy_gap_stats = _gap_stats(
            main_summary["tail_transport_entropy"],
            control_summary["tail_transport_entropy"],
        )
        transport_active_gap_stats = _gap_stats(
            main_summary["tail_transport_active"],
            control_summary["tail_transport_active"],
        )
        transport_decay_gap_stats = _gap_stats(
            main_summary["tail_transport_decay"],
            control_summary["tail_transport_decay"],
        )
        row = {
            "variant": config["variant"],
            "variant_role": classify_variant_role(config["variant"]),
            "iteration": config.get("iteration"),
            "segment_bits": config.get("segment_bits"),
            "num_segments": config.get("num_segments"),
            "segment_offset_bits": config.get("segment_offset_bits", 0),
            "scales": list(config.get("scales", [])),
            "policies": list(config.get("policies", [])),
            "main_policy": main_policy,
            "control_policy": control_policy,
            "tail_edges": tail_edges,
            "kept_first": kept_first,
            "kept_last": kept_last,
            "breadth_ratio_last_to_first": breadth_ratio,
            "transport_steps": main_profile["steps"],
            "control_transport_steps": control_profile["steps"],
            "tail_transport_entropy": main_summary["tail_transport_entropy"],
            "tail_transport_entropy_mean": main_summary["tail_transport_entropy_mean"],
            "tail_transport_entropy_min": main_summary["tail_transport_entropy_min"],
            "tail_transport_entropy_last": main_summary["tail_transport_entropy_last"],
            "tail_transport_active": main_summary["tail_transport_active"],
            "tail_transport_active_mean": main_summary["tail_transport_active_mean"],
            "tail_transport_active_min": main_summary["tail_transport_active_min"],
            "tail_transport_active_last": main_summary["tail_transport_active_last"],
            "tail_transport_decay": main_summary["tail_transport_decay"],
            "tail_transport_decay_mean": main_summary["tail_transport_decay_mean"],
            "tail_transport_decay_min": main_summary["tail_transport_decay_min"],
            "tail_transport_decay_last": main_summary["tail_transport_decay_last"],
            "tail_transport_entropy_control": control_summary["tail_transport_entropy"],
            "tail_transport_active_control": control_summary["tail_transport_active"],
            "tail_transport_decay_control": control_summary["tail_transport_decay"],
            "tail_transport_entropy_gap": transport_entropy_gap_stats["abs"],
            "tail_transport_entropy_gap_abs": transport_entropy_gap_stats["abs"],
            "tail_transport_entropy_gap_signed": transport_entropy_gap_stats["signed"],
            "tail_transport_entropy_gap_relative": transport_entropy_gap_stats["relative"],
            "tail_transport_active_gap": transport_active_gap_stats["abs"],
            "tail_transport_active_gap_abs": transport_active_gap_stats["abs"],
            "tail_transport_active_gap_signed": transport_active_gap_stats["signed"],
            "tail_transport_active_gap_relative": transport_active_gap_stats["relative"],
            "tail_transport_decay_gap": transport_decay_gap_stats["abs"],
            "tail_transport_decay_gap_abs": transport_decay_gap_stats["abs"],
            "tail_transport_decay_gap_signed": transport_decay_gap_stats["signed"],
            "tail_transport_decay_gap_relative": transport_decay_gap_stats["relative"],
            "mean_transport_entropy": main_summary["mean_transport_entropy"],
            "mean_transport_active": main_summary["mean_transport_active"],
            "mean_transport_decay": main_summary["mean_transport_decay"],
            "summary_path": run["_summary_path"],
            "run_dir": run["_run_dir"],
            "generated_at": run["generated_at"],
        }
        row["transport_hint"] = _transport_hint(row) if include_hints else None
        rows.append(row)

    return rows


def render_console_summary(rows: list[dict], *, include_hints: bool = False) -> str:
    if not rows:
        return "No matching Phase 1 transport rows found."

    grouped = split_rows_by_role(rows)
    sections: list[str] = []

    if grouped["primary"]:
        sections.extend(
            [
                "Primary transport persistence readout (B/F/A/M/N core contrast)",
                "-" * 112,
                _render_header(include_hints=include_hints),
                *_render_rows(grouped["primary"], include_hints=include_hints),
            ]
        )

    if grouped["supporting"]:
        if sections:
            sections.append("")
        sections.extend(
            [
                "Supporting transport persistence readout",
                "-" * 112,
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
                "-" * 112,
                _render_header(include_hints=include_hints),
                *_render_rows(grouped["other"], include_hints=include_hints),
            ]
        )

    return "\n".join(sections)


def render_markdown_report(rows: list[dict], selection: dict) -> str:
    grouped = split_rows_by_role(rows)
    include_hints = selection.get("include_hints", False)
    lines = [
        "# HSI v2 Phase 1 Transport Persistence Report",
        "",
        "## Selection",
        "",
        f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Iterations in report: `{', '.join(str(value) for value in selection['iterations'])}`",
        f"- Segment layout: `{selection['num_segments']} x {selection['segment_bits']}` bits",
        f"- Segment offset: `{selection['segment_offset_bits']}` bits",
        f"- Scale ladder: `{', '.join(str(value) for value in selection['scales'])}`",
        f"- Policies in run: `{', '.join(selection['policies'])}`",
        f"- Main/control transport policies: `{selection['main_policy']} / {selection['control_policy']}`",
        f"- Tail edges used: `{selection['tail_edges']}`",
        f"- Family inferred: `{selection['family_inferred']}`",
        f"- Interpretive labels enabled: `{include_hints}`",
        f"- Variants in report: `{', '.join(row['variant'] for row in rows)}`",
        "",
        "## Primary Transport Persistence Readout",
        "",
        "This report asks whether mass that branches at one edge is transported into children that still branch at the next edge.",
        "",
    ]

    if grouped["primary"]:
        lines.extend(_render_markdown_table(grouped["primary"], include_hints=include_hints))
    else:
        lines.append("No primary rows available.")

    if grouped["supporting"]:
        lines.extend(["", "## Supporting Transport Persistence Readout", ""])
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
            "- `transport entropy` measures the expected next-edge child entropy reached by the current edge mass.",
            "- `transport active` measures the expected mass that reaches children which still branch at the next edge.",
            "- `transport decay` is the ratio between transported entropy and current-edge entropy.",
            "- Tail transport observables now expose mean, min, and last summaries to avoid smoothing bias near the top of the tower.",
            "- Policy divergence now exposes signed, relative, and absolute summaries.",
            "- Signed gaps are computed as `main - control`.",
            "- Relative gaps are computed as `(main - control) / (|main| + |control|)` when the denominator is non-zero.",
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


def build_selection(
    rows: list[dict],
    *,
    family_inferred: bool,
    tail_edges: int,
    main_policy: str,
    control_policy: str,
) -> dict:
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
        "main_policy": main_policy,
        "control_policy": control_policy,
        "family_inferred": family_inferred,
        "include_hints": bool(rows and rows[0].get("transport_hint") is not None),
    }


def _render_header(*, include_hints: bool) -> str:
    header = (
        f"{'variant':8} {'iter':>4} {'b_ratio':>9} {'a_mean':>8} {'a_min':>8} {'a_last':>8} "
        f"{'d_mean':>8} {'d_min':>8} {'d_last':>8} {'gap_s':>8}"
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
            f"{_fmt(row['tail_transport_active_mean']):>8} "
            f"{_fmt(row['tail_transport_active_min']):>8} "
            f"{_fmt(row['tail_transport_active_last']):>8} "
            f"{_fmt(row['tail_transport_decay_mean']):>8} "
            f"{_fmt(row['tail_transport_decay_min']):>8} "
            f"{_fmt(row['tail_transport_decay_last']):>8} "
            f"{_fmt(row['tail_transport_active_gap_signed']):>8}"
        )
        if include_hints:
            line += f" {(row.get('transport_hint') or 'n/a'):>12}"
        rendered.append(line)
    return rendered


def _render_markdown_table(rows: list[dict], *, include_hints: bool) -> list[str]:
    if include_hints:
        lines = [
            "| Variant | Iter | Role | breadth ratio | active mean | active min | active last | decay mean | decay min | decay last | active gap signed | active gap rel | Interpretive label |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    else:
        lines = [
            "| Variant | Iter | Role | breadth ratio | active mean | active min | active last | decay mean | decay min | decay last | active gap signed | active gap rel |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    for row in rows:
        line = (
            f"| {row['variant']} | {row['iteration']} | {row['variant_role']} | "
            f"{_fmt(row['breadth_ratio_last_to_first'])} | {_fmt(row['tail_transport_active_mean'])} | "
            f"{_fmt(row['tail_transport_active_min'])} | {_fmt(row['tail_transport_active_last'])} | "
            f"{_fmt(row['tail_transport_decay_mean'])} | {_fmt(row['tail_transport_decay_min'])} | {_fmt(row['tail_transport_decay_last'])} | "
            f"{_fmt(row['tail_transport_active_gap_signed'])} | {_fmt(row['tail_transport_active_gap_relative'])}"
        )
        if include_hints:
            line += f" | {row.get('transport_hint') or 'n/a'} |"
        else:
            line += " |"
        lines.append(line)
    return lines


def _build_transport_profile(fiber_edges: list[dict], *, tail_edges: int) -> dict:
    if not fiber_edges:
        return {
            "steps": [],
            "summary": {
                "tail_transport_entropy": 0.0,
                "tail_transport_entropy_mean": 0.0,
                "tail_transport_entropy_min": 0.0,
                "tail_transport_entropy_last": 0.0,
                "tail_transport_active": 0.0,
                "tail_transport_active_mean": 0.0,
                "tail_transport_active_min": 0.0,
                "tail_transport_active_last": 0.0,
                "tail_transport_decay": 0.0,
                "tail_transport_decay_mean": 0.0,
                "tail_transport_decay_min": 0.0,
                "tail_transport_decay_last": 0.0,
                "mean_transport_entropy": 0.0,
                "mean_transport_active": 0.0,
                "mean_transport_decay": 0.0,
            },
        }

    edge_lookup = {int(edge["target_m"]): edge for edge in fiber_edges}
    ordered_targets = sorted(edge_lookup)
    steps = []
    for target_m in ordered_targets[:-1]:
        current_edge = edge_lookup[target_m]
        next_edge = edge_lookup.get(int(current_edge["source_m"]))
        if next_edge is None:
            continue
        steps.append(_build_transport_step(current_edge, next_edge))

    tail_steps = steps[-tail_edges:] if tail_edges > 0 else steps
    tail_transport_entropy = _metric_stats(tail_steps, "transport_entropy")
    tail_transport_active = _metric_stats(tail_steps, "transport_active_mass")
    tail_transport_decay = _metric_stats(tail_steps, "transport_decay")
    return {
        "steps": steps,
        "summary": {
            "tail_transport_entropy": tail_transport_entropy["mean"],
            "tail_transport_entropy_mean": tail_transport_entropy["mean"],
            "tail_transport_entropy_min": tail_transport_entropy["min"],
            "tail_transport_entropy_last": tail_transport_entropy["last"],
            "tail_transport_active": tail_transport_active["mean"],
            "tail_transport_active_mean": tail_transport_active["mean"],
            "tail_transport_active_min": tail_transport_active["min"],
            "tail_transport_active_last": tail_transport_active["last"],
            "tail_transport_decay": tail_transport_decay["mean"],
            "tail_transport_decay_mean": tail_transport_decay["mean"],
            "tail_transport_decay_min": tail_transport_decay["min"],
            "tail_transport_decay_last": tail_transport_decay["last"],
            "mean_transport_entropy": _mean_metric(steps, "transport_entropy"),
            "mean_transport_active": _mean_metric(steps, "transport_active_mass"),
            "mean_transport_decay": _mean_metric(steps, "transport_decay"),
        },
    }


def _build_transport_step(current_edge: dict, next_edge: dict) -> dict:
    next_entropy = {
        parent["pattern"]: float(parent.get("child_entropy", 0.0))
        for parent in next_edge.get("parents", [])
    }
    next_active = {
        parent["pattern"]: 1.0 if int(parent.get("fiber_size", 0)) > 1 else 0.0
        for parent in next_edge.get("parents", [])
    }

    total_parent_mass = 0.0
    current_entropy = 0.0
    transport_entropy = 0.0
    transport_active_mass = 0.0

    for parent in current_edge.get("parents", []):
        parent_mass = float(parent.get("normalized_frequency", 0.0))
        total_parent_mass += parent_mass
        current_entropy += parent_mass * float(parent.get("child_entropy", 0.0))

        child_total_mass = float(parent.get("child_total_mass", 0.0))
        if child_total_mass <= 0:
            continue

        inherited_entropy = 0.0
        inherited_active_mass = 0.0
        for child in parent.get("children", []):
            child_mass = float(child.get("normalized_frequency", 0.0))
            if child_mass <= 0:
                continue
            child_weight = child_mass / child_total_mass
            child_pattern = child.get("pattern")
            inherited_entropy += child_weight * next_entropy.get(child_pattern, 0.0)
            inherited_active_mass += child_weight * next_active.get(child_pattern, 0.0)

        transport_entropy += parent_mass * inherited_entropy
        transport_active_mass += parent_mass * inherited_active_mass

    if total_parent_mass > 0:
        current_entropy /= total_parent_mass
        transport_entropy /= total_parent_mass
        transport_active_mass /= total_parent_mass

    transport_decay = (transport_entropy / current_entropy) if current_entropy > 0 else 0.0
    return {
        "target_m": int(current_edge["target_m"]),
        "source_m": int(current_edge["source_m"]),
        "current_entropy": current_entropy,
        "transport_entropy": transport_entropy,
        "transport_active_mass": transport_active_mass,
        "transport_decay": transport_decay,
    }


def _transport_hint(row: dict) -> str:
    if (
        row["kept_last"] == 1
        and row["tail_transport_active_last"] <= 0.01
        and row["breadth_ratio_last_to_first"] <= 1.2
    ):
        return "trivial"
    if row["kept_last"] == 0 or row["tail_transport_entropy_last"] == 0 or row["tail_transport_active_last"] == 0:
        return "collapsed"
    if (
        row["breadth_ratio_last_to_first"] <= 5.0
        and row["tail_transport_active_gap"] <= 0.01
        and row["tail_transport_decay_min"] >= 0.95
    ):
        return "crystalline"
    if (
        row["breadth_ratio_last_to_first"] >= 10.0
        and row["tail_transport_entropy_last"] >= 0.35
        and row["tail_transport_active_last"] >= 0.30
        and row["tail_transport_active_gap"] >= 0.02
    ):
        return "coherent"
    if row["tail_transport_entropy_last"] >= 0.15 and row["tail_transport_decay_min"] < 0.95:
        return "rigid"
    return "mixed"


def _require_fibers_for_policy(run: dict, policy: str) -> list[dict]:
    config = run["dataset"]["config"]
    if policy not in set(config.get("policies", [])):
        raise ValueError(
            f"Phase 1 transport report expects policy '{policy}' in run for variant {config.get('variant')}: {run['_run_dir']}"
        )
    path = Path(run["_run_dir"]) / "fibers" / f"fibers_{policy}.json"
    if not path.exists():
        raise ValueError(
            f"Phase 1 transport report expects fibers_{policy}.json for variant {config.get('variant')}: {run['_run_dir']}"
        )
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        raise ValueError(
            f"Phase 1 transport report could not read fibers_{policy}.json for variant {config.get('variant')}: {run['_run_dir']}"
        )
    if payload:
        return payload
    raise ValueError(
        f"Phase 1 transport report found empty fibers_{policy}.json for variant {config.get('variant')}: {run['_run_dir']}"
    )


def _mean_metric(items: list[dict], key: str) -> float:
    if not items:
        return 0.0
    return float(sum(float(item[key]) for item in items) / len(items))


def _metric_stats(items: list[dict], key: str) -> dict[str, float]:
    if not items:
        return {"mean": 0.0, "min": 0.0, "last": 0.0}
    values = [float(item[key]) for item in items]
    return {
        "mean": float(sum(values) / len(values)),
        "min": float(min(values)),
        "last": float(values[-1]),
    }


def _gap_stats(main_value: float, control_value: float) -> dict[str, float]:
    signed = float(main_value - control_value)
    denominator = abs(main_value) + abs(control_value)
    relative = float(signed / denominator) if denominator > 0 else 0.0
    return {"abs": abs(signed), "signed": signed, "relative": relative}


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"
