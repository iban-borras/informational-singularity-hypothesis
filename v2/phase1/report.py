from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable


PRIMARY_PHASE1_VARIANTS = {"B", "F", "A", "M", "N"}
SUPPORTING_PHASE1_VARIANTS = {"E", "I", "J", "K", "L", "D", "G", "H"}
PREFERRED_VARIANT_ORDER = ["B", "F", "A", "M", "N", "E", "I", "J", "K", "L", "D", "G", "H"]


def discover_phase1_runs(phase1_dir: Path) -> list[dict]:
    runs = []
    for dataset_path in sorted(phase1_dir.glob("*/dataset.json")):
        try:
            with open(dataset_path, "r", encoding="utf-8") as handle:
                dataset = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue

        if dataset.get("stage") != "phase1_tower":
            continue

        run_dir = dataset_path.parent
        summary_path = run_dir / "phase1_summary.json"
        if not summary_path.exists():
            continue

        try:
            with open(summary_path, "r", encoding="utf-8") as handle:
                summary = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue

        moments_prefix = _load_optional_json(run_dir / "moments" / "moments_prefix.json")
        moments_suffix = _load_optional_json(run_dir / "moments" / "moments_suffix.json")

        generated_at = dataset.get("generated_at") or _timestamp_from_mtime(dataset_path)
        runs.append(
            {
                "dataset": dataset,
                "summary": summary,
                "moments_prefix": moments_prefix,
                "moments_suffix": moments_suffix,
                "_dataset_path": str(dataset_path.resolve()),
                "_summary_path": str(summary_path.resolve()),
                "_run_dir": str(run_dir.resolve()),
                "_generated_at_dt": _parse_timestamp(generated_at),
                "generated_at": generated_at,
            }
        )

    return runs


def infer_family_from_latest_run(runs: list[dict]) -> dict | None:
    if not runs:
        return None

    latest_run = max(runs, key=lambda item: item["_generated_at_dt"])
    config = latest_run["dataset"]["config"]
    return {
        "segment_bits": config.get("segment_bits"),
        "num_segments": config.get("num_segments"),
        "segment_offset_bits": config.get("segment_offset_bits", 0),
        "scales": tuple(config.get("scales", [])),
        "policies": tuple(config.get("policies", [])),
        "min_count_floor": config.get("min_count_floor"),
        "min_count_rate": config.get("min_count_rate"),
        "min_segment_support_effective": config.get("min_segment_support_effective"),
        "cv_max": config.get("cv_max"),
        "sampling_policy": config.get("sampling_policy"),
        "pattern_space_mode": config.get("pattern_space_mode"),
    }


def filter_runs(
    runs: list[dict],
    *,
    variants: list[str] | None = None,
    iteration: int | None = None,
    segment_bits: int | None = None,
    num_segments: int | None = None,
    segment_offset_bits: int | None = None,
    scales: list[int] | None = None,
    policies: list[str] | None = None,
    family: dict | None = None,
) -> list[dict]:
    requested_variants = {item.upper() for item in variants} if variants else None
    filtered = []
    for run in runs:
        config = run["dataset"]["config"]
        if requested_variants and str(config.get("variant", "")).upper() not in requested_variants:
            continue
        if iteration is not None and config.get("iteration") != iteration:
            continue
        if segment_bits is not None and config.get("segment_bits") != segment_bits:
            continue
        if num_segments is not None and config.get("num_segments") != num_segments:
            continue
        if segment_offset_bits is not None and config.get("segment_offset_bits", 0) != segment_offset_bits:
            continue
        if scales is not None and list(config.get("scales", [])) != list(scales):
            continue
        if policies is not None and list(config.get("policies", [])) != list(policies):
            continue

        if family is not None:
            if config.get("segment_bits") != family["segment_bits"]:
                continue
            if config.get("num_segments") != family["num_segments"]:
                continue
            if config.get("segment_offset_bits", 0) != family.get("segment_offset_bits", 0):
                continue
            if tuple(config.get("scales", [])) != tuple(family["scales"]):
                continue
            if tuple(config.get("policies", [])) != tuple(family["policies"]):
                continue
            if config.get("min_count_floor") != family["min_count_floor"]:
                continue
            if config.get("min_count_rate") != family["min_count_rate"]:
                continue
            if config.get("min_segment_support_effective") != family["min_segment_support_effective"]:
                continue
            if config.get("cv_max") != family["cv_max"]:
                continue
            if config.get("sampling_policy") != family["sampling_policy"]:
                continue
            if config.get("pattern_space_mode") != family["pattern_space_mode"]:
                continue

        filtered.append(run)

    return filtered


def select_latest_per_variant(runs: list[dict], variant_order: list[str] | None = None) -> list[dict]:
    latest: dict[str, dict] = {}
    for run in runs:
        variant = run["dataset"]["config"]["variant"]
        current = latest.get(variant)
        if current is None or run["_generated_at_dt"] > current["_generated_at_dt"]:
            latest[variant] = run

    ordered_variants = variant_order or sorted(latest)
    if variant_order:
        variant_lookup = {variant.upper(): variant for variant in latest}
        selected = [latest[variant_lookup[item.upper()]] for item in ordered_variants if item.upper() in variant_lookup]
    else:
        selected = [latest[variant] for variant in ordered_variants if variant in latest]
    if variant_order:
        return selected
    role_rank = {"primary": 0, "supporting": 1, "other": 2}
    variant_rank = {variant: index for index, variant in enumerate(PREFERRED_VARIANT_ORDER)}
    return sorted(
        selected,
        key=lambda item: (
            role_rank.get(classify_variant_role(item["dataset"]["config"]["variant"]), 9),
            variant_rank.get(item["dataset"]["config"]["variant"], 999),
            item["dataset"]["config"]["variant"],
        ),
    )


def build_report_rows(runs: Iterable[dict], *, include_hints: bool = False) -> list[dict]:
    rows = []
    for run in runs:
        config = run["dataset"]["config"]
        summary = run["summary"]
        scale_summary = summary.get("scale_summary", [])
        fiber_summary = summary.get("fiber_summary", {})
        prefix_summary = _require_policy_series(fiber_summary, "prefix", config["variant"], run["_run_dir"])
        suffix_summary = _require_policy_series(fiber_summary, "suffix", config["variant"], run["_run_dir"])
        prefix_moments = _require_payload(
            run.get("moments_prefix") or [],
            "moments_prefix",
            config["variant"],
            run["_run_dir"],
        )
        suffix_moments = _require_payload(
            run.get("moments_suffix") or [],
            "moments_suffix",
            config["variant"],
            run["_run_dir"],
        )

        scale_counts = {str(item["m"]): item["kept_pattern_count"] for item in scale_summary}
        last_scale = scale_summary[-1] if scale_summary else None
        first_scale = scale_summary[0] if scale_summary else None
        mid_scale = scale_summary[len(scale_summary) // 2] if scale_summary else None

        prefix_last = prefix_summary[-1] if prefix_summary else _empty_fiber_row()
        suffix_last = suffix_summary[-1] if suffix_summary else _empty_fiber_row()
        prefix_q_last = _extract_last_ratio(prefix_moments)
        suffix_q_last = _extract_last_ratio(suffix_moments)

        entropy_gap_stats = _mean_pair_gap_stats(prefix_summary, suffix_summary, "mean_child_entropy")
        singleton_gap_stats = _mean_pair_gap_stats(prefix_summary, suffix_summary, "singleton_fraction")

        row = {
            "variant": config["variant"],
            "variant_role": classify_variant_role(config["variant"]),
            "iteration": config.get("iteration"),
            "segment_bits": config.get("segment_bits"),
            "num_segments": config.get("num_segments"),
            "segment_offset_bits": config.get("segment_offset_bits", 0),
            "scales": list(config.get("scales", [])),
            "policies": list(config.get("policies", [])),
            "scale_counts": scale_counts,
            "kept_first": first_scale["kept_pattern_count"] if first_scale else 0,
            "kept_mid": mid_scale["kept_pattern_count"] if mid_scale else 0,
            "kept_last": last_scale["kept_pattern_count"] if last_scale else 0,
            "candidate_last": last_scale["candidate_pattern_count"] if last_scale else 0,
            "breadth_ratio_last_to_first": _safe_ratio(
                float(last_scale["kept_pattern_count"]) if last_scale else 0.0,
                float(first_scale["kept_pattern_count"]) if first_scale else 0.0,
            ),
            "retention_last": _safe_ratio(
                float(last_scale["kept_pattern_count"]) if last_scale else 0.0,
                float(last_scale["candidate_pattern_count"]) if last_scale else 0.0,
            ),
            "prefix_survival_last": prefix_last["survival_fraction"],
            "suffix_survival_last": suffix_last["survival_fraction"],
            "prefix_mean_fiber_last": prefix_last["mean_fiber_size"],
            "suffix_mean_fiber_last": suffix_last["mean_fiber_size"],
            "prefix_singleton_last": prefix_last["singleton_fraction"],
            "suffix_singleton_last": suffix_last["singleton_fraction"],
            "prefix_child_entropy_last": prefix_last["mean_child_entropy"],
            "suffix_child_entropy_last": suffix_last["mean_child_entropy"],
            "prefix_mean_survival": _mean_metric(prefix_summary, "survival_fraction"),
            "suffix_mean_survival": _mean_metric(suffix_summary, "survival_fraction"),
            "prefix_mean_child_entropy": _mean_metric(prefix_summary, "mean_child_entropy"),
            "suffix_mean_child_entropy": _mean_metric(suffix_summary, "mean_child_entropy"),
            "prefix_q1_last_ratio": prefix_q_last.get("1"),
            "prefix_q2_last_ratio": prefix_q_last.get("2"),
            "prefix_q3_last_ratio": prefix_q_last.get("3"),
            "suffix_q1_last_ratio": suffix_q_last.get("1"),
            "suffix_q2_last_ratio": suffix_q_last.get("2"),
            "suffix_q3_last_ratio": suffix_q_last.get("3"),
            "prefix_suffix_entropy_gap_mean": entropy_gap_stats["abs"],
            "prefix_suffix_entropy_gap_mean_abs": entropy_gap_stats["abs"],
            "prefix_suffix_entropy_gap_mean_signed": entropy_gap_stats["signed"],
            "prefix_suffix_entropy_gap_mean_relative": entropy_gap_stats["relative"],
            "prefix_suffix_singleton_gap_mean": singleton_gap_stats["abs"],
            "prefix_suffix_singleton_gap_mean_abs": singleton_gap_stats["abs"],
            "prefix_suffix_singleton_gap_mean_signed": singleton_gap_stats["signed"],
            "prefix_suffix_singleton_gap_mean_relative": singleton_gap_stats["relative"],
            "run_dir": run["_run_dir"],
            "dataset_path": run["_dataset_path"],
            "summary_path": run["_summary_path"],
            "generated_at": run["generated_at"],
        }
        row["selective_persistence_index_simple"] = _safe_product(
            row["retention_last"],
            row["prefix_survival_last"],
        )
        row["status_hint"] = _status_hint(row) if include_hints else None
        rows.append(row)

    return rows


def classify_variant_role(variant: str) -> str:
    if variant in PRIMARY_PHASE1_VARIANTS:
        return "primary"
    if variant in SUPPORTING_PHASE1_VARIANTS:
        return "supporting"
    return "other"


def split_rows_by_role(rows: list[dict]) -> dict[str, list[dict]]:
    grouped = {"primary": [], "supporting": [], "other": []}
    for row in rows:
        grouped.setdefault(row["variant_role"], []).append(row)
    return grouped


def render_console_summary(rows: list[dict], *, include_hints: bool = False) -> str:
    if not rows:
        return "No matching Phase 1 runs found."

    grouped = split_rows_by_role(rows)
    sections: list[str] = []

    if grouped["primary"]:
        sections.extend(
            [
                "Primary Phase 1 readout (B/F/A/M/N core contrast)",
                "-" * 104,
                _render_scale_header(include_hints=include_hints),
                *_render_scale_rows(grouped["primary"], include_hints=include_hints),
                "",
                "Primary richness metrics",
                "-" * 104,
                _render_richness_header(),
                *_render_richness_rows(grouped["primary"]),
            ]
        )

    if grouped["supporting"]:
        if sections:
            sections.append("")
        sections.extend(
            [
                "Supporting variants (robustness + weak controls)",
                "-" * 104,
                _render_scale_header(include_hints=include_hints),
                *_render_scale_rows(grouped["supporting"], include_hints=include_hints),
                "",
                "Supporting richness metrics",
                "-" * 104,
                _render_richness_header(),
                *_render_richness_rows(grouped["supporting"]),
            ]
        )

    if grouped["other"]:
        if sections:
            sections.append("")
        sections.extend(
            [
                "Other variants",
                "-" * 104,
                _render_scale_header(include_hints=include_hints),
                *_render_scale_rows(grouped["other"], include_hints=include_hints),
            ]
        )

    return "\n".join(sections)


def render_markdown_report(rows: list[dict], selection: dict) -> str:
    grouped = split_rows_by_role(rows)
    include_hints = selection.get("include_hints", False)
    lines = [
        "# HSI v2 Phase 1 Report",
        "",
        "## Selection",
        "",
        f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Iterations in report: `{', '.join(str(value) for value in selection['iterations'])}`",
        f"- Segment layout: `{selection['num_segments']} x {selection['segment_bits']}` bits",
        f"- Segment offset: `{selection['segment_offset_bits']}` bits",
        f"- Scale ladder: `{', '.join(str(value) for value in selection['scales'])}`",
        f"- Policies: `{', '.join(selection['policies'])}`",
        f"- Family inferred: `{selection['family_inferred']}`",
        f"- Interpretive labels enabled: `{include_hints}`",
        f"- Variants in report: `{', '.join(row['variant'] for row in rows)}`" if rows else "- Variants in report: none",
        "",
        "## Primary Phase 1 Readout",
        "",
        "This section drives the main comparative judgment for the first Phase 1 gate.",
        "",
    ]

    if grouped["primary"]:
        lines.extend(_render_markdown_scale_table(grouped["primary"], include_hints=include_hints))
        lines.extend(["", "### Primary Richness Metrics", ""])
        lines.extend(_render_markdown_richness_table(grouped["primary"]))
    else:
        lines.append("No primary rows available.")

    if grouped["supporting"]:
        lines.extend(
            [
                "",
                "## Supporting Variants",
                "",
                "These rows provide robustness checks and weak-control calibration.",
                "",
            ]
        )
        lines.extend(_render_markdown_scale_table(grouped["supporting"], include_hints=include_hints))
        lines.extend(["", "### Supporting Richness Metrics", ""])
        lines.extend(_render_markdown_richness_table(grouped["supporting"]))

    if grouped["other"]:
        lines.extend(["", "## Other Variants", ""])
        lines.extend(_render_markdown_scale_table(grouped["other"], include_hints=include_hints))

    lines.extend(["", "## Run Paths", ""])
    for row in rows:
        lines.append(f"- `{row['variant']}`: `{row['summary_path']}`")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `SPI_simple` is the minimal selective-persistence index defined as `retention@last * prefix survival@last`.",
            "- `prefix/suffix` divergence now exposes signed, relative, and absolute gap summaries.",
            "- Signed gaps are computed as `prefix - suffix`.",
            "- Relative gaps are computed as `(prefix - suffix) / (|prefix| + |suffix|)` when the denominator is non-zero.",
            "- Absolute gaps are still preserved in the JSON payload for magnitude-only inspection.",
        ]
    )
    if include_hints:
        lines.extend(
            [
                "- Interpretive labels are heuristic reading aids derived from the current batch, not canonical outputs of Phase 1.",
                "- Treat `rich / rigid / collapsed / trivial` as descriptive synthesis, not as mathematically final classes.",
            ]
        )

    return "\n".join(lines)


def _render_scale_header(*, include_hints: bool) -> str:
    header = (
        f"{'variant':8} {'iter':>4} {'k@8':>8} {'k@mid':>8} {'k@last':>8} "
        f"{'ret_last':>9} {'spi':>9} {'b_ratio':>9}"
    )
    if include_hints:
        header += f" {'hint':>10}"
    return header


def _render_scale_rows(rows: list[dict], *, include_hints: bool) -> list[str]:
    rendered = []
    for row in rows:
        line = (
            f"{row['variant']:8} "
            f"{row['iteration']:>4} "
            f"{row['kept_first']:>8} "
            f"{row['kept_mid']:>8} "
            f"{row['kept_last']:>8} "
            f"{_fmt(row['retention_last']):>9} "
            f"{_fmt(row['selective_persistence_index_simple']):>9} "
            f"{_fmt(row['breadth_ratio_last_to_first']):>9}"
        )
        if include_hints:
            line += f" {(row.get('status_hint') or 'n/a'):>10}"
        rendered.append(line)
    return rendered


def _render_richness_header() -> str:
    return (
        f"{'variant':8} {'p_surv':>8} {'s_surv':>8} {'p_ent':>8} {'s_ent':>8} "
        f"{'p_q1':>8} {'p_q2':>8} {'p_q3':>8} {'gap_s':>8} {'gap_r':>8}"
    )


def _render_richness_rows(rows: list[dict]) -> list[str]:
    return [
        (
            f"{row['variant']:8} "
            f"{_fmt(row['prefix_survival_last']):>8} "
            f"{_fmt(row['suffix_survival_last']):>8} "
            f"{_fmt(row['prefix_child_entropy_last']):>8} "
            f"{_fmt(row['suffix_child_entropy_last']):>8} "
            f"{_fmt(row['prefix_q1_last_ratio']):>8} "
            f"{_fmt(row['prefix_q2_last_ratio']):>8} "
            f"{_fmt(row['prefix_q3_last_ratio']):>8} "
            f"{_fmt(row['prefix_suffix_entropy_gap_mean_signed']):>8} "
            f"{_fmt(row['prefix_suffix_entropy_gap_mean_relative']):>8}"
        )
        for row in rows
    ]


def _render_markdown_scale_table(rows: list[dict], *, include_hints: bool) -> list[str]:
    if include_hints:
        lines = [
            "| Variant | Iter | Role | kept@first | kept@mid | kept@last | retention@last | SPI_simple | breadth ratio | Interpretive label |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    else:
        lines = [
            "| Variant | Iter | Role | kept@first | kept@mid | kept@last | retention@last | SPI_simple | breadth ratio |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    for row in rows:
        base = (
            f"| {row['variant']} | {row['iteration']} | {row['variant_role']} | {row['kept_first']} | "
            f"{row['kept_mid']} | {row['kept_last']} | {_fmt(row['retention_last'])} | "
            f"{_fmt(row['selective_persistence_index_simple'])} | {_fmt(row['breadth_ratio_last_to_first'])}"
        )
        if include_hints:
            base += f" | {row.get('status_hint') or 'n/a'} |"
        else:
            base += " |"
        lines.append(base)
    return lines


def _render_markdown_richness_table(rows: list[dict]) -> list[str]:
    lines = [
        "| Variant | prefix survival | suffix survival | prefix entropy | suffix entropy | prefix q1 | prefix q2 | prefix q3 | entropy gap signed | entropy gap rel |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {_fmt(row['prefix_survival_last'])} | {_fmt(row['suffix_survival_last'])} | "
            f"{_fmt(row['prefix_child_entropy_last'])} | {_fmt(row['suffix_child_entropy_last'])} | "
            f"{_fmt(row['prefix_q1_last_ratio'])} | {_fmt(row['prefix_q2_last_ratio'])} | "
            f"{_fmt(row['prefix_q3_last_ratio'])} | {_fmt(row['prefix_suffix_entropy_gap_mean_signed'])} | "
            f"{_fmt(row['prefix_suffix_entropy_gap_mean_relative'])} |"
        )
    return lines


def _status_hint(row: dict) -> str:
    kept_last = row["kept_last"]
    prefix_survival = row["prefix_survival_last"] or 0.0
    prefix_entropy = row["prefix_child_entropy_last"] or 0.0
    prefix_q1 = row["prefix_q1_last_ratio"] or 0.0
    gap = row["prefix_suffix_entropy_gap_mean"] or 0.0

    if kept_last == 0 or prefix_survival == 0:
        return "collapsed"
    if kept_last <= 1 and prefix_q1 <= 1.01:
        return "trivial"
    if kept_last >= 100 and prefix_survival >= 0.95 and prefix_entropy >= 0.15 and prefix_q1 >= 1.25:
        return "rich"
    if kept_last >= 20 and prefix_survival >= 0.95 and prefix_q1 >= 1.10:
        return "rigid"
    if gap >= 0.05 and prefix_entropy >= 0.15:
        return "rich"
    return "mixed"


def _mean_metric(items: list[dict], key: str) -> float:
    if not items:
        return 0.0
    return float(sum(item[key] for item in items) / len(items))


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
        signed_gap = left_value - right_value
        abs_sum += abs(signed_gap)
        signed_sum += signed_gap
        relative_sum += _relative_gap(left_value, right_value)

    return {
        "abs": float(abs_sum / pair_count),
        "signed": float(signed_sum / pair_count),
        "relative": float(relative_sum / pair_count),
    }


def _relative_gap(left_value: float, right_value: float) -> float:
    denominator = abs(left_value) + abs(right_value)
    if denominator == 0.0:
        return 0.0
    return float((left_value - right_value) / denominator)


def _extract_last_ratio(moment_payload: list[dict]) -> dict[str, float | None]:
    if not moment_payload:
        return {"1": None, "2": None, "3": None}
    ratios = moment_payload[-1].get("global_ratios_to_previous_edge", {})
    return {
        "1": ratios.get("1"),
        "2": ratios.get("2"),
        "3": ratios.get("3"),
    }


def _empty_fiber_row() -> dict:
    return {
        "survival_fraction": 0.0,
        "mean_fiber_size": 0.0,
        "singleton_fraction": 0.0,
        "mean_child_entropy": 0.0,
    }


def _safe_ratio(current: float, previous: float) -> float | None:
    if previous == 0:
        return None
    return float(current / previous)


def _safe_product(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left * right)


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _load_optional_json(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return []


def _timestamp_from_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")


def _parse_timestamp(raw: str) -> datetime:
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return datetime.now()


def _require_policy_series(fiber_summary: dict, policy: str, variant: str, run_dir: str) -> list[dict]:
    payload = fiber_summary.get(policy) or []
    if payload:
        return payload
    raise ValueError(
        f"Phase 1 report expects '{policy}' fiber summary for variant {variant}: {run_dir}"
    )


def _require_payload(payload: list[dict], label: str, variant: str, run_dir: str) -> list[dict]:
    if payload:
        return payload
    raise ValueError(
        f"Phase 1 report expects '{label}' artifacts for variant {variant}: {run_dir}"
    )
