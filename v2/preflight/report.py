from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable


PRIMARY_HSI_VARIANTS = {"B", "D", "E", "F", "G", "H", "I"}
BASELINE_CONTROL_VARIANTS = {"A", "J"}
CONTRAST_CONTROL_VARIANTS = {"K", "L", "M", "N"}


def discover_preflight_runs(preflight_dir: Path) -> list[dict]:
    runs = []
    for result_path in sorted(preflight_dir.glob("*/factor_complexity.json")):
        try:
            with open(result_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue

        if payload.get("stage") != "preflight_factor_complexity":
            continue

        generated_at = payload.get("generated_at") or _timestamp_from_mtime(result_path)
        payload["_result_path"] = str(result_path.resolve())
        payload["_run_dir"] = str(result_path.parent.resolve())
        payload["_generated_at_dt"] = _parse_timestamp(generated_at)
        runs.append(payload)

    return runs


def infer_family_from_latest_run(runs: list[dict]) -> dict | None:
    if not runs:
        return None

    latest_run = max(runs, key=lambda item: item["_generated_at_dt"])
    config = latest_run["config"]
    return {
        "iteration": config.get("iteration"),
        "segment_bits": config.get("segment_bits"),
        "num_segments": config.get("num_segments"),
        "m_min": config.get("m_min"),
        "m_max": config.get("m_max"),
        "nulls": tuple(config.get("nulls", [])),
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
    m_min: int | None = None,
    m_max: int | None = None,
    nulls: list[str] | None = None,
    family: dict | None = None,
) -> list[dict]:
    filtered = []

    for run in runs:
        config = run["config"]
        if variants and config.get("variant") not in variants:
            continue
        if iteration is not None and config.get("iteration") != iteration:
            continue
        if segment_bits is not None and config.get("segment_bits") != segment_bits:
            continue
        if num_segments is not None and config.get("num_segments") != num_segments:
            continue
        if m_min is not None and config.get("m_min") != m_min:
            continue
        if m_max is not None and config.get("m_max") != m_max:
            continue
        if nulls is not None and list(config.get("nulls", [])) != list(nulls):
            continue

        if family is not None:
            if config.get("iteration") != family["iteration"]:
                continue
            if config.get("segment_bits") != family["segment_bits"]:
                continue
            if config.get("num_segments") != family["num_segments"]:
                continue
            if config.get("m_min") != family["m_min"]:
                continue
            if config.get("m_max") != family["m_max"]:
                continue
            if tuple(config.get("nulls", [])) != tuple(family["nulls"]):
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
        variant = run["config"]["variant"]
        current = latest.get(variant)
        if current is None or run["_generated_at_dt"] > current["_generated_at_dt"]:
            latest[variant] = run

    ordered_variants = variant_order or sorted(latest)
    selected = [latest[variant] for variant in ordered_variants if variant in latest]
    if not variant_order:
        return sorted(selected, key=lambda item: (item["config"]["variant"], item["_generated_at_dt"]))
    return selected


def build_report_rows(runs: Iterable[dict]) -> list[dict]:
    rows = []
    for run in runs:
        config = run["config"]
        results = run["results"]
        observed = results["observed"]["aggregate"]
        shuffled = results.get("shuffled", {}).get("aggregate")
        same_density = results.get("same-density", {}).get("aggregate")
        markov1 = results.get("markov1", {}).get("aggregate")

        random_baseline = _mean_aggregate([agg for agg in (shuffled, same_density) if agg is not None])

        row = {
            "variant": config["variant"],
            "variant_role": classify_variant_role(config["variant"]),
            "iteration": config["iteration"],
            "segment_bits": config["segment_bits"],
            "num_segments": config["num_segments"],
            "m_range": f"{config['m_min']}-{config['m_max']}",
            "observed_h_eff": observed["mean_tail_h_eff"],
            "observed_right": observed["mean_tail_right_branching"],
            "observed_left": observed["mean_tail_left_branching"],
            "observed_rho": observed["mean_tail_rho"],
            "markov1_h_eff": markov1["mean_tail_h_eff"] if markov1 else None,
            "markov1_right": markov1["mean_tail_right_branching"] if markov1 else None,
            "markov1_rho": markov1["mean_tail_rho"] if markov1 else None,
            "random_h_eff": random_baseline["mean_tail_h_eff"] if random_baseline else None,
            "random_right": random_baseline["mean_tail_right_branching"] if random_baseline else None,
            "random_rho": random_baseline["mean_tail_rho"] if random_baseline else None,
            "result_path": run["_result_path"],
            "run_dir": run["_run_dir"],
            "generated_at": run["generated_at"],
        }

        row["delta_h_eff_vs_markov1"] = _delta(row["observed_h_eff"], row["markov1_h_eff"])
        row["delta_right_vs_markov1"] = _delta(row["observed_right"], row["markov1_right"])
        row["delta_rho_vs_markov1"] = _delta(row["observed_rho"], row["markov1_rho"])
        row["delta_h_eff_vs_random"] = _delta(row["observed_h_eff"], row["random_h_eff"])
        row["delta_right_vs_random"] = _delta(row["observed_right"], row["random_right"])
        row["delta_rho_vs_random"] = _delta(row["observed_rho"], row["random_rho"])
        row["status_hint"] = _status_hint(row)

        rows.append(row)

    return rows


def classify_variant_role(variant: str) -> str:
    if variant in PRIMARY_HSI_VARIANTS:
        return "hsi"
    if variant in BASELINE_CONTROL_VARIANTS:
        return "baseline_control"
    if variant in CONTRAST_CONTROL_VARIANTS:
        return "contrast_control"
    return "other"


def split_rows_by_role(rows: list[dict]) -> dict[str, list[dict]]:
    grouped = {
        "main": [],
        "contrast": [],
        "other": [],
    }
    for row in rows:
        role = row.get("variant_role")
        if role in {"hsi", "baseline_control"}:
            grouped["main"].append(row)
        elif role == "contrast_control":
            grouped["contrast"].append(row)
        else:
            grouped["other"].append(row)
    return grouped


def render_console_summary(rows: list[dict]) -> str:
    if not rows:
        return "No matching preflight runs found."

    grouped = split_rows_by_role(rows)
    sections: list[str] = []

    if grouped["main"]:
        sections.extend(
            [
                "Main preflight readout (HSI + baseline controls)",
                "-" * 86,
                _render_observed_header(),
                *_render_observed_rows(grouped["main"]),
                "",
                "Main deltas vs strong nulls (observed - control)",
                "-" * 86,
                _render_delta_header(),
                *_render_delta_rows(grouped["main"]),
            ]
        )

    if grouped["contrast"]:
        if sections:
            sections.append("")
        sections.extend(
            [
                "Contrast controls (structural comparison, not null falsation)",
                "-" * 86,
                _render_observed_header(),
                *_render_observed_rows(grouped["contrast"]),
                "",
                "Contrast deltas vs strong nulls (observed - control)",
                "-" * 86,
                _render_delta_header(),
                *_render_delta_rows(grouped["contrast"]),
            ]
        )

    if grouped["other"]:
        if sections:
            sections.append("")
        sections.extend(
            [
                "Other variants",
                "-" * 86,
                _render_observed_header(),
                *_render_observed_rows(grouped["other"]),
            ]
        )

    return "\n".join(sections)


def render_markdown_report(rows: list[dict], selection: dict) -> str:
    grouped = split_rows_by_role(rows)

    lines = [
        "# HSI v2 Preflight Report",
        "",
        "## Selection",
        "",
        f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Iteration: `{selection['iteration']}`",
        f"- Segment layout: `{selection['num_segments']} x {selection['segment_bits']}` bits",
        f"- m range: `{selection['m_min']}..{selection['m_max']}`",
        f"- Nulls: `{', '.join(selection['nulls'])}`",
        f"- Variants in report: `{', '.join(row['variant'] for row in rows)}`" if rows else "- Variants in report: none",
        "",
        "## Main Preflight Readout",
        "",
        "These rows drive the primary go/no-go judgment for the HSI v2 preflight.",
        "",
    ]

    if grouped["main"]:
        lines.extend(_render_markdown_table(grouped["main"]))
    else:
        lines.append("No main rows available.")

    lines.extend(
        [
            "",
            "## Main Deltas vs Strong Nulls",
            "",
            "Negative deltas mean the observed run is more constrained than the control on that tail metric.",
            "",
        ]
    )
    if grouped["main"]:
        lines.extend(_render_markdown_delta_table(grouped["main"]))
    else:
        lines.append("No main deltas available.")

    if grouped["contrast"]:
        lines.extend(
            [
                "",
                "## Contrast Controls",
                "",
                "These rows are useful for structural discrimination, not for null falsation.",
                "",
            ]
        )
        lines.extend(_render_markdown_table(grouped["contrast"]))
        lines.extend(
            [
                "",
                "## Contrast Deltas vs Strong Nulls",
                "",
                "These comparisons help distinguish HSI-like projective structure from other non-random structures.",
                "",
            ]
        )
        lines.extend(_render_markdown_delta_table(grouped["contrast"]))

    if grouped["other"]:
        lines.extend(["", "## Other Variants", ""])
        lines.extend(_render_markdown_table(grouped["other"]))

    lines.extend(["", "## Run Paths", ""])
    for row in rows:
        lines.append(f"- `{row['variant']}`: `{row['result_path']}`")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This report summarizes the preflight tail metrics only.",
            "- It does not replace inspection of the full per-scale curves.",
            "- The `hint` column is only a compact heuristic for separation, not a scientific conclusion.",
            "- `K`, `L`, `M`, and `N` are contrast controls; they should not be read as null baselines.",
        ]
    )

    return "\n".join(lines)


def _render_observed_header() -> str:
    return f"{'variant':8} {'iter':>4} {'h_eff':>10} {'right':>10} {'left':>10} {'rho':>10} {'hint':>12}"


def _render_observed_rows(rows: list[dict]) -> list[str]:
    return [
        (
            f"{row['variant']:8} "
            f"{row['iteration']:>4} "
            f"{row['observed_h_eff']:>10.4f} "
            f"{row['observed_right']:>10.4f} "
            f"{row['observed_left']:>10.4f} "
            f"{row['observed_rho']:>10.4f} "
            f"{row['status_hint']:>12}"
        )
        for row in rows
    ]


def _render_delta_header() -> str:
    return f"{'variant':8} {'d_h_rand':>10} {'d_h_mk1':>10} {'d_rho_rand':>12} {'d_rho_mk1':>11} {'d_right_rand':>13} {'d_right_mk1':>12}"


def _render_delta_rows(rows: list[dict]) -> list[str]:
    return [
        (
            f"{row['variant']:8} "
            f"{_fmt(row['delta_h_eff_vs_random']):>10} "
            f"{_fmt(row['delta_h_eff_vs_markov1']):>10} "
            f"{_fmt(row['delta_rho_vs_random']):>12} "
            f"{_fmt(row['delta_rho_vs_markov1']):>11} "
            f"{_fmt(row['delta_right_vs_random']):>13} "
            f"{_fmt(row['delta_right_vs_markov1']):>12}"
        )
        for row in rows
    ]


def _render_markdown_table(rows: list[dict]) -> list[str]:
    lines = [
        "| Variant | Iter | Role | h_eff | Right | Left | rho | Hint |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['iteration']} | {row['variant_role']} | {row['observed_h_eff']:.4f} | "
            f"{row['observed_right']:.4f} | {row['observed_left']:.4f} | {row['observed_rho']:.4f} | {row['status_hint']} |"
        )
    return lines


def _render_markdown_delta_table(rows: list[dict]) -> list[str]:
    lines = [
        "| Variant | d_h_eff vs random | d_h_eff vs Markov-1 | d_rho vs random | d_rho vs Markov-1 | d_right vs random | d_right vs Markov-1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {_fmt(row['delta_h_eff_vs_random'])} | {_fmt(row['delta_h_eff_vs_markov1'])} | "
            f"{_fmt(row['delta_rho_vs_random'])} | {_fmt(row['delta_rho_vs_markov1'])} | "
            f"{_fmt(row['delta_right_vs_random'])} | {_fmt(row['delta_right_vs_markov1'])} |"
        )
    return lines


def _mean_aggregate(aggregates: list[dict]) -> dict | None:
    if not aggregates:
        return None

    return {
        "mean_tail_h_eff": sum(item["mean_tail_h_eff"] for item in aggregates) / len(aggregates),
        "mean_tail_right_branching": sum(item["mean_tail_right_branching"] for item in aggregates) / len(aggregates),
        "mean_tail_left_branching": sum(item["mean_tail_left_branching"] for item in aggregates) / len(aggregates),
        "mean_tail_rho": sum(item["mean_tail_rho"] for item in aggregates) / len(aggregates),
    }


def _status_hint(row: dict) -> str:
    epsilon = 0.02

    random_h = row.get("random_h_eff")
    random_right = row.get("random_right")
    random_rho = row.get("random_rho")
    markov_h = row.get("markov1_h_eff")
    markov_right = row.get("markov1_right")
    markov_rho = row.get("markov1_rho")

    random_separated = False
    if random_h is not None and random_right is not None and random_rho is not None:
        random_separated = (
            row["observed_h_eff"] < random_h - epsilon
            and abs(row["observed_right"] - random_right) > epsilon
            and abs(row["observed_rho"] - random_rho) > epsilon
        )

    markov_separated = False
    if markov_h is not None and markov_right is not None and markov_rho is not None:
        markov_separated = (
            row["observed_h_eff"] < markov_h - epsilon
            and row["observed_right"] < markov_right - epsilon
            and row["observed_rho"] < markov_rho - epsilon
        )

    if random_separated and markov_separated:
        return "promising"
    if random_separated or markov_separated:
        return "partial"
    return "weak"


def _delta(observed: float | None, control: float | None) -> float | None:
    if observed is None or control is None:
        return None
    return observed - control


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _timestamp_from_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")


def _parse_timestamp(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return datetime.min
