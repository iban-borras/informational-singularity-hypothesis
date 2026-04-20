#!/usr/bin/env python3
"""
HSI v2 Phase 2 Strict-Band Transport-Defect Launcher

Strict orchestration path for the first formal N2-01 pilot:
- discovers Phase 1 runs recursively inside the high-scale repository
- freezes one or more deep-window offsets
- anchors low/high pattern selection to the observed variant run
- reuses the existing return-lag and transport-defect core without new math
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from v2.common.cli import parse_variants, resolve_dir
from v2.common.naming import compact_int
from v2.phase1.report import filter_runs, infer_family_from_latest_run, select_latest_per_variant
from v2.phase1.tower import parse_scales
from v2.phase2.null_pressure import (
    discover_phase1_runs_recursive,
    is_observed_run,
    parse_null_models,
    parse_seed_list,
    select_null_pressure_runs,
    supports_pattern_scale,
)
from v2.phase2.return_lag import build_return_lag_rows, prepare_pattern_selection
from v2.phase2.transport_defect import compute_transport_defect_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the HSI v2 strict-band Phase 2 N2-01 transport-defect pilot."
    )
    parser.add_argument(
        "--phase1-dir",
        type=str,
        default="results/hsi_v2/phase1_high_scales",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/hsi_v2/phase2/transport_defect_strict",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="observed",
        choices=("observed", "nulls", "all"),
        help="observed = B/E strict-band envelope; nulls = strong-null envelope only; all = both.",
    )
    parser.add_argument("--variants", type=str, default="B,E")
    parser.add_argument("--iteration", type=int, default=None)
    parser.add_argument("--segment-bits", type=int, default=None)
    parser.add_argument("--num-segments", type=int, default=None)
    parser.add_argument("--segment-offset-bits", type=int, default=None)
    parser.add_argument("--scales", type=str, default="")
    parser.add_argument("--phase1-policies", type=str, default="")
    parser.add_argument("--low-scale", type=int, default=40)
    parser.add_argument("--high-scale", type=int, default=48)
    parser.add_argument("--top-patterns", type=int, default=64)
    parser.add_argument(
        "--pattern-selection",
        type=str,
        default="bridge-linked",
        choices=("top", "rare-stable", "bridge-linked"),
    )
    parser.add_argument("--long-lag-threshold", type=int, default=4096)
    parser.add_argument(
        "--offsets",
        type=str,
        default="717000000",
        help="Comma-separated list of window start offsets in bits.",
    )
    parser.add_argument("--null-models", type=str, default="markov1,matched-lz")
    parser.add_argument("--matched-lz-seeds", type=str, default="")
    parser.add_argument("--no-family-inference", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    variants = parse_variants(args.variants)
    if not variants:
        parser.error("At least one variant is required.")
    if args.low_scale <= 0 or args.high_scale <= 0 or args.low_scale >= args.high_scale:
        parser.error("--low-scale and --high-scale must satisfy 1 <= low < high.")
    if args.high_scale > 64:
        parser.error("--high-scale must be in the range 1..64.")
    if args.top_patterns <= 0:
        parser.error("--top-patterns must be positive.")
    if args.long_lag_threshold <= 0:
        parser.error("--long-lag-threshold must be positive.")

    try:
        offsets = parse_int_list(args.offsets, label="--offsets", allow_zero=True)
    except ValueError as exc:
        parser.error(str(exc))
    if any(offset < 0 for offset in offsets):
        parser.error("--offsets must contain non-negative integers.")

    null_models: list[str] = []
    matched_lz_seeds: list[int] | None = None
    if args.stage in {"nulls", "all"}:
        try:
            null_models = parse_null_models(args.null_models)
        except ValueError as exc:
            parser.error(str(exc))
        if not null_models:
            parser.error("At least one null model is required for --stage nulls/all.")
        try:
            matched_lz_seeds = (
                parse_seed_list(args.matched_lz_seeds)
                if args.matched_lz_seeds.strip()
                else None
            )
        except ValueError:
            parser.error("--matched-lz-seeds must be a comma-separated list of integers.")

    phase1_dir = resolve_dir(args.phase1_dir, anchor_file=__file__)
    output_dir = resolve_dir(args.output_dir, anchor_file=__file__)
    output_dir.mkdir(parents=True, exist_ok=True)

    phase1_scales = parse_scales(args.scales) if args.scales.strip() else None
    phase1_policies = [item.strip() for item in args.phase1_policies.split(",") if item.strip()] or None

    phase_print(
        "Preparing strict-band N2-01 transport-defect pilot",
        (
            f"stage={args.stage} | variants={','.join(variants)} | "
            f"m={args.low_scale}->{args.high_scale} | sel={args.pattern_selection} | "
            f"offsets={','.join(str(value) for value in offsets)}"
        ),
        quiet=args.quiet,
    )

    runs = discover_phase1_runs_recursive(phase1_dir)
    if not runs:
        parser.error(f"No valid Phase 1 runs found in {phase1_dir}")

    observed_runs, family_inferred = select_observed_runs(
        runs,
        variants=variants,
        iteration=args.iteration,
        segment_bits=args.segment_bits,
        num_segments=args.num_segments,
        segment_offset_bits=args.segment_offset_bits,
        scales=phase1_scales,
        policies=phase1_policies,
        low_scale=args.low_scale,
        high_scale=args.high_scale,
        no_family_inference=args.no_family_inference,
    )
    observed_by_variant = {
        str(run["dataset"]["config"]["variant"]).upper(): run
        for run in observed_runs
    }

    evaluation_specs = []
    if args.stage in {"observed", "all"}:
        evaluation_specs.extend(
            build_observed_specs(observed_runs)
        )

    if args.stage in {"nulls", "all"}:
        for variant in variants:
            try:
                selected = select_null_pressure_runs(
                    runs,
                    observed_variant=variant,
                    null_models=null_models,
                    matched_lz_seeds=matched_lz_seeds,
                    iteration=args.iteration,
                    segment_bits=args.segment_bits,
                    num_segments=args.num_segments,
                    segment_offset_bits=args.segment_offset_bits,
                    scales=phase1_scales,
                    policies=phase1_policies,
                    pattern_scale=args.high_scale,
                    no_family_inference=args.no_family_inference,
                )
            except ValueError as exc:
                parser.error(f"{variant}: {exc}")
            anchor_run = observed_by_variant[variant]
            evaluation_specs.extend(
                build_null_specs(
                    selected["null_specs"],
                    anchor_run=anchor_run,
                )
            )

    if not evaluation_specs:
        parser.error("No evaluation specs remain after selection.")

    segment_bits, num_segments = infer_segment_protocol(observed_runs)
    window_span_bits = segment_bits * num_segments

    phase_print(
        "Running strict-band transport defect",
        f"cases={len(evaluation_specs)} | windows={len(offsets)} | span={window_span_bits}",
        quiet=args.quiet,
    )

    rows = []
    artifacts = []
    for case_index, spec in enumerate(evaluation_specs, start=1):
        case_note(
            (
                f"{case_index}/{len(evaluation_specs)} | variant={spec['variant']} | "
                f"source={spec['source_label']} | anchor={spec['anchor_variant']}"
            ),
            quiet=args.quiet,
        )
        low_selection = prepare_pattern_selection(
            spec["anchor_run"],
            pattern_scale=args.low_scale,
            top_patterns=args.top_patterns,
            pattern_selection=args.pattern_selection,
        )
        high_selection = prepare_pattern_selection(
            spec["anchor_run"],
            pattern_scale=args.high_scale,
            top_patterns=args.top_patterns,
            pattern_selection=args.pattern_selection,
        )

        for offset_index, offset_bits in enumerate(offsets, start=1):
            window_note(
                (
                    f"case {case_index}/{len(evaluation_specs)} | "
                    f"window {offset_index}/{len(offsets)} | offset={offset_bits} | "
                    f"variant={spec['variant']} | source={spec['source_label']}"
                ),
                quiet=args.quiet,
            )
            run_with_offset = clone_run_with_offset(spec["run"], offset_bits)
            low_row = build_return_lag_row_for_spec(
                run_with_offset,
                selection_info=low_selection,
                variant=spec["variant"],
                pattern_scale=args.low_scale,
                top_patterns=args.top_patterns,
                pattern_selection=args.pattern_selection,
                long_lag_threshold=args.long_lag_threshold,
                anchor_kind=spec["anchor_kind"],
                show_progress=not args.quiet,
            )
            high_row = build_return_lag_row_for_spec(
                run_with_offset,
                selection_info=high_selection,
                variant=spec["variant"],
                pattern_scale=args.high_scale,
                top_patterns=args.top_patterns,
                pattern_selection=args.pattern_selection,
                long_lag_threshold=args.long_lag_threshold,
                anchor_kind=spec["anchor_kind"],
                show_progress=not args.quiet,
            )
            result = compute_transport_defect_result(low_row, high_row)
            rows.append(
                build_output_row(
                    result=result,
                    spec=spec,
                    low_row=low_row,
                    high_row=high_row,
                    offset_bits=offset_bits,
                    window_span_bits=window_span_bits,
                )
            )

        artifacts.append(
            {
                "variant": spec["variant"],
                "source_label": spec["source_label"],
                "source_kind": spec["source_kind"],
                "null_model": spec.get("null_model"),
                "null_seed": spec.get("null_seed"),
                "run_dir": spec["run"]["_run_dir"],
                "anchor_run_dir": spec["anchor_run"]["_run_dir"],
            }
        )

    selection = build_selection(
        args=args,
        variants=variants,
        offsets=offsets,
        segment_bits=segment_bits,
        num_segments=num_segments,
        family_inferred=family_inferred,
        null_models=null_models,
        matched_lz_seeds=matched_lz_seeds or [],
    )
    grouped_summary = summarize_rows(rows)

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_slug = build_run_slug(selection, timestamp)
    run_dir = output_dir / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = run_dir / "dataset.json"
    summary_path = run_dir / "summary.json"
    report_path = run_dir / "report.md"
    manifest_path = run_dir / "manifest.json"
    generated_at = datetime.now().isoformat(timespec="seconds")

    dataset_payload = {
        "stage": "phase2_transport_defect_strict",
        "generated_at": generated_at,
        "selection": selection,
        "rows": rows,
        "grouped_summary": grouped_summary,
        "artifacts": artifacts,
        "notes": [
            "Pattern selection is frozen from the observed anchor run for each evaluated variant.",
            "Return-lag rows are rebuilt at the requested deep offsets before the transport defect is computed.",
            "The transport kernel always comes from the evaluated run itself; only the selected pattern list is anchored exogenously.",
            "This launcher is an orchestration path only; it reuses the existing return-lag and transport-defect math.",
        ],
    }
    summary_payload = {
        "generated_at": generated_at,
        "selection": selection,
        "grouped_summary": grouped_summary,
        "compact_rows": [
            {
                "variant": row["variant"],
                "source_label": row["source_label"],
                "source_kind": row["source_kind"],
                "null_model": row["null_model"],
                "null_seed": row["null_seed"],
                "segment_offset_bits": row["segment_offset_bits"],
                "terminal_fraction": row["terminal_fraction"],
                "defect_js_mean": row["defect_js_mean"],
                "defect_wj_mean": row["defect_wj_mean"],
                "defect_flow_mean": row["defect_flow_mean"],
                "defect_top_mass": row["defect_top_mass"],
                "defect_mean_gap": row["defect_mean_gap"],
                "segment_defect_stability": row["segment_defect_stability"],
                "connected_pattern_count": row["connected_pattern_count"],
            }
            for row in rows
        ],
    }
    manifest_payload = {
        "run_slug": run_slug,
        "generated_at": generated_at,
        "script": "hsi_v2_phase2_transport_defect_strict.py",
        "cwd": str(Path.cwd()),
        "outputs": {
            "dataset": str(dataset_path),
            "summary": str(summary_path),
            "report": str(report_path),
            "manifest": str(manifest_path),
        },
        "inputs": {
            "phase1_dir": str(phase1_dir),
            "observed_phase1_runs": [run["_run_dir"] for run in observed_runs],
            "evaluated_phase1_runs": [artifact["run_dir"] for artifact in artifacts],
        },
        "arguments": vars(args),
    }

    with open(dataset_path, "w", encoding="utf-8") as handle:
        json.dump(dataset_payload, handle, indent=2)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(render_markdown_report(selection, grouped_summary, rows, artifacts) + "\n")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, indent=2)

    if not args.quiet:
        if family_inferred:
            print("Using the latest coherent Phase 1 family inferred from the newest observed run.")
            print("")
        print(render_console_summary(rows))
        print("")
        print(f"Saved dataset to: {dataset_path}")
        print(f"Saved summary to: {summary_path}")
        print(f"Saved report to: {report_path}")
        print(f"Saved manifest to: {manifest_path}")

    return 0


def select_observed_runs(
    runs: list[dict],
    *,
    variants: list[str],
    iteration: int | None,
    segment_bits: int | None,
    num_segments: int | None,
    segment_offset_bits: int | None,
    scales: list[int] | None,
    policies: list[str] | None,
    low_scale: int,
    high_scale: int,
    no_family_inference: bool,
) -> tuple[list[dict], bool]:
    observed_runs = [run for run in runs if is_observed_run(run)]
    family = None
    explicit_filters = any(
        value is not None
        for value in (iteration, segment_bits, num_segments, segment_offset_bits)
    ) or scales is not None or policies is not None
    if not no_family_inference and not explicit_filters:
        family_candidates = filter_runs(observed_runs, variants=variants)
        family_candidates = [
            run
            for run in family_candidates
            if supports_pattern_scale(run, pattern_scale=low_scale)
            and supports_pattern_scale(run, pattern_scale=high_scale)
        ]
        family = infer_family_from_latest_run(family_candidates)

    matching = filter_runs(
        observed_runs,
        variants=variants,
        iteration=iteration,
        segment_bits=segment_bits,
        num_segments=num_segments,
        segment_offset_bits=segment_offset_bits,
        scales=scales,
        policies=policies,
        family=family,
    )
    matching = [
        run
        for run in matching
        if supports_pattern_scale(run, pattern_scale=low_scale)
        and supports_pattern_scale(run, pattern_scale=high_scale)
    ]
    selected = select_latest_per_variant(matching, variant_order=variants)
    selected_variants = {str(run["dataset"]["config"]["variant"]).upper() for run in selected}
    missing = [variant for variant in variants if variant not in selected_variants]
    if missing:
        raise ValueError(
            "Missing observed strict-band Phase 1 runs for variants: "
            + ", ".join(missing)
        )
    return selected, family is not None


def build_observed_specs(observed_runs: list[dict]) -> list[dict]:
    specs = []
    for run in observed_runs:
        variant = str(run["dataset"]["config"]["variant"]).upper()
        specs.append(
            {
                "variant": variant,
                "run": run,
                "anchor_run": run,
                "anchor_variant": variant,
                "anchor_kind": "self-observed",
                "source_kind": "observed",
                "source_label": "observed",
                "null_model": None,
                "null_seed": None,
            }
        )
    return specs


def build_null_specs(null_specs: list[dict], *, anchor_run: dict) -> list[dict]:
    variant = str(anchor_run["dataset"]["config"]["variant"]).upper()
    specs = []
    for item in null_specs:
        specs.append(
            {
                "variant": variant,
                "run": item["run"],
                "anchor_run": anchor_run,
                "anchor_variant": variant,
                "anchor_kind": "observed-fixed",
                "source_kind": "null_surrogate",
                "source_label": item["null_label"],
                "null_model": item["null_model"],
                "null_seed": item.get("null_seed"),
            }
        )
    return specs


def build_return_lag_row_for_spec(
    run: dict,
    *,
    selection_info: dict,
    variant: str,
    pattern_scale: int,
    top_patterns: int,
    pattern_selection: str,
    long_lag_threshold: int,
    anchor_kind: str,
    show_progress: bool,
) -> dict:
    override = {
        variant: {
            **selection_info,
            "anchor_kind": anchor_kind,
        }
    }
    rows = build_return_lag_rows(
        [run],
        pattern_scale=pattern_scale,
        top_patterns=top_patterns,
        pattern_selection=pattern_selection,
        long_lag_threshold=long_lag_threshold,
        selection_overrides=override,
        show_progress=show_progress,
    )
    return rows[0]


def build_output_row(
    *,
    result: dict,
    spec: dict,
    low_row: dict,
    high_row: dict,
    offset_bits: int,
    window_span_bits: int,
) -> dict:
    return {
        "variant": result["variant"],
        "source_label": spec["source_label"],
        "source_kind": spec["source_kind"],
        "null_model": spec["null_model"],
        "null_seed": spec["null_seed"],
        "anchor_variant": spec["anchor_variant"],
        "anchor_kind": spec["anchor_kind"],
        "anchor_run_dir": spec["anchor_run"]["_run_dir"],
        "run_dir": result["run_dir"],
        "segment_offset_bits": offset_bits,
        "window_start_bits": offset_bits,
        "window_end_bits": offset_bits + window_span_bits,
        "low_scale": result["low_scale"],
        "high_scale": result["high_scale"],
        "pattern_selection": result["pattern_selection"],
        "selected_low_pattern_count": result["selected_low_pattern_count"],
        "selected_high_pattern_count": result["selected_high_pattern_count"],
        "connected_pattern_count": result["connected_pattern_count"],
        "terminal_pattern_count": result["terminal_pattern_count"],
        "terminal_fraction": result["terminal_fraction"],
        "defect_js_mean": result["defect_js_mean"],
        "defect_wj_mean": result["defect_wj_mean"],
        "defect_flow_mean": result["defect_flow_mean"],
        "defect_top_mass": result["defect_top_mass"],
        "defect_mean_gap": result["defect_mean_gap"],
        "segment_defect_stability": result["segment_defect_stability"],
        "phase1_policies": result["phase1_policies"],
        "transport_result": result,
        "low_row_path": low_row["dataset_path"],
        "high_row_path": high_row["dataset_path"],
        "low_selection_source_variant": low_row.get("selection_source_variant"),
        "high_selection_source_variant": high_row.get("selection_source_variant"),
        "low_selection_anchor_kind": low_row.get("selection_anchor_kind"),
        "high_selection_anchor_kind": high_row.get("selection_anchor_kind"),
    }


def build_selection(
    *,
    args,
    variants: list[str],
    offsets: list[int],
    segment_bits: int,
    num_segments: int,
    family_inferred: bool,
    null_models: list[str],
    matched_lz_seeds: list[int],
) -> dict:
    return {
        "stage": args.stage,
        "variants": variants,
        "iteration": args.iteration,
        "segment_bits": segment_bits,
        "num_segments": num_segments,
        "offsets": offsets,
        "low_scale": args.low_scale,
        "high_scale": args.high_scale,
        "top_patterns": args.top_patterns,
        "pattern_selection": args.pattern_selection,
        "long_lag_threshold": args.long_lag_threshold,
        "null_models": null_models,
        "matched_lz_seeds": matched_lz_seeds,
        "family_inferred": family_inferred,
    }


def build_run_slug(selection: dict, timestamp: str) -> str:
    variants = "-".join(selection["variants"])
    offsets = selection["offsets"]
    if len(offsets) == 1:
        offset_part = f"off-{compact_int(offsets[0])}"
    else:
        offset_part = f"off-{compact_int(min(offsets))}-plus-{len(offsets)}"
    return (
        f"phase2-transport-defect-strict__stage-{selection['stage']}"
        f"__m-{selection['low_scale']}-{selection['high_scale']}"
        f"__sel-{selection['pattern_selection']}__top-{selection['top_patterns']}"
        f"__{offset_part}__var-{variants}__{timestamp}"
    )


def summarize_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str | None, int | None], list[dict]] = {}
    for row in rows:
        key = (
            row["variant"],
            row["source_kind"],
            row.get("null_model"),
            row.get("null_seed"),
        )
        grouped.setdefault(key, []).append(row)

    summaries = []
    for (_, _, _, _), group in grouped.items():
        first = group[0]
        summaries.append(
            {
                "variant": first["variant"],
                "source_label": first["source_label"],
                "source_kind": first["source_kind"],
                "null_model": first.get("null_model"),
                "null_seed": first.get("null_seed"),
                "offset_count": len(group),
                "offsets": [row["segment_offset_bits"] for row in group],
                "terminal_fraction_avg": mean_of(group, "terminal_fraction"),
                "terminal_fraction_max": max_of(group, "terminal_fraction"),
                "defect_js_mean_avg": mean_of(group, "defect_js_mean"),
                "defect_js_mean_min": min_of(group, "defect_js_mean"),
                "defect_js_mean_max": max_of(group, "defect_js_mean"),
                "defect_wj_mean_avg": mean_of(group, "defect_wj_mean"),
                "defect_wj_mean_min": min_of(group, "defect_wj_mean"),
                "defect_wj_mean_max": max_of(group, "defect_wj_mean"),
                "defect_flow_mean_avg": mean_of(group, "defect_flow_mean"),
                "defect_flow_mean_min": min_of(group, "defect_flow_mean"),
                "defect_flow_mean_max": max_of(group, "defect_flow_mean"),
                "defect_mean_gap_avg": mean_of(group, "defect_mean_gap"),
                "segment_defect_stability_avg": mean_of(group, "segment_defect_stability"),
                "connected_pattern_count_avg": mean_of(group, "connected_pattern_count"),
                "worst_offset_by_flow": max(group, key=lambda item: item["defect_flow_mean"])["segment_offset_bits"],
            }
        )

    return sorted(
        summaries,
        key=lambda item: (
            item["variant"],
            source_kind_sort_key(item["source_kind"], item.get("null_model")),
            item.get("null_seed") or -1,
        ),
    )


def render_console_summary(rows: list[dict]) -> str:
    variant_width = 16
    source_width = 28
    lines = [
        "Phase 2 strict-band transport-defect pilot",
        "-" * 154,
        f"{'variant':<{variant_width}} {'source':<{source_width}} {'offset':>12}{'term':>8}{'d_JS':>10}{'d_wJ':>10}{'d_flow':>10}{'d_top':>10}{'d_mean':>10}{'seg_stab':>10}{'conn':>8}",
    ]
    for row in sorted(
        rows,
        key=lambda item: (
            item["variant"],
            source_kind_sort_key(item["source_kind"], item.get("null_model")),
            item.get("null_seed") or -1,
            item["segment_offset_bits"],
        ),
    ):
        lines.append(
            f"{truncate_label(row['variant'], variant_width):<{variant_width}} "
            f"{truncate_label(row['source_label'], source_width):<{source_width}} "
            f"{row['segment_offset_bits']:>12}"
            f"{row['terminal_fraction']:>8.4f}"
            f"{row['defect_js_mean']:>10.4f}"
            f"{row['defect_wj_mean']:>10.4f}"
            f"{row['defect_flow_mean']:>10.4f}"
            f"{row['defect_top_mass']:>10.4f}"
            f"{row['defect_mean_gap']:>10.4f}"
            f"{row['segment_defect_stability']:>10.4f}"
            f"{row['connected_pattern_count']:>8}"
        )
    return "\n".join(lines)


def render_markdown_report(
    selection: dict,
    grouped_summary: list[dict],
    rows: list[dict],
    artifacts: list[dict],
) -> str:
    lines = [
        "# Phase 2 Strict-Band Transport-Defect Pilot",
        "",
        "## Selection",
        "",
        f"- Stage: {selection['stage']}",
        f"- Variants: {', '.join(selection['variants'])}",
        f"- Segment protocol: {selection['num_segments']} x {selection['segment_bits']}",
        f"- Offsets: {', '.join(str(value) for value in selection['offsets'])}",
        f"- Low/high scales: {selection['low_scale']} -> {selection['high_scale']}",
        f"- Pattern selection: {selection['pattern_selection']}",
        f"- Top patterns: {selection['top_patterns']}",
        f"- Long-lag threshold: {selection['long_lag_threshold']}",
        f"- Family inferred: {selection['family_inferred']}",
        "",
        "## Frozen Evaluation Runs",
        "",
    ]
    for artifact in artifacts:
        seed = f" | seed={artifact['null_seed']}" if artifact.get("null_seed") is not None else ""
        null_model = (
            f" | null_model={artifact['null_model']}"
            if artifact.get("null_model")
            else ""
        )
        lines.append(
            f"- `{artifact['variant']}` | source={artifact['source_label']} | kind={artifact['source_kind']}"
            f"{null_model}{seed} | run_dir=`{artifact['run_dir']}` | anchor_run=`{artifact['anchor_run_dir']}`"
        )

    lines.extend(
        [
            "",
            "## Grouped Summary",
            "",
            "| Variant | Source | Offsets | Term avg | Term max | d_JS avg | d_JS min | d_JS max | d_flow avg | d_flow min | d_flow max | d_mean avg | Worst flow offset |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in grouped_summary:
        lines.append(
            f"| {item['variant']} | {item['source_label']} | {item['offset_count']} | "
            f"{fmt(item['terminal_fraction_avg'])} | {fmt(item['terminal_fraction_max'])} | "
            f"{fmt(item['defect_js_mean_avg'])} | {fmt(item['defect_js_mean_min'])} | {fmt(item['defect_js_mean_max'])} | "
            f"{fmt(item['defect_flow_mean_avg'])} | {fmt(item['defect_flow_mean_min'])} | {fmt(item['defect_flow_mean_max'])} | "
            f"{fmt(item['defect_mean_gap_avg'])} | {item['worst_offset_by_flow']} |"
        )

    lines.extend(
        [
            "",
            "## Offset Readout",
            "",
            "| Variant | Source | Offset | Term | d_JS | d_wJ | d_flow | d_top | d_mean | seg_stab | Connected |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(
        rows,
        key=lambda item: (
            item["variant"],
            source_kind_sort_key(item["source_kind"], item.get("null_model")),
            item.get("null_seed") or -1,
            item["segment_offset_bits"],
        ),
    ):
        lines.append(
            f"| {row['variant']} | {row['source_label']} | {row['segment_offset_bits']} | "
            f"{fmt(row['terminal_fraction'])} | {fmt(row['defect_js_mean'])} | {fmt(row['defect_wj_mean'])} | "
            f"{fmt(row['defect_flow_mean'])} | {fmt(row['defect_top_mass'])} | {fmt(row['defect_mean_gap'])} | "
            f"{fmt(row['segment_defect_stability'])} | {row['connected_pattern_count']} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The selected low/high pattern lists are frozen from the observed anchor run for each evaluated variant.",
            "- The return-lag rows are rebuilt on the requested deep offsets before the transport defect is computed.",
            "- The transport kernel still belongs to the evaluated run itself; this launcher only removes selection drift across observed/null contrasts.",
        ]
    )
    return "\n".join(lines)


def clone_run_with_offset(run: dict, segment_offset_bits: int) -> dict:
    cloned = dict(run)
    dataset = deepcopy(run["dataset"])
    config = dataset["config"]
    segment_bits = int(config["segment_bits"])
    num_segments = int(config["num_segments"])
    config["segment_offset_bits"] = segment_offset_bits
    config["loaded_observable_bits"] = segment_offset_bits + (segment_bits * num_segments)
    cloned["dataset"] = dataset
    return cloned


def infer_segment_protocol(runs: list[dict]) -> tuple[int, int]:
    first = runs[0]["dataset"]["config"]
    segment_bits = int(first["segment_bits"])
    num_segments = int(first["num_segments"])
    for run in runs[1:]:
        config = run["dataset"]["config"]
        if int(config["segment_bits"]) != segment_bits or int(config["num_segments"]) != num_segments:
            raise ValueError("All selected observed runs must share the same segment protocol.")
    return segment_bits, num_segments


def parse_int_list(raw: str, *, label: str, allow_zero: bool) -> list[int]:
    values = []
    seen = set()
    for token in raw.split(","):
        item = token.strip()
        if not item:
            continue
        value = int(item)
        if not allow_zero and value == 0:
            raise ValueError(f"{label} must not contain zero.")
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
    if not values:
        raise ValueError(f"{label} requires at least one integer.")
    return values


def mean_of(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    return float(sum(float(row[key]) for row in rows) / len(rows))


def min_of(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    return float(min(float(row[key]) for row in rows))


def max_of(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    return float(max(float(row[key]) for row in rows))


def source_kind_sort_key(source_kind: str, null_model: str | None) -> tuple[int, str]:
    if source_kind == "observed":
        return (0, "observed")
    if null_model == "markov1":
        return (1, "markov1")
    if null_model == "matched-lz":
        return (2, "matched-lz")
    return (9, source_kind)


def truncate_label(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    if width <= 3:
        return value[:width]
    return value[: width - 3] + "..."


def fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def phase_print(title: str, detail: str, *, quiet: bool) -> None:
    if quiet:
        return
    print(f"[Phase] {title}")
    print(f"        {detail}")
    print("")


def case_note(detail: str, *, quiet: bool) -> None:
    if quiet:
        return
    print(f"[Case] {detail}")


def window_note(detail: str, *, quiet: bool) -> None:
    if quiet:
        return
    print(f"[Window] {detail}")


if __name__ == "__main__":
    raise SystemExit(main())
