#!/usr/bin/env python3
"""
HSI v2 Phase 2 Counterfactual Transport-Defect Launcher

Strict orchestration path for the proposed N2-02 pilot:
- freezes the observed anchor run and its low/high pattern supports
- freezes the observed transport kernel of the anchor run
- evaluates candidate observed/null runs against that fixed law
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hsi_v2_phase2_transport_defect_strict import (
    artifact_source_info,
    build_null_specs,
    build_return_lag_row_for_spec,
    case_note,
    clone_run_with_offset,
    count_unique_run_dirs,
    fmt,
    infer_segment_protocol,
    max_of,
    mean_of,
    min_of,
    parse_int_list,
    phase_print,
    select_observed_runs,
    source_kind_sort_key,
    truncate_label,
    window_note,
)
from v2.common.cli import parse_variants, resolve_dir
from v2.common.naming import compact_int
from v2.phase1.tower import parse_scales
from v2.phase2.null_pressure import (
    discover_phase1_runs_recursive,
    parse_null_models,
    parse_seed_list,
    select_null_pressure_runs,
)
from v2.phase2.return_lag import prepare_frozen_source_cache, prepare_pattern_selection
from v2.phase2.transport_defect_counterfactual import (
    compute_counterfactual_transport_defect_result,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the HSI v2 strict N2-02 counterfactual transport-defect pilot."
    )
    parser.add_argument(
        "--phase1-dir",
        type=str,
        default="results/hsi_v2/phase1_high_scales",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/hsi_v2/phase2/transport_defect_counterfactual",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="observed",
        choices=("observed", "nulls", "all"),
        help="observed = anchored observed envelope; nulls = anchored strong-null envelope; all = both.",
    )
    parser.add_argument("--variants", type=str, default="B,E")
    parser.add_argument("--anchor-variant", type=str, default="B")
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
    anchor_variants = parse_variants(args.anchor_variant)
    if len(anchor_variants) != 1:
        parser.error("--anchor-variant must contain exactly one variant.")
    anchor_variant = anchor_variants[0]

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

    observed_variants = [anchor_variant]
    if args.stage in {"observed", "all"}:
        for variant in variants:
            if variant not in observed_variants:
                observed_variants.append(variant)

    phase_print(
        "Preparing counterfactual N2-02 transport-defect pilot",
        (
            f"stage={args.stage} | anchor={anchor_variant} | variants={','.join(observed_variants)} | "
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
        variants=observed_variants,
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
    anchor_run = observed_by_variant.get(anchor_variant)
    if anchor_run is None:
        parser.error(f"Missing observed anchor run for variant {anchor_variant}.")

    evaluation_specs = []
    if args.stage in {"observed", "all"}:
        evaluation_specs.extend(
            build_observed_specs_counterfactual(
                observed_by_variant,
                anchor_run=anchor_run,
                variants=observed_variants,
            )
        )

    if args.stage in {"nulls", "all"}:
        try:
            selected = select_null_pressure_runs(
                runs,
                observed_variant=anchor_variant,
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
            parser.error(f"{anchor_variant}: {exc}")
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
    frozen_total_bits = max(offsets) + window_span_bits

    phase_print(
        "Running counterfactual transport defect",
        f"cases={len(evaluation_specs)} | windows={len(offsets)} | span={window_span_bits}",
        quiet=args.quiet,
    )
    phase_print(
        "Freezing source realizations across the full sweep",
        f"runs={count_unique_run_dirs(evaluation_specs)} | required_bits={frozen_total_bits}",
        quiet=args.quiet,
    )

    preloaded_sources = prepare_frozen_source_cache(
        [spec["run"] for spec in evaluation_specs],
        required_bits=frozen_total_bits,
        show_progress=not args.quiet,
    )

    rows = []
    artifacts = []
    low_selection = prepare_pattern_selection(
        anchor_run,
        pattern_scale=args.low_scale,
        top_patterns=args.top_patterns,
        pattern_selection=args.pattern_selection,
    )
    high_selection = prepare_pattern_selection(
        anchor_run,
        pattern_scale=args.high_scale,
        top_patterns=args.top_patterns,
        pattern_selection=args.pattern_selection,
    )
    anchor_run_dir = Path(anchor_run["_run_dir"]).resolve()

    for case_index, spec in enumerate(evaluation_specs, start=1):
        case_note(
            (
                f"{case_index}/{len(evaluation_specs)} | variant={spec['variant']} | "
                f"source={spec['source_label']} | anchor={spec['anchor_variant']}"
            ),
            quiet=args.quiet,
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
                preloaded_sources=preloaded_sources,
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
                preloaded_sources=preloaded_sources,
                show_progress=not args.quiet,
            )
            result = compute_counterfactual_transport_defect_result(
                low_row,
                high_row,
                anchor_run_dir=anchor_run_dir,
            )
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
                "source_freeze_mode": artifact_source_info(
                    preloaded_sources,
                    spec["run"]["_run_dir"],
                    "source_freeze_mode",
                ),
                "frozen_total_bits": artifact_source_info(
                    preloaded_sources,
                    spec["run"]["_run_dir"],
                    "frozen_total_bits",
                ),
            }
        )

    selection = build_selection(
        args=args,
        anchor_variant=anchor_variant,
        variants=observed_variants,
        offsets=offsets,
        segment_bits=segment_bits,
        num_segments=num_segments,
        frozen_total_bits=frozen_total_bits,
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
        "stage": "phase2_transport_defect_counterfactual",
        "generated_at": generated_at,
        "selection": selection,
        "rows": rows,
        "grouped_summary": grouped_summary,
        "artifacts": artifacts,
        "notes": [
            "Pattern selection is frozen from the observed anchor run.",
            "The transport kernel is also frozen from the observed anchor run.",
            "Return-lag rows are rebuilt at the requested deep offsets before the counterfactual defect is computed.",
            "Source realizations are frozen once across the full offset sweep before any return-lag rebuild.",
            "Support coverage is reported separately so uncovered observed child mass is not hidden by profile renormalization.",
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
                "support_coverage_mean": row["support_coverage_mean"],
                "zero_coverage_fraction": row["zero_coverage_fraction"],
                "defect_js_mean": row["defect_js_mean"],
                "defect_wj_mean": row["defect_wj_mean"],
                "defect_flow_mean": row["defect_flow_mean"],
                "defect_top_mass": row["defect_top_mass"],
                "defect_mean_gap": row["defect_mean_gap"],
                "segment_defect_stability": row["segment_defect_stability"],
                "connected_pattern_count": row["connected_pattern_count"],
                "source_freeze_mode": row["source_freeze_mode"],
                "frozen_total_bits": row["frozen_total_bits"],
            }
            for row in rows
        ],
    }
    manifest_payload = {
        "run_slug": run_slug,
        "generated_at": generated_at,
        "script": "hsi_v2_phase2_transport_defect_counterfactual.py",
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
            "anchor_run_dir": str(anchor_run_dir),
            "frozen_source_run_dirs": sorted(preloaded_sources.keys()),
            "frozen_total_bits": frozen_total_bits,
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


def build_observed_specs_counterfactual(
    observed_by_variant: dict[str, dict],
    *,
    anchor_run: dict,
    variants: list[str],
) -> list[dict]:
    anchor_variant = str(anchor_run["dataset"]["config"]["variant"]).upper()
    specs = []
    for variant in variants:
        run = observed_by_variant[variant]
        specs.append(
            {
                "variant": variant,
                "run": run,
                "anchor_run": anchor_run,
                "anchor_variant": anchor_variant,
                "anchor_kind": "observed-anchor" if variant == anchor_variant else "observed-fixed",
                "source_kind": "observed",
                "source_label": "observed",
                "null_model": None,
                "null_seed": None,
            }
        )
    return specs


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
        "support_coverage_mean": result["support_coverage_mean"],
        "zero_coverage_fraction": result["zero_coverage_fraction"],
        "defect_js_mean": result["defect_js_mean"],
        "defect_wj_mean": result["defect_wj_mean"],
        "defect_flow_mean": result["defect_flow_mean"],
        "defect_top_mass": result["defect_top_mass"],
        "defect_mean_gap": result["defect_mean_gap"],
        "segment_defect_stability": result["segment_defect_stability"],
        "selection_source_variant": low_row["selection_source_variant"],
        "selection_source_run_dir": low_row["selection_source_run_dir"],
        "selection_anchor_kind": low_row["selection_anchor_kind"],
        "source_struct_path": low_row["source_struct_path"],
        "source_freeze_mode": low_row.get("source_freeze_mode"),
        "frozen_total_bits": low_row.get("frozen_total_bits"),
        "low_source_freeze_mode": low_row.get("source_freeze_mode"),
        "high_source_freeze_mode": high_row.get("source_freeze_mode"),
        "low_row": low_row,
        "high_row": high_row,
        "counterfactual_result": result,
    }


def build_selection(
    *,
    args,
    anchor_variant: str,
    variants: list[str],
    offsets: list[int],
    segment_bits: int,
    num_segments: int,
    frozen_total_bits: int,
    family_inferred: bool,
    null_models: list[str],
    matched_lz_seeds: list[int],
) -> dict:
    return {
        "stage": args.stage,
        "anchor_variant": anchor_variant,
        "variants": variants,
        "iteration": args.iteration,
        "segment_bits": segment_bits,
        "num_segments": num_segments,
        "offsets": offsets,
        "source_freeze_mode": "sweep-global-frozen",
        "frozen_total_bits": frozen_total_bits,
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
        f"phase2-transport-defect-counterfactual__stage-{selection['stage']}"
        f"__anchor-{selection['anchor_variant']}"
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
                "support_coverage_mean_avg": mean_of(group, "support_coverage_mean"),
                "support_coverage_mean_min": min_of(group, "support_coverage_mean"),
                "support_coverage_mean_max": max_of(group, "support_coverage_mean"),
                "zero_coverage_fraction_avg": mean_of(group, "zero_coverage_fraction"),
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
        "Phase 2 counterfactual transport-defect pilot",
        "-" * 162,
        f"{'variant':<{variant_width}} {'source':<{source_width}} {'offset':>12}{'term':>8}{'cov':>8}{'d_JS':>10}{'d_wJ':>10}{'d_flow':>10}{'d_top':>10}{'d_mean':>10}{'seg_stab':>10}{'conn':>8}",
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
            f"{row['support_coverage_mean']:>8.4f}"
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
        "# Phase 2 Counterfactual Transport-Defect Pilot",
        "",
        "## Selection",
        "",
        f"- Stage: {selection['stage']}",
        f"- Anchor variant: {selection['anchor_variant']}",
        f"- Variants: {', '.join(selection['variants'])}",
        f"- Segment protocol: {selection['num_segments']} x {selection['segment_bits']}",
        f"- Offsets: {', '.join(str(value) for value in selection['offsets'])}",
        f"- Source freeze mode: {selection['source_freeze_mode']}",
        f"- Frozen total bits: {selection['frozen_total_bits']}",
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
            f"{null_model}{seed} | freeze={artifact.get('source_freeze_mode')} | "
            f"frozen_bits={artifact.get('frozen_total_bits')} | run_dir=`{artifact['run_dir']}` | "
            f"anchor_run=`{artifact['anchor_run_dir']}`"
        )

    lines.extend(
        [
            "",
            "## Grouped Summary",
            "",
            "| Variant | Source | Offsets | Term avg | Cov avg | Cov min | Cov max | d_JS avg | d_JS min | d_JS max | d_flow avg | d_flow min | d_flow max | d_mean avg | Worst flow offset |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in grouped_summary:
        lines.append(
            f"| {item['variant']} | {item['source_label']} | {item['offset_count']} | "
            f"{fmt(item['terminal_fraction_avg'])} | "
            f"{fmt(item['support_coverage_mean_avg'])} | {fmt(item['support_coverage_mean_min'])} | {fmt(item['support_coverage_mean_max'])} | "
            f"{fmt(item['defect_js_mean_avg'])} | {fmt(item['defect_js_mean_min'])} | {fmt(item['defect_js_mean_max'])} | "
            f"{fmt(item['defect_flow_mean_avg'])} | {fmt(item['defect_flow_mean_min'])} | {fmt(item['defect_flow_mean_max'])} | "
            f"{fmt(item['defect_mean_gap_avg'])} | {item['worst_offset_by_flow']} |"
        )

    lines.extend(
        [
            "",
            "## Offset Readout",
            "",
            "| Variant | Source | Offset | Term | Cov | Zero-cov | d_JS | d_wJ | d_flow | d_top | d_mean | seg_stab | Connected |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
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
            f"{fmt(row['terminal_fraction'])} | {fmt(row['support_coverage_mean'])} | {fmt(row['zero_coverage_fraction'])} | "
            f"{fmt(row['defect_js_mean'])} | {fmt(row['defect_wj_mean'])} | {fmt(row['defect_flow_mean'])} | "
            f"{fmt(row['defect_top_mass'])} | {fmt(row['defect_mean_gap'])} | "
            f"{fmt(row['segment_defect_stability'])} | {row['connected_pattern_count']} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The selected low/high pattern lists are frozen from the observed anchor run.",
            "- The transport kernel is also frozen from the observed anchor run instead of being rebuilt from the evaluated run.",
            "- Source realizations are frozen once across the full offset sweep, so null surrogates are not regenerated window by window.",
            "- The transported profile is renormalized over covered child mass only; the missing observed child mass is reported separately through support coverage.",
            "- This launcher keeps the strict deep-band orchestration contract while changing only the transport law being tested.",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
