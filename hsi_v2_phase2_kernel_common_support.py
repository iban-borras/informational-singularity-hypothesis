#!/usr/bin/env python3
"""
HSI v2 Phase 2 Common-Support Kernel Divergence Launcher

Minimal executable scaffold for the proposed N2-03 object:
- freezes the observed anchor run
- freezes the anchor-selected parent/child support
- compares candidate transport kernels against the anchor kernel on shared support

This first scaffold is structural:
- it records the late-band anchor offset in metadata
- but the current computation still acts on Phase 1 transport kernels
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
    case_note,
    count_unique_run_dirs,
    fmt,
    max_of,
    mean_of,
    min_of,
    phase_print,
    select_observed_runs,
    source_kind_sort_key,
    truncate_label,
)
from v2.common.cli import parse_variants, resolve_dir
from v2.common.naming import compact_int
from v2.phase2.return_lag import prepare_frozen_source_cache
from v2.phase1.tower import parse_scales
from v2.phase2.kernel_common_support import (
    compute_common_support_kernel_result,
    compute_window_local_common_support_kernel_result,
)
from v2.phase2.null_pressure import (
    discover_phase1_runs_recursive,
    parse_null_models,
    parse_seed_list,
    select_null_pressure_runs,
)
from v2.phase2.return_lag import prepare_pattern_selection
from v2.phase2.defects import classify_variant_role


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the HSI v2 Phase 2 common-support kernel-divergence scaffold."
    )
    parser.add_argument(
        "--phase1-dir",
        type=str,
        default="results/hsi_v2/phase1_high_scales",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/hsi_v2/phase2/kernel_common_support",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="observed",
        choices=("observed", "nulls", "all"),
        help="observed = anchor plus observed contrasts; nulls = anchor plus strong nulls; all = both.",
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
    parser.add_argument(
        "--anchor-offset-bits",
        type=int,
        default=717000000,
        help="Late-band anchor recorded in metadata for downstream alignment.",
    )
    parser.add_argument(
        "--kernel-mode",
        type=str,
        default="window-local",
        choices=("window-local", "structural"),
        help="window-local = rebuild kernel weights from child-anchored realized events in the late window; structural = global Phase 1 kernel only.",
    )
    parser.add_argument(
        "--window-bits",
        type=int,
        default=None,
        help="Late-window size in bits. Defaults to the selected Phase 1 segment protocol span.",
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
    if args.anchor_offset_bits < 0:
        parser.error("--anchor-offset-bits must be non-negative.")
    if args.window_bits is not None and args.window_bits <= 0:
        parser.error("--window-bits must be positive when provided.")

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
    phase1_policies = [
        item.strip()
        for item in args.phase1_policies.split(",")
        if item.strip()
    ] or None

    observed_variants = [anchor_variant]
    if args.stage in {"observed", "all"}:
        for variant in variants:
            if variant not in observed_variants:
                observed_variants.append(variant)

    phase_print(
        "Preparing common-support kernel-divergence scaffold",
        (
            f"stage={args.stage} | anchor={anchor_variant} | variants={','.join(observed_variants)} | "
            f"m={args.low_scale}->{args.high_scale} | sel={args.pattern_selection} | "
            f"mode={args.kernel_mode} | late_anchor={args.anchor_offset_bits}"
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
    anchor_config = anchor_run["dataset"]["config"]
    window_bits = (
        int(args.window_bits)
        if args.window_bits is not None
        else int(anchor_config["segment_bits"]) * int(anchor_config["num_segments"])
    )
    if window_bits < args.high_scale:
        parser.error("--window-bits must be at least as large as --high-scale.")

    evaluation_specs = []
    if args.stage in {"observed", "all"}:
        evaluation_specs.extend(
            build_observed_specs(
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

    preloaded_sources = None
    if args.kernel_mode == "window-local":
        required_bits = args.anchor_offset_bits + window_bits
        phase_print(
            "Freezing source realizations for the local window rebuild",
            f"runs={count_unique_run_dirs(evaluation_specs)} | required_bits={required_bits}",
            quiet=args.quiet,
        )
        preloaded_sources = prepare_frozen_source_cache(
            [spec["run"] for spec in evaluation_specs] + [anchor_run],
            required_bits=required_bits,
            show_progress=not args.quiet,
        )

    phase_print(
        "Running common-support kernel divergence",
        (
            f"cases={len(evaluation_specs)} | parents={len(low_selection['selected_patterns'])} | "
            f"children={len(high_selection['selected_patterns'])} | window_bits={window_bits}"
        ),
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
        if args.kernel_mode == "window-local":
            anchor_bits = preloaded_sources[str(Path(spec["anchor_run"]["_run_dir"]).resolve())]["bits"]
            candidate_bits = preloaded_sources[str(Path(spec["run"]["_run_dir"]).resolve())]["bits"]
            window_start = args.anchor_offset_bits
            window_end = window_start + window_bits
            result = compute_window_local_common_support_kernel_result(
                anchor_run_dir=spec["anchor_run"]["_run_dir"],
                candidate_run_dir=spec["run"]["_run_dir"],
                anchor_window_bits=anchor_bits[window_start:window_end],
                candidate_window_bits=candidate_bits[window_start:window_end],
                low_scale=args.low_scale,
                high_scale=args.high_scale,
                pattern_selection=args.pattern_selection,
                low_selected_patterns=low_selection["selected_patterns"],
                high_selected_patterns=high_selection["selected_patterns"],
            )
        else:
            result = compute_common_support_kernel_result(
                anchor_run_dir=spec["anchor_run"]["_run_dir"],
                candidate_run_dir=spec["run"]["_run_dir"],
                low_scale=args.low_scale,
                high_scale=args.high_scale,
                pattern_selection=args.pattern_selection,
                low_selected_patterns=low_selection["selected_patterns"],
                high_selected_patterns=high_selection["selected_patterns"],
            )
        rows.append(
            build_output_row(
                result=result,
                spec=spec,
                low_selection=low_selection,
                high_selection=high_selection,
                anchor_offset_bits=args.anchor_offset_bits,
                window_bits=window_bits,
                kernel_mode=args.kernel_mode,
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
                "source_freeze_mode": (
                    artifact_source_info(
                        preloaded_sources,
                        spec["run"]["_run_dir"],
                        "source_freeze_mode",
                    )
                    if preloaded_sources is not None
                    else None
                ),
                "frozen_total_bits": (
                    artifact_source_info(
                        preloaded_sources,
                        spec["run"]["_run_dir"],
                        "frozen_total_bits",
                    )
                    if preloaded_sources is not None
                    else None
                ),
            }
        )

    selection = build_selection(
        args=args,
        anchor_variant=anchor_variant,
        variants=observed_variants,
        family_inferred=family_inferred,
        null_models=null_models,
        matched_lz_seeds=matched_lz_seeds or [],
        anchor_run=anchor_run,
        low_selection=low_selection,
        high_selection=high_selection,
        window_bits=window_bits,
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
        "stage": "phase2_kernel_common_support",
        "generated_at": generated_at,
        "selection": selection,
        "rows": rows,
        "grouped_summary": grouped_summary,
        "artifacts": artifacts,
        "notes": [
            "This is the first executable scaffold of N2-03.",
            (
                "Kernel weights are rebuilt locally from child-anchored realized events inside the late window."
                if args.kernel_mode == "window-local"
                else "Kernel weights come from the global Phase 1 transport fibers."
            ),
            "The late-band anchor offset is recorded explicitly for downstream alignment.",
            "The object compares anchor and candidate kernels on shared child support under the same anchor-selected parent/child pattern universe.",
            "No flow-like metric is emitted yet because the child-pattern axis still lacks a canonical ordering.",
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
                "kernel_mode": row["kernel_mode"],
                "anchor_offset_bits": row["anchor_offset_bits"],
                "window_bits": row["window_bits"],
                "anchor_local_parent_fraction": row.get("anchor_local_parent_fraction"),
                "support_coverage_anchor_mean": row["support_coverage_anchor_mean"],
                "candidate_spillover_mean": row["candidate_spillover_mean"],
                "support_jaccard_mean": row["support_jaccard_mean"],
                "kernel_js_mean": row["kernel_js_mean"],
                "kernel_wj_mean": row["kernel_wj_mean"],
                "top_child_full_match_mass": row["top_child_full_match_mass"],
                "top_child_common_match_mass": row["top_child_common_match_mass"],
                "comparable_weight_fraction": row["comparable_weight_fraction"],
                "anchor_terminal_fraction": row["anchor_terminal_fraction"],
            }
            for row in rows
        ],
    }
    manifest_payload = {
        "run_slug": run_slug,
        "generated_at": generated_at,
        "script": "hsi_v2_phase2_kernel_common_support.py",
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
            "anchor_run_dir": anchor_run["_run_dir"],
            "window_bits": window_bits,
            "frozen_source_run_dirs": (
                sorted(preloaded_sources.keys())
                if preloaded_sources is not None
                else []
            ),
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


def build_observed_specs(
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
    low_selection: dict,
    high_selection: dict,
    anchor_offset_bits: int,
    window_bits: int,
    kernel_mode: str,
) -> dict:
    return {
        "variant": spec["variant"],
        "variant_role": classify_variant_role(spec["variant"]),
        "source_variant_internal": result["variant"],
        "source_label": spec["source_label"],
        "source_kind": spec["source_kind"],
        "null_model": spec["null_model"],
        "null_seed": spec["null_seed"],
        "anchor_variant": spec["anchor_variant"],
        "anchor_kind": spec["anchor_kind"],
        "anchor_run_dir": spec["anchor_run"]["_run_dir"],
        "run_dir": spec["run"]["_run_dir"],
        "anchor_offset_bits": anchor_offset_bits,
        "window_bits": window_bits,
        "kernel_mode": kernel_mode,
        "low_scale": result["low_scale"],
        "high_scale": result["high_scale"],
        "pattern_selection": result["pattern_selection"],
        "selected_parent_count": result["selected_parent_count"],
        "selected_child_count": result["selected_child_count"],
        "anchor_terminal_fraction": result["anchor_terminal_fraction"],
        "candidate_terminal_fraction": result["candidate_terminal_fraction"],
        "anchor_local_parent_fraction": result.get("anchor_local_parent_fraction"),
        "candidate_local_parent_fraction": result.get("candidate_local_parent_fraction"),
        "empty_common_support_fraction": result["empty_common_support_fraction"],
        "comparable_weight_fraction": result["comparable_weight_fraction"],
        "support_coverage_anchor_mean": result["support_coverage_anchor_mean"],
        "anchor_dropout_mean": result["anchor_dropout_mean"],
        "candidate_common_mass_mean": result["candidate_common_mass_mean"],
        "candidate_spillover_mean": result["candidate_spillover_mean"],
        "support_jaccard_mean": result["support_jaccard_mean"],
        "kernel_js_mean": result["kernel_js_mean"],
        "kernel_wj_mean": result["kernel_wj_mean"],
        "top_child_full_match_mass": result["top_child_full_match_mass"],
        "top_child_common_match_mass": result["top_child_common_match_mass"],
        "selection_source_variant": low_selection["source_variant"],
        "selection_source_run_dir": low_selection["source_run_dir"],
        "selection_anchor_kind": low_selection["anchor_kind"],
        "kernel_result": result,
        "high_selection_source_variant": high_selection["source_variant"],
        "high_selection_source_run_dir": high_selection["source_run_dir"],
    }


def build_selection(
    *,
    args,
    anchor_variant: str,
    variants: list[str],
    family_inferred: bool,
    null_models: list[str],
    matched_lz_seeds: list[int],
    anchor_run: dict,
    low_selection: dict,
    high_selection: dict,
    window_bits: int,
) -> dict:
    anchor_config = anchor_run["dataset"]["config"]
    return {
        "stage": args.stage,
        "anchor_variant": anchor_variant,
        "variants": variants,
        "iteration": anchor_config.get("iteration"),
        "segment_bits": anchor_config.get("segment_bits"),
        "num_segments": anchor_config.get("num_segments"),
        "segment_offset_bits": anchor_config.get("segment_offset_bits", 0),
        "phase1_scales": list(anchor_config.get("scales", [])),
        "phase1_policies": list(anchor_config.get("policies", [])),
        "low_scale": args.low_scale,
        "high_scale": args.high_scale,
        "top_patterns": args.top_patterns,
        "pattern_selection": args.pattern_selection,
        "kernel_mode": args.kernel_mode,
        "anchor_offset_bits": args.anchor_offset_bits,
        "window_bits": window_bits,
        "null_models": null_models,
        "matched_lz_seeds": matched_lz_seeds,
        "family_inferred": family_inferred,
        "selection_source_variant": low_selection["source_variant"],
        "selection_source_run_dir": low_selection["source_run_dir"],
        "selected_parent_count": len(low_selection["selected_patterns"]),
        "selected_child_count": len(high_selection["selected_patterns"]),
        "structural_only": args.kernel_mode == "structural",
    }


def build_run_slug(selection: dict, timestamp: str) -> str:
    variants = "-".join(selection["variants"])
    return (
        f"phase2-kernel-common-support__stage-{selection['stage']}"
        f"__anchor-{selection['anchor_variant']}"
        f"__m-{selection['low_scale']}-{selection['high_scale']}"
        f"__sel-{selection['pattern_selection']}__top-{selection['top_patterns']}"
        f"__late-{compact_int(selection['anchor_offset_bits'])}"
        f"__var-{variants}__{timestamp}"
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
    for group in grouped.values():
        first = group[0]
        summaries.append(
            {
                "variant": first["variant"],
                "source_label": first["source_label"],
                "source_kind": first["source_kind"],
                "null_model": first.get("null_model"),
                "null_seed": first.get("null_seed"),
                "row_count": len(group),
                "anchor_local_parent_fraction_avg": mean_present(
                    group,
                    "anchor_local_parent_fraction",
                ),
                "support_coverage_anchor_mean_avg": mean_of(
                    group,
                    "support_coverage_anchor_mean",
                ),
                "support_coverage_anchor_mean_min": min_of(
                    group,
                    "support_coverage_anchor_mean",
                ),
                "support_coverage_anchor_mean_max": max_of(
                    group,
                    "support_coverage_anchor_mean",
                ),
                "candidate_spillover_mean_avg": mean_of(
                    group,
                    "candidate_spillover_mean",
                ),
                "candidate_spillover_mean_min": min_of(
                    group,
                    "candidate_spillover_mean",
                ),
                "candidate_spillover_mean_max": max_of(
                    group,
                    "candidate_spillover_mean",
                ),
                "support_jaccard_mean_avg": mean_of(group, "support_jaccard_mean"),
                "kernel_js_mean_avg": mean_present(group, "kernel_js_mean"),
                "kernel_js_mean_min": min_present(group, "kernel_js_mean"),
                "kernel_js_mean_max": max_present(group, "kernel_js_mean"),
                "kernel_wj_mean_avg": mean_present(group, "kernel_wj_mean"),
                "kernel_wj_mean_min": min_present(group, "kernel_wj_mean"),
                "kernel_wj_mean_max": max_present(group, "kernel_wj_mean"),
                "top_child_full_match_mass_avg": mean_of(
                    group,
                    "top_child_full_match_mass",
                ),
                "top_child_common_match_mass_avg": mean_present(
                    group,
                    "top_child_common_match_mass",
                ),
                "comparable_weight_fraction_avg": mean_of(
                    group,
                    "comparable_weight_fraction",
                ),
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
        "Phase 2 common-support kernel-divergence scaffold",
        "-" * 166,
        f"{'variant':<{variant_width}} {'source':<{source_width}} {'par_a':>8}{'cov_a':>8}{'spill_x':>10}{'sup_j':>9}{'js_cs':>9}{'wj_cs':>9}{'top_f':>9}{'top_c':>9}{'comp_w':>9}{'term_a':>9}",
    ]
    for row in sorted(
        rows,
        key=lambda item: (
            item["variant"],
            source_kind_sort_key(item["source_kind"], item.get("null_model")),
            item.get("null_seed") or -1,
        ),
    ):
        lines.append(
            f"{truncate_label(row['variant'], variant_width):<{variant_width}} "
            f"{truncate_label(row['source_label'], source_width):<{source_width}} "
            f"{fmt(row.get('anchor_local_parent_fraction')):>8}"
            f"{row['support_coverage_anchor_mean']:>8.4f}"
            f"{row['candidate_spillover_mean']:>10.4f}"
            f"{row['support_jaccard_mean']:>9.4f}"
            f"{fmt(row['kernel_js_mean']):>9}"
            f"{fmt(row['kernel_wj_mean']):>9}"
            f"{row['top_child_full_match_mass']:>9.4f}"
            f"{fmt(row['top_child_common_match_mass']):>9}"
            f"{row['comparable_weight_fraction']:>9.4f}"
            f"{row['anchor_terminal_fraction']:>9.4f}"
        )
    return "\n".join(lines)


def render_markdown_report(
    selection: dict,
    grouped_summary: list[dict],
    rows: list[dict],
    artifacts: list[dict],
) -> str:
    lines = [
        "# Phase 2 Common-Support Kernel-Divergence Scaffold",
        "",
        "## Selection",
        "",
        f"- Stage: {selection['stage']}",
        f"- Anchor variant: {selection['anchor_variant']}",
        f"- Variants: {', '.join(selection['variants'])}",
        f"- Segment protocol: {selection['num_segments']} x {selection['segment_bits']}",
        f"- Phase 1 scales: {', '.join(str(value) for value in selection['phase1_scales'])}",
        f"- Phase 1 policies: {', '.join(selection['phase1_policies'])}",
        f"- Low/high scales: {selection['low_scale']} -> {selection['high_scale']}",
        f"- Pattern selection: {selection['pattern_selection']}",
        f"- Kernel mode: {selection['kernel_mode']}",
        f"- Top patterns: {selection['top_patterns']}",
        f"- Late anchor offset (metadata): {selection['anchor_offset_bits']}",
        f"- Window bits: {selection['window_bits']}",
        f"- Structural only: {selection['structural_only']}",
        f"- Selection source variant: {selection['selection_source_variant']}",
        f"- Selected parents: {selection['selected_parent_count']}",
        f"- Selected children: {selection['selected_child_count']}",
        f"- Family inferred: {selection['family_inferred']}",
        "",
        "## Frozen Evaluation Runs",
        "",
    ]
    for artifact in artifacts:
        seed = (
            f" | seed={artifact['null_seed']}"
            if artifact.get("null_seed") is not None
            else ""
        )
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
            "| Variant | Source | Rows | Par_a avg | Cov_a avg | Cov_a min | Cov_a max | Spill_x avg | JS_cs avg | JS_cs min | JS_cs max | WJ_cs avg | Top_full avg | Top_common avg | Comp_w avg |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in grouped_summary:
        lines.append(
            f"| {item['variant']} | {item['source_label']} | {item['row_count']} | "
            f"{fmt(item.get('anchor_local_parent_fraction_avg'))} | "
            f"{fmt(item['support_coverage_anchor_mean_avg'])} | {fmt(item['support_coverage_anchor_mean_min'])} | {fmt(item['support_coverage_anchor_mean_max'])} | "
            f"{fmt(item['candidate_spillover_mean_avg'])} | "
            f"{fmt(item['kernel_js_mean_avg'])} | {fmt(item['kernel_js_mean_min'])} | {fmt(item['kernel_js_mean_max'])} | "
            f"{fmt(item['kernel_wj_mean_avg'])} | "
            f"{fmt(item['top_child_full_match_mass_avg'])} | {fmt(item['top_child_common_match_mass_avg'])} | "
            f"{fmt(item['comparable_weight_fraction_avg'])} |"
        )

    lines.extend(
        [
            "",
            "## Readout",
            "",
            "| Variant | Source | Par_a | Cov_a | Spill_x | Sup_j | JS_cs | WJ_cs | Top_full | Top_common | Comp_w | Term_a | Empty_common |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(
        rows,
        key=lambda item: (
            item["variant"],
            source_kind_sort_key(item["source_kind"], item.get("null_model")),
            item.get("null_seed") or -1,
        ),
    ):
        lines.append(
            f"| {row['variant']} | {row['source_label']} | "
            f"{fmt(row.get('anchor_local_parent_fraction'))} | "
            f"{fmt(row['support_coverage_anchor_mean'])} | {fmt(row['candidate_spillover_mean'])} | {fmt(row['support_jaccard_mean'])} | "
            f"{fmt(row['kernel_js_mean'])} | {fmt(row['kernel_wj_mean'])} | "
            f"{fmt(row['top_child_full_match_mass'])} | {fmt(row['top_child_common_match_mass'])} | "
            f"{fmt(row['comparable_weight_fraction'])} | {fmt(row['anchor_terminal_fraction'])} | "
            f"{fmt(row['empty_common_support_fraction'])} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This scaffold compares transport kernels directly on shared child support under a frozen anchor-selected parent/child universe.",
            "- In `window-local` mode the kernel weights are rebuilt from child-anchored realized events inside the late window, counting each parent-child pair at most once per child occurrence.",
            "- In `window-local` mode the parent weighting is driven by the anchor realized event mass on the frozen child universe, not by the raw low-scale parent count.",
            "- In `structural` mode the kernel weights come from the global Phase 1 fibers and the late anchor remains metadata only.",
            "- Coverage and spillover are reported separately from the shared-support divergences so support dropout is not hidden by renormalization.",
            "- No flow metric is emitted yet because the child-pattern axis still lacks a canonical order.",
        ]
    )
    return "\n".join(lines)


def mean_present(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def min_present(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return float(min(values))


def max_present(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return float(max(values))


if __name__ == "__main__":
    raise SystemExit(main())
