#!/usr/bin/env python3
"""
HSI v2 Phase 2 Parent Survival Band Stability

Observed-only band sweep over the corrected N2-03 window-local kernel,
followed by the hardened N2-04 parent-survival readout.
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
    case_note,
    count_unique_run_dirs,
    infer_segment_protocol,
    parse_int_list,
    phase_print,
    select_observed_runs,
    window_note,
)
from v2.common.cli import parse_variants, resolve_dir
from v2.common.naming import compact_int
from v2.phase1.tower import parse_scales
from v2.phase2.kernel_common_support import (
    compute_window_local_common_support_kernel_result,
)
from v2.phase2.parent_survival import (
    build_parent_survival_rows,
    write_parent_patterns_csv,
    write_parent_survival_csv,
)
from v2.phase2.parent_survival_band import (
    render_parent_survival_band_console_summary,
    render_parent_survival_band_report,
    summarize_parent_survival_band_rows,
    write_parent_survival_band_summary_csv,
)
from v2.phase2.return_lag import (
    prepare_frozen_source_cache,
    prepare_pattern_selection,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a short-band parent-survival stability sweep on the corrected Phase 2 window-local kernel."
    )
    parser.add_argument(
        "--phase1-dir",
        type=str,
        default="results/hsi_v2/phase1_high_scales",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/hsi_v2/phase2/parent_survival_band",
    )
    parser.add_argument("--variants", type=str, default="E,B")
    parser.add_argument("--anchor-variant", type=str, default="E")
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
        "--offsets",
        type=str,
        default="714000000,717000000,720000000,723000000",
        help="Comma-separated list of late-band window start offsets in bits.",
    )
    parser.add_argument(
        "--window-bits",
        type=int,
        default=None,
        help="Late-window size in bits. Defaults to the selected Phase 1 segment protocol span.",
    )
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

    if anchor_variant not in variants:
        variants = [anchor_variant] + [variant for variant in variants if variant != anchor_variant]
    if args.low_scale <= 0 or args.high_scale <= 0 or args.low_scale >= args.high_scale:
        parser.error("--low-scale and --high-scale must satisfy 1 <= low < high.")
    if args.high_scale > 64:
        parser.error("--high-scale must be in the range 1..64.")
    if args.top_patterns <= 0:
        parser.error("--top-patterns must be positive.")

    try:
        offsets = parse_int_list(args.offsets, label="--offsets", allow_zero=True)
    except ValueError as exc:
        parser.error(str(exc))
    if any(offset < 0 for offset in offsets):
        parser.error("--offsets must contain non-negative integers.")

    phase1_dir = resolve_dir(args.phase1_dir, anchor_file=__file__)
    output_dir = resolve_dir(args.output_dir, anchor_file=__file__)
    output_dir.mkdir(parents=True, exist_ok=True)

    phase1_scales = parse_scales(args.scales) if args.scales.strip() else None
    phase1_policies = [
        item.strip()
        for item in args.phase1_policies.split(",")
        if item.strip()
    ] or None

    phase_print(
        "Preparing parent-survival band stability",
        (
            f"anchor={anchor_variant} | variants={','.join(variants)} | "
            f"m={args.low_scale}->{args.high_scale} | sel={args.pattern_selection} | "
            f"offsets={','.join(str(value) for value in offsets)}"
        ),
        quiet=args.quiet,
    )

    runs = discover_runs_or_die(phase1_dir, parser)
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
    anchor_run = observed_by_variant.get(anchor_variant)
    if anchor_run is None:
        parser.error(f"Missing observed anchor run for variant {anchor_variant}.")

    segment_bits, num_segments = infer_segment_protocol(observed_runs)
    window_bits = (
        int(args.window_bits)
        if args.window_bits is not None
        else segment_bits * num_segments
    )
    if window_bits < args.high_scale:
        parser.error("--window-bits must be at least as large as --high-scale.")

    evaluation_specs = build_observed_specs(observed_by_variant, anchor_run=anchor_run, variants=variants)

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

    frozen_total_bits = max(offsets) + window_bits
    phase_print(
        "Freezing source realizations for the local band sweep",
        f"runs={count_unique_run_dirs(evaluation_specs)} | required_bits={frozen_total_bits}",
        quiet=args.quiet,
    )
    preloaded_sources = prepare_frozen_source_cache(
        [spec["run"] for spec in evaluation_specs] + [anchor_run],
        required_bits=frozen_total_bits,
        show_progress=not args.quiet,
    )

    phase_print(
        "Running parent-survival band stability",
        (
            f"cases={len(evaluation_specs)} | windows={len(offsets)} | "
            f"parents={len(low_selection['selected_patterns'])} | children={len(high_selection['selected_patterns'])}"
        ),
        quiet=args.quiet,
    )

    kernel_rows = []
    artifacts = []
    for case_index, spec in enumerate(evaluation_specs, start=1):
        case_note(
            (
                f"{case_index}/{len(evaluation_specs)} | variant={spec['variant']} | "
                f"source={spec['source_label']} | anchor={spec['anchor_variant']}"
            ),
            quiet=args.quiet,
        )
        anchor_bits = preloaded_sources[str(Path(spec["anchor_run"]["_run_dir"]).resolve())]["bits"]
        candidate_bits = preloaded_sources[str(Path(spec["run"]["_run_dir"]).resolve())]["bits"]

        for offset_index, offset_bits in enumerate(offsets, start=1):
            window_note(
                (
                    f"case {case_index}/{len(evaluation_specs)} | "
                    f"window {offset_index}/{len(offsets)} | offset={offset_bits} | "
                    f"variant={spec['variant']} | source={spec['source_label']}"
                ),
                quiet=args.quiet,
            )
            window_start = offset_bits
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
            kernel_rows.append(
                build_kernel_row(
                    result=result,
                    spec=spec,
                    offset_bits=offset_bits,
                    window_bits=window_bits,
                    source_freeze_mode=artifact_source_info(
                        preloaded_sources,
                        spec["run"]["_run_dir"],
                        "source_freeze_mode",
                    ),
                    frozen_total_bits=artifact_source_info(
                        preloaded_sources,
                        spec["run"]["_run_dir"],
                        "frozen_total_bits",
                    ),
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

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    selection = build_selection(
        args=args,
        anchor_variant=anchor_variant,
        variants=variants,
        offsets=offsets,
        segment_bits=segment_bits,
        num_segments=num_segments,
        window_bits=window_bits,
        frozen_total_bits=frozen_total_bits,
        family_inferred=family_inferred,
        low_selection=low_selection,
        high_selection=high_selection,
    )
    run_slug = build_run_slug(selection, timestamp)
    run_dir = output_dir / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = run_dir / "dataset.json"
    summary_path = run_dir / "summary.json"
    report_path = run_dir / "report.md"
    table_csv_path = run_dir / "table.csv"
    patterns_csv_path = run_dir / "patterns.csv"
    band_csv_path = run_dir / "band_summary.csv"
    manifest_path = run_dir / "manifest.json"
    generated_at = datetime.now().isoformat(timespec="seconds")

    survival_payload = {
        "selection": selection,
        "_dataset_path": str(dataset_path),
        "rows": kernel_rows,
    }
    survival_rows, pattern_rows = build_parent_survival_rows(survival_payload)
    grouped_summary = summarize_parent_survival_band_rows(survival_rows, pattern_rows)

    dataset_payload = {
        "stage": "phase2_parent_survival_band",
        "generated_at": generated_at,
        "selection": selection,
        "kernel_rows": kernel_rows,
        "rows": survival_rows,
        "pattern_rows": pattern_rows,
        "grouped_summary": grouped_summary,
        "artifacts": artifacts,
        "notes": [
            "The observed anchor selection is frozen once and reused across the full late-band sweep.",
            "Common-support kernels are rebuilt window-locally from realized child-anchored events at each offset.",
            "The hardened parent-survival taxonomy separates dead parents from dark continuation.",
            "Band stability is read on exact/dead/dark counts, mass fractions, and pattern-identity persistence across offsets.",
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
                "anchor_offset_bits": row["anchor_offset_bits"],
                "exact_parent_count": row["exact_parent_count"],
                "dead_parent_count": row["dead_parent_count"],
                "dark_parent_count": row["dark_parent_count"],
                "partial_parent_count": row["partial_parent_count"],
                "exact_mass_fraction": row["exact_mass_fraction"],
                "dead_mass_fraction": row["dead_mass_fraction"],
                "dark_mass_fraction": row["dark_mass_fraction"],
                "kernel_wj_mean": row["kernel_wj_mean"],
            }
            for row in survival_rows
        ],
    }
    manifest_payload = {
        "run_slug": run_slug,
        "generated_at": generated_at,
        "script": "hsi_v2_phase2_parent_survival_band.py",
        "cwd": str(Path.cwd()),
        "outputs": {
            "dataset": str(dataset_path),
            "summary": str(summary_path),
            "report": str(report_path),
            "table_csv": str(table_csv_path),
            "patterns_csv": str(patterns_csv_path),
            "band_summary_csv": str(band_csv_path),
            "manifest": str(manifest_path),
        },
        "inputs": {
            "phase1_dir": str(phase1_dir),
            "observed_phase1_runs": [run["_run_dir"] for run in observed_runs],
            "evaluated_phase1_runs": [artifact["run_dir"] for artifact in artifacts],
            "anchor_run_dir": anchor_run["_run_dir"],
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
        handle.write(
            render_parent_survival_band_report(selection, grouped_summary, survival_rows) + "\n"
        )
    write_parent_survival_csv(survival_rows, table_csv_path)
    write_parent_patterns_csv(pattern_rows, patterns_csv_path)
    write_parent_survival_band_summary_csv(grouped_summary, band_csv_path)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, indent=2)

    if not args.quiet:
        if family_inferred:
            print("Using the latest coherent Phase 1 family inferred from the newest observed run.")
            print("")
        print(render_parent_survival_band_console_summary(grouped_summary))
        print("")
        print(f"Saved dataset to: {dataset_path}")
        print(f"Saved summary to: {summary_path}")
        print(f"Saved report to: {report_path}")
        print(f"Saved CSV table to: {table_csv_path}")
        print(f"Saved patterns CSV to: {patterns_csv_path}")
        print(f"Saved band summary CSV to: {band_csv_path}")
        print(f"Saved manifest to: {manifest_path}")

    return 0


def discover_runs_or_die(phase1_dir: Path, parser: argparse.ArgumentParser) -> list[dict]:
    from v2.phase2.null_pressure import discover_phase1_runs_recursive

    runs = discover_phase1_runs_recursive(phase1_dir)
    if not runs:
        parser.error(f"No valid Phase 1 runs found in {phase1_dir}")
    return runs


def build_observed_specs(
    observed_by_variant: dict[str, dict],
    *,
    anchor_run: dict,
    variants: list[str],
) -> list[dict]:
    anchor_variant = str(anchor_run["dataset"]["config"]["variant"]).upper()
    specs = []
    for variant in variants:
        run = observed_by_variant.get(variant)
        if run is None:
            continue
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


def build_kernel_row(
    *,
    result: dict,
    spec: dict,
    offset_bits: int,
    window_bits: int,
    source_freeze_mode: str | None,
    frozen_total_bits: int | None,
) -> dict:
    return {
        "variant": result["variant"],
        "source_label": spec["source_label"],
        "source_kind": spec["source_kind"],
        "null_model": spec["null_model"],
        "null_seed": spec["null_seed"],
        "anchor_variant": spec["anchor_variant"],
        "anchor_kind": spec["anchor_kind"],
        "anchor_run_dir": result["anchor_run_dir"],
        "run_dir": result["candidate_run_dir"],
        "anchor_offset_bits": offset_bits,
        "window_bits": window_bits,
        "low_scale": result["low_scale"],
        "high_scale": result["high_scale"],
        "pattern_selection": result["pattern_selection"],
        "kernel_mode": "window-local",
        "selected_parent_count": result["selected_parent_count"],
        "support_coverage_anchor_mean": result["support_coverage_anchor_mean"],
        "kernel_wj_mean": result["kernel_wj_mean"],
        "top_child_full_match_mass": result["top_child_full_match_mass"],
        "source_freeze_mode": source_freeze_mode,
        "frozen_total_bits": frozen_total_bits,
        "kernel_result": result,
    }


def build_selection(
    *,
    args,
    anchor_variant: str,
    variants: list[str],
    offsets: list[int],
    segment_bits: int,
    num_segments: int,
    window_bits: int,
    frozen_total_bits: int,
    family_inferred: bool,
    low_selection: dict,
    high_selection: dict,
) -> dict:
    return {
        "stage": "observed",
        "anchor_variant": anchor_variant,
        "variants": variants,
        "iteration": args.iteration,
        "segment_bits": segment_bits,
        "num_segments": num_segments,
        "offsets": offsets,
        "source_freeze_mode": "sweep-global-frozen",
        "frozen_total_bits": frozen_total_bits,
        "window_bits": window_bits,
        "low_scale": args.low_scale,
        "high_scale": args.high_scale,
        "top_patterns": args.top_patterns,
        "pattern_selection": args.pattern_selection,
        "family_inferred": family_inferred,
        "kernel_mode": "window-local",
        "selection_source_variant": low_selection.get("selection_source_variant"),
        "selection_source_run_dir": low_selection.get("selection_source_run_dir"),
        "selected_low_pattern_count": len(low_selection["selected_patterns"]),
        "selected_high_pattern_count": len(high_selection["selected_patterns"]),
    }


def build_run_slug(selection: dict, timestamp: str) -> str:
    offsets = selection["offsets"]
    if len(offsets) == 1:
        offset_part = f"off-{compact_int(offsets[0])}"
    else:
        offset_part = f"off-{compact_int(min(offsets))}-plus-{len(offsets)}"
    variants = "-".join(selection["variants"])
    return (
        f"phase2-parent-survival-band__anchor-{selection['anchor_variant']}"
        f"__m-{selection['low_scale']}-{selection['high_scale']}"
        f"__sel-{selection['pattern_selection']}__top-{selection['top_patterns']}"
        f"__{offset_part}__var-{variants}__{timestamp}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
