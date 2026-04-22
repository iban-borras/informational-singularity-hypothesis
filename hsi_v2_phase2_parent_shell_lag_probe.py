#!/usr/bin/env python3
"""
HSI v2 Phase 2 Parent Shell Lag Probe

Independent diagnostic for the low-scale parent shell defined from the anchor
alone, used to estimate temporal lag before any lag-aware survival comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hsi_v2_phase2_transport_defect_strict import (
    infer_segment_protocol,
    parse_int_list,
    phase_print,
    select_observed_runs,
)
from v2.common.cli import parse_variants, resolve_dir
from v2.common.naming import compact_int
from v2.phase1.tower import parse_scales
from v2.phase2.parent_shell_lag import (
    build_window_pattern_counts,
    render_parent_shell_lag_console_summary,
    render_parent_shell_lag_report,
    summarize_shell_lag_probe,
    write_parent_shell_lag_csv,
)
from v2.phase2.return_lag import (
    prepare_frozen_source_cache,
    prepare_pattern_selection,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe temporal lag of the anchor-defined parent shell before lag-aware survival analysis."
    )
    parser.add_argument(
        "--phase1-dir",
        type=str,
        default="results/hsi_v2/phase1_high_scales",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/hsi_v2/phase2/parent_shell_lag_probe",
    )
    parser.add_argument("--variants", type=str, default="E,B")
    parser.add_argument("--anchor-variant", type=str, default="E")
    parser.add_argument("--iteration", type=int, default=None)
    parser.add_argument("--segment-bits", type=int, default=None)
    parser.add_argument("--num-segments", type=int, default=None)
    parser.add_argument("--segment-offset-bits", type=int, default=None)
    parser.add_argument("--scales", type=str, default="")
    parser.add_argument("--phase1-policies", type=str, default="")
    parser.add_argument("--pattern-scale", type=int, default=40)
    parser.add_argument("--top-patterns", type=int, default=64)
    parser.add_argument(
        "--pattern-selection",
        type=str,
        default="bridge-linked",
        choices=("top", "rare-stable", "bridge-linked"),
    )
    parser.add_argument(
        "--definition-offsets",
        type=str,
        default="714000000,714500000,715000000,715500000,716000000,716500000,717000000,717500000,718000000,718500000,719000000,719500000,720000000,720500000,721000000,721500000,722000000,722500000,723000000",
    )
    parser.add_argument("--window-bits", type=int, default=None)
    parser.add_argument("--scan-start-bits", type=int, default=None)
    parser.add_argument("--scan-stop-bits", type=int, default=None)
    parser.add_argument("--scan-step-bits", type=int, default=500000)
    parser.add_argument("--scan-forward-bits", type=int, default=12000000)
    parser.add_argument("--scan-backward-bits", type=int, default=0)
    parser.add_argument("--no-family-inference", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    variants = parse_variants(args.variants)
    if len(variants) != 2:
        parser.error("--variants must contain exactly two observed variants.")
    anchor_variants = parse_variants(args.anchor_variant)
    if len(anchor_variants) != 1:
        parser.error("--anchor-variant must contain exactly one variant.")
    anchor_variant = anchor_variants[0]
    if anchor_variant not in variants:
        parser.error("--anchor-variant must be included in --variants.")
    candidate_variant = next(variant for variant in variants if variant != anchor_variant)

    if args.pattern_scale <= 0 or args.pattern_scale > 64:
        parser.error("--pattern-scale must be in the range 1..64.")
    if args.top_patterns <= 0:
        parser.error("--top-patterns must be positive.")
    if args.scan_step_bits <= 0:
        parser.error("--scan-step-bits must be positive.")
    if args.scan_forward_bits < 0 or args.scan_backward_bits < 0:
        parser.error("--scan-forward-bits and --scan-backward-bits must be non-negative.")

    try:
        definition_offsets = parse_int_list(
            args.definition_offsets,
            label="--definition-offsets",
            allow_zero=True,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if any(offset < 0 for offset in definition_offsets):
        parser.error("--definition-offsets must contain non-negative integers.")

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
        "Preparing parent shell lag probe",
        (
            f"anchor={anchor_variant} | candidate={candidate_variant} | "
            f"m={args.pattern_scale} | sel={args.pattern_selection}"
        ),
        quiet=args.quiet,
    )

    from v2.phase2.null_pressure import discover_phase1_runs_recursive

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
        low_scale=args.pattern_scale,
        high_scale=args.pattern_scale,
        no_family_inference=args.no_family_inference,
    )
    observed_by_variant = {
        str(run["dataset"]["config"]["variant"]).upper(): run
        for run in observed_runs
    }
    anchor_run = observed_by_variant.get(anchor_variant)
    candidate_run = observed_by_variant.get(candidate_variant)
    if anchor_run is None or candidate_run is None:
        parser.error("Missing observed anchor or candidate run.")

    segment_bits, num_segments = infer_segment_protocol(observed_runs)
    window_bits = (
        int(args.window_bits)
        if args.window_bits is not None
        else segment_bits * num_segments
    )
    if window_bits < args.pattern_scale:
        parser.error("--window-bits must be at least as large as --pattern-scale.")

    scan_start_bits = (
        int(args.scan_start_bits)
        if args.scan_start_bits is not None
        else min(definition_offsets) - int(args.scan_backward_bits)
    )
    scan_stop_bits = (
        int(args.scan_stop_bits)
        if args.scan_stop_bits is not None
        else max(definition_offsets) + int(args.scan_forward_bits)
    )
    if scan_start_bits < 0:
        parser.error("The derived scan start offset is negative.")
    if scan_stop_bits < scan_start_bits:
        parser.error("--scan-stop-bits must be >= --scan-start-bits.")

    scan_offsets = list(range(scan_start_bits, scan_stop_bits + 1, int(args.scan_step_bits)))
    required_bits = max(scan_offsets) + window_bits
    selection_info = prepare_pattern_selection(
        anchor_run,
        pattern_scale=args.pattern_scale,
        top_patterns=args.top_patterns,
        pattern_selection=args.pattern_selection,
    )

    phase_print(
        "Freezing source realizations for lag probe",
        f"runs=2 | required_bits={required_bits}",
        quiet=args.quiet,
    )
    preloaded_sources = prepare_frozen_source_cache(
        [anchor_run, candidate_run],
        required_bits=required_bits,
        show_progress=not args.quiet,
    )
    anchor_bits = preloaded_sources[str(Path(anchor_run["_run_dir"]).resolve())]["bits"]
    candidate_bits = preloaded_sources[str(Path(candidate_run["_run_dir"]).resolve())]["bits"]

    phase_print(
        "Scanning parent shell profiles",
        f"definition_windows={len(definition_offsets)} | scan_windows={len(scan_offsets)}",
        quiet=args.quiet,
    )
    anchor_counts = build_window_pattern_counts(
        anchor_bits,
        scan_offsets,
        window_bits=window_bits,
        pattern_scale=args.pattern_scale,
        selected_patterns=selection_info["selected_patterns"],
    )
    candidate_counts = build_window_pattern_counts(
        candidate_bits,
        scan_offsets,
        window_bits=window_bits,
        pattern_scale=args.pattern_scale,
        selected_patterns=selection_info["selected_patterns"],
    )
    summary, rows = summarize_shell_lag_probe(
        anchor_counts=anchor_counts,
        candidate_counts=candidate_counts,
        definition_offsets=definition_offsets,
        scan_offsets=scan_offsets,
        scan_step_bits=args.scan_step_bits,
    )

    generated_at = datetime.now().isoformat(timespec="seconds")
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    selection = {
        "anchor_variant": anchor_variant,
        "candidate_variant": candidate_variant,
        "variants": variants,
        "iteration": args.iteration,
        "segment_bits": segment_bits,
        "num_segments": num_segments,
        "definition_offsets": definition_offsets,
        "scan_start_bits": scan_start_bits,
        "scan_stop_bits": scan_stop_bits,
        "scan_step_bits": int(args.scan_step_bits),
        "source_freeze_mode": "sweep-global-frozen",
        "frozen_total_bits": required_bits,
        "pattern_scale": args.pattern_scale,
        "top_patterns": args.top_patterns,
        "pattern_selection": args.pattern_selection,
        "window_bits": window_bits,
        "family_inferred": family_inferred,
    }
    run_slug = build_run_slug(selection, timestamp)
    run_dir = output_dir / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = run_dir / "dataset.json"
    summary_path = run_dir / "summary.json"
    report_path = run_dir / "report.md"
    table_csv_path = run_dir / "table.csv"
    manifest_path = run_dir / "manifest.json"

    dataset_payload = {
        "stage": "phase2_parent_shell_lag_probe",
        "generated_at": generated_at,
        "selection": selection,
        "summary": summary,
        "rows": rows,
        "notes": [
            "The anchor shell is defined from the anchor alone as union(anchor-active) minus intersection(anchor-active) over the definition offsets.",
            "The candidate is then scanned independently over a wider offset range before any lag-aware survival object is built.",
            "The probe reports multiple lag heuristics instead of a single forced estimate.",
        ],
    }
    summary_payload = {
        "generated_at": generated_at,
        "selection": selection,
        "summary": summary,
    }
    manifest_payload = {
        "run_slug": run_slug,
        "generated_at": generated_at,
        "script": "hsi_v2_phase2_parent_shell_lag_probe.py",
        "cwd": str(Path.cwd()),
        "outputs": {
            "dataset": str(dataset_path),
            "summary": str(summary_path),
            "report": str(report_path),
            "table_csv": str(table_csv_path),
            "manifest": str(manifest_path),
        },
        "inputs": {
            "phase1_dir": str(phase1_dir),
            "anchor_run_dir": anchor_run["_run_dir"],
            "candidate_run_dir": candidate_run["_run_dir"],
            "frozen_total_bits": required_bits,
        },
        "arguments": vars(args),
    }

    with open(dataset_path, "w", encoding="utf-8") as handle:
        json.dump(dataset_payload, handle, indent=2)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(render_parent_shell_lag_report(selection, summary, rows) + "\n")
    write_parent_shell_lag_csv(rows, table_csv_path)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, indent=2)

    if not args.quiet:
        if family_inferred:
            print("Using the latest coherent Phase 1 family inferred from the newest observed run.")
            print("")
        print(render_parent_shell_lag_console_summary(selection, summary))
        print("")
        print(f"Saved dataset to: {dataset_path}")
        print(f"Saved summary to: {summary_path}")
        print(f"Saved report to: {report_path}")
        print(f"Saved CSV table to: {table_csv_path}")
        print(f"Saved manifest to: {manifest_path}")

    return 0


def build_run_slug(selection: dict, timestamp: str) -> str:
    return (
        f"phase2-parent-shell-lag-probe__anchor-{selection['anchor_variant']}"
        f"__cand-{selection['candidate_variant']}__m-{selection['pattern_scale']}"
        f"__sel-{selection['pattern_selection']}__top-{selection['top_patterns']}"
        f"__scan-{compact_int(selection['scan_start_bits'])}-to-{compact_int(selection['scan_stop_bits'])}"
        f"__step-{compact_int(selection['scan_step_bits'])}__{timestamp}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
