#!/usr/bin/env python3
"""
HSI v2 Phase 2 Return-Lag Spectrum Pilot

First Gate 2 pilot aimed explicitly at longer-range order:
- selects top patterns from a stored Phase 1 pattern space
- reconstructs the analyzed observable slice for observed or null-surrogate runs
- measures the return-distance spectrum of those patterns inside each segment
- writes reproducible artifacts, manifests, and a compact markdown summary
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from v2.common.cli import parse_variants, resolve_dir
from v2.common.naming import compact_int
from v2.phase1.report import (
    discover_phase1_runs,
    filter_runs,
    infer_family_from_latest_run,
    select_latest_per_variant,
)
from v2.phase1.tower import parse_scales
from v2.phase2.return_lag import (
    build_return_lag_rows,
    render_console_summary,
    render_markdown_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the HSI v2 Gate 2 return-lag spectrum pilot from existing Phase 1 artifacts."
    )
    parser.add_argument("--phase1-dir", type=str, default="results/hsi_v2/phase1")
    parser.add_argument("--output-dir", type=str, default="results/hsi_v2/phase2")
    parser.add_argument("--variants", type=str, default="B")
    parser.add_argument("--iteration", type=int, default=None)
    parser.add_argument("--segment-bits", type=int, default=None)
    parser.add_argument("--num-segments", type=int, default=None)
    parser.add_argument("--segment-offset-bits", type=int, default=None)
    parser.add_argument("--scales", type=str, default="")
    parser.add_argument("--phase1-policies", type=str, default="")
    parser.add_argument(
        "--pattern-scale",
        type=int,
        default=32,
        help="Pattern scale to scan. Requires a stored pattern_space_mX.json in the selected Phase 1 runs. Current scanner supports 1..64.",
    )
    parser.add_argument("--top-patterns", type=int, default=16)
    parser.add_argument(
        "--pattern-selection",
        type=str,
        default="top",
        choices=("top", "rare-stable", "bridge-linked"),
        help="How to choose patterns from the stored Phase 1 pattern space.",
    )
    parser.add_argument("--long-lag-threshold", type=int, default=4096)
    parser.add_argument("--no-family-inference", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    variants = parse_variants(args.variants)
    if not variants:
        parser.error("At least one variant is required.")
    if args.pattern_scale <= 0 or args.pattern_scale > 64:
        parser.error("--pattern-scale must be in the range 1..64.")
    if args.top_patterns <= 0:
        parser.error("--top-patterns must be positive.")
    if args.long_lag_threshold <= 0:
        parser.error("--long-lag-threshold must be positive.")

    phase1_dir = resolve_dir(args.phase1_dir, anchor_file=__file__)
    output_dir = resolve_dir(args.output_dir, anchor_file=__file__)
    output_dir.mkdir(parents=True, exist_ok=True)

    phase1_scales = parse_scales(args.scales) if args.scales.strip() else None
    phase1_policies = [item.strip() for item in args.phase1_policies.split(",") if item.strip()] or None

    phase_print(
        "Preparing Gate 2 return-lag spectrum pilot",
        f"variants={','.join(variants)} | pattern_scale={args.pattern_scale} | top={args.top_patterns} | sel={args.pattern_selection}",
        quiet=args.quiet,
    )

    runs = discover_phase1_runs(phase1_dir)
    if not runs:
        parser.error(f"No valid Phase 1 runs found in {phase1_dir}")

    family = None
    explicit_filters = any(
        value is not None
        for value in (args.iteration, args.segment_bits, args.num_segments, args.segment_offset_bits)
    ) or phase1_scales is not None or phase1_policies is not None
    if not args.no_family_inference and not explicit_filters:
        family_candidates = filter_runs(runs, variants=variants or None)
        family_candidates = [
            run
            for run in family_candidates
            if _supports_pattern_scale(run, pattern_scale=args.pattern_scale)
        ]
        family = infer_family_from_latest_run(family_candidates)

    matching_runs = filter_runs(
        runs,
        variants=variants or None,
        iteration=args.iteration,
        segment_bits=args.segment_bits,
        num_segments=args.num_segments,
        segment_offset_bits=args.segment_offset_bits,
        scales=phase1_scales,
        policies=phase1_policies,
        family=family,
    )
    if not matching_runs:
        parser.error("No Phase 1 runs matched the requested selection.")

    selected = select_latest_per_variant(matching_runs, variant_order=variants)
    if not selected:
        parser.error("No latest-per-variant Phase 1 runs remain after filtering.")

    unsupported = [
        run["dataset"]["config"]["variant"]
        for run in selected
        if not _supports_pattern_scale(run, pattern_scale=args.pattern_scale)
    ]
    if unsupported:
        parser.error(
            "Selected runs do not contain the requested pattern scale for: "
            + ", ".join(unsupported)
        )

    phase_print(
        "Analyzing return-distance spectra",
        f"{len(selected)} runs | long_lag_threshold={args.long_lag_threshold}",
        quiet=args.quiet,
    )
    rows = build_return_lag_rows(
        selected,
        pattern_scale=args.pattern_scale,
        top_patterns=args.top_patterns,
        pattern_selection=args.pattern_selection,
        long_lag_threshold=args.long_lag_threshold,
        show_progress=not args.quiet,
    )
    selection = build_selection(rows, family_inferred=family is not None)

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_slug = build_run_slug(selection, timestamp)
    run_dir = output_dir / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = run_dir / "dataset.json"
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "manifest.json"
    report_path = run_dir / "report.md"

    generated_at = datetime.now().isoformat(timespec="seconds")
    dataset_payload = {
        "stage": "phase2_return_lag_spectrum",
        "generated_at": generated_at,
        "selection": selection,
        "rows": rows,
        "notes": [
            "This pilot targets longer-range recurrence order rather than parent-local transport.",
            "selected patterns are taken from the stored Phase 1 pattern space at the chosen scale.",
            "return lag = distance in window-start positions between consecutive occurrences of the same pattern within a segment.",
            "For null-surrogate runs, the analyzed surrogate slice is reconstructed from the stored Phase 1 config and null seed.",
        ],
    }
    summary_payload = {
        "generated_at": generated_at,
        "selection": selection,
        "compact_rows": [
            {
                "variant": row["variant"],
                "iteration": row["iteration"],
                "selected_pattern_count": row["selected_pattern_count"],
                "pattern_selection": row["pattern_selection"],
                "total_occurrence_count": row["total_occurrence_count"],
                "total_return_count": row["total_return_count"],
                "mean_return_lag": row["mean_return_lag"],
                "median_return_lag": row["median_return_lag"],
                "lag_entropy": row["lag_entropy"],
                "dominant_bin_mass_fraction": row["dominant_bin_mass_fraction"],
                "long_lag_fraction": row["long_lag_fraction"],
                "segment_profile_jaccard": row["segment_profile_jaccard"],
                "segment_profile_weighted_jaccard": row["segment_profile_weighted_jaccard"],
            }
            for row in rows
        ],
    }
    manifest_payload = {
        "run_slug": run_slug,
        "generated_at": generated_at,
        "script": "hsi_v2_phase2_return_lag_spectrum.py",
        "cwd": str(Path.cwd()),
        "outputs": {
            "dataset": str(dataset_path),
            "summary": str(summary_path),
            "manifest": str(manifest_path),
            "report": str(report_path),
        },
        "inputs": {
            "phase1_dir": str(phase1_dir),
            "phase1_runs": [row["run_dir"] for row in rows],
        },
        "arguments": vars(args),
    }

    phase_print("Writing artifacts", str(run_dir), quiet=args.quiet)
    with open(dataset_path, "w", encoding="utf-8") as handle:
        json.dump(dataset_payload, handle, indent=2)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, indent=2)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(render_markdown_report(rows, selection) + "\n")

    if not args.quiet:
        if selection["family_inferred"]:
            print("Using the latest coherent Phase 1 batch inferred from the newest run.")
            print("")
        print(render_console_summary(rows))
        print("")
        print(f"Saved dataset to: {dataset_path}")
        print(f"Saved summary to: {summary_path}")
        print(f"Saved report to: {report_path}")
        print(f"Saved manifest to: {manifest_path}")

    return 0


def build_selection(rows: list[dict], *, family_inferred: bool) -> dict:
    first = rows[0]
    return {
        "iterations": sorted({row["iteration"] for row in rows if row["iteration"] is not None}),
        "segment_bits": first["segment_bits"],
        "num_segments": first["num_segments"],
        "segment_offset_bits": first["segment_offset_bits"],
        "phase1_scales": list(first["scales"]),
        "pattern_scale": first["pattern_scale"],
        "top_patterns": first["top_patterns"],
        "pattern_selection": first["pattern_selection"],
        "long_lag_threshold": first["long_lag_threshold"],
        "variants": [row["variant"] for row in rows],
        "family_inferred": family_inferred,
    }


def build_run_slug(selection: dict, timestamp: str) -> str:
    variants = "-".join(selection["variants"])
    iteration_values = selection["iterations"]
    iter_part = "iter-mixed" if len(iteration_values) != 1 else f"iter-{iteration_values[0]}"
    offset_bits = selection.get("segment_offset_bits", 0)
    offset_part = f"__off-{compact_int(offset_bits)}" if offset_bits else ""
    return (
        f"phase2-return-lag__{iter_part}__seg-{selection['num_segments']}x{compact_int(selection['segment_bits'])}"
        f"{offset_part}__m-{selection['pattern_scale']}__sel-{selection['pattern_selection']}__top-{selection['top_patterns']}"
        f"__lag-{compact_int(selection['long_lag_threshold'])}__var-{variants}__{timestamp}"
    )


def phase_print(title: str, detail: str, *, quiet: bool) -> None:
    if quiet:
        return
    print(f"[Phase] {title}")
    print(f"        {detail}")
    print("")


def _supports_pattern_scale(run: dict, *, pattern_scale: int) -> bool:
    config = run["dataset"]["config"]
    return int(pattern_scale) in {int(value) for value in config.get("scales", [])}


if __name__ == "__main__":
    raise SystemExit(main())
