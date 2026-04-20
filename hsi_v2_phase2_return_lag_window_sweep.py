#!/usr/bin/env python3
"""
HSI v2 Phase 2 Return-Lag Window Sweep

Observed-only comparative sweep that freezes the selected Phase 1 runs and
slides the analyzed observable window through the bitstream by changing only
`segment_offset_bits`.
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
    filter_runs,
    infer_family_from_latest_run,
    select_latest_per_variant,
)
from v2.phase1.tower import parse_scales
from v2.phase2.window_sweep import (
    build_window_sweep,
    parse_comparison_pairs,
    render_window_sweep_console_summary,
    render_window_sweep_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the HSI v2 Phase 2 observed-only return-lag window sweep."
    )
    parser.add_argument(
        "--phase1-dir",
        type=str,
        default="results/hsi_v2/phase1_high_scales",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/hsi_v2/phase2/window_sweep",
    )
    parser.add_argument("--variants", type=str, default="B,E,I")
    parser.add_argument("--comparison-pairs", type=str, default="")
    parser.add_argument("--iteration", type=int, default=None)
    parser.add_argument("--segment-bits", type=int, default=None)
    parser.add_argument("--num-segments", type=int, default=None)
    parser.add_argument("--segment-offset-bits", type=int, default=None)
    parser.add_argument("--scales", type=str, default="")
    parser.add_argument("--phase1-policies", type=str, default="")
    parser.add_argument("--pattern-scale", type=int, default=40)
    parser.add_argument("--top-patterns", type=int, default=16)
    parser.add_argument(
        "--pattern-selection",
        type=str,
        default="bridge-linked",
        choices=("top", "rare-stable", "bridge-linked"),
    )
    parser.add_argument("--long-lag-threshold", type=int, default=4096)
    parser.add_argument("--start-offset-bits", type=int, default=0)
    parser.add_argument("--window-step-bits", type=int, default=0)
    parser.add_argument("--window-count", type=int, required=True)
    parser.add_argument("--no-family-inference", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    variants = parse_variants(args.variants)
    if len(variants) < 2:
        parser.error("At least two variants are required for a comparative window sweep.")
    if args.pattern_scale <= 0 or args.pattern_scale > 64:
        parser.error("--pattern-scale must be in the range 1..64.")
    if args.top_patterns <= 0:
        parser.error("--top-patterns must be positive.")
    if args.long_lag_threshold <= 0:
        parser.error("--long-lag-threshold must be positive.")
    if args.start_offset_bits < 0:
        parser.error("--start-offset-bits must be non-negative.")
    if args.window_step_bits < 0:
        parser.error("--window-step-bits must be non-negative.")
    if args.window_count <= 0:
        parser.error("--window-count must be positive.")

    phase1_dir = resolve_dir(args.phase1_dir, anchor_file=__file__)
    output_dir = resolve_dir(args.output_dir, anchor_file=__file__)
    output_dir.mkdir(parents=True, exist_ok=True)

    phase1_scales = parse_scales(args.scales) if args.scales.strip() else None
    phase1_policies = [item.strip() for item in args.phase1_policies.split(",") if item.strip()] or None

    phase_print(
        "Preparing Phase 2 observed window sweep",
        f"variants={','.join(variants)} | pattern_scale={args.pattern_scale} | top={args.top_patterns} | sel={args.pattern_selection}",
        quiet=args.quiet,
    )

    runs = _discover_phase1_runs_recursive(phase1_dir)
    if not runs:
        parser.error(f"No valid Phase 1 runs found in {phase1_dir}")

    runs = [run for run in runs if _is_observed_run(run)]
    if not runs:
        parser.error(f"No observed Phase 1 runs found in {phase1_dir}")

    family = None
    explicit_filters = any(
        value is not None
        for value in (args.iteration, args.segment_bits, args.num_segments, args.segment_offset_bits)
    ) or phase1_scales is not None or phase1_policies is not None
    if not args.no_family_inference and not explicit_filters:
        family_candidates = filter_runs(runs, variants=variants or None)
        family_candidates = [
            run for run in family_candidates if _supports_pattern_scale(run, pattern_scale=args.pattern_scale)
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
        parser.error("No observed Phase 1 runs matched the requested selection.")

    selected = select_latest_per_variant(matching_runs, variant_order=variants)
    if len(selected) != len(variants):
        found = [run["dataset"]["config"]["variant"] for run in selected]
        missing = [variant for variant in variants if variant not in found]
        parser.error(f"Missing observed variants for window sweep: {', '.join(missing)}")

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

    try:
        comparison_pairs = parse_comparison_pairs(args.comparison_pairs, variants=variants)
    except ValueError as exc:
        parser.error(str(exc))
    if not comparison_pairs:
        parser.error("No comparison pairs remain after parsing.")

    segment_bits = int(selected[0]["dataset"]["config"]["segment_bits"])
    num_segments = int(selected[0]["dataset"]["config"]["num_segments"])
    window_span_bits = segment_bits * num_segments
    window_step_bits = args.window_step_bits or window_span_bits

    phase_print(
        "Sweeping observed windows",
        f"windows={args.window_count} | start={args.start_offset_bits} | step={window_step_bits}",
        quiet=args.quiet,
    )

    sweep = build_window_sweep(
        selected,
        pattern_scale=args.pattern_scale,
        top_patterns=args.top_patterns,
        pattern_selection=args.pattern_selection,
        long_lag_threshold=args.long_lag_threshold,
        comparison_pairs=comparison_pairs,
        start_offset_bits=args.start_offset_bits,
        window_step_bits=window_step_bits,
        window_count=args.window_count,
        show_progress=not args.quiet,
    )

    selection = build_selection(
        selected,
        variants=variants,
        comparison_pairs=comparison_pairs,
        pattern_scale=args.pattern_scale,
        top_patterns=args.top_patterns,
        pattern_selection=args.pattern_selection,
        long_lag_threshold=args.long_lag_threshold,
        start_offset_bits=args.start_offset_bits,
        window_step_bits=window_step_bits,
        window_count=args.window_count,
        family_inferred=family is not None,
    )

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
        "stage": "phase2_return_lag_window_sweep",
        "generated_at": generated_at,
        "selection": selection,
        "sweep": sweep,
    }
    summary_payload = {
        "generated_at": generated_at,
        "selection": selection,
        "pair_rows": sweep["pair_rows"],
        "variant_rows": sweep["variant_rows"],
        "top_pair_rows": sweep["top_pair_rows"],
    }
    manifest_payload = {
        "run_slug": run_slug,
        "generated_at": generated_at,
        "script": "hsi_v2_phase2_return_lag_window_sweep.py",
        "cwd": str(Path.cwd()),
        "outputs": {
            "dataset": str(dataset_path),
            "summary": str(summary_path),
            "report": str(report_path),
            "manifest": str(manifest_path),
        },
        "inputs": {
            "phase1_dir": str(phase1_dir),
            "phase1_runs": [run["_run_dir"] for run in selected],
        },
        "arguments": vars(args),
    }

    phase_print("Writing artifacts", str(run_dir), quiet=args.quiet)
    with open(dataset_path, "w", encoding="utf-8") as handle:
        json.dump(dataset_payload, handle, indent=2)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(
            render_window_sweep_report(
                sweep,
                variants=variants,
                pattern_scale=args.pattern_scale,
                top_patterns=args.top_patterns,
                pattern_selection=args.pattern_selection,
                long_lag_threshold=args.long_lag_threshold,
                frozen_runs=selected,
            )
            + "\n"
        )
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, indent=2)

    if not args.quiet:
        if selection["family_inferred"]:
            print("Using the latest coherent observed Phase 1 batch inferred from the newest run.")
            print("")
        print(render_window_sweep_console_summary(sweep))
        print("")
        print(f"Saved dataset to: {dataset_path}")
        print(f"Saved summary to: {summary_path}")
        print(f"Saved report to: {report_path}")
        print(f"Saved manifest to: {manifest_path}")

    return 0


def build_selection(
    selected: list[dict],
    *,
    variants: list[str],
    comparison_pairs: list[tuple[str, str]],
    pattern_scale: int,
    top_patterns: int,
    pattern_selection: str,
    long_lag_threshold: int,
    start_offset_bits: int,
    window_step_bits: int,
    window_count: int,
    family_inferred: bool,
) -> dict:
    first_config = selected[0]["dataset"]["config"]
    return {
        "variants": variants,
        "comparison_pairs": [f"{left}:{right}" for left, right in comparison_pairs],
        "iteration": first_config.get("iteration"),
        "segment_bits": first_config.get("segment_bits"),
        "num_segments": first_config.get("num_segments"),
        "phase1_scales": list(first_config.get("scales", [])),
        "phase1_policies": list(first_config.get("policies", [])),
        "pattern_scale": pattern_scale,
        "top_patterns": top_patterns,
        "pattern_selection": pattern_selection,
        "long_lag_threshold": long_lag_threshold,
        "start_offset_bits": start_offset_bits,
        "window_step_bits": window_step_bits,
        "window_count": window_count,
        "family_inferred": family_inferred,
    }


def build_run_slug(selection: dict, timestamp: str) -> str:
    variant_part = "-".join(selection["variants"])
    pair_part = "-".join(item.replace(":", "-") for item in selection["comparison_pairs"])
    start_part = (
        f"__off-{compact_int(selection['start_offset_bits'])}"
        if selection["start_offset_bits"] > 0
        else ""
    )
    return (
        f"phase2-return-lag-window-sweep__var-{variant_part}__pairs-{pair_part}"
        f"__m-{selection['pattern_scale']}__sel-{selection['pattern_selection']}"
        f"__top-{selection['top_patterns']}__w-{selection['window_count']}"
        f"{start_part}__step-{compact_int(selection['window_step_bits'])}__{timestamp}"
    )


def phase_print(title: str, detail: str, *, quiet: bool) -> None:
    if quiet:
        return
    print(f"[Phase] {title}")
    print(f"        {detail}")
    print("")


def _is_observed_run(run: dict) -> bool:
    config = run["dataset"]["config"]
    return str(config.get("sequence_kind", "observed")) != "null_surrogate"


def _supports_pattern_scale(run: dict, *, pattern_scale: int) -> bool:
    run_dir = Path(run["_run_dir"])
    path = run_dir / "pattern_spaces" / f"pattern_space_m{pattern_scale}.json"
    return path.is_file()


def _discover_phase1_runs_recursive(phase1_dir: Path) -> list[dict]:
    runs = []
    for dataset_path in sorted(phase1_dir.rglob("dataset.json")):
        try:
            with open(dataset_path, "r", encoding="utf-8") as handle:
                dataset = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if dataset.get("stage") != "phase1_tower":
            continue

        run_dir = dataset_path.parent
        summary_path = run_dir / "phase1_summary.json"
        if not summary_path.is_file():
            continue
        try:
            with open(summary_path, "r", encoding="utf-8") as handle:
                summary = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue

        generated_at = dataset.get("generated_at")
        if not generated_at:
            generated_at = datetime.fromtimestamp(dataset_path.stat().st_mtime).isoformat(timespec="seconds")
        runs.append(
            {
                "dataset": dataset,
                "summary": summary,
                "_dataset_path": str(dataset_path.resolve()),
                "_summary_path": str(summary_path.resolve()),
                "_run_dir": str(run_dir.resolve()),
                "_generated_at_dt": datetime.fromisoformat(generated_at),
                "generated_at": generated_at,
            }
        )
    return runs


if __name__ == "__main__":
    raise SystemExit(main())
