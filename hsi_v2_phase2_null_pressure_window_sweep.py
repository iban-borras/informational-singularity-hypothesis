#!/usr/bin/env python3
"""
HSI v2 Phase 2 Null Pressure Window Sweep

Strong-null window sweep that freezes one observed Phase 1 run plus one or more
null-surrogate Phase 1 runs, then slides the analyzed window through deeper
regions of the bitstream to compare late return-lag microstructure.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from v2.common.cli import resolve_dir
from v2.phase1.tower import parse_scales
from v2.phase2.null_pressure import (
    build_null_pressure_aggregate_slug,
    build_null_pressure_pair_slug,
    discover_phase1_runs_recursive,
    parse_null_models,
    parse_seed_list,
    render_null_pressure_console_summary,
    render_null_pressure_report,
    select_null_pressure_runs,
    summarize_null_pressure_rows,
)
from v2.phase2.return_lag import prepare_pattern_selection
from v2.phase2.window_sweep import build_window_sweep, render_window_sweep_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the HSI v2 Phase 2 strong-null pressure window sweep."
    )
    parser.add_argument(
        "--phase1-dir",
        type=str,
        default="results/hsi_v2/phase1_high_scales",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/hsi_v2/phase2/window_sweep_deep/null_pressure",
    )
    parser.add_argument("--observed-variant", type=str, default="B")
    parser.add_argument("--null-models", type=str, default="markov1,matched-lz")
    parser.add_argument("--matched-lz-seeds", type=str, default="")
    parser.add_argument("--iteration", type=int, default=None)
    parser.add_argument("--segment-bits", type=int, default=None)
    parser.add_argument("--num-segments", type=int, default=None)
    parser.add_argument("--segment-offset-bits", type=int, default=None)
    parser.add_argument("--scales", type=str, default="")
    parser.add_argument("--phase1-policies", type=str, default="")
    parser.add_argument("--pattern-scale", type=int, default=48)
    parser.add_argument("--top-patterns", type=int, default=64)
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

    observed_variant = args.observed_variant.strip().upper()
    if not observed_variant:
        parser.error("--observed-variant must not be empty.")
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

    try:
        null_models = parse_null_models(args.null_models)
    except ValueError as exc:
        parser.error(str(exc))
    if not null_models:
        parser.error("At least one null model is required.")

    try:
        matched_lz_seeds = parse_seed_list(args.matched_lz_seeds) if args.matched_lz_seeds.strip() else None
    except ValueError:
        parser.error("--matched-lz-seeds must be a comma-separated list of integers.")

    phase1_dir = resolve_dir(args.phase1_dir, anchor_file=__file__)
    output_dir = resolve_dir(args.output_dir, anchor_file=__file__)
    output_dir.mkdir(parents=True, exist_ok=True)

    phase1_scales = parse_scales(args.scales) if args.scales.strip() else None
    phase1_policies = [item.strip() for item in args.phase1_policies.split(",") if item.strip()] or None

    phase_print(
        "Preparing Phase 2 null-pressure window sweep",
        f"obs={observed_variant} | nulls={','.join(null_models)} | m={args.pattern_scale} | top={args.top_patterns}",
        quiet=args.quiet,
    )

    runs = discover_phase1_runs_recursive(phase1_dir)
    if not runs:
        parser.error(f"No valid Phase 1 runs found in {phase1_dir}")

    try:
        selected = select_null_pressure_runs(
            runs,
            observed_variant=observed_variant,
            null_models=null_models,
            matched_lz_seeds=matched_lz_seeds,
            iteration=args.iteration,
            segment_bits=args.segment_bits,
            num_segments=args.num_segments,
            segment_offset_bits=args.segment_offset_bits,
            scales=phase1_scales,
            policies=phase1_policies,
            pattern_scale=args.pattern_scale,
            no_family_inference=args.no_family_inference,
        )
    except ValueError as exc:
        parser.error(str(exc))

    observed_run = selected["observed_run"]
    null_specs = selected["null_specs"]
    if not null_specs:
        parser.error("No null comparators remain after selection.")

    observed_selection_anchor = prepare_pattern_selection(
        observed_run,
        pattern_scale=args.pattern_scale,
        top_patterns=args.top_patterns,
        pattern_selection=args.pattern_selection,
    )

    segment_bits = int(observed_run["dataset"]["config"]["segment_bits"])
    num_segments = int(observed_run["dataset"]["config"]["num_segments"])
    window_span_bits = segment_bits * num_segments
    window_step_bits = args.window_step_bits or window_span_bits

    phase_print(
        "Running null-pressure sweep",
        f"pairs={len(null_specs)} | windows={args.window_count} | start={args.start_offset_bits} | step={window_step_bits}",
        quiet=args.quiet,
    )

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_slug = build_null_pressure_aggregate_slug(
        observed_variant=observed_variant,
        pattern_scale=args.pattern_scale,
        pattern_selection=args.pattern_selection,
        top_patterns=args.top_patterns,
        start_offset_bits=args.start_offset_bits,
        window_step_bits=window_step_bits,
        window_count=args.window_count,
        timestamp=timestamp,
    )
    run_dir = output_dir / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)

    aggregate_rows = []
    artifacts = []

    for pair_index, spec in enumerate(null_specs, start=1):
        phase_print(
            "Null-pressure pair",
            f"{pair_index}/{len(null_specs)} | obs={observed_variant} | null={spec['null_label']}",
            quiet=args.quiet,
        )

        progress_callback = None
        if not args.quiet:
            progress_callback = _build_window_progress_callback(
                pair_index=pair_index,
                pair_total=len(null_specs),
                null_label=spec["null_label"],
            )
        selection_overrides = {
            observed_variant: {
                **observed_selection_anchor,
                "anchor_kind": "observed-fixed",
            },
            spec["comparison_variant"]: {
                **observed_selection_anchor,
                "anchor_kind": "observed-fixed",
            },
        }

        sweep = build_window_sweep(
            [observed_run, spec["run"]],
            pattern_scale=args.pattern_scale,
            top_patterns=args.top_patterns,
            pattern_selection=args.pattern_selection,
            long_lag_threshold=args.long_lag_threshold,
            comparison_pairs=[(observed_variant, spec["comparison_variant"])],
            start_offset_bits=args.start_offset_bits,
            window_step_bits=window_step_bits,
            window_count=args.window_count,
            selection_overrides=selection_overrides,
            show_progress=not args.quiet,
            progress_callback=progress_callback,
        )

        pair_selection = build_pair_selection(
            observed_variant=observed_variant,
            observed_run=observed_run,
            null_spec=spec,
            pattern_scale=args.pattern_scale,
            top_patterns=args.top_patterns,
            pattern_selection=args.pattern_selection,
            long_lag_threshold=args.long_lag_threshold,
            start_offset_bits=args.start_offset_bits,
            window_step_bits=window_step_bits,
            window_count=args.window_count,
            family_inferred=selected["family_inferred"],
            selection_anchor_variant=observed_variant,
            selection_anchor_run_dir=observed_run["_run_dir"],
            source_freeze_mode=sweep.get("source_freeze_mode", "run-local"),
            frozen_total_bits=sweep.get("frozen_total_bits"),
        )
        pair_dir = run_dir / build_null_pressure_pair_slug(
            observed_variant=observed_variant,
            null_spec=spec,
            pattern_scale=args.pattern_scale,
            pattern_selection=args.pattern_selection,
            top_patterns=args.top_patterns,
            start_offset_bits=args.start_offset_bits,
            window_step_bits=window_step_bits,
            window_count=args.window_count,
            timestamp=datetime.now().strftime("%Y%m%dT%H%M%S"),
        )
        pair_dir.mkdir(parents=True, exist_ok=True)

        pair_dataset_path = pair_dir / "dataset.json"
        pair_summary_path = pair_dir / "summary.json"
        pair_report_path = pair_dir / "report.md"
        pair_manifest_path = pair_dir / "manifest.json"
        generated_at = datetime.now().isoformat(timespec="seconds")

        pair_dataset_payload = {
            "stage": "phase2_return_lag_null_pressure_pair",
            "generated_at": generated_at,
            "selection": pair_selection,
            "sweep": sweep,
        }
        pair_summary_payload = {
            "generated_at": generated_at,
            "selection": pair_selection,
            "source_freeze_mode": sweep.get("source_freeze_mode", "run-local"),
            "frozen_total_bits": sweep.get("frozen_total_bits"),
            "pair_rows": sweep["pair_rows"],
            "variant_rows": sweep["variant_rows"],
            "top_pair_rows": sweep["top_pair_rows"],
        }
        pair_manifest_payload = {
            "generated_at": generated_at,
            "script": "hsi_v2_phase2_null_pressure_window_sweep.py",
            "cwd": str(Path.cwd()),
            "outputs": {
                "dataset": str(pair_dataset_path),
                "summary": str(pair_summary_path),
                "report": str(pair_report_path),
                "manifest": str(pair_manifest_path),
            },
            "inputs": {
                "observed_phase1_run": observed_run["_run_dir"],
                "null_phase1_run": spec["run"]["_run_dir"],
            },
            "selection": pair_selection,
            "source_freeze_mode": sweep.get("source_freeze_mode", "run-local"),
            "frozen_total_bits": sweep.get("frozen_total_bits"),
        }

        with open(pair_dataset_path, "w", encoding="utf-8") as handle:
            json.dump(pair_dataset_payload, handle, indent=2)
        with open(pair_summary_path, "w", encoding="utf-8") as handle:
            json.dump(pair_summary_payload, handle, indent=2)
        with open(pair_report_path, "w", encoding="utf-8") as handle:
            handle.write(
                render_window_sweep_report(
                    sweep,
                    variants=[observed_variant, spec["variant"]],
                    pattern_scale=args.pattern_scale,
                    top_patterns=args.top_patterns,
                    pattern_selection=args.pattern_selection,
                    long_lag_threshold=args.long_lag_threshold,
                    frozen_runs=[observed_run, spec["run"]],
                )
                + "\n"
            )
        with open(pair_manifest_path, "w", encoding="utf-8") as handle:
            json.dump(pair_manifest_payload, handle, indent=2)

        artifacts.append(
            {
                "null_label": spec["null_label"],
                "null_model": spec["null_model"],
                "null_seed": spec.get("null_seed"),
                "run_dir": spec["run"]["_run_dir"],
                "pair_dir": str(pair_dir),
                "report_path": str(pair_report_path),
            }
        )
        for row in sweep["pair_rows"]:
            aggregate_rows.append(
                {
                    **row,
                    "observed_variant": observed_variant,
                    "null_variant": spec["variant"],
                    "null_label": spec["null_label"],
                    "null_model": spec["null_model"],
                    "null_seed": spec.get("null_seed"),
                    "report_path": str(pair_report_path),
                }
            )

    aggregate_selection = {
        "observed_variant": observed_variant,
        "null_models": null_models,
        "matched_lz_seeds": matched_lz_seeds or [],
        "iteration": observed_run["dataset"]["config"].get("iteration"),
        "segment_bits": segment_bits,
        "num_segments": num_segments,
        "phase1_scales": list(observed_run["dataset"]["config"].get("scales", [])),
        "phase1_policies": list(observed_run["dataset"]["config"].get("policies", [])),
        "pattern_scale": args.pattern_scale,
        "top_patterns": args.top_patterns,
        "pattern_selection": args.pattern_selection,
        "long_lag_threshold": args.long_lag_threshold,
        "selection_anchor_mode": "observed-fixed",
        "selection_anchor_variant": observed_variant,
        "selection_anchor_run_dir": observed_run["_run_dir"],
        "start_offset_bits": args.start_offset_bits,
        "window_step_bits": window_step_bits,
        "window_count": args.window_count,
        "family_inferred": selected["family_inferred"],
        "source_freeze_mode": "sweep-global-frozen",
        "frozen_total_bits": args.start_offset_bits + ((args.window_count - 1) * window_step_bits) + window_span_bits,
    }

    dataset_path = run_dir / "dataset.json"
    summary_path = run_dir / "summary.json"
    report_path = run_dir / "report.md"
    manifest_path = run_dir / "manifest.json"
    generated_at = datetime.now().isoformat(timespec="seconds")
    aggregate_summaries = summarize_null_pressure_rows(aggregate_rows)

    dataset_payload = {
        "stage": "phase2_return_lag_null_pressure",
        "generated_at": generated_at,
        "selection": aggregate_selection,
        "pair_rows": aggregate_rows,
        "summaries": aggregate_summaries,
        "artifacts": artifacts,
    }
    summary_payload = {
        "generated_at": generated_at,
        "selection": aggregate_selection,
        "pair_rows": aggregate_rows,
        "summaries": aggregate_summaries,
        "artifacts": artifacts,
    }
    manifest_payload = {
        "generated_at": generated_at,
        "script": "hsi_v2_phase2_null_pressure_window_sweep.py",
        "cwd": str(Path.cwd()),
        "outputs": {
            "dataset": str(dataset_path),
            "summary": str(summary_path),
            "report": str(report_path),
            "manifest": str(manifest_path),
        },
        "inputs": {
            "phase1_dir": str(phase1_dir),
            "observed_phase1_run": observed_run["_run_dir"],
            "null_phase1_runs": [spec["run"]["_run_dir"] for spec in null_specs],
        },
        "arguments": vars(args),
    }

    with open(dataset_path, "w", encoding="utf-8") as handle:
        json.dump(dataset_payload, handle, indent=2)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(
            render_null_pressure_report(
                selection=aggregate_selection,
                rows=aggregate_rows,
                artifacts=artifacts,
            )
            + "\n"
        )
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, indent=2)

    if not args.quiet:
        if aggregate_selection["family_inferred"]:
            print("Using the latest coherent Phase 1 family inferred from the newest observed run.")
            print("")
        print(render_null_pressure_console_summary(aggregate_rows))
        print("")
        print(f"Saved dataset to: {dataset_path}")
        print(f"Saved summary to: {summary_path}")
        print(f"Saved report to: {report_path}")
        print(f"Saved manifest to: {manifest_path}")

    return 0


def build_pair_selection(
    *,
    observed_variant: str,
    observed_run: dict,
    null_spec: dict,
    pattern_scale: int,
    top_patterns: int,
    pattern_selection: str,
    long_lag_threshold: int,
    start_offset_bits: int,
    window_step_bits: int,
    window_count: int,
    family_inferred: bool,
    selection_anchor_variant: str,
    selection_anchor_run_dir: str,
    source_freeze_mode: str,
    frozen_total_bits: int | None,
) -> dict:
    config = observed_run["dataset"]["config"]
    return {
        "observed_variant": observed_variant,
        "null_variant": null_spec["variant"],
        "null_label": null_spec["null_label"],
        "null_model": null_spec["null_model"],
        "null_seed": null_spec.get("null_seed"),
        "iteration": config.get("iteration"),
        "segment_bits": config.get("segment_bits"),
        "num_segments": config.get("num_segments"),
        "phase1_scales": list(config.get("scales", [])),
        "phase1_policies": list(config.get("policies", [])),
        "pattern_scale": pattern_scale,
        "top_patterns": top_patterns,
        "pattern_selection": pattern_selection,
        "long_lag_threshold": long_lag_threshold,
        "selection_anchor_mode": "observed-fixed",
        "selection_anchor_variant": selection_anchor_variant,
        "selection_anchor_run_dir": selection_anchor_run_dir,
        "start_offset_bits": start_offset_bits,
        "window_step_bits": window_step_bits,
        "window_count": window_count,
        "family_inferred": family_inferred,
        "source_freeze_mode": source_freeze_mode,
        "frozen_total_bits": frozen_total_bits,
    }


def phase_print(title: str, detail: str, *, quiet: bool) -> None:
    if quiet:
        return
    print(f"[Phase] {title}")
    print(f"        {detail}")
    print("")


def window_note(detail: str, *, quiet: bool) -> None:
    if quiet:
        return
    print(f"[Window] {detail}")


def _build_window_progress_callback(
    *,
    pair_index: int,
    pair_total: int,
    null_label: str,
):
    def _callback(window_index: int, segment_offset_bits: int, window_count: int) -> None:
        window_note(
            f"pair {pair_index}/{pair_total} | window {window_index + 1}/{window_count} | "
            f"offset={segment_offset_bits} | null={null_label}",
            quiet=False,
        )

    return _callback


if __name__ == "__main__":
    raise SystemExit(main())
