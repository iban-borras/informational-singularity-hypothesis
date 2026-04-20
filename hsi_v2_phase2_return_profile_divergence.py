#!/usr/bin/env python3
"""
HSI v2 Phase 2 Return-Profile Divergence

Compare per-pattern lag-bin profiles between an observed return-lag run and a null run.
This is intended as a post-analysis layer over existing `hsi_v2_phase2_return_lag_spectrum.py` outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from v2.common.cli import resolve_dir
from v2.phase2.return_profile_compare import (
    compare_return_profiles,
    load_return_lag_row,
    render_profile_comparison_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare per-pattern return-lag profiles between an observed run and a null run."
    )
    parser.add_argument("--observed-dataset", type=str, required=True)
    parser.add_argument("--null-dataset", type=str, required=True)
    parser.add_argument("--observed-variant", type=str, default="B")
    parser.add_argument("--null-variant", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/hsi_v2/phase2")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    observed_dataset = resolve_dir(args.observed_dataset, anchor_file=__file__)
    null_dataset = resolve_dir(args.null_dataset, anchor_file=__file__)
    if observed_dataset.is_dir():
        observed_dataset = observed_dataset / "dataset.json"
    if null_dataset.is_dir():
        null_dataset = null_dataset / "dataset.json"
    if not observed_dataset.is_file():
        parser.error(f"Observed dataset not found: {observed_dataset}")
    if not null_dataset.is_file():
        parser.error(f"Null dataset not found: {null_dataset}")

    output_dir = resolve_dir(args.output_dir, anchor_file=__file__)
    output_dir.mkdir(parents=True, exist_ok=True)

    phase_print(
        "Preparing return-profile divergence comparison",
        f"obs={args.observed_variant} | null={args.null_variant}",
        quiet=args.quiet,
    )
    observed_row = load_return_lag_row(observed_dataset, variant=args.observed_variant)
    null_row = load_return_lag_row(null_dataset, variant=args.null_variant)
    comparison = compare_return_profiles(observed_row, null_row)

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_slug = build_run_slug(comparison, timestamp)
    run_dir = output_dir / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_path = run_dir / "summary.json"
    report_path = run_dir / "report.md"
    manifest_path = run_dir / "manifest.json"

    generated_at = datetime.now().isoformat(timespec="seconds")
    payload = {
        "generated_at": generated_at,
        "comparison": comparison,
    }
    manifest = {
        "generated_at": generated_at,
        "script": "hsi_v2_phase2_return_profile_divergence.py",
        "cwd": str(Path.cwd()),
        "inputs": {
            "observed_dataset": str(observed_dataset),
            "null_dataset": str(null_dataset),
            "observed_variant": args.observed_variant,
            "null_variant": args.null_variant,
        },
        "outputs": {
            "summary": str(summary_path),
            "report": str(report_path),
            "manifest": str(manifest_path),
        },
        "arguments": vars(args),
    }

    phase_print("Writing artifacts", str(run_dir), quiet=args.quiet)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(render_profile_comparison_report(comparison) + "\n")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    if not args.quiet:
        print(render_console_summary(comparison))
        print("")
        print(f"Saved summary to: {summary_path}")
        print(f"Saved report to: {report_path}")
        print(f"Saved manifest to: {manifest_path}")

    return 0


def build_run_slug(comparison: dict, timestamp: str) -> str:
    return (
        f"phase2-return-profile-divergence__obs-{comparison['observed_variant']}"
        f"__null-{comparison['null_variant']}__m-{comparison['pattern_scale']}"
        f"__sel-{comparison['pattern_selection']}__shared-{comparison['shared_pattern_count']}"
        f"__{timestamp}"
    )


def render_console_summary(comparison: dict) -> str:
    lines = [
        "Phase 2 return-profile divergence",
        "-" * 118,
        f"{'obs':<12}{'null':<18}{'m':>6}{'shared':>8}{'sel_J':>10}{'prof_wJ':>10}{'prof_JS':>10}{'d_mean':>10}{'top_eq':>10}",
        f"{comparison['observed_variant']:<12}"
        f"{comparison['null_variant']:<18}"
        f"{comparison['pattern_scale']:>6}"
        f"{comparison['shared_pattern_count']:>8}"
        f"{comparison['selection_jaccard']:>10.4f}"
        f"{comparison['weighted_profile_jaccard']:>10.4f}"
        f"{comparison['weighted_profile_js_divergence']:>10.4f}"
        f"{comparison['weighted_relative_mean_lag_gap']:>10.4f}"
        f"{comparison['weighted_top_bin_match_fraction']:>10.4f}",
    ]
    return "\n".join(lines)


def phase_print(title: str, detail: str, *, quiet: bool) -> None:
    if quiet:
        return
    print(f"[Phase] {title}")
    print(f"        {detail}")
    print("")


if __name__ == "__main__":
    raise SystemExit(main())
