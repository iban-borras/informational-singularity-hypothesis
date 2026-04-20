#!/usr/bin/env python3
"""
HSI v2 Phase 2 Minimum Commutation-Defect Pilot

First executable Gate 2 pilot built strictly on top of existing Phase 1 artifacts:
- loads canonical Phase 1 runs
- extracts a dominant continuation readout from the tail fibers
- measures whether local mass dominance conflicts with one-step-ahead branching richness
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
from v2.phase1.tower import parse_policies, parse_scales
from v2.phase2.defects import build_defect_rows, render_console_summary, render_markdown_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the minimum HSI v2 Gate 2 commutation-defect pilot from existing Phase 1 artifacts."
    )
    parser.add_argument(
        "--phase1-dir",
        type=str,
        default="results/hsi_v2/phase1",
        help="Directory containing Phase 1 run folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/hsi_v2/phase2",
        help="Directory where Phase 2 pilot artifacts will be written.",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="B,F,M,N",
        help="Comma-separated variant list. Default: B,F,M,N",
    )
    parser.add_argument("--iteration", type=int, default=None, help="Optional iteration filter.")
    parser.add_argument("--segment-bits", type=int, default=None, help="Optional segment size filter.")
    parser.add_argument("--num-segments", type=int, default=None, help="Optional segment-count filter.")
    parser.add_argument(
        "--segment-offset-bits",
        type=int,
        default=None,
        help="Optional segment-offset filter in bits.",
    )
    parser.add_argument(
        "--scales",
        type=str,
        default="",
        help="Optional exact Phase 1 scale ladder filter.",
    )
    parser.add_argument(
        "--phase1-policies",
        type=str,
        default="",
        help="Optional exact Phase 1 policy filter.",
    )
    parser.add_argument(
        "--window-scales",
        type=str,
        default="24,28,32",
        help="Exactly three ascending scales used for the defect pilot, e.g. 24,28,32.",
    )
    parser.add_argument(
        "--policies",
        type=str,
        default="prefix,suffix",
        help="Pilot policies to analyze from the stored fiber artifacts.",
    )
    parser.add_argument(
        "--no-family-inference",
        action="store_true",
        help="Do not infer a coherent Phase 1 family from the newest available run.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress console tables after writing artifacts.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    variants = parse_variants(args.variants)
    if not variants:
        parser.error("At least one variant is required.")

    phase1_dir = resolve_dir(args.phase1_dir, anchor_file=__file__)
    output_dir = resolve_dir(args.output_dir, anchor_file=__file__)
    output_dir.mkdir(parents=True, exist_ok=True)

    phase1_scales = parse_scales(args.scales) if args.scales.strip() else None
    phase1_policies = parse_policies(args.phase1_policies) if args.phase1_policies.strip() else None
    pilot_policies = parse_policies(args.policies)
    window_scales = tuple(parse_scales(args.window_scales))
    if len(window_scales) != 3:
        parser.error("--window-scales must contain exactly three ascending scales.")

    phase_print(
        "Preparing Gate 2 defect pilot",
        f"variants={','.join(variants)} | window={','.join(str(value) for value in window_scales)}",
        quiet=args.quiet,
    )

    runs = discover_phase1_runs(phase1_dir)
    if not runs:
        parser.error(f"No valid Phase 1 runs found in {phase1_dir}")

    family = None
    explicit_filters = any(
        value is not None for value in (args.iteration, args.segment_bits, args.num_segments, args.segment_offset_bits)
    ) or phase1_scales is not None or phase1_policies is not None
    if not args.no_family_inference and not explicit_filters:
        family_candidates = filter_runs(runs, variants=variants or None)
        family_candidates = [
            run
            for run in family_candidates
            if _supports_window(run, window_scales=window_scales, policies=pilot_policies)
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
        if not _supports_window(run, window_scales=window_scales, policies=pilot_policies)
    ]
    if unsupported:
        parser.error(
            "Selected runs do not contain the requested defect window/policies for: "
            + ", ".join(unsupported)
        )

    phase_print(
        "Analyzing dominant-continuation defects",
        f"{len(selected)} runs | policies={','.join(pilot_policies)}",
        quiet=args.quiet,
    )
    try:
        rows = build_defect_rows(
            selected,
            window_scales=window_scales,
            policies=pilot_policies,
        )
    except ValueError as exc:
        parser.error(str(exc))

    selection = build_selection(rows, family_inferred=family is not None)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_slug = build_run_slug(selection, timestamp)
    run_dir = output_dir / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = run_dir / "dataset.json"
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "manifest.json"
    report_path = run_dir / "report.md"

    dataset_payload = {
        "stage": "phase2_defect_pilot",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "selection": selection,
        "rows": rows,
        "notes": [
            "This pilot measures dominant-continuation instability over a two-edge tail window.",
            "defect = mismatch between the locally mass-dominant child and the child favored by one-step-ahead branching richness.",
            "terminal closure = no child of the current parent retains positive one-step-ahead branching richness.",
            "The pilot is a Gate 2 precursor, not yet a graph or spectral construction.",
        ],
    }
    summary_payload = {
        "generated_at": dataset_payload["generated_at"],
        "selection": selection,
        "compact_rows": [
            {
                "variant": row["variant"],
                "iteration": row["iteration"],
                "prefix_defect_rate": row.get("prefix_defect_rate"),
                "suffix_defect_rate": row.get("suffix_defect_rate"),
                "prefix_defect_mass_rate": row.get("prefix_defect_mass_rate"),
                "suffix_defect_mass_rate": row.get("suffix_defect_mass_rate"),
                "prefix_future_parent_fraction": row.get("prefix_future_parent_fraction"),
                "suffix_future_parent_fraction": row.get("suffix_future_parent_fraction"),
                "prefix_terminal_parent_fraction": row.get("prefix_terminal_parent_fraction"),
                "suffix_terminal_parent_fraction": row.get("suffix_terminal_parent_fraction"),
                "prefix_defect_entropy": row.get("prefix_defect_entropy"),
                "suffix_defect_entropy": row.get("suffix_defect_entropy"),
                "defect_rate_gap_abs": row.get("defect_rate_gap_abs"),
                "defect_mass_gap_abs": row.get("defect_mass_gap_abs"),
            }
            for row in rows
        ],
    }
    manifest_payload = {
        "run_slug": run_slug,
        "generated_at": dataset_payload["generated_at"],
        "script": "hsi_v2_phase2_defect_pilot.py",
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
        print(f"\nSaved dataset to: {dataset_path}")
        print(f"Saved summary to: {summary_path}")
        print(f"Saved manifest to: {manifest_path}")
        print(f"Saved report to: {report_path}")

    return 0


def build_selection(rows: list[dict], *, family_inferred: bool) -> dict:
    iterations = sorted({row["iteration"] for row in rows if row["iteration"] is not None})
    segment_bits = sorted({row["segment_bits"] for row in rows})
    num_segments = sorted({row["num_segments"] for row in rows})
    segment_offset_bits = sorted({row["segment_offset_bits"] for row in rows})
    phase1_scales = rows[0]["scales"]
    policies = rows[0]["policies"]
    window_scales = rows[0]["window_scales"]
    return {
        "iterations": iterations,
        "segment_bits": segment_bits[0] if len(segment_bits) == 1 else "mixed",
        "num_segments": num_segments[0] if len(num_segments) == 1 else "mixed",
        "segment_offset_bits": segment_offset_bits[0] if len(segment_offset_bits) == 1 else "mixed",
        "phase1_scales": phase1_scales,
        "window_scales": window_scales,
        "policies": policies,
        "variants": [row["variant"] for row in rows],
        "family_inferred": family_inferred,
    }


def build_run_slug(selection: dict, timestamp: str) -> str:
    variants = "-".join(selection["variants"])
    iteration_part = str(selection["iterations"][0]) if len(selection["iterations"]) == 1 else "mixed"
    window_part = "-".join(str(value) for value in selection["window_scales"])
    policy_part = "-".join(selection["policies"])
    offset_part = ""
    if selection["segment_offset_bits"] not in (0, "mixed"):
        offset_part = f"__off-{compact_int(selection['segment_offset_bits'])}"
    elif selection["segment_offset_bits"] == "mixed":
        offset_part = "__off-mixed"
    return (
        f"phase2-defect-pilot__iter-{iteration_part}__seg-{selection['num_segments']}x{selection['segment_bits']}"
        f"{offset_part}__window-{window_part}__pol-{policy_part}__var-{variants}__{timestamp}"
    )


def phase_print(title: str, detail: str, *, quiet: bool) -> None:
    if quiet:
        return
    print(f"\n[Phase] {title}")
    print(f"        {detail}")


def _supports_window(run: dict, *, window_scales: tuple[int, int, int], policies: list[str]) -> bool:
    config = run["dataset"]["config"]
    available_scales = list(config.get("scales", []))
    available_policies = list(config.get("policies", []))
    return all(scale in available_scales for scale in window_scales) and all(
        policy in available_policies for policy in policies
    )


if __name__ == "__main__":
    raise SystemExit(main())
