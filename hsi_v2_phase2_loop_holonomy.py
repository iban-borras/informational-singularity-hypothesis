#!/usr/bin/env python3
"""
HSI v2 Phase 2 Cross-Policy Loop Holonomy Pilot

Minimal follow-up to the parent-local defect pilot:
- reuses existing Phase 1 fiber artifacts
- builds two-step dominant continuation paths under prefix and suffix
- returns each path to the low scale through the opposite projection
- measures loop non-closure and path disagreement
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
from v2.phase2.holonomy import build_holonomy_rows, render_console_summary, render_markdown_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the HSI v2 Gate 2 cross-policy loop holonomy pilot from existing Phase 1 artifacts."
    )
    parser.add_argument("--phase1-dir", type=str, default="results/hsi_v2/phase1")
    parser.add_argument("--output-dir", type=str, default="results/hsi_v2/phase2")
    parser.add_argument("--variants", type=str, default="B,F,M,N")
    parser.add_argument("--iteration", type=int, default=None)
    parser.add_argument("--segment-bits", type=int, default=None)
    parser.add_argument("--num-segments", type=int, default=None)
    parser.add_argument("--segment-offset-bits", type=int, default=None)
    parser.add_argument("--scales", type=str, default="")
    parser.add_argument("--phase1-policies", type=str, default="")
    parser.add_argument("--window-scales", type=str, default="24,28,32")
    parser.add_argument("--policies", type=str, default="prefix,suffix")
    parser.add_argument("--no-family-inference", action="store_true")
    parser.add_argument("--quiet", action="store_true")
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
    if len(pilot_policies) != 2:
        parser.error("--policies must contain exactly two policies.")
    window_scales = tuple(parse_scales(args.window_scales))
    if len(window_scales) != 3:
        parser.error("--window-scales must contain exactly three ascending scales.")

    phase_print(
        "Preparing Gate 2 loop holonomy pilot",
        f"variants={','.join(variants)} | window={','.join(str(value) for value in window_scales)}",
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
            "Selected runs do not contain the requested loop window/policies for: "
            + ", ".join(unsupported)
        )

    phase_print(
        "Analyzing cross-policy loop holonomy",
        f"{len(selected)} runs | policies={','.join(pilot_policies)}",
        quiet=args.quiet,
    )
    rows = build_holonomy_rows(selected, window_scales=window_scales, policies=pilot_policies)
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
        "stage": "phase2_loop_holonomy",
        "generated_at": generated_at,
        "selection": selection,
        "rows": rows,
        "notes": [
            "This pilot measures path dependence across a two-step tail loop.",
            "left path = dominant continuation under the left policy, returned to the low scale through the right projection.",
            "right path = dominant continuation under the right policy, returned to the low scale through the left projection.",
            "holonomy = disagreement between the two returned low-scale patterns.",
            "drift = either returned low-scale pattern differs from the original low-scale parent.",
        ],
    }
    summary_payload = {
        "generated_at": generated_at,
        "selection": selection,
        "compact_rows": [
            {
                "variant": row["variant"],
                "iteration": row["iteration"],
                "loopable_parent_fraction": row.get("loopable_parent_fraction"),
                "holonomy_rate": row.get("holonomy_rate"),
                "drift_rate": row.get("drift_rate"),
                "flat_loop_rate": row.get("flat_loop_rate"),
                "mean_cross_return_distance": row.get("mean_cross_return_distance"),
                "mean_left_return_drift": row.get("mean_left_return_drift"),
                "mean_right_return_drift": row.get("mean_right_return_drift"),
                "holonomy_entropy": row.get("holonomy_entropy"),
                "segment_consistency": row.get("segment_consistency"),
            }
            for row in rows
        ],
    }
    manifest_payload = {
        "run_slug": run_slug,
        "generated_at": generated_at,
        "script": "hsi_v2_phase2_loop_holonomy.py",
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
        "window_scales": list(first["window_scales"]),
        "policies": list(first["policies"]),
        "variants": [row["variant"] for row in rows],
        "family_inferred": family_inferred,
    }


def build_run_slug(selection: dict, timestamp: str) -> str:
    variants = "-".join(selection["variants"])
    iteration_values = selection["iterations"]
    iter_part = "iter-mixed" if len(iteration_values) != 1 else f"iter-{iteration_values[0]}"
    offset_bits = selection.get("segment_offset_bits", 0)
    offset_part = f"__off-{compact_int(offset_bits)}" if offset_bits else ""
    window_part = "-".join(str(value) for value in selection["window_scales"])
    policy_part = "-".join(selection["policies"])
    return (
        f"phase2-loop-holonomy__{iter_part}__seg-{selection['num_segments']}x{compact_int(selection['segment_bits'])}"
        f"{offset_part}__window-{window_part}__pol-{policy_part}__var-{variants}__{timestamp}"
    )


def phase_print(title: str, detail: str, *, quiet: bool) -> None:
    if quiet:
        return
    print(f"[Phase] {title}")
    print(f"        {detail}")
    print("")


def _supports_window(run: dict, *, window_scales: tuple[int, int, int], policies: list[str]) -> bool:
    run_dir = Path(run["_run_dir"])
    available = list(run["dataset"]["config"].get("scales", []))
    if any(value not in available for value in window_scales):
        return False
    for policy in policies:
        if policy not in run["dataset"]["config"].get("policies", []):
            return False
        fiber_path = run_dir / "fibers" / f"fibers_{policy}.json"
        if not fiber_path.exists():
            return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
