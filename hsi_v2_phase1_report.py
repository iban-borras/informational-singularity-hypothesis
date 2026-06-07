#!/usr/bin/env python3
"""
HSI v2 Phase 1 Report

Build a compact comparative report from Phase 1 runs:
- primary contrast: B, F, A, M, N
- supporting variants: E, I, J, K, L
- outputs in console, JSON, and Markdown
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from v2.common.cli import parse_variants, resolve_results_dir
from v2.phase1.report import (
    build_report_rows,
    discover_phase1_runs,
    filter_runs,
    infer_family_from_latest_run,
    render_console_summary,
    render_markdown_report,
    select_latest_per_variant,
    split_rows_by_role,
)
from v2.phase1.tower import parse_policies, parse_scales


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a compact report from HSI v2 Phase 1 runs."
    )
    parser.add_argument(
        "--phase1-dir",
        type=str,
        default="results/hsi_v2/phase1",
        help="Directory containing Phase 1 run folders.",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="results/hsi_v2/phase1/reports",
        help="Directory where summary artifacts will be written.",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="",
        help="Optional comma-separated variant list, e.g. B,F,A,M,N.",
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
        help="Optional exact scale ladder filter, e.g. 8,12,16,20,24,28,32.",
    )
    parser.add_argument(
        "--policies",
        type=str,
        default="",
        help="Optional exact policy filter, e.g. prefix,suffix.",
    )
    parser.add_argument(
        "--no-family-inference",
        action="store_true",
        help="Do not infer a coherent batch from the newest available Phase 1 run.",
    )
    parser.add_argument(
        "--with-hints",
        action="store_true",
        help="Include heuristic interpretive labels in the report output.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress console tables after writing artifacts.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    phase1_dir = resolve_results_dir(args.phase1_dir, anchor_file=__file__)
    report_dir = resolve_results_dir(args.report_dir, anchor_file=__file__)
    report_dir.mkdir(parents=True, exist_ok=True)

    variants = parse_variants(args.variants)
    scales = parse_scales(args.scales) if args.scales.strip() else None
    policies = parse_policies(args.policies) if args.policies.strip() else None

    runs = discover_phase1_runs(phase1_dir)
    if not runs:
        parser.error(f"No valid Phase 1 runs found in {phase1_dir}")

    family = None
    explicit_filters = any(
        value is not None for value in (args.iteration, args.segment_bits, args.num_segments, args.segment_offset_bits)
    ) or scales is not None or policies is not None
    if not args.no_family_inference and not explicit_filters:
        family = infer_family_from_latest_run(runs)

    matching_runs = filter_runs(
        runs,
        variants=variants or None,
        iteration=args.iteration,
        segment_bits=args.segment_bits,
        num_segments=args.num_segments,
        segment_offset_bits=args.segment_offset_bits,
        scales=scales,
        policies=policies,
        family=family,
    )
    if not matching_runs:
        parser.error("No Phase 1 runs matched the requested selection.")

    selected = select_latest_per_variant(matching_runs, variant_order=variants or None)
    try:
        rows = build_report_rows(selected, include_hints=args.with_hints)
    except ValueError as exc:
        parser.error(str(exc))
    if not rows:
        parser.error("No latest-per-variant runs available after filtering.")

    selection = build_selection(
        rows,
        family_inferred=family is not None,
        include_hints=args.with_hints,
    )
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    slug = build_report_slug(selection, timestamp)

    json_path = report_dir / f"{slug}.json"
    md_path = report_dir / f"{slug}.md"

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "selection": selection,
        "rows": rows,
        "grouped_rows": split_rows_by_role(rows),
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    markdown = render_markdown_report(rows, selection)
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(markdown + "\n")

    if not args.quiet:
        if selection["family_inferred"]:
            print("Using the latest coherent Phase 1 batch inferred from the newest run.")
            print("")
        print(render_console_summary(rows, include_hints=args.with_hints))
        print(f"\nSaved JSON summary to: {json_path}")
        print(f"Saved Markdown report to: {md_path}")

    return 0


def build_selection(rows: list[dict], *, family_inferred: bool, include_hints: bool) -> dict:
    iterations = sorted({row["iteration"] for row in rows if row["iteration"] is not None})
    segment_bits = sorted({row["segment_bits"] for row in rows})
    num_segments = sorted({row["num_segments"] for row in rows})
    segment_offset_bits = sorted({row["segment_offset_bits"] for row in rows})
    scales = rows[0]["scales"]
    policies = rows[0]["policies"]
    return {
        "iterations": iterations,
        "segment_bits": segment_bits[0] if len(segment_bits) == 1 else "mixed",
        "num_segments": num_segments[0] if len(num_segments) == 1 else "mixed",
        "segment_offset_bits": segment_offset_bits[0] if len(segment_offset_bits) == 1 else "mixed",
        "scales": scales,
        "policies": policies,
        "variants": [row["variant"] for row in rows],
        "family_inferred": family_inferred,
        "include_hints": include_hints,
    }


def build_report_slug(selection: dict, timestamp: str) -> str:
    variants = "-".join(selection["variants"])
    iteration_part = (
        str(selection["iterations"][0])
        if len(selection["iterations"]) == 1
        else "mixed"
    )
    scale_part = "-".join(str(value) for value in selection["scales"])
    policy_part = "-".join(selection["policies"])
    offset_part = ""
    if selection["segment_offset_bits"] not in (0, "mixed"):
        offset_part = f"__off-{selection['segment_offset_bits']}"
    elif selection["segment_offset_bits"] == "mixed":
        offset_part = "__off-mixed"
    return (
        f"phase1-report__iter-{iteration_part}__seg-{selection['num_segments']}x{selection['segment_bits']}"
        f"{offset_part}__m-{scale_part}__proj-{policy_part}__var-{variants}__{timestamp}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
