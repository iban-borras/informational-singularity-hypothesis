#!/usr/bin/env python3
"""
HSI v2 Phase 2 Transport-Defect Pilot

Minimal executable N2-01 pilot:
- loads low/high bridge-linked return-lag datasets
- reconstructs an observable transport kernel from Phase 1 fibers
- transports high-scale return profiles down to the lower selected patterns
- measures the first transport-defect tuple
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from v2.common.cli import parse_variants, resolve_dir
from v2.phase2.transport_defect import (
    build_transport_defect_rows,
    render_console_summary,
    render_markdown_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the HSI v2 N2-01 transport-defect pilot from existing return-lag datasets."
    )
    parser.add_argument("--low-dataset", type=str, required=True)
    parser.add_argument("--high-dataset", type=str, required=True)
    parser.add_argument("--variants", type=str, default="B")
    parser.add_argument("--output-dir", type=str, default="results/hsi_v2/phase2")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    variants = parse_variants(args.variants)
    if not variants:
        parser.error("At least one variant is required.")

    low_dataset = resolve_input_dataset(args.low_dataset)
    high_dataset = resolve_input_dataset(args.high_dataset)
    if not low_dataset.is_file():
        parser.error(f"Low-scale dataset not found: {low_dataset}")
    if not high_dataset.is_file():
        parser.error(f"High-scale dataset not found: {high_dataset}")

    output_dir = resolve_dir(args.output_dir, anchor_file=__file__)
    output_dir.mkdir(parents=True, exist_ok=True)

    phase_print(
        "Preparing N2-01 transport-defect pilot",
        f"variants={','.join(variants)}",
        quiet=args.quiet,
    )
    rows = build_transport_defect_rows(
        low_dataset_path=low_dataset,
        high_dataset_path=high_dataset,
        variants=variants,
    )
    selection = build_selection(rows)

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
        "stage": "phase2_transport_defect",
        "generated_at": generated_at,
        "selection": selection,
        "rows": rows,
        "notes": [
            "This pilot builds a bridge-aware transport kernel from the direct Phase 1 48->40 fiber edge.",
            "Transport is restricted to the selected high-scale bridge-linked return-profile patterns.",
            "The defect tuple compares observed low-scale return profiles against transported high-scale mixtures.",
        ],
    }
    summary_payload = {
        "generated_at": generated_at,
        "selection": selection,
        "compact_rows": [
            {
                "variant": row["variant"],
                "source_kind": row["source_kind"],
                "null_model": row["null_model"],
                "low_scale": row["low_scale"],
                "high_scale": row["high_scale"],
                "connected_pattern_count": row["connected_pattern_count"],
                "terminal_fraction": row["terminal_fraction"],
                "defect_js_mean": row["defect_js_mean"],
                "defect_wj_mean": row["defect_wj_mean"],
                "defect_flow_mean": row["defect_flow_mean"],
                "defect_top_mass": row["defect_top_mass"],
                "defect_mean_gap": row["defect_mean_gap"],
                "segment_defect_stability": row["segment_defect_stability"],
            }
            for row in rows
        ],
    }
    manifest_payload = {
        "run_slug": run_slug,
        "generated_at": generated_at,
        "script": "hsi_v2_phase2_transport_defect.py",
        "cwd": str(Path.cwd()),
        "inputs": {
            "low_dataset": str(low_dataset),
            "high_dataset": str(high_dataset),
            "variants": variants,
        },
        "outputs": {
            "dataset": str(dataset_path),
            "summary": str(summary_path),
            "manifest": str(manifest_path),
            "report": str(report_path),
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
        print(render_console_summary(rows))
        print("")
        print(f"Saved dataset to: {dataset_path}")
        print(f"Saved summary to: {summary_path}")
        print(f"Saved report to: {report_path}")
        print(f"Saved manifest to: {manifest_path}")

    return 0


def resolve_input_dataset(raw_path: str) -> Path:
    path = resolve_dir(raw_path, anchor_file=__file__)
    if path.is_dir():
        return path / "dataset.json"
    return path


def build_selection(rows: list[dict]) -> dict:
    first = rows[0]
    return {
        "variants": [row["variant"] for row in rows],
        "low_scale": first["low_scale"],
        "high_scale": first["high_scale"],
        "pattern_selection": first["pattern_selection"],
        "phase1_policies": list(first.get("phase1_policies", [])),
    }


def build_run_slug(selection: dict, timestamp: str) -> str:
    variant_part = "-".join(selection["variants"])
    return (
        f"phase2-transport-defect__m-{selection['low_scale']}-{selection['high_scale']}"
        f"__sel-{selection['pattern_selection']}__var-{variant_part}__{timestamp}"
    )


def phase_print(title: str, detail: str, *, quiet: bool) -> None:
    if quiet:
        return
    print(f"[Phase] {title}")
    print(f"        {detail}")
    print("")


if __name__ == "__main__":
    raise SystemExit(main())
