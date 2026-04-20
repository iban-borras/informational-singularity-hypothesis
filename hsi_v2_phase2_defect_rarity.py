#!/usr/bin/env python3
"""
HSI v2 Phase 2 Defect Rarity Against a Reference Null

Small post-analysis layer on top of existing Phase 2 defect-pilot outputs.
It compares observed defect-family weights against a reference null and measures
how much observed defect mass remains genuinely rare with respect to that null.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from v2.common.cli import resolve_dir
from v2.common.naming import compact_int
from v2.phase2.rarity import (
    build_rarity_report,
    load_phase2_dataset,
    render_console_summary,
    render_markdown_report,
    select_row,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare Phase 2 defect-family weights against a reference null run."
    )
    parser.add_argument(
        "--observed-run",
        type=str,
        required=True,
        help="Observed Phase 2 run directory or dataset.json path.",
    )
    parser.add_argument(
        "--null-run",
        type=str,
        required=True,
        help="Reference-null Phase 2 run directory or dataset.json path.",
    )
    parser.add_argument(
        "--observed-variant",
        type=str,
        default="",
        help="Optional variant selector inside the observed dataset if it contains multiple rows.",
    )
    parser.add_argument(
        "--null-variant",
        type=str,
        default="",
        help="Optional variant selector inside the null dataset if it contains multiple rows.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/hsi_v2/phase2",
        help="Directory where rarity comparison artifacts will be written.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress console table after writing artifacts.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    observed_run = resolve_input_path(args.observed_run)
    null_run = resolve_input_path(args.null_run)
    output_dir = resolve_dir(args.output_dir, anchor_file=__file__)
    output_dir.mkdir(parents=True, exist_ok=True)

    phase_print(
        "Preparing defect-rarity comparison",
        f"observed={observed_run.name} | null={null_run.name}",
        quiet=args.quiet,
    )

    observed_payload = load_phase2_dataset(observed_run)
    null_payload = load_phase2_dataset(null_run)
    observed_row = select_row(observed_payload, args.observed_variant.strip() or None)
    null_row = select_row(null_payload, args.null_variant.strip() or None)

    report = build_rarity_report(observed_row, null_row)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_slug = build_run_slug(report, timestamp)
    run_dir = output_dir / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = run_dir / "dataset.json"
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "manifest.json"
    report_path = run_dir / "report.md"

    generated_at = datetime.now().isoformat(timespec="seconds")
    dataset_payload = {
        "stage": "phase2_defect_rarity",
        "generated_at": generated_at,
        "observed_run": str(observed_run),
        "null_run": str(null_run),
        "report": report,
        "notes": [
            "rarity_mass_fraction = observed defect mass not explained by the reference null family weights",
            "weighted_jaccard = shared defect mass divided by max-weight family union",
        ],
    }
    summary_payload = {
        "generated_at": generated_at,
        "observed_variant": report["observed_variant"],
        "null_variant": report["null_variant"],
        "window_scales": report["window_scales"],
        "policies": report["policies"],
        "policy_reports": report["policy_reports"],
    }
    manifest_payload = {
        "run_slug": run_slug,
        "generated_at": generated_at,
        "script": "hsi_v2_phase2_defect_rarity.py",
        "cwd": str(Path.cwd()),
        "inputs": {
            "observed_run": str(observed_run),
            "null_run": str(null_run),
        },
        "arguments": vars(args),
        "outputs": {
            "dataset": str(dataset_path),
            "summary": str(summary_path),
            "manifest": str(manifest_path),
            "report": str(report_path),
        },
    }

    phase_print("Writing artifacts", str(run_dir), quiet=args.quiet)
    with open(dataset_path, "w", encoding="utf-8") as handle:
        json.dump(dataset_payload, handle, indent=2)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, indent=2)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(render_markdown_report(report) + "\n")

    if not args.quiet:
        print(render_console_summary(report))
        print(f"\nSaved dataset to: {dataset_path}")
        print(f"Saved summary to: {summary_path}")
        print(f"Saved manifest to: {manifest_path}")
        print(f"Saved report to: {report_path}")

    return 0


def build_run_slug(report: dict, timestamp: str) -> str:
    window = "-".join(str(value) for value in report["window_scales"])
    policy_part = "-".join(report["policies"])
    return (
        f"phase2-defect-rarity__obs-{report['observed_variant']}__null-{report['null_variant']}"
        f"__iter-{compact_int(report['observed_iteration'])}__window-{window}__pol-{policy_part}__{timestamp}"
    )


def phase_print(title: str, detail: str, *, quiet: bool) -> None:
    if quiet:
        return
    print(f"\n[Phase] {title}")
    print(f"        {detail}")


def resolve_input_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    if path.exists():
        return path.resolve()
    return resolve_dir(raw_path, anchor_file=__file__)


if __name__ == "__main__":
    raise SystemExit(main())
