#!/usr/bin/env python3
"""
HSI v2 Phase 2 Gate Report

Aggregate a small set of return-profile divergence summaries into one compact Gate 2 table.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from v2.common.cli import resolve_dir
from v2.phase2.gate_report import (
    build_gate_rows,
    load_profile_comparison,
    render_gate_report,
    write_gate_csv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate Phase 2 return-profile divergence summaries into a compact Gate 2 report."
    )
    parser.add_argument("--comparisons", type=str, nargs="+", required=True)
    parser.add_argument("--output-dir", type=str, default="results/hsi_v2/phase2")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    comparison_paths = []
    for raw_path in args.comparisons:
        path = resolve_dir(raw_path, anchor_file=__file__)
        if path.is_dir():
            path = path / "summary.json"
        if not path.is_file():
            parser.error(f"Comparison summary not found: {path}")
        comparison_paths.append(path)

    phase_print("Loading Gate 2 comparisons", f"{len(comparison_paths)} summaries", quiet=args.quiet)
    comparisons = [load_profile_comparison(path) for path in comparison_paths]
    rows = build_gate_rows(comparisons)

    output_dir = resolve_dir(args.output_dir, anchor_file=__file__)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = output_dir / f"phase2-gate-report__rows-{len(rows)}__{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_path = run_dir / "summary.json"
    report_path = run_dir / "report.md"
    csv_path = run_dir / "table.csv"
    manifest_path = run_dir / "manifest.json"

    generated_at = datetime.now().isoformat(timespec="seconds")
    payload = {
        "generated_at": generated_at,
        "rows": rows,
    }
    manifest = {
        "generated_at": generated_at,
        "script": "hsi_v2_phase2_gate_report.py",
        "cwd": str(Path.cwd()),
        "inputs": {
            "comparisons": [str(path) for path in comparison_paths],
        },
        "outputs": {
            "summary": str(summary_path),
            "report": str(report_path),
            "table_csv": str(csv_path),
            "manifest": str(manifest_path),
        },
        "arguments": vars(args),
    }

    phase_print("Writing artifacts", str(run_dir), quiet=args.quiet)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(render_gate_report(rows) + "\n")
    write_gate_csv(rows, csv_path)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    if not args.quiet:
        print(render_console_summary(rows))
        print("")
        print(f"Saved summary to: {summary_path}")
        print(f"Saved report to: {report_path}")
        print(f"Saved CSV table to: {csv_path}")
        print(f"Saved manifest to: {manifest_path}")

    return 0


def render_console_summary(rows: list[dict]) -> str:
    lines = [
        "Phase 2 Gate Report",
        "-" * 128,
        f"{'null':<18}{'m':>6}{'shared':>8}{'union':>8}{'sel_J':>10}{'prof_wJ':>10}{'prof_JS':>10}{'d_mean':>10}{'top_eq':>10}",
    ]
    for row in rows:
        lines.append(
            f"{row['null_variant']:<18}"
            f"{row['pattern_scale']:>6}"
            f"{row['shared_pattern_count']:>8}"
            f"{row['union_pattern_count']:>8}"
            f"{row['selection_jaccard']:>10.4f}"
            f"{row['weighted_profile_jaccard']:>10.4f}"
            f"{row['weighted_profile_js_divergence']:>10.4f}"
            f"{row['weighted_relative_mean_lag_gap']:>10.4f}"
            f"{row['weighted_top_bin_match_fraction']:>10.4f}"
        )
    return "\n".join(lines)


def phase_print(title: str, detail: str, *, quiet: bool) -> None:
    if quiet:
        return
    print(f"[Phase] {title}")
    print(f"        {detail}")
    print("")


if __name__ == "__main__":
    raise SystemExit(main())
