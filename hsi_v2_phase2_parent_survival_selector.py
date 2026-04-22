#!/usr/bin/env python3
"""
HSI v2 Phase 2 Parent Survival Selector

Derive a parent-survival frontier from an existing corrected
`hsi_v2_phase2_kernel_common_support.py` dataset artifact.
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
from v2.phase2.parent_survival import (
    build_parent_survival_rows,
    load_kernel_dataset,
    render_parent_survival_report,
    summarize_parent_survival_rows,
    write_parent_patterns_csv,
    write_parent_survival_csv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Derive a parent-survival frontier from a Phase 2 common-support kernel dataset."
    )
    parser.add_argument("--kernel-dataset", type=str, required=True)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/hsi_v2/phase2/parent_survival_selector",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    dataset_path = resolve_dir(args.kernel_dataset, anchor_file=__file__)
    if dataset_path.is_dir():
        dataset_path = dataset_path / "dataset.json"
    if not dataset_path.is_file():
        parser.error(f"Kernel dataset not found: {dataset_path}")

    output_dir = resolve_dir(args.output_dir, anchor_file=__file__)
    output_dir.mkdir(parents=True, exist_ok=True)

    phase_print("Loading kernel dataset", str(dataset_path), quiet=args.quiet)
    payload = load_kernel_dataset(dataset_path)
    rows, pattern_rows = build_parent_survival_rows(payload)
    grouped_summary = summarize_parent_survival_rows(rows)

    selection = dict(payload["selection"])
    selection["source_dataset_path"] = str(dataset_path)

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_slug = build_run_slug(selection, timestamp)
    run_dir = output_dir / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_path = run_dir / "summary.json"
    report_path = run_dir / "report.md"
    table_csv_path = run_dir / "table.csv"
    patterns_csv_path = run_dir / "patterns.csv"
    manifest_path = run_dir / "manifest.json"

    generated_at = datetime.now().isoformat(timespec="seconds")
    summary_payload = {
        "generated_at": generated_at,
        "selection": selection,
        "grouped_summary": grouped_summary,
        "rows": rows,
        "pattern_rows": pattern_rows,
    }
    manifest = {
        "generated_at": generated_at,
        "script": "hsi_v2_phase2_parent_survival_selector.py",
        "cwd": str(Path.cwd()),
        "inputs": {
            "kernel_dataset": str(dataset_path),
        },
        "outputs": {
            "summary": str(summary_path),
            "report": str(report_path),
            "table_csv": str(table_csv_path),
            "patterns_csv": str(patterns_csv_path),
            "manifest": str(manifest_path),
        },
        "arguments": vars(args),
    }

    phase_print("Writing artifacts", str(run_dir), quiet=args.quiet)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(render_parent_survival_report(selection, grouped_summary, rows) + "\n")
    write_parent_survival_csv(rows, table_csv_path)
    write_parent_patterns_csv(pattern_rows, patterns_csv_path)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    if not args.quiet:
        print(render_console_summary(rows))
        print("")
        print(f"Saved summary to: {summary_path}")
        print(f"Saved report to: {report_path}")
        print(f"Saved CSV table to: {table_csv_path}")
        print(f"Saved patterns CSV to: {patterns_csv_path}")
        print(f"Saved manifest to: {manifest_path}")

    return 0


def build_run_slug(selection: dict, timestamp: str) -> str:
    variants = "-".join(selection["variants"])
    return (
        f"phase2-parent-survival-selector__anchor-{selection['anchor_variant']}"
        f"__m-{selection['low_scale']}-{selection['high_scale']}"
        f"__sel-{selection['pattern_selection']}__top-{selection['top_patterns']}"
        f"__late-{compact_int(selection['anchor_offset_bits'])}"
        f"__var-{variants}__{timestamp}"
    )


def render_console_summary(rows: list[dict]) -> str:
    lines = [
        "Phase 2 parent survival selector",
        "-" * 152,
        f"{'variant':<12}{'source':<28}{'act_p':>8}{'surv_p':>8}{'exact_p':>8}{'dead_p':>8}{'dark_p':>8}{'part_p':>8}{'exact_m':>10}{'dead_m':>10}{'dark_m':>10}{'wj_cs':>10}",
    ]
    for row in rows:
        lines.append(
            f"{truncate_label(row['variant'], 12):<12}"
            f"{truncate_label(row['source_label'], 28):<28}"
            f"{row['active_parent_count']:>8}"
            f"{row['survivor_parent_count']:>8}"
            f"{row['exact_parent_count']:>8}"
            f"{row['dead_parent_count']:>8}"
            f"{row['dark_parent_count']:>8}"
            f"{row['partial_parent_count']:>8}"
            f"{fmt(row['exact_mass_fraction']):>10}"
            f"{fmt(row['dead_mass_fraction']):>10}"
            f"{fmt(row['dark_mass_fraction']):>10}"
            f"{fmt(row['kernel_wj_mean']):>10}"
        )
    return "\n".join(lines)


def phase_print(title: str, detail: str, *, quiet: bool) -> None:
    if quiet:
        return
    print(f"[Phase] {title}")
    print(f"        {detail}")
    print("")


def fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def truncate_label(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    if width <= 3:
        return value[:width]
    return value[: width - 3] + "..."


if __name__ == "__main__":
    raise SystemExit(main())
