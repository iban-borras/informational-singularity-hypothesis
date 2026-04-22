#!/usr/bin/env python3
"""
HSI v2 Phase 2 Kernel Throughput Readout

Derive a transport-throughput readout from an existing
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
from v2.phase2.kernel_throughput import (
    build_throughput_rows,
    load_kernel_dataset,
    render_throughput_report,
    summarize_throughput_rows,
    write_throughput_csv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Derive a throughput readout from a Phase 2 common-support kernel dataset."
    )
    parser.add_argument("--kernel-dataset", type=str, required=True)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/hsi_v2/phase2/kernel_throughput_readout",
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
    rows = build_throughput_rows(payload)
    grouped_summary = summarize_throughput_rows(rows)

    selection = dict(payload["selection"])
    selection["source_dataset_path"] = str(dataset_path)

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_slug = build_run_slug(selection, timestamp)
    run_dir = output_dir / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_path = run_dir / "summary.json"
    report_path = run_dir / "report.md"
    csv_path = run_dir / "table.csv"
    manifest_path = run_dir / "manifest.json"

    generated_at = datetime.now().isoformat(timespec="seconds")
    summary_payload = {
        "generated_at": generated_at,
        "selection": selection,
        "grouped_summary": grouped_summary,
        "rows": rows,
    }
    manifest = {
        "generated_at": generated_at,
        "script": "hsi_v2_phase2_kernel_throughput_readout.py",
        "cwd": str(Path.cwd()),
        "inputs": {
            "kernel_dataset": str(dataset_path),
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
        json.dump(summary_payload, handle, indent=2)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(render_throughput_report(selection, grouped_summary, rows) + "\n")
    write_throughput_csv(rows, csv_path)
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


def build_run_slug(selection: dict, timestamp: str) -> str:
    variants = "-".join(selection["variants"])
    return (
        f"phase2-kernel-throughput-readout__anchor-{selection['anchor_variant']}"
        f"__m-{selection['low_scale']}-{selection['high_scale']}"
        f"__sel-{selection['pattern_selection']}__top-{selection['top_patterns']}"
        f"__late-{compact_int(selection['anchor_offset_bits'])}"
        f"__var-{variants}__{timestamp}"
    )


def render_console_summary(rows: list[dict]) -> str:
    lines = [
        "Phase 2 kernel throughput readout",
        "-" * 140,
        f"{'variant':<12}{'source':<28}{'ev_a':>10}{'ev_x':>10}{'ret_x':>10}{'gap_x':>10}{'ret_min':>10}{'ret_max':>10}{'wj_cs':>10}{'top_f':>10}",
    ]
    for row in rows:
        lines.append(
            f"{truncate_label(row['variant'], 12):<12}"
            f"{truncate_label(row['source_label'], 28):<28}"
            f"{row['anchor_event_total']:>10}"
            f"{row['candidate_event_total']:>10}"
            f"{fmt(row['throughput_retention_ratio']):>10}"
            f"{fmt(row['throughput_gap_signed']):>10}"
            f"{fmt(row['parent_retention_min']):>10}"
            f"{fmt(row['parent_retention_max']):>10}"
            f"{fmt(row['kernel_wj_mean']):>10}"
            f"{fmt(row['top_child_full_match_mass']):>10}"
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
