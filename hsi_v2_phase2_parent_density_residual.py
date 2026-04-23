#!/usr/bin/env python3
"""
HSI v2 Phase 2 Parent Density Residual Readout

Derived readout over a canonical parent-survival revalidation artifact.
Measures partition-level mass retention and survivor-internal density deformation.
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
from v2.phase2.parent_density_residual import (
    build_parent_density_residual_rows,
    render_parent_density_residual_report,
    summarize_parent_density_residual_rows,
    write_parent_density_residual_csv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Derive a residual density readout from a canonical parent-survival "
            "revalidation artifact."
        )
    )
    parser.add_argument("--revalidation-run", type=str, required=True)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/hsi_v2/phase2/parent_density_residual_readout",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    revalidation_path = resolve_dir(args.revalidation_run, anchor_file=__file__)
    if revalidation_path.is_dir():
        revalidation_path = revalidation_path / "summary.json"
    if not revalidation_path.is_file():
        parser.error(f"Revalidation summary not found: {revalidation_path}")

    output_dir = resolve_dir(args.output_dir, anchor_file=__file__)
    output_dir.mkdir(parents=True, exist_ok=True)

    phase_print("Loading canonical revalidation", str(revalidation_path), quiet=args.quiet)
    revalidation_summary = load_json(revalidation_path)
    lagaware_summary_path = Path(revalidation_summary["lagaware_summary_path"])
    lagaware_dataset_path = lagaware_summary_path.with_name("dataset.json")
    if not lagaware_dataset_path.is_file():
        parser.error(f"Lag-aware dataset not found: {lagaware_dataset_path}")
    lagaware_payload = load_json(lagaware_dataset_path)
    lagaware_payload["_dataset_path"] = str(lagaware_dataset_path)

    probe_summary = revalidation_summary["probe_summary"]
    rows = build_parent_density_residual_rows(
        lagaware_payload,
        anchor_core_patterns=set(probe_summary["anchor_core_patterns"]),
        anchor_shell_patterns=set(probe_summary["anchor_shell_patterns"]),
    )
    grouped_summary = summarize_parent_density_residual_rows(rows)

    selection = build_selection(revalidation_summary, revalidation_path)

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
    manifest_payload = {
        "generated_at": generated_at,
        "script": "hsi_v2_phase2_parent_density_residual.py",
        "cwd": str(Path.cwd()),
        "inputs": {
            "revalidation_summary": str(revalidation_path),
            "lagaware_dataset": str(lagaware_dataset_path),
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
        handle.write(render_parent_density_residual_report(selection, grouped_summary, rows) + "\n")
    write_parent_density_residual_csv(rows, csv_path)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, indent=2)

    if not args.quiet:
        print(render_console_summary(grouped_summary))
        print("")
        print(f"Saved summary to: {summary_path}")
        print(f"Saved report to: {report_path}")
        print(f"Saved CSV table to: {csv_path}")
        print(f"Saved manifest to: {manifest_path}")

    return 0


def build_selection(revalidation_summary: dict, revalidation_path: Path) -> dict:
    selection = dict(revalidation_summary["selection"])
    selection["revalidation_summary_path"] = str(revalidation_path)
    selection["candidate_lag_bits"] = revalidation_summary["recommended_lag_bits"]
    selection["probe_summary_path"] = revalidation_summary["probe_summary_path"]
    selection["lagaware_summary_path"] = revalidation_summary["lagaware_summary_path"]
    return selection


def build_run_slug(selection: dict, timestamp: str) -> str:
    offsets = selection["offsets"]
    if len(offsets) == 1:
        offset_part = f"off-{compact_int(offsets[0])}"
    else:
        offset_part = f"off-{compact_int(min(offsets))}-plus-{len(offsets)}"
    return (
        f"phase2-parent-density-residual__anchor-{selection['anchor_variant']}"
        f"__cand-{selection['candidate_variant']}__lag-{compact_int(selection['candidate_lag_bits'])}"
        f"__m-{selection['low_scale']}-{selection['high_scale']}"
        f"__sel-{selection['pattern_selection']}__top-{selection['top_patterns']}"
        f"__{offset_part}__{timestamp}"
    )


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def render_console_summary(grouped_summary: list[dict]) -> str:
    lines = [
        "Phase 2 parent density residual readout",
        "-" * 164,
        (
            f"{'variant':<10}{'source':<20}{'partition':<14}{'active':>9}"
            f"{'exact':>9}{'surv':>9}{'partRet':>10}{'survRet':>10}"
            f"{'survDef':>10}{'survBias':>10}{'deadM':>10}{'darkM':>10}"
        ),
    ]
    for item in grouped_summary:
        active_label = f"{item['active_offset_count']}/{item['row_count']}"
        exact_label = f"{item['exact_offset_count']}/{item['row_count']}"
        survivor_label = f"{item['survivor_offset_count']}/{item['row_count']}"
        lines.append(
            f"{truncate_label(item['variant'], 10):<10}"
            f"{truncate_label(item['source_label'], 20):<20}"
            f"{truncate_label(item['partition_label'], 14):<14}"
            f"{active_label:>9}"
            f"{exact_label:>9}"
            f"{survivor_label:>9}"
            f"{fmt(item['partition_mass_retention_pooled']):>10}"
            f"{fmt(item['survivor_internal_retention_pooled']):>10}"
            f"{fmt(item['survivor_internal_deformation_log2_mean_pooled']):>10}"
            f"{fmt(item['survivor_internal_bias_log2_mean_pooled']):>10}"
            f"{fmt(item['dead_anchor_mass_fraction_pooled']):>10}"
            f"{fmt(item['dark_anchor_mass_fraction_pooled']):>10}"
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
