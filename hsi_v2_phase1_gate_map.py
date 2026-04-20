#!/usr/bin/env python3
"""
HSI v2 Phase 1 Gate Map

Build a compact Porta 1 summary table plus a 2D visual plane that foregrounds:
- selectivity at the top of the tower
- transported tail activity
- the separation between HSI-like persistence and strong-null persistence
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from v2.common.cli import resolve_dir
from v2.phase1.gate_map import (
    DEFAULT_NULL_VARIANTS,
    DEFAULT_OBSERVED_VARIANTS,
    load_gate_map_rows,
    render_console_summary,
    render_gate_png,
    render_gate_svg,
    render_markdown_report,
    write_csv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a compact Porta 1 gate-map table and 2D visual summary."
    )
    parser.add_argument(
        "--observed-dir",
        type=str,
        default="results/hsi_v2/phase1",
        help="Directory containing canonical observed Phase 1 runs.",
    )
    parser.add_argument(
        "--null-dir",
        type=str,
        default="results/hsi_v2/phase1/nulls",
        help="Directory containing strong-null Phase 1 runs.",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="results/hsi_v2/phase1/reports",
        help="Directory where gate-map artifacts will be written.",
    )
    parser.add_argument(
        "--observed-variants",
        type=str,
        default=",".join(DEFAULT_OBSERVED_VARIANTS),
        help="Comma-separated observed variants to include.",
    )
    parser.add_argument(
        "--null-variants",
        type=str,
        default=",".join(DEFAULT_NULL_VARIANTS),
        help="Comma-separated strong-null variants to include.",
    )
    parser.add_argument(
        "--tail-edges",
        type=int,
        default=2,
        help="Number of tail edges used for coherence/transport summaries.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console table after writing artifacts.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    observed_dir = resolve_dir(args.observed_dir, anchor_file=__file__)
    null_dir = resolve_dir(args.null_dir, anchor_file=__file__)
    report_dir = resolve_dir(args.report_dir, anchor_file=__file__)
    report_dir.mkdir(parents=True, exist_ok=True)

    observed_variants = _parse_variant_list(args.observed_variants)
    null_variants = _parse_variant_list(args.null_variants)

    try:
        rows, selection = load_gate_map_rows(
            observed_dir,
            null_dir,
            observed_variants=observed_variants,
            null_variants=null_variants,
            tail_edges=args.tail_edges,
        )
    except ValueError as exc:
        parser.error(str(exc))

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    slug = (
        f"phase1-gate-map__obs-{len(selection['observed_variants'])}"
        f"__null-{len(selection['null_variants'])}"
        f"__tail-{selection['tail_edges']}__{timestamp}"
    )

    json_path = report_dir / f"{slug}.json"
    md_path = report_dir / f"{slug}.md"
    csv_path = report_dir / f"{slug}.csv"
    svg_path = report_dir / f"{slug}.svg"
    png_path = report_dir / f"{slug}.png"

    payload = {
        "generated_at": selection["generated_at"],
        "selection": selection,
        "rows": rows,
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    write_csv(rows, csv_path)

    svg = render_gate_svg(rows)
    with open(svg_path, "w", encoding="utf-8") as handle:
        handle.write(svg + "\n")

    png_written = render_gate_png(rows, png_path)

    markdown = render_markdown_report(
        rows,
        selection,
        svg_filename=png_path.name if png_written else svg_path.name,
    )
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(markdown + "\n")

    if not args.quiet:
        print(render_console_summary(rows))
        print(f"\nSaved JSON summary to: {json_path}")
        print(f"Saved CSV table to: {csv_path}")
        print(f"Saved SVG figure to: {svg_path}")
        if png_written:
            print(f"Saved PNG figure to: {png_path}")
        print(f"Saved Markdown report to: {md_path}")

    return 0


def _parse_variant_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
