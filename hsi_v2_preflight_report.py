#!/usr/bin/env python3
"""
HSI v2 Preflight Report

Scan preflight factor-complexity runs and build a compact comparison report
for the latest coherent batch across variants.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from v2.preflight.report import (
    build_report_rows,
    discover_preflight_runs,
    filter_runs,
    infer_family_from_latest_run,
    render_console_summary,
    render_markdown_report,
    select_latest_per_variant,
    split_rows_by_role,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a compact report from HSI v2 preflight factor-complexity runs."
    )
    parser.add_argument(
        "--preflight-dir",
        type=str,
        default="results/hsi_v2/preflight",
        help="Directory containing preflight run folders.",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="results/hsi_v2/preflight/reports",
        help="Directory where summary artifacts will be written.",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="",
        help="Optional comma-separated variant list, e.g. B,E,I,F,A.",
    )
    parser.add_argument("--iteration", type=int, default=None, help="Optional iteration filter.")
    parser.add_argument("--segment-bits", type=int, default=None, help="Optional segment size filter.")
    parser.add_argument("--num-segments", type=int, default=None, help="Optional segment-count filter.")
    parser.add_argument("--m-min", type=int, default=None, help="Optional minimum m filter.")
    parser.add_argument("--m-max", type=int, default=None, help="Optional maximum m filter.")
    parser.add_argument(
        "--nulls",
        type=str,
        default="",
        help="Optional exact null list filter, comma-separated.",
    )
    parser.add_argument(
        "--no-family-inference",
        action="store_true",
        help="Do not infer a coherent batch from the latest run when explicit filters are missing.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress console tables after writing artifacts.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    preflight_dir = resolve_dir(args.preflight_dir)
    report_dir = resolve_dir(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    variants = parse_csv(args.variants)
    nulls = parse_csv(args.nulls)

    runs = discover_preflight_runs(preflight_dir)
    if not runs:
        parser.error(f"No valid preflight runs found in {preflight_dir}")

    family = None
    explicit_filters = any(
        value is not None
        for value in (args.iteration, args.segment_bits, args.num_segments, args.m_min, args.m_max)
    ) or bool(nulls)

    if not args.no_family_inference and not explicit_filters:
        family = infer_family_from_latest_run(runs)

    matching_runs = filter_runs(
        runs,
        variants=variants or None,
        iteration=args.iteration,
        segment_bits=args.segment_bits,
        num_segments=args.num_segments,
        m_min=args.m_min,
        m_max=args.m_max,
        nulls=nulls or None,
        family=family,
    )

    if not matching_runs:
        parser.error("No preflight runs matched the requested selection.")

    selected = select_latest_per_variant(matching_runs, variant_order=variants or None)
    rows = build_report_rows(selected)

    if not rows:
        parser.error("No latest-per-variant runs available after filtering.")

    selection = {
        "iteration": rows[0]["iteration"],
        "segment_bits": rows[0]["segment_bits"],
        "num_segments": rows[0]["num_segments"],
        "m_min": int(rows[0]["m_range"].split("-")[0]),
        "m_max": int(rows[0]["m_range"].split("-")[1]),
        "nulls": list(selected[0]["config"].get("nulls", [])),
        "variants": [row["variant"] for row in rows],
        "family_inferred": family is not None,
    }

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
            print("Using the latest coherent batch inferred from the newest run.")
            print("")
        print(render_console_summary(rows))
        print(f"\nSaved JSON summary to: {json_path}")
        print(f"Saved Markdown report to: {md_path}")

    return 0


def parse_csv(raw: str) -> list[str]:
    if not raw.strip():
        return []
    return [item.strip().upper() if len(item.strip()) == 1 else item.strip().lower() for item in raw.split(",") if item.strip()]


def resolve_dir(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = Path(__file__).parent / path
    return path.resolve()


def build_report_slug(selection: dict, timestamp: str) -> str:
    variants = "-".join(selection["variants"])
    nulls = "-".join(selection["nulls"]) if selection["nulls"] else "observed"
    return (
        f"preflight-report__iter-{selection['iteration']}__seg-{selection['num_segments']}x{selection['segment_bits']}"
        f"__m-{selection['m_min']}-{selection['m_max']}__var-{variants}__ctrl-{nulls}__{timestamp}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
