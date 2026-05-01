#!/usr/bin/env python3
"""HSI v2 Phase 3 minimal defect/spectrum pilot."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from v2.common.cli import parse_variants, resolve_dir
from v2.common.naming import compact_int
from v2.phase1.report import filter_runs, infer_family_from_latest_run, select_latest_per_variant
from v2.phase1.tower import parse_scales
from v2.phase2.null_pressure import discover_phase1_runs_recursive
from v2.phase3.defect_spectrum import (
    build_defect_spectrum_rows,
    compact_rows,
    render_console_summary,
    render_markdown_report,
)


CSV_FIELDS = [
    "variant",
    "source_kind",
    "null_model",
    "null_seed",
    "iteration",
    "future_bridge_fraction",
    "squareable_bridge_fraction",
    "anchored_square_rate",
    "square_defect_rate",
    "square_defect_mass_rate",
    "bridge_disagreement_rate",
    "mean_square_distance",
    "square_entropy",
    "square_segment_consistency",
    "node_count",
    "edge_count",
    "largest_component_fraction",
    "recurrent_edge_fraction",
    "recurrent_edge_mass_fraction",
    "core_node_fraction",
    "degree_entropy",
    "hub_mass_fraction",
    "graph_segment_core_jaccard",
    "lambda2_norm_laplacian",
    "zero_eigenvalue_count",
    "spectral_radius_adjacency",
    "spectral_status",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the HSI v2 Phase 3 minimal defect/spectrum pilot from existing "
            "Phase 1 projective-tower artifacts."
        )
    )
    parser.add_argument("--phase1-dir", default="results/hsi_v2/phase1_high_scales")
    parser.add_argument("--output-dir", default="results/hsi_v2/phase3/min_defect_spectrum")
    parser.add_argument("--variants", default="B,E,B-markov1,B-matched-lz")
    parser.add_argument("--iteration", type=int, default=None)
    parser.add_argument("--segment-bits", type=int, default=None)
    parser.add_argument("--num-segments", type=int, default=None)
    parser.add_argument("--segment-offset-bits", type=int, default=None)
    parser.add_argument("--scales", default="")
    parser.add_argument("--window-scales", default="24,28,32,40")
    parser.add_argument("--left-policy", default="prefix")
    parser.add_argument("--right-policy", default="suffix")
    parser.add_argument("--min-segment-recurrence", type=int, default=2)
    parser.add_argument("--top-hub-k", type=int, default=8)
    parser.add_argument("--no-family-inference", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    variants = parse_variants(args.variants)
    if not variants:
        parser.error("At least one variant/source is required.")
    if args.left_policy == args.right_policy:
        parser.error("--left-policy and --right-policy must differ.")
    if args.min_segment_recurrence < 1:
        parser.error("--min-segment-recurrence must be at least 1.")
    if args.top_hub_k < 1:
        parser.error("--top-hub-k must be at least 1.")

    phase1_dir = resolve_dir(args.phase1_dir, anchor_file=__file__)
    output_root = resolve_dir(args.output_dir, anchor_file=__file__)
    output_root.mkdir(parents=True, exist_ok=True)

    phase1_scales = parse_scales(args.scales) if args.scales.strip() else None
    window_scales = tuple(parse_scales(args.window_scales))
    if len(window_scales) != 4:
        parser.error("--window-scales must contain exactly four ascending scales.")
    if sorted(window_scales) != list(window_scales) or len(set(window_scales)) != 4:
        parser.error("--window-scales must contain four strictly ascending scales.")

    phase_print(
        "Preparing Phase 3 minimal defect/spectrum pilot",
        (
            f"variants={','.join(variants)} | window={','.join(str(value) for value in window_scales)} | "
            f"pol={args.left_policy}-{args.right_policy}"
        ),
        quiet=args.quiet,
    )

    runs = discover_phase1_runs_recursive(phase1_dir)
    if not runs:
        parser.error(f"No valid Phase 1 runs found in {phase1_dir}")

    family = None
    explicit_filters = any(
        value is not None for value in (args.iteration, args.segment_bits, args.num_segments, args.segment_offset_bits)
    ) or phase1_scales is not None
    if not args.no_family_inference and not explicit_filters:
        family_candidates = filter_runs(runs, variants=variants or None)
        family_candidates = [
            run
            for run in family_candidates
            if supports_window(
                run,
                window_scales=window_scales,
                left_policy=args.left_policy,
                right_policy=args.right_policy,
            )
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
        family=family,
    )
    selected = select_latest_per_variant(matching_runs, variant_order=variants)
    if not selected:
        parser.error("No latest-per-variant Phase 1 runs remain after filtering.")

    unsupported = [
        run["dataset"]["config"]["variant"]
        for run in selected
        if not supports_window(
            run,
            window_scales=window_scales,
            left_policy=args.left_policy,
            right_policy=args.right_policy,
        )
    ]
    if unsupported:
        parser.error("Selected runs do not support the requested Phase 3 window: " + ", ".join(unsupported))

    phase_print(
        "Computing square defects and sparse-graph spectrum",
        f"runs={len(selected)} | recurrence>={args.min_segment_recurrence}",
        quiet=args.quiet,
    )
    rows = build_defect_spectrum_rows(
        selected,
        window_scales=window_scales,
        left_policy=args.left_policy,
        right_policy=args.right_policy,
        min_segment_recurrence=args.min_segment_recurrence,
        top_hub_k=args.top_hub_k,
        show_progress=not args.quiet,
    )
    selection = build_selection(rows, family_inferred=family is not None, args=args)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_slug = build_run_slug(selection, timestamp)
    run_dir = output_root / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    dataset_path = run_dir / "dataset.json"
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "manifest.json"
    report_path = run_dir / "report.md"
    csv_path = run_dir / "compact_rows.csv"

    dataset_payload = {
        "stage": "phase3_min_defect_spectrum_pilot",
        "generated_at": generated_at,
        "selection": selection,
        "rows": rows,
        "notes": [
            "This is a Phase 3 pre-geometric pilot over existing Phase 1 artifacts.",
            "delta_square is measured by the commuting-square bridge defect.",
            "G_m is the sparse bridge graph; spectrum is the normalized Laplacian spectrum of that sampled graph.",
            "No geometry, curvature, transport law, or Gate 2 resolution is claimed by this run.",
        ],
    }
    summary_payload = {
        "generated_at": generated_at,
        "selection": selection,
        "compact_rows": compact_rows(rows),
    }
    manifest_payload = {
        "run_slug": run_slug,
        "generated_at": generated_at,
        "script": Path(__file__).name,
        "cwd": str(Path.cwd()),
        "inputs": {
            "phase1_dir": str(phase1_dir),
            "phase1_runs": [row["run_dir"] for row in rows],
        },
        "outputs": {
            "dataset": str(dataset_path),
            "summary": str(summary_path),
            "report": str(report_path),
            "compact_rows_csv": str(csv_path),
            "manifest": str(manifest_path),
        },
        "arguments": vars(args),
    }

    phase_print("Writing Phase 3 pilot artifacts", str(run_dir), quiet=args.quiet)
    write_json(dataset_path, dataset_payload)
    write_json(summary_path, summary_payload)
    write_json(manifest_path, manifest_payload)
    report_path.write_text(render_markdown_report(rows, selection) + "\n", encoding="utf-8")
    write_csv(csv_path, compact_rows(rows), CSV_FIELDS)

    if not args.quiet:
        if selection["family_inferred"]:
            print("Using the latest coherent Phase 1 family inferred from the newest compatible run.\n")
        print(render_console_summary(rows))
        print(f"\nSaved dataset to: {dataset_path}")
        print(f"Saved summary to: {summary_path}")
        print(f"Saved report to: {report_path}")
        print(f"Saved compact CSV to: {csv_path}")
        print(f"Saved manifest to: {manifest_path}")

    return 0


def supports_window(run: dict[str, Any], *, window_scales: tuple[int, int, int, int], left_policy: str, right_policy: str) -> bool:
    run_dir = Path(run["_run_dir"])
    available_scales = list(run["dataset"]["config"].get("scales", []))
    if any(scale not in available_scales for scale in window_scales):
        return False
    for policy in (left_policy, right_policy):
        if policy not in run["dataset"]["config"].get("policies", []):
            return False
        if not (run_dir / "fibers" / f"fibers_{policy}.json").is_file():
            return False
    return True


def build_selection(rows: list[dict[str, Any]], *, family_inferred: bool, args: argparse.Namespace) -> dict[str, Any]:
    first = rows[0]
    return {
        "variants": [row["variant"] for row in rows],
        "iterations": sorted({row["iteration"] for row in rows if row["iteration"] is not None}),
        "segment_bits": first["segment_bits"],
        "num_segments": first["num_segments"],
        "segment_offset_bits": first["segment_offset_bits"],
        "phase1_scales": list(first["scales"]),
        "window_scales": list(first["window_scales"]),
        "left_policy": first["left_policy"],
        "right_policy": first["right_policy"],
        "min_segment_recurrence": args.min_segment_recurrence,
        "top_hub_k": args.top_hub_k,
        "family_inferred": family_inferred,
    }


def build_run_slug(selection: dict[str, Any], timestamp: str) -> str:
    variants = "-".join(_slug_token(value) for value in selection["variants"])
    iteration_values = selection["iterations"]
    iter_part = "iter-mixed" if len(iteration_values) != 1 else f"iter-{iteration_values[0]}"
    offset_bits = selection.get("segment_offset_bits", 0)
    offset_part = f"__off-{compact_int(offset_bits)}" if offset_bits else ""
    window_part = "-".join(str(value) for value in selection["window_scales"])
    return (
        f"phase3-min-defect-spectrum__{iter_part}__seg-{selection['num_segments']}x{compact_int(selection['segment_bits'])}"
        f"{offset_part}__window-{window_part}__pol-{selection['left_policy']}-{selection['right_policy']}"
        f"__core-{selection['min_segment_recurrence']}__var-{variants}__{timestamp}"
    )


def phase_print(title: str, detail: str, *, quiet: bool) -> None:
    if quiet:
        return
    print(f"\n[Phase] {title}")
    print(f"        {detail}")


def write_json(path: Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _slug_token(value: str) -> str:
    return value.replace("_", "-").replace(" ", "-")


if __name__ == "__main__":
    raise SystemExit(main())
