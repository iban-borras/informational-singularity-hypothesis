#!/usr/bin/env python3
"""HSI v2 Phase 3 N3-05b weighted quotient child-routing spectral graph."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hsi_v2_phase3_quotient_child_routing_spectral_graph import (
    DEFAULT_CHILD_ROUTING_ROOT,
    DEFAULT_TRANSITION_ROOT,
    SUMMARY_FIELDS,
    render_console_summary,
    render_report,
    resolve_latest_run,
    write_csv,
    write_csv_dynamic,
    write_json,
)
from v2.common.cli import resolve_dir
from v2.phase3.quotient_child_routing_spectrum import (
    build_quotient_child_routing_spectrum,
)


DEFAULT_OUTPUT_DIR = "results/hsi_v2/phase3/weighted_quotient_child_routing_spectral_graph"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build N3-05b: the pooled-mass weighted quotient spectral graph. "
            "This is a separate object from the unit N3-05 graph."
        )
    )
    parser.add_argument("--transition-run", default="")
    parser.add_argument("--transition-root", default=DEFAULT_TRANSITION_ROOT)
    parser.add_argument("--child-routing-run", default="")
    parser.add_argument("--child-routing-root", default=DEFAULT_CHILD_ROUTING_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-patterns", type=int, default=128)
    parser.add_argument("--parent-scale", type=int, default=40)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.top_patterns <= 0:
        raise SystemExit("--top-patterns must be positive")
    if args.parent_scale <= 0:
        raise SystemExit("--parent-scale must be positive")

    transition_run = resolve_latest_run(
        explicit=args.transition_run,
        root=args.transition_root,
        required=("nodes.csv", "edges.csv"),
    )
    child_routing_run = resolve_latest_run(
        explicit=args.child_routing_run,
        root=args.child_routing_root,
        required=("summary.json",),
        recursive_required="routing_rows.csv",
    )
    output_root = resolve_dir(args.output_dir, anchor_file=__file__)
    output_root.mkdir(parents=True, exist_ok=True)

    if not args.quiet:
        print("[Phase] Preparing N3-05b weighted quotient child-routing spectral graph")
        print(f"        transition={transition_run.name}")
        print(f"        child_routing={child_routing_run.name}")
        print(f"        top={args.top_patterns} | parent_scale={args.parent_scale}")
        print("        edge_weight_mode=pooled-retained-mass")

    readout = build_quotient_child_routing_spectrum(
        transition_run=transition_run,
        child_routing_run=child_routing_run,
        top_patterns=args.top_patterns,
        parent_scale=args.parent_scale,
        edge_weight_mode="pooled-retained-mass",
    )
    if readout["missing_cells"]:
        lines = [
            "Missing routed cells; refusing to emit an incomplete N3-05b artifact.",
            "Provide child-routing rows for every transition cell before rerunning.",
        ]
        for cell in readout["missing_cells"]:
            lines.append(
                f"- {cell['channel']} {cell['band']} lag={cell['best_margin_lag_bits']}"
            )
        raise SystemExit("\n".join(lines))

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_slug = (
        "phase3-weighted-quotient-child-routing-spectral-graph"
        f"__src-{transition_run.name}"
        f"__top-{args.top_patterns}__m-{args.parent_scale}__{timestamp}"
    )
    run_dir = output_root / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_path = run_dir / "summary.json"
    report_path = run_dir / "report.md"
    graph_summary_path = run_dir / "graph_summary.csv"
    graph_nodes_path = run_dir / "graph_nodes.csv"
    graph_edges_path = run_dir / "graph_edges.csv"
    manifest_path = run_dir / "manifest.json"

    payload = {
        "stage": "phase3_weighted_quotient_child_routing_spectral_graph",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selection": readout["selection"],
        "notes": readout["notes"],
        "transition_cells": readout["transition_cells"],
        "missing_cells": readout["missing_cells"],
        "graph_summary": readout["graph_summary"],
    }
    manifest = {
        "run_slug": run_slug,
        "script": Path(__file__).name,
        "cwd": str(Path.cwd()),
        "inputs": {
            "transition_run": str(transition_run),
            "child_routing_run": str(child_routing_run),
        },
        "outputs": {
            "summary": str(summary_path),
            "report": str(report_path),
            "graph_summary_csv": str(graph_summary_path),
            "graph_nodes_csv": str(graph_nodes_path),
            "graph_edges_csv": str(graph_edges_path),
            "manifest": str(manifest_path),
        },
        "arguments": vars(args),
    }

    write_json(summary_path, payload)
    write_json(manifest_path, manifest)
    write_csv(graph_summary_path, readout["graph_summary"], SUMMARY_FIELDS)
    write_csv_dynamic(graph_nodes_path, readout["graph_nodes"])
    write_csv_dynamic(graph_edges_path, readout["graph_edges"])
    report_path.write_text(render_report(payload), encoding="utf-8")

    if not args.quiet:
        print(render_console_summary(readout["graph_summary"]))
        print(f"\nSaved summary to: {summary_path}")
        print(f"Saved report to: {report_path}")
        print(f"Saved graph summary CSV to: {graph_summary_path}")
        print(f"Saved manifest to: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
