#!/usr/bin/env python3
"""HSI v2 Phase 3 N3-04 micro child-routing spectral graph."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from v2.common.cli import resolve_dir
from v2.phase3.micro_child_routing_spectrum import build_micro_child_routing_spectrum


DEFAULT_TRANSITION_ROOT = "results/hsi_v2/phase2/phase_channel_transition_graph"
DEFAULT_CHILD_ROUTING_ROOT = "results/hsi_v2/phase2/phase_channel_width_n2_12_full_reval"
DEFAULT_OUTPUT_DIR = "results/hsi_v2/phase3/micro_child_routing_spectral_graph"

SUMMARY_FIELDS = [
    "variant",
    "source_label",
    "channel",
    "cell_count",
    "parent_count",
    "node_count",
    "edge_count",
    "retained_child_edges",
    "dropout_edges",
    "reroute_edges",
    "continuity_edges",
    "defect_edge_fraction",
    "conserved_parent_fraction",
    "mean_child_retention",
    "largest_component_fraction",
    "lambda2_norm_laplacian",
    "zero_eigenvalue_count",
    "spectral_radius_adjacency",
    "verdict",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build N3-04: a unit-weighted micro-spectral graph over child-routing "
            "events inside the N3-03 phase-channel scaffold."
        )
    )
    parser.add_argument(
        "--transition-run",
        default="",
        help="Path to an N2-14 transition graph run. Defaults to latest under --transition-root.",
    )
    parser.add_argument("--transition-root", default=DEFAULT_TRANSITION_ROOT)
    parser.add_argument(
        "--child-routing-run",
        default="",
        help=(
            "Path to a child-routing lag-response run. Defaults to latest run under "
            "--child-routing-root."
        ),
    )
    parser.add_argument("--child-routing-root", default=DEFAULT_CHILD_ROUTING_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-patterns", type=int, default=128)
    parser.add_argument("--retention-threshold", type=float, default=0.75)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.top_patterns <= 0:
        raise SystemExit("--top-patterns must be positive")
    if not (0.0 <= args.retention_threshold <= 1.0):
        raise SystemExit("--retention-threshold must be between 0 and 1")

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
        print("[Phase] Preparing N3-04 micro child-routing spectral graph")
        print(f"        transition={transition_run.name}")
        print(f"        child_routing={child_routing_run.name}")
        print(f"        top={args.top_patterns} | retention_threshold={args.retention_threshold:.2f}")

    readout = build_micro_child_routing_spectrum(
        transition_run=transition_run,
        child_routing_run=child_routing_run,
        top_patterns=args.top_patterns,
        retention_threshold=args.retention_threshold,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_slug = (
        "phase3-micro-child-routing-spectral-graph"
        f"__src-{transition_run.name}"
        f"__top-{args.top_patterns}__{timestamp}"
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
        "stage": "phase3_micro_child_routing_spectral_graph",
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
        if readout["missing_cells"]:
            print(f"\n[warn] Missing routed cells: {len(readout['missing_cells'])}")
        print(f"\nSaved summary to: {summary_path}")
        print(f"Saved report to: {report_path}")
        print(f"Saved graph summary CSV to: {graph_summary_path}")
        print(f"Saved manifest to: {manifest_path}")
    return 0


def resolve_latest_run(
    *,
    explicit: str,
    root: str,
    required: tuple[str, ...],
    recursive_required: str | None = None,
) -> Path:
    if explicit.strip():
        run = resolve_dir(explicit, anchor_file=__file__)
        if run.is_file():
            run = run.parent
        ensure_required(run, required, recursive_required)
        return run

    root_path = resolve_dir(root, anchor_file=__file__)
    if not root_path.is_dir():
        raise SystemExit(f"Run root not found: {root_path}")
    candidates = []
    for child in root_path.iterdir():
        if not child.is_dir():
            continue
        try:
            ensure_required(child, required, recursive_required)
        except SystemExit:
            continue
        candidates.append(child)
    if not candidates:
        raise SystemExit(f"No matching runs found under {root_path}")
    return max(candidates, key=lambda item: item.stat().st_mtime)


def ensure_required(run: Path, required: tuple[str, ...], recursive_required: str | None) -> None:
    for filename in required:
        if not (run / filename).is_file():
            raise SystemExit(f"Missing {filename}: {run}")
    if recursive_required and not any(run.rglob(recursive_required)):
        raise SystemExit(f"Missing {recursive_required} under {run}")


def render_console_summary(rows: list[dict[str, Any]]) -> str:
    lines = [
        "Phase 3 N3-04 micro child-routing spectral graph",
        "-" * 142,
        (
            f"{'variant':<13} {'source':<28} {'ch':<8} {'cells':>5} {'parents':>7} "
            f"{'nodes':>7} {'edges':>7} {'def':>7} {'cons':>7} {'ret':>7} "
            f"{'lcc':>7} {'lam2':>8} {'zero':>5}  verdict"
        ),
    ]
    for row in rows:
        lines.append(
            f"{clip(row['variant'], 13):<13} {clip(row['source_label'], 28):<28} "
            f"{row['channel']:<8} {int(row['cell_count']):>5} {int(row['parent_count']):>7} "
            f"{int(row['node_count']):>7} {int(row['edge_count']):>7} "
            f"{fmt(row['defect_edge_fraction']):>7} {fmt(row['conserved_parent_fraction']):>7} "
            f"{fmt(row['mean_child_retention']):>7} {fmt(row['largest_component_fraction']):>7} "
            f"{fmt(row['lambda2_norm_laplacian']):>8} {int(row['zero_eigenvalue_count']):>5}  "
            f"{row['verdict']}"
        )
    return "\n".join(lines)


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase 3 N3-04 Micro Child-Routing Spectral Graph",
        "",
        "## Contract",
        "",
        "- This readout builds unit-weighted graphs from existing Phase 2 child-routing artifacts.",
        "- Nodes are parent patterns, retained child destinations, and explicit dropout/reroute terminals inside phase-channel cells.",
        "- Edges are retained child destinations, dropout/reroute defects, and same-parent continuity across adjacent bands.",
        "- No raw bits are regenerated; no metric weights are fitted.",
        "- This is pre-geometric and internal, not a curvature or transport-law claim.",
        "",
        "## Selection",
        "",
        f"- Transition run: `{payload['selection']['transition_run']}`",
        f"- Child-routing run: `{payload['selection']['child_routing_run']}`",
        f"- Top patterns: {payload['selection']['top_patterns']}",
        f"- Retention threshold: {payload['selection']['retention_threshold']}",
        "",
        "## Compact Readout",
        "",
        "| Variant | Source | Channel | Cells | Parents | Nodes | Edges | Defect edge frac | Conserved parent frac | Mean retention | LCC | Lambda2 | Zero | Verdict |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["graph_summary"]:
        lines.append(
            f"| {row['variant']} | {row['source_label']} | {row['channel']} | "
            f"{row['cell_count']} | {row['parent_count']} | {row['node_count']} | {row['edge_count']} | "
            f"{fmt(row['defect_edge_fraction'])} | {fmt(row['conserved_parent_fraction'])} | "
            f"{fmt(row['mean_child_retention'])} | {fmt(row['largest_component_fraction'])} | "
            f"{fmt(row['lambda2_norm_laplacian'])} | {row['zero_eigenvalue_count']} | {row['verdict']} |"
        )
    if payload["missing_cells"]:
        lines.extend(["", "## Missing Cells", ""])
        for cell in payload["missing_cells"]:
            lines.append(
                f"- {cell['channel']} {cell['band']} lag={cell['best_margin_lag_bits']}"
            )
    return "\n".join(lines) + "\n"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fields})


def write_csv_dynamic(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    write_csv(path, rows, fields)


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def fmt(value: Any) -> str:
    if value is None or value == "":
        return "-"
    return f"{float(value):.4f}"


def clip(value: Any, width: int) -> str:
    text = str(value)
    if len(text) <= width:
        return text
    return text[: max(0, width - 3)] + "..."


if __name__ == "__main__":
    raise SystemExit(main())
