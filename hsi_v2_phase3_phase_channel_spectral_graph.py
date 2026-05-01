#!/usr/bin/env python3
"""HSI v2 Phase 3 N3-03 phase-channel spectral graph."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from v2.common.cli import resolve_dir
from v2.phase3.defect_spectrum import compute_graph_spectrum


DEFAULT_TRANSITION_ROOT = "results/hsi_v2/phase2/phase_channel_transition_graph"
DEFAULT_OUTPUT_DIR = "results/hsi_v2/phase3/phase_channel_spectral_graph"

SERIES_FIELDS = [
    "top_patterns",
    "channel",
    "node_count",
    "edge_count",
    "active_node_count",
    "boundary_node_count",
    "largest_component_fraction",
    "lambda2_norm_laplacian",
    "zero_eigenvalue_count",
    "spectral_radius_adjacency",
    "active_path_bands",
    "verdict",
]

COUNT_FIELDS = ["top_patterns", "channel", "transition_type", "count"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Phase 3 N3-03 non-local spectral readout over the audited "
            "N2-14 phase-channel transition graph."
        )
    )
    parser.add_argument(
        "--transition-run",
        default="",
        help="Path to an N2-14 transition graph run. Defaults to latest under --transition-root.",
    )
    parser.add_argument("--transition-root", default=DEFAULT_TRANSITION_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reference-top", type=int, default=128)
    parser.add_argument("--min-path-bands", type=int, default=3)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.reference_top <= 0:
        raise SystemExit("--reference-top must be positive")
    if args.min_path_bands <= 0:
        raise SystemExit("--min-path-bands must be positive")

    transition_run = resolve_transition_run(args)
    output_root = resolve_dir(args.output_dir, anchor_file=__file__)
    output_root.mkdir(parents=True, exist_ok=True)

    n2_nodes = read_csv(transition_run / "nodes.csv")
    n2_edges = read_csv(transition_run / "edges.csv")
    n2_paths = read_csv(transition_run / "paths.csv")

    graph_nodes, graph_edges = build_graph_records(n2_nodes, n2_edges)
    series_rows = build_series_rows(
        graph_nodes,
        graph_edges,
        n2_paths,
        reference_top=args.reference_top,
        min_path_bands=args.min_path_bands,
    )
    transition_counts = build_transition_counts(graph_edges)
    combined = build_combined_reference_row(
        graph_nodes,
        graph_edges,
        reference_top=args.reference_top,
        min_path_bands=args.min_path_bands,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_slug = f"phase3-phase-channel-spectral-graph__src-{transition_run.name}__{timestamp}"
    run_dir = output_root / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "stage": "phase3_phase_channel_spectral_graph",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_transition_run": str(transition_run),
        "parameters": vars(args),
        "notes": [
            "N3-03 is a non-local spectral readout over the already audited N2-14 transition graph.",
            "Nodes are phase-channel states; edges are adjacent-band transitions from N2-14.",
            "All non-inactive transitions are treated as unit-weight adjacency in this first spectral pass.",
            "This is still pre-geometric: it does not claim curvature, metric geometry, or a final transport law.",
        ],
        "combined_reference": combined,
        "series_summary": series_rows,
        "transition_counts": transition_counts,
    }
    manifest = {
        "run_slug": run_slug,
        "script": Path(__file__).name,
        "cwd": str(Path.cwd()),
        "inputs": {
            "transition_run": str(transition_run),
            "nodes_csv": str(transition_run / "nodes.csv"),
            "edges_csv": str(transition_run / "edges.csv"),
            "paths_csv": str(transition_run / "paths.csv"),
        },
        "outputs": {
            "summary": str(run_dir / "summary.json"),
            "report": str(run_dir / "report.md"),
            "series_summary_csv": str(run_dir / "series_summary.csv"),
            "transition_counts_csv": str(run_dir / "transition_counts.csv"),
            "graph_nodes_csv": str(run_dir / "graph_nodes.csv"),
            "graph_edges_csv": str(run_dir / "graph_edges.csv"),
            "manifest": str(run_dir / "manifest.json"),
        },
    }

    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "manifest.json", manifest)
    write_csv(run_dir / "series_summary.csv", series_rows, SERIES_FIELDS)
    write_csv(run_dir / "transition_counts.csv", transition_counts, COUNT_FIELDS)
    write_csv(run_dir / "graph_nodes.csv", graph_nodes, sorted(graph_nodes[0]) if graph_nodes else [])
    write_csv(run_dir / "graph_edges.csv", graph_edges, sorted(graph_edges[0]) if graph_edges else [])
    (run_dir / "report.md").write_text(render_report(summary), encoding="utf-8")

    if not args.quiet:
        print(render_console_summary(series_rows, combined, reference_top=args.reference_top))
        print(f"\nSaved summary to: {run_dir / 'summary.json'}")
        print(f"Saved report to: {run_dir / 'report.md'}")
        print(f"Saved series summary CSV to: {run_dir / 'series_summary.csv'}")
        print(f"Saved manifest to: {run_dir / 'manifest.json'}")
    return 0


def resolve_transition_run(args: argparse.Namespace) -> Path:
    if args.transition_run.strip():
        run = resolve_dir(args.transition_run, anchor_file=__file__)
    else:
        root = resolve_dir(args.transition_root, anchor_file=__file__)
        if not root.is_dir():
            raise SystemExit(f"Transition root not found: {root}")
        candidates = [
            child
            for child in root.iterdir()
            if child.is_dir()
            and (child / "nodes.csv").is_file()
            and (child / "edges.csv").is_file()
            and (child / "paths.csv").is_file()
        ]
        if not candidates:
            raise SystemExit(f"No N2-14 transition graph runs found under {root}")
        run = max(candidates, key=lambda path: path.stat().st_mtime)
    for filename in ("nodes.csv", "edges.csv", "paths.csv"):
        if not (run / filename).is_file():
            raise SystemExit(f"Missing {filename}: {run}")
    return run


def build_graph_records(n2_nodes: list[dict[str, str]], n2_edges: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes = []
    node_ids = set()
    for row in n2_nodes:
        node_id = node_id_for(row["top_patterns"], row["channel"], row["band"])
        node_ids.add(node_id)
        nodes.append(
            {
                "id": node_id,
                "top_patterns": to_int(row["top_patterns"]),
                "channel": row["channel"],
                "band": row["band"],
                "band_start_bits": to_int(row["band_start_bits"]),
                "band_stop_bits": to_int(row["band_stop_bits"]),
                "node_class": row["node_class"],
                "active_flag": str(row.get("strong_count", "0")) != "0",
                "boundary_flag": row["node_class"] == "null-shadow-boundary",
                "strong_count": to_int(row.get("strong_count", "0")),
                "best_margin_lag_bits": optional_int(row.get("best_margin_lag_bits", "")),
                "best_margin": optional_float(row.get("best_margin", "")),
            }
        )

    edges = []
    for row in n2_edges:
        if row["transition_type"] == "inactive-continuity":
            continue
        left = node_id_for(row["top_patterns"], row["channel"], row["source_band"])
        right = node_id_for(row["top_patterns"], row["channel"], row["target_band"])
        if left not in node_ids or right not in node_ids:
            continue
        edges.append(
            {
                "id": f"{left}||{right}",
                "nodes": [left, right],
                "top_patterns": to_int(row["top_patterns"]),
                "channel": row["channel"],
                "source_band": row["source_band"],
                "target_band": row["target_band"],
                "transition_type": row["transition_type"],
                "edge_weight": 1.0,
                "lag_delta_bits": optional_int(row.get("lag_delta_bits", "")),
                "margin_delta": optional_float(row.get("margin_delta", "")),
            }
        )
    return nodes, edges


def build_series_rows(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    n2_paths: list[dict[str, str]],
    *,
    reference_top: int,
    min_path_bands: int,
) -> list[dict[str, Any]]:
    path_index = {
        (to_int(row["top_patterns"]), row["channel"]): row
        for row in n2_paths
    }
    rows = []
    grouped_nodes = group_by_series(nodes)
    grouped_edges = group_edges_by_series(edges)
    for key in sorted(grouped_nodes):
        top, channel = key
        series_nodes = grouped_nodes[key]
        series_edges = grouped_edges.get(key, [])
        graph = graph_result(series_nodes, series_edges)
        spectrum = compute_graph_spectrum(graph)
        path = path_index.get(key, {})
        active_path_bands = to_int(path.get("max_active_path_bands", "0"))
        boundary_count = sum(1 for node in series_nodes if node["boundary_flag"])
        row = {
            "top_patterns": top,
            "channel": channel,
            "node_count": len(series_nodes),
            "edge_count": len(series_edges),
            "active_node_count": sum(1 for node in series_nodes if node["active_flag"]),
            "boundary_node_count": boundary_count,
            "largest_component_fraction": largest_component_fraction(series_nodes, series_edges),
            "lambda2_norm_laplacian": spectrum["lambda2_norm_laplacian"],
            "zero_eigenvalue_count": spectrum["zero_eigenvalue_count"],
            "spectral_radius_adjacency": spectrum["spectral_radius_adjacency"],
            "active_path_bands": active_path_bands,
            "verdict": spectral_verdict(
                top=top,
                reference_top=reference_top,
                active_path_bands=active_path_bands,
                boundary_count=boundary_count,
                min_path_bands=min_path_bands,
                spectrum=spectrum,
            ),
        }
        rows.append(row)
    return rows


def build_combined_reference_row(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    *,
    reference_top: int,
    min_path_bands: int,
) -> dict[str, Any]:
    ref_nodes = [node for node in nodes if int(node["top_patterns"]) == reference_top]
    ref_ids = {node["id"] for node in ref_nodes}
    ref_edges = [edge for edge in edges if edge["nodes"][0] in ref_ids and edge["nodes"][1] in ref_ids]
    spectrum = compute_graph_spectrum(graph_result(ref_nodes, ref_edges))
    return {
        "top_patterns": reference_top,
        "node_count": len(ref_nodes),
        "edge_count": len(ref_edges),
        "active_node_count": sum(1 for node in ref_nodes if node["active_flag"]),
        "boundary_node_count": sum(1 for node in ref_nodes if node["boundary_flag"]),
        "largest_component_fraction": largest_component_fraction(ref_nodes, ref_edges),
        "lambda2_norm_laplacian": spectrum["lambda2_norm_laplacian"],
        "zero_eigenvalue_count": spectrum["zero_eigenvalue_count"],
        "spectral_radius_adjacency": spectrum["spectral_radius_adjacency"],
        "verdict": "reference-has-nonlocal-channel-components"
        if len(ref_nodes) >= min_path_bands and len(ref_edges) > 0
        else "reference-no-nonlocal-channel-components",
    }


def build_transition_counts(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[int, str, str], int] = defaultdict(int)
    for edge in edges:
        counts[(int(edge["top_patterns"]), edge["channel"], edge["transition_type"])] += 1
    return [
        {"top_patterns": top, "channel": channel, "transition_type": transition, "count": count}
        for (top, channel, transition), count in sorted(counts.items())
    ]


def graph_result(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "node_records": [{"id": node["id"]} for node in nodes],
        "edge_records": [{"nodes": edge["nodes"], "edge_weight": edge["edge_weight"]} for edge in edges],
    }


def spectral_verdict(
    *,
    top: int,
    reference_top: int,
    active_path_bands: int,
    boundary_count: int,
    min_path_bands: int,
    spectrum: dict[str, Any],
) -> str:
    if active_path_bands < min_path_bands:
        return "no-spectral-channel-path"
    if top == reference_top and boundary_count:
        return "spectral-channel-with-boundary"
    if spectrum["zero_eigenvalue_count"] == 1:
        return "connected-spectral-channel"
    return "fragmented-spectral-channel"


def group_by_series(nodes: list[dict[str, Any]]) -> dict[tuple[int, str], list[dict[str, Any]]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        grouped[(int(node["top_patterns"]), node["channel"])].append(node)
    for group in grouped.values():
        group.sort(key=lambda item: int(item["band_start_bits"]))
    return grouped


def group_edges_by_series(edges: list[dict[str, Any]]) -> dict[tuple[int, str], list[dict[str, Any]]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        grouped[(int(edge["top_patterns"]), edge["channel"])].append(edge)
    return grouped


def largest_component_fraction(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> float:
    if not nodes:
        return 0.0
    adjacency = {node["id"]: set() for node in nodes}
    for edge in edges:
        left, right = edge["nodes"]
        adjacency[left].add(right)
        adjacency[right].add(left)
    seen = set()
    largest = 0
    for node_id in adjacency:
        if node_id in seen:
            continue
        frontier = [node_id]
        component = set()
        while frontier:
            current = frontier.pop()
            if current in component:
                continue
            component.add(current)
            frontier.extend(adjacency[current] - component)
        seen.update(component)
        largest = max(largest, len(component))
    return largest / len(nodes)


def render_console_summary(series_rows: list[dict[str, Any]], combined: dict[str, Any], *, reference_top: int) -> str:
    lines = [
        "Phase 3 N3-03 phase-channel spectral graph",
        "-" * 112,
        f"{'top':>5} {'channel':<8} {'nodes':>6} {'edges':>6} {'active':>7} {'bound':>6} {'lcc':>8} {'lambda2':>9} {'zero':>5}  verdict",
    ]
    for row in series_rows:
        marker = "*" if int(row["top_patterns"]) == reference_top else " "
        lines.append(
            f"{marker}{int(row['top_patterns']):>4} {row['channel']:<8} "
            f"{int(row['node_count']):>6} {int(row['edge_count']):>6} "
            f"{int(row['active_node_count']):>7} {int(row['boundary_node_count']):>6} "
            f"{fmt(row['largest_component_fraction']):>8} {fmt(row['lambda2_norm_laplacian']):>9} "
            f"{int(row['zero_eigenvalue_count']):>5}  {row['verdict']}"
        )
    lines.append("")
    lines.append(
        f"reference top={reference_top}: nodes={combined['node_count']} edges={combined['edge_count']} "
        f"zero={combined['zero_eigenvalue_count']} rho={fmt(combined['spectral_radius_adjacency'])} "
        f"verdict={combined['verdict']}"
    )
    return "\n".join(lines)


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase 3 N3-03 Phase-Channel Spectral Graph",
        "",
        "## Contract",
        "",
        "- This is a non-local spectral readout over the audited N2-14 phase-channel transition graph.",
        "- Nodes are band x channel states; edges are adjacent-band transitions.",
        "- Unit edge weights are used intentionally to avoid fitting metric weights at this stage.",
        "- No curvature, geometry, or final transport law is claimed.",
        "",
        "## Combined Reference Graph",
        "",
        f"- Top-k: {summary['combined_reference']['top_patterns']}",
        f"- Nodes: {summary['combined_reference']['node_count']}",
        f"- Edges: {summary['combined_reference']['edge_count']}",
        f"- Zero eigenvalues: {summary['combined_reference']['zero_eigenvalue_count']}",
        f"- Spectral radius: {fmt(summary['combined_reference']['spectral_radius_adjacency'])}",
        f"- Verdict: `{summary['combined_reference']['verdict']}`",
        "",
        "## Series Summary",
        "",
        "| Top-k | Channel | Nodes | Edges | Active | Boundary | LCC | Lambda2 | Zero eigs | Verdict |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary["series_summary"]:
        lines.append(
            f"| {row['top_patterns']} | {row['channel']} | {row['node_count']} | {row['edge_count']} | "
            f"{row['active_node_count']} | {row['boundary_node_count']} | {fmt(row['largest_component_fraction'])} | "
            f"{fmt(row['lambda2_norm_laplacian'])} | {row['zero_eigenvalue_count']} | {row['verdict']} |"
        )
    lines.extend(
        [
            "",
            "## Reading Rule",
            "",
            "- A positive readout requires persistent reference-top channel components with explicit boundary handling.",
            "- If the reference graph decomposes into isolated or non-persistent nodes, the non-local scaffold reading fails.",
            "- This object is only a bridge toward future topology/geometry; it is not itself a geometric claim.",
        ]
    )
    return "\n".join(lines) + "\n"


def node_id_for(top: Any, channel: str, band: str) -> str:
    return f"top{to_int(top)}::{channel}::{band}"


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def to_int(value: Any) -> int:
    if value is None or value == "":
        return 0
    return int(float(str(value)))


def optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(float(str(value)))


def optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(str(value))


def fmt(value: Any) -> str:
    if value is None or value == "":
        return "-"
    return f"{float(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
