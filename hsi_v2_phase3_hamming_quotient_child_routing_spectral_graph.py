#!/usr/bin/env python3
"""HSI v2 Phase 3 N3-06 Hamming quotient child-routing spectral graph."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from hsi_v2_phase3_quotient_child_routing_spectral_graph import resolve_latest_run
from v2.common.cli import resolve_dir
from v2.phase3.hamming_quotient_spectrum import build_hamming_quotient_readout


DEFAULT_WEIGHTED_ROOT = "results/hsi_v2/phase3/weighted_quotient_child_routing_spectral_graph"
DEFAULT_OUTPUT_DIR = "results/hsi_v2/phase3/hamming_quotient_child_routing_spectral_graph"
DEFAULT_RADII = "1,2,4,8"

SUMMARY_FIELDS = [
    "variant",
    "source_label",
    "channel",
    "radius",
    "family_count",
    "hamming_cluster_count",
    "node_count",
    "edge_count",
    "inherited_edge_count",
    "collapsed_self_edges",
    "collapsed_self_weight",
    "projected_flow_weight",
    "family_continuity_weight",
    "total_edge_weight",
    "largest_component_fraction",
    "lambda2_norm_laplacian",
    "zero_eigenvalue_count",
    "spectral_radius_adjacency",
    "degree_entropy_norm",
    "verdict",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build N3-06: a predeclared Hamming quotient over the existing "
            "N3-05b weighted child-routing quotient."
        )
    )
    parser.add_argument(
        "--weighted-quotient-run",
        default="",
        help="Path to an N3-05b weighted quotient run. Defaults to latest under --weighted-root.",
    )
    parser.add_argument("--weighted-root", default=DEFAULT_WEIGHTED_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--radii",
        default=DEFAULT_RADII,
        help=f"Comma-separated Hamming radii to test. Default: {DEFAULT_RADII}",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    radii = parse_radii(args.radii)
    weighted_run = resolve_latest_run(
        explicit=args.weighted_quotient_run,
        root=args.weighted_root,
        required=("graph_nodes.csv", "graph_edges.csv", "graph_summary.csv"),
    )
    output_root = resolve_dir(args.output_dir, anchor_file=__file__)
    output_root.mkdir(parents=True, exist_ok=True)

    if not args.quiet:
        print("[Phase] Preparing N3-06 Hamming quotient child-routing spectral graph")
        print(f"        weighted_quotient={weighted_run.name}")
        print(f"        radii={','.join(str(value) for value in radii)}")

    readout = build_hamming_quotient_readout(weighted_run, radii=radii)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    radius_slug = "-".join(str(value) for value in radii)
    run_slug = f"phase3-hamming-quotient-spectral-graph__src-{weighted_run.name}__r-{radius_slug}__{timestamp}"
    run_dir = output_root / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_path = run_dir / "summary.json"
    report_path = run_dir / "report.md"
    graph_summary_path = run_dir / "graph_summary.csv"
    graph_nodes_path = run_dir / "graph_nodes.csv"
    graph_edges_path = run_dir / "graph_edges.csv"
    manifest_path = run_dir / "manifest.json"

    payload = {
        "stage": "phase3_hamming_quotient_child_routing_spectral_graph",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **readout,
    }
    manifest = {
        "run_slug": run_slug,
        "script": Path(__file__).name,
        "cwd": str(Path.cwd()),
        "inputs": {
            "weighted_quotient_run": str(weighted_run),
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
    write_csv(graph_summary_path, payload["graph_summary"], SUMMARY_FIELDS)
    write_csv_dynamic(graph_nodes_path, payload["graph_nodes"])
    write_csv_dynamic(graph_edges_path, payload["graph_edges"])
    report_path.write_text(render_report(payload), encoding="utf-8")

    if not args.quiet:
        print(render_console_summary(payload["graph_summary"]))
        print(f"\nSaved summary to: {summary_path}")
        print(f"Saved report to: {report_path}")
        print(f"Saved graph summary CSV to: {graph_summary_path}")
        print(f"Saved manifest to: {manifest_path}")
    return 0


def render_console_summary(rows: list[dict[str, Any]]) -> str:
    lines = [
        "Phase 3 N3-06 Hamming quotient child-routing spectral graph",
        "-" * 160,
        (
            f"{'variant':<13} {'source':<28} {'ch':<8} {'r':>3} {'fam':>5} {'hcl':>5} "
            f"{'nodes':>6} {'edges':>6} {'selfW':>8} {'lcc':>7} {'lam2':>8} "
            f"{'zero':>5} {'rho':>9}  verdict"
        ),
    ]
    for row in rows:
        lines.append(
            f"{clip(row['variant'], 13):<13} {clip(row['source_label'], 28):<28} "
            f"{row['channel']:<8} {int(row['radius']):>3} {int(row['family_count']):>5} "
            f"{int(row['hamming_cluster_count']):>5} {int(row['node_count']):>6} "
            f"{int(row['edge_count']):>6} {fmt(row['collapsed_self_weight']):>8} "
            f"{fmt(row['largest_component_fraction']):>7} {fmt(row['lambda2_norm_laplacian']):>8} "
            f"{int(row['zero_eigenvalue_count']):>5} {fmt(row['spectral_radius_adjacency']):>9}  "
            f"{row['verdict']}"
        )
    return "\n".join(lines)


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase 3 N3-06 Hamming Quotient Child-Routing Spectral Graph",
        "",
        "## Contract",
        "",
        "- This readout starts from an existing N3-05b weighted quotient artifact.",
        "- Families whose Hamming distance is within a predeclared radius are identified.",
        "- Existing retained-flow edges are then aggregated; no transport edge is invented.",
        "- The radius sweep is sensitivity, not post-hoc tuning.",
        "- This remains internal and pre-geometric.",
        "",
        "## Selection",
        "",
        f"- Weighted quotient run: `{payload['selection']['weighted_quotient_run']}`",
        f"- Hamming radii: {', '.join(str(value) for value in payload['selection']['radii'])}",
        "",
        "## Compact Readout",
        "",
        "| Variant | Source | Channel | r | Families | H-clusters | Nodes | Edges | Self weight | LCC | Lambda2 | Zero | Rho | Verdict |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["graph_summary"]:
        lines.append(
            f"| {row['variant']} | {row['source_label']} | {row['channel']} | "
            f"{row['radius']} | {row['family_count']} | {row['hamming_cluster_count']} | "
            f"{row['node_count']} | {row['edge_count']} | {fmt(row['collapsed_self_weight'])} | "
            f"{fmt(row['largest_component_fraction'])} | {fmt(row['lambda2_norm_laplacian'])} | "
            f"{row['zero_eigenvalue_count']} | {fmt(row['spectral_radius_adjacency'])} | {row['verdict']} |"
        )
    return "\n".join(lines) + "\n"


def parse_radii(raw: str) -> list[int]:
    radii = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    if not radii:
        raise SystemExit("--radii must contain at least one integer")
    if any(value < 0 for value in radii):
        raise SystemExit("--radii cannot contain negative values")
    return radii


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
