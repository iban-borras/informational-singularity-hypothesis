#!/usr/bin/env python3
"""HSI v2 Phase 3 N3-05c component-wise quotient normalization."""

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
from v2.phase3.component_quotient_spectrum import build_component_quotient_readout


DEFAULT_WEIGHTED_ROOT = "results/hsi_v2/phase3/weighted_quotient_child_routing_spectral_graph"
DEFAULT_OUTPUT_DIR = "results/hsi_v2/phase3/component_quotient_child_routing_spectrum"

SOURCE_FIELDS = [
    "variant",
    "source_label",
    "channel",
    "node_count",
    "edge_count",
    "component_count",
    "active_component_count",
    "total_edge_weight",
    "projected_flow_weight",
    "family_continuity_weight",
    "flow_continuity_ratio",
    "component_weight_entropy_norm",
    "effective_weight_components",
    "top1_component_weight_share",
    "top3_component_weight_share",
    "top5_component_weight_share",
    "weighted_component_lambda2",
    "weighted_mass_normalized_rho",
    "weighted_degree_entropy_norm",
    "verdict",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build N3-05c: component-wise normalized readout over the N3-05b "
            "weighted prefix/suffix quotient."
        )
    )
    parser.add_argument(
        "--weighted-quotient-run",
        default="",
        help="Path to an N3-05b weighted quotient run. Defaults to latest under --weighted-root.",
    )
    parser.add_argument("--weighted-root", default=DEFAULT_WEIGHTED_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    weighted_run = resolve_latest_run(
        explicit=args.weighted_quotient_run,
        root=args.weighted_root,
        required=("graph_nodes.csv", "graph_edges.csv", "graph_summary.csv"),
    )
    output_root = resolve_dir(args.output_dir, anchor_file=__file__)
    output_root.mkdir(parents=True, exist_ok=True)

    if not args.quiet:
        print("[Phase] Preparing N3-05c component-wise quotient normalization")
        print(f"        weighted_quotient={weighted_run.name}")

    readout = build_component_quotient_readout(weighted_run)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_slug = f"phase3-component-quotient-spectrum__src-{weighted_run.name}__{timestamp}"
    run_dir = output_root / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_path = run_dir / "summary.json"
    report_path = run_dir / "report.md"
    source_csv_path = run_dir / "source_summary.csv"
    component_csv_path = run_dir / "component_summary.csv"
    manifest_path = run_dir / "manifest.json"

    payload = {
        "stage": "phase3_component_quotient_child_routing_spectrum",
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
            "source_summary_csv": str(source_csv_path),
            "component_summary_csv": str(component_csv_path),
            "manifest": str(manifest_path),
        },
        "arguments": vars(args),
    }

    write_json(summary_path, payload)
    write_json(manifest_path, manifest)
    write_csv(source_csv_path, payload["source_summary"], SOURCE_FIELDS)
    write_csv_dynamic(component_csv_path, payload["component_summary"])
    report_path.write_text(render_report(payload), encoding="utf-8")

    if not args.quiet:
        print(render_console_summary(payload["source_summary"]))
        print(f"\nSaved summary to: {summary_path}")
        print(f"Saved report to: {report_path}")
        print(f"Saved source CSV to: {source_csv_path}")
        print(f"Saved manifest to: {manifest_path}")
    return 0


def render_console_summary(rows: list[dict[str, Any]]) -> str:
    lines = [
        "Phase 3 N3-05c component-wise quotient normalization",
        "-" * 148,
        (
            f"{'variant':<13} {'source':<28} {'ch':<8} {'comp':>7} {'act':>5} "
            f"{'ent':>7} {'eff':>7} {'top1':>7} {'top3':>7} "
            f"{'lam2w':>8} {'rhoNw':>8} {'degH':>7}  verdict"
        ),
    ]
    for row in rows:
        lines.append(
            f"{clip(row['variant'], 13):<13} {clip(row['source_label'], 28):<28} "
            f"{row['channel']:<8} {int(row['component_count']):>7} "
            f"{int(row['active_component_count']):>5} "
            f"{fmt(row['component_weight_entropy_norm']):>7} "
            f"{fmt(row['effective_weight_components']):>7} "
            f"{fmt(row['top1_component_weight_share']):>7} "
            f"{fmt(row['top3_component_weight_share']):>7} "
            f"{fmt(row['weighted_component_lambda2']):>8} "
            f"{fmt(row['weighted_mass_normalized_rho']):>8} "
            f"{fmt(row['weighted_degree_entropy_norm']):>7}  "
            f"{row['verdict']}"
        )
    return "\n".join(lines)


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase 3 N3-05c Component-wise Quotient Normalization",
        "",
        "## Contract",
        "",
        "- This readout starts from an existing N3-05b weighted quotient artifact.",
        "- It does not introduce Hamming balls, learned clustering, or fitted metrics.",
        "- Components are computed in the fixed prefix/suffix quotient graph.",
        "- Component weights are normalized to test internal shape rather than absolute mass.",
        "- This is an internal falsification check, not a geometric claim.",
        "",
        "## Selection",
        "",
        f"- Weighted quotient run: `{payload['selection']['weighted_quotient_run']}`",
        "",
        "## Compact Readout",
        "",
        "| Variant | Source | Channel | Components | Active | Entropy | Effective comps | Top1 | Top3 | Weighted lambda2 | Weighted norm-rho | Weighted degree entropy | Verdict |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["source_summary"]:
        lines.append(
            f"| {row['variant']} | {row['source_label']} | {row['channel']} | "
            f"{row['component_count']} | {row['active_component_count']} | "
            f"{fmt(row['component_weight_entropy_norm'])} | "
            f"{fmt(row['effective_weight_components'])} | "
            f"{fmt(row['top1_component_weight_share'])} | "
            f"{fmt(row['top3_component_weight_share'])} | "
            f"{fmt(row['weighted_component_lambda2'])} | "
            f"{fmt(row['weighted_mass_normalized_rho'])} | "
            f"{fmt(row['weighted_degree_entropy_norm'])} | {row['verdict']} |"
        )
    lines.extend(
        [
            "",
            "## Reading Boundary",
            "",
            "If observed B/E differ from matched-LZ only by absolute mass but not by normalized component shape, the prefix/suffix quotient should be treated as exhausted for Phase 3 geometry. If normalized component shape separates, the quotient still contains internal structure worth auditing before moving to any Hamming or metric quotient.",
        ]
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
