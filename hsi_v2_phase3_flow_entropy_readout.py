#!/usr/bin/env python3
"""HSI v2 Phase 3 N3-07 direct flow-entropy readout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from hsi_v2_phase3_quotient_child_routing_spectral_graph import resolve_latest_run
from v2.common.cli import resolve_dir
from v2.phase3.flow_entropy_readout import build_flow_entropy_readout


DEFAULT_COMPONENT_ROOT = "results/hsi_v2/phase3/component_quotient_child_routing_spectrum"
DEFAULT_OUTPUT_DIR = "results/hsi_v2/phase3/flow_entropy_readout"

SUMMARY_FIELDS = [
    "variant",
    "source_label",
    "source_class",
    "channel",
    "component_entropy",
    "effective_components",
    "top3_share",
    "top5_share",
    "matched_entropy_min",
    "matched_entropy_avg",
    "matched_entropy_max",
    "matched_effective_min",
    "matched_effective_avg",
    "matched_effective_max",
    "matched_top3_min",
    "matched_top3_avg",
    "matched_top3_max",
    "matched_top5_min",
    "matched_top5_avg",
    "matched_top5_max",
    "entropy_gap_vs_best_matched",
    "effective_gap_vs_best_matched",
    "top3_gap_vs_best_matched",
    "top5_gap_vs_best_matched",
    "flow_entropy_score",
    "verdict",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build N3-07: a direct low-entropy flow readout from the latest "
            "N3-05c component-normalized quotient artifact."
        )
    )
    parser.add_argument(
        "--component-run",
        default="",
        help="Path to an N3-05c component quotient run. Defaults to latest under --component-root.",
    )
    parser.add_argument("--component-root", default=DEFAULT_COMPONENT_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    component_run = resolve_latest_run(
        explicit=args.component_run,
        root=args.component_root,
        required=("source_summary.csv", "component_summary.csv", "summary.json"),
    )
    output_root = resolve_dir(args.output_dir, anchor_file=__file__)
    output_root.mkdir(parents=True, exist_ok=True)

    if not args.quiet:
        print("[Phase] Preparing N3-07 direct flow-entropy readout")
        print(f"        component_run={component_run.name}")

    readout = build_flow_entropy_readout(component_run)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    source_hash = hashlib.sha1(str(component_run).encode("utf-8")).hexdigest()[:10]
    run_slug = f"phase3-flow-entropy-readout__src-{source_hash}__{timestamp}"
    run_dir = output_root / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_path = run_dir / "summary.json"
    report_path = run_dir / "report.md"
    csv_path = run_dir / "flow_entropy_summary.csv"
    manifest_path = run_dir / "manifest.json"

    payload = {
        "stage": "phase3_flow_entropy_readout",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **readout,
    }
    manifest = {
        "run_slug": run_slug,
        "script": Path(__file__).name,
        "cwd": str(Path.cwd()),
        "inputs": {
            "component_run": str(component_run),
        },
        "outputs": {
            "summary": str(summary_path),
            "report": str(report_path),
            "flow_entropy_summary_csv": str(csv_path),
            "manifest": str(manifest_path),
        },
        "arguments": vars(args),
    }

    write_json(summary_path, payload)
    write_json(manifest_path, manifest)
    write_csv(csv_path, payload["flow_entropy_rows"], SUMMARY_FIELDS)
    report_path.write_text(render_report(payload), encoding="utf-8")

    if not args.quiet:
        print(render_console_summary(payload["flow_entropy_rows"]))
        print(f"\nSaved summary to: {summary_path}")
        print(f"Saved report to: {report_path}")
        print(f"Saved CSV to: {csv_path}")
        print(f"Saved manifest to: {manifest_path}")
    return 0


def render_console_summary(rows: list[dict[str, Any]]) -> str:
    lines = [
        "Phase 3 N3-07 direct flow-entropy readout",
        "-" * 158,
        (
            f"{'variant':<13} {'source':<18} {'ch':<8} {'H':>7} {'eff':>7} "
            f"{'top3':>7} {'top5':>7} {'dH':>8} {'dEff':>8} "
            f"{'dTop3':>8} {'dTop5':>8} {'score':>5}  verdict"
        ),
    ]
    for row in rows:
        lines.append(
            f"{clip(row['variant'], 13):<13} {clip(row['source_label'], 18):<18} "
            f"{row['channel']:<8} {fmt(row['component_entropy']):>7} "
            f"{fmt(row['effective_components']):>7} {fmt(row['top3_share']):>7} "
            f"{fmt(row['top5_share']):>7} "
            f"{fmt(row['entropy_gap_vs_best_matched']):>8} "
            f"{fmt(row['effective_gap_vs_best_matched']):>8} "
            f"{fmt(row['top3_gap_vs_best_matched']):>8} "
            f"{fmt(row['top5_gap_vs_best_matched']):>8} "
            f"{int(row['flow_entropy_score']):>5}  {row['verdict']}"
        )
    return "\n".join(lines)


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase 3 N3-07 Direct Flow-Entropy Readout",
        "",
        "## Contract",
        "",
        "- This readout is derived from N3-05c component-normalized quotient artifacts.",
        "- It does not define a new graph, quotient, metric, or radius.",
        "- It scores observed B/E against the matched-LZ seed envelope per channel.",
        "- Lower entropy and fewer effective components are interpreted as more canalized flow.",
        "- Higher top-k component share is interpreted as stronger concentration.",
        "- This is internal and pre-geometric; it is not a topology or curvature claim.",
        "",
        "## Selection",
        "",
        f"- Component run: `{payload['selection']['component_run']}`",
        "",
        "## Compact Readout",
        "",
        "| Variant | Source | Channel | H | Eff. comps | Top3 | Top5 | dH vs best matched | dEff vs best matched | dTop3 vs best matched | dTop5 vs best matched | Score | Verdict |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["flow_entropy_rows"]:
        lines.append(
            f"| {row['variant']} | {row['source_label']} | {row['channel']} | "
            f"{fmt(row['component_entropy'])} | {fmt(row['effective_components'])} | "
            f"{fmt(row['top3_share'])} | {fmt(row['top5_share'])} | "
            f"{fmt(row['entropy_gap_vs_best_matched'])} | "
            f"{fmt(row['effective_gap_vs_best_matched'])} | "
            f"{fmt(row['top3_gap_vs_best_matched'])} | "
            f"{fmt(row['top5_gap_vs_best_matched'])} | "
            f"{row['flow_entropy_score']} | {row['verdict']} |"
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
