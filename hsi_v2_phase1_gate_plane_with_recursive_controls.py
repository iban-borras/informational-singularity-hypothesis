#!/usr/bin/env python3
"""Re-emit the Phase 1 gate plane with Q4 recursive external controls O/P."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
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
    write_csv,
)


EXPECTED_RECURSIVE_COORDS = {
    "O": {"retention_last": 0.9433, "tail_transport_active_mean": 0.2457},
    "P": {"retention_last": 1.0000, "tail_transport_active_mean": 0.2269},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only Phase 1 gate-plane refresh that adds the pre-registered Q4 "
            "recursive external controls O/P. No upstream Phase 1 objects are regenerated."
        )
    )
    parser.add_argument("--observed-dir", default="results/hsi_v2/phase1")
    parser.add_argument("--null-dir", default="results/hsi_v2/phase1/nulls")
    parser.add_argument(
        "--recursive-controls-dir",
        default="results/hsi_v2/phase1_external_recursive_controls",
        help="Directory containing the D-0087 O/P observed Phase 1 towers.",
    )
    parser.add_argument(
        "--q4-transport-summary",
        default=(
            "results/hsi_v2/phase1_external_recursive_controls/transport_op/"
            "phase1-transport__var-O-P__d0092__20260507T074318/summary.json"
        ),
        help="Canonical D-0092/Q4 transport summary used to validate O/P coordinates.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/hsi_v2/visualizations",
        help="Root where the timestamped visualization directory is written.",
    )
    parser.add_argument(
        "--paper-png",
        default="../Paper_Latex/phase1_gate_plane_tail2.png",
        help="Paper-facing PNG path to overwrite after rendering.",
    )
    parser.add_argument(
        "--tail-edges",
        type=int,
        default=2,
        help="Number of tail edges used for active_mean_tail.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    observed_dir = resolve_dir(args.observed_dir, anchor_file=__file__)
    null_dir = resolve_dir(args.null_dir, anchor_file=__file__)
    recursive_controls_dir = resolve_dir(args.recursive_controls_dir, anchor_file=__file__)
    output_root = resolve_dir(args.output_dir, anchor_file=__file__)
    paper_png = resolve_dir(args.paper_png, anchor_file=__file__)
    q4_transport_summary = resolve_dir(args.q4_transport_summary, anchor_file=__file__)

    if not q4_transport_summary.is_file():
        raise SystemExit(f"Missing Q4 transport summary: {q4_transport_summary}")
    q4_payload = json.loads(q4_transport_summary.read_text(encoding="utf-8"))

    rows, selection = load_gate_map_rows(
        observed_dir,
        null_dir,
        observed_variants=list(DEFAULT_OBSERVED_VARIANTS),
        null_variants=list(DEFAULT_NULL_VARIANTS),
        extra_observed_dirs=[recursive_controls_dir],
        tail_edges=args.tail_edges,
    )
    validate_recursive_coordinates(rows, q4_payload)

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = output_root / f"phase1-gate-plane-with-recursive-controls__{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    slug = (
        f"phase1-gate-map__obs-{len(selection['observed_variants'])}"
        f"__null-{len(selection['null_variants'])}"
        f"__tail-{selection['tail_edges']}__{timestamp}"
    )

    json_path = run_dir / f"{slug}.json"
    csv_path = run_dir / f"{slug}.csv"
    svg_path = run_dir / f"{slug}.svg"
    png_path = run_dir / f"{slug}.png"
    canonical_png_path = run_dir / "phase1_gate_plane_tail2.png"
    report_path = run_dir / "report.md"
    manifest_path = run_dir / "manifest.json"

    payload = {
        "generated_at": selection["generated_at"],
        "selection": selection,
        "q4_recursive_controls": q4_summary_for_manifest(q4_payload, q4_transport_summary),
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(rows, csv_path)
    svg_path.write_text(render_gate_svg(rows) + "\n", encoding="utf-8")
    png_written = render_gate_png(rows, png_path)
    if not png_written:
        raise SystemExit("PNG rendering failed; matplotlib is required for the paper-facing PNG.")

    shutil.copy2(png_path, canonical_png_path)
    paper_png.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(png_path, paper_png)

    report_path.write_text(render_report(rows, selection, png_path, q4_transport_summary), encoding="utf-8")

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": Path(__file__).name,
        "command": [sys.executable, *sys.argv],
        "contract": {
            "purpose": "Refresh Paper_Latex/phase1_gate_plane_tail2.png with Q4 recursive external controls O/P.",
            "mode": "read-only derivation; no upstream Phase 1 objects regenerated",
        },
        "inputs": {
            "observed_dir": str(observed_dir),
            "null_dir": str(null_dir),
            "recursive_controls_dir": str(recursive_controls_dir),
            "q4_transport_summary": str(q4_transport_summary),
            "q4_transport_summary_sha256": sha256_file(q4_transport_summary),
            "source_rows": source_rows(rows),
        },
        "outputs": {
            "summary_json": str(json_path),
            "csv": str(csv_path),
            "svg": str(svg_path),
            "png": str(png_path),
            "canonical_png": str(canonical_png_path),
            "paper_png_copy": str(paper_png),
            "report": str(report_path),
            "manifest": str(manifest_path),
        },
        "output_hashes": {
            "png_sha256": sha256_file(png_path),
            "canonical_png_sha256": sha256_file(canonical_png_path),
            "paper_png_sha256": sha256_file(paper_png),
            "svg_sha256": sha256_file(svg_path),
            "summary_json_sha256": sha256_file(json_path),
            "csv_sha256": sha256_file(csv_path),
        },
        "represented_variants": [row["variant"] for row in rows],
        "recursive_control_coordinates": {
            row["variant"]: {
                "retention_last": row["retention_last"],
                "active_mean_tail": row["tail_transport_active_mean"],
            }
            for row in rows
            if row["variant"] in {"O", "P"}
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not args.quiet:
        print("Phase 1 gate plane with recursive external controls")
        print("-" * 86)
        print(render_console_summary(rows))
        print("")
        print(f"Saved JSON summary to: {json_path}")
        print(f"Saved CSV table to: {csv_path}")
        print(f"Saved SVG figure to: {svg_path}")
        print(f"Saved PNG figure to: {png_path}")
        print(f"Saved canonical PNG copy to: {canonical_png_path}")
        print(f"Copied paper PNG to: {paper_png}")
        print(f"Saved report to: {report_path}")
        print(f"Saved manifest to: {manifest_path}")
        print(f"PNG SHA256: {manifest['output_hashes']['png_sha256']}")
    return 0


def validate_recursive_coordinates(rows: list[dict], q4_payload: dict) -> None:
    row_by_variant = {row["variant"]: row for row in rows}
    q4_by_variant = {row["variant"]: row for row in q4_payload.get("rows", [])}
    for variant, expected in EXPECTED_RECURSIVE_COORDS.items():
        if variant not in row_by_variant:
            raise SystemExit(f"Missing recursive control in gate-plane rows: {variant}")
        if variant not in q4_by_variant:
            raise SystemExit(f"Missing recursive control in Q4 transport summary: {variant}")
        row = row_by_variant[variant]
        q4_row = q4_by_variant[variant]
        active = float(row["tail_transport_active_mean"])
        q4_active = float(q4_row["tail_transport_active_mean"])
        retention = float(row["retention_last"])
        if abs(active - q4_active) > 5e-7:
            raise SystemExit(
                f"{variant}: active_mean_tail mismatch with Q4 summary "
                f"({active:.8f} vs {q4_active:.8f})"
            )
        if abs(active - expected["tail_transport_active_mean"]) > 5e-4:
            raise SystemExit(f"{variant}: active_mean_tail differs from reference value {expected}")
        if abs(retention - expected["retention_last"]) > 5e-4:
            raise SystemExit(f"{variant}: retention@last differs from reference value {expected}")


def q4_summary_for_manifest(q4_payload: dict, path: Path) -> dict:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "contract": q4_payload.get("contract", {}),
        "selection": q4_payload.get("selection", {}),
        "coordinates": {
            row["variant"]: {
                "retention_last": _retention_from_q4_source(row),
                "active_mean_tail": row["tail_transport_active_mean"],
                "summary_path": row["summary_path"],
            }
            for row in q4_payload.get("rows", [])
            if row.get("variant") in {"O", "P"}
        },
    }


def _retention_from_q4_source(row: dict) -> float:
    summary_path = Path(row["summary_path"])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    last = summary["scale_summary"][-1]
    candidate = float(last["candidate_pattern_count"])
    return float(last["kept_pattern_count"]) / candidate if candidate else 0.0


def source_rows(rows: list[dict]) -> list[dict]:
    return [
        {
            "variant": row["variant"],
            "summary_path": row["summary_path"],
            "summary_sha256": sha256_file(Path(row["summary_path"])),
            "run_dir": row["run_dir"],
        }
        for row in rows
    ]


def render_report(rows: list[dict], selection: dict, png_path: Path, q4_transport_summary: Path) -> str:
    op_rows = [row for row in rows if row["variant"] in {"O", "P"}]
    lines = [
        "# Phase 1 Gate Plane with Recursive External Controls",
        "",
        "## Scope",
        "",
        "- Read-only refresh of the Phase 1 gate-plane PNG used by the manuscript.",
        "- Includes the pre-registered external recursive controls O and P from the Q4 cycle.",
        "- O/P coordinates are re-derived from existing `phase1-transport` artifacts; no upstream objects are regenerated.",
        "- The figure is presentational and introduces no new claim.",
        "",
        "## Q4 source",
        "",
        f"- Q4 transport summary: `{q4_transport_summary}`",
        f"- Q4 transport summary SHA256: `{sha256_file(q4_transport_summary)}`",
        "",
        "## Recursive-control coordinates",
        "",
        "| variant | retention@last | active_mean_tail | source summary |",
        "|---|---:|---:|---|",
    ]
    for row in op_rows:
        lines.append(
            "| "
            f"{row['variant']} | "
            f"{float(row['retention_last']):.4f} | "
            f"{float(row['tail_transport_active_mean']):.4f} | "
            f"`{row['summary_path']}` |"
        )
    lines.extend(
        [
            "",
            "## Figure",
            "",
            f"![Phase 1 Gate Plane](./{png_path.name})",
            "",
            "## Wording fence",
            "",
            "- Caption basis: including the pre-registered external recursive controls O and P from the Q4 cycle (read-only `phase1-transport` re-derivation, no upstream object regeneration).",
            "- The plot uses `retention@last` and `active_mean_tail` as already defined Phase 1 observables.",
            "- No geometry, physical analogy, or new Level 2 claim is introduced by this visualization refresh.",
            "",
            "## Represented variants",
            "",
            f"- Observed: `{', '.join(selection['observed_variants'])}`",
            f"- Strong nulls: `{', '.join(selection['null_variants'])}`",
            "",
        ]
    )
    return "\n".join(lines)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


if __name__ == "__main__":
    raise SystemExit(main())
