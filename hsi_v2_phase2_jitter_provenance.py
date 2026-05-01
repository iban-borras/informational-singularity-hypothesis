#!/usr/bin/env python3
"""
HSI v2 Phase 2 Jitter Provenance

N2-07 derived readout over a parent-shell atlas artifact. It asks whether
local shell losses are uncompensated collapse or temporally compensated jitter
inside the same parent-pattern shell.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hsi_v2_phase2_transport_defect_strict import phase_print
from v2.common.cli import resolve_dir
from v2.common.naming import compact_int
from v2.phase2.jitter_provenance import (
    build_jitter_provenance_readout,
    render_jitter_provenance_console_summary,
    render_jitter_provenance_report,
    source_summary_for_json,
    write_pattern_provenance_csv,
    write_source_provenance_csv,
    write_window_deltas_csv,
)


DEFAULT_ATLAS_ROOT = "results/hsi_v2/phase2/parent_shell_atlas_pipeline"
DEFAULT_OUTPUT_DIR = "results/hsi_v2/phase2/jitter_provenance"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the N2-07 same-parent jitter-provenance readout from a "
            "parent-shell atlas dataset."
        )
    )
    parser.add_argument(
        "--atlas-run",
        type=str,
        default="",
        help=(
            "Path to an atlas run, pipeline run, or dataset.json. If omitted, "
            "the latest complete atlas under --atlas-root is used."
        ),
    )
    parser.add_argument(
        "--atlas-root",
        type=str,
        default=DEFAULT_ATLAS_ROOT,
        help="Root used to discover the latest canonical parent-shell atlas pipeline run.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where N2-07 artifacts will be written.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    atlas_dataset_path = resolve_atlas_dataset_path(args)
    output_root = resolve_dir(args.output_dir, anchor_file=__file__)
    output_root.mkdir(parents=True, exist_ok=True)

    phase_print(
        "Preparing N2-07 jitter provenance",
        f"atlas={atlas_dataset_path}",
        quiet=args.quiet,
    )

    atlas_payload = load_json(atlas_dataset_path)
    readout = build_jitter_provenance_readout(atlas_payload)
    selection = readout["selection"]

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_slug = build_run_slug(selection, atlas_dataset_path, timestamp)
    run_dir = output_root / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = run_dir / "dataset.json"
    summary_path = run_dir / "summary.json"
    report_path = run_dir / "report.md"
    source_csv_path = run_dir / "source_provenance.csv"
    pattern_csv_path = run_dir / "pattern_provenance.csv"
    window_csv_path = run_dir / "window_deltas.csv"
    manifest_path = run_dir / "manifest.json"

    generated_at = datetime.now().isoformat(timespec="seconds")
    dataset_payload = {
        "stage": "phase2_jitter_provenance",
        "generated_at": generated_at,
        "selection": {
            **selection,
            "atlas_dataset_path": str(atlas_dataset_path),
        },
        "notes": readout["notes"],
        "source_summary": readout["source_summary"],
        "pattern_summary": readout["pattern_summary"],
        "window_deltas": readout["window_deltas"],
    }
    summary_payload = {
        "generated_at": generated_at,
        "selection": {
            **selection,
            "atlas_dataset_path": str(atlas_dataset_path),
        },
        "notes": readout["notes"],
        "source_summary": source_summary_for_json(readout["source_summary"]),
    }
    manifest_payload = {
        "generated_at": generated_at,
        "run_slug": run_slug,
        "script": "hsi_v2_phase2_jitter_provenance.py",
        "cwd": str(Path.cwd()),
        "inputs": {
            "atlas_dataset": str(atlas_dataset_path),
        },
        "outputs": {
            "dataset": str(dataset_path),
            "summary": str(summary_path),
            "report": str(report_path),
            "source_csv": str(source_csv_path),
            "pattern_csv": str(pattern_csv_path),
            "window_csv": str(window_csv_path),
            "manifest": str(manifest_path),
        },
        "arguments": vars(args),
    }

    phase_print("Writing N2-07 artifacts", str(run_dir), quiet=args.quiet)
    with open(dataset_path, "w", encoding="utf-8") as handle:
        json.dump(dataset_payload, handle, indent=2)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(render_jitter_provenance_report(readout) + "\n")
    write_source_provenance_csv(readout["source_summary"], source_csv_path)
    write_pattern_provenance_csv(readout["pattern_summary"], pattern_csv_path)
    write_window_deltas_csv(readout["window_deltas"], window_csv_path)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, indent=2)

    if not args.quiet:
        print(render_jitter_provenance_console_summary(readout["source_summary"]))
        print("")
        print(f"Saved dataset to: {dataset_path}")
        print(f"Saved summary to: {summary_path}")
        print(f"Saved report to: {report_path}")
        print(f"Saved source CSV to: {source_csv_path}")
        print(f"Saved pattern CSV to: {pattern_csv_path}")
        print(f"Saved window CSV to: {window_csv_path}")
        print(f"Saved manifest to: {manifest_path}")

    return 0


def resolve_atlas_dataset_path(args) -> Path:
    if args.atlas_run.strip():
        path = resolve_dir(args.atlas_run, anchor_file=__file__)
        dataset = find_atlas_dataset(path)
        if dataset is None:
            raise SystemExit(f"No parent-shell atlas dataset found under {path}")
        return dataset

    root = resolve_dir(args.atlas_root, anchor_file=__file__)
    if not root.is_dir():
        raise SystemExit(f"Atlas root not found: {root}")
    dataset = find_latest_atlas_dataset(root)
    if dataset is None:
        raise SystemExit(f"No complete parent-shell atlas dataset found under {root}")
    return dataset


def find_atlas_dataset(path: Path) -> Path | None:
    path = path.resolve()
    if path.is_file():
        return path if is_parent_shell_atlas_dataset(path) else None
    direct = path / "dataset.json"
    if direct.is_file() and is_parent_shell_atlas_dataset(direct):
        return direct
    return find_latest_atlas_dataset(path)


def find_latest_atlas_dataset(root: Path) -> Path | None:
    candidates = [
        path
        for path in root.rglob("dataset.json")
        if path.parent.name.startswith("phase2-parent-shell-atlas__")
        and is_parent_shell_atlas_dataset(path)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def is_parent_shell_atlas_dataset(path: Path) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("stage") == "phase2_parent_shell_atlas"
        and isinstance(payload.get("atlas_rows"), list)
    )


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_run_slug(selection: dict, atlas_dataset_path: Path, timestamp: str) -> str:
    anchor = str(selection.get("anchor_variant", "anchor"))
    candidate = str(selection.get("candidate_variant", "cand"))
    top = selection.get("top_patterns", "top")
    lag = selection.get("candidate_lag_bits", "lag")
    source_stamp = atlas_dataset_path.parent.name.split("__")[-1]
    return (
        f"phase2-jitter-provenance__anchor-{anchor}__cand-{candidate}"
        f"__top-{top}__lag-{compact_int(int(lag)) if str(lag).isdigit() else lag}"
        f"__src-{source_stamp}__{timestamp}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
