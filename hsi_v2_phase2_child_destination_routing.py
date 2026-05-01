#!/usr/bin/env python3
"""
HSI v2 Phase 2 Child-Destination Routing

N2-08 derived readout over a parent-shell atlas artifact. It checks whether
candidate shell mass lands on the same high-scale child destinations as the
anchor, without claiming cross-parent routing.
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
from v2.phase2.child_destination_routing import (
    build_child_destination_routing_readout,
    render_child_destination_console_summary,
    render_child_destination_report,
    write_pattern_routing_csv,
    write_routing_rows_csv,
    write_source_routing_csv,
)


DEFAULT_ATLAS_ROOT = "results/hsi_v2/phase2/parent_shell_atlas_sensitivity"
DEFAULT_OUTPUT_DIR = "results/hsi_v2/phase2/child_destination_routing"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the N2-08 child-destination routing readout from a "
            "parent-shell atlas dataset."
        )
    )
    parser.add_argument(
        "--atlas-run",
        type=str,
        default="",
        help=(
            "Path to an atlas run or dataset.json. If omitted, the latest "
            "top/lag matching atlas under --atlas-root is used."
        ),
    )
    parser.add_argument(
        "--atlas-root",
        type=str,
        default=DEFAULT_ATLAS_ROOT,
        help="Root used to discover an atlas dataset when --atlas-run is omitted.",
    )
    parser.add_argument(
        "--top-patterns",
        type=int,
        default=128,
        help="Top-k atlas to use by default. top=128 exposes non-trivial child routing.",
    )
    parser.add_argument(
        "--lag-bits",
        type=int,
        default=15000000,
        help="Candidate lag to use by default.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where N2-08 artifacts will be written.",
    )
    parser.add_argument(
        "--allow-preview-truncated",
        action="store_true",
        help=(
            "Allow exploratory output when upstream kernel previews are truncated. "
            "By default, N2-08 fails because routing needs complete monitored child maps."
        ),
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.top_patterns <= 0:
        parser.error("--top-patterns must be positive.")
    if args.lag_bits <= 0:
        parser.error("--lag-bits must be positive.")

    atlas_dataset_path = resolve_atlas_dataset_path(args)
    output_root = resolve_dir(args.output_dir, anchor_file=__file__)
    output_root.mkdir(parents=True, exist_ok=True)

    phase_print(
        "Preparing N2-08 child-destination routing",
        f"atlas={atlas_dataset_path}",
        quiet=args.quiet,
    )

    atlas_payload = load_json(atlas_dataset_path)
    readout = build_child_destination_routing_readout(
        atlas_payload,
        allow_preview_truncated=args.allow_preview_truncated,
    )
    selection = readout["selection"]

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_slug = build_run_slug(selection, atlas_dataset_path, timestamp)
    run_dir = output_root / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = run_dir / "dataset.json"
    summary_path = run_dir / "summary.json"
    report_path = run_dir / "report.md"
    source_csv_path = run_dir / "source_routing.csv"
    pattern_csv_path = run_dir / "pattern_routing.csv"
    routing_csv_path = run_dir / "routing_rows.csv"
    manifest_path = run_dir / "manifest.json"

    generated_at = datetime.now().isoformat(timespec="seconds")
    dataset_payload = {
        "stage": "phase2_child_destination_routing",
        "generated_at": generated_at,
        "selection": {
            **selection,
            "atlas_dataset_path": str(atlas_dataset_path),
        },
        "notes": readout["notes"],
        "source_summary": readout["source_summary"],
        "pattern_summary": readout["pattern_summary"],
        "routing_rows": readout["routing_rows"],
    }
    summary_payload = {
        "generated_at": generated_at,
        "selection": {
            **selection,
            "atlas_dataset_path": str(atlas_dataset_path),
        },
        "notes": readout["notes"],
        "source_summary": readout["source_summary"],
    }
    manifest_payload = {
        "generated_at": generated_at,
        "run_slug": run_slug,
        "script": "hsi_v2_phase2_child_destination_routing.py",
        "cwd": str(Path.cwd()),
        "inputs": {
            "atlas_dataset": str(atlas_dataset_path),
            "lagaware_dataset": selection.get("lagaware_dataset_path"),
        },
        "outputs": {
            "dataset": str(dataset_path),
            "summary": str(summary_path),
            "report": str(report_path),
            "source_csv": str(source_csv_path),
            "pattern_csv": str(pattern_csv_path),
            "routing_csv": str(routing_csv_path),
            "manifest": str(manifest_path),
        },
        "arguments": vars(args),
    }

    phase_print("Writing N2-08 artifacts", str(run_dir), quiet=args.quiet)
    with open(dataset_path, "w", encoding="utf-8") as handle:
        json.dump(dataset_payload, handle, indent=2)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(render_child_destination_report(readout) + "\n")
    write_source_routing_csv(readout["source_summary"], source_csv_path)
    write_pattern_routing_csv(readout["pattern_summary"], pattern_csv_path)
    write_routing_rows_csv(readout["routing_rows"], routing_csv_path)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, indent=2)

    if not args.quiet:
        print(render_child_destination_console_summary(readout["source_summary"]))
        print("")
        print(f"Saved dataset to: {dataset_path}")
        print(f"Saved summary to: {summary_path}")
        print(f"Saved report to: {report_path}")
        print(f"Saved source CSV to: {source_csv_path}")
        print(f"Saved pattern CSV to: {pattern_csv_path}")
        print(f"Saved routing CSV to: {routing_csv_path}")
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
    dataset = find_latest_matching_atlas_dataset(
        root,
        top_patterns=args.top_patterns,
        lag_bits=args.lag_bits,
    )
    if dataset is None:
        raise SystemExit(
            f"No matching parent-shell atlas dataset found under {root} "
            f"for top={args.top_patterns}, lag={args.lag_bits}."
        )
    return dataset


def find_atlas_dataset(path: Path) -> Path | None:
    path = path.resolve()
    if path.is_file():
        return path if is_parent_shell_atlas_dataset(path) else None
    direct = path / "dataset.json"
    if direct.is_file() and is_parent_shell_atlas_dataset(direct):
        return direct
    candidates = [
        item
        for item in path.rglob("dataset.json")
        if item.parent.name.startswith("phase2-parent-shell-atlas__")
        and is_parent_shell_atlas_dataset(item)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime)


def find_latest_matching_atlas_dataset(
    root: Path,
    *,
    top_patterns: int,
    lag_bits: int,
) -> Path | None:
    candidates = []
    for path in root.rglob("dataset.json"):
        if not path.parent.name.startswith("phase2-parent-shell-atlas__"):
            continue
        if not is_parent_shell_atlas_dataset(path):
            continue
        payload = load_json(path)
        selection = payload.get("selection", {})
        if int(selection.get("top_patterns", 0)) != int(top_patterns):
            continue
        if int(selection.get("candidate_lag_bits", 0)) != int(lag_bits):
            continue
        candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime)


def is_parent_shell_atlas_dataset(path: Path) -> bool:
    try:
        payload = load_json(path)
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
    lag_label = compact_int(int(lag)) if str(lag).isdigit() else str(lag)
    return (
        f"phase2-child-destination-routing__anchor-{anchor}__cand-{candidate}"
        f"__top-{top}__lag-{lag_label}__src-{source_stamp}__{timestamp}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
