#!/usr/bin/env python3
"""
HSI v2 Phase 2 Parent Shell Atlas

Lag-aware structural atlas of the anchor shell after canonical revalidation.
It localizes shell deficit pattern by pattern inside structurally typed cohorts.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hsi_v2_phase2_transport_defect_strict import phase_print, select_observed_runs
from v2.common.cli import resolve_dir
from v2.common.naming import compact_int
from v2.phase2.parent_shell_atlas import (
    build_parent_shell_atlas_rows,
    build_shell_structural_metadata,
    render_parent_shell_atlas_console_summary,
    render_parent_shell_atlas_report,
    summarize_parent_shell_patterns,
    summarize_parent_shell_subtypes,
    summarize_shell_deficit_concentration,
    write_parent_shell_atlas_csv,
    write_parent_shell_concentration_csv,
    write_parent_shell_pattern_csv,
    write_parent_shell_structural_csv,
    write_parent_shell_subtype_csv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a lag-aware shell atlas from a canonical parent-survival "
            "revalidation artifact."
        )
    )
    parser.add_argument("--revalidation-run", type=str, required=True)
    parser.add_argument(
        "--phase1-dir",
        type=str,
        default="",
        help="Optional explicit Phase 1 directory override. Defaults to the path recorded in the revalidation artifact.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/hsi_v2/phase2/parent_shell_atlas",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    revalidation_path = resolve_dir(args.revalidation_run, anchor_file=__file__)
    if revalidation_path.is_dir():
        revalidation_path = revalidation_path / "summary.json"
    if not revalidation_path.is_file():
        parser.error(f"Revalidation summary not found: {revalidation_path}")

    output_dir = resolve_dir(args.output_dir, anchor_file=__file__)
    output_dir.mkdir(parents=True, exist_ok=True)

    phase_print("Loading canonical revalidation", str(revalidation_path), quiet=args.quiet)
    revalidation_summary = load_json(revalidation_path)

    lagaware_summary_path = Path(revalidation_summary["lagaware_summary_path"])
    lagaware_dataset_path = lagaware_summary_path.with_name("dataset.json")
    probe_summary_path = Path(revalidation_summary["probe_summary_path"])
    probe_dataset_path = probe_summary_path.with_name("dataset.json")
    if not lagaware_dataset_path.is_file():
        parser.error(f"Lag-aware dataset not found: {lagaware_dataset_path}")
    if not probe_dataset_path.is_file():
        parser.error(f"Probe dataset not found: {probe_dataset_path}")

    lagaware_payload = load_json(lagaware_dataset_path)
    lagaware_payload["_dataset_path"] = str(lagaware_dataset_path)
    probe_payload = load_json(probe_dataset_path)

    phase1_dir = resolve_phase1_dir(args.phase1_dir, revalidation_summary)
    anchor_run, family_inferred = recover_anchor_run(
        phase1_dir=phase1_dir,
        lagaware_selection=lagaware_payload["selection"],
    )

    probe_summary = probe_payload["summary"]
    shell_metadata = build_shell_structural_metadata(
        anchor_run,
        low_scale=int(lagaware_payload["selection"]["low_scale"]),
        high_scale=int(lagaware_payload["selection"]["high_scale"]),
        top_patterns=int(lagaware_payload["selection"]["top_patterns"]),
        pattern_selection=str(lagaware_payload["selection"]["pattern_selection"]),
        anchor_shell_patterns=set(probe_summary["anchor_shell_patterns"]),
    )

    phase_print(
        "Building shell atlas",
        (
            f"shell={shell_metadata['shell_pattern_count']} | "
            f"monitored={shell_metadata['monitored_shell_pattern_count']}"
        ),
        quiet=args.quiet,
    )
    atlas_rows = build_parent_shell_atlas_rows(
        lagaware_payload,
        shell_metadata=shell_metadata,
    )
    pattern_summary_rows = summarize_parent_shell_patterns(atlas_rows)
    structural_summary_rows = shell_metadata["structural_summary_rows"]
    subtype_summary_rows = summarize_parent_shell_subtypes(
        atlas_rows,
        structural_summary_rows=structural_summary_rows,
    )
    concentration_rows = summarize_shell_deficit_concentration(pattern_summary_rows)

    selection = build_selection(
        revalidation_summary=revalidation_summary,
        revalidation_path=revalidation_path,
        phase1_dir=phase1_dir,
        lagaware_payload=lagaware_payload,
        probe_payload=probe_payload,
        shell_metadata=shell_metadata,
        family_inferred=family_inferred,
        anchor_run_dir=Path(anchor_run["_run_dir"]).resolve(),
    )

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_slug = build_run_slug(selection, timestamp)
    run_dir = output_dir / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = run_dir / "dataset.json"
    summary_path = run_dir / "summary.json"
    report_path = run_dir / "report.md"
    structural_csv_path = run_dir / "structure.csv"
    subtype_csv_path = run_dir / "subtypes.csv"
    pattern_csv_path = run_dir / "patterns.csv"
    concentration_csv_path = run_dir / "concentration.csv"
    atlas_csv_path = run_dir / "atlas.csv"
    manifest_path = run_dir / "manifest.json"

    generated_at = datetime.now().isoformat(timespec="seconds")
    dataset_payload = {
        "stage": "phase2_parent_shell_atlas",
        "generated_at": generated_at,
        "selection": selection,
        "structural_summary": structural_summary_rows,
        "subtype_summary": subtype_summary_rows,
        "pattern_summary": pattern_summary_rows,
        "concentration_summary": concentration_rows,
        "atlas_rows": atlas_rows,
        "notes": [
            "The shell atlas is derived from the canonical lag-aware parent-survival revalidation artifact.",
            "Structural shell subtypes are computed from the anchor Phase 1 fibers only.",
            "Only shell parents with monitored high-scale child support appear in the lag-aware atlas rows.",
            "Net deficit mass is computed pattern by pattern as max(sum(anchor) - sum(candidate), 0).",
            "Gross loss pressure is aggregated as sum(max(anchor - candidate, 0)) across windows.",
            "Jitter gap mass is defined as gross loss pressure minus pooled net deficit mass.",
        ],
    }
    summary_payload = {
        "generated_at": generated_at,
        "selection": selection,
        "structural_summary": structural_summary_rows,
        "subtype_summary": subtype_summary_rows,
        "concentration_summary": concentration_rows,
        "top_patterns": [
            {
                "variant": row["variant"],
                "source_label": row["source_label"],
                "top_deficit_patterns": row["top_deficit_patterns"][:5],
            }
            for row in concentration_rows
        ],
    }
    manifest_payload = {
        "generated_at": generated_at,
        "script": "hsi_v2_phase2_parent_shell_atlas.py",
        "cwd": str(Path.cwd()),
        "inputs": {
            "revalidation_summary": str(revalidation_path),
            "probe_dataset": str(probe_dataset_path),
            "lagaware_dataset": str(lagaware_dataset_path),
            "phase1_dir": str(phase1_dir),
            "anchor_run_dir": str(Path(anchor_run["_run_dir"]).resolve()),
        },
        "outputs": {
            "dataset": str(dataset_path),
            "summary": str(summary_path),
            "report": str(report_path),
            "structure_csv": str(structural_csv_path),
            "subtype_csv": str(subtype_csv_path),
            "pattern_csv": str(pattern_csv_path),
            "concentration_csv": str(concentration_csv_path),
            "atlas_csv": str(atlas_csv_path),
            "manifest": str(manifest_path),
        },
        "arguments": vars(args),
    }

    phase_print("Writing artifacts", str(run_dir), quiet=args.quiet)
    with open(dataset_path, "w", encoding="utf-8") as handle:
        json.dump(dataset_payload, handle, indent=2)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(
            render_parent_shell_atlas_report(
                selection,
                structural_summary_rows,
                subtype_summary_rows,
                concentration_rows,
                pattern_summary_rows,
            )
            + "\n"
        )
    write_parent_shell_structural_csv(structural_summary_rows, structural_csv_path)
    write_parent_shell_subtype_csv(subtype_summary_rows, subtype_csv_path)
    write_parent_shell_pattern_csv(pattern_summary_rows, pattern_csv_path)
    write_parent_shell_concentration_csv(concentration_rows, concentration_csv_path)
    write_parent_shell_atlas_csv(atlas_rows, atlas_csv_path)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, indent=2)

    if not args.quiet:
        print(
            render_parent_shell_atlas_console_summary(
                structural_summary_rows,
                subtype_summary_rows,
                concentration_rows,
            )
        )
        print("")
        print(f"Saved dataset to: {dataset_path}")
        print(f"Saved summary to: {summary_path}")
        print(f"Saved report to: {report_path}")
        print(f"Saved structure CSV to: {structural_csv_path}")
        print(f"Saved subtype CSV to: {subtype_csv_path}")
        print(f"Saved pattern CSV to: {pattern_csv_path}")
        print(f"Saved concentration CSV to: {concentration_csv_path}")
        print(f"Saved atlas CSV to: {atlas_csv_path}")
        print(f"Saved manifest to: {manifest_path}")

    return 0


def resolve_phase1_dir(raw_override: str, revalidation_summary: dict) -> Path:
    if raw_override.strip():
        return resolve_dir(raw_override, anchor_file=__file__)
    selection = revalidation_summary.get("selection", {})
    phase1_dir = selection.get("phase1_dir", "results/hsi_v2/phase1_high_scales")
    return resolve_dir(str(phase1_dir), anchor_file=__file__)


def recover_anchor_run(*, phase1_dir: Path, lagaware_selection: dict) -> tuple[dict, bool]:
    from v2.phase2.null_pressure import discover_phase1_runs_recursive

    runs = discover_phase1_runs_recursive(phase1_dir)
    observed_runs, family_inferred = select_observed_runs(
        runs,
        variants=[str(item).upper() for item in lagaware_selection["variants"]],
        iteration=int(lagaware_selection["iteration"]),
        segment_bits=int(lagaware_selection["segment_bits"]),
        num_segments=int(lagaware_selection["num_segments"]),
        segment_offset_bits=None,
        scales=None,
        policies=None,
        low_scale=int(lagaware_selection["low_scale"]),
        high_scale=int(lagaware_selection["high_scale"]),
        no_family_inference=False,
    )
    observed_by_variant = {
        str(run["dataset"]["config"]["variant"]).upper(): run
        for run in observed_runs
    }
    return observed_by_variant[str(lagaware_selection["anchor_variant"]).upper()], family_inferred


def build_selection(
    *,
    revalidation_summary: dict,
    revalidation_path: Path,
    phase1_dir: Path,
    lagaware_payload: dict,
    probe_payload: dict,
    shell_metadata: dict,
    family_inferred: bool,
    anchor_run_dir: Path,
) -> dict:
    lag_sel = lagaware_payload["selection"]
    probe_summary = probe_payload["summary"]
    return {
        "anchor_variant": str(lag_sel["anchor_variant"]),
        "candidate_variant": str(lag_sel["candidate_variant"]),
        "variants": [str(item) for item in lag_sel["variants"]],
        "iteration": int(lag_sel["iteration"]),
        "segment_bits": int(lag_sel["segment_bits"]),
        "num_segments": int(lag_sel["num_segments"]),
        "offsets": [int(item) for item in lag_sel["offsets"]],
        "window_bits": int(lag_sel["window_bits"]),
        "low_scale": int(lag_sel["low_scale"]),
        "high_scale": int(lag_sel["high_scale"]),
        "top_patterns": int(lag_sel["top_patterns"]),
        "pattern_selection": str(lag_sel["pattern_selection"]),
        "candidate_lag_bits": int(lag_sel["candidate_lag_bits"]),
        "source_freeze_mode": str(lag_sel["source_freeze_mode"]),
        "frozen_total_bits": int(lag_sel["frozen_total_bits"]),
        "null_models": [str(item) for item in lag_sel.get("null_models", [])],
        "matched_lz_seeds": [int(item) for item in lag_sel.get("matched_lz_seeds", [])],
        "phase1_dir": str(phase1_dir),
        "family_inferred": bool(family_inferred),
        "revalidation_summary_path": str(revalidation_path),
        "probe_dataset_path": str(Path(revalidation_summary["probe_summary_path"]).with_name("dataset.json")),
        "lagaware_dataset_path": str(Path(revalidation_summary["lagaware_summary_path"]).with_name("dataset.json")),
        "anchor_run_dir": str(anchor_run_dir),
        "shell_pattern_count": int(shell_metadata["shell_pattern_count"]),
        "monitored_shell_pattern_count": int(shell_metadata["monitored_shell_pattern_count"]),
        "anchor_core_pattern_count": int(probe_summary["anchor_core_pattern_count"]),
        "anchor_shell_pattern_count": int(probe_summary["anchor_shell_pattern_count"]),
    }


def build_run_slug(selection: dict, timestamp: str) -> str:
    offsets = selection["offsets"]
    if len(offsets) == 1:
        offset_part = f"off-{compact_int(offsets[0])}"
    else:
        offset_part = f"off-{compact_int(min(offsets))}-plus-{len(offsets)}"
    return (
        f"phase2-parent-shell-atlas__anchor-{selection['anchor_variant']}"
        f"__cand-{selection['candidate_variant']}__lag-{compact_int(selection['candidate_lag_bits'])}"
        f"__m-{selection['low_scale']}-{selection['high_scale']}"
        f"__sel-{selection['pattern_selection']}__top-{selection['top_patterns']}"
        f"__{offset_part}__{timestamp}"
    )


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    raise SystemExit(main())
