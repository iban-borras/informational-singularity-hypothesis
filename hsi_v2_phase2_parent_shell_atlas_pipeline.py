#!/usr/bin/env python3
"""
HSI v2 Phase 2 Parent Shell Atlas Pipeline

Canonical one-command launcher for the shell-atlas line:

1. Canonical parent-survival revalidation.
2. Lag-aware structural shell atlas.

This script is intentionally thin. It reuses the audited public launchers
instead of duplicating their internal logic.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hsi_v2_phase2_parent_survival_revalidation import (
    append_flag,
    flush_output,
    latest_child_file,
    load_json,
    run_subprocess,
)
from hsi_v2_phase2_transport_defect_strict import phase_print
from v2.common.cli import resolve_dir


CANONICAL_ALL_PROFILE = {
    "stage": "all",
    "variants": "E,B",
    "anchor_variant": "E",
    "iteration": 20,
    "segment_bits": 1_000_000,
    "num_segments": 3,
    "scales": "8,12,16,20,24,28,32,40,48",
    "phase1_policies": "prefix,suffix",
    "low_scale": 40,
    "high_scale": 48,
    "top_patterns": 64,
    "pattern_selection": "bridge-linked",
    "definition_offsets": (
        "714000000,714500000,715000000,715500000,716000000,716500000,"
        "717000000,717500000,718000000,718500000,719000000,719500000,"
        "720000000,720500000,721000000,721500000,722000000,722500000,"
        "723000000"
    ),
    "offsets": (
        "714000000,714500000,715000000,715500000,716000000,716500000,"
        "717000000,717500000,718000000,718500000,719000000,719500000,"
        "720000000,720500000,721000000,721500000,722000000,722500000,"
        "723000000"
    ),
    "scan_start_bits": 714_000_000,
    "scan_stop_bits": 732_000_000,
    "scan_step_bits": 500_000,
    "scan_forward_bits": 12_000_000,
    "scan_backward_bits": 0,
    "null_models": "markov1,matched-lz",
    "matched_lz_seeds": "17,101,211,307,401,503",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the canonical Phase 2 parent-shell atlas pipeline in one command: "
            "shell-lag revalidation plus lag-aware shell atlas."
        )
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="canonical-all",
        choices=("canonical-all",),
        help="Named reproducible pipeline profile.",
    )
    parser.add_argument(
        "--phase1-dir",
        type=str,
        default="results/hsi_v2/phase1_high_scales",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/hsi_v2/phase2/parent_shell_atlas_pipeline",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="",
        choices=("", "observed", "nulls", "all"),
        help="Optional override over the profile stage.",
    )
    parser.add_argument(
        "--null-models",
        type=str,
        default="",
        help="Optional override over the profile null models.",
    )
    parser.add_argument(
        "--matched-lz-seeds",
        type=str,
        default="",
        help="Optional override over the profile matched-LZ seed list.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    selection = build_selection(args)
    script_dir = Path(__file__).parent.resolve()
    phase1_dir = resolve_dir(args.phase1_dir, anchor_file=__file__)
    output_dir = resolve_dir(args.output_dir, anchor_file=__file__)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_slug = build_run_slug(selection, timestamp)
    run_dir = output_dir / run_slug
    revalidation_root = run_dir / "revalidation"
    atlas_root = run_dir / "atlas"
    run_dir.mkdir(parents=True, exist_ok=True)

    phase_print(
        "Preparing canonical parent shell atlas pipeline",
        (
            f"profile={selection['profile']} | stage={selection['stage']} | "
            f"anchor={selection['anchor_variant']} | candidate={selection['candidate_variant']}"
        ),
        quiet=args.quiet,
    )
    flush_output()

    revalidation_cmd = [
        sys.executable,
        str(script_dir / "hsi_v2_phase2_parent_survival_revalidation.py"),
        "--stage",
        selection["stage"],
        "--phase1-dir",
        str(phase1_dir),
        "--output-dir",
        str(revalidation_root),
        "--variants",
        selection["variants"],
        "--anchor-variant",
        selection["anchor_variant"],
        "--iteration",
        str(selection["iteration"]),
        "--segment-bits",
        str(selection["segment_bits"]),
        "--num-segments",
        str(selection["num_segments"]),
        "--scales",
        selection["scales"],
        "--phase1-policies",
        selection["phase1_policies"],
        "--low-scale",
        str(selection["low_scale"]),
        "--high-scale",
        str(selection["high_scale"]),
        "--top-patterns",
        str(selection["top_patterns"]),
        "--pattern-selection",
        selection["pattern_selection"],
        "--definition-offsets",
        selection["definition_offsets"],
        "--offsets",
        selection["offsets"],
        "--scan-start-bits",
        str(selection["scan_start_bits"]),
        "--scan-stop-bits",
        str(selection["scan_stop_bits"]),
        "--scan-step-bits",
        str(selection["scan_step_bits"]),
        "--scan-forward-bits",
        str(selection["scan_forward_bits"]),
        "--scan-backward-bits",
        str(selection["scan_backward_bits"]),
    ]
    if selection["stage"] in {"nulls", "all"}:
        revalidation_cmd.extend(["--null-models", selection["null_models"]])
        if selection["matched_lz_seeds"]:
            revalidation_cmd.extend(["--matched-lz-seeds", selection["matched_lz_seeds"]])
    append_flag(revalidation_cmd, "--quiet", args.quiet)

    phase_print(
        "Stage 1/2: canonical parent-survival revalidation",
        f"output_dir={revalidation_root}",
        quiet=args.quiet,
    )
    flush_output()
    run_subprocess(revalidation_cmd, cwd=script_dir)
    revalidation_summary_path = latest_child_file(revalidation_root, "summary.json")
    revalidation_report_path = latest_child_file(revalidation_root, "report.md")
    revalidation_payload = load_json(revalidation_summary_path)

    atlas_cmd = [
        sys.executable,
        str(script_dir / "hsi_v2_phase2_parent_shell_atlas.py"),
        "--revalidation-run",
        str(revalidation_summary_path),
        "--output-dir",
        str(atlas_root),
    ]
    append_flag(atlas_cmd, "--quiet", args.quiet)

    phase_print(
        "Stage 2/2: lag-aware shell atlas",
        f"output_dir={atlas_root}",
        quiet=args.quiet,
    )
    flush_output()
    run_subprocess(atlas_cmd, cwd=script_dir)
    atlas_summary_path = latest_child_file(atlas_root, "summary.json")
    atlas_report_path = latest_child_file(atlas_root, "report.md")
    atlas_payload = load_json(atlas_summary_path)

    generated_at = datetime.now().isoformat(timespec="seconds")
    summary_path = run_dir / "summary.json"
    report_path = run_dir / "report.md"
    manifest_path = run_dir / "manifest.json"

    summary_payload = {
        "generated_at": generated_at,
        "selection": {
            "profile": selection["profile"],
            "stage": selection["stage"],
            "anchor_variant": selection["anchor_variant"],
            "candidate_variant": selection["candidate_variant"],
            "variants": selection["variants_list"],
            "phase1_dir": str(phase1_dir),
            "null_models": selection["null_models"].split(",") if selection["null_models"] else [],
            "matched_lz_seeds": [
                int(item) for item in selection["matched_lz_seeds"].split(",") if item.strip()
            ],
        },
        "revalidation_summary_path": str(revalidation_summary_path),
        "atlas_summary_path": str(atlas_summary_path),
        "recommended_lag_bits": revalidation_payload.get("recommended_lag_bits"),
        "revalidation_grouped_summary": revalidation_payload.get("lagaware_grouped_summary", []),
        "atlas_structural_summary": atlas_payload.get("structural_summary", []),
        "atlas_subtype_summary": atlas_payload.get("subtype_summary", []),
        "atlas_concentration_summary": atlas_payload.get("concentration_summary", []),
    }
    manifest_payload = {
        "generated_at": generated_at,
        "run_slug": run_slug,
        "script": "hsi_v2_phase2_parent_shell_atlas_pipeline.py",
        "cwd": str(Path.cwd()),
        "outputs": {
            "summary": str(summary_path),
            "report": str(report_path),
            "manifest": str(manifest_path),
            "revalidation_summary": str(revalidation_summary_path),
            "revalidation_report": str(revalidation_report_path),
            "atlas_summary": str(atlas_summary_path),
            "atlas_report": str(atlas_report_path),
        },
        "commands": {
            "revalidation": revalidation_cmd,
            "atlas": atlas_cmd,
        },
        "arguments": vars(args),
    }

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(render_report(summary_payload) + "\n")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, indent=2)

    if not args.quiet:
        print("Phase 2 canonical parent shell atlas pipeline")
        print("-" * 118)
        print(
            f"profile={selection['profile']} stage={selection['stage']} "
            f"lag={summary_payload['recommended_lag_bits']} "
            f"revalidation={revalidation_summary_path.parent.name} "
            f"atlas={atlas_summary_path.parent.name}"
        )
        print("")
        print(f"Saved pipeline summary to: {summary_path}")
        print(f"Saved pipeline report to: {report_path}")
        print(f"Saved pipeline manifest to: {manifest_path}")

    return 0


def build_selection(args) -> dict:
    if args.profile != "canonical-all":
        raise ValueError(f"Unsupported profile {args.profile!r}")

    selection = dict(CANONICAL_ALL_PROFILE)
    selection["profile"] = args.profile
    if args.stage:
        selection["stage"] = args.stage
    if args.null_models:
        selection["null_models"] = args.null_models
    if args.matched_lz_seeds:
        selection["matched_lz_seeds"] = args.matched_lz_seeds
    if selection["stage"] == "observed":
        selection["null_models"] = ""
        selection["matched_lz_seeds"] = ""
    elif "matched-lz" not in [item.strip() for item in selection["null_models"].split(",") if item.strip()]:
        selection["matched_lz_seeds"] = ""

    variants_list = [item.strip() for item in selection["variants"].split(",") if item.strip()]
    if len(variants_list) != 2:
        raise ValueError("The canonical parent shell atlas pipeline expects exactly two variants.")
    selection["variants_list"] = variants_list
    selection["candidate_variant"] = next(
        variant for variant in variants_list if variant != selection["anchor_variant"]
    )
    return selection


def render_report(payload: dict) -> str:
    selection = payload["selection"]
    revalidation_items = payload["revalidation_grouped_summary"]
    structural_items = payload["atlas_structural_summary"]
    concentration_items = payload["atlas_concentration_summary"]

    lines = [
        "# Phase 2 Parent Shell Atlas Pipeline",
        "",
        "## Selection",
        "",
        f"- Profile: {selection['profile']}",
        f"- Stage: {selection['stage']}",
        f"- Anchor variant: {selection['anchor_variant']}",
        f"- Candidate variant: {selection['candidate_variant']}",
        f"- Variants: {', '.join(selection['variants'])}",
        f"- Null models: {', '.join(selection['null_models']) or '-'}",
        f"- Matched-LZ seeds: {', '.join(str(value) for value in selection['matched_lz_seeds']) or '-'}",
        "",
        "## Pipeline Outcome",
        "",
        f"- Recommended lag bits: {payload['recommended_lag_bits']}",
        f"- Revalidation summary path: {payload['revalidation_summary_path']}",
        f"- Atlas summary path: {payload['atlas_summary_path']}",
        "",
        "## Revalidation Summary",
        "",
        "| Variant | Source | Exact rng | Dead rng | Exact m rng | Dead m rng | Ex IDs | Dead IDs |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for item in revalidation_items:
        lines.append(
            f"| {item['variant']} | {item['source_label']} | "
            f"{item['exact_parent_count_min']}-{item['exact_parent_count_max']} | "
            f"{item['dead_parent_count_min']}-{item['dead_parent_count_max']} | "
            f"{item['exact_mass_fraction_min']:.4f}-{item['exact_mass_fraction_max']:.4f} | "
            f"{item['dead_mass_fraction_min']:.4f}-{item['dead_mass_fraction_max']:.4f} | "
            f"{'yes' if item['exact_pattern_identity_stable'] else 'no'} | "
            f"{'yes' if item['dead_pattern_identity_stable'] else 'no'} |"
        )

    lines.extend(
        [
            "",
            "## Shell Structure",
            "",
            "| Subtype | Shell patterns | Monitored | Unmonitored | Fraction |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in structural_items:
        lines.append(
            f"| {item['shell_subtype_label']} | {item['shell_pattern_count']} | "
            f"{item['monitored_shell_pattern_count']} | {item['unmonitored_shell_pattern_count']} | "
            f"{fmt(item['shell_pattern_fraction'])} |"
        )

    lines.extend(
        [
            "",
            "## Atlas Concentration Summary",
            "",
            "| Variant | Source | Net def | Gross loss | Jitter gap | Top1 | Top3 | Top5 | Dominant subtype | Dom share |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
        ]
    )
    for item in concentration_items:
        lines.append(
            f"| {item['variant']} | {item['source_label']} | "
            f"{item['net_deficit_mass_sum']} | {item['gross_loss_pressure_sum']} | "
            f"{item['jitter_gap_mass']} | "
            f"{fmt(item['top1_net_deficit_share'])} | "
            f"{fmt(item['top3_net_deficit_share'])} | "
            f"{fmt(item['top5_net_deficit_share'])} | "
            f"{item['dominant_subtype_label'] or '-'} | "
            f"{fmt(item['dominant_subtype_share'])} |"
        )

    return "\n".join(lines)


def build_run_slug(selection: dict, timestamp: str) -> str:
    return (
        f"phase2-parent-shell-atlas-pipeline__profile-{selection['profile']}"
        f"__stage-{selection['stage']}__anchor-{selection['anchor_variant']}"
        f"__cand-{selection['candidate_variant']}__{timestamp}"
    )


def fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
