#!/usr/bin/env python3
"""
HSI v2 Phase 2 Child-Routing Boundary Sweep

N2-10d reproducible wrapper. It runs the fine boundary experiment:

1. phase-lag atlas over selected bands,
2. tri-estimator audit over the atlas,
3. targeted child-routing validation over selected confidence classes.

The wrapper is intentionally thin: it delegates the audited work to the
existing N2-10/N2-10b/N2-10c scripts and records the exact commands.
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
from hsi_v2_phase2_transport_defect_strict import parse_int_list, phase_print
from v2.common.cli import resolve_dir
from v2.common.naming import compact_int


DEFAULT_OUTPUT_DIR = "results/hsi_v2/phase2/child_routing_boundary_sweep"
DEFAULT_BAND_STARTS = (
    "684000000,687000000,690000000,693000000,696000000,"
    "705000000,714000000,"
    "723000000,726000000,729000000,732000000,735000000"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the N2-10d fine boundary sweep around the child-routing "
            "phase-local plateau."
        )
    )
    parser.add_argument("--phase1-dir", type=str, default="results/hsi_v2/phase1_high_scales")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--band-starts", type=str, default=DEFAULT_BAND_STARTS)
    parser.add_argument("--window-count", type=int, default=19)
    parser.add_argument("--window-step-bits", type=int, default=500_000)
    parser.add_argument("--probe-backward-bits", type=int, default=45_000_000)
    parser.add_argument("--probe-forward-bits", type=int, default=45_000_000)
    parser.add_argument("--scan-step-bits", type=int, default=500_000)
    parser.add_argument("--top-patterns", type=int, default=128)
    parser.add_argument("--tolerance-bits", type=int, default=1_000_000)
    parser.add_argument(
        "--validation-confidences",
        type=str,
        default="strong,transition",
        help="Confidence classes sent to child-routing validation.",
    )
    parser.add_argument(
        "--validation-lag-policy",
        type=str,
        default="dual-transition",
        choices=("consensus", "first-full", "dual-transition"),
        help=(
            "Lag policy used by N2-10c validation. The default revalidates "
            "transition bands against both consensus and first-full lags."
        ),
    )
    parser.add_argument("--stage", type=str, default="all", choices=("observed", "nulls", "all"))
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)

    root = Path(__file__).parent.resolve()
    phase1_dir = resolve_dir(args.phase1_dir, anchor_file=__file__)
    output_root = resolve_dir(args.output_dir, anchor_file=__file__)
    output_root.mkdir(parents=True, exist_ok=True)
    band_starts = parse_int_list(args.band_starts, label="--band-starts", allow_zero=True)

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_slug = build_run_slug(args, band_starts, timestamp)
    run_dir = output_root / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)

    phase_print(
        "Preparing N2-10d child-routing boundary sweep",
        (
            f"bands={len(band_starts)} | top={args.top_patterns} | "
            f"stage={args.stage} | validation={not args.skip_validation}"
        ),
        quiet=args.quiet,
    )
    flush_output()

    atlas_dir = run_dir / "atlas"
    tri_dir = run_dir / "tri_estimator"
    validation_dir = run_dir / "validation"
    commands: list[dict] = []

    atlas_cmd = build_atlas_command(root, phase1_dir, atlas_dir, args, band_starts)
    tri_summary_path = None
    atlas_summary_path = None
    validation_summary_path = None

    commands.append(command_record("phase_lag_atlas", atlas_cmd))
    if args.dry_run:
        atlas_summary_path = atlas_dir / "<dry-run-summary.json>"
        tri_summary_path = tri_dir / "<dry-run-summary.json>"
    else:
        phase_print("Stage 1/3: phase-lag atlas", str(atlas_dir), quiet=args.quiet)
        flush_output()
        run_subprocess(atlas_cmd, cwd=root)
        atlas_summary_path = latest_child_file(atlas_dir, "summary.json")

        tri_cmd = build_tri_command(root, atlas_summary_path, tri_dir, args)
        commands.append(command_record("tri_estimator", tri_cmd))
        phase_print("Stage 2/3: tri-estimator audit", str(tri_dir), quiet=args.quiet)
        flush_output()
        run_subprocess(tri_cmd, cwd=root)
        tri_summary_path = latest_child_file(tri_dir, "summary.json")

        if not args.skip_validation:
            validation_cmd = build_validation_command(
                root=root,
                phase1_dir=phase1_dir,
                atlas_summary_path=atlas_summary_path,
                tri_summary_path=tri_summary_path,
                validation_dir=validation_dir,
                args=args,
            )
            commands.append(command_record("child_routing_validation", validation_cmd))
            phase_print(
                "Stage 3/3: child-routing phase validation",
                str(validation_dir),
                quiet=args.quiet,
            )
            flush_output()
            run_subprocess(validation_cmd, cwd=root)
            validation_summary_path = latest_child_file(validation_dir, "summary.json")

    generated_at = datetime.now().isoformat(timespec="seconds")
    summary_path = run_dir / "summary.json"
    report_path = run_dir / "report.md"
    manifest_path = run_dir / "manifest.json"
    summary_payload = {
        "generated_at": generated_at,
        "run_slug": run_slug,
        "dry_run": bool(args.dry_run),
        "skip_validation": bool(args.skip_validation),
        "selection": {
            "phase1_dir": str(phase1_dir),
            "band_starts": band_starts,
            "window_count": args.window_count,
            "window_step_bits": args.window_step_bits,
            "probe_backward_bits": args.probe_backward_bits,
            "probe_forward_bits": args.probe_forward_bits,
            "scan_step_bits": args.scan_step_bits,
            "top_patterns": args.top_patterns,
            "tolerance_bits": args.tolerance_bits,
            "validation_confidences": args.validation_confidences,
            "validation_lag_policy": args.validation_lag_policy,
            "stage": args.stage,
        },
        "outputs": {
            "atlas_dir": str(atlas_dir),
            "tri_estimator_dir": str(tri_dir),
            "validation_dir": str(validation_dir) if not args.skip_validation else None,
            "atlas_summary": str(atlas_summary_path) if atlas_summary_path else None,
            "tri_summary": str(tri_summary_path) if tri_summary_path else None,
            "validation_summary": (
                str(validation_summary_path) if validation_summary_path else None
            ),
        },
        "quick_reading": quick_reading(tri_summary_path, validation_summary_path),
    }
    manifest_payload = {
        "generated_at": generated_at,
        "script": "hsi_v2_phase2_child_routing_boundary_sweep.py",
        "cwd": str(Path.cwd()),
        "arguments": vars(args),
        "commands": commands,
        "outputs": {
            "summary": str(summary_path),
            "report": str(report_path),
            "manifest": str(manifest_path),
        },
    }

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, indent=2)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(render_report(summary_payload) + "\n")

    if not args.quiet:
        print(render_console_summary(summary_payload))
        print("")
        print(f"Saved summary to: {summary_path}")
        print(f"Saved report to: {report_path}")
        print(f"Saved manifest to: {manifest_path}")

    return 0


def validate_args(args, parser: argparse.ArgumentParser) -> None:
    if args.window_count <= 0:
        parser.error("--window-count must be positive.")
    if args.window_step_bits <= 0:
        parser.error("--window-step-bits must be positive.")
    if args.probe_backward_bits < 0 or args.probe_forward_bits < 0:
        parser.error("--probe-backward-bits and --probe-forward-bits must be non-negative.")
    if args.scan_step_bits <= 0:
        parser.error("--scan-step-bits must be positive.")
    if args.top_patterns <= 0:
        parser.error("--top-patterns must be positive.")
    if args.tolerance_bits < 0:
        parser.error("--tolerance-bits must be non-negative.")
    try:
        band_starts = parse_int_list(args.band_starts, label="--band-starts", allow_zero=True)
    except ValueError as exc:
        parser.error(str(exc))
    if any(value < 0 for value in band_starts):
        parser.error("--band-starts must contain non-negative offsets.")


def build_atlas_command(
    root: Path, phase1_dir: Path, output_dir: Path, args, band_starts: list[int]
) -> list[str]:
    command = [
        sys.executable,
        str(root / "hsi_v2_phase2_phase_lag_atlas.py"),
        "--phase1-dir",
        str(phase1_dir),
        "--output-dir",
        str(output_dir),
        "--band-starts",
        ",".join(str(value) for value in band_starts),
        "--window-count",
        str(args.window_count),
        "--window-step-bits",
        str(args.window_step_bits),
        "--probe-backward-bits",
        str(args.probe_backward_bits),
        "--probe-forward-bits",
        str(args.probe_forward_bits),
        "--scan-step-bits",
        str(args.scan_step_bits),
        "--top-patterns",
        str(args.top_patterns),
    ]
    append_flag(command, "--quiet", args.quiet)
    return command


def build_tri_command(root: Path, input_summary: Path, output_dir: Path, args) -> list[str]:
    command = [
        sys.executable,
        str(root / "hsi_v2_phase2_phase_lag_tri_estimator.py"),
        "--input-summary",
        str(input_summary),
        "--output-dir",
        str(output_dir),
        "--tolerance-bits",
        str(args.tolerance_bits),
    ]
    append_flag(command, "--quiet", args.quiet)
    return command


def build_validation_command(
    *,
    root: Path,
    phase1_dir: Path,
    atlas_summary_path: Path,
    tri_summary_path: Path,
    validation_dir: Path,
    args,
) -> list[str]:
    command = [
        sys.executable,
        str(root / "hsi_v2_phase2_child_routing_phase_validation.py"),
        "--phase1-dir",
        str(phase1_dir),
        "--output-dir",
        str(validation_dir),
        "--atlas-summary",
        str(atlas_summary_path),
        "--tri-summary",
        str(tri_summary_path),
        "--confidences",
        args.validation_confidences,
        "--lag-policy",
        args.validation_lag_policy,
        "--stage",
        args.stage,
    ]
    append_flag(command, "--quiet", args.quiet)
    return command


def quick_reading(tri_summary_path: Path | None, validation_summary_path: Path | None) -> dict:
    reading: dict = {}
    if tri_summary_path and tri_summary_path.exists():
        tri_payload = load_json(tri_summary_path)
        reading["tri_counts"] = tri_payload.get("counts", {})
    if validation_summary_path and validation_summary_path.exists():
        validation_payload = load_json(validation_summary_path)
        reading["validated_targets"] = len(validation_payload.get("targets", []))
        reading["validated_sources"] = len(validation_payload.get("source_validation", []))
    return reading


def render_report(payload: dict) -> str:
    selection = payload["selection"]
    outputs = payload["outputs"]
    lines = [
        "# Phase 2 N2-10d Child-Routing Boundary Sweep",
        "",
        "## Scope",
        "",
        "- Fine boundary sweep around the phase-local child-routing plateau.",
        "- Delegates computation to N2-10, N2-10b, and N2-10c.",
        "- Intended to map plateau edges, not to claim a wave law.",
        "",
        "## Selection",
        "",
        f"- Stage: {selection['stage']}",
        f"- Top patterns: {selection['top_patterns']}",
        f"- Agreement tolerance: {compact_int(selection['tolerance_bits'])}",
        f"- Validation confidences: {selection['validation_confidences']}",
        f"- Validation lag policy: {selection['validation_lag_policy']}",
        f"- Band starts: {', '.join(compact_int(value) for value in selection['band_starts'])}",
        "",
        "## Outputs",
        "",
        f"- Atlas summary: `{outputs['atlas_summary']}`",
        f"- Tri-estimator summary: `{outputs['tri_summary']}`",
        f"- Validation summary: `{outputs['validation_summary']}`",
        "",
        "## Quick Reading",
        "",
        f"```json\n{json.dumps(payload['quick_reading'], indent=2)}\n```",
    ]
    return "\n".join(lines)


def render_console_summary(payload: dict) -> str:
    selection = payload["selection"]
    outputs = payload["outputs"]
    lines = [
        "Phase 2 N2-10d child-routing boundary sweep",
        "-" * 104,
        (
            f"bands={len(selection['band_starts'])} | top={selection['top_patterns']} | "
            f"stage={selection['stage']} | dry_run={payload['dry_run']} | "
            f"skip_validation={payload['skip_validation']} | lag_policy={selection['validation_lag_policy']}"
        ),
        f"atlas_summary={outputs['atlas_summary']}",
        f"tri_summary={outputs['tri_summary']}",
        f"validation_summary={outputs['validation_summary']}",
        f"quick={json.dumps(payload['quick_reading'], sort_keys=True)}",
    ]
    return "\n".join(lines)


def command_record(stage: str, command: list[str]) -> dict:
    return {"stage": stage, "command": command}


def build_run_slug(args, band_starts: list[int], timestamp: str) -> str:
    return (
        "phase2-child-routing-boundary-sweep"
        f"__stage-{args.stage}"
        f"__top-{args.top_patterns}"
        f"__lag-{args.validation_lag_policy}"
        f"__bands-{compact_int(min(band_starts))}-{compact_int(max(band_starts))}x{len(band_starts)}"
        f"__{timestamp}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
