#!/usr/bin/env python3
"""
HSI v2 Phase 2 Parent Survival Revalidation

Canonical reproducible entry point for the parent-shell lag revalidation:

1. Probe the anchor-defined shell lag independently.
2. Re-run the parent-survival band with the externally estimated lag.

This wrapper keeps the scientific pipeline OS-agnostic and reproducible while
preserving the existing public scripts as the audited computational units.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hsi_v2_phase2_transport_defect_strict import parse_int_list, phase_print
from v2.common.cli import parse_variants, resolve_dir
from v2.common.naming import compact_int
from v2.phase2.null_pressure import parse_null_models, parse_seed_list


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full Phase 2 parent-shell revalidation pipeline: "
            "independent lag probe followed by lag-aware parent-survival band."
        )
    )
    parser.add_argument(
        "--phase1-dir",
        type=str,
        default="results/hsi_v2/phase1_high_scales",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/hsi_v2/phase2/parent_survival_revalidation",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="observed",
        choices=("observed", "nulls", "all"),
    )
    parser.add_argument("--variants", type=str, default="E,B")
    parser.add_argument("--anchor-variant", type=str, default="E")
    parser.add_argument("--iteration", type=int, default=None)
    parser.add_argument("--segment-bits", type=int, default=None)
    parser.add_argument("--num-segments", type=int, default=None)
    parser.add_argument("--segment-offset-bits", type=int, default=None)
    parser.add_argument("--scales", type=str, default="")
    parser.add_argument("--phase1-policies", type=str, default="")
    parser.add_argument("--low-scale", type=int, default=40)
    parser.add_argument("--high-scale", type=int, default=48)
    parser.add_argument("--top-patterns", type=int, default=64)
    parser.add_argument(
        "--pattern-selection",
        type=str,
        default="bridge-linked",
        choices=("top", "rare-stable", "bridge-linked"),
    )
    parser.add_argument(
        "--definition-offsets",
        type=str,
        default=(
            "714000000,714500000,715000000,715500000,716000000,716500000,"
            "717000000,717500000,718000000,718500000,719000000,719500000,"
            "720000000,720500000,721000000,721500000,722000000,722500000,"
            "723000000"
        ),
    )
    parser.add_argument(
        "--offsets",
        type=str,
        default=(
            "714000000,714500000,715000000,715500000,716000000,716500000,"
            "717000000,717500000,718000000,718500000,719000000,719500000,"
            "720000000,720500000,721000000,721500000,722000000,722500000,"
            "723000000"
        ),
    )
    parser.add_argument("--window-bits", type=int, default=None)
    parser.add_argument("--scan-start-bits", type=int, default=None)
    parser.add_argument("--scan-stop-bits", type=int, default=None)
    parser.add_argument("--scan-step-bits", type=int, default=500000)
    parser.add_argument("--scan-forward-bits", type=int, default=12000000)
    parser.add_argument("--scan-backward-bits", type=int, default=0)
    parser.add_argument("--null-models", type=str, default="markov1,matched-lz")
    parser.add_argument("--matched-lz-seeds", type=str, default="")
    parser.add_argument("--no-family-inference", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    variants = parse_variants(args.variants)
    if len(variants) != 2:
        parser.error("--variants must contain exactly two observed variants.")
    anchor_variants = parse_variants(args.anchor_variant)
    if len(anchor_variants) != 1:
        parser.error("--anchor-variant must contain exactly one variant.")
    anchor_variant = anchor_variants[0]
    if anchor_variant not in variants:
        parser.error("--anchor-variant must be included in --variants.")
    candidate_variant = next(variant for variant in variants if variant != anchor_variant)

    if args.low_scale <= 0 or args.high_scale <= 0 or args.low_scale >= args.high_scale:
        parser.error("--low-scale and --high-scale must satisfy 1 <= low < high.")
    if args.high_scale > 64:
        parser.error("--high-scale must be in the range 1..64.")
    if args.top_patterns <= 0:
        parser.error("--top-patterns must be positive.")
    if args.scan_step_bits <= 0:
        parser.error("--scan-step-bits must be positive.")
    if args.scan_forward_bits < 0 or args.scan_backward_bits < 0:
        parser.error("--scan-forward-bits and --scan-backward-bits must be non-negative.")

    null_models: list[str] = []
    matched_lz_seeds: list[int] = []
    if args.stage in {"nulls", "all"}:
        try:
            null_models = parse_null_models(args.null_models)
        except ValueError as exc:
            parser.error(str(exc))
        if not null_models:
            parser.error("At least one null model is required for --stage nulls/all.")
        try:
            matched_lz_seeds = parse_seed_list(args.matched_lz_seeds)
        except ValueError:
            parser.error("--matched-lz-seeds must be a comma-separated list of integers.")

    try:
        definition_offsets = parse_int_list(
            args.definition_offsets,
            label="--definition-offsets",
            allow_zero=True,
        )
        offsets = parse_int_list(args.offsets, label="--offsets", allow_zero=True)
    except ValueError as exc:
        parser.error(str(exc))
    if any(value < 0 for value in definition_offsets):
        parser.error("--definition-offsets must contain non-negative integers.")
    if any(value < 0 for value in offsets):
        parser.error("--offsets must contain non-negative integers.")

    script_dir = Path(__file__).parent.resolve()
    phase1_dir = resolve_dir(args.phase1_dir, anchor_file=__file__)
    output_dir = resolve_dir(args.output_dir, anchor_file=__file__)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_slug = build_run_slug(
        anchor_variant=anchor_variant,
        candidate_variant=candidate_variant,
        low_scale=args.low_scale,
        high_scale=args.high_scale,
        pattern_selection=args.pattern_selection,
        top_patterns=args.top_patterns,
        offsets=offsets,
        timestamp=timestamp,
    )
    run_dir = output_dir / run_slug
    probe_output_dir = run_dir / "probe"
    lagaware_output_dir = run_dir / "lagaware"
    run_dir.mkdir(parents=True, exist_ok=True)

    phase_print(
        "Preparing canonical parent-survival revalidation",
        (
            f"stage={args.stage} | anchor={anchor_variant} | candidate={candidate_variant} | "
            f"m={args.low_scale}->{args.high_scale} | sel={args.pattern_selection}"
        ),
        quiet=args.quiet,
    )
    flush_output()

    probe_cmd = [
        sys.executable,
        str(script_dir / "hsi_v2_phase2_parent_shell_lag_probe.py"),
        "--phase1-dir",
        str(phase1_dir),
        "--output-dir",
        str(probe_output_dir),
        "--variants",
        ",".join(variants),
        "--anchor-variant",
        anchor_variant,
        "--scales",
        args.scales,
        "--phase1-policies",
        args.phase1_policies,
        "--pattern-scale",
        str(args.low_scale),
        "--top-patterns",
        str(args.top_patterns),
        "--pattern-selection",
        args.pattern_selection,
        "--definition-offsets",
        ",".join(str(value) for value in definition_offsets),
        "--scan-step-bits",
        str(args.scan_step_bits),
        "--scan-forward-bits",
        str(args.scan_forward_bits),
        "--scan-backward-bits",
        str(args.scan_backward_bits),
    ]
    append_optional_int_arg(probe_cmd, "--iteration", args.iteration)
    append_optional_int_arg(probe_cmd, "--segment-bits", args.segment_bits)
    append_optional_int_arg(probe_cmd, "--num-segments", args.num_segments)
    append_optional_int_arg(probe_cmd, "--segment-offset-bits", args.segment_offset_bits)
    append_optional_int_arg(probe_cmd, "--window-bits", args.window_bits)
    append_optional_int_arg(probe_cmd, "--scan-start-bits", args.scan_start_bits)
    append_optional_int_arg(probe_cmd, "--scan-stop-bits", args.scan_stop_bits)
    append_flag(probe_cmd, "--no-family-inference", args.no_family_inference)
    append_flag(probe_cmd, "--quiet", args.quiet)

    phase_print(
        "Stage 1/2: independent shell-lag probe",
        f"output_dir={probe_output_dir}",
        quiet=args.quiet,
    )
    flush_output()
    run_subprocess(probe_cmd, cwd=script_dir)
    probe_summary_path = latest_child_file(probe_output_dir, "summary.json")
    probe_report_path = latest_child_file(probe_output_dir, "report.md")
    probe_payload = load_json(probe_summary_path)
    recommended_lag_raw = probe_payload.get("summary", {}).get("recommended_lag_bits")
    if recommended_lag_raw is None:
        parser.error(
            "The shell-lag probe did not yield recommended_lag_bits. "
            f"Check the probe artifact: {probe_summary_path}"
        )
    recommended_lag_bits = int(recommended_lag_raw)

    phase_print(
        "Resolved external lag estimate",
        f"recommended_lag_bits={recommended_lag_bits}",
        quiet=args.quiet,
    )
    flush_output()

    lagaware_cmd = [
        sys.executable,
        str(script_dir / "hsi_v2_phase2_parent_survival_band_lagaware.py"),
        "--phase1-dir",
        str(phase1_dir),
        "--output-dir",
        str(lagaware_output_dir),
        "--stage",
        args.stage,
        "--variants",
        ",".join(variants),
        "--anchor-variant",
        anchor_variant,
        "--scales",
        args.scales,
        "--phase1-policies",
        args.phase1_policies,
        "--low-scale",
        str(args.low_scale),
        "--high-scale",
        str(args.high_scale),
        "--top-patterns",
        str(args.top_patterns),
        "--pattern-selection",
        args.pattern_selection,
        "--offsets",
        ",".join(str(value) for value in offsets),
        "--lag-summary",
        str(probe_summary_path),
    ]
    if null_models:
        lagaware_cmd.extend(["--null-models", ",".join(null_models)])
    if matched_lz_seeds:
        lagaware_cmd.extend(
            ["--matched-lz-seeds", ",".join(str(seed) for seed in matched_lz_seeds)]
        )
    append_optional_int_arg(lagaware_cmd, "--iteration", args.iteration)
    append_optional_int_arg(lagaware_cmd, "--segment-bits", args.segment_bits)
    append_optional_int_arg(lagaware_cmd, "--num-segments", args.num_segments)
    append_optional_int_arg(lagaware_cmd, "--segment-offset-bits", args.segment_offset_bits)
    append_optional_int_arg(lagaware_cmd, "--window-bits", args.window_bits)
    append_flag(lagaware_cmd, "--no-family-inference", args.no_family_inference)
    append_flag(lagaware_cmd, "--quiet", args.quiet)

    phase_print(
        "Stage 2/2: lag-aware parent-survival band",
        f"output_dir={lagaware_output_dir}",
        quiet=args.quiet,
    )
    flush_output()
    run_subprocess(lagaware_cmd, cwd=script_dir)
    lagaware_summary_path = latest_child_file(lagaware_output_dir, "summary.json")
    lagaware_report_path = latest_child_file(lagaware_output_dir, "report.md")
    lagaware_payload = load_json(lagaware_summary_path)

    generated_at = datetime.now().isoformat(timespec="seconds")
    summary_payload = {
        "generated_at": generated_at,
        "selection": {
            "stage": args.stage,
            "anchor_variant": anchor_variant,
            "candidate_variant": candidate_variant,
            "variants": variants,
            "phase1_dir": str(phase1_dir),
            "low_scale": args.low_scale,
            "high_scale": args.high_scale,
            "top_patterns": args.top_patterns,
            "pattern_selection": args.pattern_selection,
            "definition_offsets": definition_offsets,
            "offsets": offsets,
            "null_models": null_models,
            "matched_lz_seeds": matched_lz_seeds,
        },
        "probe_summary_path": str(probe_summary_path),
        "lagaware_summary_path": str(lagaware_summary_path),
        "recommended_lag_bits": recommended_lag_bits,
        "probe_summary": probe_payload.get("summary", {}),
        "lagaware_grouped_summary": lagaware_payload.get("grouped_summary", []),
    }
    manifest_payload = {
        "generated_at": generated_at,
        "run_slug": run_slug,
        "script": "hsi_v2_phase2_parent_survival_revalidation.py",
        "cwd": str(Path.cwd()),
        "outputs": {
            "summary": str(run_dir / "summary.json"),
            "report": str(run_dir / "report.md"),
            "manifest": str(run_dir / "manifest.json"),
            "probe_summary": str(probe_summary_path),
            "probe_report": str(probe_report_path),
            "lagaware_summary": str(lagaware_summary_path),
            "lagaware_report": str(lagaware_report_path),
        },
        "commands": {
            "probe": probe_cmd,
            "lagaware": lagaware_cmd,
        },
        "arguments": vars(args),
    }

    summary_path = run_dir / "summary.json"
    report_path = run_dir / "report.md"
    manifest_path = run_dir / "manifest.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(render_report(summary_payload) + "\n")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, indent=2)

    if not args.quiet:
        print("Phase 2 canonical parent-survival revalidation")
        print("-" * 118)
        print(
            f"stage={args.stage} anchor={anchor_variant} candidate={candidate_variant} "
            f"lag={recommended_lag_bits} "
            f"probe={probe_summary_path.parent.name} "
            f"lagaware={lagaware_summary_path.parent.name}"
        )
        print("")
        print(f"Saved pipeline summary to: {summary_path}")
        print(f"Saved pipeline report to: {report_path}")
        print(f"Saved pipeline manifest to: {manifest_path}")

    return 0


def append_optional_int_arg(command: list[str], flag: str, value: int | None) -> None:
    if value is None:
        return
    command.extend([flag, str(int(value))])


def append_flag(command: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        command.append(flag)


def flush_output() -> None:
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except OSError:
        return


def run_subprocess(command: list[str], *, cwd: Path) -> None:
    subprocess.run(command, cwd=str(cwd), check=True)


def latest_child_file(root_dir: Path, filename: str) -> Path:
    candidates = [
        child / filename
        for child in root_dir.iterdir()
        if child.is_dir() and (child / filename).is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"No child artifact {filename!r} found under {root_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def render_report(payload: dict) -> str:
    selection = payload["selection"]
    probe_summary = payload["probe_summary"]
    lagaware_items = payload["lagaware_grouped_summary"]

    lines = [
        "# Phase 2 Canonical Parent-Survival Revalidation",
        "",
        "## Selection",
        "",
        f"- Anchor variant: {selection['anchor_variant']}",
        f"- Candidate variant: {selection['candidate_variant']}",
        f"- Stage: {selection['stage']}",
        f"- Low/high scales: {selection['low_scale']} -> {selection['high_scale']}",
        f"- Pattern selection: {selection['pattern_selection']}",
        f"- Top patterns: {selection['top_patterns']}",
        f"- Definition offsets: {', '.join(str(value) for value in selection['definition_offsets'])}",
        f"- Band offsets: {', '.join(str(value) for value in selection['offsets'])}",
        f"- Null models: {', '.join(selection.get('null_models', [])) or '-'}",
        f"- Matched-LZ seeds: {', '.join(str(value) for value in selection.get('matched_lz_seeds', [])) or '-'}",
        "",
        "## Pipeline Outcome",
        "",
        f"- Recommended lag bits: {payload['recommended_lag_bits']}",
        f"- Probe summary path: {payload['probe_summary_path']}",
        f"- Lag-aware summary path: {payload['lagaware_summary_path']}",
        "",
        "## Lag Probe",
        "",
        f"- Anchor core size: {probe_summary.get('anchor_core_pattern_count')}",
        f"- Anchor shell size: {probe_summary.get('anchor_shell_pattern_count')}",
        f"- First full candidate shell offset: {probe_summary.get('first_candidate_full_shell_offset_bits')}",
        f"- First-full lag estimate: {probe_summary.get('first_full_shell_lag_bits')}",
        f"- Hit-profile best shift: {probe_summary.get('shell_hits_correlation', {}).get('best_shift_bits')}",
        f"- Unique-profile best shift: {probe_summary.get('shell_unique_correlation', {}).get('best_shift_bits')}",
        "",
        "## Lag-Aware Band Summary",
        "",
        "| Variant | Source | Exact rng | Dead rng | Exact m rng | Dead m rng | Ex IDs | Dead IDs |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for item in lagaware_items:
        lines.append(
            f"| {item['variant']} | {item['source_label']} | "
            f"{item['exact_parent_count_min']}-{item['exact_parent_count_max']} | "
            f"{item['dead_parent_count_min']}-{item['dead_parent_count_max']} | "
            f"{item['exact_mass_fraction_min']:.4f}-{item['exact_mass_fraction_max']:.4f} | "
            f"{item['dead_mass_fraction_min']:.4f}-{item['dead_mass_fraction_max']:.4f} | "
            f"{'yes' if item['exact_pattern_identity_stable'] else 'no'} | "
            f"{'yes' if item['dead_pattern_identity_stable'] else 'no'} |"
        )
    return "\n".join(lines)


def build_run_slug(
    *,
    anchor_variant: str,
    candidate_variant: str,
    low_scale: int,
    high_scale: int,
    pattern_selection: str,
    top_patterns: int,
    offsets: list[int],
    timestamp: str,
) -> str:
    if len(offsets) == 1:
        offset_part = f"off-{compact_int(offsets[0])}"
    else:
        offset_part = f"off-{compact_int(min(offsets))}-plus-{len(offsets)}"
    return (
        f"phase2-parent-survival-revalidation__anchor-{anchor_variant}"
        f"__cand-{candidate_variant}__m-{low_scale}-{high_scale}"
        f"__sel-{pattern_selection}__top-{top_patterns}"
        f"__{offset_part}__{timestamp}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
