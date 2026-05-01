#!/usr/bin/env python3
"""
HSI v2 Phase 2 Child-Routing Persistence

N2-09 reproducible wrapper. It checks whether the corrected N2-08
child-destination conservation readout persists across neighboring offset
bands without changing the underlying audited computational units.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hsi_v2_phase2_parent_shell_atlas_pipeline import CANONICAL_ALL_PROFILE
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


DEFAULT_OUTPUT_DIR = "results/hsi_v2/phase2/child_routing_persistence"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the N2-09 child-routing persistence scan across offset bands. "
            "Each band runs shell-lag probe, lag-aware band, shell atlas, and "
            "corrected N2-08 child-destination routing."
        )
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="canonical-neighborhood",
        choices=("canonical-neighborhood",),
        help="Named reproducible persistence profile.",
    )
    parser.add_argument(
        "--phase1-dir",
        type=str,
        default="results/hsi_v2/phase1_high_scales",
    )
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=("observed", "nulls", "all"),
    )
    parser.add_argument(
        "--band-starts",
        type=str,
        default="705000000,714000000,723000000",
        help=(
            "Comma-separated anchor band starts. With the default window-count "
            "and step this scans 705M-714M, 714M-723M, and 723M-732M."
        ),
    )
    parser.add_argument("--window-count", type=int, default=19)
    parser.add_argument("--window-step-bits", type=int, default=500000)
    parser.add_argument(
        "--scan-span-bits",
        type=int,
        default=18000000,
        help="Shell-lag probe scan span from each band start.",
    )
    parser.add_argument(
        "--probe-scan-start-bits",
        type=int,
        default=None,
        help=(
            "Optional absolute shell-lag probe scan start. Use this for "
            "boundary checks where the candidate phase may precede the "
            "anchor band."
        ),
    )
    parser.add_argument(
        "--probe-scan-stop-bits",
        type=int,
        default=None,
        help="Optional absolute shell-lag probe scan stop.",
    )
    parser.add_argument(
        "--lag-mode",
        type=str,
        default="fixed",
        choices=("fixed", "probe"),
        help=(
            "fixed uses --candidate-lag-bits for every band; probe uses the "
            "band-local shell-lag estimate."
        ),
    )
    parser.add_argument("--candidate-lag-bits", type=int, default=15000000)
    parser.add_argument("--top-patterns", type=int, default=128)
    parser.add_argument(
        "--null-models",
        type=str,
        default="",
        help="Optional override over the canonical null models.",
    )
    parser.add_argument(
        "--matched-lz-seeds",
        type=str,
        default="",
        help="Optional override over the canonical matched-LZ seed list.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.window_count <= 0:
        parser.error("--window-count must be positive.")
    if args.window_step_bits <= 0:
        parser.error("--window-step-bits must be positive.")
    if args.scan_span_bits <= 0:
        parser.error("--scan-span-bits must be positive.")
    if args.probe_scan_start_bits is not None and args.probe_scan_start_bits < 0:
        parser.error("--probe-scan-start-bits must be non-negative.")
    if args.probe_scan_stop_bits is not None and args.probe_scan_stop_bits < 0:
        parser.error("--probe-scan-stop-bits must be non-negative.")
    if (
        args.probe_scan_start_bits is not None
        and args.probe_scan_stop_bits is not None
        and args.probe_scan_stop_bits < args.probe_scan_start_bits
    ):
        parser.error("--probe-scan-stop-bits must be >= --probe-scan-start-bits.")
    if args.top_patterns <= 0:
        parser.error("--top-patterns must be positive.")

    try:
        band_starts = parse_int_list(
            args.band_starts,
            label="--band-starts",
            allow_zero=True,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if any(value < 0 for value in band_starts):
        parser.error("--band-starts must contain non-negative offsets.")

    selection = build_selection(args)
    script_dir = Path(__file__).parent.resolve()
    phase1_dir = resolve_dir(args.phase1_dir, anchor_file=__file__)
    output_dir = resolve_dir(args.output_dir, anchor_file=__file__)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_slug = build_run_slug(selection, args, band_starts, timestamp)
    run_dir = output_dir / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)

    phase_print(
        "Preparing N2-09 child-routing persistence",
        (
            f"stage={selection['stage']} | anchor={selection['anchor_variant']} | "
            f"candidate={selection['candidate_variant']} | top={args.top_patterns} | "
            f"lag_mode={args.lag_mode}"
        ),
        quiet=args.quiet,
    )
    flush_output()

    commands: list[dict] = []
    band_rows: list[dict] = []
    source_rows: list[dict] = []

    for band_index, band_start in enumerate(band_starts, start=1):
        band_offsets = build_offsets(
            start_bits=band_start,
            count=args.window_count,
            step_bits=args.window_step_bits,
        )
        band_end = band_offsets[-1]
        band_label = f"band-{compact_int(band_start)}-{compact_int(band_end)}"
        band_root = run_dir / band_label
        probe_root = band_root / "probe"
        lagaware_root = band_root / "lagaware"
        atlas_root = band_root / "atlas"
        routing_root = band_root / "child_routing"
        probe_scan_start_bits = (
            args.probe_scan_start_bits
            if args.probe_scan_start_bits is not None
            else band_start
        )
        probe_scan_stop_bits = (
            args.probe_scan_stop_bits
            if args.probe_scan_stop_bits is not None
            else band_start + args.scan_span_bits
        )
        if probe_scan_stop_bits < probe_scan_start_bits:
            parser.error(
                "The resolved probe scan stop must be >= the resolved probe scan start."
            )

        phase_print(
            "Persistence band",
            (
                f"{band_index}/{len(band_starts)} | {band_label} | "
                f"offsets={len(band_offsets)}"
            ),
            quiet=args.quiet,
        )
        flush_output()

        probe_cmd = build_probe_cmd(
            script_dir=script_dir,
            phase1_dir=phase1_dir,
            output_dir=probe_root,
            selection=selection,
            top_patterns=args.top_patterns,
            definition_offsets=band_offsets,
            scan_start_bits=probe_scan_start_bits,
            scan_stop_bits=probe_scan_stop_bits,
            scan_step_bits=args.window_step_bits,
        )
        append_flag(probe_cmd, "--quiet", args.quiet)
        phase_print("Stage 1/4: shell-lag probe", band_label, quiet=args.quiet)
        flush_output()
        run_subprocess(probe_cmd, cwd=script_dir)
        probe_summary_path = latest_child_file(probe_root, "summary.json")
        probe_payload = load_json(probe_summary_path)
        probe_lag = probe_payload.get("summary", {}).get("recommended_lag_bits")
        if probe_lag is None and args.lag_mode == "probe":
            phase_print(
                "Skipping band: no probe-derived lag",
                f"{band_label} | probe={probe_summary_path}",
                quiet=args.quiet,
            )
            flush_output()
            commands.append(command_record("probe", band_label, probe_cmd))
            band_rows.append(
                build_band_common(
                    band_index=band_index,
                    band_label=band_label,
                    band_start=band_start,
                    band_end=band_end,
                    selected_lag=None,
                    probe_lag=None,
                    probe_summary_path=probe_summary_path,
                    lagaware_summary_path=None,
                    atlas_summary_path=None,
                    routing_summary_path=None,
                    probe_scan_start_bits=probe_scan_start_bits,
                    probe_scan_stop_bits=probe_scan_stop_bits,
                    band_status="no-probe-lag",
                    skip_reason="shell-lag probe did not emit recommended_lag_bits",
                )
            )
            continue
        probe_lag = int(probe_lag) if probe_lag is not None else None
        selected_lag = int(args.candidate_lag_bits if args.lag_mode == "fixed" else probe_lag)
        min_candidate_offset = min(offset + selected_lag for offset in band_offsets)
        if min_candidate_offset < 0:
            parser.error(
                "The selected candidate lag places at least one candidate window before bit 0."
            )

        lagaware_cmd = build_lagaware_cmd(
            script_dir=script_dir,
            phase1_dir=phase1_dir,
            output_dir=lagaware_root,
            selection=selection,
            top_patterns=args.top_patterns,
            offsets=band_offsets,
            lag_bits=selected_lag,
        )
        append_flag(lagaware_cmd, "--quiet", args.quiet)
        phase_print(
            "Stage 2/4: lag-aware parent-survival band",
            f"{band_label} | lag={selected_lag}",
            quiet=args.quiet,
        )
        flush_output()
        run_subprocess(lagaware_cmd, cwd=script_dir)
        lagaware_summary_path = latest_child_file(lagaware_root, "summary.json")

        synthetic_revalidation_path = write_synthetic_revalidation_summary(
            band_root=band_root,
            selection=selection,
            phase1_dir=phase1_dir,
            top_patterns=args.top_patterns,
            offsets=band_offsets,
            probe_summary_path=probe_summary_path,
            lagaware_summary_path=lagaware_summary_path,
            probe_lag_bits=probe_lag,
            selected_lag_bits=selected_lag,
            lag_mode=args.lag_mode,
        )

        atlas_cmd = [
            sys.executable,
            str(script_dir / "hsi_v2_phase2_parent_shell_atlas.py"),
            "--revalidation-run",
            str(synthetic_revalidation_path),
            "--output-dir",
            str(atlas_root),
        ]
        append_flag(atlas_cmd, "--quiet", args.quiet)
        phase_print("Stage 3/4: parent shell atlas", band_label, quiet=args.quiet)
        flush_output()
        run_subprocess(atlas_cmd, cwd=script_dir)
        atlas_summary_path = latest_child_file(atlas_root, "summary.json")
        atlas_dataset_path = atlas_summary_path.with_name("dataset.json")

        routing_cmd = [
            sys.executable,
            str(script_dir / "hsi_v2_phase2_child_destination_routing.py"),
            "--atlas-run",
            str(atlas_dataset_path),
            "--output-dir",
            str(routing_root),
        ]
        append_flag(routing_cmd, "--quiet", args.quiet)
        phase_print("Stage 4/4: child-destination routing", band_label, quiet=args.quiet)
        flush_output()
        run_subprocess(routing_cmd, cwd=script_dir)
        routing_summary_path = latest_child_file(routing_root, "summary.json")
        routing_payload = load_json(routing_summary_path)

        commands.extend(
            [
                command_record("probe", band_label, probe_cmd),
                command_record("lagaware", band_label, lagaware_cmd),
                command_record("atlas", band_label, atlas_cmd),
                command_record("child_routing", band_label, routing_cmd),
            ]
        )
        collect_rows(
            band_rows,
            source_rows,
            band_index=band_index,
            band_label=band_label,
            band_start=band_start,
            band_end=band_end,
            selected_lag=selected_lag,
            probe_lag=probe_lag,
            probe_summary_path=probe_summary_path,
            lagaware_summary_path=lagaware_summary_path,
            atlas_summary_path=atlas_summary_path,
            routing_summary_path=routing_summary_path,
            probe_scan_start_bits=probe_scan_start_bits,
            probe_scan_stop_bits=probe_scan_stop_bits,
            routing_payload=routing_payload,
        )

    generated_at = datetime.now().isoformat(timespec="seconds")
    summary_path = run_dir / "summary.json"
    report_path = run_dir / "report.md"
    source_csv_path = run_dir / "source_persistence.csv"
    band_csv_path = run_dir / "band_persistence.csv"
    manifest_path = run_dir / "manifest.json"

    summary_payload = {
        "generated_at": generated_at,
        "selection": {
            "profile": args.profile,
            "stage": selection["stage"],
            "anchor_variant": selection["anchor_variant"],
            "candidate_variant": selection["candidate_variant"],
            "variants": selection["variants_list"],
            "phase1_dir": str(phase1_dir),
            "top_patterns": args.top_patterns,
            "lag_mode": args.lag_mode,
            "candidate_lag_bits": args.candidate_lag_bits,
            "band_starts": band_starts,
            "window_count": args.window_count,
            "window_step_bits": args.window_step_bits,
            "scan_span_bits": args.scan_span_bits,
            "probe_scan_start_bits": args.probe_scan_start_bits,
            "probe_scan_stop_bits": args.probe_scan_stop_bits,
            "null_models": selection["null_models"].split(",") if selection["null_models"] else [],
            "matched_lz_seeds": [
                int(item) for item in selection["matched_lz_seeds"].split(",") if item.strip()
            ],
        },
        "band_summary": band_rows,
        "source_persistence": source_rows,
    }
    manifest_payload = {
        "generated_at": generated_at,
        "run_slug": run_slug,
        "script": "hsi_v2_phase2_child_routing_persistence.py",
        "cwd": str(Path.cwd()),
        "outputs": {
            "summary": str(summary_path),
            "report": str(report_path),
            "source_csv": str(source_csv_path),
            "band_csv": str(band_csv_path),
            "manifest": str(manifest_path),
        },
        "commands": commands,
        "arguments": vars(args),
    }

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(render_report(summary_payload) + "\n")
    write_csv(source_rows, source_csv_path)
    write_csv(band_rows, band_csv_path)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, indent=2)

    if not args.quiet:
        print(render_console_summary(source_rows))
        print("")
        print(f"Saved summary to: {summary_path}")
        print(f"Saved report to: {report_path}")
        print(f"Saved source CSV to: {source_csv_path}")
        print(f"Saved band CSV to: {band_csv_path}")
        print(f"Saved manifest to: {manifest_path}")

    return 0


def build_selection(args) -> dict:
    selection = dict(CANONICAL_ALL_PROFILE)
    selection["profile"] = args.profile
    selection["stage"] = args.stage
    selection["top_patterns"] = args.top_patterns
    if args.null_models:
        selection["null_models"] = args.null_models
    if args.matched_lz_seeds:
        selection["matched_lz_seeds"] = args.matched_lz_seeds
    if selection["stage"] == "observed":
        selection["null_models"] = ""
        selection["matched_lz_seeds"] = ""
    elif "matched-lz" not in [
        item.strip() for item in selection["null_models"].split(",") if item.strip()
    ]:
        selection["matched_lz_seeds"] = ""

    variants_list = [item.strip() for item in selection["variants"].split(",") if item.strip()]
    if len(variants_list) != 2:
        raise ValueError("The persistence profile expects exactly two variants.")
    selection["variants_list"] = variants_list
    selection["candidate_variant"] = next(
        variant for variant in variants_list if variant != selection["anchor_variant"]
    )
    return selection


def build_offsets(*, start_bits: int, count: int, step_bits: int) -> list[int]:
    return [int(start_bits + index * step_bits) for index in range(count)]


def build_probe_cmd(
    *,
    script_dir: Path,
    phase1_dir: Path,
    output_dir: Path,
    selection: dict,
    top_patterns: int,
    definition_offsets: list[int],
    scan_start_bits: int,
    scan_stop_bits: int,
    scan_step_bits: int,
) -> list[str]:
    return [
        sys.executable,
        str(script_dir / "hsi_v2_phase2_parent_shell_lag_probe.py"),
        "--phase1-dir",
        str(phase1_dir),
        "--output-dir",
        str(output_dir),
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
        "--pattern-scale",
        str(selection["low_scale"]),
        "--top-patterns",
        str(top_patterns),
        "--pattern-selection",
        selection["pattern_selection"],
        "--definition-offsets",
        ",".join(str(value) for value in definition_offsets),
        "--scan-start-bits",
        str(scan_start_bits),
        "--scan-stop-bits",
        str(scan_stop_bits),
        "--scan-step-bits",
        str(scan_step_bits),
        "--scan-forward-bits",
        str(selection["scan_forward_bits"]),
        "--scan-backward-bits",
        str(selection["scan_backward_bits"]),
    ]


def build_lagaware_cmd(
    *,
    script_dir: Path,
    phase1_dir: Path,
    output_dir: Path,
    selection: dict,
    top_patterns: int,
    offsets: list[int],
    lag_bits: int,
) -> list[str]:
    command = [
        sys.executable,
        str(script_dir / "hsi_v2_phase2_parent_survival_band_lagaware.py"),
        "--phase1-dir",
        str(phase1_dir),
        "--output-dir",
        str(output_dir),
        "--stage",
        selection["stage"],
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
        str(top_patterns),
        "--pattern-selection",
        selection["pattern_selection"],
        "--offsets",
        ",".join(str(value) for value in offsets),
        "--candidate-lag-bits",
        str(lag_bits),
    ]
    if selection["stage"] in {"nulls", "all"}:
        command.extend(["--null-models", selection["null_models"]])
        if selection["matched_lz_seeds"]:
            command.extend(["--matched-lz-seeds", selection["matched_lz_seeds"]])
    return command


def write_synthetic_revalidation_summary(
    *,
    band_root: Path,
    selection: dict,
    phase1_dir: Path,
    top_patterns: int,
    offsets: list[int],
    probe_summary_path: Path,
    lagaware_summary_path: Path,
    probe_lag_bits: int | None,
    selected_lag_bits: int,
    lag_mode: str,
) -> Path:
    path = band_root / "synthetic_revalidation_summary.json"
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "selection": {
            "profile": selection["profile"],
            "stage": selection["stage"],
            "anchor_variant": selection["anchor_variant"],
            "candidate_variant": selection["candidate_variant"],
            "variants": selection["variants_list"],
            "phase1_dir": str(phase1_dir),
            "top_patterns": top_patterns,
            "offsets": offsets,
            "candidate_lag_bits": selected_lag_bits,
            "probe_recommended_lag_bits": probe_lag_bits,
            "lag_mode": lag_mode,
        },
        "probe_summary_path": str(probe_summary_path),
        "lagaware_summary_path": str(lagaware_summary_path),
        "recommended_lag_bits": selected_lag_bits,
        "notes": [
            "Synthetic revalidation pointer generated by N2-09 persistence wrapper.",
            "The lag-aware dataset is built per band with either fixed or probe-derived lag.",
        ],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def collect_rows(
    band_rows: list[dict],
    source_rows: list[dict],
    *,
    band_index: int,
    band_label: str,
    band_start: int,
    band_end: int,
    selected_lag: int,
    probe_lag: int | None,
    probe_summary_path: Path,
    lagaware_summary_path: Path,
    atlas_summary_path: Path,
    routing_summary_path: Path,
    probe_scan_start_bits: int,
    probe_scan_stop_bits: int,
    routing_payload: dict,
) -> None:
    common = build_band_common(
        band_index=band_index,
        band_label=band_label,
        band_start=band_start,
        band_end=band_end,
        selected_lag=selected_lag,
        probe_lag=probe_lag,
        probe_summary_path=probe_summary_path,
        lagaware_summary_path=lagaware_summary_path,
        atlas_summary_path=atlas_summary_path,
        routing_summary_path=routing_summary_path,
        probe_scan_start_bits=probe_scan_start_bits,
        probe_scan_stop_bits=probe_scan_stop_bits,
        band_status="completed",
        skip_reason="",
    )
    band_rows.append(common)
    for row in routing_payload.get("source_summary", []):
        source_rows.append(
            {
                **common,
                "variant": row["variant"],
                "source_label": row["source_label"],
                "source_kind": row["source_kind"],
                "null_model": row.get("null_model"),
                "null_seed": row.get("null_seed"),
                "pattern_count": row.get("pattern_count"),
                "anchor_event_total_sum": row.get("anchor_event_total_sum"),
                "candidate_event_total_sum": row.get("candidate_event_total_sum"),
                "child_destination_retention_pooled": row.get(
                    "child_destination_retention_pooled"
                ),
                "window_synchronous_child_destination_retention_pooled": row.get(
                    "window_synchronous_child_destination_retention_pooled"
                ),
                "monitored_candidate_reroute_share_pooled": row.get(
                    "monitored_candidate_reroute_share_pooled"
                ),
                "anchor_child_deficit_mass_sum": row.get("anchor_child_deficit_mass_sum"),
                "top_child_full_match_mass_fraction": row.get(
                    "top_child_full_match_mass_fraction"
                ),
                "dominant_routing_class": row.get("dominant_routing_class"),
                "preview_truncated": row.get("preview_truncated"),
            }
        )


def build_band_common(
    *,
    band_index: int,
    band_label: str,
    band_start: int,
    band_end: int,
    selected_lag: int | None,
    probe_lag: int | None,
    probe_summary_path: Path | None,
    lagaware_summary_path: Path | None,
    atlas_summary_path: Path | None,
    routing_summary_path: Path | None,
    probe_scan_start_bits: int,
    probe_scan_stop_bits: int,
    band_status: str,
    skip_reason: str,
) -> dict:
    return {
        "band_index": band_index,
        "band_label": band_label,
        "band_start_bits": band_start,
        "band_end_bits": band_end,
        "selected_lag_bits": selected_lag,
        "probe_lag_bits": probe_lag,
        "probe_scan_start_bits": probe_scan_start_bits,
        "probe_scan_stop_bits": probe_scan_stop_bits,
        "band_status": band_status,
        "skip_reason": skip_reason,
        "probe_summary_path": path_text(probe_summary_path),
        "lagaware_summary_path": path_text(lagaware_summary_path),
        "atlas_summary_path": path_text(atlas_summary_path),
        "routing_summary_path": path_text(routing_summary_path),
    }


def path_text(path: Path | None) -> str:
    return str(path) if path is not None else ""


def command_record(stage: str, band_label: str, command: list[str]) -> dict:
    return {
        "stage": stage,
        "band_label": band_label,
        "command": command,
    }


def render_console_summary(rows: list[dict]) -> str:
    lines = [
        "Phase 2 child-routing persistence",
        "-" * 132,
        (
            f"{'band':<18} {'variant':<12} {'source':<28} {'lag':>8} "
            f"{'childRet':>9} {'syncRet':>9} {'monRer':>9} {'deficit':>10} {'class':>28}"
        ),
    ]
    for row in rows:
        lines.append(
            f"{row['band_label']:<18} "
            f"{truncate(row['variant'], 12):<12} "
            f"{truncate(row['source_label'], 28):<28} "
            f"{compact_int(int(row['selected_lag_bits'])):>8} "
            f"{fmt(row['child_destination_retention_pooled']):>9} "
            f"{fmt(row['window_synchronous_child_destination_retention_pooled']):>9} "
            f"{fmt(row['monitored_candidate_reroute_share_pooled']):>9} "
            f"{fmt_num(row['anchor_child_deficit_mass_sum']):>10} "
            f"{truncate(row['dominant_routing_class'], 28):>28}"
        )
    return "\n".join(lines)


def render_report(payload: dict) -> str:
    selection = payload["selection"]
    lines = [
        "# Phase 2 Child-Routing Persistence",
        "",
        "## Selection",
        "",
        f"- Profile: {selection['profile']}",
        f"- Stage: {selection['stage']}",
        f"- Anchor variant: {selection['anchor_variant']}",
        f"- Candidate variant: {selection['candidate_variant']}",
        f"- Top patterns: {selection['top_patterns']}",
        f"- Lag mode: {selection['lag_mode']}",
        f"- Candidate lag bits: {selection['candidate_lag_bits']}",
        f"- Band starts: {', '.join(str(value) for value in selection['band_starts'])}",
        f"- Window count: {selection['window_count']}",
        f"- Window step bits: {selection['window_step_bits']}",
        f"- Probe scan start bits override: {selection['probe_scan_start_bits']}",
        f"- Probe scan stop bits override: {selection['probe_scan_stop_bits']}",
        "",
        "## Band Summary",
        "",
        "| Band | Probe lag | Selected lag | Status | Skip reason |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for row in payload["band_summary"]:
        lines.append(
            f"| {row['band_label']} | {fmt_optional_int(row['probe_lag_bits'])} | "
            f"{fmt_optional_int(row['selected_lag_bits'])} | {row['band_status']} | "
            f"{row['skip_reason'] or '-'} |"
        )
    lines.extend(
        [
            "",
        "## Source Persistence",
        "",
        "| Band | Lag | Variant | Source | Child retention | Sync retention | Monitored reroute | Deficit | Class |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    if not payload["source_persistence"]:
        lines.append("")
        lines.append("_No source persistence rows were generated._")
        return "\n".join(lines)
    for row in payload["source_persistence"]:
        lines.append(
            f"| {row['band_label']} | {row['selected_lag_bits']} | {row['variant']} | "
            f"{row['source_label']} | {fmt(row['child_destination_retention_pooled'])} | "
            f"{fmt(row['window_synchronous_child_destination_retention_pooled'])} | "
            f"{fmt(row['monitored_candidate_reroute_share_pooled'])} | "
            f"{fmt_num(row['anchor_child_deficit_mass_sum'])} | "
            f"{row['dominant_routing_class']} |"
        )
    return "\n".join(lines)


def build_run_slug(
    selection: dict,
    args,
    band_starts: list[int],
    timestamp: str,
) -> str:
    if args.lag_mode == "fixed":
        lag_part = f"lag-fixed-{compact_int(args.candidate_lag_bits)}"
    else:
        lag_part = "lag-probe"
    if args.probe_scan_start_bits is not None or args.probe_scan_stop_bits is not None:
        scan_start = (
            compact_int(args.probe_scan_start_bits)
            if args.probe_scan_start_bits is not None
            else "auto"
        )
        scan_stop = (
            compact_int(args.probe_scan_stop_bits)
            if args.probe_scan_stop_bits is not None
            else "auto"
        )
        lag_part = f"{lag_part}__scan-{scan_start}-{scan_stop}"
    return (
        f"phase2-child-routing-persistence__stage-{selection['stage']}"
        f"__anchor-{selection['anchor_variant']}__cand-{selection['candidate_variant']}"
        f"__top-{args.top_patterns}__{lag_part}"
        f"__starts-{compact_int(min(band_starts))}-{compact_int(max(band_starts))}x{len(band_starts)}"
        f"__{timestamp}"
    )


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt(value: float | int | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def fmt_num(value: float | int | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.0f}"


def fmt_optional_int(value: int | None) -> str:
    if value is None or value == "":
        return "-"
    return str(int(value))


def truncate(value: str, width: int) -> str:
    value = str(value)
    if len(value) <= width:
        return value
    if width <= 3:
        return value[:width]
    return value[: width - 3] + "..."


if __name__ == "__main__":
    raise SystemExit(main())
