#!/usr/bin/env python3
"""
HSI v2 Phase 2 Parent Shell Atlas Sensitivity

Reproducible N2-06 wrapper for checking whether the parent shell atlas reading
depends on the monitored top-k boundary or the exact candidate lag.
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
from v2.phase2.null_pressure import PHASE2_SEEDED_NULLS, discover_phase1_runs_recursive


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the N2-06 parent-shell atlas sensitivity sweep over top-k "
            "selection and candidate-lag neighborhoods."
        )
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="canonical-sensitivity",
        choices=("canonical-sensitivity",),
        help="Named reproducible sensitivity profile.",
    )
    parser.add_argument(
        "--phase1-dir",
        type=str,
        default="results/hsi_v2/phase1_high_scales",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/hsi_v2/phase2/parent_shell_atlas_sensitivity",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=("observed", "nulls", "all"),
    )
    parser.add_argument("--top-patterns-list", type=str, default="32,64,128")
    parser.add_argument("--lag-bits-list", type=str, default="14500000,15000000,15500000")
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

    try:
        top_values = parse_int_list(
            args.top_patterns_list,
            label="--top-patterns-list",
            allow_zero=False,
        )
        lag_values = parse_int_list(
            args.lag_bits_list,
            label="--lag-bits-list",
            allow_zero=False,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if any(value <= 0 for value in top_values):
        parser.error("--top-patterns-list must contain positive integers.")

    selection = build_selection(args)
    script_dir = Path(__file__).parent.resolve()
    phase1_dir = resolve_dir(args.phase1_dir, anchor_file=__file__)
    output_dir = resolve_dir(args.output_dir, anchor_file=__file__)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_slug = build_run_slug(selection, top_values, lag_values, timestamp)
    run_dir = output_dir / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)

    phase_print(
        "Preparing parent shell atlas sensitivity",
        (
            f"profile={args.profile} | stage={selection['stage']} | "
            f"top={','.join(str(value) for value in top_values)} | "
            f"lags={','.join(compact_int(value) for value in lag_values)}"
        ),
        quiet=args.quiet,
    )
    flush_output()

    phase1_input_rows = ensure_required_phase1_inputs(
        script_dir=script_dir,
        phase1_dir=phase1_dir,
        selection=selection,
        quiet=args.quiet,
    )

    result_rows: list[dict] = []
    structure_rows: list[dict] = []
    subtype_rows: list[dict] = []
    probe_rows: list[dict] = []
    commands: list[dict] = [
        {
            "stage": "phase1-input",
            "variant": row["variant"],
            "source_kind": row["source_kind"],
            "null_model": row.get("null_model"),
            "null_seed": row.get("null_seed"),
            "action": row["action"],
            "command": row.get("command"),
        }
        for row in phase1_input_rows
    ]

    for top_patterns in top_values:
        top_root = run_dir / f"top-{top_patterns}"
        probe_root = top_root / "probe"
        probe_cmd = build_probe_cmd(
            script_dir=script_dir,
            phase1_dir=phase1_dir,
            output_dir=probe_root,
            selection=selection,
            top_patterns=top_patterns,
        )
        append_flag(probe_cmd, "--quiet", args.quiet)

        phase_print(
            "Stage 1: shell-lag probe",
            f"top={top_patterns} | output_dir={probe_root}",
            quiet=args.quiet,
        )
        flush_output()
        run_subprocess(probe_cmd, cwd=script_dir)
        probe_summary_path = latest_child_file(probe_root, "summary.json")
        probe_payload = load_json(probe_summary_path)
        recommended_lag_bits = probe_payload.get("summary", {}).get("recommended_lag_bits")
        probe_rows.append(
            {
                "top_patterns": top_patterns,
                "recommended_lag_bits": recommended_lag_bits,
                "probe_summary_path": str(probe_summary_path),
            }
        )
        commands.append(
            {
                "stage": "probe",
                "top_patterns": top_patterns,
                "command": probe_cmd,
            }
        )

        for lag_bits in lag_values:
            combo_root = top_root / f"lag-{compact_int(lag_bits)}"
            lagaware_root = combo_root / "lagaware"
            atlas_root = combo_root / "atlas"

            lagaware_cmd = build_lagaware_cmd(
                script_dir=script_dir,
                phase1_dir=phase1_dir,
                output_dir=lagaware_root,
                selection=selection,
                top_patterns=top_patterns,
                lag_bits=lag_bits,
            )
            append_flag(lagaware_cmd, "--quiet", args.quiet)

            phase_print(
                "Stage 2: lag-aware band",
                f"top={top_patterns} | lag={lag_bits} | output_dir={lagaware_root}",
                quiet=args.quiet,
            )
            flush_output()
            run_subprocess(lagaware_cmd, cwd=script_dir)
            lagaware_summary_path = latest_child_file(lagaware_root, "summary.json")

            synthetic_revalidation_path = write_synthetic_revalidation_summary(
                combo_root=combo_root,
                selection=selection,
                phase1_dir=phase1_dir,
                top_patterns=top_patterns,
                lag_bits=lag_bits,
                probe_summary_path=probe_summary_path,
                lagaware_summary_path=lagaware_summary_path,
                recommended_lag_bits=recommended_lag_bits,
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

            phase_print(
                "Stage 3: shell atlas",
                f"top={top_patterns} | lag={lag_bits} | output_dir={atlas_root}",
                quiet=args.quiet,
            )
            flush_output()
            run_subprocess(atlas_cmd, cwd=script_dir)
            atlas_summary_path = latest_child_file(atlas_root, "summary.json")
            atlas_payload = load_json(atlas_summary_path)

            commands.extend(
                [
                    {
                        "stage": "lagaware",
                        "top_patterns": top_patterns,
                        "lag_bits": lag_bits,
                        "command": lagaware_cmd,
                    },
                    {
                        "stage": "atlas",
                        "top_patterns": top_patterns,
                        "lag_bits": lag_bits,
                        "command": atlas_cmd,
                    },
                ]
            )
            collect_result_rows(
                result_rows,
                structure_rows,
                subtype_rows,
                top_patterns=top_patterns,
                lag_bits=lag_bits,
                recommended_lag_bits=recommended_lag_bits,
                synthetic_revalidation_path=synthetic_revalidation_path,
                lagaware_summary_path=lagaware_summary_path,
                atlas_summary_path=atlas_summary_path,
                atlas_payload=atlas_payload,
            )

    generated_at = datetime.now().isoformat(timespec="seconds")
    summary_path = run_dir / "summary.json"
    report_path = run_dir / "report.md"
    result_csv_path = run_dir / "concentration_sensitivity.csv"
    structure_csv_path = run_dir / "structure_sensitivity.csv"
    subtype_csv_path = run_dir / "subtype_sensitivity.csv"
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
            "top_patterns_list": top_values,
            "lag_bits_list": lag_values,
            "null_models": selection["null_models"].split(",") if selection["null_models"] else [],
            "matched_lz_seeds": [
                int(item) for item in selection["matched_lz_seeds"].split(",") if item.strip()
            ],
        },
        "phase1_input_recovery": phase1_input_rows,
        "probe_summary": probe_rows,
        "structure_sensitivity": structure_rows,
        "concentration_sensitivity": result_rows,
        "subtype_sensitivity": subtype_rows,
    }
    manifest_payload = {
        "generated_at": generated_at,
        "run_slug": run_slug,
        "script": "hsi_v2_phase2_parent_shell_atlas_sensitivity.py",
        "cwd": str(Path.cwd()),
        "outputs": {
            "summary": str(summary_path),
            "report": str(report_path),
            "concentration_csv": str(result_csv_path),
            "structure_csv": str(structure_csv_path),
            "subtype_csv": str(subtype_csv_path),
            "manifest": str(manifest_path),
        },
        "commands": commands,
        "arguments": vars(args),
    }

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(render_report(summary_payload) + "\n")
    write_csv(result_rows, result_csv_path)
    write_csv(structure_rows, structure_csv_path)
    write_csv(subtype_rows, subtype_csv_path)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, indent=2)

    if not args.quiet:
        print(render_console_summary(result_rows))
        print("")
        print(f"Saved summary to: {summary_path}")
        print(f"Saved report to: {report_path}")
        print(f"Saved concentration CSV to: {result_csv_path}")
        print(f"Saved structure CSV to: {structure_csv_path}")
        print(f"Saved subtype CSV to: {subtype_csv_path}")
        print(f"Saved manifest to: {manifest_path}")

    return 0


def build_selection(args) -> dict:
    selection = dict(CANONICAL_ALL_PROFILE)
    selection["profile"] = args.profile
    selection["stage"] = args.stage
    if args.null_models:
        selection["null_models"] = args.null_models
    if args.matched_lz_seeds:
        selection["matched_lz_seeds"] = args.matched_lz_seeds
    if selection["stage"] == "observed":
        selection["null_models"] = ""
        selection["matched_lz_seeds"] = ""
    elif not selection_uses_seeded_nulls(selection):
        selection["matched_lz_seeds"] = ""

    variants_list = [item.strip() for item in selection["variants"].split(",") if item.strip()]
    if len(variants_list) != 2:
        raise ValueError("The canonical sensitivity profile expects exactly two variants.")
    selection["variants_list"] = variants_list
    selection["candidate_variant"] = next(
        variant for variant in variants_list if variant != selection["anchor_variant"]
    )
    return selection


def ensure_required_phase1_inputs(
    *,
    script_dir: Path,
    phase1_dir: Path,
    selection: dict,
    quiet: bool,
) -> list[dict]:
    runs = discover_phase1_runs_recursive(phase1_dir)
    recovery_rows: list[dict] = []

    for variant in selection["variants_list"]:
        if phase1_run_exists(runs, selection=selection, variant=variant, source_kind="observed"):
            recovery_rows.append(
                {
                    "variant": variant,
                    "source_kind": "observed",
                    "action": "present",
                }
            )
            continue
        command = build_phase1_run_cmd(
            script_dir=script_dir,
            phase1_dir=phase1_dir,
            selection=selection,
            variant=variant,
            source_kind="observed",
        )
        phase_print(
            "Phase 0: generating missing Phase 1 observed tower",
            f"variant={variant}",
            quiet=quiet,
        )
        flush_output()
        run_subprocess(command, cwd=script_dir)
        recovery_rows.append(
            {
                "variant": variant,
                "source_kind": "observed",
                "action": "generated",
                "command": command,
            }
        )

    if selection["stage"] not in {"nulls", "all"}:
        return recovery_rows

    null_models = parse_csv_tokens(selection["null_models"])
    candidate_variant = selection["candidate_variant"]
    if "markov1" in null_models:
        seed = 17
        if phase1_run_exists(
            runs,
            selection=selection,
            variant=candidate_variant,
            source_kind="null_surrogate",
            null_model="markov1",
            null_seed=seed,
        ):
            recovery_rows.append(
                {
                    "variant": candidate_variant,
                    "source_kind": "null_surrogate",
                    "null_model": "markov1",
                    "null_seed": seed,
                    "action": "present",
                }
            )
        else:
            command = build_phase1_run_cmd(
                script_dir=script_dir,
                phase1_dir=phase1_dir,
                selection=selection,
                variant=candidate_variant,
                source_kind="null_surrogate",
                null_model="markov1",
                null_seed=seed,
            )
            phase_print(
                "Phase 0: generating missing Phase 1 null tower",
                f"variant={candidate_variant} | null=markov1 | seed={seed}",
                quiet=quiet,
            )
            flush_output()
            run_subprocess(command, cwd=script_dir)
            recovery_rows.append(
                {
                    "variant": candidate_variant,
                    "source_kind": "null_surrogate",
                    "null_model": "markov1",
                    "null_seed": seed,
                    "action": "generated",
                    "command": command,
                }
            )

    for null_model in [model for model in null_models if model in PHASE2_SEEDED_NULLS]:
        seeds = parse_seed_csv(selection["matched_lz_seeds"])
        for seed in seeds:
            if phase1_run_exists(
                runs,
                selection=selection,
                variant=candidate_variant,
                source_kind="null_surrogate",
                null_model=null_model,
                null_seed=seed,
            ):
                recovery_rows.append(
                    {
                        "variant": candidate_variant,
                        "source_kind": "null_surrogate",
                        "null_model": null_model,
                        "null_seed": seed,
                        "action": "present",
                    }
                )
                continue
            command = build_phase1_run_cmd(
                script_dir=script_dir,
                phase1_dir=phase1_dir,
                selection=selection,
                variant=candidate_variant,
                source_kind="null_surrogate",
                null_model=null_model,
                null_seed=seed,
            )
            phase_print(
                "Phase 0: generating missing Phase 1 null tower",
                f"variant={candidate_variant} | null={null_model} | seed={seed}",
                quiet=quiet,
            )
            flush_output()
            run_subprocess(command, cwd=script_dir)
            recovery_rows.append(
                    {
                        "variant": candidate_variant,
                        "source_kind": "null_surrogate",
                        "null_model": null_model,
                        "null_seed": seed,
                        "action": "generated",
                        "command": command,
                }
            )
    return recovery_rows


def phase1_run_exists(
    runs: list[dict],
    *,
    selection: dict,
    variant: str,
    source_kind: str,
    null_model: str | None = None,
    null_seed: int | None = None,
) -> bool:
    for run in runs:
        config = run["dataset"].get("config", {})
        if str(config.get("sequence_kind", "observed")) != source_kind:
            continue
        if source_kind == "observed":
            run_variant = str(config.get("variant") or config.get("source_variant") or "").upper()
        else:
            run_variant = str(config.get("source_variant") or "").upper()
        if run_variant != variant.upper():
            continue
        if source_kind == "null_surrogate":
            if normalize_null_name(config.get("null_model")) != normalize_null_name(null_model):
                continue
            if int_or_default(config.get("null_seed")) != int(null_seed):
                continue
        if int_or_default(config.get("iteration")) != int(selection["iteration"]):
            continue
        if int_or_default(config.get("segment_bits")) != int(selection["segment_bits"]):
            continue
        if int_or_default(config.get("num_segments")) != int(selection["num_segments"]):
            continue
        if list(config.get("scales", [])) != parse_seed_scales(selection["scales"]):
            continue
        if list(config.get("policies", [])) != parse_csv_tokens(selection["phase1_policies"]):
            continue
        return True
    return False


def build_phase1_run_cmd(
    *,
    script_dir: Path,
    phase1_dir: Path,
    selection: dict,
    variant: str,
    source_kind: str,
    null_model: str | None = None,
    null_seed: int | None = None,
) -> list[str]:
    if source_kind == "observed":
        output_dir = phase1_dir / "auto_observed" / variant
    else:
        output_dir = phase1_dir / "auto_nulls" / str(null_model) / f"seed-{null_seed}" / variant
    command = [
        sys.executable,
        str(script_dir / "hsi_v2_phase1_run.py"),
        "--variant",
        variant,
        "--iteration",
        str(selection["iteration"]),
        "--segment-bits",
        str(selection["segment_bits"]),
        "--num-segments",
        str(selection["num_segments"]),
        "--scales",
        selection["scales"],
        "--policies",
        selection["phase1_policies"],
        "--output-dir",
        str(output_dir),
    ]
    if source_kind == "null_surrogate":
        command.extend(["--null-model", str(null_model), "--null-seed", str(null_seed)])
    return command


def parse_csv_tokens(raw) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple, set)):
        return [str(item).strip() for item in raw if str(item).strip()]
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def parse_seed_csv(raw: str) -> list[int]:
    values: list[int] = []
    seen: set[int] = set()
    for item in parse_csv_tokens(raw):
        value = int(item)
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
    return values


def parse_seed_scales(raw: str) -> list[int]:
    return [int(item) for item in parse_csv_tokens(raw)]


def selection_uses_seeded_nulls(selection: dict) -> bool:
    return any(model in PHASE2_SEEDED_NULLS for model in parse_csv_tokens(selection.get("null_models", "")))


def int_or_default(value, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_null_name(value) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def build_probe_cmd(
    *,
    script_dir: Path,
    phase1_dir: Path,
    output_dir: Path,
    selection: dict,
    top_patterns: int,
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
        selection["definition_offsets"],
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


def build_lagaware_cmd(
    *,
    script_dir: Path,
    phase1_dir: Path,
    output_dir: Path,
    selection: dict,
    top_patterns: int,
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
        selection["offsets"],
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
    combo_root: Path,
    selection: dict,
    phase1_dir: Path,
    top_patterns: int,
    lag_bits: int,
    probe_summary_path: Path,
    lagaware_summary_path: Path,
    recommended_lag_bits,
) -> Path:
    combo_root.mkdir(parents=True, exist_ok=True)
    path = combo_root / "synthetic_revalidation_summary.json"
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
            "candidate_lag_bits": lag_bits,
            "probe_recommended_lag_bits": recommended_lag_bits,
        },
        "probe_summary_path": str(probe_summary_path),
        "lagaware_summary_path": str(lagaware_summary_path),
        "recommended_lag_bits": lag_bits,
        "notes": [
            "Synthetic revalidation pointer generated by N2-06 sensitivity wrapper.",
            "The lag-aware dataset is built with an explicit candidate lag for sensitivity testing.",
        ],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def collect_result_rows(
    result_rows: list[dict],
    structure_rows: list[dict],
    subtype_rows: list[dict],
    *,
    top_patterns: int,
    lag_bits: int,
    recommended_lag_bits,
    synthetic_revalidation_path: Path,
    lagaware_summary_path: Path,
    atlas_summary_path: Path,
    atlas_payload: dict,
) -> None:
    common = {
        "top_patterns": top_patterns,
        "lag_bits": lag_bits,
        "probe_recommended_lag_bits": recommended_lag_bits,
        "synthetic_revalidation_path": str(synthetic_revalidation_path),
        "lagaware_summary_path": str(lagaware_summary_path),
        "atlas_summary_path": str(atlas_summary_path),
    }
    for row in atlas_payload.get("structural_summary", []):
        structure_rows.append(
            {
                **common,
                "shell_subtype": row.get("shell_subtype"),
                "shell_subtype_label": row.get("shell_subtype_label"),
                "shell_pattern_count": row.get("shell_pattern_count"),
                "monitored_shell_pattern_count": row.get("monitored_shell_pattern_count"),
                "unmonitored_shell_pattern_count": row.get("unmonitored_shell_pattern_count"),
                "shell_pattern_fraction": row.get("shell_pattern_fraction"),
            }
        )
    for row in atlas_payload.get("concentration_summary", []):
        result_rows.append(
            {
                **common,
                "variant": row["variant"],
                "source_label": row["source_label"],
                "source_kind": row["source_kind"],
                "null_model": row.get("null_model"),
                "null_seed": row.get("null_seed"),
                "net_deficit_mass_sum": row.get("net_deficit_mass_sum"),
                "gross_loss_pressure_sum": row.get("gross_loss_pressure_sum"),
                "jitter_gap_mass": row.get("jitter_gap_mass"),
                "top1_net_deficit_share": row.get("top1_net_deficit_share"),
                "top3_net_deficit_share": row.get("top3_net_deficit_share"),
                "top5_net_deficit_share": row.get("top5_net_deficit_share"),
                "dominant_subtype_label": row.get("dominant_subtype_label"),
                "dominant_subtype_share": row.get("dominant_subtype_share"),
            }
        )
    for row in atlas_payload.get("subtype_summary", []):
        subtype_rows.append(
            {
                **common,
                "variant": row["variant"],
                "source_label": row["source_label"],
                "source_kind": row["source_kind"],
                "null_model": row.get("null_model"),
                "null_seed": row.get("null_seed"),
                "shell_subtype_label": row.get("shell_subtype_label"),
                "partition_mass_retention_pooled": row.get("partition_mass_retention_pooled"),
                "survivor_internal_deformation_log2_mean_pooled": row.get(
                    "survivor_internal_deformation_log2_mean_pooled"
                ),
                "dead_anchor_mass_fraction": row.get("dead_anchor_mass_fraction"),
                "dark_anchor_mass_fraction": row.get("dark_anchor_mass_fraction"),
                "net_deficit_mass": row.get("net_deficit_mass"),
                "gross_loss_pressure_sum": row.get("gross_loss_pressure_sum"),
                "jitter_gap_mass": row.get("jitter_gap_mass"),
            }
        )


def render_report(payload: dict) -> str:
    selection = payload["selection"]
    lines = [
        "# Phase 2 Parent Shell Atlas Sensitivity",
        "",
        "## Selection",
        "",
        f"- Profile: {selection['profile']}",
        f"- Stage: {selection['stage']}",
        f"- Anchor variant: {selection['anchor_variant']}",
        f"- Candidate variant: {selection['candidate_variant']}",
        f"- Top-k values: {', '.join(str(value) for value in selection['top_patterns_list'])}",
        f"- Lag values: {', '.join(str(value) for value in selection['lag_bits_list'])}",
        f"- Null models: {', '.join(parse_csv_tokens(selection['null_models'])) or '-'}",
        f"- Matched-LZ seeds: {', '.join(str(value) for value in parse_seed_csv(selection['matched_lz_seeds'])) or '-'}",
        "",
        "## Phase 1 Input Recovery",
        "",
        "| Variant | Source kind | Null | Seed | Action |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for row in payload.get("phase1_input_recovery", []):
        lines.append(
            f"| {row['variant']} | {row['source_kind']} | "
            f"{row.get('null_model') or '-'} | {row.get('null_seed') or '-'} | "
            f"{row['action']} |"
        )
    lines.extend(
        [
            "",
            "## Lag Probe Summary",
            "",
            "| Top-k | Probe recommended lag | Probe summary |",
            "| ---: | ---: | --- |",
        ]
    )
    for row in payload["probe_summary"]:
        lines.append(
            f"| {row['top_patterns']} | {row['recommended_lag_bits']} | "
            f"`{row['probe_summary_path']}` |"
        )

    lines.extend(
        [
            "",
            "## Shell Structure Sensitivity",
            "",
            "| Top-k | Lag | Shell subtype | Shell patterns | Monitored | Unmonitored | Fraction |",
            "| ---: | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["structure_sensitivity"]:
        lines.append(
            f"| {row['top_patterns']} | {row['lag_bits']} | "
            f"{row['shell_subtype_label']} | {row['shell_pattern_count']} | "
            f"{row['monitored_shell_pattern_count']} | "
            f"{row['unmonitored_shell_pattern_count']} | "
            f"{fmt(row['shell_pattern_fraction'])} |"
        )

    lines.extend(
        [
            "",
            "## Concentration Sensitivity",
            "",
            "| Top-k | Lag | Variant | Source | Net def | Gross | Jitter gap | Top1 | Top3 | Top5 |",
            "| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["concentration_sensitivity"]:
        lines.append(
            f"| {row['top_patterns']} | {row['lag_bits']} | {row['variant']} | "
            f"{row['source_label']} | {row['net_deficit_mass_sum']} | "
            f"{row['gross_loss_pressure_sum']} | {row['jitter_gap_mass']} | "
            f"{fmt(row['top1_net_deficit_share'])} | {fmt(row['top3_net_deficit_share'])} | "
            f"{fmt(row['top5_net_deficit_share'])} |"
        )
    return "\n".join(lines)


def render_console_summary(result_rows: list[dict]) -> str:
    lines = [
        "Phase 2 parent shell atlas sensitivity",
        "-" * 132,
        f"{'top':>5}{'lag':>10} {'variant':<10} {'source':<28}{'netDef':>10}{'gross':>10}{'jGap':>10}{'top1':>10}{'top3':>10}{'top5':>10}",
    ]
    if not result_rows:
        lines.append("No concentration rows were emitted; inspect structure_sensitivity.csv for empty monitored shells.")
        return "\n".join(lines)
    for row in result_rows:
        lines.append(
            f"{int(row['top_patterns']):>5}{compact_int(int(row['lag_bits'])):>10} "
            f"{truncate(row['variant'], 10):<10} "
            f"{truncate(row['source_label'], 28):<28}"
            f"{int(row['net_deficit_mass_sum']):>10}"
            f"{int(row['gross_loss_pressure_sum']):>10}"
            f"{int(row['jitter_gap_mass']):>10}"
            f"{fmt(row['top1_net_deficit_share']):>10}"
            f"{fmt(row['top3_net_deficit_share']):>10}"
            f"{fmt(row['top5_net_deficit_share']):>10}"
        )
    return "\n".join(lines)


def build_run_slug(
    selection: dict,
    top_values: list[int],
    lag_values: list[int],
    timestamp: str,
) -> str:
    return (
        f"phase2-parent-shell-atlas-sensitivity__stage-{selection['stage']}"
        f"__anchor-{selection['anchor_variant']}__cand-{selection['candidate_variant']}"
        f"__top-{min(top_values)}-{max(top_values)}x{len(top_values)}"
        f"__lag-{compact_int(min(lag_values))}-{compact_int(max(lag_values))}x{len(lag_values)}"
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


def fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def truncate(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    if width <= 3:
        return value[:width]
    return value[: width - 3] + "..."


if __name__ == "__main__":
    raise SystemExit(main())
