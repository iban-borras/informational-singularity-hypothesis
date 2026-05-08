#!/usr/bin/env python3
"""Run the D-0087 Phase 1 external recursive controls O/P."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from utils.progress import format_time
from v2.common.cli import resolve_dir
from v2.phase1.external_recursive_controls import (
    CONTRACT_DOC,
    CONTRACT_SHA256,
    ITERATION_LABEL,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LEVEL0_ROOT = "results/level0/phi_snapshots"
DEFAULT_OUTPUT_DIR = "results/hsi_v2/phase1_external_recursive_controls"
DEFAULT_CONTROLS = ("O", "P")
DEFAULT_NULL_SEEDS = (607, 709)
DEFAULT_SCALES = "8,12,16,20,24,28,32"
DEFAULT_POLICIES = "prefix,suffix"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate D-0087 O/P snapshots if needed, run observed Phase 1 and "
            "phase-matched-LZ controls, and emit a compact recovery report."
        )
    )
    parser.add_argument("--level0-root", default=DEFAULT_LEVEL0_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--segment-bits", type=int, default=1_000_000)
    parser.add_argument("--num-segments", type=int, default=3)
    parser.add_argument("--scales", default=DEFAULT_SCALES)
    parser.add_argument("--policies", default=DEFAULT_POLICIES)
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Require existing D-0087-compatible snapshots instead of invoking the generator.",
    )
    parser.add_argument(
        "--force-snapshots",
        action="store_true",
        help="Pass --force to snapshot generation. Does not change D-0087 rules.",
    )
    parser.add_argument("--quiet-children", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.segment_bits <= 0 or args.num_segments <= 0:
        raise SystemExit("--segment-bits and --num-segments must be positive")

    level0_root = resolve_dir(args.level0_root, anchor_file=__file__)
    output_root = resolve_dir(args.output_dir, anchor_file=__file__)
    output_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_slug = f"phase1-external-recursive-controls__ctrl-O-P__seeds-607-709__{timestamp}"
    run_dir = output_root / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)

    state: dict[str, Any] = {
        "run_id": run_slug,
        "timestamp_utc": timestamp,
        "status": "dry-run" if args.dry_run else "running",
        "contract": {
            "doc": CONTRACT_DOC,
            "sha256": CONTRACT_SHA256,
            "iteration": ITERATION_LABEL,
        },
        "parameters": {
            "level0_root": str(level0_root),
            "output_dir": str(output_root),
            "segment_bits": args.segment_bits,
            "num_segments": args.num_segments,
            "scales": args.scales,
            "policies": args.policies,
            "controls": list(DEFAULT_CONTROLS),
            "null_model": "phase-matched-lz",
            "null_seeds": list(DEFAULT_NULL_SEEDS),
            "skip_generate": bool(args.skip_generate),
            "force_snapshots": bool(args.force_snapshots),
            "dry_run": bool(args.dry_run),
        },
        "commands": [],
        "phase1_runs": [],
        "rows": [],
    }

    started = time.perf_counter()
    try:
        phase_print(
            "Preparing D-0087 Phase 1 external recursive controls",
            f"controls=O,P | null=phase-matched-lz | seeds=607,709 | output={output_root}",
            quiet=args.quiet,
        )

        if not args.skip_generate:
            generator_cmd = build_generator_command(
                level0_root=level0_root,
                report_dir=output_root / "snapshot_generation",
                force=args.force_snapshots,
                dry_run=args.dry_run,
                quiet=args.quiet_children,
            )
            state["commands"].append({"stage": "snapshot_generation", "command": generator_cmd})
            if not args.dry_run:
                run_command(generator_cmd, cwd=SCRIPT_DIR, quiet=args.quiet)

        commands = build_phase1_commands(args=args, level0_root=level0_root, output_root=output_root)
        state["commands"].extend(commands)
        if args.dry_run:
            state["status"] = "dry-run"
        else:
            for index, item in enumerate(commands, start=1):
                phase_print(
                    f"Phase 1 run {index}/{len(commands)}",
                    f"{item['label']} | variant={item['source_variant']} | seed={item.get('null_seed', '-')}",
                    quiet=args.quiet,
                )
                run_dir_path = run_phase1_and_capture_dir(item, output_root=output_root, quiet=args.quiet)
                item["run_dir"] = str(run_dir_path.resolve())
                state["phase1_runs"].append(
                    {
                        "label": item["label"],
                        "source_variant": item["source_variant"],
                        "sequence_kind": item["sequence_kind"],
                        "null_model": item.get("null_model"),
                        "null_seed": item.get("null_seed"),
                        "run_dir": str(run_dir_path.resolve()),
                    }
                )
                state["rows"].append(load_recovery_row(item, run_dir_path))
            state["status"] = "completed"

        state["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        write_outputs(run_dir, state)
        if not args.quiet:
            print(render_console_summary(state))
            print(f"\nSaved summary to: {run_dir / 'summary.json'}")
            print(f"Saved report to: {run_dir / 'report.md'}")
            print(f"Saved CSV table to: {run_dir / 'table.csv'}")
            print(f"Saved manifest to: {run_dir / 'manifest.json'}")
        return 0
    except subprocess.CalledProcessError as exc:
        state["status"] = "failed"
        state["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        state["error"] = {
            "returncode": int(exc.returncode),
            "command": [str(part) for part in exc.cmd],
        }
        write_outputs(run_dir, state)
        print(
            f"[x] D-0087 Phase 1 external controls failed after {format_time(time.perf_counter() - started)}",
            file=sys.stderr,
        )
        print(f"Saved failure summary to: {run_dir / 'summary.json'}", file=sys.stderr)
        return 2
    except Exception as exc:
        state["status"] = "failed"
        state["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        state["error"] = str(exc)
        write_outputs(run_dir, state)
        print(f"[x] D-0087 Phase 1 external controls failed: {exc}", file=sys.stderr)
        print(f"Saved failure summary to: {run_dir / 'summary.json'}", file=sys.stderr)
        return 2


def build_generator_command(
    *,
    level0_root: Path,
    report_dir: Path,
    force: bool,
    dry_run: bool,
    quiet: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(SCRIPT_DIR / "hsi_v2_generate_external_recursive_controls.py"),
        "--level0-root",
        str(level0_root),
        "--report-dir",
        str(report_dir),
    ]
    if force:
        command.append("--force")
    if dry_run:
        command.append("--dry-run")
    if quiet:
        command.append("--quiet")
    return command


def build_phase1_commands(*, args, level0_root: Path, output_root: Path) -> list[dict]:
    commands: list[dict] = []
    for variant in DEFAULT_CONTROLS:
        command = base_phase1_command(
            args=args,
            level0_root=level0_root,
            output_root=output_root,
            variant=variant,
        )
        commands.append(
            {
                "stage": "phase1_observed",
                "label": f"{variant} observed",
                "source_variant": variant,
                "sequence_kind": "observed",
                "command": command,
            }
        )
    for variant in DEFAULT_CONTROLS:
        for seed in DEFAULT_NULL_SEEDS:
            command = base_phase1_command(
                args=args,
                level0_root=level0_root,
                output_root=output_root,
                variant=variant,
            )
            command.extend(["--null-model", "phase-matched-lz", "--null-seed", str(seed)])
            commands.append(
                {
                    "stage": "phase1_null",
                    "label": f"{variant} phase-matched-lz seed {seed}",
                    "source_variant": variant,
                    "sequence_kind": "null_surrogate",
                    "null_model": "phase-matched-lz",
                    "null_seed": seed,
                    "command": command,
                }
            )
    return commands


def base_phase1_command(*, args, level0_root: Path, output_root: Path, variant: str) -> list[str]:
    command = [
        sys.executable,
        str(SCRIPT_DIR / "hsi_v2_phase1_run.py"),
        "--variant",
        variant,
        "--iteration",
        str(ITERATION_LABEL),
        "--level0-root",
        str(level0_root),
        "--segment-bits",
        str(args.segment_bits),
        "--num-segments",
        str(args.num_segments),
        "--scales",
        str(args.scales),
        "--policies",
        str(args.policies),
        "--output-dir",
        str(output_root),
    ]
    if args.quiet_children:
        command.append("--quiet")
    return command


def run_phase1_and_capture_dir(item: dict, *, output_root: Path, quiet: bool) -> Path:
    run_root = output_root / "nulls" if item["sequence_kind"] == "null_surrogate" else output_root
    before = {path.resolve() for path in run_root.glob("*/dataset.json")}
    run_command(item["command"], cwd=SCRIPT_DIR, quiet=quiet)
    after = {path.resolve() for path in run_root.glob("*/dataset.json")}
    created = sorted(after - before, key=lambda path: path.stat().st_mtime, reverse=True)
    if created:
        return created[0].parent
    candidates = sorted(after, key=lambda path: path.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0].parent
    raise RuntimeError(f"Could not locate Phase 1 output directory for {item['label']}")


def run_command(command: list[str], *, cwd: Path, quiet: bool) -> None:
    if not quiet:
        print(f"[Run] {' '.join(str(part) for part in command)}")
    subprocess.run(command, cwd=str(cwd), check=True)


def load_recovery_row(item: dict, run_dir: Path) -> dict:
    with open(run_dir / "dataset.json", "r", encoding="utf-8") as handle:
        dataset = json.load(handle)
    with open(run_dir / "phase1_summary.json", "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    config = dataset["config"]
    source_metadata = dataset.get("source_metadata") or {}
    scale_last = summary["scale_summary"][-1]
    prefix_last = summary["fiber_summary"]["prefix"][-1]
    suffix_last = summary["fiber_summary"]["suffix"][-1]
    candidate_last = int(scale_last["candidate_pattern_count"])
    kept_last = int(scale_last["kept_pattern_count"])
    return {
        "label": item["label"],
        "variant": config["variant"],
        "source_variant": config["source_variant"],
        "sequence_kind": config["sequence_kind"],
        "null_model": config.get("null_model"),
        "null_seed": config.get("null_seed"),
        "kept_last": kept_last,
        "candidate_last": candidate_last,
        "retention_last": safe_ratio(kept_last, candidate_last),
        "prefix_survival_last": prefix_last["survival_fraction"],
        "suffix_survival_last": suffix_last["survival_fraction"],
        "prefix_mean_fiber_last": prefix_last["mean_fiber_size"],
        "suffix_mean_fiber_last": suffix_last["mean_fiber_size"],
        "prefix_entropy_last": prefix_last["mean_child_entropy"],
        "suffix_entropy_last": suffix_last["mean_child_entropy"],
        "terminalization_ratio": source_metadata.get("terminalization_ratio"),
        "freq_zero": source_metadata.get("freq_zero"),
        "freq_one": source_metadata.get("freq_one"),
        "snapshot_sha256": source_metadata.get("sha256_struct_gz"),
        "run_dir": str(run_dir.resolve()),
        "summary_path": str((run_dir / "phase1_summary.json").resolve()),
    }


def write_outputs(run_dir: Path, state: dict) -> None:
    summary_path = run_dir / "summary.json"
    report_path = run_dir / "report.md"
    csv_path = run_dir / "table.csv"
    manifest_path = run_dir / "manifest.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)
    write_csv(state.get("rows", []), csv_path)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(render_markdown_report(state) + "\n")
    manifest = {
        "run_id": state["run_id"],
        "timestamp_utc": state["timestamp_utc"],
        "status": state["status"],
        "contract": state["contract"],
        "parameters": state["parameters"],
        "commands": state["commands"],
        "phase1_runs": state["phase1_runs"],
        "outputs": {
            "summary": str(summary_path.resolve()),
            "report": str(report_path.resolve()),
            "table_csv": str(csv_path.resolve()),
            "manifest": str(manifest_path.resolve()),
        },
    }
    if "error" in state:
        manifest["error"] = state["error"]
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def write_csv(rows: list[dict], csv_path: Path) -> None:
    fieldnames = [
        "label",
        "variant",
        "source_variant",
        "sequence_kind",
        "null_model",
        "null_seed",
        "kept_last",
        "candidate_last",
        "retention_last",
        "prefix_survival_last",
        "suffix_survival_last",
        "prefix_mean_fiber_last",
        "suffix_mean_fiber_last",
        "prefix_entropy_last",
        "suffix_entropy_last",
        "terminalization_ratio",
        "freq_zero",
        "freq_one",
        "snapshot_sha256",
        "summary_path",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def render_console_summary(state: dict) -> str:
    lines = [
        "D-0087 Phase 1 external recursive controls",
        "-" * 132,
        (
            f"{'label':<34} {'kind':<14} {'seed':>6} {'kept':>8} {'cand':>8} "
            f"{'ret':>8} {'p_surv':>8} {'s_surv':>8} {'term':>8}"
        ),
    ]
    for row in state.get("rows", []):
        lines.append(
            f"{row['label']:<34} "
            f"{row['sequence_kind']:<14} "
            f"{format_optional(row.get('null_seed')):>6} "
            f"{int(row['kept_last']):>8d} "
            f"{int(row['candidate_last']):>8d} "
            f"{format_float(row['retention_last']):>8} "
            f"{format_float(row['prefix_survival_last']):>8} "
            f"{format_float(row['suffix_survival_last']):>8} "
            f"{format_float(row.get('terminalization_ratio')):>8}"
        )
    return "\n".join(lines)


def render_markdown_report(state: dict) -> str:
    lines = [
        "# D-0087 Phase 1 External Recursive Controls",
        "",
        "## Contract",
        "",
        f"- Document: `{state['contract']['doc']}`",
        f"- SHA256: `{state['contract']['sha256']}`",
        f"- Iteration: `{state['contract']['iteration']}`",
        "",
        "## Recovery Table",
        "",
        "| Label | Kind | Seed | kept@last | cand@last | retention | prefix survival | suffix survival | term ratio |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in state.get("rows", []):
        lines.append(
            "| "
            f"`{row['label']}` | "
            f"`{row['sequence_kind']}` | "
            f"{format_optional(row.get('null_seed'))} | "
            f"{int(row['kept_last'])} | "
            f"{int(row['candidate_last'])} | "
            f"{format_float(row['retention_last'])} | "
            f"{format_float(row['prefix_survival_last'])} | "
            f"{format_float(row['suffix_survival_last'])} | "
            f"{format_float(row.get('terminalization_ratio'))} |"
        )
    lines.extend(["", "## Run Paths", ""])
    for row in state.get("rows", []):
        lines.append(f"- `{row['label']}`: `{row['summary_path']}`")
    if state.get("error"):
        lines.extend(["", "## Error", "", f"`{state['error']}`"])
    return "\n".join(lines)


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def format_float(value) -> str:
    return "-" if value is None else f"{float(value):.4f}"


def format_optional(value) -> str:
    return "-" if value is None else str(value)


def phase_print(title: str, detail: str = "", quiet: bool = False) -> None:
    if quiet:
        return
    print(f"\n[Phase] {title}")
    if detail:
        print(f"        {detail}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
