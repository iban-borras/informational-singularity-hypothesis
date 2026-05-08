#!/usr/bin/env python3
"""Generate D-0087 external recursive control snapshots O/P."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.progress import ProgressIndicator, format_time
from v2.common.cli import resolve_dir
from v2.phase1.external_recursive_controls import (
    CONTRACT_DOC,
    CONTRACT_SHA256,
    ITERATION_LABEL,
    TARGET_OBSERVABLE_BITS,
    default_command,
    generate_control,
    write_control_snapshot,
)


DEFAULT_LEVEL0_ROOT = "results/level0/phi_snapshots"
DEFAULT_REPORT_DIR = "results/hsi_v2/phase1_external_recursive_controls/snapshot_generation"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Level-0-compatible structural snapshots for D-0087 external "
            "recursive controls O and P."
        )
    )
    parser.add_argument(
        "--level0-root",
        default=DEFAULT_LEVEL0_ROOT,
        help="Level-0 phi_snapshots root. Scientific rule contract is unchanged.",
    )
    parser.add_argument(
        "--report-dir",
        default=DEFAULT_REPORT_DIR,
        help="Directory for generation summary/report/manifest artifacts.",
    )
    parser.add_argument(
        "--controls",
        default="O,P",
        help="Comma-separated subset of controls to generate/reuse. Default: O,P.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate snapshots even if D-0087-compatible files already exist.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    controls = parse_controls(args.controls)
    level0_root = resolve_dir(args.level0_root, anchor_file=__file__)
    report_root = resolve_dir(args.report_dir, anchor_file=__file__)
    report_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_slug = f"phase1-external-recursive-snapshots__ctrl-{'-'.join(controls)}__{timestamp}"
    run_dir = report_root / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "run_id": run_slug,
        "timestamp_utc": timestamp,
        "status": "dry-run" if args.dry_run else "running",
        "contract": {
            "doc": CONTRACT_DOC,
            "sha256": CONTRACT_SHA256,
            "iteration": ITERATION_LABEL,
            "target_observable_bits": TARGET_OBSERVABLE_BITS,
        },
        "parameters": {
            "level0_root": str(level0_root),
            "report_dir": str(report_root),
            "controls": controls,
            "force": bool(args.force),
            "dry_run": bool(args.dry_run),
        },
        "snapshots": [],
    }

    started = time.perf_counter()
    try:
        phase_print(
            "Preparing D-0087 external recursive snapshots",
            f"controls={','.join(controls)} | iteration={ITERATION_LABEL} | target={TARGET_OBSERVABLE_BITS:,}",
            quiet=args.quiet,
        )

        if args.dry_run:
            for control in controls:
                state["snapshots"].append(
                    {
                        "variant": control,
                        "status": "dry-run",
                        "struct_path": str((level0_root / f"var_{control}" / f"phi_iter{ITERATION_LABEL}.struct.gz").resolve()),
                    }
                )
        else:
            progress = ProgressIndicator("Generate/reuse O/P snapshots", total=len(controls)) if not args.quiet else None
            if progress is not None:
                with progress:
                    for index, control in enumerate(controls, start=1):
                        progress.update(index - 1, message=f"var_{control}")
                        snapshot = build_snapshot(
                            control=control,
                            level0_root=level0_root,
                            force=args.force,
                            quiet=args.quiet,
                        )
                        state["snapshots"].append(snapshot)
                        progress.update(index, message=f"var_{control} {snapshot['status']}")
            else:
                for control in controls:
                    state["snapshots"].append(
                        build_snapshot(
                            control=control,
                            level0_root=level0_root,
                            force=args.force,
                            quiet=args.quiet,
                        )
                    )

        state["status"] = "dry-run" if args.dry_run else "completed"
        state["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        write_outputs(run_dir, state)
        if not args.quiet:
            print(render_console_summary(state))
            print(f"\nSaved summary to: {run_dir / 'summary.json'}")
            print(f"Saved report to: {run_dir / 'report.md'}")
            print(f"Saved manifest to: {run_dir / 'manifest.json'}")
        return 0
    except Exception as exc:
        state["status"] = "failed"
        state["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        state["error"] = str(exc)
        write_outputs(run_dir, state)
        print(f"[x] Snapshot generation failed after {format_time(time.perf_counter() - started)}: {exc}", file=sys.stderr)
        print(f"Saved failure summary to: {run_dir / 'summary.json'}", file=sys.stderr)
        return 2


def parse_controls(raw: str) -> list[str]:
    controls = [item.strip().upper() for item in raw.split(",") if item.strip()]
    allowed = {"O", "P"}
    if not controls:
        raise SystemExit("--controls cannot be empty")
    invalid = [item for item in controls if item not in allowed]
    if invalid:
        raise SystemExit(f"Unsupported controls: {', '.join(invalid)}")
    return controls


def build_snapshot(*, control: str, level0_root: Path, force: bool, quiet: bool) -> dict:
    generated = generate_control(control)
    return write_control_snapshot(
        generated=generated,
        level0_root=level0_root,
        command=default_command(),
        force=force,
        quiet=quiet,
    )


def write_outputs(run_dir: Path, state: dict) -> None:
    summary_path = run_dir / "summary.json"
    report_path = run_dir / "report.md"
    manifest_path = run_dir / "manifest.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(render_markdown_report(state) + "\n")
    manifest = {
        "run_id": state["run_id"],
        "timestamp_utc": state["timestamp_utc"],
        "status": state["status"],
        "contract": state["contract"],
        "parameters": state["parameters"],
        "outputs": {
            "summary": str(summary_path.resolve()),
            "report": str(report_path.resolve()),
            "manifest": str(manifest_path.resolve()),
        },
        "snapshots": [
            {
                "variant": item.get("variant"),
                "status": item.get("status"),
                "struct_path": item.get("struct_path"),
                "metadata_path": item.get("metadata_path"),
                "sha256_struct_gz": item.get("sha256_struct_gz"),
            }
            for item in state.get("snapshots", [])
        ],
    }
    if "error" in state:
        manifest["error"] = state["error"]
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def render_console_summary(state: dict) -> str:
    lines = [
        "D-0087 external recursive snapshots",
        "-" * 96,
        f"{'variant':<8} {'status':<10} {'obs_bits':>12} {'term_ratio':>11} {'freq0':>8} {'freq1':>8}",
    ]
    for item in state.get("snapshots", []):
        metadata = item.get("metadata") or {}
        lines.append(
            f"{item.get('variant', '-'):<8} "
            f"{item.get('status', '-'):<10} "
            f"{format_int(metadata.get('final_observable_bits')):>12} "
            f"{format_float(metadata.get('terminalization_ratio')):>11} "
            f"{format_float(metadata.get('freq_zero')):>8} "
            f"{format_float(metadata.get('freq_one')):>8}"
        )
    return "\n".join(lines)


def render_markdown_report(state: dict) -> str:
    lines = [
        "# D-0087 External Recursive Snapshot Generation",
        "",
        "## Contract",
        "",
        f"- Document: `{state['contract']['doc']}`",
        f"- SHA256: `{state['contract']['sha256']}`",
        f"- Iteration label: `{state['contract']['iteration']}`",
        f"- Target observable bits: `{state['contract']['target_observable_bits']}`",
        "",
        "## Snapshots",
        "",
        "| Variant | Status | Observable bits | Terminalization ratio | freq_zero | freq_one | Path |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for item in state.get("snapshots", []):
        metadata = item.get("metadata") or {}
        lines.append(
            "| "
            f"`{item.get('variant', '-')}` | "
            f"`{item.get('status', '-')}` | "
            f"{format_int(metadata.get('final_observable_bits'))} | "
            f"{format_float(metadata.get('terminalization_ratio'))} | "
            f"{format_float(metadata.get('freq_zero'))} | "
            f"{format_float(metadata.get('freq_one'))} | "
            f"`{item.get('struct_path', '-')}` |"
        )
    if state.get("error"):
        lines.extend(["", "## Error", "", f"`{state['error']}`"])
    return "\n".join(lines)


def format_float(value) -> str:
    return "-" if value is None else f"{float(value):.6f}"


def format_int(value) -> str:
    return "-" if value is None else f"{int(value)}"


def phase_print(title: str, detail: str = "", quiet: bool = False) -> None:
    if quiet:
        return
    print(f"\n[Phase] {title}")
    if detail:
        print(f"        {detail}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
