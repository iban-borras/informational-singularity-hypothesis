#!/usr/bin/env python3
"""
HSI v2 Phase 2 Phase-Lag Atlas

N2-10 scout object. It maps the local shell-lag field across neighboring
anchor bands without running the heavier null/routing stack at every point.
The output is a band-level lag table plus a deterministic PNG phase map.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from hsi_v2_phase2_parent_shell_atlas_pipeline import CANONICAL_ALL_PROFILE
from hsi_v2_phase2_parent_survival_revalidation import (
    latest_child_file,
    load_json,
    run_subprocess,
)
from hsi_v2_phase2_transport_defect_strict import parse_int_list, phase_print
from v2.common.cli import resolve_dir
from v2.common.naming import compact_int


DEFAULT_OUTPUT_DIR = "results/hsi_v2/phase2/phase_lag_atlas"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run an N2-10 phase-lag atlas scout by applying the observed "
            "parent-shell lag probe over successive anchor bands."
        )
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="phase-scout",
        choices=("phase-scout",),
        help="Named reproducible scout profile.",
    )
    parser.add_argument(
        "--phase1-dir",
        type=str,
        default="results/hsi_v2/phase1_high_scales",
    )
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--band-starts",
        type=str,
        default="",
        help="Optional comma-separated band starts. Overrides start/stop/step.",
    )
    parser.add_argument("--band-start-bits", type=int, default=660_000_000)
    parser.add_argument("--band-stop-bits", type=int, default=732_000_000)
    parser.add_argument("--band-step-bits", type=int, default=9_000_000)
    parser.add_argument("--window-count", type=int, default=19)
    parser.add_argument("--window-step-bits", type=int, default=500_000)
    parser.add_argument("--probe-backward-bits", type=int, default=45_000_000)
    parser.add_argument("--probe-forward-bits", type=int, default=45_000_000)
    parser.add_argument("--scan-step-bits", type=int, default=500_000)
    parser.add_argument("--top-patterns", type=int, default=128)
    parser.add_argument("--no-plot", action="store_true")
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
    band_starts = resolve_band_starts(args, parser)
    selection = build_selection(args, phase1_dir, band_starts)

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_slug = build_run_slug(selection, args, band_starts, timestamp)
    run_dir = output_root / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)

    phase_print(
        "Preparing N2-10 phase-lag atlas",
        (
            f"anchor={selection['anchor_variant']} | candidate={selection['candidate_variant']} | "
            f"bands={len(band_starts)} | top={args.top_patterns}"
        ),
        quiet=args.quiet,
    )

    commands: list[dict] = []
    rows: list[dict] = []
    for band_index, band_start in enumerate(band_starts, start=1):
        band_offsets = build_offsets(
            start_bits=band_start,
            count=args.window_count,
            step_bits=args.window_step_bits,
        )
        band_end = band_offsets[-1]
        band_label = f"band-{compact_int(band_start)}-{compact_int(band_end)}"
        probe_root = run_dir / band_label / "probe"
        scan_start = max(0, band_start - args.probe_backward_bits)
        scan_stop = band_start + args.probe_forward_bits

        phase_print(
            "Phase-lag band",
            (
                f"{band_index}/{len(band_starts)} | {band_label} | "
                f"scan={compact_int(scan_start)}-{compact_int(scan_stop)}"
            ),
            quiet=args.quiet,
        )
        command = build_probe_cmd(
            script_dir=root,
            phase1_dir=phase1_dir,
            output_dir=probe_root,
            selection=selection,
            top_patterns=args.top_patterns,
            definition_offsets=band_offsets,
            scan_start_bits=scan_start,
            scan_stop_bits=scan_stop,
            scan_step_bits=args.scan_step_bits,
            quiet=args.quiet,
        )
        run_subprocess(command, cwd=root)
        summary_path = latest_child_file(probe_root, "summary.json")
        payload = load_json(summary_path)
        row = summarize_probe_band(
            band_index=band_index,
            band_label=band_label,
            band_start=band_start,
            band_end=band_end,
            scan_start=scan_start,
            scan_stop=scan_stop,
            summary_path=summary_path,
            payload=payload,
        )
        rows.append(row)
        commands.append({"band_label": band_label, "stage": "probe", "command": command})

    report_path = run_dir / "report.md"
    csv_path = run_dir / "phase_lag_atlas.csv"
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "manifest.json"
    png_path = run_dir / "phase_lag_atlas.png"

    write_csv(rows, csv_path)
    if not args.no_plot:
        plot_phase_lag_atlas(rows, png_path)
    else:
        png_path = None

    generated_at = datetime.now().isoformat(timespec="seconds")
    summary = {
        "generated_at": generated_at,
        "selection": selection,
        "rows": rows,
        "outputs": {
            "report": str(report_path),
            "csv": str(csv_path),
            "png": str(png_path) if png_path else None,
            "summary": str(summary_path),
            "manifest": str(manifest_path),
        },
    }
    manifest = {
        "generated_at": generated_at,
        "script": "hsi_v2_phase2_phase_lag_atlas.py",
        "cwd": str(Path.cwd()),
        "arguments": vars(args),
        "commands": commands,
        "outputs": summary["outputs"],
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(render_report(summary) + "\n")

    copy_latest_artifacts(run_dir, output_root)

    if not args.quiet:
        print(render_console_table(rows))
        print("")
        print(f"Saved summary to: {summary_path}")
        print(f"Saved report to: {report_path}")
        print(f"Saved CSV to: {csv_path}")
        if png_path:
            print(f"Saved PNG to: {png_path}")
        print(f"Saved manifest to: {manifest_path}")
        print(f"Updated latest artifacts in: {output_root}")

    return 0


def validate_args(args, parser: argparse.ArgumentParser) -> None:
    if args.band_step_bits <= 0:
        parser.error("--band-step-bits must be positive.")
    if args.band_stop_bits < args.band_start_bits:
        parser.error("--band-stop-bits must be >= --band-start-bits.")
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


def resolve_band_starts(args, parser: argparse.ArgumentParser) -> list[int]:
    if args.band_starts.strip():
        try:
            values = parse_int_list(args.band_starts, label="--band-starts", allow_zero=True)
        except ValueError as exc:
            parser.error(str(exc))
    else:
        values = list(
            range(args.band_start_bits, args.band_stop_bits + 1, args.band_step_bits)
        )
    if any(value < 0 for value in values):
        parser.error("Band starts must be non-negative.")
    return values


def build_selection(args, phase1_dir: Path, band_starts: list[int]) -> dict:
    base = dict(CANONICAL_ALL_PROFILE)
    variants = [item.strip() for item in base["variants"].split(",") if item.strip()]
    anchor_variant = base["anchor_variant"]
    candidate_variant = next(variant for variant in variants if variant != anchor_variant)
    return {
        "profile": args.profile,
        "phase1_dir": str(phase1_dir),
        "variants": variants,
        "variants_raw": base["variants"],
        "anchor_variant": anchor_variant,
        "candidate_variant": candidate_variant,
        "iteration": base["iteration"],
        "segment_bits": base["segment_bits"],
        "num_segments": base["num_segments"],
        "scales": base["scales"],
        "phase1_policies": base["phase1_policies"],
        "pattern_scale": base["low_scale"],
        "top_patterns": args.top_patterns,
        "pattern_selection": base["pattern_selection"],
        "band_starts": band_starts,
        "window_count": args.window_count,
        "window_step_bits": args.window_step_bits,
        "probe_backward_bits": args.probe_backward_bits,
        "probe_forward_bits": args.probe_forward_bits,
        "scan_step_bits": args.scan_step_bits,
    }


def build_offsets(*, start_bits: int, count: int, step_bits: int) -> list[int]:
    return [start_bits + index * step_bits for index in range(count)]


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
    quiet: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(script_dir / "hsi_v2_phase2_parent_shell_lag_probe.py"),
        "--phase1-dir",
        str(phase1_dir),
        "--output-dir",
        str(output_dir),
        "--variants",
        selection["variants_raw"],
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
        str(selection["pattern_scale"]),
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
        "--no-family-inference",
    ]
    if quiet:
        command.append("--quiet")
    return command


def summarize_probe_band(
    *,
    band_index: int,
    band_label: str,
    band_start: int,
    band_end: int,
    scan_start: int,
    scan_stop: int,
    summary_path: Path,
    payload: dict,
) -> dict:
    summary = payload.get("summary", {})
    lag = as_int(summary.get("recommended_lag_bits"))
    hit_corr = summary.get("shell_hits_correlation") or {}
    unique_corr = summary.get("shell_unique_correlation") or {}
    return {
        "band_index": band_index,
        "band_label": band_label,
        "band_start_bits": band_start,
        "band_end_bits": band_end,
        "scan_start_bits": scan_start,
        "scan_stop_bits": scan_stop,
        "recommended_lag_bits": lag,
        "lag_class": classify_lag(lag),
        "anchor_shell_pattern_count": as_int(summary.get("anchor_shell_pattern_count")),
        "anchor_core_pattern_count": as_int(summary.get("anchor_core_pattern_count")),
        "first_anchor_shell_offset_bits": as_int(
            summary.get("first_anchor_shell_offset_bits")
        ),
        "first_candidate_shell_offset_bits": as_int(
            summary.get("first_candidate_shell_offset_bits")
        ),
        "first_candidate_full_shell_offset_bits": as_int(
            summary.get("first_candidate_full_shell_offset_bits")
        ),
        "first_full_shell_lag_bits": as_int(summary.get("first_full_shell_lag_bits")),
        "hit_corr_lag_bits": as_int(hit_corr.get("best_shift_bits")),
        "hit_corr_score": as_float(hit_corr.get("best_score")),
        "unique_corr_lag_bits": as_int(unique_corr.get("best_shift_bits")),
        "unique_corr_score": as_float(unique_corr.get("best_score")),
        "probe_summary_path": str(summary_path),
    }


def classify_lag(value: int | None) -> str:
    if value is None:
        return "no-probe-lag"
    if value < 0:
        return "negative"
    if value == 0:
        return "zero"
    return "positive"


def plot_phase_lag_atlas(rows: list[dict], path: Path) -> None:
    x_values = [row["band_start_bits"] / 1_000_000 for row in rows]
    y_values = [
        row["recommended_lag_bits"] / 1_000_000
        if row["recommended_lag_bits"] is not None
        else None
        for row in rows
    ]
    fig, ax = plt.subplots(figsize=(11.5, 5.8), dpi=180)
    fig.patch.set_facecolor("#fbfaf5")
    ax.set_facecolor("#fbfaf5")
    ax.axhline(0, color="#6b7280", linewidth=1.2, alpha=0.75)

    valid_x = [x for x, y in zip(x_values, y_values) if y is not None]
    valid_y = [y for y in y_values if y is not None]
    ax.plot(valid_x, valid_y, color="#0f766e", linewidth=2.2, alpha=0.85, zorder=2)

    colors = {
        "negative": "#0f766e",
        "zero": "#315f72",
        "positive": "#b45f2a",
        "no-probe-lag": "#7c7568",
    }
    markers = {
        "negative": "o",
        "zero": "D",
        "positive": "o",
        "no-probe-lag": "x",
    }
    for x, y, row in zip(x_values, y_values, rows):
        cls = row["lag_class"]
        plot_y = 0 if y is None else y
        ax.scatter(
            [x],
            [plot_y],
            s=110,
            marker=markers[cls],
            color=colors[cls],
            edgecolor="#143b36" if cls != "no-probe-lag" else colors[cls],
            linewidth=1.5,
            zorder=3,
        )
        ax.text(
            x,
            plot_y + (2.6 if plot_y >= 0 else -3.6),
            lag_label(row["recommended_lag_bits"]),
            ha="center",
            va="bottom" if plot_y >= 0 else "top",
            fontsize=8.5,
            color="#143b36",
            fontweight="bold",
        )

    ax.set_title("N2-10 Phase-Lag Atlas Scout", fontsize=15, fontweight="bold")
    ax.set_xlabel("anchor band start (M bits)", fontsize=11, fontweight="bold")
    ax.set_ylabel("recommended local lag (M bits)", fontsize=11, fontweight="bold")
    ax.grid(axis="y", color="#dbe3ef", linewidth=1, alpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    if x_values:
        ax.set_xlim(min(x_values) - 3, max(x_values) + 3)

    for cls, label in [
        ("negative", "negative lag"),
        ("positive", "positive lag"),
        ("zero", "zero lag"),
        ("no-probe-lag", "no probe lag"),
    ]:
        ax.scatter([], [], s=90, marker=markers[cls], color=colors[cls], label=label)
    ax.legend(loc="best", frameon=True, facecolor="#fbfaf5", edgecolor="#d8d1c2")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.88)
    fig.savefig(path)
    plt.close(fig)


def render_report(payload: dict) -> str:
    selection = payload["selection"]
    rows = payload["rows"]
    lines = [
        "# Phase 2 N2-10 Phase-Lag Atlas Scout",
        "",
        "## Selection",
        "",
        f"- Anchor variant: {selection['anchor_variant']}",
        f"- Candidate variant: {selection['candidate_variant']}",
        f"- Top patterns: {selection['top_patterns']}",
        f"- Pattern scale: {selection['pattern_scale']}",
        f"- Pattern selection: {selection['pattern_selection']}",
        f"- Probe backward bits: {selection['probe_backward_bits']}",
        f"- Probe forward bits: {selection['probe_forward_bits']}",
        f"- Band starts: {', '.join(str(value) for value in selection['band_starts'])}",
        "",
        "## Phase-Lag Table",
        "",
        "| Band | Lag | Class | Shell | Full shell lag | Hit corr lag | Unique corr lag |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['band_label']}` | {lag_label(row['recommended_lag_bits'])} | "
            f"{row['lag_class']} | {fmt_int(row['anchor_shell_pattern_count'])} | "
            f"{lag_label(row['first_full_shell_lag_bits'])} | "
            f"{lag_label(row['hit_corr_lag_bits'])} | "
            f"{lag_label(row['unique_corr_lag_bits'])} |"
        )
    lines.extend(
        [
            "",
            "## Reading Boundary",
            "",
            "- This is a phase scout, not a null-validated child-routing result.",
            "- Lags found here are candidates for later observed-routing or full-null revalidation.",
            "- A negative lag means the candidate shell phase precedes the anchor band under this contract.",
            "- A visually smooth curve should not be called a wave until it survives predefined follow-up tests.",
        ]
    )
    return "\n".join(lines)


def render_console_table(rows: list[dict]) -> str:
    lines = [
        "Phase 2 N2-10 phase-lag atlas scout",
        "-" * 96,
        f"{'band':<16} {'lag':>8} {'class':<14} {'shell':>7} {'hitCorr':>9} {'uniqCorr':>9}",
    ]
    for row in rows:
        lines.append(
            f"{short_band(row['band_label']):<16} "
            f"{lag_label(row['recommended_lag_bits']):>8} "
            f"{row['lag_class']:<14} "
            f"{fmt_int(row['anchor_shell_pattern_count']):>7} "
            f"{lag_label(row['hit_corr_lag_bits']):>9} "
            f"{lag_label(row['unique_corr_lag_bits']):>9}"
        )
    return "\n".join(lines)


def build_run_slug(selection: dict, args, band_starts: list[int], timestamp: str) -> str:
    return (
        f"phase2-phase-lag-atlas__anchor-{selection['anchor_variant']}"
        f"__cand-{selection['candidate_variant']}__top-{args.top_patterns}"
        f"__bands-{compact_int(min(band_starts))}-{compact_int(max(band_starts))}x{len(band_starts)}"
        f"__probe-back-{compact_int(args.probe_backward_bits)}__fwd-{compact_int(args.probe_forward_bits)}"
        f"__{timestamp}"
    )


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def copy_latest_artifacts(run_dir: Path, output_root: Path) -> None:
    for name in ("phase_lag_atlas.csv", "report.md", "summary.json", "phase_lag_atlas.png"):
        source = run_dir / name
        if not source.exists():
            continue
        suffix = source.suffix
        stem = "phase_lag_atlas_latest" if name != "report.md" else "report_latest"
        if name == "summary.json":
            stem = "summary_latest"
        shutil.copyfile(source, output_root / f"{stem}{suffix}")


def as_int(value) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def as_float(value) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def fmt_int(value: int | None) -> str:
    return "-" if value is None else str(int(value))


def lag_label(value: int | None) -> str:
    if value is None:
        return "-"
    if value == 0:
        return "0"
    if value % 1_000_000 == 0:
        return f"{value // 1_000_000}M"
    if value % 500_000 == 0:
        return f"{value / 1_000_000:g}M"
    if value % 1_000 == 0:
        return f"{value // 1_000}K"
    return str(value)


def short_band(label: str) -> str:
    return label.replace("band-", "")


if __name__ == "__main__":
    raise SystemExit(main())
