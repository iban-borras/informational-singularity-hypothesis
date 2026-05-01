#!/usr/bin/env python3
"""
HSI v2 Phase 2 Phase-Lag Tri-Estimator Audit

N2-10b derived readout. It does not rerun the expensive probe. It reads an
existing N2-10 phase-lag atlas summary and compares the three lag estimators:
first full shell, hit-profile correlation, and unique-profile correlation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


DEFAULT_INPUT_SUMMARY = "results/hsi_v2/phase2/phase_lag_atlas/summary_latest.json"
DEFAULT_OUTPUT_DIR = "results/hsi_v2/phase2/phase_lag_tri_estimator"

ESTIMATORS = (
    ("first_full", "first_full_shell_lag_bits"),
    ("hit_corr", "hit_corr_lag_bits"),
    ("unique_corr", "unique_corr_lag_bits"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit N2-10 phase-lag atlas rows by comparing first-full, "
            "hit-correlation, and unique-correlation lag estimators."
        )
    )
    parser.add_argument("--input-summary", type=str, default=DEFAULT_INPUT_SUMMARY)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--tolerance-bits",
        type=int,
        default=1_000_000,
        help="Maximum absolute lag difference for estimator agreement.",
    )
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.tolerance_bits < 0:
        parser.error("--tolerance-bits must be non-negative.")

    root = Path(__file__).parent.resolve()
    input_summary = resolve_path(args.input_summary, root)
    output_root = resolve_path(args.output_dir, root)
    output_root.mkdir(parents=True, exist_ok=True)

    payload = load_json(input_summary)
    rows = [audit_row(row, args.tolerance_bits) for row in payload.get("rows", [])]

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_slug = build_run_slug(payload, args.tolerance_bits, timestamp)
    run_dir = output_root / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)

    csv_path = run_dir / "phase_lag_tri_estimator.csv"
    report_path = run_dir / "report.md"
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "manifest.json"
    png_path = run_dir / "phase_lag_tri_estimator.png"

    write_csv(rows, csv_path)
    if not args.no_plot:
        plot_audit(rows, png_path)
    else:
        png_path = None

    generated_at = datetime.now().isoformat(timespec="seconds")
    summary = {
        "generated_at": generated_at,
        "source_summary": str(input_summary),
        "tolerance_bits": args.tolerance_bits,
        "source_selection": payload.get("selection", {}),
        "counts": confidence_counts(rows),
        "rows": rows,
        "outputs": {
            "csv": str(csv_path),
            "report": str(report_path),
            "summary": str(summary_path),
            "manifest": str(manifest_path),
            "png": str(png_path) if png_path else None,
        },
    }
    manifest = {
        "generated_at": generated_at,
        "script": "hsi_v2_phase2_phase_lag_tri_estimator.py",
        "arguments": vars(args),
        "input_summary": str(input_summary),
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


def resolve_path(value: str, root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def audit_row(row: dict, tolerance_bits: int) -> dict:
    lags = {name: as_int(row.get(key)) for name, key in ESTIMATORS}
    agreements = agreement_pairs(lags, tolerance_bits)
    confidence, consensus_lag, reason = classify_row(lags, agreements, tolerance_bits)
    spread = lag_spread(lags)
    return {
        "band_index": as_int(row.get("band_index")),
        "band_label": row.get("band_label", ""),
        "band_start_bits": as_int(row.get("band_start_bits")),
        "band_end_bits": as_int(row.get("band_end_bits")),
        "first_full_lag_bits": lags["first_full"],
        "hit_corr_lag_bits": lags["hit_corr"],
        "hit_corr_score": as_float(row.get("hit_corr_score")),
        "unique_corr_lag_bits": lags["unique_corr"],
        "unique_corr_score": as_float(row.get("unique_corr_score")),
        "lag_spread_bits": spread,
        "agreement_pairs": ",".join(agreements) if agreements else "-",
        "consensus_lag_bits": consensus_lag,
        "confidence": confidence,
        "reading": reason,
        "anchor_shell_pattern_count": as_int(row.get("anchor_shell_pattern_count")),
        "anchor_core_pattern_count": as_int(row.get("anchor_core_pattern_count")),
    }


def agreement_pairs(lags: dict[str, int | None], tolerance_bits: int) -> list[str]:
    pairs = []
    names = list(lags.keys())
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            if agrees(lags[left], lags[right], tolerance_bits):
                pairs.append(f"{left}+{right}")
    return pairs


def classify_row(
    lags: dict[str, int | None], agreements: list[str], tolerance_bits: int
) -> tuple[str, int | None, str]:
    values = [value for value in lags.values() if value is not None]
    if not values:
        return "missing", None, "no lag estimator was available"

    if len(values) == 3 and max(values) - min(values) <= tolerance_bits:
        values.sort()
        return "strong", values[len(values) // 2], "all three estimators agree"

    if agreements:
        consensus = consensus_from_pairs(lags, agreements)
        return (
            "transition",
            consensus,
            f"partial estimator agreement: {', '.join(agreements)}",
        )

    return "ambiguous", None, "no estimator pair agrees within tolerance"


def consensus_from_pairs(lags: dict[str, int | None], pairs: list[str]) -> int | None:
    selected: list[int] = []
    for pair in pairs:
        for name in pair.split("+"):
            value = lags.get(name)
            if value is not None:
                selected.append(value)
    if not selected:
        return None
    selected.sort()
    return selected[len(selected) // 2]


def agrees(left: int | None, right: int | None, tolerance_bits: int) -> bool:
    return left is not None and right is not None and abs(left - right) <= tolerance_bits


def lag_spread(lags: dict[str, int | None]) -> int | None:
    values = [value for value in lags.values() if value is not None]
    if len(values) < 2:
        return None
    return max(values) - min(values)


def confidence_counts(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = row["confidence"]
        counts[key] = counts.get(key, 0) + 1
    return counts


def plot_audit(rows: list[dict], path: Path) -> None:
    x_values = [row["band_start_bits"] / 1_000_000 for row in rows]
    fig, ax = plt.subplots(figsize=(12.2, 6.4), dpi=180)
    fig.patch.set_facecolor("#fbfaf5")
    ax.set_facecolor("#fbfaf5")

    shade = {
        "strong": "#dff3e7",
        "transition": "#fff0cf",
        "ambiguous": "#ece7de",
        "missing": "#f2d7d5",
    }
    for row in rows:
        left = row["band_start_bits"] / 1_000_000
        right = row["band_end_bits"] / 1_000_000
        ax.axvspan(
            left,
            right,
            color=shade.get(row["confidence"], "#ece7de"),
            alpha=0.45,
            linewidth=0,
            zorder=0,
        )

    series = [
        ("first_full_lag_bits", "first full shell", "#0f766e", "o", "-"),
        ("hit_corr_lag_bits", "hit-profile corr", "#b45f2a", "s", "--"),
        ("unique_corr_lag_bits", "unique-profile corr", "#315f72", "^", ":"),
    ]
    for key, label, color, marker, linestyle in series:
        y_values = [bits_to_m(row.get(key)) for row in rows]
        ax.plot(
            x_values,
            y_values,
            label=label,
            color=color,
            marker=marker,
            linewidth=2.0,
            markersize=6.5,
            linestyle=linestyle,
            zorder=2,
        )

    ax.axhline(0, color="#6b7280", linewidth=1.2, alpha=0.8)
    ax.set_title("N2-10b Phase-Lag Tri-Estimator Audit", fontsize=15, fontweight="bold")
    ax.set_xlabel("anchor band start (M bits)", fontsize=11, fontweight="bold")
    ax.set_ylabel("lag estimate (M bits)", fontsize=11, fontweight="bold")
    ax.grid(axis="y", color="#dbe3ef", linewidth=1, alpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    line_legend = ax.legend(
        loc="upper right",
        frameon=True,
        facecolor="#fbfaf5",
        edgecolor="#d8d1c2",
    )
    ax.add_artist(line_legend)
    confidence_handles = [
        Patch(facecolor=shade["strong"], edgecolor="none", alpha=0.7, label="strong"),
        Patch(
            facecolor=shade["transition"],
            edgecolor="none",
            alpha=0.7,
            label="transition",
        ),
        Patch(
            facecolor=shade["ambiguous"],
            edgecolor="none",
            alpha=0.7,
            label="ambiguous",
        ),
    ]
    ax.legend(
        handles=confidence_handles,
        loc="lower right",
        frameon=True,
        facecolor="#fbfaf5",
        edgecolor="#d8d1c2",
        title="confidence",
    )

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.14, top=0.88)
    fig.savefig(path)
    plt.close(fig)


def render_report(payload: dict) -> str:
    rows = payload["rows"]
    selection = payload.get("source_selection", {})
    lines = [
        "# Phase 2 N2-10b Phase-Lag Tri-Estimator Audit",
        "",
        "## Scope",
        "",
        "- This is a derived audit over an existing N2-10 phase-lag atlas summary.",
        "- It does not rerun the probe and does not validate child-routing against nulls.",
        "- Its purpose is to separate stable phase candidates from transition or ambiguous bands.",
        "",
        "## Selection",
        "",
        f"- Source summary: `{payload['source_summary']}`",
        f"- Anchor variant: {selection.get('anchor_variant', '-')}",
        f"- Candidate variant: {selection.get('candidate_variant', '-')}",
        f"- Top patterns: {selection.get('top_patterns', '-')}",
        f"- Agreement tolerance: {lag_label(payload['tolerance_bits'])}",
        f"- Confidence counts: {payload['counts']}",
        "",
        "## Tri-Estimator Table",
        "",
        "| Band | First full | Hit corr | Unique corr | Spread | Confidence | Reading |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['band_label']}` | {lag_label(row['first_full_lag_bits'])} | "
            f"{lag_label(row['hit_corr_lag_bits'])} | "
            f"{lag_label(row['unique_corr_lag_bits'])} | "
            f"{lag_label(row['lag_spread_bits'])} | "
            f"{row['confidence']} | {row['reading']} |"
        )
    lines.extend(
        [
            "",
            "## Reading Boundary",
            "",
            "- Strong bands are candidates for targeted child-routing validation.",
            "- Transition bands should be treated as phase-boundary candidates, not as failures.",
            "- Ambiguous bands are not evidence for a stable local lag under this contract.",
            "- This audit keeps the earlier caution intact: a smooth visual profile is not yet a wave claim.",
        ]
    )
    return "\n".join(lines)


def render_console_table(rows: list[dict]) -> str:
    lines = [
        "Phase 2 N2-10b phase-lag tri-estimator audit",
        "-" * 118,
        (
            f"{'band':<16} {'first':>9} {'hit':>9} {'unique':>9} "
            f"{'spread':>9} {'conf':<11} {'pairs':<30}"
        ),
    ]
    for row in rows:
        lines.append(
            f"{short_band(row['band_label']):<16} "
            f"{lag_label(row['first_full_lag_bits']):>9} "
            f"{lag_label(row['hit_corr_lag_bits']):>9} "
            f"{lag_label(row['unique_corr_lag_bits']):>9} "
            f"{lag_label(row['lag_spread_bits']):>9} "
            f"{row['confidence']:<11} "
            f"{row['agreement_pairs']:<30}"
        )
    return "\n".join(lines)


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def copy_latest_artifacts(run_dir: Path, output_root: Path) -> None:
    mapping = {
        "phase_lag_tri_estimator.csv": "phase_lag_tri_estimator_latest.csv",
        "phase_lag_tri_estimator.png": "phase_lag_tri_estimator_latest.png",
        "summary.json": "summary_latest.json",
        "report.md": "report_latest.md",
    }
    for source_name, target_name in mapping.items():
        source = run_dir / source_name
        if source.exists():
            shutil.copyfile(source, output_root / target_name)


def build_run_slug(payload: dict, tolerance_bits: int, timestamp: str) -> str:
    selection = payload.get("selection", {})
    starts = selection.get("band_starts") or []
    if starts:
        band_part = f"{compact_int(min(starts))}-{compact_int(max(starts))}x{len(starts)}"
    else:
        band_part = "unknown-bands"
    return (
        "phase2-phase-lag-tri-estimator"
        f"__anchor-{selection.get('anchor_variant', 'unknown')}"
        f"__cand-{selection.get('candidate_variant', 'unknown')}"
        f"__top-{selection.get('top_patterns', 'unknown')}"
        f"__bands-{band_part}"
        f"__tol-{compact_int(tolerance_bits)}"
        f"__{timestamp}"
    )


def as_int(value) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def as_float(value) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def bits_to_m(value: int | None) -> float | None:
    if value is None:
        return math.nan
    return value / 1_000_000


def lag_label(value: int | None) -> str:
    if value is None:
        return "-"
    if value == 0:
        return "0"
    sign = "-" if value < 0 else ""
    abs_value = abs(value)
    if abs_value % 1_000_000 == 0:
        return f"{sign}{abs_value // 1_000_000}M"
    if abs_value % 500_000 == 0:
        return f"{sign}{abs_value / 1_000_000:g}M"
    if abs_value % 1_000 == 0:
        return f"{sign}{abs_value // 1_000}K"
    return str(value)


def compact_int(value: int) -> str:
    sign = "-" if value < 0 else ""
    abs_value = abs(value)
    if abs_value % 1_000_000 == 0:
        return f"{sign}{abs_value // 1_000_000}M"
    if abs_value % 1_000 == 0:
        return f"{sign}{abs_value // 1_000}K"
    return str(value)


def short_band(label: str) -> str:
    return label.replace("band-", "")


if __name__ == "__main__":
    raise SystemExit(main())
