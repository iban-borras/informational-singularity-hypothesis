#!/usr/bin/env python3
"""
HSI v2 Phase 2 Deficit-Jitter Plane

Build a deterministic visual readout from parent-shell atlas sensitivity runs.
The plot compares pooled net shell deficit against jitter-gap mass, both
normalized by the Markov-1 collapse mass at the same top-k and lag.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from v2.common.cli import resolve_dir
from v2.common.naming import compact_int


DEFAULT_SENSITIVITY_ROOT = "results/hsi_v2/phase2/parent_shell_atlas_sensitivity"
DEFAULT_OUTPUT_DIR = "results/hsi_v2/phase2/visual_readouts"

SOURCE_STYLE = {
    "B observed": {
        "color": "#0f766e",
        "edge": "#064e3b",
        "label": "B observed",
        "zorder": 5,
        "size": 90,
    },
    "E observed": {
        "color": "#111827",
        "edge": "#111827",
        "label": "E reference",
        "zorder": 4,
        "size": 75,
    },
    "Markov-1 null": {
        "color": "#2563eb",
        "edge": "#1e3a8a",
        "label": "Markov-1 null",
        "zorder": 3,
        "size": 80,
    },
    "Matched-LZ null": {
        "color": "#64748b",
        "edge": "#334155",
        "label": "Matched-LZ null seeds",
        "zorder": 2,
        "size": 48,
    },
}

TOP_MARKERS = {
    64: "o",
    128: "s",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render the Phase 2 deficit-jitter plane from a parent-shell atlas sensitivity run."
    )
    parser.add_argument(
        "--sensitivity-run",
        type=str,
        default="",
        help="Path to one phase2-parent-shell-atlas-sensitivity__... run. Defaults to latest stage-all run.",
    )
    parser.add_argument(
        "--sensitivity-root",
        type=str,
        default=DEFAULT_SENSITIVITY_ROOT,
        help="Root used to discover the latest sensitivity run when --sensitivity-run is omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where visual readout artifacts will be written.",
    )
    parser.add_argument(
        "--top-patterns",
        type=str,
        default="64,128",
        help="Comma-separated top-k values to include. Defaults to the informative regimes.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        top_values = parse_int_list(args.top_patterns)
    except ValueError as exc:
        parser.error(str(exc))

    sensitivity_run = resolve_sensitivity_run(args)
    concentration_path = sensitivity_run / "concentration_sensitivity.csv"
    structure_path = sensitivity_run / "structure_sensitivity.csv"
    if not concentration_path.is_file():
        parser.error(f"Missing concentration_sensitivity.csv: {concentration_path}")
    if not structure_path.is_file():
        parser.error(f"Missing structure_sensitivity.csv: {structure_path}")

    output_root = resolve_dir(args.output_dir, anchor_file=__file__)
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_slug = (
        f"phase2-deficit-jitter-plane__top-{'-'.join(str(value) for value in top_values)}"
        f"__src-{sensitivity_run.name.split('__')[-1]}__{timestamp}"
    )
    run_dir = output_root / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)

    concentration_rows = read_csv(concentration_path)
    structure_rows = read_csv(structure_path)
    plot_rows = build_plot_rows(concentration_rows, top_values=top_values)
    if not plot_rows:
        parser.error("No plottable concentration rows remain after top-k filtering.")

    generated_at = datetime.now().isoformat(timespec="seconds")
    csv_path = run_dir / "deficit_jitter_points.csv"
    json_path = run_dir / "summary.json"
    report_path = run_dir / "report.md"
    svg_path = run_dir / "deficit_jitter_plane.svg"
    png_path = run_dir / "deficit_jitter_plane.png"
    manifest_path = run_dir / "manifest.json"

    write_csv(plot_rows, csv_path)
    render_deficit_jitter_plot(plot_rows, svg_path=svg_path, png_path=png_path)

    summary_payload = {
        "generated_at": generated_at,
        "selection": {
            "sensitivity_run": str(sensitivity_run),
            "concentration_path": str(concentration_path),
            "structure_path": str(structure_path),
            "top_patterns": top_values,
            "normalization": "net_deficit_mass_sum and jitter_gap_mass divided by Markov-1 net_deficit_mass_sum at the same top-k and lag",
        },
        "structure_summary": summarize_structure(structure_rows, top_values=top_values),
        "class_summary": summarize_plot_rows(plot_rows),
        "plot_rows": plot_rows,
    }
    manifest_payload = {
        "generated_at": generated_at,
        "run_slug": run_slug,
        "script": "hsi_v2_phase2_deficit_jitter_plane.py",
        "cwd": str(Path.cwd()),
        "inputs": {
            "sensitivity_run": str(sensitivity_run),
            "concentration_csv": str(concentration_path),
            "structure_csv": str(structure_path),
        },
        "outputs": {
            "summary": str(json_path),
            "report": str(report_path),
            "points_csv": str(csv_path),
            "svg": str(svg_path),
            "png": str(png_path),
            "manifest": str(manifest_path),
        },
        "arguments": vars(args),
    }

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(render_report(summary_payload, svg_path.name, png_path.name) + "\n")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, indent=2)

    if not args.quiet:
        print(render_console_summary(summary_payload))
        print("")
        print(f"Saved summary to: {json_path}")
        print(f"Saved report to: {report_path}")
        print(f"Saved CSV points to: {csv_path}")
        print(f"Saved SVG figure to: {svg_path}")
        print(f"Saved PNG figure to: {png_path}")
        print(f"Saved manifest to: {manifest_path}")

    return 0


def resolve_sensitivity_run(args) -> Path:
    if args.sensitivity_run.strip():
        path = resolve_dir(args.sensitivity_run, anchor_file=__file__)
        if not path.is_dir():
            raise SystemExit(f"Sensitivity run not found: {path}")
        return path

    root = resolve_dir(args.sensitivity_root, anchor_file=__file__)
    if not root.is_dir():
        raise SystemExit(f"Sensitivity root not found: {root}")
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir()
        and path.name.startswith("phase2-parent-shell-atlas-sensitivity__")
        and "__stage-all__" in path.name
        and (path / "concentration_sensitivity.csv").is_file()
    ]
    if not candidates:
        raise SystemExit(f"No complete stage-all sensitivity runs found under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def build_plot_rows(rows: list[dict], *, top_values: list[int]) -> list[dict]:
    selected_tops = set(top_values)
    denominators: dict[tuple[int, int], float] = {}
    for row in rows:
        top = int(row["top_patterns"])
        lag = int(row["lag_bits"])
        if top not in selected_tops:
            continue
        if row.get("null_model") == "markov1":
            denominators[(top, lag)] = float(row["net_deficit_mass_sum"])

    plot_rows = []
    for row in rows:
        top = int(row["top_patterns"])
        lag = int(row["lag_bits"])
        if top not in selected_tops:
            continue
        collapse_mass = denominators.get((top, lag), 0.0)
        if collapse_mass <= 0:
            continue
        source_class = classify_source(row)
        net = float_or_zero(row.get("net_deficit_mass_sum"))
        gross = float_or_zero(row.get("gross_loss_pressure_sum"))
        jitter = float_or_zero(row.get("jitter_gap_mass"))
        plot_rows.append(
            {
                "top_patterns": top,
                "lag_bits": lag,
                "lag_label": format_lag(lag),
                "variant": row["variant"],
                "source_label": row["source_label"],
                "source_class": source_class,
                "null_model": row.get("null_model") or "",
                "null_seed": row.get("null_seed") or "",
                "net_deficit_mass_sum": net,
                "gross_loss_pressure_sum": gross,
                "jitter_gap_mass": jitter,
                "markov1_collapse_mass": collapse_mass,
                "net_deficit_norm": net / collapse_mass,
                "jitter_gap_norm": jitter / collapse_mass,
                "jitter_share_of_gross": jitter / gross if gross > 0 else None,
                "dominant_subtype_label": row.get("dominant_subtype_label") or "",
                "dominant_subtype_share": float_or_none(row.get("dominant_subtype_share")),
            }
        )
    return plot_rows


def render_deficit_jitter_plot(plot_rows: list[dict], *, svg_path: Path, png_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(10.8, 7.2), dpi=180)
    fig.patch.set_facecolor("#f8f4ec")
    ax.set_facecolor("#fffdf7")

    ax.axvspan(0.0, 0.22, ymin=0.0, ymax=1.0, color="#d1fae5", alpha=0.22, zorder=0)
    ax.axvspan(0.55, 1.05, ymin=0.0, ymax=1.0, color="#dbeafe", alpha=0.18, zorder=0)
    ax.text(0.035, 0.345, "redistribution\nregime", color="#047857", fontsize=10, weight="bold")
    ax.text(0.71, 0.345, "net-collapse\nregime", color="#1d4ed8", fontsize=10, weight="bold")

    for source_class in ["Matched-LZ null", "Markov-1 null", "E observed", "B observed"]:
        rows = [row for row in plot_rows if row["source_class"] == source_class]
        if not rows:
            continue
        style = SOURCE_STYLE[source_class]
        for top in sorted({row["top_patterns"] for row in rows}):
            top_rows = [row for row in rows if row["top_patterns"] == top]
            top_rows.sort(key=lambda row: row["lag_bits"])
            marker = TOP_MARKERS.get(top, "D")
            alpha = 0.42 if source_class == "Matched-LZ null" else 0.9
            if source_class == "B observed" and len(top_rows) > 1:
                ax.plot(
                    [row["net_deficit_norm"] for row in top_rows],
                    [row["jitter_gap_norm"] for row in top_rows],
                    color=style["edge"],
                    linewidth=1.8,
                    alpha=0.78,
                    zorder=style["zorder"] - 0.1,
                )
            ax.scatter(
                [row["net_deficit_norm"] for row in top_rows],
                [row["jitter_gap_norm"] for row in top_rows],
                s=style["size"],
                marker=marker,
                c=style["color"],
                edgecolors=style["edge"],
                linewidths=1.2,
                alpha=alpha,
                zorder=style["zorder"],
            )

    annotate_representatives(ax, plot_rows)

    ax.set_title("Phase 2 Deficit-Jitter Plane", loc="left", fontsize=18, weight="bold", color="#172033")
    ax.set_xlabel("pooled net shell deficit / Markov-1 collapse mass", fontsize=11, weight="bold")
    ax.set_ylabel("jitter-gap mass / Markov-1 collapse mass", fontsize=11, weight="bold")
    ax.set_xlim(-0.03, 1.06)
    ax.set_ylim(-0.02, 0.40)
    ax.grid(True, color="#dbe4ef", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94a3b8")
    ax.spines["bottom"].set_color("#94a3b8")

    source_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=SOURCE_STYLE[name]["color"], markeredgecolor=SOURCE_STYLE[name]["edge"], markersize=8, label=SOURCE_STYLE[name]["label"])
        for name in ["B observed", "E observed", "Markov-1 null", "Matched-LZ null"]
    ]
    top_handles = [
        Line2D([0], [0], marker=TOP_MARKERS.get(top, "D"), color="#334155", markerfacecolor="#f8fafc", linestyle="None", markersize=8, label=f"top={top}")
        for top in sorted({row["top_patterns"] for row in plot_rows})
    ]
    legend1 = ax.legend(handles=source_handles, loc="upper right", frameon=True, facecolor="#fffdf7", edgecolor="#cbd5e1")
    ax.add_artist(legend1)
    ax.legend(handles=top_handles, loc="center right", bbox_to_anchor=(0.995, 0.55), frameon=True, facecolor="#fffdf7", edgecolor="#cbd5e1")

    fig.text(
        0.12,
        0.02,
        "Deterministic readout from parent_shell_atlas_sensitivity: axes normalized per top-k/lag by Markov-1 net collapse.",
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(svg_path, format="svg")
    fig.savefig(png_path, format="png")
    plt.close(fig)


def annotate_representatives(ax, plot_rows: list[dict]) -> None:
    b_rows = [row for row in plot_rows if row["source_class"] == "B observed"]
    for top in sorted({row["top_patterns"] for row in b_rows}):
        top_rows = sorted(
            [row for row in b_rows if row["top_patterns"] == top],
            key=lambda row: row["jitter_gap_norm"],
        )
        if not top_rows:
            continue
        low = top_rows[0]
        high = top_rows[-1]
        ax.annotate(
            f"B top={top}\n{low['lag_label']}-{high['lag_label']}",
            (high["net_deficit_norm"], high["jitter_gap_norm"]),
            xytext=(44 if top == 128 else 14, 10 if top == 64 else -4),
            textcoords="offset points",
            fontsize=8,
            color="#064e3b",
            weight="bold",
        )

    markov_rows = [row for row in plot_rows if row["source_class"] == "Markov-1 null"]
    if markov_rows:
        row = markov_rows[0]
        ax.annotate(
            "Markov-1\npure collapse",
            (row["net_deficit_norm"], row["jitter_gap_norm"]),
            xytext=(-80, 24),
            textcoords="offset points",
            fontsize=8,
            color="#1e3a8a",
            weight="bold",
        )

    matched_rows = [row for row in plot_rows if row["source_class"] == "Matched-LZ null"]
    if matched_rows:
        med_x = sorted(row["net_deficit_norm"] for row in matched_rows)[len(matched_rows) // 2]
        med_y = sorted(row["jitter_gap_norm"] for row in matched_rows)[len(matched_rows) // 2]
        ax.annotate(
            "matched-LZ seeds\nnet-deficit dominated",
            (med_x, med_y),
            xytext=(-8, 28),
            textcoords="offset points",
            fontsize=8,
            color="#334155",
            weight="bold",
        )


def summarize_plot_rows(plot_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[int, str], list[dict]] = {}
    for row in plot_rows:
        grouped.setdefault((row["top_patterns"], row["source_class"]), []).append(row)

    summary = []
    for (top, source_class), rows in sorted(grouped.items()):
        summary.append(
            {
                "top_patterns": top,
                "source_class": source_class,
                "n": len(rows),
                "net_deficit_norm_min": min(row["net_deficit_norm"] for row in rows),
                "net_deficit_norm_max": max(row["net_deficit_norm"] for row in rows),
                "jitter_gap_norm_min": min(row["jitter_gap_norm"] for row in rows),
                "jitter_gap_norm_max": max(row["jitter_gap_norm"] for row in rows),
                "jitter_share_of_gross_min": min_ignore_none(row["jitter_share_of_gross"] for row in rows),
                "jitter_share_of_gross_max": max_ignore_none(row["jitter_share_of_gross"] for row in rows),
            }
        )
    return summary


def summarize_structure(rows: list[dict], *, top_values: list[int]) -> list[dict]:
    selected_tops = set(top_values)
    summary = []
    for row in rows:
        top = int(row["top_patterns"])
        if top not in selected_tops:
            continue
        summary.append(
            {
                "top_patterns": top,
                "lag_bits": int(row["lag_bits"]),
                "shell_subtype_label": row["shell_subtype_label"],
                "shell_pattern_count": int(row["shell_pattern_count"]),
                "monitored_shell_pattern_count": int(row["monitored_shell_pattern_count"]),
                "shell_pattern_fraction": float(row["shell_pattern_fraction"]),
            }
        )
    return summary


def render_report(payload: dict, svg_name: str, png_name: str) -> str:
    selection = payload["selection"]
    lines = [
        "# Phase 2 Deficit-Jitter Plane",
        "",
        "## Selection",
        "",
        f"- Sensitivity run: `{selection['sensitivity_run']}`",
        f"- Top-k values: `{', '.join(str(value) for value in selection['top_patterns'])}`",
        f"- Normalization: `{selection['normalization']}`",
        "",
        "## Figure",
        "",
        f"![Deficit-Jitter Plane](./{png_name})",
        "",
        f"SVG version: `{svg_name}`",
        "",
        "## Reading",
        "",
        "- `E` sits at the origin by construction of the anchor reference.",
        "- `markov1` defines the pure collapse axis: high normalized net deficit, zero jitter gap.",
        "- `matched-LZ` stays net-deficit dominated across seeds.",
        "- observed `B` stays in the redistribution regime: low normalized net deficit with high normalized jitter gap.",
        "",
        "## Class Summary",
        "",
        "| Top-k | Source | n | net norm | jitter norm | jitter/gross |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["class_summary"]:
        lines.append(
            f"| {row['top_patterns']} | {row['source_class']} | {row['n']} | "
            f"{fmt_range(row['net_deficit_norm_min'], row['net_deficit_norm_max'])} | "
            f"{fmt_range(row['jitter_gap_norm_min'], row['jitter_gap_norm_max'])} | "
            f"{fmt_range(row['jitter_share_of_gross_min'], row['jitter_share_of_gross_max'])} |"
        )
    lines.extend(
        [
            "",
            "## Methodological Note",
            "",
            "This is a deterministic visual readout, not a new metric.",
            "The plotted coordinates are derived from the audited N2-06 sensitivity artifacts.",
            "The normalization removes the top-k mass-scale difference by dividing each point by the Markov-1 collapse mass at the same top-k and lag.",
        ]
    )
    return "\n".join(lines)


def render_console_summary(payload: dict) -> str:
    lines = [
        "Phase 2 Deficit-Jitter Plane",
        "-" * 104,
        f"{'top':>5} {'source':<18} {'n':>3} {'net_norm':>17} {'jitter_norm':>17} {'jitter/gross':>17}",
    ]
    for row in payload["class_summary"]:
        lines.append(
            f"{row['top_patterns']:>5} {row['source_class']:<18} {row['n']:>3} "
            f"{fmt_range(row['net_deficit_norm_min'], row['net_deficit_norm_max']):>17} "
            f"{fmt_range(row['jitter_gap_norm_min'], row['jitter_gap_norm_max']):>17} "
            f"{fmt_range(row['jitter_share_of_gross_min'], row['jitter_share_of_gross_max']):>17}"
        )
    return "\n".join(lines)


def classify_source(row: dict) -> str:
    if row["variant"] == "B" and row["source_kind"] == "observed":
        return "B observed"
    if row["variant"] == "E" and row["source_kind"] == "observed":
        return "E observed"
    if row.get("null_model") == "markov1":
        return "Markov-1 null"
    if row.get("null_model") == "matched-lz":
        return "Matched-LZ null"
    return row["variant"]


def parse_int_list(raw: str) -> list[int]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError("--top-patterns must contain positive integers.")
        values.append(value)
    if not values:
        raise ValueError("--top-patterns must contain at least one integer.")
    return values


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def float_or_zero(value) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def float_or_none(value) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def min_ignore_none(values) -> float | None:
    clean = [value for value in values if value is not None]
    return min(clean) if clean else None


def max_ignore_none(values) -> float | None:
    clean = [value for value in values if value is not None]
    return max(clean) if clean else None


def fmt_range(lo, hi) -> str:
    if lo is None or hi is None:
        return "-"
    if abs(float(lo) - float(hi)) < 1e-12:
        return f"{float(lo):.3f}"
    return f"{float(lo):.3f}-{float(hi):.3f}"


def format_lag(value: int) -> str:
    if value % 1_000_000 == 0:
        return f"{value // 1_000_000}M"
    if value % 100_000 == 0:
        return f"{value / 1_000_000:.1f}M"
    return compact_int(value)


if __name__ == "__main__":
    raise SystemExit(main())
