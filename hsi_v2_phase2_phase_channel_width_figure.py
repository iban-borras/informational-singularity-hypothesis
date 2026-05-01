#!/usr/bin/env python3
"""Render the HSI v2 Phase 2 N2-12 phase-channel width figure."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from v2.common.cli import resolve_dir
from v2.common.naming import compact_int


DEFAULT_RESPONSE_ROOT = "results/hsi_v2/phase2/phase_channel_width_n2_12_full_reval"
DEFAULT_OUTPUT_DIR = "results/hsi_v2/phase2/visual_readouts"
BOUNDARY_BAND_NOTES = {"729M-738M": "top-k-sensitive boundary"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a figure from an N2-12 phase-channel width revalidation run."
    )
    parser.add_argument(
        "--lag-response-run",
        default="",
        help="Path to one N2-12 full-revalidation lag-response run. Defaults to latest.",
    )
    parser.add_argument("--lag-response-root", default=DEFAULT_RESPONSE_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--b-threshold", type=float, default=0.90)
    parser.add_argument("--margin-threshold", type=float, default=0.30)
    parser.add_argument("--write-latest", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    response_run = resolve_response_run(args)
    response_csv = response_run / "lag_response.csv"
    if not response_csv.is_file():
        raise SystemExit(f"Missing lag_response.csv: {response_csv}")

    rows = build_rows(read_csv(response_csv))
    if not rows:
        raise SystemExit("No plottable rows found.")
    bands = ordered_bands(rows)
    lags = ordered_lags(rows)
    envelopes = build_envelopes(
        rows,
        b_threshold=args.b_threshold,
        margin_threshold=args.margin_threshold,
    )
    best_by_band = build_best_by_band(rows)

    output_root = resolve_dir(args.output_dir, anchor_file=__file__)
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    source_stamp = response_run.name.split("__")[-1]
    run_dir = output_root / f"phase2-phase-channel-width-figure__src-{source_stamp}__{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    png_path = run_dir / "phase_channel_width.png"
    svg_path = run_dir / "phase_channel_width.svg"
    points_csv = run_dir / "phase_channel_width_points.csv"
    envelopes_csv = run_dir / "phase_channel_width_envelopes.csv"
    summary_path = run_dir / "summary.json"
    report_path = run_dir / "report.md"
    manifest_path = run_dir / "manifest.json"

    render_figure(
        rows,
        envelopes,
        bands=bands,
        lags=lags,
        png_path=png_path,
        svg_path=svg_path,
        b_threshold=args.b_threshold,
        margin_threshold=args.margin_threshold,
    )
    write_csv(points_csv, rows)
    write_csv(envelopes_csv, envelopes)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "selection": {
            "lag_response_run": str(response_run),
            "lag_response_csv": str(response_csv),
            "b_threshold": args.b_threshold,
            "margin_threshold": args.margin_threshold,
            "reading": (
                "N2-12 phase-channel width: sampled lag cells where observed B "
                "retains child destination against the strongest matched-LZ seed."
            ),
        },
        "best_by_band": best_by_band,
        "channel_envelopes": envelopes,
        "plot_rows": rows,
    }
    manifest = {
        "generated_at": summary["generated_at"],
        "script": Path(__file__).name,
        "inputs": {
            "lag_response_run": str(response_run),
            "lag_response_csv": str(response_csv),
        },
        "outputs": {
            "summary": str(summary_path),
            "report": str(report_path),
            "points_csv": str(points_csv),
            "envelopes_csv": str(envelopes_csv),
            "png": str(png_path),
            "svg": str(svg_path),
            "manifest": str(manifest_path),
        },
        "arguments": vars(args),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_path.write_text(render_report(summary, png_path.name, svg_path.name), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.write_latest:
        latest_png = output_root / "phase_channel_width_latest.png"
        latest_svg = output_root / "phase_channel_width_latest.svg"
        shutil.copyfile(png_path, latest_png)
        shutil.copyfile(svg_path, latest_svg)

    if not args.quiet:
        print("Phase 2 N2-12 phase-channel width figure")
        print("-" * 72)
        for item in best_by_band:
            print(
                f"{item['band']:<10} best_margin={item['best_margin_lag']:>8} "
                f"B={item['best_margin_B_retention']:.4f} "
                f"margin={item['best_margin']:+.4f}"
            )
        print("")
        print(f"Saved summary to: {summary_path}")
        print(f"Saved report to: {report_path}")
        print(f"Saved PNG figure to: {png_path}")
        print(f"Saved SVG figure to: {svg_path}")
        print(f"Saved manifest to: {manifest_path}")
    return 0


def resolve_response_run(args: argparse.Namespace) -> Path:
    if args.lag_response_run.strip():
        path = resolve_dir(args.lag_response_run, anchor_file=__file__)
        if not path.is_dir():
            raise SystemExit(f"Lag-response run not found: {path}")
        return path

    root = resolve_dir(args.lag_response_root, anchor_file=__file__)
    if not root.is_dir():
        raise SystemExit(f"Lag-response root not found: {root}")
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir()
        and path.name.startswith("phase2-child-routing-lag-response__")
        and (path / "lag_response.csv").is_file()
    ]
    if not candidates:
        raise SystemExit(f"No complete lag-response runs found under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_rows(raw_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in raw_rows:
        lag_bits = int(row["lag_bits"])
        rows.append(
            {
                "band": row["band"],
                "lag": row["lag"],
                "lag_bits": lag_bits,
                "channel": "negative" if lag_bits < 0 else "positive",
                "B_retention": to_float(row["B_retention"]),
                "markov1_retention": to_float(row["markov1_retention"]),
                "matched_lz_min": to_float(row["matched_lz_min"]),
                "matched_lz_avg": to_float(row["matched_lz_avg"]),
                "matched_lz_max": to_float(row["matched_lz_max"]),
                "B_minus_matched_lz_max": to_float(row["B_minus_matched_lz_max"]),
            }
        )
    return sorted(rows, key=lambda item: (band_start(item["band"]), item["lag_bits"]))


def to_float(value: str) -> float | None:
    if value == "":
        return None
    return float(value)


def band_start(label: str) -> int:
    return int(label.split("M", 1)[0])


def ordered_bands(rows: list[dict[str, Any]]) -> list[str]:
    return sorted({row["band"] for row in rows}, key=band_start)


def ordered_lags(rows: list[dict[str, Any]]) -> list[str]:
    by_label = {row["lag"]: row["lag_bits"] for row in rows}
    return [label for label, _bits in sorted(by_label.items(), key=lambda item: item[1])]


def build_envelopes(
    rows: list[dict[str, Any]],
    *,
    b_threshold: float,
    margin_threshold: float,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for band in ordered_bands(rows):
        for channel in ("negative", "positive"):
            channel_rows = [row for row in rows if row["band"] == band and row["channel"] == channel]
            if not channel_rows:
                continue
            strong = [
                row
                for row in channel_rows
                if row["B_retention"] is not None
                and row["B_minus_matched_lz_max"] is not None
                and row["B_retention"] >= b_threshold
                and row["B_minus_matched_lz_max"] >= margin_threshold
            ]
            best_margin = max(
                channel_rows,
                key=lambda row: value_or_low(row["B_minus_matched_lz_max"]),
            )
            output.append(
                {
                    "band": band,
                    "channel": channel,
                    "strong_count": len(strong),
                    "strong_lag_min_bits": min_lag(strong),
                    "strong_lag_max_bits": max_lag(strong),
                    "strong_width_bits": width_bits(strong),
                    "best_margin_lag_bits": best_margin["lag_bits"],
                    "best_margin_lag": best_margin["lag"],
                    "best_margin_B_retention": best_margin["B_retention"],
                    "best_margin": best_margin["B_minus_matched_lz_max"],
                }
            )
    return output


def build_best_by_band(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for band in ordered_bands(rows):
        items = [row for row in rows if row["band"] == band]
        best_margin = max(items, key=lambda row: value_or_low(row["B_minus_matched_lz_max"]))
        best_b = max(items, key=lambda row: value_or_low(row["B_retention"]))
        output.append(
            {
                "band": band,
                "best_B_lag": best_b["lag"],
                "best_B_retention": best_b["B_retention"],
                "best_B_margin": best_b["B_minus_matched_lz_max"],
                "best_margin_lag": best_margin["lag"],
                "best_margin_B_retention": best_margin["B_retention"],
                "best_margin": best_margin["B_minus_matched_lz_max"],
            }
        )
    return output


def value_or_low(value: float | None) -> float:
    return value if value is not None else -1e9


def min_lag(rows: list[dict[str, Any]]) -> int | None:
    return min((int(row["lag_bits"]) for row in rows), default=None)


def max_lag(rows: list[dict[str, Any]]) -> int | None:
    return max((int(row["lag_bits"]) for row in rows), default=None)


def width_bits(rows: list[dict[str, Any]]) -> int | None:
    if not rows:
        return None
    return int(max_lag(rows) - min_lag(rows))


def matrix(rows: list[dict[str, Any]], key: str, bands: list[str], lags: list[str]) -> list[list[float]]:
    by_cell = {(row["band"], row["lag"]): row for row in rows}
    return [[float(by_cell[(band, lag)][key]) for lag in lags] for band in bands]


def render_figure(
    rows: list[dict[str, Any]],
    envelopes: list[dict[str, Any]],
    *,
    bands: list[str],
    lags: list[str],
    png_path: Path,
    svg_path: Path,
    b_threshold: float,
    margin_threshold: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Rectangle

    margin_matrix = matrix(rows, "B_minus_matched_lz_max", bands, lags)
    strong_cells = {
        (bands.index(row["band"]), lags.index(row["lag"]))
        for row in rows
        if row["B_retention"] is not None
        and row["B_minus_matched_lz_max"] is not None
        and row["B_retention"] >= b_threshold
        and row["B_minus_matched_lz_max"] >= margin_threshold
    }

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.8), gridspec_kw={"width_ratios": [1.42, 1.0]})
    fig.patch.set_facecolor("#f7f3ea")
    ax_heat, ax_width = axes
    for ax in axes:
        ax.set_facecolor("#fffdf8")

    norm = TwoSlopeNorm(vmin=-0.35, vcenter=0.0, vmax=0.70)
    image = ax_heat.imshow(margin_matrix, cmap="RdBu_r", norm=norm, aspect="auto")
    ax_heat.set_title("N2-12 B margin over strongest matched-LZ seed", fontsize=12, fontweight="bold")
    ax_heat.set_xticks(range(len(lags)), lags, rotation=45, ha="right")
    band_labels = [f"{band}*" if band in BOUNDARY_BAND_NOTES else band for band in bands]
    ax_heat.set_yticks(range(len(bands)), band_labels)
    ax_heat.set_ylabel("anchor band")
    ax_heat.set_xlabel("candidate lag")
    ax_heat.set_xticks([x - 0.5 for x in range(1, len(lags))], minor=True)
    ax_heat.set_yticks([y - 0.5 for y in range(1, len(bands))], minor=True)
    ax_heat.grid(which="minor", color="#f7f3ea", linewidth=1.4)
    ax_heat.tick_params(which="minor", bottom=False, left=False)
    for y, band in enumerate(bands):
        for x, lag in enumerate(lags):
            value = margin_matrix[y][x]
            color = text_color_for_cell(value, image.cmap, image.norm)
            ax_heat.text(x, y, f"{value:+.2f}", ha="center", va="center", fontsize=7.5, color=color)
    for y, x in strong_cells:
        ax_heat.add_patch(
            Rectangle((x - 0.48, y - 0.48), 0.96, 0.96, fill=False, edgecolor="#111827", linewidth=2.0)
        )
    for band in BOUNDARY_BAND_NOTES:
        if band in bands:
            y = bands.index(band)
            ax_heat.add_patch(
                Rectangle(
                    (-0.5, y - 0.5),
                    len(lags),
                    1.0,
                    fill=False,
                    edgecolor="#92400e",
                    linewidth=1.8,
                    linestyle="--",
                )
            )
    cbar = fig.colorbar(image, ax=ax_heat, fraction=0.045, pad=0.04)
    cbar.set_label("B - max(matched-LZ)", fontsize=9)

    width_rows = build_width_plot_rows(envelopes)
    y_positions = list(range(len(width_rows)))
    ax_width.axvline(0, color="#9ca3af", linewidth=1.0, linestyle="--")
    for y, item in zip(y_positions, width_rows):
        all_min, all_max = item["domain"]
        if item["band"] in BOUNDARY_BAND_NOTES:
            ax_width.axhspan(y - 0.42, y + 0.42, color="#f59e0b", alpha=0.08, zorder=0)
        ax_width.plot([all_min / 1_000_000, all_max / 1_000_000], [y, y], color="#d1d5db", linewidth=5)
        if item["strong_min"] is not None:
            ax_width.plot(
                [item["strong_min"] / 1_000_000, item["strong_max"] / 1_000_000],
                [y, y],
                color=item["color"],
                linewidth=7,
                solid_capstyle="round",
            )
        ax_width.scatter(
            [item["best_lag"] / 1_000_000],
            [y],
            color="#111827",
            s=34,
            marker="D",
            zorder=3,
        )
        label = f"{item['strong_count']} cells"
        if item["strong_width"] is not None:
            label += f" / {compact_int(item['strong_width'])}"
        if item["channel"] == "positive":
            label_x = all_min / 1_000_000 - 1.2
            label_align = "right"
        else:
            label_x = all_max / 1_000_000 + 1.2
            label_align = "left"
        ax_width.text(
            label_x,
            y,
            label,
            ha=label_align,
            va="center",
            fontsize=8,
            color="#374151",
        )
    ax_width.set_title("Strong channel width envelope", fontsize=12, fontweight="bold")
    ax_width.set_yticks(y_positions, [item["label"] for item in width_rows])
    ax_width.set_xlabel("candidate lag (M bits)")
    ax_width.set_xlim(-33, 33)
    ax_width.invert_yaxis()
    ax_width.grid(axis="x", color="#e5e7eb", linewidth=1)

    fig.suptitle("HSI v2 Phase 2 N2-12 Phase-Channel Width", fontsize=17, fontweight="bold", y=0.98)
    fig.text(
        0.5,
        0.045,
        f"Black boxes/bars mark strong cells: B >= {b_threshold:.2f} and B - max(matched-LZ) >= {margin_threshold:.2f}. "
        "Diamond = best margin within each channel.\n"
        "* D-0060: 729M-738M is top-k-sensitive boundary, not a stable late-channel claim.",
        ha="center",
        fontsize=8.7,
        color="#374151",
    )
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.20, top=0.89, wspace=0.32)
    fig.savefig(png_path, dpi=180)
    fig.savefig(svg_path)
    plt.close(fig)


def build_width_plot_rows(envelopes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    colors = {"negative": "#0f766e", "positive": "#b45309"}
    domains = {"negative": (-30_000_000, -24_000_000), "positive": (23_500_000, 29_500_000)}
    for item in envelopes:
        channel = item["channel"]
        rows.append(
            {
                "label": f"{item['band']} {channel[:3]}",
                "band": item["band"],
                "channel": channel,
                "domain": domains[channel],
                "strong_min": item["strong_lag_min_bits"],
                "strong_max": item["strong_lag_max_bits"],
                "strong_width": item["strong_width_bits"],
                "strong_count": item["strong_count"],
                "best_lag": item["best_margin_lag_bits"],
                "color": colors[channel],
            }
        )
    return rows


def text_color_for_cell(value: float, cmap, norm) -> str:
    red, green, blue, _alpha = cmap(norm(value))
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "#111827" if luminance > 0.55 else "white"


def render_report(summary: dict[str, Any], png_name: str, svg_name: str) -> str:
    lines = [
        "# Phase 2 N2-12 Phase-Channel Width Figure",
        "",
        "## Figure",
        "",
        f"![Phase-channel width](./{png_name})",
        "",
        f"SVG version: `{svg_name}`",
        "",
        "## Reading",
        "",
        "The heatmap shows `B - max(matched-LZ)` over the sampled fine lag grid.",
        "Outlined cells satisfy the strong-cell criterion used for the width envelope.",
        "The right panel summarizes the sampled width of negative and positive phase channels per band.",
        "",
        "## Best Cells By Band",
        "",
        "| Band | Best B lag | B retention | Best-margin lag | Margin |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["best_by_band"]:
        lines.append(
            f"| {row['band']} | {row['best_B_lag']} | {row['best_B_retention']:.4f} | "
            f"{row['best_margin_lag']} | {row['best_margin']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Strong Channel Envelopes",
            "",
            "| Band | Channel | Strong cells | Lag min | Lag max | Width | Best margin lag | Margin |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["channel_envelopes"]:
        lines.append(
            f"| {row['band']} | {row['channel']} | {row['strong_count']} | "
            f"{compact_optional(row['strong_lag_min_bits'])} | {compact_optional(row['strong_lag_max_bits'])} | "
            f"{compact_optional(row['strong_width_bits'])} | {row['best_margin_lag']} | "
            f"{row['best_margin']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def compact_optional(value: int | None) -> str:
    return "-" if value is None else compact_int(int(value))


if __name__ == "__main__":
    raise SystemExit(main())
