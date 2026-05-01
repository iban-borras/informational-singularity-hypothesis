#!/usr/bin/env python3
"""Render the HSI v2 Phase 2 N2-11 lag-response heatmap."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from v2.common.cli import resolve_dir


DEFAULT_RESPONSE_ROOT = "results/hsi_v2/phase2/child_routing_lag_response_n2_11"
DEFAULT_OUTPUT_DIR = "results/hsi_v2/phase2/visual_readouts"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a two-panel heatmap from an N2-11 lag-response run."
    )
    parser.add_argument(
        "--lag-response-run",
        default="",
        help="Path to one phase2-child-routing-lag-response__... run. Defaults to latest run.",
    )
    parser.add_argument(
        "--lag-response-root",
        default=DEFAULT_RESPONSE_ROOT,
        help="Root used to discover the latest N2-11 run when --lag-response-run is omitted.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    response_run = resolve_response_run(args)
    response_csv = response_run / "lag_response.csv"
    targets_csv = response_run / "targets.csv"
    if not response_csv.is_file():
        raise SystemExit(f"Missing lag_response.csv: {response_csv}")
    if not targets_csv.is_file():
        raise SystemExit(f"Missing targets.csv: {targets_csv}")

    rows = read_csv(response_csv)
    targets = read_csv(targets_csv)
    plot_rows = build_plot_rows(rows, targets)
    if not plot_rows:
        raise SystemExit("No plottable rows found in lag_response.csv")

    output_root = resolve_dir(args.output_dir, anchor_file=__file__)
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    source_stamp = response_run.name.split("__")[-1]
    run_slug = f"phase2-lag-response-heatmap__src-{source_stamp}__{timestamp}"
    run_dir = output_root / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)

    png_path = run_dir / "lag_response_heatmap.png"
    svg_path = run_dir / "lag_response_heatmap.svg"
    csv_path = run_dir / "lag_response_heatmap_points.csv"
    summary_path = run_dir / "summary.json"
    report_path = run_dir / "report.md"
    manifest_path = run_dir / "manifest.json"

    write_csv(csv_path, plot_rows)
    render_heatmap(plot_rows, png_path=png_path, svg_path=svg_path)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "selection": {
            "lag_response_run": str(response_run),
            "lag_response_csv": str(response_csv),
            "targets_csv": str(targets_csv),
            "reading": "N2-11 child-routing lag-response: observed B retention and B-minus-max(matched-LZ) over band x lag.",
        },
        "best_by_band": best_by_band(plot_rows),
        "lag_averages": lag_averages(plot_rows),
        "plot_rows": plot_rows,
    }
    manifest = {
        "generated_at": summary["generated_at"],
        "script": "hsi_v2_phase2_lag_response_heatmap.py",
        "inputs": {
            "lag_response_run": str(response_run),
            "lag_response_csv": str(response_csv),
            "targets_csv": str(targets_csv),
        },
        "outputs": {
            "summary": str(summary_path),
            "report": str(report_path),
            "points_csv": str(csv_path),
            "png": str(png_path),
            "svg": str(svg_path),
            "manifest": str(manifest_path),
        },
        "arguments": vars(args),
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_path.write_text(render_report(summary, png_path.name, svg_path.name), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not args.quiet:
        print("Phase 2 N2-11 lag-response heatmap")
        print("-" * 72)
        for item in summary["best_by_band"]:
            print(
                f"{item['band']:<10} best_B={item['best_B_lag']:>7} "
                f"B={item['best_B_retention']:.4f} "
                f"best_margin={item['best_margin_lag']:>7} "
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
        and (path / "targets.csv").is_file()
    ]
    if not candidates:
        raise SystemExit(f"No complete lag-response runs found under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_plot_rows(rows: list[dict], targets: list[dict]) -> list[dict]:
    recommended = {
        (row["band"], row["lag"]): row.get("probe_recommended_lag") == row.get("lag")
        for row in targets
    }
    plot_rows = []
    for row in rows:
        item = {
            "band": row["band"],
            "lag": row["lag"],
            "lag_bits": int(row["lag_bits"]),
            "B_retention": to_float(row["B_retention"]),
            "E_retention": to_float(row["E_retention"]),
            "markov1_retention": to_float(row["markov1_retention"]),
            "matched_lz_min": to_float(row["matched_lz_min"]),
            "matched_lz_avg": to_float(row["matched_lz_avg"]),
            "matched_lz_max": to_float(row["matched_lz_max"]),
            "B_minus_matched_lz_max": to_float(row["B_minus_matched_lz_max"]),
            "is_probe_recommended": bool(recommended.get((row["band"], row["lag"]), False)),
        }
        plot_rows.append(item)
    return sorted(plot_rows, key=lambda item: (band_start(item["band"]), item["lag_bits"]))


def to_float(value: str) -> float | None:
    if value == "":
        return None
    return float(value)


def band_start(label: str) -> int:
    return int(label.split("M", 1)[0])


def best_by_band(rows: list[dict]) -> list[dict]:
    output = []
    for band in ordered_bands(rows):
        items = [row for row in rows if row["band"] == band]
        best_b = max(items, key=lambda row: value_or_low(row["B_retention"]))
        best_margin = max(items, key=lambda row: value_or_low(row["B_minus_matched_lz_max"]))
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


def lag_averages(rows: list[dict]) -> list[dict]:
    output = []
    for lag in ordered_lags(rows):
        items = [row for row in rows if row["lag"] == lag]
        output.append(
            {
                "lag": lag,
                "lag_bits": items[0]["lag_bits"],
                "avg_B_retention": mean(row["B_retention"] for row in items),
                "avg_margin": mean(row["B_minus_matched_lz_max"] for row in items),
            }
        )
    return output


def value_or_low(value: float | None) -> float:
    return value if value is not None else -1e9


def mean(values) -> float:
    data = [float(value) for value in values if value is not None]
    return sum(data) / len(data) if data else 0.0


def ordered_bands(rows: list[dict]) -> list[str]:
    return sorted({row["band"] for row in rows}, key=band_start)


def ordered_lags(rows: list[dict]) -> list[str]:
    by_label = {row["lag"]: row["lag_bits"] for row in rows}
    return [label for label, _bits in sorted(by_label.items(), key=lambda item: item[1])]


def matrix(rows: list[dict], key: str, bands: list[str], lags: list[str]) -> list[list[float]]:
    by_cell = {(row["band"], row["lag"]): row for row in rows}
    output = []
    for band in bands:
        output.append([float(by_cell[(band, lag)][key]) for lag in lags])
    return output


def render_heatmap(rows: list[dict], *, png_path: Path, svg_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Rectangle

    bands = ordered_bands(rows)
    lags = ordered_lags(rows)
    b_matrix = matrix(rows, "B_retention", bands, lags)
    margin_matrix = matrix(rows, "B_minus_matched_lz_max", bands, lags)
    rec_cells = {
        (bands.index(row["band"]), lags.index(row["lag"]))
        for row in rows
        if row["is_probe_recommended"]
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 7.4))
    fig.patch.set_facecolor("#f7f3ea")
    for ax in axes:
        ax.set_facecolor("#fffdf8")

    panels = [
        (
            axes[0],
            b_matrix,
            "Observed B child-destination retention",
            "viridis",
            0.0,
            1.0,
            None,
        ),
        (
            axes[1],
            margin_matrix,
            "B minus strongest matched-LZ seed",
            "RdBu_r",
            None,
            None,
            TwoSlopeNorm(vmin=-0.9, vcenter=0.0, vmax=0.9),
        ),
    ]

    for ax, data, title, cmap, vmin, vmax, norm in panels:
        image = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, norm=norm, aspect="auto")
        color_mapper = image.cmap
        normalizer = image.norm
        ax.set_title(title, fontsize=13, fontweight="bold", color="#1f2937", pad=12)
        ax.set_xticks(range(len(lags)), lags, rotation=45, ha="right")
        ax.set_yticks(range(len(bands)), bands)
        ax.set_ylabel("anchor band")
        ax.set_xticks([x - 0.5 for x in range(1, len(lags))], minor=True)
        ax.set_yticks([y - 0.5 for y in range(1, len(bands))], minor=True)
        ax.grid(which="minor", color="#f7f3ea", linewidth=1.5)
        ax.tick_params(which="minor", bottom=False, left=False)
        for y, band in enumerate(bands):
            for x, lag in enumerate(lags):
                value = data[y][x]
                color = text_color_for_cell(value, color_mapper, normalizer)
                ax.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=8, color=color)
        for y, x in rec_cells:
            ax.add_patch(
                Rectangle(
                    (x - 0.48, y - 0.48),
                    0.96,
                    0.96,
                    fill=False,
                    edgecolor="#111827",
                    linewidth=2.4,
                )
            )
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        "HSI v2 Phase 2 N2-11 Lag-Response Profile",
        fontsize=17,
        fontweight="bold",
        color="#111827",
        y=0.98,
    )
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.20, top=0.88, wspace=0.28)
    fig.text(
        0.5,
        0.055,
        "Black cell outline = lag recommended by the local shell-lag probe. "
        "Cells show pooled child-destination retention over monitored top-128 shell children.",
        ha="center",
        fontsize=9,
        color="#374151",
    )
    fig.savefig(png_path, dpi=180)
    fig.savefig(svg_path)
    plt.close(fig)


def text_color_for_cell(value: float, cmap, norm) -> str:
    red, green, blue, _alpha = cmap(norm(value))
    # Perceptual luminance for sRGB-like matplotlib colors.
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "#111827" if luminance > 0.55 else "white"


def render_report(summary: dict, png_name: str, svg_name: str) -> str:
    lines = [
        "# Phase 2 N2-11 Lag-Response Heatmap",
        "",
        "## Figure",
        "",
        f"![Lag-response heatmap](./{png_name})",
        "",
        f"SVG version: `{svg_name}`",
        "",
        "## Reading",
        "",
        "The left panel shows observed `B` child-destination retention over the explicit lag grid.",
        "The right panel shows `B - max(matched-LZ)`, so positive cells are places where observed `B` beats every matched-LZ seed.",
        "Black outlines mark the lag recommended by the local shell-lag probe.",
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
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
