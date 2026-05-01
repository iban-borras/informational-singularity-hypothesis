#!/usr/bin/env python3
"""
HSI v2 Phase 2 Child-Routing Regime Map

Compact N2-09 visual/readout summarizer. It merges child-routing persistence
runs into a band-level regime table and a deterministic timeline figure.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_OUTPUT_DIR = "results/hsi_v2/phase2/child_routing_regime_map"
DEFAULT_RUN_ROOTS = (
    "results/hsi_v2/phase2/child_routing_persistence_probe_map",
    "results/hsi_v2/phase2/child_routing_persistence_boundary705_wide",
    "results/hsi_v2/phase2/child_routing_persistence_probe_extension",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a compact N2-09 child-routing regime map from persistence runs."
        )
    )
    parser.add_argument(
        "--run-dirs",
        type=str,
        default="",
        help=(
            "Comma-separated exact persistence run directories. If omitted, the "
            "latest run under each canonical N2-09 regime-map root is used."
        ),
    )
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-plot", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(__file__).parent.resolve()
    output_root = resolve_path(args.output_dir, root)
    output_root.mkdir(parents=True, exist_ok=True)

    run_dirs = resolve_run_dirs(args.run_dirs, root)
    if not run_dirs:
        raise SystemExit("No N2-09 persistence runs found.")

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = output_root / f"phase2-child-routing-regime-map__{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = build_regime_rows(run_dirs)
    summary_path = run_dir / "summary.json"
    report_path = run_dir / "report.md"
    csv_path = run_dir / "regime_map.csv"
    manifest_path = run_dir / "manifest.json"
    png_path = run_dir / "child_routing_regime_map.png"

    write_csv(rows, csv_path)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(render_report(rows, run_dirs) + "\n")

    plot_path = None
    if not args.no_plot:
        plot_regime_map(rows, png_path)
        plot_path = png_path

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_run_dirs": [str(path) for path in run_dirs],
        "rows": rows,
        "outputs": {
            "report": str(report_path),
            "csv": str(csv_path),
            "png": str(plot_path) if plot_path else None,
            "summary": str(summary_path),
            "manifest": str(manifest_path),
        },
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "generated_at": summary["generated_at"],
                "script": "hsi_v2_phase2_child_routing_regime_map.py",
                "arguments": vars(args),
                "source_run_dirs": summary["source_run_dirs"],
                "outputs": summary["outputs"],
            },
            handle,
            indent=2,
        )

    print("Phase 2 child-routing regime map")
    print("-" * 96)
    print(render_console_table(rows))
    print("")
    print(f"Saved summary to: {summary_path}")
    print(f"Saved report to: {report_path}")
    print(f"Saved CSV to: {csv_path}")
    if plot_path:
        print(f"Saved PNG to: {plot_path}")
    print(f"Saved manifest to: {manifest_path}")
    return 0


def resolve_path(value: str, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def resolve_run_dirs(raw: str, root: Path) -> list[Path]:
    if raw.strip():
        return [resolve_path(item.strip(), root) for item in raw.split(",") if item.strip()]

    run_dirs: list[Path] = []
    for rel_root in DEFAULT_RUN_ROOTS:
        folder = root / rel_root
        if not folder.exists():
            continue
        children = [child for child in folder.iterdir() if child.is_dir()]
        if children:
            run_dirs.append(max(children, key=lambda child: child.stat().st_mtime))
    return run_dirs


def build_regime_rows(run_dirs: list[Path]) -> list[dict]:
    by_band: dict[tuple[int, str], dict] = {}
    for run_dir in run_dirs:
        band_path = run_dir / "band_persistence.csv"
        source_path = run_dir / "source_persistence.csv"
        if not band_path.exists():
            continue
        bands = read_csv(band_path)
        sources = read_csv(source_path) if source_path.exists() else []
        sources_by_band: dict[str, list[dict]] = {}
        for row in sources:
            sources_by_band.setdefault(row["band_label"], []).append(row)
        for band in bands:
            key = (int(band["band_start_bits"]), band["band_label"])
            by_band[key] = summarize_band(band, sources_by_band.get(band["band_label"], []))
    return [by_band[key] for key in sorted(by_band)]


def summarize_band(band: dict, source_rows: list[dict]) -> dict:
    b_row = first(row for row in source_rows if row.get("variant") == "B")
    markov_row = first(row for row in source_rows if row.get("variant") == "B-markov1")
    matched_rows = [row for row in source_rows if row.get("variant") == "B-matched-lz"]
    matched_values = [
        parse_float(row.get("child_destination_retention_pooled")) for row in matched_rows
    ]
    matched_values = [value for value in matched_values if value is not None]
    b_value = parse_float(b_row.get("child_destination_retention_pooled")) if b_row else None
    matched_max = max(matched_values) if matched_values else None
    selected_lag = parse_int(band.get("selected_lag_bits"))
    status = band.get("band_status") or "completed"
    regime = classify_regime(status, selected_lag, b_value, matched_max)
    return {
        "band_label": band["band_label"],
        "band_start_bits": parse_int(band["band_start_bits"]),
        "band_end_bits": parse_int(band["band_end_bits"]),
        "probe_lag_bits": parse_int(band.get("probe_lag_bits")),
        "selected_lag_bits": selected_lag,
        "band_status": status,
        "regime": regime,
        "b_child_retention": b_value,
        "b_sync_retention": parse_float(
            b_row.get("window_synchronous_child_destination_retention_pooled")
        )
        if b_row
        else None,
        "matched_lz_min": min(matched_values) if matched_values else None,
        "matched_lz_avg": sum(matched_values) / len(matched_values) if matched_values else None,
        "matched_lz_max": matched_max,
        "margin_vs_matched_lz_max": (b_value - matched_max)
        if b_value is not None and matched_max is not None
        else None,
        "markov1_child_retention": parse_float(
            markov_row.get("child_destination_retention_pooled")
        )
        if markov_row
        else None,
        "b_deficit": parse_float(b_row.get("anchor_child_deficit_mass_sum")) if b_row else None,
        "b_class": b_row.get("dominant_routing_class") if b_row else "",
        "skip_reason": band.get("skip_reason", ""),
    }


def classify_regime(
    status: str,
    selected_lag: int | None,
    b_value: float | None,
    matched_max: float | None,
) -> str:
    if status != "completed":
        return "no-lag boundary"
    if selected_lag == 0:
        return "zero-lag coincident"
    if b_value is None:
        return "unread"
    if b_value >= 0.9:
        return "strong phase"
    if b_value >= 0.5:
        return "partial / liminal"
    if matched_max is not None and b_value <= matched_max:
        return "null-shadowed"
    return "weak"


def plot_regime_map(rows: list[dict], path: Path) -> None:
    labels = [short_band(row["band_label"]) for row in rows]
    x_values = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(10.5, 5.6), dpi=180)
    ax.set_facecolor("#fbfaf5")
    fig.patch.set_facecolor("#fbfaf5")

    for x, row in zip(x_values, rows):
        if row["band_status"] != "completed":
            ax.axvspan(x - 0.42, x + 0.42, color="#d9d6cc", alpha=0.65, zorder=0)
            ax.text(
                x,
                0.52,
                "no\nprobe\nlag",
                ha="center",
                va="center",
                fontsize=9,
                color="#56524a",
                fontweight="bold",
            )
            continue

        lo = row["matched_lz_min"]
        hi = row["matched_lz_max"]
        avg = row["matched_lz_avg"]
        if lo is not None and hi is not None:
            ax.vlines(x, lo, hi, color="#b45f2a", linewidth=8, alpha=0.35, zorder=1)
            ax.scatter([x], [avg], s=52, color="#b45f2a", edgecolor="#6f3616", zorder=2)
        marker = "D" if row["selected_lag_bits"] == 0 else "o"
        ax.scatter(
            [x],
            [row["b_child_retention"]],
            s=118,
            marker=marker,
            color="#0f766e",
            edgecolor="#064e48",
            linewidth=1.5,
            zorder=3,
        )
        ax.text(
            x,
            min(1.08, row["b_child_retention"] + 0.08),
            lag_label(row["selected_lag_bits"]),
            ha="center",
            va="bottom",
            fontsize=9,
            color="#143b36",
            fontweight="bold",
        )

    ax.axhline(0, color="#64748b", linewidth=1, alpha=0.6)
    ax.set_ylim(-0.04, 1.12)
    ax.set_xlim(-0.55, len(rows) - 0.45)
    ax.set_xticks(x_values)
    ax.set_xticklabels(labels, rotation=0, fontsize=9)
    ax.set_ylabel("child-destination retention", fontsize=11, fontweight="bold")
    ax.set_title("N2-09 Child-Routing Regime Map", fontsize=15, fontweight="bold")
    ax.grid(axis="y", color="#dbe3ef", linewidth=1, alpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)

    ax.scatter([], [], s=118, color="#0f766e", edgecolor="#064e48", label="observed B")
    ax.vlines([], [], [], color="#b45f2a", linewidth=8, alpha=0.35, label="matched-LZ envelope")
    ax.scatter([], [], s=52, color="#b45f2a", edgecolor="#6f3616", label="matched-LZ average")
    ax.legend(loc="lower right", frameon=True, facecolor="#fbfaf5", edgecolor="#d8d1c2")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def render_report(rows: list[dict], run_dirs: list[Path]) -> str:
    lines = [
        "# Phase 2 N2-09 Child-Routing Regime Map",
        "",
        "## Source Runs",
        "",
    ]
    lines.extend(f"- `{path}`" for path in run_dirs)
    lines.extend(
        [
            "",
            "## Regime Table",
            "",
            "| Band | Lag | Status | Regime | B childRet | matched-LZ envelope | markov1 | Margin vs matched max |",
            "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| `{row['band_label']}` | {lag_label(row['selected_lag_bits'])} | "
            f"{row['band_status']} | {row['regime']} | {fmt(row['b_child_retention'])} | "
            f"{fmt_range(row['matched_lz_min'], row['matched_lz_max'])} | "
            f"{fmt(row['markov1_child_retention'])} | {fmt(row['margin_vs_matched_lz_max'])} |"
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- The regime is structured, not globally stationary.",
            "- A narrow forward probe can expose scan boundaries; a wide probe can resolve them as phase-shifted transport.",
            "- `705M-714M` is resolved by the wide boundary probe as a negative-lag phase, not as an intrinsic absence of transport.",
            "- `714M-723M` and `723M-732M` remain the strongest positive transport bands.",
            "- `732M-741M` is a zero-lag coincident phase: observed `B` stays conserved, but `matched-LZ` approaches the monitored destination channel.",
            "- The figure is a compact diagnostic map, not a paper-final transport law.",
        ]
    )
    return "\n".join(lines)


def render_console_table(rows: list[dict]) -> str:
    lines = [
        f"{'band':<16} {'lag':>8} {'regime':<22} {'B':>8} {'mLZ range':>17} {'margin':>8}"
    ]
    for row in rows:
        lines.append(
            f"{short_band(row['band_label']):<16} {lag_label(row['selected_lag_bits']):>8} "
            f"{row['regime']:<22} {fmt(row['b_child_retention']):>8} "
            f"{fmt_range(row['matched_lz_min'], row['matched_lz_max']):>17} "
            f"{fmt(row['margin_vs_matched_lz_max']):>8}"
        )
    return "\n".join(lines)


def read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def first(items) -> dict | None:
    for item in items:
        return item
    return None


def parse_float(value) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def parse_int(value) -> int | None:
    if value in (None, ""):
        return None
    return int(float(value))


def fmt(value: float | int | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def fmt_range(lo: float | None, hi: float | None) -> str:
    if lo is None or hi is None:
        return "-"
    return f"{lo:.4f}-{hi:.4f}"


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
