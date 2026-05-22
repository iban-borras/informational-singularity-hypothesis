#!/usr/bin/env python3
"""Render the Phase 1 projective tower prefix-survival figure."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path

from v2.common.cli import resolve_dir


DEFAULT_PHASE1_DIR = "results/hsi_v2/phase1"
DEFAULT_OUTPUT_DIR = "results/hsi_v2/visualizations"
DEFAULT_SCALES = (8, 12, 16, 20, 24, 28, 32)
DEFAULT_GROUPS = (
    ("B/E", ("B", "E"), "canonical HSI"),
    ("I", ("I",), "inverse HSI"),
    ("F", ("F",), "rigid"),
    ("M/N", ("M", "N"), "crystalline"),
    ("A", ("A",), "collapse contrast"),
    ("L", ("L",), "collapse contrast"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a paper-facing Phase 1 projective tower figure from existing "
            "phase1_summary.json artifacts. No Phase 1 data are regenerated."
        )
    )
    parser.add_argument(
        "--phase1-dir",
        default=DEFAULT_PHASE1_DIR,
        help="Directory containing canonical observed Phase 1 tower runs.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the visualization run directory will be written.",
    )
    parser.add_argument(
        "--scales",
        default=",".join(str(item) for item in DEFAULT_SCALES),
        help="Comma-separated sampled scales to render.",
    )
    parser.add_argument(
        "--title",
        default="Prefix survival across the Phase 1 projective tower",
        help="Figure title.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress console summary.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    phase1_dir = resolve_dir(args.phase1_dir, anchor_file=__file__)
    output_root = resolve_dir(args.output_dir, anchor_file=__file__)
    scales = parse_scales(args.scales)

    groups = build_groups(phase1_dir, scales)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = output_root / f"phase1-projective-tower-survival__{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    data_path = run_dir / "projective_tower_survival_data.json"
    png_path = run_dir / "projective_tower_survival.png"
    pdf_path = run_dir / "projective_tower_survival.pdf"
    report_path = run_dir / "report.md"
    manifest_path = run_dir / "manifest.json"

    data_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": Path(__file__).name,
        "reading": (
            "For each sampled prefix edge m -> next_m, surviving_patterns is the "
            "number of patterns in P_m with non-empty prefix fiber in P_next_m. "
            "dropped_patterns is the remaining count after the stability-filtered "
            "next level is applied."
        ),
        "wording_fence": {
            "allowed_terms": [
                "pattern cardinality",
                "surviving patterns",
                "dropped patterns",
                "prefix projection",
                "prefix fiber",
                "retention",
                "survival",
            ],
            "figure_note": (
                "Figure segments encode fractions of patterns. They are not physical "
                "masses or charges."
            ),
        },
        "scales": scales,
        "groups": groups,
    }
    data_path.write_text(json.dumps(data_payload, indent=2), encoding="utf-8")

    render_figure(groups, scales, png_path=png_path, pdf_path=pdf_path, title=args.title)

    report_path.write_text(render_report(data_payload, png_path, pdf_path), encoding="utf-8")
    manifest = {
        "generated_at": data_payload["generated_at"],
        "script": Path(__file__).name,
        "command": [sys.executable, *sys.argv],
        "inputs": source_manifest(groups),
        "outputs": {
            "data_json": str(data_path),
            "png": str(png_path),
            "pdf": str(pdf_path),
            "report": str(report_path),
            "manifest": str(manifest_path),
        },
        "arguments": vars(args),
        "notes": [
            "Read-only visualization from canonical Phase 1 observed runs.",
            "No strong null variants are included in this figure.",
            "Grouped labels B/E and M/N are validated as identical before rendering.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not args.quiet:
        print("Phase 1 projective tower prefix-survival figure")
        print("-" * 78)
        for group in groups:
            counts = [group["pattern_cardinality"][str(scale)] for scale in scales]
            print(f"{group['label']:<4} |P_m|: {', '.join(format_count(item) for item in counts)}")
        print("")
        print(f"Saved data JSON to: {data_path}")
        print(f"Saved PNG figure to: {png_path}")
        print(f"Saved PDF figure to: {pdf_path}")
        print(f"Saved report to: {report_path}")
        print(f"Saved manifest to: {manifest_path}")
    return 0


def parse_scales(raw: str) -> list[int]:
    scales = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if len(scales) < 2:
        raise SystemExit("At least two scales are required.")
    if scales != sorted(scales):
        raise SystemExit("Scales must be strictly increasing.")
    if len(set(scales)) != len(scales):
        raise SystemExit("Scales must not contain duplicates.")
    return scales


def build_groups(phase1_dir: Path, scales: list[int]) -> list[dict]:
    groups: list[dict] = []
    for label, variants, family in DEFAULT_GROUPS:
        loaded = [load_variant_summary(phase1_dir, variant, scales) for variant in variants]
        reference = loaded[0]
        for item in loaded[1:]:
            if item["pattern_cardinality"] != reference["pattern_cardinality"]:
                raise SystemExit(f"Grouped variants {label} do not have identical |P_m|.")
            if edge_figure_signature(item["edges"]) != edge_figure_signature(reference["edges"]):
                raise SystemExit(
                    f"Grouped variants {label} do not have identical rendered prefix survival."
                )
        groups.append(
            {
                "label": label,
                "represented_variants": list(variants),
                "family": family,
                "source_summaries": [item["source_summary"] for item in loaded],
                "pattern_cardinality": reference["pattern_cardinality"],
                "edges": reference["edges"],
                "represented_variant_edge_details": {
                    item["variant"]: item["edges"] for item in loaded
                },
            }
        )
    return groups


def edge_figure_signature(edges: list[dict]) -> list[tuple]:
    return [
        (
            item["from_m"],
            item["to_m"],
            item["target_pattern_cardinality"],
            item["surviving_patterns"],
            item["dropped_patterns"],
            round(float(item["survival_fraction"]), 12),
        )
        for item in edges
    ]


def load_variant_summary(phase1_dir: Path, variant: str, scales: list[int]) -> dict:
    scale_slug = "-".join(str(item) for item in scales)
    pattern = (
        f"phase1-tower__var-{variant}__iter-*__seg-3x1M__m-{scale_slug}"
        "__proj-prefix-suffix__*"
    )
    candidates = sorted(
        path / "phase1_summary.json"
        for path in phase1_dir.glob(pattern)
        if (path / "phase1_summary.json").is_file()
    )
    if not candidates:
        raise SystemExit(f"No canonical Phase 1 summary found for variant {variant} under {phase1_dir}")
    summary_path = max(candidates, key=lambda path: path.stat().st_mtime)
    data = json.loads(summary_path.read_text(encoding="utf-8"))

    counts = {int(row["m"]): int(row["kept_pattern_count"]) for row in data["scale_summary"]}
    missing_scales = [scale for scale in scales if scale not in counts]
    if missing_scales:
        raise SystemExit(f"Variant {variant} is missing scales: {missing_scales}")

    prefix_edges = {
        (int(row["target_m"]), int(row["source_m"])): row
        for row in data["fiber_summary"].get("prefix", [])
    }
    edges: list[dict] = []
    for target_m, source_m in zip(scales[:-1], scales[1:]):
        row = prefix_edges.get((target_m, source_m))
        if row is None:
            raise SystemExit(f"Variant {variant} missing prefix edge {source_m}->{target_m}")
        target_count = counts[target_m]
        survival_fraction = float(row["survival_fraction"])
        surviving = int(round(target_count * survival_fraction))
        dropped = target_count - surviving
        if surviving < 0 or dropped < 0:
            raise SystemExit(f"Invalid survival count for variant {variant}, edge {source_m}->{target_m}")
        edges.append(
            {
                "from_m": target_m,
                "to_m": source_m,
                "prefix_projection": f"pi_{{{source_m},{target_m}}}",
                "target_pattern_cardinality": target_count,
                "survival_fraction": survival_fraction,
                "surviving_patterns": surviving,
                "dropped_patterns": dropped,
                "mean_fiber_size": float(row["mean_fiber_size"]),
                "singleton_fraction": float(row["singleton_fraction"]),
                "mean_child_entropy": float(row["mean_child_entropy"]),
            }
        )

    return {
        "variant": variant,
        "source_summary": {
            "path": str(summary_path),
            "sha256": sha256_file(summary_path),
        },
        "pattern_cardinality": {str(scale): counts[scale] for scale in scales},
        "edges": edges,
    }


def render_figure(
    groups: list[dict],
    scales: list[int],
    *,
    png_path: Path,
    pdf_path: Path,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(13.0, 7.8))
    fig.patch.set_facecolor("#fbfaf7")
    ax.set_facecolor("#fbfaf7")
    ax.set_xlim(-0.85, len(scales) - 0.25)
    ax.set_ylim(-1.05, len(groups) + 0.85)
    ax.axis("off")

    x_positions = {scale: idx for idx, scale in enumerate(scales)}
    max_log = max(
        math.log10(count + 1)
        for group in groups
        for count in group["pattern_cardinality"].values()
    )
    min_log = min(
        math.log10(count + 1)
        for group in groups
        for count in group["pattern_cardinality"].values()
    )
    span = max(max_log - min_log, 1e-9)

    palette = {
        "B/E": "#2f5d7c",
        "I": "#4a708b",
        "F": "#6c6f55",
        "M/N": "#8a7a5a",
        "A": "#6f6673",
        "L": "#7d6a63",
    }
    survivor_color = "#314f64"
    dropped_color = "#d8d2c7"
    node_edge = "#403f3a"
    text_color = "#272521"
    muted_text = "#6f6a62"

    for idx, scale in enumerate(scales):
        ax.text(
            idx,
            len(groups) - 0.17,
            f"m={scale}",
            ha="center",
            va="bottom",
            fontsize=11,
            color=text_color,
            fontweight="bold",
        )
        ax.plot([idx, idx], [-0.25, len(groups) - 0.27], color="#ece6db", lw=0.8, zorder=0)

    for row_index, group in enumerate(groups):
        y = len(groups) - 1 - row_index
        ax.add_patch(
            Rectangle(
                (-0.72, y - 0.33),
                len(scales) + 0.16,
                0.66,
                facecolor="#ffffff" if row_index % 2 == 0 else "#f5f1e9",
                edgecolor="none",
                alpha=0.76,
                zorder=0,
            )
        )
        ax.text(
            -0.78,
            y,
            group["label"],
            ha="right",
            va="center",
            fontsize=12,
            color=palette[group["label"]],
            fontweight="bold",
        )
        ax.text(
            -0.78,
            y - 0.22,
            group["family"],
            ha="right",
            va="center",
            fontsize=7.5,
            color=muted_text,
        )

        for edge in group["edges"]:
            x0 = x_positions[edge["from_m"]]
            x1 = x_positions[edge["to_m"]]
            bar_start = x0 + 0.22
            bar_end = x1 - 0.22
            width = bar_end - bar_start
            survival = max(0.0, min(1.0, edge["survival_fraction"]))
            ax.plot([bar_start, bar_end], [y, y], color=dropped_color, lw=8.0, solid_capstyle="butt", zorder=1)
            ax.plot(
                [bar_start, bar_start + width * survival],
                [y, y],
                color=survivor_color,
                lw=8.0,
                solid_capstyle="butt",
                zorder=2,
            )
            ax.text(
                (bar_start + bar_end) * 0.5,
                y + 0.18,
                format_percent(survival),
                ha="center",
                va="center",
                fontsize=7.2,
                color=muted_text,
            )

        for scale in scales:
            x = x_positions[scale]
            count = group["pattern_cardinality"][str(scale)]
            log_count = math.log10(count + 1)
            size = 82 + 420 * ((log_count - min_log) / span)
            face = palette[group["label"]] if count else "#fbfaf7"
            ax.scatter(
                [x],
                [y],
                s=size,
                facecolor=face,
                edgecolor=node_edge,
                linewidth=1.1,
                zorder=4,
                alpha=0.94 if count else 1.0,
            )
            ax.text(
                x,
                y - 0.38,
                format_count(count),
                ha="center",
                va="top",
                fontsize=8.0,
                color=text_color if count else muted_text,
            )

    ax.text(
        -0.72,
        len(groups) + 0.70,
        title,
        ha="left",
        va="top",
        fontsize=15,
        color=text_color,
        fontweight="bold",
    )
    ax.text(
        -0.72,
        len(groups) + 0.36,
        (
            r"Nodes encode $|P_m|$; segments show prefix survival between adjacent sampled scales."
        ),
        ha="left",
        va="top",
        fontsize=9.0,
        color=muted_text,
    )
    ax.text(
        -0.72,
        -0.78,
        (
            "Segment lengths are fractions of patterns under prefix projection; "
            "they are not masses or charges."
        ),
        ha="left",
        va="center",
        fontsize=8.5,
        color=muted_text,
    )

    legend_handles = [
        Line2D([0], [0], color=survivor_color, lw=7, label="surviving patterns"),
        Line2D([0], [0], color=dropped_color, lw=7, label="dropped patterns"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#6f6673",
            markeredgecolor=node_edge,
            markersize=8,
            label=r"$|P_m|$ node",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        frameon=False,
        fontsize=8.5,
        bbox_to_anchor=(0.995, -0.07),
    )

    fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def render_report(payload: dict, png_path: Path, pdf_path: Path) -> str:
    lines = [
        "# Phase 1 Projective Tower Prefix-Survival Figure",
        "",
        "This read-only visualization is derived from canonical Phase 1 `phase1_summary.json` artifacts.",
        "",
        "## Figure semantics",
        "",
        "- Nodes encode `|P_m|` with a logarithmic node-size scale.",
        "- Dark horizontal segments encode the fraction of patterns in `P_m` with non-empty prefix fiber in the next sampled level.",
        "- Pale horizontal segments encode the remaining dropped patterns after the stability-filtered next level is applied.",
        "- The figure intentionally avoids physical analogy language; segment lengths are fractions of patterns only.",
        "",
        "## Included groups",
        "",
    ]
    for group in payload["groups"]:
        counts = ", ".join(
            f"m={scale}: {format_count(group['pattern_cardinality'][str(scale)])}"
            for scale in payload["scales"]
        )
        lines.append(f"- `{group['label']}` ({', '.join(group['represented_variants'])}): {counts}")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- PNG: `{png_path.name}`",
            f"- PDF: `{pdf_path.name}`",
            "- Raw data: `projective_tower_survival_data.json`",
            "",
        ]
    )
    return "\n".join(lines)


def source_manifest(groups: list[dict]) -> list[dict]:
    items: list[dict] = []
    for group in groups:
        for source in group["source_summaries"]:
            items.append(
                {
                    "group": group["label"],
                    "path": source["path"],
                    "sha256": source["sha256"],
                }
            )
    return items


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def format_count(value: int) -> str:
    return f"{int(value):,}".replace(",", " ")


def format_percent(value: float) -> str:
    if value >= 0.995:
        return "100%"
    if value <= 0.005:
        return "0%"
    return f"{100.0 * value:.1f}%"


if __name__ == "__main__":
    raise SystemExit(main())
