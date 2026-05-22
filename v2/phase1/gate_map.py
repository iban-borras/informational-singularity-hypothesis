from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from v2.phase1.coherence import build_coherence_rows
from v2.phase1.report import (
    build_report_rows,
    discover_phase1_runs,
    filter_runs,
    select_latest_per_variant,
)
from v2.phase1.transport import build_transport_rows


DEFAULT_OBSERVED_VARIANTS = ["B", "E", "I", "F", "M", "N", "A", "J", "K", "L", "O", "P"]
DEFAULT_NULL_VARIANTS = [
    "B-shuffled",
    "B-same-density",
    "B-markov1",
    "B-matched-lz",
    "E-matched-lz",
    "I-matched-lz",
]

PLOT_STYLE = {
    "HSI coherent": {"fill": "#0f766e", "stroke": "#0b4f4a"},
    "Rigid / crystalline control": {"fill": "#b7791f", "stroke": "#7c4f14"},
    "Collapsed control": {"fill": "#c2410c", "stroke": "#8a2d08"},
    "Trivial control": {"fill": "#6b7280", "stroke": "#4b5563"},
    "Recursive external control": {"fill": "#7c3aed", "stroke": "#5b21b6"},
    "Randomized null": {"fill": "#dc6b19", "stroke": "#9a4a11"},
    "Markov-1 null": {"fill": "#2563eb", "stroke": "#1d4ed8"},
    "Matched-LZ null": {"fill": "#64748b", "stroke": "#475569"},
}


def load_gate_map_rows(
    observed_dir: Path,
    null_dir: Path,
    *,
    observed_variants: list[str] | None = None,
    null_variants: list[str] | None = None,
    extra_observed_dirs: list[Path] | None = None,
    tail_edges: int = 2,
) -> tuple[list[dict], dict]:
    observed_variants = observed_variants or list(DEFAULT_OBSERVED_VARIANTS)
    null_variants = null_variants or list(DEFAULT_NULL_VARIANTS)

    observed_runs = discover_phase1_runs(observed_dir)
    for extra_dir in extra_observed_dirs or []:
        observed_runs.extend(discover_phase1_runs(extra_dir))
    null_runs = discover_phase1_runs(null_dir)
    if not observed_runs:
        raise ValueError(f"No valid observed Phase 1 runs found in {observed_dir}")
    if not null_runs:
        raise ValueError(f"No valid null Phase 1 runs found in {null_dir}")

    observed_selected, observed_family = _select_consensus_batch(
        observed_runs,
        variants=observed_variants,
    )
    null_selected, null_family = _select_consensus_batch(
        null_runs,
        variants=null_variants,
    )

    combined = observed_selected + null_selected
    report_rows = {row["variant"]: row for row in build_report_rows(combined)}
    coherence_rows = {row["variant"]: row for row in build_coherence_rows(combined, tail_edges=tail_edges)}
    transport_rows = {row["variant"]: row for row in build_transport_rows(combined, tail_edges=tail_edges)}

    order = _ordered_existing_variants(observed_variants, report_rows)
    order.extend(_ordered_existing_variants(null_variants, report_rows))

    rows = []
    for variant in order:
        report_row = report_rows[variant]
        coherence_row = coherence_rows[variant]
        transport_row = transport_rows[variant]
        rows.append(
            {
                "variant": variant,
                "source_kind": _source_kind(variant),
                "regime_bucket": _regime_bucket(variant),
                "plot_group": _plot_group(variant),
                "iteration": report_row["iteration"],
                "segment_bits": report_row["segment_bits"],
                "num_segments": report_row["num_segments"],
                "segment_offset_bits": report_row["segment_offset_bits"],
                "kept_last": report_row["kept_last"],
                "candidate_last": report_row["candidate_last"],
                "breadth_ratio_last_to_first": report_row["breadth_ratio_last_to_first"],
                "retention_last": report_row["retention_last"],
                "tail_survival_last": coherence_row["tail_survival_last"],
                "tail_survival_mean": coherence_row["tail_survival_mean"],
                "tail_entropy_last": coherence_row["tail_entropy_last"],
                "selective_persistence_index_simple": coherence_row["selective_persistence_index_simple"],
                "tail_transport_active_mean": transport_row["tail_transport_active_mean"],
                "tail_transport_active_last": transport_row["tail_transport_active_last"],
                "tail_transport_decay_last": transport_row["tail_transport_decay_last"],
                "run_dir": report_row["run_dir"],
                "summary_path": report_row["summary_path"],
                "generated_at": report_row["generated_at"],
            }
        )

    selection = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "tail_edges": tail_edges,
        "observed_variants": [row["variant"] for row in rows if row["source_kind"] == "observed"],
        "null_variants": [row["variant"] for row in rows if row["source_kind"] != "observed"],
        "observed_family": observed_family,
        "null_family": null_family,
        "extra_observed_dirs": [str(path) for path in extra_observed_dirs or []],
    }
    return rows, selection


def render_console_summary(rows: list[dict]) -> str:
    lines = [
        "Phase 1 Gate Map (selectivity vs persistence summary)",
        "-" * 126,
        f"{'variant':<16} {'source':<12} {'regime':<28} {'k@last':>8} {'cand@last':>10} {'ret':>8} {'surv':>8} {'spi':>8} {'act_mean':>10}",
    ]
    for row in rows:
        lines.append(
            f"{row['variant']:<16} "
            f"{row['source_kind']:<12} "
            f"{row['regime_bucket']:<28} "
            f"{row['kept_last']:>8d} "
            f"{row['candidate_last']:>10d} "
            f"{_format_float(row['retention_last']):>8} "
            f"{_format_float(row['tail_survival_last']):>8} "
            f"{_format_float(row['selective_persistence_index_simple']):>8} "
            f"{_format_float(row['tail_transport_active_mean']):>10}"
        )
    return "\n".join(lines)


def render_markdown_report(rows: list[dict], selection: dict, *, svg_filename: str | None = None) -> str:
    observed_rows = [row for row in rows if row["source_kind"] == "observed"]
    null_rows = [row for row in rows if row["source_kind"] != "observed"]

    lines = [
        "# HSI v2 Phase 1 Gate Map",
        "",
        "## Selection",
        "",
        f"- Generated at: `{selection['generated_at']}`",
        f"- Tail edges used: `{selection['tail_edges']}`",
        f"- Observed variants: `{', '.join(selection['observed_variants'])}`",
        f"- Strong-null variants: `{', '.join(selection['null_variants'])}`",
        f"- Observed consensus family: `{_format_family(selection['observed_family'])}`",
        f"- Null consensus family: `{_format_family(selection['null_family'])}`",
        "",
        "## Reading",
        "",
        "- `retention@last` asks how selective the last scale remains relative to its candidate space.",
        "- `surv_last` asks whether the top edge is still alive.",
        "- `SPI_simple = retention@last * surv_last` is the minimal selective-persistence index.",
        "- `active_mean_tail` asks whether top-edge activity still reaches children that keep branching.",
        "",
        "## Complete Table",
        "",
        "### Observed Variants",
        "",
    ]
    lines.extend(_render_markdown_table(observed_rows))
    lines.extend(["", "### Strong Nulls", ""])
    lines.extend(_render_markdown_table(null_rows))

    lines.extend(["", "## Gate Plane", ""])
    lines.append(
        "The SVG plot below uses `retention@last` on the x-axis and `active_mean_tail` on the y-axis."
    )
    lines.append(
        "This plane is purely presentational: it separates compact-selective persistence from diffuse or rigid persistence more clearly than `surv_last` alone."
    )
    if svg_filename:
        lines.extend(["", f"![Phase 1 Gate Plane](./{svg_filename})"])

    lines.extend(["", "## Run Paths", ""])
    for row in rows:
        lines.append(f"- `{row['variant']}`: `{row['summary_path']}`")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `regime` is a descriptive synthesis for the Phase 1 gate map, not a new canonical classifier.",
            "- `B/E/I` represent compact-selective HSI persistence in this batch.",
            "- `Markov-1` and `matched-LZ` can keep the top of the tower alive, but do so with much weaker selectivity.",
            "- `shuffled` and `same-density` collapse completely at the top of the tower.",
        ]
    )
    return "\n".join(lines)


def write_csv(rows: list[dict], csv_path: Path) -> None:
    fieldnames = [
        "variant",
        "source_kind",
        "regime_bucket",
        "iteration",
        "kept_last",
        "candidate_last",
        "breadth_ratio_last_to_first",
        "retention_last",
        "tail_survival_last",
        "tail_survival_mean",
        "selective_persistence_index_simple",
        "tail_transport_active_mean",
        "tail_transport_active_last",
        "tail_transport_decay_last",
        "summary_path",
        "run_dir",
        "generated_at",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def render_gate_svg(rows: list[dict], *, width: int = 1180, height: int = 760) -> str:
    margin_left = 88
    margin_top = 72
    margin_right = 330
    margin_bottom = 88
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    x_min, x_max = -0.04, 1.06
    y_values = [float(row["tail_transport_active_mean"] or 0.0) for row in rows]
    y_max = max(0.62, max(y_values, default=0.0) * 1.12)
    y_min = -0.03

    plot_rows = _cluster_plot_rows(rows)

    def x_pos(value: float) -> float:
        return margin_left + ((value - x_min) / max(x_max - x_min, 1e-12)) * plot_width

    def y_pos(value: float) -> float:
        return margin_top + plot_height - ((value - y_min) / max(y_max - y_min, 1e-12)) * plot_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        "<defs>",
        '<linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">',
        '<stop offset="0%" stop-color="#faf7f1" />',
        '<stop offset="100%" stop-color="#f4efe4" />',
        "</linearGradient>",
        "</defs>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="url(#bg)" />',
        '<title id="title">HSI v2 Phase 1 Gate Plane</title>',
        '<desc id="desc">Retention at the top scale versus mean tail activity for observed variants and strong nulls.</desc>',
        f'<text x="{margin_left}" y="34" font-family="Arial, Helvetica, sans-serif" font-size="26" font-weight="700" fill="#1f2937">Phase 1 Gate Plane</text>',
        f'<text x="{margin_left}" y="58" font-family="Arial, Helvetica, sans-serif" font-size="14" fill="#475569">x = retention@last | y = active_mean_tail</text>',
        f'<rect x="{margin_left}" y="{margin_top}" width="{plot_width}" height="{plot_height}" fill="#fffdf8" stroke="#cbd5e1" stroke-width="1.2" />',
    ]

    for tick in range(6):
        value = tick / 5
        xpos = x_pos(value)
        parts.append(
            f'<line x1="{xpos:.2f}" y1="{margin_top}" x2="{xpos:.2f}" y2="{margin_top + plot_height}" stroke="#e2e8f0" stroke-width="1" />'
        )
        parts.append(
            f'<text x="{xpos:.2f}" y="{margin_top + plot_height + 28}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="12" fill="#475569">{value:.1f}</text>'
        )

    y_ticks = _build_y_ticks(y_max)
    for value in y_ticks:
        ypos = y_pos(value)
        parts.append(
            f'<line x1="{margin_left}" y1="{ypos:.2f}" x2="{margin_left + plot_width}" y2="{ypos:.2f}" stroke="#e2e8f0" stroke-width="1" />'
        )
        parts.append(
            f'<text x="{margin_left - 16}" y="{ypos + 4:.2f}" text-anchor="end" font-family="Arial, Helvetica, sans-serif" font-size="12" fill="#475569">{value:.1f}</text>'
        )

    parts.append(
        f'<text x="{margin_left + (plot_width / 2):.2f}" y="{height - 28}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="15" font-weight="600" fill="#334155">retention@last</text>'
    )
    parts.append(
        f'<text transform="translate(26 {margin_top + (plot_height / 2):.2f}) rotate(-90)" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="15" font-weight="600" fill="#334155">active_mean_tail</text>'
    )

    for index, row in enumerate(plot_rows):
        style = PLOT_STYLE[row["plot_group"]]
        x_value = float(row["retention_last"] or 0.0)
        y_value = float(row["tail_transport_active_mean"] or 0.0)
        cx = x_pos(x_value)
        cy = y_pos(y_value)
        dx, dy, label_anchor = _label_offset(row, index=index, x_value=x_value, y_value=y_value, y_max=y_max)
        parts.append(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="7" fill="{style["fill"]}" stroke="{style["stroke"]}" stroke-width="1.8" />'
        )
        parts.append(
            f'<line x1="{cx:.2f}" y1="{cy:.2f}" x2="{cx + dx:.2f}" y2="{cy + dy:.2f}" stroke="{style["stroke"]}" stroke-width="1.2" />'
        )
        parts.append(
            f'<text x="{cx + dx:.2f}" y="{cy + dy - 2:.2f}" text-anchor="{label_anchor}" font-family="Arial, Helvetica, sans-serif" font-size="13" font-weight="600" fill="#1f2937">{_svg_escape(row["plot_label"])}</text>'
        )
    parts.extend(_render_svg_legend(width=width, margin_top=margin_top))
    parts.append("</svg>")
    return "\n".join(parts)


def render_gate_png(
    rows: list[dict],
    png_path: Path,
    *,
    width: int = 11.8,
    height: float = 7.6,
    dpi: int = 180,
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    x_min, x_max = -0.04, 1.06
    y_min = -0.03
    y_values = [float(row["tail_transport_active_mean"] or 0.0) for row in rows]
    y_max = max(0.62, max(y_values, default=0.0) * 1.12)
    plot_rows = _cluster_plot_rows(rows)
    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    fig.patch.set_facecolor("#faf7f1")
    ax.set_facecolor("#fffdf8")

    for row in plot_rows:
        style = PLOT_STYLE[row["plot_group"]]
        x = float(row["retention_last"] or 0.0)
        y = float(row["tail_transport_active_mean"] or 0.0)
        ax.scatter(
            [x],
            [y],
            s=90,
            c=style["fill"],
            edgecolors=style["stroke"],
            linewidths=1.6,
            zorder=3,
            clip_on=False,
        )

    for index, row in enumerate(plot_rows):
        x = float(row["retention_last"] or 0.0)
        y = float(row["tail_transport_active_mean"] or 0.0)
        dx, dy, ha = _label_offset(row, index=index, x_value=x, y_value=y, y_max=y_max)
        ax.annotate(
            row["plot_label"],
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=_matplotlib_anchor(ha),
            va="center",
            fontsize=10,
            fontweight="semibold",
            color="#1f2937",
            arrowprops={"arrowstyle": "-", "color": PLOT_STYLE[row["plot_group"]]["stroke"], "lw": 1.0},
            zorder=4,
            annotation_clip=False,
        )

    for label, style in PLOT_STYLE.items():
        ax.scatter([], [], s=70, c=style["fill"], edgecolors=style["stroke"], linewidths=1.4, label=label)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("retention@last", fontsize=12, fontweight="semibold", color="#334155")
    ax.set_ylabel("active_mean_tail", fontsize=12, fontweight="semibold", color="#334155")
    ax.set_title("Phase 1 Gate Plane", fontsize=16, fontweight="bold", color="#1f2937", loc="left", pad=14)
    ax.grid(True, which="major", color="#e2e8f0", linewidth=1.0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")
    ax.tick_params(colors="#475569")

    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        title="Legend",
        borderaxespad=0.0,
    )
    if legend is not None:
        legend.get_title().set_fontweight("bold")

    fig.tight_layout()
    fig.savefig(png_path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return True


def _select_consensus_batch(runs: list[dict], *, variants: list[str]) -> tuple[list[dict], dict]:
    candidate_runs = filter_runs(runs, variants=variants)
    if not candidate_runs:
        raise ValueError("No runs matched the requested gate-map variants.")

    family = _infer_consensus_family(candidate_runs, variants=variants)
    filtered = filter_runs(candidate_runs, variants=variants, family=family)
    selected = select_latest_per_variant(filtered, variant_order=variants)
    return selected, family


def _ordered_existing_variants(requested: list[str], rows_by_variant: dict[str, dict]) -> list[str]:
    lookup = {variant.upper(): variant for variant in rows_by_variant}
    return [lookup[item.upper()] for item in requested if item.upper() in lookup]


def _infer_consensus_family(runs: list[dict], *, variants: list[str]) -> dict:
    requested = {item.upper() for item in variants}
    grouped: dict[tuple, dict] = {}
    for run in runs:
        signature = _family_signature(run)
        bucket = grouped.setdefault(
            signature,
            {
                "family": _family_from_signature(signature),
                "variants": set(),
                "latest": run["_generated_at_dt"],
            },
        )
        bucket["variants"].add(str(run["dataset"]["config"]["variant"]).upper())
        if run["_generated_at_dt"] > bucket["latest"]:
            bucket["latest"] = run["_generated_at_dt"]

    best = max(
        grouped.values(),
        key=lambda item: (
            len(item["variants"] & requested),
            item["latest"],
        ),
    )
    return best["family"]


def _family_signature(run: dict) -> tuple:
    config = run["dataset"]["config"]
    return (
        config.get("segment_bits"),
        config.get("num_segments"),
        config.get("segment_offset_bits", 0),
        tuple(config.get("scales", [])),
        tuple(config.get("policies", [])),
        config.get("min_count_floor"),
        config.get("min_count_rate"),
        config.get("min_segment_support_effective"),
        config.get("cv_max"),
        config.get("sampling_policy"),
        config.get("pattern_space_mode"),
    )


def _family_from_signature(signature: tuple) -> dict:
    (
        segment_bits,
        num_segments,
        segment_offset_bits,
        scales,
        policies,
        min_count_floor,
        min_count_rate,
        min_segment_support_effective,
        cv_max,
        sampling_policy,
        pattern_space_mode,
    ) = signature
    return {
        "segment_bits": segment_bits,
        "num_segments": num_segments,
        "segment_offset_bits": segment_offset_bits,
        "scales": list(scales),
        "policies": list(policies),
        "min_count_floor": min_count_floor,
        "min_count_rate": min_count_rate,
        "min_segment_support_effective": min_segment_support_effective,
        "cv_max": cv_max,
        "sampling_policy": sampling_policy,
        "pattern_space_mode": pattern_space_mode,
    }


def _source_kind(variant: str) -> str:
    return "observed" if "-" not in variant else "strong_null"


def _regime_bucket(variant: str) -> str:
    if variant in {"B", "E", "I"}:
        return "HSI coherent"
    if variant in {"F", "M", "N"}:
        return "Rigid / crystalline control"
    if variant in {"A", "J", "L"}:
        return "Collapsed control"
    if variant == "K":
        return "Trivial control"
    if variant in {"O", "P"}:
        return "Recursive external control"
    if variant.endswith("shuffled") or variant.endswith("same-density"):
        return "Randomized null"
    if variant.endswith("markov1"):
        return "Markov-1 null"
    if variant.endswith("matched-lz"):
        return "Matched-LZ null"
    return "Other"


def _plot_group(variant: str) -> str:
    regime = _regime_bucket(variant)
    if regime == "HSI coherent":
        return "HSI coherent"
    if regime == "Rigid / crystalline control":
        return "Rigid / crystalline control"
    if regime == "Collapsed control":
        return "Collapsed control"
    if regime == "Trivial control":
        return "Trivial control"
    if regime == "Recursive external control":
        return "Recursive external control"
    if regime == "Randomized null":
        return "Randomized null"
    if regime == "Markov-1 null":
        return "Markov-1 null"
    return "Matched-LZ null"


def _render_markdown_table(rows: list[dict]) -> list[str]:
    if not rows:
        return ["No rows available."]
    lines = [
        "| variant | source | regime | iter | k@last | cand@last | ret@last | surv_last | SPI_simple | active_mean_tail |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['variant']} | "
            f"{row['source_kind']} | "
            f"{row['regime_bucket']} | "
            f"{row['iteration']} | "
            f"{row['kept_last']} | "
            f"{row['candidate_last']} | "
            f"{_format_float(row['retention_last'])} | "
            f"{_format_float(row['tail_survival_last'])} | "
            f"{_format_float(row['selective_persistence_index_simple'])} | "
            f"{_format_float(row['tail_transport_active_mean'])} |"
        )
    return lines


def _cluster_plot_rows(rows: list[dict]) -> list[dict]:
    clusters: dict[tuple, dict] = {}
    for row in rows:
        key = (
            row["plot_group"],
            round(float(row["retention_last"] or 0.0), 4),
            round(float(row["tail_transport_active_mean"] or 0.0), 4),
        )
        bucket = clusters.setdefault(
            key,
            {
                "plot_group": row["plot_group"],
                "retention_last": row["retention_last"],
                "tail_transport_active_mean": row["tail_transport_active_mean"],
                "variants": [],
            },
        )
        bucket["variants"].append(row["variant"])

    clustered = []
    for bucket in clusters.values():
        variants = sorted(bucket["variants"], key=lambda item: (len(item), item))
        clustered.append(
            {
                "plot_group": bucket["plot_group"],
                "retention_last": bucket["retention_last"],
                "tail_transport_active_mean": bucket["tail_transport_active_mean"],
                "plot_label": "/".join(variants),
            }
        )
    return sorted(
        clustered,
        key=lambda item: (
            float(item["retention_last"] or 0.0),
            float(item["tail_transport_active_mean"] or 0.0),
            item["plot_label"],
        ),
    )


def _label_offset(
    row: dict,
    *,
    index: int,
    x_value: float,
    y_value: float,
    y_max: float,
) -> tuple[int, int, str]:
    manual = {
        "A/J": (14, 18, "start"),
        "B-shuffled/B-same-density": (18, -14, "start"),
        "L": (-14, 26, "end"),
        "K": (-12, -18, "end"),
        "F": (12, -2, "start"),
        "M/N": (0, 28, "start"),
        "O": (-24, -18, "end"),
        "P": (-14, -24, "end"),
    }
    if row["plot_label"] in manual:
        return manual[row["plot_label"]]

    offsets = [(14, -12), (14, 18), (-12, -14), (-12, 18), (18, 2), (-18, 2)]
    dx, dy = offsets[index % len(offsets)]
    if x_value > 0.9:
        dx = -abs(dx) - 6
    if y_value > y_max * 0.92:
        dy = -abs(dy)
    anchor = "start" if dx >= 0 else "end"
    return dx, dy, anchor


def _matplotlib_anchor(svg_anchor: str) -> str:
    return "left" if svg_anchor == "start" else "right"


def _render_svg_legend(*, width: int, margin_top: int) -> list[str]:
    x = width - 282
    y = margin_top + 12
    lines = [
        f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" font-size="16" font-weight="700" fill="#1f2937">Legend</text>'
    ]
    y += 26
    for label, style in PLOT_STYLE.items():
        lines.append(
            f'<circle cx="{x + 10}" cy="{y - 5}" r="6" fill="{style["fill"]}" stroke="{style["stroke"]}" stroke-width="1.5" />'
        )
        lines.append(
            f'<text x="{x + 24}" y="{y - 1}" font-family="Arial, Helvetica, sans-serif" font-size="13" fill="#334155">{_svg_escape(label)}</text>'
        )
        y += 24
    return lines


def _build_y_ticks(y_max: float) -> list[float]:
    step = 0.1
    count = int(round(y_max / step))
    return [index * step for index in range(count + 1)]


def _format_family(family: dict) -> str:
    return (
        f"{family['num_segments']} x {family['segment_bits']} bits"
        f" | offset={family['segment_offset_bits']}"
        f" | scales={','.join(str(item) for item in family['scales'])}"
        f" | policies={','.join(family['policies'])}"
    )


def _format_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
