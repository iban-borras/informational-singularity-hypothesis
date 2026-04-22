from __future__ import annotations

from copy import deepcopy
from itertools import combinations
from typing import Callable

from v2.phase2.return_lag import build_return_lag_rows, prepare_frozen_source_cache
from v2.phase2.return_profile_compare import compare_return_profiles


def build_window_sweep(
    runs: list[dict],
    *,
    pattern_scale: int,
    top_patterns: int,
    pattern_selection: str,
    long_lag_threshold: int,
    comparison_pairs: list[tuple[str, str]],
    start_offset_bits: int,
    window_step_bits: int,
    window_count: int,
    selection_overrides: dict[str, dict] | None = None,
    show_progress: bool = False,
    progress_callback: Callable[[int, int, int], None] | None = None,
) -> dict:
    if not runs:
        raise ValueError("At least one Phase 1 run is required.")
    if window_count <= 0:
        raise ValueError("window_count must be positive.")
    if window_step_bits <= 0:
        raise ValueError("window_step_bits must be positive.")

    segment_bits, num_segments = _infer_segment_protocol(runs)
    window_span_bits = segment_bits * num_segments
    frozen_total_bits = start_offset_bits + ((window_count - 1) * window_step_bits) + window_span_bits
    selected_variants = [str(run["dataset"]["config"]["variant"]).upper() for run in runs]
    _validate_pairs(comparison_pairs, selected_variants)
    selection_override_summary = _summarize_selection_overrides(selection_overrides)
    frozen_source_cache = prepare_frozen_source_cache(
        runs,
        required_bits=frozen_total_bits,
        show_progress=show_progress,
    )

    windows = []
    pair_rows = []
    variant_rows = []

    for window_index in range(window_count):
        segment_offset_bits = start_offset_bits + (window_index * window_step_bits)
        if progress_callback is not None:
            progress_callback(window_index, segment_offset_bits, window_count)
        window_runs = [_clone_run_with_offset(run, segment_offset_bits) for run in runs]
        return_lag_rows = build_return_lag_rows(
            window_runs,
            pattern_scale=pattern_scale,
            top_patterns=top_patterns,
            pattern_selection=pattern_selection,
            long_lag_threshold=long_lag_threshold,
            selection_overrides=selection_overrides,
            preloaded_sources=frozen_source_cache,
            show_progress=show_progress,
        )
        rows_by_variant = {str(row["variant"]).upper(): row for row in return_lag_rows}

        comparisons = []
        for left_variant, right_variant in comparison_pairs:
            comparison = compare_return_profiles(
                rows_by_variant[left_variant],
                rows_by_variant[right_variant],
            )
            comparison["window_index"] = window_index
            comparison["segment_offset_bits"] = segment_offset_bits
            comparison["window_start_bits"] = segment_offset_bits
            comparison["window_end_bits"] = segment_offset_bits + window_span_bits
            comparison["pair"] = f"{left_variant}:{right_variant}"
            comparisons.append(comparison)
            pair_rows.append(
                {
                    "window_index": window_index,
                    "segment_offset_bits": segment_offset_bits,
                    "window_start_bits": segment_offset_bits,
                    "window_end_bits": segment_offset_bits + window_span_bits,
                    "pair": comparison["pair"],
                    "left_variant": left_variant,
                    "right_variant": right_variant,
                    "shared_pattern_count": comparison["shared_pattern_count"],
                    "union_pattern_count": comparison["union_pattern_count"],
                    "selection_jaccard": comparison["selection_jaccard"],
                    "weighted_profile_jaccard": comparison["weighted_profile_jaccard"],
                    "weighted_profile_js_divergence": comparison["weighted_profile_js_divergence"],
                    "weighted_relative_mean_lag_gap": comparison["weighted_relative_mean_lag_gap"],
                    "weighted_relative_median_lag_gap": comparison["weighted_relative_median_lag_gap"],
                    "weighted_top_bin_match_fraction": comparison["weighted_top_bin_match_fraction"],
                    "weighted_long_lag_match_fraction": comparison["weighted_long_lag_match_fraction"],
                }
            )

        for row in return_lag_rows:
            variant_rows.append(
                {
                    "window_index": window_index,
                    "segment_offset_bits": segment_offset_bits,
                    "window_start_bits": segment_offset_bits,
                    "window_end_bits": segment_offset_bits + window_span_bits,
                    "variant": row["variant"],
                    "selected_pattern_count": row["selected_pattern_count"],
                    "total_occurrence_count": row["total_occurrence_count"],
                    "total_return_count": row["total_return_count"],
                    "mean_return_lag": row["mean_return_lag"],
                    "median_return_lag": row["median_return_lag"],
                    "lag_entropy": row["lag_entropy"],
                    "dominant_bin_mass_fraction": row["dominant_bin_mass_fraction"],
                    "long_lag_fraction": row["long_lag_fraction"],
                    "segment_profile_jaccard": row["segment_profile_jaccard"],
                    "segment_profile_weighted_jaccard": row["segment_profile_weighted_jaccard"],
                    "selection_source_variant": row.get("selection_source_variant"),
                    "selection_anchor_kind": row.get("selection_anchor_kind"),
                    "source_freeze_mode": row.get("source_freeze_mode"),
                    "frozen_total_bits": row.get("frozen_total_bits"),
                }
            )

        windows.append(
            {
                "window_index": window_index,
                "segment_offset_bits": segment_offset_bits,
                "window_start_bits": segment_offset_bits,
                "window_end_bits": segment_offset_bits + window_span_bits,
                "return_lag_rows": return_lag_rows,
                "comparisons": comparisons,
            }
        )

    ranked_pairs = sorted(
        pair_rows,
        key=lambda item: (
            -item["weighted_profile_js_divergence"],
            item["weighted_profile_jaccard"],
            -item["weighted_relative_mean_lag_gap"],
            item["window_index"],
            item["pair"],
        ),
    )

    return {
        "segment_bits": segment_bits,
        "num_segments": num_segments,
        "window_span_bits": window_span_bits,
        "window_count": window_count,
        "start_offset_bits": start_offset_bits,
        "window_step_bits": window_step_bits,
        "source_freeze_mode": "sweep-global-frozen",
        "frozen_total_bits": frozen_total_bits,
        "comparison_pairs": [f"{left}:{right}" for left, right in comparison_pairs],
        "selection_override_variants": selection_override_summary["variants"],
        "selection_override_sources": selection_override_summary["sources"],
        "windows": windows,
        "pair_rows": pair_rows,
        "variant_rows": variant_rows,
        "top_pair_rows": ranked_pairs[: min(10, len(ranked_pairs))],
        "notes": [
            "Window sweep freezes the selected Phase 1 runs and the chosen pattern list for each evaluated variant.",
            "The only intentional mutation across windows is segment_offset_bits; source bitstreams are generated once at the sweep-global required length and then sliced window-by-window.",
            "Pairwise comparisons reuse the existing return-profile divergence machinery on observed rows window-by-window.",
        ],
    }


def render_window_sweep_report(
    sweep: dict,
    *,
    variants: list[str],
    pattern_scale: int,
    top_patterns: int,
    pattern_selection: str,
    long_lag_threshold: int,
    frozen_runs: list[dict],
) -> str:
    lines = [
        "# Phase 2 Return-Lag Window Sweep",
        "",
        "## Selection",
        "",
        f"- Variants: {', '.join(variants)}",
        f"- Segment protocol: {sweep['num_segments']} x {sweep['segment_bits']}",
        f"- Pattern scale: {pattern_scale}",
        f"- Top patterns: {top_patterns}",
        f"- Pattern selection: {pattern_selection}",
        f"- Source freeze mode: {sweep.get('source_freeze_mode', 'run-local')}",
        f"- Frozen source bits: {sweep.get('frozen_total_bits', '-')}",
    ]
    if sweep.get("selection_override_variants"):
        sources = ", ".join(sweep.get("selection_override_sources", [])) or "external source"
        variants = ", ".join(sweep["selection_override_variants"])
        lines.append(
            f"- Selection anchor: fixed external pattern list from {sources} applied to {variants}"
        )
    lines.extend(
        [
        f"- Long-lag threshold: {long_lag_threshold}",
        f"- Start offset bits: {sweep['start_offset_bits']}",
        f"- Window step bits: {sweep['window_step_bits']}",
        f"- Window count: {sweep['window_count']}",
        f"- Window span bits: {sweep['window_span_bits']}",
        f"- Comparison pairs: {', '.join(sweep['comparison_pairs']) if sweep['comparison_pairs'] else 'none'}",
        "",
        "## Frozen Phase 1 Runs",
        "",
        ]
    )
    for run in frozen_runs:
        config = run["dataset"]["config"]
        lines.append(
            f"- `{config['variant']}` | run_dir=`{run['_run_dir']}` | generated_at={run['generated_at']}"
        )

    lines.extend(
        [
            "",
            "## Top Pair Windows",
            "",
            "| Window | Offset | Pair | Shared | Sel J | Prof wJ | Prof JS | d_mean | top_eq |",
            "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sweep["top_pair_rows"]:
        lines.append(
            f"| {row['window_index']} | {row['segment_offset_bits']} | {row['pair']} | "
            f"{row['shared_pattern_count']} | {_fmt(row['selection_jaccard'])} | "
            f"{_fmt(row['weighted_profile_jaccard'])} | {_fmt(row['weighted_profile_js_divergence'])} | "
            f"{_fmt(row['weighted_relative_mean_lag_gap'])} | {_fmt(row['weighted_top_bin_match_fraction'])} |"
        )

    lines.extend(
        [
            "",
            "## Window Pair Readout",
            "",
            "| Window | Offset | Pair | Shared | Sel J | Prof wJ | Prof JS | d_mean | top_eq | long_eq |",
            "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sweep["pair_rows"]:
        lines.append(
            f"| {row['window_index']} | {row['segment_offset_bits']} | {row['pair']} | "
            f"{row['shared_pattern_count']} | {_fmt(row['selection_jaccard'])} | "
            f"{_fmt(row['weighted_profile_jaccard'])} | {_fmt(row['weighted_profile_js_divergence'])} | "
            f"{_fmt(row['weighted_relative_mean_lag_gap'])} | "
            f"{_fmt(row['weighted_top_bin_match_fraction'])} | "
            f"{_fmt(row['weighted_long_lag_match_fraction'])} |"
        )

    lines.extend(
        [
            "",
            "## Variant Window Readout",
            "",
            "| Window | Offset | Variant | Freeze mode | Selected | Occ | Ret | Mean lag | Lag H | Dom bin | Long | seg J | seg wJ |",
            "| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sweep["variant_rows"]:
        lines.append(
            f"| {row['window_index']} | {row['segment_offset_bits']} | {row['variant']} | {row.get('source_freeze_mode', '-')} | "
            f"{row['selected_pattern_count']} | {row['total_occurrence_count']} | {row['total_return_count']} | "
            f"{_fmt(row['mean_return_lag'])} | {_fmt(row['lag_entropy'])} | "
            f"{_fmt(row['dominant_bin_mass_fraction'])} | {_fmt(row['long_lag_fraction'])} | "
            f"{_fmt(row['segment_profile_jaccard'])} | {_fmt(row['segment_profile_weighted_jaccard'])} |"
        )

    return "\n".join(lines)


def render_window_sweep_console_summary(sweep: dict) -> str:
    lines = [
        "Phase 2 return-lag window sweep",
        "-" * 126,
        f"{'win':>4}{'offset':>12}{'pair':>10}{'shared':>8}{'sel_J':>10}{'prof_wJ':>10}{'prof_JS':>10}{'d_mean':>10}{'top_eq':>10}",
    ]
    for row in sweep["pair_rows"]:
        lines.append(
            f"{row['window_index']:>4}"
            f"{row['segment_offset_bits']:>12}"
            f"{row['pair']:>10}"
            f"{row['shared_pattern_count']:>8}"
            f"{row['selection_jaccard']:>10.4f}"
            f"{row['weighted_profile_jaccard']:>10.4f}"
            f"{row['weighted_profile_js_divergence']:>10.4f}"
            f"{row['weighted_relative_mean_lag_gap']:>10.4f}"
            f"{row['weighted_top_bin_match_fraction']:>10.4f}"
        )
    return "\n".join(lines)


def default_comparison_pairs(variants: list[str]) -> list[tuple[str, str]]:
    ordered = [item.upper() for item in variants]
    return [(left, right) for left, right in combinations(ordered, 2)]


def parse_comparison_pairs(raw: str, *, variants: list[str]) -> list[tuple[str, str]]:
    if not raw.strip():
        return default_comparison_pairs(variants)

    pairs = []
    seen = set()
    valid_variants = {item.upper() for item in variants}
    for token in raw.split(","):
        item = token.strip()
        if not item:
            continue
        left, separator, right = item.partition(":")
        if separator != ":":
            raise ValueError(f"Invalid comparison pair {item!r}; expected LEFT:RIGHT.")
        left = left.strip().upper()
        right = right.strip().upper()
        if left == right:
            raise ValueError(f"Comparison pair {item!r} repeats the same variant.")
        if left not in valid_variants or right not in valid_variants:
            raise ValueError(
                f"Comparison pair {item!r} references variants outside the selected set {sorted(valid_variants)}."
            )
        key = (left, right)
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)
    return pairs


def _clone_run_with_offset(run: dict, segment_offset_bits: int) -> dict:
    cloned = dict(run)
    dataset = deepcopy(run["dataset"])
    config = dataset["config"]
    segment_bits = int(config["segment_bits"])
    num_segments = int(config["num_segments"])
    config["segment_offset_bits"] = segment_offset_bits
    config["loaded_observable_bits"] = segment_offset_bits + (segment_bits * num_segments)
    cloned["dataset"] = dataset
    return cloned


def _infer_segment_protocol(runs: list[dict]) -> tuple[int, int]:
    first = runs[0]["dataset"]["config"]
    segment_bits = int(first["segment_bits"])
    num_segments = int(first["num_segments"])
    for run in runs[1:]:
        config = run["dataset"]["config"]
        if int(config["segment_bits"]) != segment_bits or int(config["num_segments"]) != num_segments:
            raise ValueError("All selected runs must share the same segment protocol.")
    return segment_bits, num_segments


def _validate_pairs(pairs: list[tuple[str, str]], variants: list[str]) -> None:
    available = {item.upper() for item in variants}
    for left, right in pairs:
        if left not in available or right not in available:
            raise ValueError(
                f"Comparison pair {left}:{right} is not available in the selected runs {sorted(available)}."
            )


def _summarize_selection_overrides(selection_overrides: dict[str, dict] | None) -> dict[str, list[str]]:
    if not selection_overrides:
        return {"variants": [], "sources": []}
    return {
        "variants": sorted({str(key).upper() for key in selection_overrides}),
        "sources": sorted(
            {
                str(value.get("source_variant"))
                for value in selection_overrides.values()
                if value.get("source_variant")
            }
        ),
    }


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"
