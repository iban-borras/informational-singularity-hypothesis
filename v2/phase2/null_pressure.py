from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from v2.common.naming import compact_int
from v2.phase1.report import (
    filter_runs,
    infer_family_from_latest_run,
    select_latest_per_variant,
)


def discover_phase1_runs_recursive(phase1_dir: Path) -> list[dict]:
    runs = []
    for dataset_path in sorted(phase1_dir.rglob("dataset.json")):
        try:
            with open(dataset_path, "r", encoding="utf-8") as handle:
                dataset = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if dataset.get("stage") != "phase1_tower":
            continue

        run_dir = dataset_path.parent
        summary_path = run_dir / "phase1_summary.json"
        if not summary_path.is_file():
            continue
        try:
            with open(summary_path, "r", encoding="utf-8") as handle:
                summary = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue

        generated_at = dataset.get("generated_at")
        if not generated_at:
            generated_at = datetime.fromtimestamp(dataset_path.stat().st_mtime).isoformat(timespec="seconds")
        runs.append(
            {
                "dataset": dataset,
                "summary": summary,
                "_dataset_path": str(dataset_path.resolve()),
                "_summary_path": str(summary_path.resolve()),
                "_run_dir": str(run_dir.resolve()),
                "_generated_at_dt": datetime.fromisoformat(generated_at),
                "generated_at": generated_at,
            }
        )
    return runs


def is_observed_run(run: dict) -> bool:
    config = run["dataset"]["config"]
    return str(config.get("sequence_kind", "observed")) != "null_surrogate"


def supports_pattern_scale(run: dict, *, pattern_scale: int) -> bool:
    run_dir = Path(run["_run_dir"])
    path = run_dir / "pattern_spaces" / f"pattern_space_m{pattern_scale}.json"
    return path.is_file()


def parse_null_models(raw: str) -> list[str]:
    models = []
    seen = set()
    for token in raw.split(","):
        item = _normalize_null_model_name(token)
        if not item:
            continue
        if item not in {"markov1", "matched-lz"}:
            raise ValueError(f"Unsupported null model {token!r}.")
        if item in seen:
            continue
        seen.add(item)
        models.append(item)
    return models


def parse_seed_list(raw: str) -> list[int]:
    seeds = []
    seen = set()
    for token in raw.split(","):
        item = token.strip()
        if not item:
            continue
        value = int(item)
        if value in seen:
            continue
        seen.add(value)
        seeds.append(value)
    return seeds


def select_null_pressure_runs(
    runs: list[dict],
    *,
    observed_variant: str,
    null_models: list[str],
    matched_lz_seeds: list[int] | None,
    iteration: int | None,
    segment_bits: int | None,
    num_segments: int | None,
    segment_offset_bits: int | None,
    scales: list[int] | None,
    policies: list[str] | None,
    pattern_scale: int,
    no_family_inference: bool,
) -> dict:
    observed_variant = observed_variant.upper()
    observed_runs = [run for run in runs if is_observed_run(run)]
    null_runs = [run for run in runs if not is_observed_run(run)]

    family = None
    explicit_filters = any(
        value is not None
        for value in (iteration, segment_bits, num_segments, segment_offset_bits)
    ) or scales is not None or policies is not None
    if not no_family_inference and not explicit_filters:
        family_candidates = filter_runs(observed_runs, variants=[observed_variant])
        family_candidates = [
            run for run in family_candidates if supports_pattern_scale(run, pattern_scale=pattern_scale)
        ]
        family = infer_family_from_latest_run(family_candidates)

    matching_observed = filter_runs(
        observed_runs,
        variants=[observed_variant],
        iteration=iteration,
        segment_bits=segment_bits,
        num_segments=num_segments,
        segment_offset_bits=segment_offset_bits,
        scales=scales,
        policies=policies,
        family=family,
    )
    matching_observed = [
        run for run in matching_observed if supports_pattern_scale(run, pattern_scale=pattern_scale)
    ]
    selected_observed = select_latest_per_variant(matching_observed, variant_order=[observed_variant])
    if len(selected_observed) != 1:
        raise ValueError(f"Expected exactly one observed run for {observed_variant}.")
    observed_run = selected_observed[0]

    matching_nulls = filter_runs(
        null_runs,
        iteration=iteration,
        segment_bits=segment_bits,
        num_segments=num_segments,
        segment_offset_bits=segment_offset_bits,
        scales=scales,
        policies=policies,
        family=family,
    )
    matching_nulls = [
        run
        for run in matching_nulls
        if supports_pattern_scale(run, pattern_scale=pattern_scale)
        and str(run["dataset"]["config"].get("source_variant", "")).upper() == observed_variant
    ]
    if not matching_nulls:
        raise ValueError(f"No null runs matched the requested selection for source variant {observed_variant}.")

    null_specs = []
    for model in null_models:
        if model == "markov1":
            candidates = [
                run
                for run in matching_nulls
                if _normalize_null_model_name(run["dataset"]["config"].get("null_model")) == "markov1"
            ]
            if not candidates:
                raise ValueError("No markov1 null runs matched the requested selection.")
            latest = max(candidates, key=lambda item: item["_generated_at_dt"])
            null_specs.append(_build_null_spec(latest))
            continue

        if model == "matched-lz":
            candidates = [
                run
                for run in matching_nulls
                if _normalize_null_model_name(run["dataset"]["config"].get("null_model")) == "matched-lz"
            ]
            if not candidates:
                raise ValueError("No matched-lz null runs matched the requested selection.")

            latest_by_seed: dict[int, dict] = {}
            for run in candidates:
                seed = int(run["dataset"]["config"].get("null_seed", 0))
                current = latest_by_seed.get(seed)
                if current is None or run["_generated_at_dt"] > current["_generated_at_dt"]:
                    latest_by_seed[seed] = run

            selected_seeds = matched_lz_seeds or sorted(latest_by_seed)
            missing = [seed for seed in selected_seeds if seed not in latest_by_seed]
            if missing:
                raise ValueError(
                    "Missing matched-lz runs for seeds: " + ", ".join(str(seed) for seed in missing)
                )
            for seed in selected_seeds:
                null_specs.append(_build_null_spec(latest_by_seed[seed]))
            continue

    return {
        "observed_run": observed_run,
        "null_specs": null_specs,
        "family_inferred": family is not None,
    }


def build_null_pressure_aggregate_slug(
    *,
    observed_variant: str,
    pattern_scale: int,
    pattern_selection: str,
    top_patterns: int,
    start_offset_bits: int,
    window_step_bits: int,
    window_count: int,
    timestamp: str,
) -> str:
    start_part = f"__off-{compact_int(start_offset_bits)}" if start_offset_bits > 0 else ""
    return (
        f"phase2-return-lag-null-pressure__obs-{observed_variant}"
        f"__m-{pattern_scale}__sel-{pattern_selection}__top-{top_patterns}"
        f"__w-{window_count}{start_part}__step-{compact_int(window_step_bits)}__{timestamp}"
    )


def build_null_pressure_pair_slug(
    *,
    observed_variant: str,
    null_spec: dict,
    pattern_scale: int,
    pattern_selection: str,
    top_patterns: int,
    start_offset_bits: int,
    window_step_bits: int,
    window_count: int,
    timestamp: str,
) -> str:
    start_part = f"__off-{compact_int(start_offset_bits)}" if start_offset_bits > 0 else ""
    seed_part = (
        f"__seed-{null_spec['null_seed']}"
        if null_spec.get("null_seed") is not None
        else ""
    )
    return (
        f"phase2-return-lag-window-sweep__obs-{observed_variant}"
        f"__null-{null_spec['slug_token']}{seed_part}"
        f"__m-{pattern_scale}__sel-{pattern_selection}__top-{top_patterns}"
        f"__w-{window_count}{start_part}__step-{compact_int(window_step_bits)}__{timestamp}"
    )


def render_null_pressure_console_summary(rows: list[dict]) -> str:
    lines = [
        "Phase 2 null-pressure window sweep",
        "-" * 132,
        f"{'null':<28}{'seed':>8}{'js_min':>10}{'js_avg':>10}{'js_max':>10}{'d_avg':>10}{'top_avg':>10}{'worst_off':>12}{'close_off':>12}",
    ]
    for summary in summarize_null_pressure_rows(rows):
        lines.append(
            f"{summary['null_label']:<28}"
            f"{_fmt_seed(summary['null_seed']):>8}"
            f"{summary['min_profile_js']:>10.4f}"
            f"{summary['avg_profile_js']:>10.4f}"
            f"{summary['max_profile_js']:>10.4f}"
            f"{summary['avg_d_mean']:>10.4f}"
            f"{summary['avg_top_eq']:>10.4f}"
            f"{summary['worst_offset_bits']:>12}"
            f"{summary['closest_offset_bits']:>12}"
        )
    return "\n".join(lines)


def render_null_pressure_report(
    *,
    selection: dict,
    rows: list[dict],
    artifacts: list[dict],
) -> str:
    lines = [
        "# Phase 2 Return-Lag Null Pressure Sweep",
        "",
        "## Selection",
        "",
        f"- Observed variant: {selection['observed_variant']}",
        f"- Null models: {', '.join(selection['null_models'])}",
        f"- Segment protocol: {selection['num_segments']} x {selection['segment_bits']}",
        f"- Pattern scale: {selection['pattern_scale']}",
        f"- Top patterns: {selection['top_patterns']}",
        f"- Pattern selection: {selection['pattern_selection']}",
        f"- Selection anchor: {selection.get('selection_anchor_mode', 'run-local')} from {selection.get('selection_anchor_variant', selection['observed_variant'])}",
        f"- Long-lag threshold: {selection['long_lag_threshold']}",
        f"- Start offset bits: {selection['start_offset_bits']}",
        f"- Window step bits: {selection['window_step_bits']}",
        f"- Window count: {selection['window_count']}",
        "",
        "## Frozen Null Runs",
        "",
    ]
    for artifact in artifacts:
        seed = (
            f" | seed={artifact['null_seed']}"
            if artifact.get("null_seed") is not None
            else ""
        )
        lines.append(
            f"- `{artifact['null_label']}` | model={artifact['null_model']}{seed} | run_dir=`{artifact['run_dir']}`"
        )

    lines.extend(
        [
            "",
            "## Envelope By Null",
            "",
            "| Null | Model | Seed | Prof JS min | Prof JS avg | Prof JS max | Prof wJ min | Prof wJ avg | Prof wJ max | d_mean min | d_mean avg | d_mean max | top_eq min | top_eq avg | top_eq max | Worst offset | Closest offset |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for summary in summarize_null_pressure_rows(rows):
        lines.append(
            f"| {summary['null_label']} | {summary['null_model']} | {_fmt_seed(summary['null_seed'])} | "
            f"{summary['min_profile_js']:.4f} | {summary['avg_profile_js']:.4f} | {summary['max_profile_js']:.4f} | "
            f"{summary['min_profile_wj']:.4f} | {summary['avg_profile_wj']:.4f} | {summary['max_profile_wj']:.4f} | "
            f"{summary['min_d_mean']:.4f} | {summary['avg_d_mean']:.4f} | {summary['max_d_mean']:.4f} | "
            f"{summary['min_top_eq']:.4f} | {summary['avg_top_eq']:.4f} | {summary['max_top_eq']:.4f} | "
            f"{summary['worst_offset_bits']} | {summary['closest_offset_bits']} |"
        )

    lines.extend(
        [
            "",
            "Interpretation note: `worst_offset_bits` marks the window of maximum profile JS divergence, while `closest_offset_bits` marks the window where the null most closely imitates the observed return profile on that same axis.",
            "",
            "## Window Pair Readout",
            "",
            "| Null | Model | Seed | Window | Offset | Shared | Sel J | Prof wJ | Prof JS | d_mean | top_eq | long_eq |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in _sort_null_pressure_rows(rows):
        lines.append(
            f"| {row['null_label']} | {row['null_model']} | {_fmt_seed(row.get('null_seed'))} | "
            f"{row['window_index']} | {row['segment_offset_bits']} | {row['shared_pattern_count']} | "
            f"{row['selection_jaccard']:.4f} | {row['weighted_profile_jaccard']:.4f} | "
            f"{row['weighted_profile_js_divergence']:.4f} | {row['weighted_relative_mean_lag_gap']:.4f} | "
            f"{row['weighted_top_bin_match_fraction']:.4f} | {row['weighted_long_lag_match_fraction']:.4f} |"
        )

    return "\n".join(lines)


def summarize_null_pressure_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int | None], list[dict]] = {}
    for row in rows:
        key = (row["null_model"], row.get("null_seed"))
        grouped.setdefault(key, []).append(row)

    summaries = []
    for (_null_model, _null_seed), group in grouped.items():
        min_js_row = min(group, key=lambda item: item["weighted_profile_js_divergence"])
        max_js_row = max(group, key=lambda item: item["weighted_profile_js_divergence"])
        min_wj_row = min(group, key=lambda item: item["weighted_profile_jaccard"])
        max_wj_row = max(group, key=lambda item: item["weighted_profile_jaccard"])
        min_d_mean_row = min(group, key=lambda item: item["weighted_relative_mean_lag_gap"])
        max_d_mean_row = max(group, key=lambda item: item["weighted_relative_mean_lag_gap"])
        min_top_eq_row = min(group, key=lambda item: item["weighted_top_bin_match_fraction"])
        best_top_eq_row = max(group, key=lambda item: item["weighted_top_bin_match_fraction"])
        group_size = float(len(group))
        avg_profile_js = sum(item["weighted_profile_js_divergence"] for item in group) / group_size
        avg_profile_wj = sum(item["weighted_profile_jaccard"] for item in group) / group_size
        avg_d_mean = sum(item["weighted_relative_mean_lag_gap"] for item in group) / group_size
        avg_top_eq = sum(item["weighted_top_bin_match_fraction"] for item in group) / group_size
        summaries.append(
            {
                "null_label": max_js_row["null_label"],
                "null_model": max_js_row["null_model"],
                "null_seed": max_js_row.get("null_seed"),
                "min_profile_js": min_js_row["weighted_profile_js_divergence"],
                "avg_profile_js": avg_profile_js,
                "max_profile_js": max_js_row["weighted_profile_js_divergence"],
                "min_profile_wj": min_wj_row["weighted_profile_jaccard"],
                "avg_profile_wj": avg_profile_wj,
                "max_profile_wj": max_wj_row["weighted_profile_jaccard"],
                "min_d_mean": min_d_mean_row["weighted_relative_mean_lag_gap"],
                "avg_d_mean": avg_d_mean,
                "max_d_mean": max_d_mean_row["weighted_relative_mean_lag_gap"],
                "min_top_eq": min_top_eq_row["weighted_top_bin_match_fraction"],
                "avg_top_eq": avg_top_eq,
                "max_top_eq": best_top_eq_row["weighted_top_bin_match_fraction"],
                "worst_offset_bits": max_js_row["segment_offset_bits"],
                "closest_offset_bits": min_js_row["segment_offset_bits"],
                "best_top_eq": best_top_eq_row["weighted_top_bin_match_fraction"],
            }
        )

    return sorted(
        summaries,
        key=lambda item: (_null_model_sort_key(item["null_model"]), item["null_seed"] or -1),
    )


def _build_null_spec(run: dict) -> dict:
    config = run["dataset"]["config"]
    null_model = _normalize_null_model_name(config.get("null_model"))
    null_seed = config.get("null_seed")
    null_seed = int(null_seed) if null_seed is not None else None
    variant = str(config.get("variant", "null"))
    return {
        "run": run,
        "variant": variant,
        "comparison_variant": variant.upper(),
        "null_model": null_model,
        "null_seed": null_seed,
        "null_label": _build_null_label(variant, null_model, null_seed),
        "slug_token": _slugify_variant(variant),
    }


def _build_null_label(variant: str, null_model: str, null_seed: int | None) -> str:
    if null_seed is None:
        return variant
    return f"{variant} [seed {null_seed}]"


def _normalize_null_model_name(raw: object) -> str:
    if raw is None:
        return ""
    item = str(raw).strip().lower().replace("_", "-")
    if item == "markov-1":
        return "markov1"
    return item


def _slugify_variant(variant: str) -> str:
    return variant.strip().lower().replace(" ", "-").replace("_", "-")


def _fmt_seed(seed: int | None) -> str:
    return "" if seed is None else str(seed)


def _sort_null_pressure_rows(rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda item: (
            _null_model_sort_key(item["null_model"]),
            item.get("null_seed") or -1,
            item["window_index"],
        ),
    )


def _null_model_sort_key(null_model: str) -> tuple[int, str]:
    if null_model == "markov1":
        return (0, null_model)
    if null_model == "matched-lz":
        return (1, null_model)
    return (9, null_model)
