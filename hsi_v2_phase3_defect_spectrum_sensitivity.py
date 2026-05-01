#!/usr/bin/env python3
"""HSI v2 Phase 3 defect/spectrum sensitivity sweep."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from v2.common.cli import parse_variants, resolve_dir
from v2.common.naming import compact_int
from v2.phase1.report import filter_runs, infer_family_from_latest_run
from v2.phase1.tower import parse_scales
from v2.phase2.null_pressure import discover_phase1_runs_recursive, is_observed_run
from v2.phase3.defect_spectrum import build_defect_spectrum_rows, compact_rows
from hsi_v2_phase3_min_defect_spectrum_pilot import supports_window


CSV_FIELDS = [
    "window_label",
    "window_scales",
    "variant",
    "source_kind",
    "null_model",
    "null_seed",
    "iteration",
    "future_bridge_fraction",
    "squareable_bridge_fraction",
    "anchored_square_rate",
    "square_defect_rate",
    "square_defect_mass_rate",
    "bridge_disagreement_rate",
    "mean_square_distance",
    "square_entropy",
    "square_segment_consistency",
    "node_count",
    "edge_count",
    "largest_component_fraction",
    "recurrent_edge_fraction",
    "recurrent_edge_mass_fraction",
    "core_node_fraction",
    "degree_entropy",
    "hub_mass_fraction",
    "graph_segment_core_jaccard",
    "lambda2_norm_laplacian",
    "zero_eigenvalue_count",
    "spectral_radius_adjacency",
    "spectral_status",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a Phase 3 sensitivity sweep over scale windows and matched-LZ seeds "
            "for the minimal defect/spectrum pilot."
        )
    )
    parser.add_argument("--phase1-dir", default="results/hsi_v2/phase1_high_scales")
    parser.add_argument("--output-dir", default="results/hsi_v2/phase3/defect_spectrum_sensitivity")
    parser.add_argument("--observed-variants", default="B,E")
    parser.add_argument("--null-source", default="B")
    parser.add_argument("--null-models", default="markov1,matched-lz")
    parser.add_argument(
        "--matched-lz-seeds",
        default="all",
        help="'all', 'latest', or comma-separated seed list.",
    )
    parser.add_argument(
        "--windows",
        default="20,24,28,32;24,28,32,40;28,32,40,48",
        help="Semicolon-separated four-scale windows.",
    )
    parser.add_argument("--iteration", type=int, default=None)
    parser.add_argument("--segment-bits", type=int, default=None)
    parser.add_argument("--num-segments", type=int, default=None)
    parser.add_argument("--segment-offset-bits", type=int, default=None)
    parser.add_argument("--scales", default="")
    parser.add_argument("--left-policy", default="prefix")
    parser.add_argument("--right-policy", default="suffix")
    parser.add_argument("--min-segment-recurrence", type=int, default=2)
    parser.add_argument("--top-hub-k", type=int, default=8)
    parser.add_argument("--no-family-inference", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    observed_variants = parse_variants(args.observed_variants)
    if not observed_variants:
        parser.error("--observed-variants cannot be empty.")
    null_source = args.null_source.strip().upper()
    if not null_source:
        parser.error("--null-source cannot be empty.")
    null_models = parse_null_models(args.null_models)
    windows = parse_windows(args.windows)
    if args.left_policy == args.right_policy:
        parser.error("--left-policy and --right-policy must differ.")
    if args.min_segment_recurrence < 1:
        parser.error("--min-segment-recurrence must be at least 1.")

    phase1_dir = resolve_dir(args.phase1_dir, anchor_file=__file__)
    output_root = resolve_dir(args.output_dir, anchor_file=__file__)
    output_root.mkdir(parents=True, exist_ok=True)
    phase1_scales = parse_scales(args.scales) if args.scales.strip() else None

    phase_print(
        "Preparing Phase 3 defect/spectrum sensitivity sweep",
        (
            f"observed={','.join(observed_variants)} | null-source={null_source} | "
            f"windows={len(windows)} | matched-lz-seeds={args.matched_lz_seeds}"
        ),
        quiet=args.quiet,
    )

    runs = discover_phase1_runs_recursive(phase1_dir)
    if not runs:
        parser.error(f"No valid Phase 1 runs found in {phase1_dir}")

    family = None
    explicit_filters = any(
        value is not None for value in (args.iteration, args.segment_bits, args.num_segments, args.segment_offset_bits)
    ) or phase1_scales is not None
    if not args.no_family_inference and not explicit_filters:
        compatible_runs = [
            run
            for run in runs
            if any(
                supports_window(
                    run,
                    window_scales=window,
                    left_policy=args.left_policy,
                    right_policy=args.right_policy,
                )
                for window in windows
            )
        ]
        family = infer_family_from_latest_run(compatible_runs)

    matching_runs = filter_runs(
        runs,
        iteration=args.iteration,
        segment_bits=args.segment_bits,
        num_segments=args.num_segments,
        segment_offset_bits=args.segment_offset_bits,
        scales=phase1_scales,
        family=family,
    )
    if not matching_runs:
        parser.error("No Phase 1 runs matched the requested family/filters.")

    selected_runs = select_sources(
        matching_runs,
        observed_variants=observed_variants,
        null_source=null_source,
        null_models=null_models,
        matched_lz_seeds=args.matched_lz_seeds,
        windows=windows,
        left_policy=args.left_policy,
        right_policy=args.right_policy,
    )
    if not selected_runs:
        parser.error("No sources selected for sensitivity sweep.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_slug = build_run_slug(
        observed_variants=observed_variants,
        null_source=null_source,
        selected_runs=selected_runs,
        windows=windows,
        timestamp=timestamp,
    )
    run_dir = output_root / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    for index, window in enumerate(windows, start=1):
        window_runs = [
            run
            for run in selected_runs
            if supports_window(
                run,
                window_scales=window,
                left_policy=args.left_policy,
                right_policy=args.right_policy,
            )
        ]
        if not window_runs:
            continue
        phase_print(
            "Running window",
            f"{index}/{len(windows)} | {window_label(window)} | sources={len(window_runs)}",
            quiet=args.quiet,
        )
        rows = build_defect_spectrum_rows(
            window_runs,
            window_scales=window,
            left_policy=args.left_policy,
            right_policy=args.right_policy,
            min_segment_recurrence=args.min_segment_recurrence,
            top_hub_k=args.top_hub_k,
            show_progress=not args.quiet,
        )
        for row in compact_rows(rows):
            row["window_label"] = window_label(window)
            row["window_scales"] = "-".join(str(value) for value in window)
            all_rows.append(row)

    if not all_rows:
        parser.error("No rows were produced by the sensitivity sweep.")

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    selection = {
        "observed_variants": observed_variants,
        "null_source": null_source,
        "null_models": null_models,
        "matched_lz_seeds": args.matched_lz_seeds,
        "windows": [list(window) for window in windows],
        "left_policy": args.left_policy,
        "right_policy": args.right_policy,
        "min_segment_recurrence": args.min_segment_recurrence,
        "top_hub_k": args.top_hub_k,
        "family_inferred": family is not None,
        "selected_sources": [source_label(run) for run in selected_runs],
    }
    summary = {
        "stage": "phase3_defect_spectrum_sensitivity",
        "generated_at": generated_at,
        "selection": selection,
        "compact_rows": all_rows,
        "aggregate": aggregate_rows(all_rows),
        "notes": [
            "This sweep is a falsification layer over the minimal Phase 3 defect/spectrum pilot.",
            "It tests whether the first pilot's matched-LZ shadowing is seed/window-local or structurally persistent.",
            "Rows are compact by design; rerun the single-window pilot for full node/edge payloads.",
        ],
    }
    manifest = {
        "run_slug": run_slug,
        "generated_at": generated_at,
        "script": Path(__file__).name,
        "cwd": str(Path.cwd()),
        "inputs": {
            "phase1_dir": str(phase1_dir),
            "selected_phase1_runs": [str(Path(run["_run_dir"])) for run in selected_runs],
        },
        "outputs": {
            "summary": str(run_dir / "summary.json"),
            "report": str(run_dir / "report.md"),
            "compact_rows_csv": str(run_dir / "compact_rows.csv"),
            "manifest": str(run_dir / "manifest.json"),
        },
        "arguments": vars(args),
    }

    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "manifest.json", manifest)
    write_csv(run_dir / "compact_rows.csv", all_rows, CSV_FIELDS)
    (run_dir / "report.md").write_text(render_report(summary), encoding="utf-8")

    if not args.quiet:
        print(render_console_summary(summary))
        print(f"\nSaved summary to: {run_dir / 'summary.json'}")
        print(f"Saved report to: {run_dir / 'report.md'}")
        print(f"Saved compact CSV to: {run_dir / 'compact_rows.csv'}")
        print(f"Saved manifest to: {run_dir / 'manifest.json'}")

    return 0


def select_sources(
    runs: list[dict[str, Any]],
    *,
    observed_variants: list[str],
    null_source: str,
    null_models: list[str],
    matched_lz_seeds: str,
    windows: list[tuple[int, int, int, int]],
    left_policy: str,
    right_policy: str,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for variant in observed_variants:
        pool = [
            run
            for run in runs
            if is_observed_run(run)
            and str(run["dataset"]["config"].get("variant", "")).upper() == variant
            and any(supports_window(run, window_scales=w, left_policy=left_policy, right_policy=right_policy) for w in windows)
        ]
        selected.append(latest_run(pool, label=f"observed {variant}"))

    for model in null_models:
        pool = [
            run
            for run in runs
            if not is_observed_run(run)
            and str(run["dataset"]["config"].get("source_variant", "")).upper() == null_source
            and str(run["dataset"]["config"].get("null_model", "")).lower() == model
            and any(supports_window(run, window_scales=w, left_policy=left_policy, right_policy=right_policy) for w in windows)
        ]
        if model == "matched-lz":
            selected.extend(select_seed_runs(pool, matched_lz_seeds))
        else:
            selected.append(latest_run(pool, label=f"{null_source}-{model}"))

    return sorted(selected, key=lambda run: source_sort_key(run))


def select_seed_runs(runs: list[dict[str, Any]], seed_spec: str) -> list[dict[str, Any]]:
    if not runs:
        raise SystemExit("No matched-LZ runs found for requested null source.")
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for run in runs:
        seed = run["dataset"]["config"].get("null_seed")
        if seed is None:
            continue
        by_seed.setdefault(int(seed), []).append(run)
    if not by_seed:
        raise SystemExit("Matched-LZ runs have no null_seed metadata.")

    normalized = seed_spec.strip().lower()
    if normalized == "latest":
        return [latest_run(runs, label="matched-lz latest")]
    if normalized == "all":
        seeds = sorted(by_seed)
    else:
        try:
            seeds = [int(item.strip()) for item in seed_spec.split(",") if item.strip()]
        except ValueError as exc:
            raise SystemExit("--matched-lz-seeds must be 'all', 'latest', or comma-separated integers.") from exc

    selected = []
    missing = []
    for seed in seeds:
        pool = by_seed.get(seed)
        if not pool:
            missing.append(seed)
            continue
        selected.append(latest_run(pool, label=f"matched-lz seed {seed}"))
    if missing:
        raise SystemExit("Missing matched-LZ seeds: " + ",".join(str(seed) for seed in missing))
    return selected


def latest_run(runs: list[dict[str, Any]], *, label: str) -> dict[str, Any]:
    if not runs:
        raise SystemExit(f"No compatible run found for {label}.")
    return max(runs, key=lambda run: run["_generated_at_dt"])


def parse_windows(raw: str) -> list[tuple[int, int, int, int]]:
    windows = []
    for chunk in raw.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        scales = tuple(parse_scales(chunk))
        if len(scales) != 4:
            raise SystemExit("--windows entries must each contain exactly four scales.")
        if sorted(scales) != list(scales) or len(set(scales)) != 4:
            raise SystemExit("--windows entries must be strictly ascending.")
        windows.append(scales)
    if not windows:
        raise SystemExit("--windows cannot be empty.")
    return windows


def parse_null_models(raw: str) -> list[str]:
    models = []
    for item in raw.split(","):
        model = item.strip().lower()
        if not model:
            continue
        if model not in {"markov1", "matched-lz"}:
            raise SystemExit(f"Unsupported null model: {item}")
        if model not in models:
            models.append(model)
    return models


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["window_label"], row["source_kind"]), []).append(row)
    aggregate = []
    for (window, source_kind), group in sorted(groups.items()):
        aggregate.append(
            {
                "window_label": window,
                "source_kind": source_kind,
                "count": len(group),
                "square_defect_mass_avg": mean(row["square_defect_mass_rate"] for row in group),
                "mean_square_distance_avg": mean(row["mean_square_distance"] for row in group),
                "node_count_avg": mean(row["node_count"] for row in group),
                "edge_count_avg": mean(row["edge_count"] for row in group),
                "zero_eigenvalue_count_avg": mean(row["zero_eigenvalue_count"] for row in group),
                "spectral_radius_avg": mean(row["spectral_radius_adjacency"] for row in group),
            }
        )
    return aggregate


def render_console_summary(summary: dict[str, Any]) -> str:
    lines = [
        "Phase 3 defect/spectrum sensitivity sweep",
        "-" * 118,
        f"{'window':<14}{'source':<12}{'n':>4}{'sq_mass':>10}{'dist':>10}{'nodes':>10}{'edges':>10}{'zero':>10}{'rho':>10}",
    ]
    for row in summary["aggregate"]:
        lines.append(
            f"{row['window_label']:<14}"
            f"{row['source_kind']:<12}"
            f"{row['count']:>4}"
            f"{fmt(row['square_defect_mass_avg']):>10}"
            f"{fmt(row['mean_square_distance_avg']):>10}"
            f"{fmt(row['node_count_avg']):>10}"
            f"{fmt(row['edge_count_avg']):>10}"
            f"{fmt(row['zero_eigenvalue_count_avg']):>10}"
            f"{fmt(row['spectral_radius_avg']):>10}"
        )
    return "\n".join(lines)


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase 3 Defect/Spectrum Sensitivity Sweep",
        "",
        "## Scope",
        "",
        "This is a falsification layer over the minimal Phase 3 pilot. It tests scale-window and matched-LZ seed sensitivity before any Phase 3 claim is allowed.",
        "",
        "## Aggregate Readout",
        "",
        "| Window | Source | Count | Avg defect mass | Avg distance | Avg nodes | Avg edges | Avg zero eigs | Avg spectral radius |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["aggregate"]:
        lines.append(
            f"| {row['window_label']} | {row['source_kind']} | {row['count']} | "
            f"{fmt(row['square_defect_mass_avg'])} | {fmt(row['mean_square_distance_avg'])} | "
            f"{fmt(row['node_count_avg'])} | {fmt(row['edge_count_avg'])} | "
            f"{fmt(row['zero_eigenvalue_count_avg'])} | {fmt(row['spectral_radius_avg'])} |"
        )
    lines.extend(
        [
            "",
            "## Reading Rule",
            "",
            "- If matched-LZ tracks observed B across all windows and seeds, this minimal defect/spectrum object is not a strong-null separator.",
            "- If matched-LZ separates in a stable window or seed envelope, that window becomes the next Phase 3 candidate contract.",
            "- If only markov1 separates, the object remains useful as a collapse control but not as a Phase 3 opening.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_json(path: Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def build_run_slug(
    *,
    observed_variants: list[str],
    null_source: str,
    selected_runs: list[dict[str, Any]],
    windows: list[tuple[int, int, int, int]],
    timestamp: str,
) -> str:
    seed_count = sum(1 for run in selected_runs if run["dataset"]["config"].get("null_model") == "matched-lz")
    return (
        f"phase3-defect-spectrum-sensitivity__obs-{'-'.join(observed_variants)}"
        f"__null-{null_source}__w-{len(windows)}__lzseeds-{seed_count}__{timestamp}"
    )


def source_sort_key(run: dict[str, Any]) -> tuple[int, str, int]:
    config = run["dataset"]["config"]
    if is_observed_run(run):
        return (0, str(config.get("variant", "")), -1)
    model = str(config.get("null_model", ""))
    seed = int(config.get("null_seed") or -1)
    return (1 if model == "markov1" else 2, model, seed)


def source_label(run: dict[str, Any]) -> str:
    config = run["dataset"]["config"]
    if is_observed_run(run):
        return str(config.get("variant"))
    return f"{config.get('variant')} [seed {config.get('null_seed')}]"


def window_label(window: tuple[int, int, int, int]) -> str:
    return "-".join(str(value) for value in window)


def mean(values) -> float | None:
    vals = [float(value) for value in values if value is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def phase_print(title: str, detail: str, *, quiet: bool) -> None:
    if quiet:
        return
    print(f"\n[Phase] {title}")
    print(f"        {detail}")


if __name__ == "__main__":
    raise SystemExit(main())
