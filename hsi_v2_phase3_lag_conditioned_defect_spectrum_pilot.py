#!/usr/bin/env python3
"""HSI v2 Phase 3 lag-conditioned defect/spectrum pilot."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from hsi_v2_phase2_parent_survival_band import build_observed_specs
from hsi_v2_phase2_parent_survival_band_lagaware import build_null_specs_for_candidate
from hsi_v2_phase2_transport_defect_strict import (
    artifact_source_info,
    case_note,
    infer_segment_protocol,
    parse_int_list,
    phase_print,
    select_observed_runs,
    window_note,
)
from v2.common.cli import parse_variants, resolve_dir
from v2.common.naming import compact_int
from v2.phase1.tower import parse_scales
from v2.phase2.null_pressure import (
    discover_phase1_runs_recursive,
    parse_null_models,
    parse_seed_list,
    select_null_pressure_runs,
)
from v2.phase2.return_lag import prepare_frozen_source_cache
from v2.phase3.lag_conditioned_defect import (
    compact_lag_conditioned_row,
    compute_lag_conditioned_defect_result,
    summarize_lag_conditioned_rows,
)


CSV_FIELDS = [
    "variant",
    "source_label",
    "source_kind",
    "null_model",
    "null_seed",
    "anchor_offset_bits",
    "candidate_offset_bits",
    "candidate_lag_bits",
    "monitored_bridge_count",
    "anchor_future_bridge_fraction",
    "candidate_future_bridge_fraction",
    "squareable_bridge_fraction",
    "anchored_square_mass_rate",
    "square_defect_mass_rate",
    "dropout_mass_rate",
    "bridge_disagreement_mass_rate",
    "mean_square_distance",
    "square_entropy",
    "defect_node_count",
    "defect_edge_count",
    "lambda2_norm_laplacian",
    "zero_eigenvalue_count",
    "spectral_radius_adjacency",
    "spectral_status",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Phase 3 N3-02 lag-conditioned defect/spectrum pilot. "
            "The anchor window remains fixed; candidate windows are shifted by an explicit lag."
        )
    )
    parser.add_argument("--phase1-dir", default="results/hsi_v2/phase1_high_scales")
    parser.add_argument("--output-dir", default="results/hsi_v2/phase3/lag_conditioned_defect_spectrum")
    parser.add_argument("--stage", choices=("observed", "nulls", "all"), default="all")
    parser.add_argument("--variants", default="E,B")
    parser.add_argument("--anchor-variant", default="E")
    parser.add_argument("--null-models", default="markov1,matched-lz")
    parser.add_argument(
        "--matched-lz-seeds",
        default="17",
        help="Comma-separated matched-LZ seeds. Use an empty string to select all available seeds.",
    )
    parser.add_argument("--iteration", type=int, default=None)
    parser.add_argument("--segment-bits", type=int, default=None)
    parser.add_argument("--num-segments", type=int, default=None)
    parser.add_argument("--segment-offset-bits", type=int, default=None)
    parser.add_argument("--scales", default="")
    parser.add_argument("--phase1-policies", default="")
    parser.add_argument("--window-scales", default="20,24,28,32")
    parser.add_argument("--left-policy", default="prefix")
    parser.add_argument("--right-policy", default="suffix")
    parser.add_argument("--top-bridges", type=int, default=128)
    parser.add_argument("--offsets", default="714000000,717000000,720000000,723000000")
    parser.add_argument("--window-bits", type=int, default=3000000)
    parser.add_argument("--candidate-lag-bits", type=int, default=-27000000)
    parser.add_argument("--no-family-inference", action="store_true")
    parser.add_argument(
        "--log-windows",
        action="store_true",
        help="Print one line per case/window. Off by default to keep long runs readable.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    variants = parse_variants(args.variants)
    anchor_variants = parse_variants(args.anchor_variant)
    if len(variants) != 2:
        parser.error("--variants must contain exactly two observed variants.")
    if len(anchor_variants) != 1:
        parser.error("--anchor-variant must contain exactly one variant.")
    anchor_variant = anchor_variants[0]
    if anchor_variant not in variants:
        parser.error("--anchor-variant must be included in --variants.")
    candidate_variant = next(variant for variant in variants if variant != anchor_variant)
    if args.left_policy == args.right_policy:
        parser.error("--left-policy and --right-policy must differ.")
    if args.top_bridges < 1:
        parser.error("--top-bridges must be positive.")
    if args.window_bits < 1:
        parser.error("--window-bits must be positive.")

    window_scales = tuple(parse_scales(args.window_scales))
    if len(window_scales) != 4:
        parser.error("--window-scales must contain exactly four scales.")
    if sorted(window_scales) != list(window_scales) or len(set(window_scales)) != 4:
        parser.error("--window-scales must be strictly ascending.")

    try:
        offsets = parse_int_list(args.offsets, label="--offsets", allow_zero=True)
    except ValueError as exc:
        parser.error(str(exc))
    if any(offset < 0 for offset in offsets):
        parser.error("--offsets must contain non-negative integers.")
    if any(offset + args.candidate_lag_bits < 0 for offset in offsets):
        parser.error("--candidate-lag-bits creates a negative candidate window offset.")

    null_models = []
    matched_lz_seeds: list[int] | None = []
    if args.stage in {"nulls", "all"}:
        try:
            null_models = parse_null_models(args.null_models)
        except ValueError as exc:
            parser.error(str(exc))
        if not null_models:
            parser.error("--null-models cannot be empty for --stage nulls/all.")
        try:
            matched_lz_seeds = parse_seed_list(args.matched_lz_seeds)
        except ValueError:
            parser.error("--matched-lz-seeds must be empty or comma-separated integers.")
        if not args.matched_lz_seeds.strip():
            matched_lz_seeds = None

    phase1_dir = resolve_dir(args.phase1_dir, anchor_file=__file__)
    output_root = resolve_dir(args.output_dir, anchor_file=__file__)
    output_root.mkdir(parents=True, exist_ok=True)
    phase1_scales = parse_scales(args.scales) if args.scales.strip() else None
    phase1_policies = [item.strip() for item in args.phase1_policies.split(",") if item.strip()] or None

    phase_print(
        "Preparing Phase 3 lag-conditioned defect/spectrum pilot",
        (
            f"stage={args.stage} | anchor={anchor_variant} | candidate={candidate_variant} | "
            f"window={','.join(str(value) for value in window_scales)} | lag={args.candidate_lag_bits}"
        ),
        quiet=args.quiet,
    )

    runs = discover_phase1_runs_recursive(phase1_dir)
    if not runs:
        parser.error(f"No valid Phase 1 runs found in {phase1_dir}")

    observed_runs, family_inferred = select_observed_runs(
        runs,
        variants=variants,
        iteration=args.iteration,
        segment_bits=args.segment_bits,
        num_segments=args.num_segments,
        segment_offset_bits=args.segment_offset_bits,
        scales=phase1_scales,
        policies=phase1_policies,
        low_scale=window_scales[0],
        high_scale=window_scales[-1],
        no_family_inference=args.no_family_inference,
    )
    observed_by_variant = {
        str(run["dataset"]["config"]["variant"]).upper(): run
        for run in observed_runs
    }
    anchor_run = observed_by_variant.get(anchor_variant)
    candidate_run = observed_by_variant.get(candidate_variant)
    if anchor_run is None or candidate_run is None:
        parser.error("Missing observed anchor or candidate run.")

    evaluation_specs: list[dict[str, Any]] = []
    if args.stage in {"observed", "all"}:
        evaluation_specs.extend(
            build_observed_specs(
                observed_by_variant,
                anchor_run=anchor_run,
                variants=variants,
            )
        )
    if args.stage in {"nulls", "all"}:
        try:
            selected_nulls = select_null_pressure_runs(
                runs,
                observed_variant=candidate_variant,
                null_models=null_models,
                matched_lz_seeds=matched_lz_seeds,
                iteration=args.iteration,
                segment_bits=args.segment_bits,
                num_segments=args.num_segments,
                segment_offset_bits=args.segment_offset_bits,
                scales=phase1_scales,
                policies=phase1_policies,
                pattern_scale=window_scales[-1],
                no_family_inference=args.no_family_inference,
            )
        except ValueError as exc:
            parser.error(f"{candidate_variant}: {exc}")
        evaluation_specs.extend(
            build_null_specs_for_candidate(
                selected_nulls["null_specs"],
                anchor_run=anchor_run,
                candidate_variant=candidate_variant,
            )
        )
    if not evaluation_specs:
        parser.error("No evaluation specs remain after selection.")

    segment_bits, num_segments = infer_segment_protocol(observed_runs)
    if args.window_bits < window_scales[-1]:
        parser.error("--window-bits must be at least as large as the highest requested scale.")
    required_bits = max(
        max(offset + args.window_bits for offset in offsets),
        max(offset + args.candidate_lag_bits + args.window_bits for offset in offsets),
    )

    phase_print(
        "Freezing source realizations for lag-conditioned windows",
        f"runs={count_unique_run_dirs(evaluation_specs + [{'run': anchor_run}])} | required_bits={required_bits}",
        quiet=args.quiet,
    )
    preloaded_sources = prepare_frozen_source_cache(
        [spec["run"] for spec in evaluation_specs] + [anchor_run],
        required_bits=required_bits,
        show_progress=not args.quiet,
    )

    phase_print(
        "Running lag-conditioned defect/spectrum windows",
        f"cases={len(evaluation_specs)} | windows={len(offsets)} | top_bridges={args.top_bridges}",
        quiet=args.quiet,
    )
    rows = []
    for case_index, spec in enumerate(evaluation_specs, start=1):
        case_note(
            (
                f"{case_index}/{len(evaluation_specs)} | variant={spec['variant']} | "
                f"source={spec['source_label']} | anchor={spec['anchor_variant']}"
            ),
            quiet=args.quiet,
        )
        anchor_bits = preloaded_sources[str(Path(spec["anchor_run"]["_run_dir"]))]["bits"]
        candidate_bits = preloaded_sources[str(Path(spec["run"]["_run_dir"]))]["bits"]
        for window_index, anchor_offset in enumerate(offsets, start=1):
            candidate_lag = 0 if spec["variant"] == anchor_variant and spec["source_kind"] == "observed" else args.candidate_lag_bits
            candidate_offset = anchor_offset + candidate_lag
            window_note(
                (
                    f"case {case_index}/{len(evaluation_specs)} | window {window_index}/{len(offsets)} | "
                    f"anchor_off={anchor_offset} | cand_off={candidate_offset} | source={spec['source_label']}"
                ),
                quiet=args.quiet or not args.log_windows,
            )
            result = compute_lag_conditioned_defect_result(
                anchor_run_dir=spec["anchor_run"]["_run_dir"],
                candidate_run_dir=spec["run"]["_run_dir"],
                anchor_window_bits=anchor_bits[anchor_offset : anchor_offset + args.window_bits],
                candidate_window_bits=candidate_bits[candidate_offset : candidate_offset + args.window_bits],
                window_scales=window_scales,
                left_policy=args.left_policy,
                right_policy=args.right_policy,
                top_bridges=args.top_bridges,
            )
            rows.append(
                {
                    "variant": spec["variant"],
                    "source_label": spec["source_label"],
                    "source_kind": spec["source_kind"],
                    "null_model": spec.get("null_model"),
                    "null_seed": spec.get("null_seed"),
                    "anchor_variant": spec["anchor_variant"],
                    "anchor_kind": spec["anchor_kind"],
                    "anchor_run_dir": spec["anchor_run"]["_run_dir"],
                    "run_dir": spec["run"]["_run_dir"],
                    "anchor_offset_bits": anchor_offset,
                    "candidate_offset_bits": candidate_offset,
                    "candidate_lag_bits": candidate_lag,
                    "window_bits": args.window_bits,
                    "source_freeze_mode": artifact_source_info(
                        preloaded_sources,
                        spec["run"]["_run_dir"],
                        "source_freeze_mode",
                    ),
                    "frozen_total_bits": artifact_source_info(
                        preloaded_sources,
                        spec["run"]["_run_dir"],
                        "frozen_total_bits",
                    ),
                    "result": result,
                }
            )

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    selection = {
        "stage": args.stage,
        "anchor_variant": anchor_variant,
        "candidate_variant": candidate_variant,
        "variants": variants,
        "iteration": args.iteration,
        "segment_bits": segment_bits,
        "num_segments": num_segments,
        "offsets": offsets,
        "window_bits": args.window_bits,
        "window_scales": list(window_scales),
        "left_policy": args.left_policy,
        "right_policy": args.right_policy,
        "top_bridges": args.top_bridges,
        "candidate_lag_bits": args.candidate_lag_bits,
        "null_models": null_models,
        "matched_lz_seeds": matched_lz_seeds,
        "family_inferred": family_inferred,
    }
    run_slug = build_run_slug(selection, timestamp)
    run_dir = output_root / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)

    compact_rows = [compact_lag_conditioned_row(row) for row in rows]
    source_summary = summarize_lag_conditioned_rows(rows)
    dataset_payload = {
        "stage": "phase3_lag_conditioned_defect_spectrum_pilot",
        "generated_at": generated_at,
        "selection": selection,
        "rows": rows,
        "notes": [
            "N3-02 conditions the Phase 3 local square on a Phase 2 temporal channel.",
            "Bridge monitoring is anchor-defined and window-local; candidate windows are lag-shifted.",
            "This is an internal falsification object, not a paper-facing geometric claim.",
        ],
    }
    summary_payload = {
        "generated_at": generated_at,
        "selection": selection,
        "source_summary": source_summary,
        "compact_rows": compact_rows,
    }
    manifest_payload = {
        "run_slug": run_slug,
        "generated_at": generated_at,
        "script": Path(__file__).name,
        "cwd": str(Path.cwd()),
        "inputs": {
            "phase1_dir": str(phase1_dir),
            "phase1_runs": sorted({str(Path(row["run_dir"])) for row in rows} | {str(Path(anchor_run["_run_dir"]))}),
        },
        "outputs": {
            "dataset": str(run_dir / "dataset.json"),
            "summary": str(run_dir / "summary.json"),
            "report": str(run_dir / "report.md"),
            "compact_rows_csv": str(run_dir / "compact_rows.csv"),
            "source_summary_csv": str(run_dir / "source_summary.csv"),
            "manifest": str(run_dir / "manifest.json"),
        },
        "arguments": vars(args),
    }

    write_json(run_dir / "dataset.json", dataset_payload)
    write_json(run_dir / "summary.json", summary_payload)
    write_json(run_dir / "manifest.json", manifest_payload)
    write_csv(run_dir / "compact_rows.csv", compact_rows, CSV_FIELDS)
    write_csv(run_dir / "source_summary.csv", source_summary, list(source_summary[0]) if source_summary else [])
    (run_dir / "report.md").write_text(render_report(summary_payload), encoding="utf-8")

    if not args.quiet:
        print(render_console_summary(source_summary))
        print(f"\nSaved dataset to: {run_dir / 'dataset.json'}")
        print(f"Saved summary to: {run_dir / 'summary.json'}")
        print(f"Saved report to: {run_dir / 'report.md'}")
        print(f"Saved compact CSV to: {run_dir / 'compact_rows.csv'}")
        print(f"Saved source summary CSV to: {run_dir / 'source_summary.csv'}")
        print(f"Saved manifest to: {run_dir / 'manifest.json'}")
    return 0


def render_console_summary(source_summary: list[dict[str, Any]]) -> str:
    lines = [
        "Phase 3 N3-02 lag-conditioned defect/spectrum pilot",
        "-" * 118,
        f"{'variant':<12}{'source':<28}{'wins':>6}{'anch':>9}{'cand':>9}{'def':>9}{'drop':>9}{'dist':>9}{'edges':>9}{'rho':>9}",
    ]
    for row in source_summary:
        lines.append(
            f"{row['variant']:<12}"
            f"{truncate(row['source_label'], 28):<28}"
            f"{row['window_count']:>6}"
            f"{fmt(row['anchored_mass_avg']):>9}"
            f"{fmt(row['candidate_future_avg']):>9}"
            f"{fmt(row['defect_mass_avg']):>9}"
            f"{fmt(row['dropout_mass_avg']):>9}"
            f"{fmt(row['distance_avg']):>9}"
            f"{fmt(row['edge_count_avg']):>9}"
            f"{fmt(row['spectral_radius_avg']):>9}"
        )
    return "\n".join(lines)


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase 3 N3-02 Lag-Conditioned Defect/Spectrum Pilot",
        "",
        "## Contract",
        "",
        "- Anchor bridge monitoring is defined only from the observed anchor window.",
        "- Candidate windows are shifted by the explicit candidate lag.",
        "- The object tests whether the local Phase 3 square becomes discriminative when constrained by Phase 2 temporal phase.",
        "- No geometry, curvature, or final transport law is claimed.",
        "",
        "## Source Summary",
        "",
        "| Variant | Source | Windows | Anchored mass | Candidate future | Defect mass | Dropout mass | Mean distance | Defect edges | Spectral radius |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["source_summary"]:
        lines.append(
            f"| {row['variant']} | {row['source_label']} | {row['window_count']} | "
            f"{fmt(row['anchored_mass_avg'])} | {fmt(row['candidate_future_avg'])} | "
            f"{fmt(row['defect_mass_avg'])} | {fmt(row['dropout_mass_avg'])} | "
            f"{fmt(row['distance_avg'])} | {fmt(row['edge_count_avg'])} | "
            f"{fmt(row['spectral_radius_avg'])} |"
        )
    lines.extend(
        [
            "",
            "## Reading Rule",
            "",
            "- If observed B separates from matched-LZ under the lag-conditioned square, N3-02 becomes the next Phase 3 candidate contract.",
            "- If matched-LZ continues to shadow B, the defect must become more explicitly non-local or use a phase-channel grid.",
            "- If only markov1 separates, N3-02 remains a useful collapse control but not a strongest-null separator.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_run_slug(selection: dict[str, Any], timestamp: str) -> str:
    offsets = selection["offsets"]
    offset_part = f"off-{compact_int(min(offsets))}-plus-{len(offsets)}" if len(offsets) > 1 else f"off-{compact_int(offsets[0])}"
    window = "-".join(str(value) for value in selection["window_scales"])
    return (
        f"phase3-lag-conditioned-defect-spectrum__stage-{selection['stage']}"
        f"__anchor-{selection['anchor_variant']}__cand-{selection['candidate_variant']}"
        f"__lag-{compact_int(selection['candidate_lag_bits'])}__window-{window}"
        f"__top-{selection['top_bridges']}__{offset_part}__{timestamp}"
    )


def count_unique_run_dirs(specs: list[dict[str, Any]]) -> int:
    return len({str(Path(spec["run"]["_run_dir"])) for spec in specs})


def write_json(path: Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def truncate(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    return value[: width - 3] + "..."


if __name__ == "__main__":
    raise SystemExit(main())
