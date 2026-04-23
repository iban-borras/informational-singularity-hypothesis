#!/usr/bin/env python3
"""
HSI v2 Phase 2 Parent Survival Band Stability (Lag-Aware)

Lag-aware variant of the parent survival band sweep. The anchor window stays
fixed; the candidate window is shifted by an externally estimated lag.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hsi_v2_phase2_parent_survival_band import (
    build_kernel_row,
    build_observed_specs,
)
from hsi_v2_phase2_transport_defect_strict import (
    artifact_source_info,
    case_note,
    count_unique_run_dirs,
    infer_segment_protocol,
    parse_int_list,
    phase_print,
    select_observed_runs,
    window_note,
)
from v2.common.cli import parse_variants, resolve_dir
from v2.common.naming import compact_int
from v2.phase1.tower import parse_scales
from v2.phase2.null_pressure import parse_null_models, parse_seed_list, select_null_pressure_runs
from v2.phase2.kernel_common_support import (
    compute_window_local_common_support_kernel_result,
)
from v2.phase2.parent_survival import (
    build_parent_survival_rows,
    write_parent_patterns_csv,
    write_parent_survival_csv,
)
from v2.phase2.parent_survival_band import (
    render_parent_survival_band_console_summary,
    render_parent_survival_band_report,
    summarize_parent_survival_band_rows,
    write_parent_survival_band_summary_csv,
)
from v2.phase2.return_lag import (
    prepare_frozen_source_cache,
    prepare_pattern_selection,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a lag-aware short-band parent-survival sweep using an externally estimated candidate lag."
    )
    parser.add_argument(
        "--phase1-dir",
        type=str,
        default="results/hsi_v2/phase1_high_scales",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/hsi_v2/phase2/parent_survival_band_lagaware",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="observed",
        choices=("observed", "nulls", "all"),
    )
    parser.add_argument("--variants", type=str, default="E,B")
    parser.add_argument("--anchor-variant", type=str, default="E")
    parser.add_argument("--iteration", type=int, default=None)
    parser.add_argument("--segment-bits", type=int, default=None)
    parser.add_argument("--num-segments", type=int, default=None)
    parser.add_argument("--segment-offset-bits", type=int, default=None)
    parser.add_argument("--scales", type=str, default="")
    parser.add_argument("--phase1-policies", type=str, default="")
    parser.add_argument("--low-scale", type=int, default=40)
    parser.add_argument("--high-scale", type=int, default=48)
    parser.add_argument("--top-patterns", type=int, default=64)
    parser.add_argument(
        "--pattern-selection",
        type=str,
        default="bridge-linked",
        choices=("top", "rare-stable", "bridge-linked"),
    )
    parser.add_argument(
        "--offsets",
        type=str,
        default="714000000,717000000,720000000,723000000",
    )
    parser.add_argument("--window-bits", type=int, default=None)
    parser.add_argument("--candidate-lag-bits", type=int, default=None)
    parser.add_argument("--lag-summary", type=str, default="")
    parser.add_argument("--null-models", type=str, default="markov1,matched-lz")
    parser.add_argument("--matched-lz-seeds", type=str, default="")
    parser.add_argument("--no-family-inference", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    variants = parse_variants(args.variants)
    if len(variants) != 2:
        parser.error("--variants must contain exactly two observed variants.")
    anchor_variants = parse_variants(args.anchor_variant)
    if len(anchor_variants) != 1:
        parser.error("--anchor-variant must contain exactly one variant.")
    anchor_variant = anchor_variants[0]
    if anchor_variant not in variants:
        parser.error("--anchor-variant must be included in --variants.")
    candidate_variant = next(variant for variant in variants if variant != anchor_variant)

    if args.low_scale <= 0 or args.high_scale <= 0 or args.low_scale >= args.high_scale:
        parser.error("--low-scale and --high-scale must satisfy 1 <= low < high.")
    if args.high_scale > 64:
        parser.error("--high-scale must be in the range 1..64.")
    if args.top_patterns <= 0:
        parser.error("--top-patterns must be positive.")

    null_models: list[str] = []
    matched_lz_seeds: list[int] = []
    if args.stage in {"nulls", "all"}:
        try:
            null_models = parse_null_models(args.null_models)
        except ValueError as exc:
            parser.error(str(exc))
        if not null_models:
            parser.error("At least one null model is required for --stage nulls/all.")
        try:
            matched_lz_seeds = parse_seed_list(args.matched_lz_seeds)
        except ValueError:
            parser.error("--matched-lz-seeds must be a comma-separated list of integers.")

    try:
        offsets = parse_int_list(args.offsets, label="--offsets", allow_zero=True)
    except ValueError as exc:
        parser.error(str(exc))
    if any(offset < 0 for offset in offsets):
        parser.error("--offsets must contain non-negative integers.")

    candidate_lag_bits = resolve_candidate_lag_bits(args, parser)

    phase1_dir = resolve_dir(args.phase1_dir, anchor_file=__file__)
    output_dir = resolve_dir(args.output_dir, anchor_file=__file__)
    output_dir.mkdir(parents=True, exist_ok=True)

    phase1_scales = parse_scales(args.scales) if args.scales.strip() else None
    phase1_policies = [
        item.strip()
        for item in args.phase1_policies.split(",")
        if item.strip()
    ] or None

    phase_print(
        "Preparing lag-aware parent-survival band",
        (
            f"stage={args.stage} | anchor={anchor_variant} | candidate={candidate_variant} | "
            f"m={args.low_scale}->{args.high_scale} | sel={args.pattern_selection} | "
            f"lag={candidate_lag_bits}"
        ),
        quiet=args.quiet,
    )

    from v2.phase2.null_pressure import discover_phase1_runs_recursive

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
        low_scale=args.low_scale,
        high_scale=args.high_scale,
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

    evaluation_specs = []
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
                pattern_scale=args.high_scale,
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
    window_bits = (
        int(args.window_bits)
        if args.window_bits is not None
        else segment_bits * num_segments
    )
    if window_bits < args.high_scale:
        parser.error("--window-bits must be at least as large as --high-scale.")

    low_selection = prepare_pattern_selection(
        anchor_run,
        pattern_scale=args.low_scale,
        top_patterns=args.top_patterns,
        pattern_selection=args.pattern_selection,
    )
    high_selection = prepare_pattern_selection(
        anchor_run,
        pattern_scale=args.high_scale,
        top_patterns=args.top_patterns,
        pattern_selection=args.pattern_selection,
    )

    required_bits = max(offsets) + max(window_bits, window_bits + max(candidate_lag_bits, 0))
    if candidate_lag_bits < 0:
        required_bits = max(required_bits, max(offsets) + window_bits)

    phase_print(
        "Freezing source realizations for lag-aware sweep",
        f"runs={count_unique_run_dirs(evaluation_specs)} | required_bits={required_bits}",
        quiet=args.quiet,
    )
    preloaded_sources = prepare_frozen_source_cache(
        [spec["run"] for spec in evaluation_specs] + [anchor_run],
        required_bits=required_bits,
        show_progress=not args.quiet,
    )

    phase_print(
        "Running lag-aware parent-survival band",
        (
            f"cases={len(evaluation_specs)} | windows={len(offsets)} | "
            f"parents={len(low_selection['selected_patterns'])} | children={len(high_selection['selected_patterns'])}"
        ),
        quiet=args.quiet,
    )

    kernel_rows = []
    artifacts = []
    for case_index, spec in enumerate(evaluation_specs, start=1):
        case_note(
            (
                f"{case_index}/{len(evaluation_specs)} | variant={spec['variant']} | "
                f"source={spec['source_label']} | anchor={spec['anchor_variant']}"
            ),
            quiet=args.quiet,
        )
        anchor_bits = preloaded_sources[str(Path(spec["anchor_run"]["_run_dir"]).resolve())]["bits"]
        candidate_bits = preloaded_sources[str(Path(spec["run"]["_run_dir"]).resolve())]["bits"]

        for offset_index, offset_bits in enumerate(offsets, start=1):
            variant_lag_bits = 0 if spec["variant"] == anchor_variant else candidate_lag_bits
            candidate_offset_bits = offset_bits + variant_lag_bits
            if candidate_offset_bits < 0:
                parser.error(
                    f"Computed negative candidate window start ({candidate_offset_bits}) for offset {offset_bits}."
                )
            window_note(
                (
                    f"case {case_index}/{len(evaluation_specs)} | window {offset_index}/{len(offsets)} | "
                    f"anchor_off={offset_bits} | cand_off={candidate_offset_bits} | "
                    f"variant={spec['variant']} | source={spec['source_label']}"
                ),
                quiet=args.quiet,
            )
            result = compute_window_local_common_support_kernel_result(
                anchor_run_dir=spec["anchor_run"]["_run_dir"],
                candidate_run_dir=spec["run"]["_run_dir"],
                anchor_window_bits=anchor_bits[offset_bits : offset_bits + window_bits],
                candidate_window_bits=candidate_bits[
                    candidate_offset_bits : candidate_offset_bits + window_bits
                ],
                low_scale=args.low_scale,
                high_scale=args.high_scale,
                pattern_selection=args.pattern_selection,
                low_selected_patterns=low_selection["selected_patterns"],
                high_selected_patterns=high_selection["selected_patterns"],
            )
            row = build_kernel_row(
                result=result,
                spec=spec,
                offset_bits=offset_bits,
                window_bits=window_bits,
                source_freeze_mode=artifact_source_info(
                    preloaded_sources,
                    spec["run"]["_run_dir"],
                    "source_freeze_mode",
                ),
                frozen_total_bits=artifact_source_info(
                    preloaded_sources,
                    spec["run"]["_run_dir"],
                    "frozen_total_bits",
                ),
            )
            row["candidate_offset_bits"] = candidate_offset_bits
            row["candidate_lag_bits"] = variant_lag_bits
            kernel_rows.append(row)

        artifacts.append(
            {
                "variant": spec["variant"],
                "source_label": spec["source_label"],
                "source_kind": spec["source_kind"],
                "null_model": spec.get("null_model"),
                "null_seed": spec.get("null_seed"),
                "run_dir": spec["run"]["_run_dir"],
                "anchor_run_dir": spec["anchor_run"]["_run_dir"],
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
            }
        )

    generated_at = datetime.now().isoformat(timespec="seconds")
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    selection = {
        "stage": f"{args.stage}-lag-aware",
        "anchor_variant": anchor_variant,
        "candidate_variant": candidate_variant,
        "variants": variants,
        "iteration": args.iteration,
        "segment_bits": segment_bits,
        "num_segments": num_segments,
        "offsets": offsets,
        "source_freeze_mode": "sweep-global-frozen",
        "frozen_total_bits": required_bits,
        "window_bits": window_bits,
        "low_scale": args.low_scale,
        "high_scale": args.high_scale,
        "top_patterns": args.top_patterns,
        "pattern_selection": args.pattern_selection,
        "family_inferred": family_inferred,
        "kernel_mode": "window-local-lag-aware",
        "candidate_lag_bits": candidate_lag_bits,
        "null_models": null_models,
        "matched_lz_seeds": matched_lz_seeds,
    }
    run_slug = build_run_slug(selection, timestamp)
    run_dir = output_dir / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = run_dir / "dataset.json"
    summary_path = run_dir / "summary.json"
    report_path = run_dir / "report.md"
    table_csv_path = run_dir / "table.csv"
    patterns_csv_path = run_dir / "patterns.csv"
    band_csv_path = run_dir / "band_summary.csv"
    manifest_path = run_dir / "manifest.json"

    survival_payload = {
        "selection": selection,
        "_dataset_path": str(dataset_path),
        "rows": kernel_rows,
    }
    survival_rows, pattern_rows = build_parent_survival_rows(survival_payload)
    grouped_summary = summarize_parent_survival_band_rows(survival_rows, pattern_rows)

    dataset_payload = {
        "stage": "phase2_parent_survival_band_lagaware",
        "generated_at": generated_at,
        "selection": selection,
        "kernel_rows": kernel_rows,
        "rows": survival_rows,
        "pattern_rows": pattern_rows,
        "grouped_summary": grouped_summary,
        "artifacts": artifacts,
        "notes": [
            "The anchor window stays fixed at each offset.",
            "The candidate window is shifted by the externally estimated candidate lag.",
            "This object is the lag-aware follow-up to the independent parent-shell lag probe.",
        ],
    }
    summary_payload = {
        "generated_at": generated_at,
        "selection": selection,
        "grouped_summary": grouped_summary,
    }
    manifest_payload = {
        "run_slug": run_slug,
        "generated_at": generated_at,
        "script": "hsi_v2_phase2_parent_survival_band_lagaware.py",
        "cwd": str(Path.cwd()),
        "outputs": {
            "dataset": str(dataset_path),
            "summary": str(summary_path),
            "report": str(report_path),
            "table_csv": str(table_csv_path),
            "patterns_csv": str(patterns_csv_path),
            "band_summary_csv": str(band_csv_path),
            "manifest": str(manifest_path),
        },
        "inputs": {
            "phase1_dir": str(phase1_dir),
            "anchor_run_dir": anchor_run["_run_dir"],
            "candidate_run_dir": candidate_run["_run_dir"],
            "frozen_total_bits": required_bits,
        },
        "arguments": vars(args),
    }

    with open(dataset_path, "w", encoding="utf-8") as handle:
        json.dump(dataset_payload, handle, indent=2)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(render_parent_survival_band_report(selection, grouped_summary, survival_rows) + "\n")
    write_parent_survival_csv(survival_rows, table_csv_path)
    write_parent_patterns_csv(pattern_rows, patterns_csv_path)
    write_parent_survival_band_summary_csv(grouped_summary, band_csv_path)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, indent=2)

    if not args.quiet:
        if family_inferred:
            print("Using the latest coherent Phase 1 family inferred from the newest observed run.")
            print("")
        print(render_parent_survival_band_console_summary(grouped_summary))
        print("")
        print(f"Saved dataset to: {dataset_path}")
        print(f"Saved summary to: {summary_path}")
        print(f"Saved report to: {report_path}")
        print(f"Saved CSV table to: {table_csv_path}")
        print(f"Saved patterns CSV to: {patterns_csv_path}")
        print(f"Saved band summary CSV to: {band_csv_path}")
        print(f"Saved manifest to: {manifest_path}")

    return 0


def resolve_candidate_lag_bits(args, parser: argparse.ArgumentParser) -> int:
    if args.candidate_lag_bits is not None:
        return int(args.candidate_lag_bits)
    if args.lag_summary:
        lag_summary_path = resolve_dir(args.lag_summary, anchor_file=__file__)
        if lag_summary_path.is_dir():
            lag_summary_path = lag_summary_path / "summary.json"
        if not lag_summary_path.is_file():
            parser.error(f"Lag summary not found: {lag_summary_path}")
        with open(lag_summary_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        summary = payload.get("summary", {})
        lag_bits = summary.get("recommended_lag_bits")
        if lag_bits is None:
            parser.error(f"The lag summary does not contain recommended_lag_bits: {lag_summary_path}")
        return int(lag_bits)
    parser.error("Provide either --candidate-lag-bits or --lag-summary.")


def build_null_specs_for_candidate(
    null_specs: list[dict],
    *,
    anchor_run: dict,
    candidate_variant: str,
) -> list[dict]:
    specs = []
    for item in null_specs:
        specs.append(
            {
                "variant": candidate_variant,
                "run": item["run"],
                "anchor_run": anchor_run,
                "anchor_variant": str(anchor_run["dataset"]["config"]["variant"]).upper(),
                "anchor_kind": "observed-fixed",
                "source_kind": "null_surrogate",
                "source_label": item["null_label"],
                "null_model": item["null_model"],
                "null_seed": item.get("null_seed"),
            }
        )
    return specs


def build_run_slug(selection: dict, timestamp: str) -> str:
    offsets = selection["offsets"]
    if len(offsets) == 1:
        offset_part = f"off-{compact_int(offsets[0])}"
    else:
        offset_part = f"off-{compact_int(min(offsets))}-plus-{len(offsets)}"
    variants = "-".join(selection["variants"])
    lag_part = compact_int(selection["candidate_lag_bits"])
    return (
        f"phase2-parent-survival-band-lagaware__anchor-{selection['anchor_variant']}"
        f"__cand-{selection['candidate_variant']}__lag-{lag_part}"
        f"__m-{selection['low_scale']}-{selection['high_scale']}"
        f"__sel-{selection['pattern_selection']}__top-{selection['top_patterns']}"
        f"__{offset_part}__var-{variants}__{timestamp}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
