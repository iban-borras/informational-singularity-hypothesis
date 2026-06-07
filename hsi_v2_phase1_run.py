#!/usr/bin/env python3
"""
HSI v2 Phase 1 - Pattern Spaces, Projections, Fibers, and Moments

First executable vertical slice for the HSI v2 Phase 1 gate:
- loads observable bits from a real Level 0 structural snapshot
- builds locally stable pattern spaces P_m across a scale ladder
- constructs adjacent-scale projection towers under explicit policies
- computes fibers and q-moments for q = 0, 1, 2, 3
- writes reproducible artifacts, manifests, and a concise run summary
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from v2.common.io import (
    companion_metadata_path,
    load_observable_prefix_bits,
    load_struct_metadata,
    normalize_variant,
    resolve_struct_path,
    split_into_segments,
)
from v2.common.cli import resolve_results_dir
from v2.common.naming import build_phase1_run_slug
from v2.common.null_models import SUPPORTED_NULLS, generate_control
from v2.phase1.tower import (
    PRIMARY_Q_VALUES,
    build_pattern_spaces,
    build_projection_bundle,
    parse_policies,
    parse_scales,
    serialize_pattern_spaces,
    summarize_phase1,
)
from utils.progress import ProgressIndicator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HSI v2 Phase 1: build P_m, projections, fibers, and moments over a scale ladder."
    )
    parser.add_argument("--variant", type=str, help="HSI variant code, e.g. B, F, A, M.")
    parser.add_argument("--iteration", type=int, help="Iteration to analyze. Defaults to latest available.")
    parser.add_argument("--input", type=str, help="Explicit path to phi_iter*.struct.gz. Overrides variant/iteration.")
    parser.add_argument(
        "--level0-root",
        type=str,
        default=None,
        help="Optional override for the Level 0 snapshots root directory.",
    )
    parser.add_argument("--segment-bits", type=int, default=1_000_000, help="Bits per segment.")
    parser.add_argument("--num-segments", type=int, default=3, help="Number of consecutive segments.")
    parser.add_argument(
        "--segment-offset-bits",
        type=int,
        default=0,
        help="Optional offset, in bits, before the first analyzed segment begins.",
    )
    parser.add_argument(
        "--scales",
        type=str,
        default="8,12,16,20,24,28,32",
        help="Comma-separated ascending scale ladder.",
    )
    parser.add_argument(
        "--policies",
        type=str,
        default="prefix,suffix",
        help="Comma-separated projection policies. Phase 1 defaults to prefix + suffix control.",
    )
    parser.add_argument(
        "--min-count-floor",
        type=int,
        default=16,
        help="Absolute minimum total count for a pattern to enter P_m.",
    )
    parser.add_argument(
        "--min-count-rate",
        type=float,
        default=1e-6,
        help="Relative minimum total count threshold applied to total windows per scale.",
    )
    parser.add_argument(
        "--min-segment-support",
        type=int,
        default=None,
        help="Minimum number of segments in which a pattern must appear. Default: max(2, ceil(S/2)).",
    )
    parser.add_argument(
        "--cv-max",
        type=float,
        default=1.5,
        help="Maximum coefficient of variation across supporting segments.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/hsi_v2/phase1",
        help=(
            "Output directory, relative to hsi_agents_project or absolute. "
            "If relative and rooted at results/, HSI_RESULTS_BASE_DIR can redirect that root."
        ),
    )
    parser.add_argument(
        "--null-model",
        type=str,
        choices=SUPPORTED_NULLS,
        default=None,
        help="Optional strong null surrogate built from the observed source sequence.",
    )
    parser.add_argument(
        "--null-seed",
        type=int,
        default=17,
        help="Deterministic seed used when --null-model is enabled.",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce console output.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.segment_bits <= 0 or args.num_segments <= 0:
        parser.error("--segment-bits and --num-segments must be positive.")
    if args.segment_offset_bits < 0:
        parser.error("--segment-offset-bits cannot be negative.")
    if args.min_count_floor <= 0:
        parser.error("--min-count-floor must be positive.")
    if args.min_count_rate < 0:
        parser.error("--min-count-rate cannot be negative.")
    if args.min_segment_support is not None and args.min_segment_support <= 0:
        parser.error("--min-segment-support must be positive when provided.")
    if args.cv_max <= 0:
        parser.error("--cv-max must be positive.")

    try:
        scales = parse_scales(args.scales)
        policies = parse_policies(args.policies)
    except ValueError as exc:
        parser.error(str(exc))

    if args.segment_bits < max(scales):
        parser.error("--segment-bits must be at least as large as the maximum scale.")

    try:
        struct_path = resolve_struct_path(
            input_path=args.input,
            variant=args.variant,
            iteration=args.iteration,
            level0_root=args.level0_root,
        )
    except FileNotFoundError as exc:
        print(f"\n[Error] {exc}", file=sys.stderr)
        return 2

    metadata = load_struct_metadata(struct_path)
    variant_label = infer_variant_label(args.variant, metadata)
    iteration = args.iteration if args.iteration is not None else metadata.get("iteration")
    analyzed_bits = args.segment_bits * args.num_segments
    loaded_bits = args.segment_offset_bits + analyzed_bits
    analysis_variant_label = variant_label

    phase_print(
        "Resolved input snapshot",
        f"variant={variant_label} | iteration={iteration} | source={struct_path.name}",
        quiet=args.quiet,
    )
    if args.segment_offset_bits > 0:
        phase_print(
            "Loading observable slice",
            (
                f"{loaded_bits:,} bits from structural snapshot"
                f" | offset={args.segment_offset_bits:,}"
                f" | analyzed={analyzed_bits:,}"
            ),
            quiet=args.quiet,
        )
    else:
        phase_print(
            "Loading observable prefix",
            f"{loaded_bits:,} bits from structural snapshot",
            quiet=args.quiet,
        )

    load_progress = ProgressIndicator("Loading observable bits", total=loaded_bits) if not args.quiet else None
    if load_progress is not None:
        with load_progress:
            bits = load_observable_prefix_bits(struct_path, total_bits=loaded_bits, progress=load_progress)
    else:
        bits = load_observable_prefix_bits(struct_path, total_bits=loaded_bits)

    source_bits = bits
    sequence_kind = "observed"
    null_metadata: dict[str, object] | None = None
    if args.null_model:
        phase_print(
            "Generating strong null surrogate",
            f"model={args.null_model} | seed={args.null_seed} | source_variant={variant_label}",
            quiet=args.quiet,
        )
        bits, null_metadata = generate_control(
            source_bits,
            args.null_model,
            args.null_seed,
            show_progress=not args.quiet,
            progress_label=f"{variant_label}-{args.null_model}",
        )
        analysis_variant_label = f"{variant_label}-{args.null_model}"
        sequence_kind = "null_surrogate"

    segments = split_into_segments(
        bits,
        args.segment_bits,
        args.num_segments,
        start_offset_bits=args.segment_offset_bits,
    )
    effective_segment_support = args.min_segment_support or max(2, (args.num_segments + 1) // 2)
    phase_print(
        "Prepared segment protocol",
        (
            f"{len(segments)} segments x {args.segment_bits:,} bits"
            f" | offset={args.segment_offset_bits:,}"
            f" | scales={','.join(str(value) for value in scales)}"
            f" | policies={','.join(policies)}"
        ),
        quiet=args.quiet,
    )

    phase_print(
        "Building local pattern spaces P_m",
        (
            f"{len(scales)} scales | min_count=max({args.min_count_floor}, ceil({args.min_count_rate:g} * n_windows))"
            f" | min_support={effective_segment_support} | cv_max={args.cv_max}"
        ),
        quiet=args.quiet,
    )
    pattern_progress = ProgressIndicator("Pattern space construction", total=len(scales)) if not args.quiet else None
    if pattern_progress is not None:
        with pattern_progress:
            pattern_spaces = build_pattern_spaces(
                segments,
                scales,
                min_count_floor=args.min_count_floor,
                min_count_rate=args.min_count_rate,
                min_segment_support=args.min_segment_support,
                cv_max=args.cv_max,
                progress=pattern_progress,
            )
    else:
        pattern_spaces = build_pattern_spaces(
            segments,
            scales,
            min_count_floor=args.min_count_floor,
            min_count_rate=args.min_count_rate,
            min_segment_support=args.min_segment_support,
            cv_max=args.cv_max,
        )

    projection_steps = len(policies) * (len(scales) - 1)
    phase_print(
        "Building projection towers and fibers",
        f"{projection_steps} adjacent-scale edges | q={','.join(str(value) for value in PRIMARY_Q_VALUES)}",
        quiet=args.quiet,
    )
    projection_progress = ProgressIndicator("Projection + fiber analysis", total=projection_steps) if not args.quiet else None
    if projection_progress is not None:
        with projection_progress:
            projection_bundle = build_projection_bundle(
                pattern_spaces,
                scales,
                policies,
                q_values=PRIMARY_Q_VALUES,
                progress=projection_progress,
            )
    else:
        projection_bundle = build_projection_bundle(
            pattern_spaces,
            scales,
            policies,
            q_values=PRIMARY_Q_VALUES,
        )

    serialized_pattern_spaces = serialize_pattern_spaces(pattern_spaces)
    phase1_summary = summarize_phase1(pattern_spaces, projection_bundle)

    output_dir = resolve_output_dir(args.output_dir)
    run_slug = build_phase1_run_slug(
        variant=variant_label,
        iteration=int(iteration) if iteration is not None else None,
        segment_bits=args.segment_bits,
        num_segments=args.num_segments,
        segment_offset_bits=args.segment_offset_bits,
        scales=scales,
        policies=policies,
        null_model=args.null_model,
    )
    run_root = output_dir / "nulls" if args.null_model else output_dir
    run_dir = run_root / run_slug
    pattern_dir = run_dir / "pattern_spaces"
    projection_dir = run_dir / "projection_towers"
    fibers_dir = run_dir / "fibers"
    moments_dir = run_dir / "moments"
    for directory in (pattern_dir, projection_dir, fibers_dir, moments_dir):
        directory.mkdir(parents=True, exist_ok=True)

    dataset_payload = {
        "stage": "phase1_tower",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "variant": analysis_variant_label,
            "iteration": iteration,
            "source_variant": variant_label,
            "source_iteration": iteration,
            "sequence_kind": sequence_kind,
            "null_model": args.null_model,
            "null_seed": args.null_seed if args.null_model else None,
            "null_model_details": null_metadata,
            "input_struct_path": str(struct_path),
            "input_metadata_path": str(companion_metadata_path(struct_path)),
            "segment_bits": args.segment_bits,
            "num_segments": args.num_segments,
            "segment_offset_bits": args.segment_offset_bits,
            "total_observable_bits": analyzed_bits,
            "loaded_observable_bits": loaded_bits,
            "scales": scales,
            "policies": policies,
            "min_count_floor": args.min_count_floor,
            "min_count_rate": args.min_count_rate,
            "min_segment_support_requested": args.min_segment_support,
            "min_segment_support_effective": effective_segment_support,
            "cv_max": args.cv_max,
            "sampling_policy": "prefix_consecutive_segments" if args.segment_offset_bits == 0 else "offset_consecutive_segments",
            "pattern_space_mode": "locally_stable_patterns",
            "projection_main_policy": "prefix" if "prefix" in policies else policies[0],
            "projection_control_policies": [policy for policy in policies if policy != "prefix"],
            "moment_orders": list(PRIMARY_Q_VALUES),
        },
        "source_metadata": metadata,
        "segment_protocol": {
            "segment_count": len(segments),
            "segment_bits": args.segment_bits,
            "start_offset_bits": args.segment_offset_bits,
            "loaded_observable_bits": loaded_bits,
            "segment_indexing": "consecutive_prefix_segments" if args.segment_offset_bits == 0 else "consecutive_offset_segments",
            "segment_window_counts": {
                str(m): [args.segment_bits - m + 1 for _ in range(args.num_segments)]
                for m in scales
            },
        },
        "notes": [
            "P_m is defined within scale using count, segment support, and CV stability thresholds.",
            "Inter-scale coherence is tested after P_m construction through explicit projection policies and fibers.",
            "Prefix is the primary Phase 1 projection; suffix is a mandatory control.",
        ],
    }
    if args.null_model:
        dataset_payload["notes"].append(
            f"Observable bits were replaced by a deterministic {args.null_model} surrogate with seed {args.null_seed}."
        )
        if null_metadata is not None:
            dataset_payload["notes"].append(
                f"{args.null_model} calibration: {json.dumps(null_metadata, sort_keys=True)}"
            )
    if args.segment_offset_bits > 0:
        dataset_payload["notes"].append(
            f"Analyzed segments start at observable bit offset {args.segment_offset_bits}."
        )

    manifest = {
        "run_slug": run_slug,
        "generated_at": dataset_payload["generated_at"],
        "script": "hsi_v2_phase1_run.py",
        "cwd": str(Path.cwd()),
        "outputs": {
            "dataset": str(run_dir / "dataset.json"),
            "pattern_spaces": str(pattern_dir),
            "projection_towers": str(projection_dir),
            "fibers": str(fibers_dir),
            "moments": str(moments_dir),
            "summary": str(run_dir / "phase1_summary.json"),
            "manifest": str(run_dir / "manifest.json"),
        },
        "inputs": {
            "struct_path": str(struct_path),
            "metadata_path": str(companion_metadata_path(struct_path)),
            "sequence_kind": sequence_kind,
            "source_variant": variant_label,
            "source_iteration": iteration,
            "null_model": args.null_model,
            "null_seed": args.null_seed if args.null_model else None,
            "null_model_details": null_metadata,
            "segment_offset_bits": args.segment_offset_bits,
        },
        "arguments": vars(args),
    }

    write_steps = 3 + len(scales) + (3 * len(policies))
    phase_print("Writing artifacts", str(run_dir), quiet=args.quiet)
    write_progress = ProgressIndicator("Writing Phase 1 artifacts", total=write_steps) if not args.quiet else None
    if write_progress is not None:
        with write_progress:
            write_phase1_artifacts(
                run_dir=run_dir,
                pattern_dir=pattern_dir,
                projection_dir=projection_dir,
                fibers_dir=fibers_dir,
                moments_dir=moments_dir,
                dataset_payload=dataset_payload,
                pattern_spaces=serialized_pattern_spaces,
                projection_bundle=projection_bundle,
                phase1_summary=phase1_summary,
                manifest=manifest,
                progress=write_progress,
            )
    else:
        write_phase1_artifacts(
            run_dir=run_dir,
            pattern_dir=pattern_dir,
            projection_dir=projection_dir,
            fibers_dir=fibers_dir,
            moments_dir=moments_dir,
            dataset_payload=dataset_payload,
            pattern_spaces=serialized_pattern_spaces,
            projection_bundle=projection_bundle,
            phase1_summary=phase1_summary,
            manifest=manifest,
        )

    if not args.quiet:
        print_summary(phase1_summary)
        print(f"\nSaved dataset to: {run_dir / 'dataset.json'}")
        print(f"Saved summary to: {run_dir / 'phase1_summary.json'}")
        print(f"Saved manifest to: {run_dir / 'manifest.json'}")

    return 0


def write_phase1_artifacts(
    *,
    run_dir: Path,
    pattern_dir: Path,
    projection_dir: Path,
    fibers_dir: Path,
    moments_dir: Path,
    dataset_payload: dict,
    pattern_spaces: dict[int, dict],
    projection_bundle: dict,
    phase1_summary: dict,
    manifest: dict,
    progress=None,
) -> None:
    step = 0

    save_json(run_dir / "dataset.json", dataset_payload)
    step += 1
    if progress is not None:
        progress.update(step, message="dataset.json")

    for m, payload in sorted(pattern_spaces.items()):
        save_json(pattern_dir / f"pattern_space_m{m}.json", payload)
        step += 1
        if progress is not None:
            progress.update(step, message=f"pattern_space_m{m}.json")

    for policy, artifacts in projection_bundle["projection_towers"].items():
        save_json(projection_dir / f"projection_{policy}.json", artifacts)
        step += 1
        if progress is not None:
            progress.update(step, message=f"projection_{policy}.json")

    for policy, artifacts in projection_bundle["fibers"].items():
        save_json(fibers_dir / f"fibers_{policy}.json", artifacts)
        step += 1
        if progress is not None:
            progress.update(step, message=f"fibers_{policy}.json")

    for policy, artifacts in projection_bundle["moments"].items():
        save_json(moments_dir / f"moments_{policy}.json", artifacts)
        step += 1
        if progress is not None:
            progress.update(step, message=f"moments_{policy}.json")

    save_json(run_dir / "phase1_summary.json", phase1_summary)
    step += 1
    if progress is not None:
        progress.update(step, message="phase1_summary.json")

    save_json(run_dir / "manifest.json", manifest)
    step += 1
    if progress is not None:
        progress.update(step, message="manifest.json")


def infer_variant_label(raw_variant: str | None, metadata: dict) -> str:
    if raw_variant:
        return normalize_variant(raw_variant)

    for key in ("variant", "variant_code", "variant_name"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return normalize_variant(value)
    return "CUSTOM"


def save_json(path: Path, payload: dict | list) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def resolve_output_dir(raw_output_dir: str) -> Path:
    return resolve_results_dir(raw_output_dir, anchor_file=__file__)


def print_summary(summary: dict) -> None:
    print("\nPhase 1 scale summary")
    print("-" * 86)
    print(f"{'m':>4} {'kept':>8} {'candidate':>10} {'min_count':>10} {'support':>8} {'cv_max':>8}")
    for item in summary["scale_summary"]:
        print(
            f"{item['m']:>4} "
            f"{item['kept_pattern_count']:>8} "
            f"{item['candidate_pattern_count']:>10} "
            f"{item['min_count_threshold']:>10} "
            f"{item['min_segment_support']:>8} "
            f"{item['cv_max']:>8.2f}"
        )

    for policy, items in summary["fiber_summary"].items():
        print(f"\nFiber summary | policy={policy}")
        print("-" * 86)
        print(
            f"{'edge':>12} {'survival':>10} {'mean_fiber':>12} "
            f"{'singleton':>12} {'entropy':>10} {'orphans':>8}"
        )
        for item in items:
            edge = f"{item['source_m']}->{item['target_m']}"
            print(
                f"{edge:>12} "
                f"{item['survival_fraction']:>10.4f} "
                f"{item['mean_fiber_size']:>12.4f} "
                f"{item['singleton_fraction']:>12.4f} "
                f"{item['mean_child_entropy']:>10.4f} "
                f"{item['orphan_child_count']:>8}"
            )


def phase_print(title: str, detail: str = "", quiet: bool = False) -> None:
    if quiet:
        return
    print(f"\n[Phase] {title}")
    if detail:
        print(f"        {detail}")


if __name__ == "__main__":
    raise SystemExit(main())
