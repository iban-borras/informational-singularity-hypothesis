#!/usr/bin/env python3
"""
HSI v2 Preflight - Factor Complexity and Branching

Minimal vertical slice for HSI v2:
- loads observable bits from a real structural Level 0 snapshot
- splits a prefix into consecutive segments
- computes factor complexity p(m), h_eff(m), and left/right branching
- compares the observed segments against strong cheap nulls

This preflight studies observed factor spaces L_m, not thresholded P_m yet.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from v2.common.io import (
    companion_metadata_path,
    load_observable_prefix_bits,
    load_struct_metadata,
    normalize_variant,
    resolve_struct_path,
    split_into_segments,
)
from v2.common.naming import build_preflight_run_slug
from v2.common.null_models import generate_control, parse_nulls
from v2.preflight.factor_complexity import analyze_factor_complexity
from utils.progress import ProgressIndicator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HSI v2 preflight: factor complexity, branching, h_eff, and cheap strong nulls."
    )
    parser.add_argument("--variant", type=str, help="HSI variant code, e.g. B, E, I, F, A.")
    parser.add_argument("--iteration", type=int, help="Iteration to analyze. Defaults to latest available.")
    parser.add_argument("--input", type=str, help="Explicit path to phi_iter*.struct.gz. Overrides variant/iteration.")
    parser.add_argument(
        "--level0-root",
        type=str,
        default=None,
        help="Optional override for the Level 0 snapshots root directory.",
    )
    parser.add_argument("--segment-bits", type=int, default=1_000_000, help="Bits per segment.")
    parser.add_argument("--num-segments", type=int, default=3, help="Number of consecutive segments to analyze.")
    parser.add_argument("--m-min", type=int, default=8, help="Minimum factor size m.")
    parser.add_argument("--m-max", type=int, default=32, help="Maximum factor size m.")
    parser.add_argument(
        "--nulls",
        type=str,
        default="shuffled,same-density,markov1",
        help="Comma-separated nulls to generate inside the run.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base seed for null generation.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/hsi_v2/preflight",
        help="Output directory, relative to hsi_agents_project or absolute.",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce console output.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.m_min > args.m_max:
        parser.error("--m-min cannot be greater than --m-max.")
    if args.segment_bits <= 0 or args.num_segments <= 0:
        parser.error("--segment-bits and --num-segments must be positive.")

    variant_label = normalize_variant(args.variant) if args.variant else "CUSTOM"
    nulls = parse_nulls(args.nulls)
    m_values = list(range(args.m_min, args.m_max + 1))

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
    iteration = args.iteration if args.iteration is not None else metadata.get("iteration")

    total_bits = args.segment_bits * args.num_segments
    phase_print(
        "Resolved input snapshot",
        f"variant={variant_label} | iteration={iteration} | source={struct_path.name}",
        quiet=args.quiet,
    )
    phase_print(
        "Loading observable prefix",
        f"{total_bits:,} bits from structural snapshot",
        quiet=args.quiet,
    )
    load_progress = ProgressIndicator("Loading observable bits", total=total_bits) if not args.quiet else None
    if load_progress is not None:
        with load_progress:
            bits = load_observable_prefix_bits(struct_path, total_bits=total_bits, progress=load_progress)
    else:
        bits = load_observable_prefix_bits(struct_path, total_bits=total_bits)

    segments = split_into_segments(bits, args.segment_bits, args.num_segments)
    phase_print(
        "Prepared segment protocol",
        f"{len(segments)} segments x {args.segment_bits:,} bits | m={args.m_min}..{args.m_max}",
        quiet=args.quiet,
    )

    phase_print(
        "Analyzing observed segments",
        f"{len(segments) * len(m_values)} scale evaluations",
        quiet=args.quiet,
    )
    observed_progress = ProgressIndicator("Observed analysis", total=len(segments) * len(m_values)) if not args.quiet else None
    if observed_progress is not None:
        with observed_progress:
            observed_result = analyze_segments(
                segments,
                m_values,
                base_seed=args.seed,
                label="observed",
                progress=observed_progress,
            )
    else:
        observed_result = analyze_segments(
            segments,
            m_values,
            base_seed=args.seed,
            label="observed",
        )

    results = {
        "observed": observed_result,
    }
    null_generation: dict[str, list[dict[str, object]]] = {}

    for null_name in nulls:
        phase_print(
            f"Generating and analyzing null: {null_name}",
            f"{len(segments)} segments + {len(segments) * len(m_values)} scale evaluations",
            quiet=args.quiet,
        )
        null_segments = []
        null_generation[null_name] = []
        null_progress = ProgressIndicator(
            f"{null_name} pipeline",
            total=len(segments) + len(segments) * len(m_values),
        ) if not args.quiet else None
        if null_progress is not None:
            with null_progress:
                for index, segment in enumerate(segments, start=1):
                    generated, metadata = generate_control(segment, null_name, seed=args.seed + index - 1)
                    null_segments.append(generated)
                    if metadata is not None:
                        null_generation[null_name].append(
                            {"segment_index": index, **metadata}
                        )
                    null_progress.update(index, message=f"generated segment {index}/{len(segments)}")
                results[null_name] = analyze_segments(
                    null_segments,
                    m_values,
                    base_seed=args.seed,
                    label=null_name,
                    progress=null_progress,
                    progress_start=len(segments),
                )
        else:
            for index, segment in enumerate(segments):
                generated, metadata = generate_control(segment, null_name, seed=args.seed + index)
                null_segments.append(generated)
                if metadata is not None:
                    null_generation[null_name].append(
                        {"segment_index": index + 1, **metadata}
                    )
            results[null_name] = analyze_segments(null_segments, m_values, base_seed=args.seed, label=null_name)

    output_dir = resolve_output_dir(args.output_dir)
    run_slug = build_preflight_run_slug(
        variant=variant_label,
        iteration=int(iteration) if iteration is not None else None,
        segment_bits=args.segment_bits,
        num_segments=args.num_segments,
        m_min=args.m_min,
        m_max=args.m_max,
        nulls=nulls,
    )
    run_dir = output_dir / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)
    phase_print("Writing artifacts", str(run_dir), quiet=args.quiet)

    result_payload = {
        "stage": "preflight_factor_complexity",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "variant": variant_label,
            "iteration": iteration,
            "input_struct_path": str(struct_path),
            "input_metadata_path": str(companion_metadata_path(struct_path)),
            "segment_bits": args.segment_bits,
            "num_segments": args.num_segments,
            "m_min": args.m_min,
            "m_max": args.m_max,
            "m_values": m_values,
            "nulls": nulls,
            "seed": args.seed,
            "sampling_policy": "prefix_consecutive_segments",
            "pattern_space_mode": "observed_factors_L_m",
        },
        "source_metadata": metadata,
        "results": results,
        "null_generation": null_generation,
        "notes": [
            "This preflight studies observed factor spaces L_m, not thresholded P_m.",
            "Raw binary factor branching p(m+1)/p(m) cannot exceed 2; any phi+1 bridge must use a derived observable.",
            "Cheap strong nulls are generated inside the same run to support immediate go/no-go judgment.",
        ],
    }

    result_path = run_dir / "factor_complexity.json"
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result_payload, handle, indent=2)

    manifest_path = run_dir / "manifest.json"
    manifest = {
        "run_slug": run_slug,
        "generated_at": result_payload["generated_at"],
        "script": "hsi_v2_preflight_factor_complexity.py",
        "cwd": str(Path.cwd()),
        "outputs": {
            "result_json": str(result_path),
        },
        "inputs": {
            "struct_path": str(struct_path),
            "metadata_path": str(companion_metadata_path(struct_path)),
        },
        "arguments": vars(args),
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    if not args.quiet:
        print_summary(results)
        print(f"\nSaved result to: {result_path}")
        print(f"Saved manifest to: {manifest_path}")

    return 0


def analyze_segments(
    segments: list[np.ndarray],
    m_values: list[int],
    base_seed: int,
    label: str,
    progress=None,
    progress_start: int = 0,
) -> dict:
    analyzed_segments = []
    for index, segment_bits in enumerate(segments, start=1):
        analysis = analyze_factor_complexity(
            segment_bits,
            m_values,
            progress=progress,
            progress_base=progress_start + (index - 1) * len(m_values),
            progress_message_prefix=f"seg {index}/{len(segments)} | ",
        )
        analysis["segment_index"] = index
        analysis["analysis_label"] = label
        analysis["seed"] = int(base_seed + index - 1)
        analyzed_segments.append(analysis)

    tail_h = [item["tail_summary"]["mean_h_eff_tail"] for item in analyzed_segments]
    tail_right = [item["tail_summary"]["mean_right_branching_tail"] for item in analyzed_segments]
    tail_left = [item["tail_summary"]["mean_left_branching_tail"] for item in analyzed_segments]
    tail_rho = [item["tail_summary"]["mean_rho_tail"] for item in analyzed_segments if item["tail_summary"]["mean_rho_tail"] is not None]

    return {
        "segments": analyzed_segments,
        "aggregate": {
            "segment_count": len(analyzed_segments),
            "mean_tail_h_eff": float(np.mean(tail_h)),
            "mean_tail_right_branching": float(np.mean(tail_right)),
            "mean_tail_left_branching": float(np.mean(tail_left)),
            "mean_tail_rho": float(np.mean(tail_rho)) if tail_rho else None,
        },
    }


def print_summary(results: dict) -> None:
    print("\nPreflight tail summary")
    print("-" * 72)
    for label, payload in results.items():
        aggregate = payload["aggregate"]
        print(
            f"{label:14} "
            f"h_eff={aggregate['mean_tail_h_eff']:.4f}  "
            f"right={aggregate['mean_tail_right_branching']:.4f}  "
            f"left={aggregate['mean_tail_left_branching']:.4f}  "
            f"rho={aggregate['mean_tail_rho']:.4f}" if aggregate["mean_tail_rho"] is not None else
            f"{label:14} "
            f"h_eff={aggregate['mean_tail_h_eff']:.4f}  "
            f"right={aggregate['mean_tail_right_branching']:.4f}  "
            f"left={aggregate['mean_tail_left_branching']:.4f}  "
            "rho=n/a"
        )


def resolve_output_dir(raw_output_dir: str) -> Path:
    output_path = Path(raw_output_dir)
    if not output_path.is_absolute():
        output_path = Path(__file__).parent / output_path
    return output_path.resolve()


def phase_print(title: str, detail: str = "", quiet: bool = False) -> None:
    if quiet:
        return
    print(f"\n[Phase] {title}")
    if detail:
        print(f"        {detail}")


if __name__ == "__main__":
    raise SystemExit(main())
