#!/usr/bin/env python3
"""Raw-sequence orthogonal metrics for HSI v2 hostile-review follow-up.

The script deliberately stays outside the Phase 1 projective tower. It reads
frozen Level 0 observable bits, derives B-based null surrogates in memory, and
computes raw-sequence metrics that do not use pattern projections or fibers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v2.common.io import (
    detect_latest_iteration,
    load_observable_prefix_bits,
    load_struct_metadata,
    resolve_struct_path,
)
from v2.common.null_models import generate_control

try:  # Optional only; the deterministic proxy below is always emitted.
    from pybdm import BDM  # type: ignore

    HAS_PYBDM = True
except Exception:  # pragma: no cover - depends on local environment.
    BDM = None
    HAS_PYBDM = False


DEFAULT_VARIANTS = ("B", "F", "M", "N", "O", "P")
DEFAULT_MI_LAGS = (1_024, 4_096, 16_384, 65_536, 262_144, 1_048_576)
PRIMARY_FEATURES = (
    "long_range_mi_tail_mean_bits",
    "long_range_mi_auc_mean_bits",
    "spectral_entropy_norm",
    "spectral_low_frequency_power_fraction",
    "spectral_peak_to_mean_power",
    "block_entropy_norm",
    "block_decomposition_proxy_per_bit",
    "zlib_ratio",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute raw-sequence metrics that do not use the Phase 1 projective "
            "tower: long-range block MI, Fourier spectrum, and block-decomposition "
            "complexity proxy/BDM when available."
        )
    )
    parser.add_argument("--level0-root", default="results/level0/phi_snapshots")
    parser.add_argument("--output-dir", default="results/hsi_v2/raw_sequence_orthogonal_metrics")
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--iteration", type=int, default=20)
    parser.add_argument(
        "--latest-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the latest available snapshot when a variant lacks --iteration.",
    )
    parser.add_argument("--sample-bits", type=int, default=3_000_000)
    parser.add_argument("--null-source-variant", default="B")
    parser.add_argument("--include-shuffled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shuffled-seeds", default="17")
    parser.add_argument("--include-matched-lz", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--matched-lz-seeds", default="17")
    parser.add_argument("--include-phase-matched-lz", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--phase-matched-lz-seeds", default="607,709")
    parser.add_argument("--mi-block-bits", type=int, default=8)
    parser.add_argument("--mi-lags", default=",".join(str(value) for value in DEFAULT_MI_LAGS))
    parser.add_argument("--mi-max-pairs", type=int, default=200_000)
    parser.add_argument("--fft-window-bits", type=int, default=65_536)
    parser.add_argument("--fft-max-windows", type=int, default=32)
    parser.add_argument("--block-bits", type=int, default=16)
    parser.add_argument("--pybdm-max-bits", type=int, default=100_000)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    started = time.perf_counter()

    level0_root = resolve_project_path(args.level0_root)
    output_root = resolve_project_path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    variants = parse_csv(args.variants)
    mi_lags = parse_int_csv(args.mi_lags)
    source_specs = build_source_specs(args)

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_slug = (
        f"raw-sequence-orthogonal-metrics__obs-{'-'.join(variants)}"
        f"__bits-{format_slug_int(args.sample_bits)}__{timestamp}"
    )
    run_dir = output_root / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)

    phase_print(
        "Preparing raw-sequence orthogonal metrics",
        (
            f"variants={','.join(variants)} | null-source={args.null_source_variant.upper()} "
            f"| sample_bits={args.sample_bits:,}"
        ),
        quiet=args.quiet,
    )

    observed_bits: dict[str, np.ndarray] = {}
    source_rows: list[dict[str, Any]] = []
    for index, variant in enumerate(variants, start=1):
        phase_print(
            f"Loading observed {variant}",
            f"{index}/{len(variants)} | first {args.sample_bits:,} observable bits",
            quiet=args.quiet,
        )
        struct_path, actual_iteration = resolve_variant_struct_path(
            variant=variant,
            iteration=args.iteration,
            level0_root=level0_root,
            latest_fallback=args.latest_fallback,
        )
        bits = load_observable_prefix_bits(struct_path, args.sample_bits)
        observed_bits[variant.upper()] = bits
        metadata = load_struct_metadata(struct_path)
        source_rows.append(
            {
                "label": variant.upper(),
                "variant": variant.upper(),
                "kind": "observed",
                "source_variant": variant.upper(),
                "requested_iteration": int(args.iteration),
                "actual_iteration": int(actual_iteration),
                "seed": None,
                "null_model": None,
                "struct_path": str(struct_path),
                "struct_sha256": sha256_file(struct_path),
                "metadata_path": str(struct_path.with_suffix("").with_suffix(".json")),
                "metadata_sha256_struct_gz": metadata.get("sha256_struct_gz"),
                "observable_bits": int(bits.size),
            }
        )

    null_source_variant = args.null_source_variant.upper()
    if null_source_variant not in observed_bits:
        phase_print(
            f"Loading null source {null_source_variant}",
            "null source is not in observed variants, loading separately",
            quiet=args.quiet,
        )
        struct_path, _actual_iteration = resolve_variant_struct_path(
            variant=null_source_variant,
            iteration=args.iteration,
            level0_root=level0_root,
            latest_fallback=args.latest_fallback,
        )
        observed_bits[null_source_variant] = load_observable_prefix_bits(struct_path, args.sample_bits)

    sequences: list[dict[str, Any]] = []
    for row in source_rows:
        sequences.append({"row": row, "bits": observed_bits[row["variant"]]})

    base_bits = observed_bits[null_source_variant]
    for spec in source_specs:
        phase_print(
            f"Generating {spec['null_model']} null",
            f"source={null_source_variant} | seed={spec['seed']}",
            quiet=args.quiet,
        )
        generated, metadata = generate_control(
            base_bits,
            spec["null_model"],
            int(spec["seed"]),
            show_progress=not args.quiet,
            progress_label=f"{null_source_variant}-{spec['null_model']}-seed-{spec['seed']}",
        )
        label = f"{null_source_variant}-{spec['null_model']} seed {spec['seed']}"
        row = {
            "label": label,
            "variant": f"{null_source_variant}-{spec['null_model']}",
            "kind": "null",
            "source_variant": null_source_variant,
            "seed": int(spec["seed"]),
            "null_model": spec["null_model"],
            "null_metadata": metadata,
            "observable_bits": int(generated.size),
        }
        sequences.append({"row": row, "bits": generated})

    metric_rows: list[dict[str, Any]] = []
    detail_payload: dict[str, Any] = {}
    for index, item in enumerate(sequences, start=1):
        row = item["row"]
        bits = item["bits"]
        phase_print(
            f"Computing metrics for {row['label']}",
            f"{index}/{len(sequences)} | bits={bits.size:,}",
            quiet=args.quiet,
        )
        metrics, details = compute_all_metrics(bits, args=args, mi_lags=mi_lags)
        metric_row = dict(row)
        metric_row.update(metrics)
        metric_rows.append(metric_row)
        detail_payload[row["label"]] = details

    margin_rows, feature_summary = build_margin_rows(metric_rows)
    grouped_feature_summary = build_group_feature_summary(metric_rows)

    summary_path = run_dir / "summary.json"
    metrics_csv_path = run_dir / "metrics.csv"
    margins_csv_path = run_dir / "margins.csv"
    report_path = run_dir / "report.md"
    manifest_path = run_dir / "manifest.json"

    summary = {
        "generated_at": now_utc(),
        "script": Path(__file__).name,
        "status": "completed",
        "question": {
            "id": "hostile-review-Q1",
            "reading": (
                "Raw-sequence metrics that do not use projective pattern spaces, "
                "prefix/suffix projections, or fiber survival."
            ),
        },
        "parameters": {
            "level0_root": str(level0_root),
            "iteration": args.iteration,
            "latest_fallback": args.latest_fallback,
            "sample_bits": args.sample_bits,
            "variants": variants,
            "null_source_variant": null_source_variant,
            "null_specs": source_specs,
            "mi_block_bits": args.mi_block_bits,
            "mi_lags": mi_lags,
            "mi_max_pairs": args.mi_max_pairs,
            "fft_window_bits": args.fft_window_bits,
            "fft_max_windows": args.fft_max_windows,
            "block_bits": args.block_bits,
            "pybdm_available": HAS_PYBDM,
            "pybdm_max_bits": args.pybdm_max_bits,
        },
        "metric_details": detail_payload,
        "feature_summary": feature_summary,
        "grouped_feature_summary": grouped_feature_summary,
        "rows": metric_rows,
        "margins": margin_rows,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(metrics_csv_path, metric_rows)
    write_csv(margins_csv_path, margin_rows)
    report_path.write_text(render_report(summary), encoding="utf-8")

    manifest = {
        "generated_at": now_utc(),
        "script": Path(__file__).name,
        "command": [sys.executable, *sys.argv],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "contract": {
            "mode": "read-only Level 0 snapshot analysis plus in-memory deterministic null reconstruction",
            "no_projective_tower_inputs": True,
            "no_upstream_regeneration": True,
        },
        "inputs": {
            "source_rows": source_rows,
            "null_source_variant": null_source_variant,
            "null_specs": source_specs,
        },
        "outputs": {
            "summary_json": str(summary_path),
            "metrics_csv": str(metrics_csv_path),
            "margins_csv": str(margins_csv_path),
            "report_md": str(report_path),
            "manifest_json": str(manifest_path),
        },
        "output_hashes": {
            "summary_json_sha256": sha256_file(summary_path),
            "metrics_csv_sha256": sha256_file(metrics_csv_path),
            "margins_csv_sha256": sha256_file(margins_csv_path),
            "report_md_sha256": sha256_file(report_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not args.quiet:
        print("")
        print(render_console(metric_rows, feature_summary, grouped_feature_summary))
        print("")
        print(f"Saved summary to: {summary_path}")
        print(f"Saved metrics CSV to: {metrics_csv_path}")
        print(f"Saved margins CSV to: {margins_csv_path}")
        print(f"Saved report to: {report_path}")
        print(f"Saved manifest to: {manifest_path}")
    return 0


def validate_args(args: argparse.Namespace) -> None:
    if args.sample_bits <= 0:
        raise SystemExit("--sample-bits must be positive")
    if args.mi_block_bits <= 0 or args.mi_block_bits > 16:
        raise SystemExit("--mi-block-bits must be in 1..16")
    if args.mi_max_pairs <= 0:
        raise SystemExit("--mi-max-pairs must be positive")
    if args.fft_window_bits <= 8:
        raise SystemExit("--fft-window-bits must be > 8")
    if args.fft_max_windows <= 0:
        raise SystemExit("--fft-max-windows must be positive")
    if args.block_bits <= 0 or args.block_bits > 24:
        raise SystemExit("--block-bits must be in 1..24")
    if args.pybdm_max_bits <= 0:
        raise SystemExit("--pybdm-max-bits must be positive")


def build_source_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    if args.include_shuffled:
        for seed in parse_int_csv(args.shuffled_seeds):
            specs.append({"null_model": "shuffled", "seed": seed})
    if args.include_matched_lz:
        for seed in parse_int_csv(args.matched_lz_seeds):
            specs.append({"null_model": "matched-lz", "seed": seed})
    if args.include_phase_matched_lz:
        for seed in parse_int_csv(args.phase_matched_lz_seeds):
            specs.append({"null_model": "phase-matched-lz", "seed": seed})
    return specs


def compute_all_metrics(
    bits: np.ndarray,
    *,
    args: argparse.Namespace,
    mi_lags: list[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics: dict[str, Any] = {
        "bit_density_one": float(bits.mean()),
        "bit_density_zero": float(1.0 - bits.mean()),
        "run_mean_bits": mean_run_length(bits),
        "zlib_ratio": zlib_ratio(bits),
    }

    mi_metrics, mi_details = long_range_mi_metrics(
        bits,
        block_bits=args.mi_block_bits,
        lag_bits_values=mi_lags,
        max_pairs=args.mi_max_pairs,
    )
    spectral_metrics, spectral_details = spectral_metrics_fft(
        bits,
        window_bits=args.fft_window_bits,
        max_windows=args.fft_max_windows,
    )
    block_metrics, block_details = block_decomposition_metrics(
        bits,
        block_bits=args.block_bits,
        pybdm_max_bits=args.pybdm_max_bits,
    )
    metrics.update(mi_metrics)
    metrics.update(spectral_metrics)
    metrics.update(block_metrics)
    details = {
        "long_range_mi": mi_details,
        "spectrum": spectral_details,
        "block_decomposition": block_details,
    }
    return metrics, details


def long_range_mi_metrics(
    bits: np.ndarray,
    *,
    block_bits: int,
    lag_bits_values: list[int],
    max_pairs: int,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    codes = encode_nonoverlap_blocks(bits, block_bits)
    details: list[dict[str, Any]] = []
    mi_values: list[float] = []
    nmi_values: list[float] = []

    for lag_bits in lag_bits_values:
        lag_blocks = max(1, int(round(lag_bits / block_bits)))
        if lag_blocks >= len(codes):
            details.append(
                {
                    "lag_bits": int(lag_bits),
                    "lag_blocks": int(lag_blocks),
                    "pairs_used": 0,
                    "mi_bits": None,
                    "nmi": None,
                    "status": "insufficient-length",
                }
            )
            continue
        x = codes[:-lag_blocks]
        y = codes[lag_blocks:]
        if len(x) > max_pairs:
            step = max(1, len(x) // max_pairs)
            x = x[::step][:max_pairs]
            y = y[::step][:max_pairs]
        mi, nmi = mutual_information_discrete(x, y)
        mi_values.append(mi)
        nmi_values.append(nmi)
        details.append(
            {
                "lag_bits": int(lag_bits),
                "lag_blocks": int(lag_blocks),
                "pairs_used": int(len(x)),
                "mi_bits": float(mi),
                "nmi": float(nmi),
                "status": "ok",
            }
        )

    tail = mi_values[-2:] if len(mi_values) >= 2 else mi_values
    metrics = {
        "long_range_mi_auc_mean_bits": float(np.mean(mi_values)) if mi_values else math.nan,
        "long_range_mi_tail_mean_bits": float(np.mean(tail)) if tail else math.nan,
        "long_range_mi_max_bits": float(np.max(mi_values)) if mi_values else math.nan,
        "long_range_mi_nmi_auc_mean": float(np.mean(nmi_values)) if nmi_values else math.nan,
    }
    return metrics, details


def spectral_metrics_fft(
    bits: np.ndarray,
    *,
    window_bits: int,
    max_windows: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    usable = (bits.size // window_bits) * window_bits
    if usable <= 0:
        return {
            "spectral_entropy_norm": math.nan,
            "spectral_low_frequency_power_fraction": math.nan,
            "spectral_peak_to_mean_power": math.nan,
            "spectral_loglog_slope": math.nan,
        }, {"status": "insufficient-length"}
    window_count = min(max_windows, usable // window_bits)
    arr = bits[: window_count * window_bits].astype(np.float64)
    arr = arr.reshape(window_count, window_bits)
    spectra = []
    slopes = []
    for segment in arr:
        centered = segment - segment.mean()
        power = np.abs(np.fft.rfft(centered)) ** 2
        power = power[1:]  # drop DC
        if not np.any(power > 0):
            continue
        spectra.append(power)
        upper = min(4096, len(power))
        if upper > 16:
            freqs = np.arange(1, upper + 1, dtype=np.float64)
            values = power[:upper] + 1e-300
            slope = np.polyfit(np.log2(freqs), np.log2(values), 1)[0]
            slopes.append(float(slope))
    if not spectra:
        return {
            "spectral_entropy_norm": math.nan,
            "spectral_low_frequency_power_fraction": math.nan,
            "spectral_peak_to_mean_power": math.nan,
            "spectral_loglog_slope": math.nan,
        }, {"status": "zero-power", "window_count": int(window_count)}
    mean_power = np.mean(np.vstack(spectra), axis=0)
    total = float(mean_power.sum())
    probabilities = mean_power / total
    entropy = -float(np.sum(probabilities * np.log2(probabilities + 1e-300)))
    entropy_norm = entropy / math.log2(len(probabilities)) if len(probabilities) > 1 else 0.0
    low_bins = max(1, int(math.ceil(len(probabilities) * 0.01)))
    metrics = {
        "spectral_entropy_norm": float(entropy_norm),
        "spectral_low_frequency_power_fraction": float(probabilities[:low_bins].sum()),
        "spectral_peak_to_mean_power": float(mean_power.max() / (mean_power.mean() + 1e-300)),
        "spectral_loglog_slope": float(np.mean(slopes)) if slopes else math.nan,
    }
    details = {
        "status": "ok",
        "window_bits": int(window_bits),
        "window_count": int(window_count),
        "low_frequency_bins": int(low_bins),
        "fft_bins_excluding_dc": int(len(probabilities)),
    }
    return metrics, details


def block_decomposition_metrics(
    bits: np.ndarray,
    *,
    block_bits: int,
    pybdm_max_bits: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    codes = encode_nonoverlap_blocks(bits, block_bits)
    counts = np.bincount(codes, minlength=1 << block_bits)
    nonzero = counts[counts > 0].astype(np.float64)
    total_blocks = float(nonzero.sum())
    probabilities = nonzero / total_blocks
    entropy = -float(np.sum(probabilities * np.log2(probabilities)))
    max_entropy = min(float(block_bits), math.log2(total_blocks)) if total_blocks > 1 else 1.0
    unique_blocks = int(nonzero.size)
    proxy_total = float(np.sum(block_bits + np.log2(nonzero)))
    metrics: dict[str, Any] = {
        "block_bits": int(block_bits),
        "block_count": int(total_blocks),
        "block_unique_count": unique_blocks,
        "block_unique_fraction": float(unique_blocks / total_blocks) if total_blocks else math.nan,
        "block_entropy_bits": entropy,
        "block_entropy_norm": float(entropy / max_entropy) if max_entropy else math.nan,
        "block_decomposition_proxy_total": proxy_total,
        "block_decomposition_proxy_per_bit": float(proxy_total / (total_blocks * block_bits))
        if total_blocks
        else math.nan,
        "pybdm_available": HAS_PYBDM,
        "pybdm_value": None,
        "pybdm_normalized": None,
    }
    details: dict[str, Any] = {
        "status": "ok",
        "proxy_definition": "sum_unique(block_bits + log2(count)) / total_bits",
        "pybdm_available": HAS_PYBDM,
    }
    if HAS_PYBDM and BDM is not None:
        length = min(pybdm_max_bits, int(bits.size))
        sample = np.ascontiguousarray(bits[:length].astype(np.int32))
        try:
            bdm = BDM(ndim=1)
            value = float(bdm.bdm(sample))
            metrics["pybdm_value"] = value
            metrics["pybdm_normalized"] = float(value / length) if length else math.nan
            details["pybdm_length_bits"] = int(length)
        except Exception as exc:  # pragma: no cover - optional path.
            details["pybdm_error"] = str(exc)
    return metrics, details


def encode_nonoverlap_blocks(bits: np.ndarray, block_bits: int) -> np.ndarray:
    usable = (bits.size // block_bits) * block_bits
    if usable == 0:
        return np.array([], dtype=np.uint32)
    matrix = bits[:usable].reshape(-1, block_bits).astype(np.uint32)
    weights = (1 << np.arange(block_bits - 1, -1, -1, dtype=np.uint32))
    return np.ascontiguousarray(matrix @ weights, dtype=np.uint32)


def mutual_information_discrete(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) == 0 or len(y) == 0:
        return math.nan, math.nan
    x_unique, x_inverse, x_counts = np.unique(x, return_inverse=True, return_counts=True)
    y_unique, y_inverse, y_counts = np.unique(y, return_inverse=True, return_counts=True)
    joint_codes = x_inverse.astype(np.int64) * len(y_unique) + y_inverse.astype(np.int64)
    _, joint_counts = np.unique(joint_codes, return_counts=True)
    n = float(len(x))
    px = x_counts.astype(np.float64) / n
    py = y_counts.astype(np.float64) / n
    pxy = joint_counts.astype(np.float64) / n
    hx = -float(np.sum(px * np.log2(px)))
    hy = -float(np.sum(py * np.log2(py)))
    hxy = -float(np.sum(pxy * np.log2(pxy)))
    mi = max(0.0, hx + hy - hxy)
    denom = min(hx, hy)
    nmi = mi / denom if denom > 0 else 0.0
    return float(mi), float(nmi)


def mean_run_length(bits: np.ndarray) -> float:
    if bits.size == 0:
        return math.nan
    changes = np.flatnonzero(bits[1:] != bits[:-1]) + 1
    boundaries = np.concatenate(([0], changes, [bits.size]))
    lengths = np.diff(boundaries)
    return float(lengths.mean()) if lengths.size else math.nan


def zlib_ratio(bits: np.ndarray) -> float:
    packed = np.packbits(bits.astype(np.uint8)).tobytes()
    if not packed:
        return math.nan
    compressed = zlib.compress(packed, level=6)
    return float(len(compressed) / len(packed))


def build_margin_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    b_rows = [row for row in rows if row["label"] == "B"]
    if len(b_rows) != 1:
        raise RuntimeError("Expected exactly one observed B row for margin readout.")
    b = b_rows[0]
    others = [row for row in rows if row["label"] != "B"]
    margin_rows: list[dict[str, Any]] = []
    feature_summary: list[dict[str, Any]] = []
    for feature in PRIMARY_FEATURES:
        b_value = as_float_or_nan(b.get(feature))
        other_values = [as_float_or_nan(row.get(feature)) for row in others]
        finite = [value for value in other_values if math.isfinite(value)]
        if not math.isfinite(b_value) or not finite:
            status = "not-evaluable"
            envelope_min = envelope_max = envelope_span = math.nan
            outside_margin = math.nan
        else:
            envelope_min = min(finite)
            envelope_max = max(finite)
            envelope_span = envelope_max - envelope_min
            if b_value < envelope_min:
                outside_margin = envelope_min - b_value
                status = "B-below-control-envelope"
            elif b_value > envelope_max:
                outside_margin = b_value - envelope_max
                status = "B-above-control-envelope"
            else:
                outside_margin = 0.0
                status = "B-inside-control-envelope"
        tolerance = max(1e-9, 0.05 * (abs(envelope_span) if math.isfinite(envelope_span) else 0.0))
        separates = bool(math.isfinite(outside_margin) and outside_margin > tolerance)
        feature_summary.append(
            {
                "feature": feature,
                "B_value": b_value,
                "control_envelope_min": envelope_min,
                "control_envelope_max": envelope_max,
                "control_envelope_span": envelope_span,
                "outside_margin": outside_margin,
                "separates_B_from_all_controls": separates,
                "status": status,
                "separation_rule": "B outside non-B envelope by > max(1e-9, 5% envelope span)",
            }
        )
        for row in others:
            value = as_float_or_nan(row.get(feature))
            margin_rows.append(
                {
                    "feature": feature,
                    "B_value": b_value,
                    "comparison_label": row["label"],
                    "comparison_kind": row["kind"],
                    "comparison_value": value,
                    "B_minus_comparison": b_value - value
                    if math.isfinite(b_value) and math.isfinite(value)
                    else math.nan,
                    "abs_margin": abs(b_value - value)
                    if math.isfinite(b_value) and math.isfinite(value)
                    else math.nan,
                }
            )
    return margin_rows, feature_summary


def build_group_feature_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    b_rows = [row for row in rows if row["label"] == "B"]
    if len(b_rows) != 1:
        raise RuntimeError("Expected exactly one observed B row for grouped readout.")
    b = b_rows[0]
    groups = {
        "observed_controls": [
            row for row in rows if row["label"] != "B" and row.get("kind") == "observed"
        ],
        "lz_family_nulls": [
            row
            for row in rows
            if row.get("null_model") in {"matched-lz", "phase-matched-lz"}
        ],
        "all_nulls": [row for row in rows if row.get("kind") == "null"],
        "all_non_B": [row for row in rows if row["label"] != "B"],
    }
    summaries: list[dict[str, Any]] = []
    for group_name, group_rows in groups.items():
        for feature in PRIMARY_FEATURES:
            b_value = as_float_or_nan(b.get(feature))
            values = [as_float_or_nan(row.get(feature)) for row in group_rows]
            finite = [value for value in values if math.isfinite(value)]
            if not math.isfinite(b_value) or not finite:
                envelope_min = envelope_max = envelope_span = outside_margin = math.nan
                status = "not-evaluable"
            else:
                envelope_min = min(finite)
                envelope_max = max(finite)
                envelope_span = envelope_max - envelope_min
                if b_value < envelope_min:
                    outside_margin = envelope_min - b_value
                    status = "B-below-group-envelope"
                elif b_value > envelope_max:
                    outside_margin = b_value - envelope_max
                    status = "B-above-group-envelope"
                else:
                    outside_margin = 0.0
                    status = "B-inside-group-envelope"
            tolerance = max(1e-9, 0.05 * (abs(envelope_span) if math.isfinite(envelope_span) else 0.0))
            summaries.append(
                {
                    "group": group_name,
                    "feature": feature,
                    "B_value": b_value,
                    "group_size": len(group_rows),
                    "group_envelope_min": envelope_min,
                    "group_envelope_max": envelope_max,
                    "group_envelope_span": envelope_span,
                    "outside_margin": outside_margin,
                    "separates_B_from_group": bool(
                        math.isfinite(outside_margin) and outside_margin > tolerance
                    ),
                    "status": status,
                }
            )
    return summaries


def render_console(
    rows: list[dict[str, Any]],
    feature_summary: list[dict[str, Any]],
    grouped_feature_summary: list[dict[str, Any]],
) -> str:
    lines = [
        "Raw-sequence orthogonal metrics",
        "-" * 112,
        (
            f"{'label':<34} {'kind':<9} {'MI_tail':>10} {'MI_auc':>10} {'spec_H':>10} "
            f"{'lowF':>10} {'blk_H':>10} {'BDMproxy':>10} {'zlib':>10}"
        ),
    ]
    for row in rows:
        lines.append(
            f"{row['label']:<34} {row['kind']:<9} "
            f"{fmt(row.get('long_range_mi_tail_mean_bits')):>10} "
            f"{fmt(row.get('long_range_mi_auc_mean_bits')):>10} "
            f"{fmt(row.get('spectral_entropy_norm')):>10} "
            f"{fmt(row.get('spectral_low_frequency_power_fraction')):>10} "
            f"{fmt(row.get('block_entropy_norm')):>10} "
            f"{fmt(row.get('block_decomposition_proxy_per_bit')):>10} "
            f"{fmt(row.get('zlib_ratio')):>10}"
        )
    lines.extend(["", "Feature-envelope readout", "-" * 112])
    for item in feature_summary:
        flag = "separates" if item["separates_B_from_all_controls"] else "no"
        lines.append(
            f"{item['feature']:<42} {flag:<10} "
            f"B={fmt(item['B_value'])} envelope=[{fmt(item['control_envelope_min'])}, "
            f"{fmt(item['control_envelope_max'])}] status={item['status']}"
        )
    lines.extend(["", "Grouped envelope readout", "-" * 112])
    for group in ("observed_controls", "lz_family_nulls", "all_nulls"):
        separating = [
            item["feature"]
            for item in grouped_feature_summary
            if item["group"] == group and item["separates_B_from_group"]
        ]
        lines.append(f"{group:<24} separating_features={', '.join(separating) if separating else '-'}")
    return "\n".join(lines)


def render_report(summary: dict[str, Any]) -> str:
    params = summary["parameters"]
    null_specs = ", ".join(
        f"{item['null_model']} seed {item['seed']}" for item in params["null_specs"]
    )
    lines = [
        "# Raw-Sequence Orthogonal Metrics",
        "",
        "This is a read-only diagnostic over frozen Level 0 observable bits. It does not use Phase 1 pattern spaces, prefix/suffix projections, fibers, or any projective-tower observable.",
        "",
        "## Scope",
        "",
        f"- Variants: {', '.join(params['variants'])}",
        f"- Observable bits per source: {params['sample_bits']:,}",
        f"- Null source: {params['null_source_variant']}",
        f"- Null specs: {null_specs}",
        f"- PyBDM available: {params['pybdm_available']}",
        "",
        "## Metrics",
        "",
        "- Long-range MI: mutual information between non-overlapping raw bit blocks separated by fixed bit lags.",
        "- Spectrum: averaged FFT power profile over raw observable bits after local mean subtraction.",
        "- Block decomposition: deterministic repeated-block complexity proxy; true PyBDM is reported only when the library is installed.",
        "- zlib ratio: auxiliary compression context, not a projective observable.",
        "",
        "## Results",
        "",
        "| Label | Kind | MI tail | MI AUC | Spectral H | Low-freq power | Block H | BDM proxy | zlib |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        lines.append(
            f"| {row['label']} | {row['kind']} | "
            f"{fmt(row.get('long_range_mi_tail_mean_bits'))} | "
            f"{fmt(row.get('long_range_mi_auc_mean_bits'))} | "
            f"{fmt(row.get('spectral_entropy_norm'))} | "
            f"{fmt(row.get('spectral_low_frequency_power_fraction'))} | "
            f"{fmt(row.get('block_entropy_norm'))} | "
            f"{fmt(row.get('block_decomposition_proxy_per_bit'))} | "
            f"{fmt(row.get('zlib_ratio'))} |"
        )
    lines.extend(
        [
            "",
            "## Envelope Readout",
            "",
            "The envelope rule is intentionally simple: B separates on a scalar feature only when its value lies outside the non-B envelope by more than `max(1e-9, 5% of envelope span)`. Direction is not pre-assumed.",
            "",
            "| Feature | B value | Non-B envelope | Verdict |",
            "|---|---:|---:|---|",
        ]
    )
    for item in summary["feature_summary"]:
        verdict = "separates" if item["separates_B_from_all_controls"] else "does not separate"
        lines.append(
            f"| {item['feature']} | {fmt(item['B_value'])} | "
            f"[{fmt(item['control_envelope_min'])}, {fmt(item['control_envelope_max'])}] | "
            f"{verdict} ({item['status']}) |"
        )
    lines.extend(
        [
            "",
            "## Grouped Envelope Readout",
            "",
            "This separates four diagnostic envelopes: observed structural controls, LZ-family nulls, all nulls, and all non-B sources. The grouping prevents a crystalline control such as M/N from silently swallowing a more local B-vs-null distinction.",
            "",
            "| Group | Feature | B value | Group envelope | Verdict |",
            "|---|---|---:|---:|---|",
        ]
    )
    for item in summary["grouped_feature_summary"]:
        verdict = "separates" if item["separates_B_from_group"] else "does not separate"
        lines.append(
            f"| {item['group']} | {item['feature']} | {fmt(item['B_value'])} | "
            f"[{fmt(item['group_envelope_min'])}, {fmt(item['group_envelope_max'])}] | "
            f"{verdict} ({item['status']}) |"
        )
    lines.extend(
        [
            "",
            "## Interpretive Guard",
            "",
            "A positive scalar-envelope separation here is not a new Gate claim. It only answers whether a raw-sequence, non-projective diagnostic can still distinguish observed B from the chosen controls/nulls. A negative or mixed result is equally valid.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: scalar_for_csv(row.get(key)) for key in keys})


def scalar_for_csv(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def parse_csv(raw: str) -> list[str]:
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


def parse_int_csv(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def resolve_project_path(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_variant_struct_path(
    *,
    variant: str,
    iteration: int,
    level0_root: Path,
    latest_fallback: bool,
) -> tuple[Path, int]:
    try:
        return (
            resolve_struct_path(
                variant=variant,
                iteration=iteration,
                level0_root=str(level0_root),
            ),
            iteration,
        )
    except FileNotFoundError:
        if not latest_fallback:
            raise
        var_dir = level0_root / f"var_{variant.upper()}"
        latest = detect_latest_iteration(var_dir)
        return (
            resolve_struct_path(
                variant=variant,
                iteration=latest,
                level0_root=str(level0_root),
            ),
            latest,
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def format_slug_int(value: int) -> str:
    if value % 1_000_000 == 0:
        return f"{value // 1_000_000}M"
    if value % 1_000 == 0:
        return f"{value // 1_000}K"
    return str(value)


def phase_print(title: str, detail: str = "", *, quiet: bool = False) -> None:
    if quiet:
        return
    print(f"[Phase] {title}", flush=True)
    if detail:
        print(f"        {detail}", flush=True)


def as_float_or_nan(value: Any) -> float:
    try:
        if value is None:
            return math.nan
        return float(value)
    except Exception:
        return math.nan


def fmt(value: Any) -> str:
    value = as_float_or_nan(value)
    if not math.isfinite(value):
        return "-"
    if abs(value) >= 100:
        return f"{value:.3f}"
    if abs(value) >= 1:
        return f"{value:.5f}"
    return f"{value:.6f}"


if __name__ == "__main__":
    raise SystemExit(main())
