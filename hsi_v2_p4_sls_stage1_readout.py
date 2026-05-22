#!/usr/bin/env python3
"""P4-SLS Stage 1 readout for D-0148.

This script executes only the probes frozen in
``prereg_P4_subshift_like_signature_v0.3.1.md``. It consumes the completed
Stage 0 inventory/cache artifacts and emits numeric tables plus a manifest.
It creates no figures and introduces no additional probes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from hsi_v2_phase2_transport_defect_strict import phase_print
from hsi_v2_p4_sls_stage0_inventory import (
    BLOCK_ENTROPY_POLICY,
    CONTRACT_SHA256,
    LAGS,
    NULL_FAMILIES,
    WINDOW_BITS,
    WINDOWS,
    extract_seed_inventory,
    load_source_manifests,
    select_null_runs,
    select_observed_b_run,
)
from v2.common.cli import resolve_dir
from v2.common.io import load_observable_prefix_bits
from v2.phase1.tower import rolling_codes_uint64_max64
from v2.phase2.null_pressure import discover_phase1_runs_recursive
from v2.phase2.return_lag import load_run_source_bits

try:
    from numba import njit
except Exception:  # pragma: no cover
    def njit(*args, **kwargs):  # type: ignore
        def decorator(func):
            return func

        return decorator


DEFAULT_OUTPUT_DIR = "results/hsi_v2/phase4/p4_sls_subshift_like_signature/stage1_readout"
DEFAULT_PHASE1_DIR = "results/hsi_v2/phase1_high_scales"
DEFAULT_STAGE0_DIR = (
    "results/hsi_v2/phase4/p4_sls_subshift_like_signature/stage0_inventory/"
    "p4-sls-stage0-inventory__windows-21__20260521T171944"
)
DEFAULT_CONTRACT = "../prereg_P4_subshift_like_signature_v0.3.1.md"
DEFAULT_P4_04_RUN = (
    "results/hsi_v2/phase4/p4_04_forward_regime_forecast/"
    "phase4-p4-04-forward-regime-forecast__bands-3__lags-6__20260507T200920"
)
DEFAULT_P4_05_RUN = (
    "results/hsi_v2/phase4/p4_05_horizon_bracket_forecast/"
    "phase4-p4-05-horizon-bracket-forecast__bands-4__lags-6__20260507T225929"
)
DEFAULT_BMINI_RUN = (
    "results/hsi_v2/phase5_cross_arc_horizon/bmini_v0_2/"
    "phase5-bmini-cross-arc-horizon__arcs-3__20260508T153520"
)

BLOCK_N_GRID = (8, 10, 12, 14, 16, 18, 20, 22, 24)
BLOCK_PRIMARY_N = (22, 24)
MI_BLOCK_BITS = 6
RECURRENCE_K_GRID = (12, 16, 20)
SUBWINDOW_BITS = 1_000_000
SUBWINDOW_COUNT = WINDOW_BITS // SUBWINDOW_BITS
BOOTSTRAP_SAMPLES = 5000
BOOTSTRAP_SEED = 20260522
PROBE1_EFFECT_FLOOR = 0.01
PROBE2_EFFECT_FLOOR = 0.002
PROBE3_EFFECT_FLOOR = 0.10
PROBE3_TOP_KMERS = 4096


@njit(cache=True)
def recurrence_gap_accumulators(bits: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    size = 1 << k
    counts = np.zeros(size, dtype=np.int64)
    gap_counts = np.zeros(size, dtype=np.int64)
    sum_gaps = np.zeros(size, dtype=np.float64)
    sum_sq_gaps = np.zeros(size, dtype=np.float64)
    last_pos = np.full(size, -1, dtype=np.int64)

    n = bits.size
    if n < k:
        return counts, gap_counts, sum_gaps, sum_sq_gaps

    code = 0
    for idx in range(k):
        code = (code << 1) | int(bits[idx])
    mask = (1 << k) - 1

    for pos in range(n - k + 1):
        if pos > 0:
            code = ((code << 1) & mask) | int(bits[pos + k - 1])
        counts[code] += 1
        previous = last_pos[code]
        if previous >= 0:
            gap = float(pos - previous)
            gap_counts[code] += 1
            sum_gaps[code] += gap
            sum_sq_gaps[code] += gap * gap
        last_pos[code] = pos
    return counts, gap_counts, sum_gaps, sum_sq_gaps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="D-0148 P4-SLS Stage 1 readout.")
    parser.add_argument("--stage0-dir", default=DEFAULT_STAGE0_DIR)
    parser.add_argument("--contract", default=DEFAULT_CONTRACT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--phase1-dir", default=DEFAULT_PHASE1_DIR)
    parser.add_argument("--p4-04-run", default=DEFAULT_P4_04_RUN)
    parser.add_argument("--p4-05-run", default=DEFAULT_P4_05_RUN)
    parser.add_argument("--bmini-run", default=DEFAULT_BMINI_RUN)
    parser.add_argument("--debug-limit-windows", type=int, default=0)
    parser.add_argument("--debug-limit-null-sources", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.perf_counter()

    stage0_dir = resolve_dir(args.stage0_dir, anchor_file=__file__)
    contract = resolve_dir(args.contract, anchor_file=__file__)
    output_root = resolve_dir(args.output_dir, anchor_file=__file__)
    phase1_dir = resolve_dir(args.phase1_dir, anchor_file=__file__)
    p4_04_run = resolve_dir(args.p4_04_run, anchor_file=__file__)
    p4_05_run = resolve_dir(args.p4_05_run, anchor_file=__file__)
    bmini_run = resolve_dir(args.bmini_run, anchor_file=__file__)

    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_dir = output_root / f"p4-sls-stage1-readout__windows-21__{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    phase_print(
        "Preparing D-0148 P4-SLS Stage 1",
        "frozen probes only | no figures | no extra null families",
        quiet=args.quiet,
    )

    stage0_summary = load_json(stage0_dir / "summary.json")
    stage0_manifest = load_json(stage0_dir / "manifest.json")
    validate_stage0(stage0_summary, stage0_manifest)

    windows = list(WINDOWS)
    if args.debug_limit_windows:
        windows = windows[: args.debug_limit_windows]

    source_manifests = load_source_manifests(p4_04_run, p4_05_run, bmini_run)
    seed_inventory = extract_seed_inventory(source_manifests)
    if not seed_inventory["seed_sets_consistent"]:
        raise RuntimeError("Stage 1 requires a consistent manifest-derived seed list.")
    seeds = list(seed_inventory["seeds"])

    phase1_runs = discover_phase1_runs_recursive(phase1_dir)
    observed_run = select_observed_b_run(phase1_runs)
    if observed_run is None:
        raise RuntimeError("Could not locate the observed B Phase 1 source run.")
    null_run_index = select_null_runs(phase1_runs, seeds)

    source_specs = [{"label": "B", "kind": "observed", "family": "B", "seed": None, "run": observed_run}]
    for family in NULL_FAMILIES:
        for seed in seeds:
            run = null_run_index.get((family, seed))
            if run is None:
                raise RuntimeError(f"Missing Phase 1 null source for {family} seed {seed}.")
            source_specs.append(
                {
                    "label": f"{family} seed {seed}",
                    "kind": "null_surrogate",
                    "family": family,
                    "seed": seed,
                    "run": run,
                }
            )
    if args.debug_limit_null_sources:
        source_specs = [source_specs[0], *source_specs[1 : 1 + args.debug_limit_null_sources]]

    required_bits = int(stage0_summary["source_requirements"]["max_required_prefix_bits"])
    source_metric_payload: dict[str, dict[str, Any]] = {}
    source_rows: list[dict[str, Any]] = []
    for index, spec in enumerate(source_specs, start=1):
        phase_print(
            f"Loading source {spec['label']}",
            f"{index}/{len(source_specs)} | required_prefix={required_bits:,} bits",
            quiet=args.quiet,
        )
        bits, source_info = load_source_bits(spec, required_bits=required_bits, quiet=args.quiet)
        source_rows.append(
            {
                "source_label": spec["label"],
                "source_kind": spec["kind"],
                "family": spec["family"],
                "seed": "" if spec["seed"] is None else spec["seed"],
                "loaded_bits": int(bits.size),
                "source_cache_status": source_info.get("source_cache_status", "not-applicable"),
                "source_cache_path": source_info.get("source_cache_path", ""),
            }
        )
        payload = compute_source_metrics(
            bits=bits,
            spec=spec,
            windows=windows,
            workers=max(1, int(args.workers)),
            quiet=args.quiet,
        )
        source_metric_payload[spec["label"]] = payload

    window_readout, probe_detail = build_window_readout(source_metric_payload, windows)
    region_readout = build_region_readout(window_readout)
    overall = build_overall_readout(region_readout)

    summary = {
        "run_id": run_dir.name,
        "timestamp_utc": timestamp,
        "status": "completed",
        "stage1_verdict": overall["outcome"],
        "authorization_note": (
            "Stage 1 executed as cache-ready continuation of D-0148/v0.3.1 after "
            "Stage 0 returned inputs-derivable and Iban authorized complete-prefix cache continuation."
        ),
        "contract": {
            "path": str(contract.resolve()),
            "sha256": sha256_file(contract),
            "expected_sha256": CONTRACT_SHA256,
        },
        "stage0": {
            "path": str(stage0_dir.resolve()),
            "manifest_sha256": sha256_file(stage0_dir / "manifest.json"),
            "summary_sha256": sha256_file(stage0_dir / "summary.json"),
            "verdict": stage0_summary.get("stage0_verdict"),
        },
        "block_entropy_policy": BLOCK_ENTROPY_POLICY,
        "parameters": {
            "window_bits": WINDOW_BITS,
            "subwindow_bits": SUBWINDOW_BITS,
            "subwindow_count": SUBWINDOW_COUNT,
            "lags": LAGS,
            "null_families": NULL_FAMILIES,
            "seeds": seeds,
            "block_n_grid": BLOCK_N_GRID,
            "block_primary_n": BLOCK_PRIMARY_N,
            "mi_block_bits": MI_BLOCK_BITS,
            "recurrence_k_grid": RECURRENCE_K_GRID,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "probe1_effect_floor": PROBE1_EFFECT_FLOOR,
            "probe2_effect_floor": PROBE2_EFFECT_FLOOR,
            "probe3_effect_floor": PROBE3_EFFECT_FLOOR,
            "probe3_top_kmers": PROBE3_TOP_KMERS,
            "debug_limit_windows": args.debug_limit_windows,
            "debug_limit_null_sources": args.debug_limit_null_sources,
            "workers": args.workers,
        },
        "overall_readout": overall,
        "region_readout": region_readout,
        "counts": {
            "windows": len(windows),
            "sources": len(source_specs),
            "null_sources": len(source_specs) - 1,
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }

    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "source_metric_details.json", source_metric_payload)
    write_json(run_dir / "probe_detail.json", probe_detail)
    write_csv(run_dir / "source_inventory.csv", source_rows)
    write_csv(run_dir / "window_readout.csv", window_readout)
    write_csv(run_dir / "region_readout.csv", region_readout)
    write_csv(run_dir / "seed_inventory.csv", seed_inventory["rows"])
    write_report(run_dir / "report.md", summary, region_readout, window_readout)
    manifest = build_manifest(run_dir, summary, source_manifests)
    write_json(run_dir / "manifest.json", manifest)

    print(render_console_summary(summary))
    print_saved_paths(run_dir)
    return 0


def validate_stage0(summary: dict[str, Any], manifest: dict[str, Any]) -> None:
    if summary.get("stage0_verdict") != "inputs-derivable":
        raise RuntimeError(f"Stage 0 is not inputs-derivable: {summary.get('stage0_verdict')}")
    contract = manifest.get("contract", {})
    if contract.get("sha256") != CONTRACT_SHA256:
        raise RuntimeError("Stage 0 manifest does not match the frozen P4-SLS contract SHA.")
    if summary.get("block_entropy_policy") != BLOCK_ENTROPY_POLICY:
        raise RuntimeError("Stage 0 block-entropy policy mismatch.")


def load_source_bits(spec: dict[str, Any], *, required_bits: int, quiet: bool) -> tuple[np.ndarray, dict[str, Any]]:
    run = spec["run"]
    config = run["dataset"]["config"]
    if spec["kind"] == "observed":
        struct_path = Path(config["input_struct_path"]).resolve()
        bits = load_observable_prefix_bits(struct_path, required_bits)
        return bits, {
            "source_kind": "observed",
            "source_struct_path": str(struct_path),
            "loaded_observable_bits": required_bits,
        }
    return load_run_source_bits(
        run["dataset"],
        required_bits=required_bits,
        show_progress=not quiet,
        progress_label=f"P4-SLS Stage1 {spec['label']}",
    )


def compute_source_metrics(
    *,
    bits: np.ndarray,
    spec: dict[str, Any],
    windows: list[dict[str, Any]],
    workers: int,
    quiet: bool,
) -> dict[str, Any]:
    source_payload: dict[str, Any] = {
        "source_label": spec["label"],
        "family": spec["family"],
        "seed": spec["seed"],
        "windows": {},
    }
    if workers <= 1:
        items = [
            compute_source_window_metrics(bits, spec, window, index, len(windows), quiet)
            for index, window in enumerate(windows, start=1)
        ]
    else:
        items = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(compute_source_window_metrics, bits, spec, window, index, len(windows), quiet)
                for index, window in enumerate(windows, start=1)
            ]
            for future in as_completed(futures):
                items.append(future.result())
        items.sort(key=lambda item: item["window_order"])

    for item in items:
        source_payload["windows"][item["band"]] = item["metrics"]
    return source_payload


def compute_source_window_metrics(
    bits: np.ndarray,
    spec: dict[str, Any],
    window: dict[str, Any],
    index: int,
    total_windows: int,
    quiet: bool,
) -> dict[str, Any]:
    band = window["band"]
    phase_print(
        f"Stage 1 metrics {spec['label']}",
        f"{index}/{total_windows} | {window['region']} {band}",
        quiet=quiet,
    )
    start = int(window["start_bits"])
    segment = as_uint8_view(bits[start : start + WINDOW_BITS])
    return {
        "window_order": index,
        "band": band,
        "metrics": {
            "region": window["region"],
            "role": window["role"],
            "start_bits": start,
            "probe1": compute_probe1_block_complexity(segment),
            "probe2": compute_probe2_lag_mi(bits, start),
            "probe3": compute_probe3_recurrence(segment),
        },
    }


def compute_probe1_block_complexity(bits: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for n in BLOCK_N_GRID:
        codes = rolling_codes_uint64_max64(bits, int(n))
        p_full = int(np.unique(codes).size)
        max_possible = min(1 << int(n), int(codes.size))
        sub_h = []
        sub_p = []
        if n in BLOCK_PRIMARY_N:
            for sub_start in range(0, WINDOW_BITS, SUBWINDOW_BITS):
                sub_len = SUBWINDOW_BITS - int(n) + 1
                sub_codes = codes[sub_start : sub_start + sub_len]
                p_sub = int(np.unique(sub_codes).size)
                sub_p.append(p_sub)
                sub_h.append(log2_safe(p_sub) / float(n))
        result[str(n)] = {
            "p_full": p_full,
            "h_full": log2_safe(p_full) / float(n),
            "max_possible": max_possible,
            "saturated_exact": p_full >= max_possible,
            "sub_p": sub_p,
            "sub_h": sub_h,
        }
    return result


def compute_probe2_lag_mi(bits: np.ndarray, start: int) -> dict[str, Any]:
    by_lag: dict[str, Any] = {}
    full_values = []
    sub_values_by_lag: dict[int, list[float]] = {}
    base = as_uint8_view(bits[start : start + WINDOW_BITS])
    x_full = encode_nonoverlap_blocks(base, MI_BLOCK_BITS)
    for lag in LAGS:
        lagged = as_uint8_view(bits[start + int(lag) : start + int(lag) + WINDOW_BITS])
        y_full = encode_nonoverlap_blocks(lagged, MI_BLOCK_BITS)
        _mi, nmi = mutual_information_discrete(x_full, y_full)
        full_values.append((int(lag), nmi))
        sub_values = []
        for sub_offset in range(0, WINDOW_BITS, SUBWINDOW_BITS):
            x_sub = encode_nonoverlap_blocks(
                as_uint8_view(bits[start + sub_offset : start + sub_offset + SUBWINDOW_BITS]),
                MI_BLOCK_BITS,
            )
            y_sub = encode_nonoverlap_blocks(
                as_uint8_view(bits[start + int(lag) + sub_offset : start + int(lag) + sub_offset + SUBWINDOW_BITS]),
                MI_BLOCK_BITS,
            )
            _sub_mi, sub_nmi = mutual_information_discrete(x_sub, y_sub)
            sub_values.append(sub_nmi)
        sub_values_by_lag[int(lag)] = sub_values
        by_lag[str(lag)] = {"nmi_full": nmi, "nmi_subwindows": sub_values}
    best_lag, best_nmi = max(full_values, key=lambda item: item[1])
    sub_max = [
        max(sub_values_by_lag[int(lag)][sub_index] for lag in LAGS)
        for sub_index in range(SUBWINDOW_COUNT)
    ]
    return {
        "by_lag": by_lag,
        "max_nmi_full": float(best_nmi),
        "max_lag_bits": int(best_lag),
        "max_nmi_subwindows": [float(value) for value in sub_max],
    }


def compute_probe3_recurrence(bits: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for k in RECURRENCE_K_GRID:
        full = recurrence_cv_for_bits(bits, int(k))
        sub_values = []
        for sub_start in range(0, WINDOW_BITS, SUBWINDOW_BITS):
            sub_bits = as_uint8_view(bits[sub_start : sub_start + SUBWINDOW_BITS])
            sub_values.append(recurrence_cv_for_bits(sub_bits, int(k)))
        result[str(k)] = {
            **full,
            "sub_cv_gap_median": [item["cv_gap_median"] for item in sub_values],
            "sub_valid_kmers": [item["valid_kmers"] for item in sub_values],
            "sub_status": [item["status"] for item in sub_values],
        }
    return result


def recurrence_cv_for_bits(bits: np.ndarray, k: int) -> dict[str, Any]:
    counts, gap_counts, sum_gaps, sum_sq_gaps = recurrence_gap_accumulators(bits, k)
    valid = np.flatnonzero((counts >= 4) & (gap_counts > 1))
    if valid.size == 0:
        return {
            "status": "power-insufficient",
            "valid_kmers": 0,
            "selected_kmers": 0,
            "cv_gap_median": math.nan,
            "h_gap": math.nan,
        }
    if valid.size > PROBE3_TOP_KMERS:
        valid_counts = counts[valid]
        selected_local = np.argpartition(valid_counts, -PROBE3_TOP_KMERS)[-PROBE3_TOP_KMERS:]
        selected = valid[selected_local]
    else:
        selected = valid
    gc = gap_counts[selected].astype(np.float64)
    means = sum_gaps[selected] / gc
    variances = np.maximum((sum_sq_gaps[selected] / gc) - (means * means), 0.0)
    cvs = np.sqrt(variances) / np.maximum(means, 1e-300)
    weights = counts[selected].astype(np.float64)
    probabilities = weights / max(float(weights.sum()), 1.0)
    h_gap = -float(np.sum(probabilities * np.log2(probabilities + 1e-300)))
    return {
        "status": "ok",
        "valid_kmers": int(valid.size),
        "selected_kmers": int(selected.size),
        "cv_gap_median": float(np.median(cvs)),
        "h_gap": h_gap,
    }


def build_window_readout(
    payload: dict[str, dict[str, Any]],
    windows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    b_payload = payload["B"]
    null_labels = [label for label in payload if label != "B"]
    rows = []
    details: dict[str, Any] = {}
    for window in windows:
        band = window["band"]
        b_window = b_payload["windows"][band]
        null_windows = [payload[label]["windows"][band] for label in null_labels]
        p1 = evaluate_probe1_window(b_window, null_windows, band)
        p2 = evaluate_probe2_window(b_window, null_windows, band)
        p3 = evaluate_probe3_window(b_window, null_windows, band)
        rows.append(
            {
                "region": window["region"],
                "role": window["role"],
                "band": band,
                "start_bits": window["start_bits"],
                "probe1_positive": p1["positive"],
                "probe1_status": p1["status"],
                "probe1_gap22": p1["gap_full"]["22"],
                "probe1_gap24": p1["gap_full"]["24"],
                "probe1_ci22_low": p1["ci"]["22"]["low"],
                "probe1_ci24_low": p1["ci"]["24"]["low"],
                "probe2_positive": p2["positive"],
                "probe2_status": p2["status"],
                "probe2_gap_mi": p2["gap_full"],
                "probe2_ci_low": p2["ci"]["low"],
                "probe2_b_best_lag": p2["b_best_lag"],
                "probe3_positive": p3["positive"],
                "probe3_status": p3["status"],
                "probe3_positive_k_count": p3["positive_k_count"],
                "probe3_gap12": p3["gap_full"]["12"],
                "probe3_gap16": p3["gap_full"]["16"],
                "probe3_gap20": p3["gap_full"]["20"],
                "probe3_ci_low": p3["ci"]["low"],
            }
        )
        details[band] = {"probe1": p1, "probe2": p2, "probe3": p3}
    return rows, details


def evaluate_probe1_window(b_window: dict[str, Any], null_windows: list[dict[str, Any]], band: str) -> dict[str, Any]:
    gap_full = {}
    ci = {}
    median_bootstrap = {}
    for n in BLOCK_PRIMARY_N:
        key = str(n)
        b_metric = b_window["probe1"][key]
        null_metrics = [item["probe1"][key] for item in null_windows]
        null_min = min(float(item["h_full"]) for item in null_metrics)
        gap_full[key] = float(null_min - float(b_metric["h_full"]))
        sub_gaps = []
        for index in range(SUBWINDOW_COUNT):
            sub_null_min = min(float(item["sub_h"][index]) for item in null_metrics)
            sub_gaps.append(float(sub_null_min - float(b_metric["sub_h"][index])))
        median, low, high = bootstrap_median_ci(sub_gaps, seed=stable_seed("probe1", band, key))
        median_bootstrap[key] = median
        ci[key] = {"low": low, "high": high}
    status = "ok"
    if any(not math.isfinite(value) for value in gap_full.values()):
        status = "power-insufficient"
    positive = (
        status == "ok"
        and gap_full["22"] > 0
        and gap_full["24"] > 0
        and median_bootstrap["22"] >= PROBE1_EFFECT_FLOOR
        and median_bootstrap["24"] >= PROBE1_EFFECT_FLOOR
        and (
            (ci["22"]["low"] > 0 and ci["24"]["low"] >= 0)
            or (ci["24"]["low"] > 0 and ci["22"]["low"] >= 0)
        )
    )
    return {
        "status": status,
        "positive": bool(positive),
        "gap_full": gap_full,
        "median_bootstrap": median_bootstrap,
        "ci": ci,
    }


def evaluate_probe2_window(b_window: dict[str, Any], null_windows: list[dict[str, Any]], band: str) -> dict[str, Any]:
    b_metric = b_window["probe2"]
    null_max = max(float(item["probe2"]["max_nmi_full"]) for item in null_windows)
    gap_full = float(float(b_metric["max_nmi_full"]) - null_max)
    sub_gaps = []
    for index in range(SUBWINDOW_COUNT):
        sub_null_max = max(float(item["probe2"]["max_nmi_subwindows"][index]) for item in null_windows)
        sub_gaps.append(float(float(b_metric["max_nmi_subwindows"][index]) - sub_null_max))
    median, low, high = bootstrap_median_ci(sub_gaps, seed=stable_seed("probe2", band))
    status = "ok" if math.isfinite(gap_full) else "power-insufficient"
    positive = status == "ok" and gap_full >= PROBE2_EFFECT_FLOOR and low > 0
    return {
        "status": status,
        "positive": bool(positive),
        "gap_full": gap_full,
        "median_bootstrap": median,
        "ci": {"low": low, "high": high},
        "b_best_lag": b_metric["max_lag_bits"],
    }


def evaluate_probe3_window(b_window: dict[str, Any], null_windows: list[dict[str, Any]], band: str) -> dict[str, Any]:
    gap_full = {}
    sub_gap_by_k = []
    positive_k_count = 0
    valid_k_count = 0
    for k in RECURRENCE_K_GRID:
        key = str(k)
        b_metric = b_window["probe3"][key]
        null_values = [
            float(item["probe3"][key]["cv_gap_median"])
            for item in null_windows
            if math.isfinite(float(item["probe3"][key]["cv_gap_median"]))
        ]
        if b_metric["status"] != "ok" or not null_values:
            gap_full[key] = math.nan
            continue
        valid_k_count += 1
        null_min = min(null_values)
        gap = float(null_min - float(b_metric["cv_gap_median"]))
        gap_full[key] = gap
        if gap >= PROBE3_EFFECT_FLOOR:
            positive_k_count += 1
        sub_gaps = []
        for index in range(SUBWINDOW_COUNT):
            sub_null_values = [
                float(item["probe3"][key]["sub_cv_gap_median"][index])
                for item in null_windows
                if math.isfinite(float(item["probe3"][key]["sub_cv_gap_median"][index]))
            ]
            sub_b = float(b_metric["sub_cv_gap_median"][index])
            if not sub_null_values or not math.isfinite(sub_b):
                continue
            sub_gaps.append(float(min(sub_null_values) - sub_b))
        if sub_gaps:
            sub_gap_by_k.append(sub_gaps)
    if valid_k_count == 0:
        return {
            "status": "power-insufficient",
            "positive": False,
            "positive_k_count": 0,
            "gap_full": {str(k): math.nan for k in RECURRENCE_K_GRID},
            "ci": {"low": math.nan, "high": math.nan},
        }
    aggregate_sub_gaps = []
    for index in range(SUBWINDOW_COUNT):
        values = [values[index] for values in sub_gap_by_k if index < len(values)]
        if values:
            aggregate_sub_gaps.append(float(np.median(values)))
    _median, low, high = bootstrap_median_ci(aggregate_sub_gaps, seed=stable_seed("probe3", band))
    positive = positive_k_count >= 2 and low > 0
    return {
        "status": "ok",
        "positive": bool(positive),
        "positive_k_count": int(positive_k_count),
        "gap_full": gap_full,
        "ci": {"low": low, "high": high},
    }


def build_region_readout(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    regions = ["P4-04", "300M", "1110M", "P4-05", "597M ARC-B"]
    for probe in ("probe1", "probe2", "probe3"):
        for region in regions:
            items = [row for row in rows if row["region"] == region]
            positives = sum(1 for row in items if truthy(row[f"{probe}_positive"]))
            power_insufficient = sum(1 for row in items if row[f"{probe}_status"] == "power-insufficient")
            denominator = len(items)
            output.append(
                {
                    "probe": probe,
                    "region": region,
                    "positive_windows": positives,
                    "denominator": denominator,
                    "power_insufficient_windows": power_insufficient,
                    "region_positive": region_positive(probe, region, positives, denominator),
                }
            )
    return output


def build_overall_readout(region_rows: list[dict[str, Any]]) -> dict[str, Any]:
    primary_support = {}
    for probe in ("probe1", "probe2"):
        p4 = get_region(region_rows, probe, "P4-04")
        c300 = get_region(region_rows, probe, "300M")
        c1110 = get_region(region_rows, probe, "1110M")
        primary_support[probe] = (
            int(p4["positive_windows"]) >= 2
            and int(c300["positive_windows"]) <= 1
            and int(c1110["positive_windows"]) <= 1
        )
    both_primary_power_insufficient = all(
        int(get_region(region_rows, probe, "P4-04")["power_insufficient_windows"])
        == int(get_region(region_rows, probe, "P4-04")["denominator"])
        for probe in ("probe1", "probe2")
    )
    if both_primary_power_insufficient:
        outcome = "inputs-or-power-insufficient"
    elif not any(primary_support.values()):
        outcome = "null-mimicry-or-no-structure"
    elif all(primary_support.values()):
        outcome = "local-subshift-like-support"
    else:
        outcome = "intermittent-structural-signal"

    arc_b_support = {}
    for probe in ("probe1", "probe2"):
        arc = get_region(region_rows, probe, "597M ARC-B")
        arc_b_support[probe] = int(arc["positive_windows"]) >= 2
    qualifier = ""
    if outcome == "local-subshift-like-support":
        qualifier = "positive-region-consistent" if all(arc_b_support.values()) else "P4-local only"
    return {
        "outcome": outcome,
        "primary_support": primary_support,
        "arc_b_support": arc_b_support,
        "arc_b_qualifier": qualifier,
        "stage1_readout_scope": "numeric-only-no-figures",
    }


def region_positive(probe: str, region: str, positives: int, denominator: int) -> bool:
    if region == "P4-04":
        return positives >= 2
    if region in {"300M", "1110M"}:
        return positives <= 1
    if region == "597M ARC-B":
        return positives >= 2
    return False


def get_region(rows: list[dict[str, Any]], probe: str, region: str) -> dict[str, Any]:
    for row in rows:
        if row["probe"] == probe and row["region"] == region:
            return row
    raise KeyError((probe, region))


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
    return float(mi), float(mi / denom if denom > 0 else 0.0)


def bootstrap_median_ci(values: list[float], *, seed: int) -> tuple[float, float, float]:
    clean = np.array([value for value in values if math.isfinite(value)], dtype=np.float64)
    if clean.size == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, clean.size, size=(BOOTSTRAP_SAMPLES, clean.size))
    medians = np.median(clean[draws], axis=1)
    return (
        float(np.median(medians)),
        float(np.percentile(medians, 2.5)),
        float(np.percentile(medians, 97.5)),
    )


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(f"{BOOTSTRAP_SEED}|{payload}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def as_uint8_view(bits: np.ndarray) -> np.ndarray:
    return np.asarray(bits, dtype=np.uint8)


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def log2_safe(value: int | float) -> float:
    return math.log2(max(float(value), 1.0))


def build_manifest(run_dir: Path, summary: dict[str, Any], source_manifests: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "run_id": summary["run_id"],
        "timestamp_utc": summary["timestamp_utc"],
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__)),
        "contract": summary["contract"],
        "stage0": summary["stage0"],
        "stage_boundary": {
            "stage1_executed": True,
            "figures_created": False,
            "extra_probes_created": False,
            "block_entropy_included": False,
        },
        "source_manifests": {
            key: {field: value for field, value in item.items() if field != "payload"}
            for key, item in source_manifests.items()
        },
        "outputs": {
            "summary": "summary.json",
            "manifest": "manifest.json",
            "report": "report.md",
            "source_inventory": "source_inventory.csv",
            "window_readout": "window_readout.csv",
            "region_readout": "region_readout.csv",
            "seed_inventory": "seed_inventory.csv",
            "source_metric_details": "source_metric_details.json",
            "probe_detail": "probe_detail.json",
        },
        "output_hashes": {
            path.name: sha256_file(path)
            for path in sorted(run_dir.iterdir())
            if path.is_file() and path.name != "manifest.json"
        },
    }


def write_report(path: Path, summary: dict[str, Any], region_rows: list[dict[str, Any]], window_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# D-0148 P4-SLS Stage 1 Readout",
        "",
        f"- Run: `{summary['run_id']}`",
        f"- Verdict: `{summary['stage1_verdict']}`",
        f"- Stage 0: `{summary['stage0']['path']}`",
        f"- Contract SHA: `{summary['contract']['sha256']}`",
        f"- Block-entropy policy: `{summary['block_entropy_policy']}`",
        "",
        "This is a numeric readout only. It creates no figures, adds no probes, and keeps `block-entropy` outside the hard-null envelope because the frozen contract treats P4-SLS as sequence-level over generative null streams only.",
        "",
        "## Overall",
        "",
        f"- Outcome: `{summary['overall_readout']['outcome']}`",
        f"- Primary support: `{summary['overall_readout']['primary_support']}`",
        f"- ARC-B qualifier: `{summary['overall_readout']['arc_b_qualifier'] or 'not-applicable'}`",
        "",
        "## Region Counts",
        "",
        "| Probe | Region | Positive | Denominator | Power insufficient | Region flag |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in region_rows:
        lines.append(
            f"| {row['probe']} | {row['region']} | {row['positive_windows']} | {row['denominator']} | "
            f"{row['power_insufficient_windows']} | {row['region_positive']} |"
        )
    lines.extend(
        [
            "",
            "## Window Readout",
            "",
            "| Region | Band | P1 | P1 gap22 | P1 gap24 | P2 | P2 gap | P3 | P3 pos-k |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in window_rows:
        lines.append(
            f"| {row['region']} | {row['band']} | {row['probe1_positive']} | {fmt(row['probe1_gap22'])} | "
            f"{fmt(row['probe1_gap24'])} | {row['probe2_positive']} | {fmt(row['probe2_gap_mi'])} | "
            f"{row['probe3_positive']} | {row['probe3_positive_k_count']} |"
        )
    lines.extend(
        [
            "",
            "## Wording Boundary",
            "",
            "The readout is a finite-block diagnostic over frozen Phase 4 coordinates. It does not establish asymptotic subshift properties, does not open Gate 2, and does not authorize manuscript changes by itself.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def render_console_summary(summary: dict[str, Any]) -> str:
    lines = [
        "",
        "D-0148 P4-SLS Stage 1 readout",
        "-" * 88,
        f"status: {summary['status']}",
        f"outcome: {summary['stage1_verdict']}",
        f"primary_support: {summary['overall_readout']['primary_support']}",
        f"arc_b_qualifier: {summary['overall_readout']['arc_b_qualifier'] or 'not-applicable'}",
        f"elapsed_seconds: {summary['elapsed_seconds']}",
    ]
    return "\n".join(lines)


def print_saved_paths(run_dir: Path) -> None:
    print(f"Saved summary to: {run_dir / 'summary.json'}")
    print(f"Saved report to: {run_dir / 'report.md'}")
    print(f"Saved window CSV to: {run_dir / 'window_readout.csv'}")
    print(f"Saved manifest to: {run_dir / 'manifest.json'}")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "nan"
    return f"{number:.6g}"


if __name__ == "__main__":
    raise SystemExit(main())
