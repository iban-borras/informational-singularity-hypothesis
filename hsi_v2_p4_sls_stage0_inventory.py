#!/usr/bin/env python3
"""P4-SLS Stage 0 inventory for D-0148.

This script implements only the read-only inventory authorized by the
P4-SLS v0.3.1 contract. It does not compute any Stage 1 probe, gap, p-value,
figure, or readout statistic.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hsi_v2_phase2_transport_defect_strict import phase_print
from v2.common.cli import resolve_dir
from v2.common.null_models import SUPPORTED_NULLS
from v2.phase2.return_lag import load_run_source_bits
from v2.phase2.null_pressure import discover_phase1_runs_recursive


CONTRACT_SHA256 = "FB150162FBBD9CF7D85B6AF364DA942052D26D3115D783D1C0D254AC4A57B451"

DEFAULT_CONTRACT = "../prereg_P4_subshift_like_signature_v0.3.1.md"
DEFAULT_OUTPUT_DIR = "results/hsi_v2/phase4/p4_sls_subshift_like_signature/stage0_inventory"
DEFAULT_PHASE1_DIR = "results/hsi_v2/phase1_high_scales"
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
DEFAULT_SOURCE_CACHE = "results/hsi_v2/source_cache/frozen_sources"

WINDOW_BITS = 9_000_000
LAGS = [-29_500_000, -27_000_000, -26_500_000, -24_000_000, 24_500_000, 26_500_000]
NULL_FAMILIES = ["phase-matched-lz", "matched-lz"]
BLOCK_ENTROPY_POLICY = "excluded-from-hard-null-envelope-sequence-level"

WINDOWS = [
    {"region": "P4-04", "role": "SUPPORTED", "band": "768M-777M", "start_bits": 768_000_000, "source": "P4-04"},
    {"region": "P4-04", "role": "SUPPORTED", "band": "777M-786M", "start_bits": 777_000_000, "source": "P4-04"},
    {"region": "P4-04", "role": "SUPPORTED", "band": "786M-795M", "start_bits": 786_000_000, "source": "P4-04"},
    {"region": "P4-05", "role": "BOUNDARY/CATCH-UP", "band": "795M-804M", "start_bits": 795_000_000, "source": "P4-05"},
    {"region": "P4-05", "role": "BOUNDARY/CATCH-UP", "band": "804M-813M", "start_bits": 804_000_000, "source": "P4-05"},
    {"region": "P4-05", "role": "BOUNDARY/CATCH-UP", "band": "813M-822M", "start_bits": 813_000_000, "source": "P4-05"},
    {"region": "300M", "role": "NON-REPLICATION", "band": "300M-309M", "start_bits": 300_000_000, "source": "B-mini"},
    {"region": "300M", "role": "NON-REPLICATION", "band": "309M-318M", "start_bits": 309_000_000, "source": "B-mini"},
    {"region": "300M", "role": "NON-REPLICATION", "band": "318M-327M", "start_bits": 318_000_000, "source": "B-mini"},
    {"region": "300M", "role": "NON-REPLICATION", "band": "327M-336M", "start_bits": 327_000_000, "source": "B-mini"},
    {"region": "300M", "role": "NON-REPLICATION", "band": "336M-345M", "start_bits": 336_000_000, "source": "B-mini"},
    {"region": "1110M", "role": "NON-REPLICATION", "band": "1110M-1119M", "start_bits": 1_110_000_000, "source": "B-mini"},
    {"region": "1110M", "role": "NON-REPLICATION", "band": "1119M-1128M", "start_bits": 1_119_000_000, "source": "B-mini"},
    {"region": "1110M", "role": "NON-REPLICATION", "band": "1128M-1137M", "start_bits": 1_128_000_000, "source": "B-mini"},
    {"region": "1110M", "role": "NON-REPLICATION", "band": "1137M-1146M", "start_bits": 1_137_000_000, "source": "B-mini"},
    {"region": "1110M", "role": "NON-REPLICATION", "band": "1146M-1155M", "start_bits": 1_146_000_000, "source": "B-mini"},
    {"region": "597M ARC-B", "role": "SANITY-CHECK", "band": "597M-606M", "start_bits": 597_000_000, "source": "B-mini"},
    {"region": "597M ARC-B", "role": "SANITY-CHECK", "band": "606M-615M", "start_bits": 606_000_000, "source": "B-mini"},
    {"region": "597M ARC-B", "role": "SANITY-CHECK", "band": "615M-624M", "start_bits": 615_000_000, "source": "B-mini"},
    {"region": "597M ARC-B", "role": "SANITY-CHECK", "band": "624M-633M", "start_bits": 624_000_000, "source": "B-mini"},
    {"region": "597M ARC-B", "role": "SANITY-CHECK", "band": "633M-642M", "start_bits": 633_000_000, "source": "B-mini"},
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="D-0148 P4-SLS Stage 0 inventory only.")
    parser.add_argument("--contract", default=DEFAULT_CONTRACT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--phase1-dir", default=DEFAULT_PHASE1_DIR)
    parser.add_argument("--p4-04-run", default=DEFAULT_P4_04_RUN)
    parser.add_argument("--p4-05-run", default=DEFAULT_P4_05_RUN)
    parser.add_argument("--bmini-run", default=DEFAULT_BMINI_RUN)
    parser.add_argument("--source-cache-dir", default=DEFAULT_SOURCE_CACHE)
    parser.add_argument(
        "--prepare-cache",
        action="store_true",
        help="Generate/extend missing generative-null source cache only; no probes/readout.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.perf_counter()

    contract = resolve_dir(args.contract, anchor_file=__file__)
    output_root = resolve_dir(args.output_dir, anchor_file=__file__)
    phase1_dir = resolve_dir(args.phase1_dir, anchor_file=__file__)
    p4_04_run = resolve_dir(args.p4_04_run, anchor_file=__file__)
    p4_05_run = resolve_dir(args.p4_05_run, anchor_file=__file__)
    bmini_run = resolve_dir(args.bmini_run, anchor_file=__file__)
    source_cache_dir = resolve_dir(args.source_cache_dir, anchor_file=__file__)

    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_dir = output_root / f"p4-sls-stage0-inventory__windows-21__{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    phase_print(
        "Preparing D-0148 P4-SLS Stage 0",
        "inventory only | no probes | no readout | no figures",
        quiet=args.quiet,
    )

    source_manifests = load_source_manifests(p4_04_run, p4_05_run, bmini_run)
    seed_inventory = extract_seed_inventory(source_manifests)
    source_requirements = build_source_requirements()
    phase1_runs = discover_phase1_runs_recursive(phase1_dir)
    observed_run = select_observed_b_run(phase1_runs)
    observed_source_info = source_info_from_run(observed_run)
    source_requirements["source_length_bits"] = observed_source_info["source_length_bits"]
    null_run_index = select_null_runs(phase1_runs, seed_inventory["seeds"])
    cache_index = scan_source_cache(source_cache_dir)
    source_inventory = build_source_inventory(
        observed_run=observed_run,
        null_run_index=null_run_index,
        seeds=seed_inventory["seeds"],
        required_bits=source_requirements["max_required_prefix_bits"],
        cache_index=cache_index,
    )
    cache_preparation_rows: list[dict[str, Any]] = []
    if args.prepare_cache:
        cache_preparation_rows = prepare_missing_cache(
            null_run_index=null_run_index,
            seeds=seed_inventory["seeds"],
            required_bits=source_requirements["max_required_prefix_bits"],
            source_inventory=source_inventory,
            quiet=args.quiet,
        )
        cache_index = scan_source_cache(source_cache_dir)
        source_inventory = build_source_inventory(
            observed_run=observed_run,
            null_run_index=null_run_index,
            seeds=seed_inventory["seeds"],
            required_bits=source_requirements["max_required_prefix_bits"],
            cache_index=cache_index,
        )
    window_inventory = build_window_inventory(observed_run, source_requirements["source_length_bits"])
    verdict = build_verdict(
        contract=contract,
        source_manifests=source_manifests,
        seed_inventory=seed_inventory,
        source_inventory=source_inventory,
        window_inventory=window_inventory,
    )

    summary = {
        "run_id": run_dir.name,
        "timestamp_utc": timestamp,
        "status": "completed",
        "stage0_verdict": verdict["status"],
        "contract": {"path": str(contract.resolve()), "sha256": sha256_file(contract), "expected_sha256": CONTRACT_SHA256},
        "authorized_scope": [
            "manifest-derived seed inventory for phase-matched-lz and matched-lz",
            "raw sequence extractability checks for B and generative null families",
            "I/O and memory cost estimates",
            "optional source-cache preparation when --prepare-cache is explicitly passed",
            "no Stage 1 probes, no readout, no figures",
        ],
        "block_entropy_policy": BLOCK_ENTROPY_POLICY,
        "parameters": {
            "window_bits": WINDOW_BITS,
            "lags": LAGS,
            "null_families": NULL_FAMILIES,
            "phase1_dir": str(phase1_dir),
            "source_cache_dir": str(source_cache_dir),
        },
        "source_requirements": source_requirements,
        "seed_inventory": seed_inventory,
        "counts": {
            "windows": len(window_inventory),
            "generative_null_families": len(NULL_FAMILIES),
            "seeds": len(seed_inventory["seeds"]),
            "source_inventory_rows": len(source_inventory),
            "cache_preparation_rows": len(cache_preparation_rows),
        },
        "cache_preparation": {
            "requested": bool(args.prepare_cache),
            "rows": cache_preparation_rows,
        },
        "verdict": verdict,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }

    summary_path = run_dir / "summary.json"
    write_json(summary_path, summary)
    manifest = build_manifest(summary, source_manifests)
    manifest["summary_sha256"] = sha256_file(summary_path)
    write_json(run_dir / "manifest.json", manifest)
    write_csv(run_dir / "window_inventory.csv", window_inventory)
    write_csv(run_dir / "source_inventory.csv", source_inventory)
    write_csv(run_dir / "seed_inventory.csv", seed_inventory["rows"])
    write_csv(run_dir / "cache_preparation.csv", cache_preparation_rows)
    (run_dir / "report.md").write_text(render_report(summary, source_inventory, window_inventory), encoding="utf-8")

    print(render_console_summary(summary))
    print_saved_paths(run_dir)
    return 0


def load_source_manifests(p4_04_run: Path, p4_05_run: Path, bmini_run: Path) -> dict[str, dict[str, Any]]:
    paths = {
        "P4-04": p4_04_run / "manifest.json",
        "P4-05": p4_05_run / "manifest.json",
        "B-mini": bmini_run / "manifest.json",
    }
    manifests = {}
    for label, path in paths.items():
        payload = load_json(path) if path.is_file() else {}
        manifests[label] = {
            "label": label,
            "path": str(path.resolve()),
            "exists": path.is_file(),
            "sha256": sha256_file(path) if path.is_file() else None,
            "payload": payload,
        }
    return manifests


def extract_seed_inventory(source_manifests: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = []
    seed_sets = []
    for label, item in source_manifests.items():
        payload = item.get("payload", {})
        seeds = parse_seed_string(deep_get(payload, ["parameters", "seeds"]))
        if not seeds:
            seeds = parse_seed_string(command_value(payload.get("command", []), "--matched-lz-seeds"))
        if not seeds:
            seeds = parse_seed_string(command_value(deep_get(payload, ["commands", "strict_p4"], []), "--matched-lz-seeds"))
        null_models = parse_models(deep_get(payload, ["parameters", "null_models"]))
        if not null_models:
            null_models = parse_models(command_value(payload.get("command", []), "--null-models"))
        if not null_models:
            null_models = parse_models(command_value(deep_get(payload, ["commands", "strict_p4"], []), "--null-models"))
        if seeds:
            seed_sets.append(tuple(seeds))
        rows.append(
            {
                "source": label,
                "manifest_exists": item["exists"],
                "seeds": ",".join(str(seed) for seed in seeds),
                "null_models_declared": ",".join(null_models),
                "phase_matched_lz_declared": "phase-matched-lz" in null_models,
                "matched_lz_declared": "matched-lz" in null_models,
                "matched_lz_seed_list_present": bool(seeds),
                "note": "matched-lz seed list is manifest-derived from the shared LZ seed parameter; matched-lz was not part of the original P4/B-mini null_models",
            }
        )
    unique_sets = sorted(set(seed_sets))
    seeds = list(unique_sets[0]) if len(unique_sets) == 1 else []
    return {
        "seeds": seeds,
        "seed_sets_consistent": len(unique_sets) == 1 and bool(seeds),
        "seed_sets_seen": [list(values) for values in unique_sets],
        "rows": rows,
    }


def build_source_requirements() -> dict[str, Any]:
    min_start = min(window["start_bits"] for window in WINDOWS)
    max_end = max(window["start_bits"] + WINDOW_BITS for window in WINDOWS)
    min_lag_start = min(window["start_bits"] + lag for window in WINDOWS for lag in LAGS)
    max_lag_end = max(window["start_bits"] + lag + WINDOW_BITS for window in WINDOWS for lag in LAGS)
    max_required = max(max_end, max_lag_end)
    return {
        "window_bits": WINDOW_BITS,
        "window_count": len(WINDOWS),
        "lags": LAGS,
        "min_window_start_bits": min_start,
        "max_window_end_bits": max_end,
        "min_lagged_start_bits": min_lag_start,
        "max_lagged_end_bits": max_lag_end,
        "max_required_prefix_bits": max_required,
        "max_required_prefix_gib_uint8": round(max_required / (1024**3), 3),
        "source_length_bits": None,
    }


def select_observed_b_run(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = []
    for run in runs:
        config = run["dataset"].get("config", {})
        if str(config.get("sequence_kind", "observed")) == "null_surrogate":
            continue
        if str(config.get("variant", "")).upper() != "B":
            continue
        if int(config.get("iteration", -1)) != 20:
            continue
        if 48 not in [int(value) for value in config.get("scales", [])]:
            continue
        candidates.append(run)
    return max(candidates, key=lambda item: item["_generated_at_dt"]) if candidates else None


def select_null_runs(runs: list[dict[str, Any]], seeds: list[int]) -> dict[tuple[str, int], dict[str, Any]]:
    latest = {}
    for run in runs:
        config = run["dataset"].get("config", {})
        if str(config.get("sequence_kind")) != "null_surrogate":
            continue
        if str(config.get("source_variant", "")).upper() != "B":
            continue
        if int(config.get("iteration", -1)) != 20:
            continue
        family = str(config.get("null_model", "")).strip()
        if family not in NULL_FAMILIES:
            continue
        seed = int(config.get("null_seed", -1))
        if seeds and seed not in seeds:
            continue
        key = (family, seed)
        current = latest.get(key)
        if current is None or run["_generated_at_dt"] > current["_generated_at_dt"]:
            latest[key] = run
    return latest


def scan_source_cache(cache_dir: Path) -> dict[tuple[str, int], list[dict[str, Any]]]:
    index: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    if not cache_dir.is_dir():
        return index
    for meta_path in sorted(cache_dir.glob("*.json")):
        try:
            payload = load_json(meta_path)
        except Exception:
            continue
        identity = payload.get("identity", {})
        family = str(identity.get("null_model", ""))
        if family not in NULL_FAMILIES:
            continue
        seed = int(identity.get("null_seed", -1))
        bits_path = Path(str(payload.get("bits_path", "")))
        index[(family, seed)].append(
            {
                "metadata_path": str(meta_path.resolve()),
                "bits_path": str(bits_path),
                "bits_exists": bits_path.is_file(),
                "required_bits": int(payload.get("required_bits", 0)),
                "identity_hash": str(payload.get("identity_hash", "")),
            }
        )
    return index


def build_source_inventory(
    *,
    observed_run: dict[str, Any] | None,
    null_run_index: dict[tuple[str, int], dict[str, Any]],
    seeds: list[int],
    required_bits: int,
    cache_index: dict[tuple[str, int], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    observed_source = source_info_from_run(observed_run)
    rows.append(
        {
            "source_kind": "observed",
            "family": "B",
            "seed": "",
            **observed_source,
            "source_cache_status": "not-applicable",
            "cache_bits_available": "",
            "derivable": observed_source["input_struct_exists"] and observed_source["source_length_bits"] >= required_bits,
        }
    )
    for family in NULL_FAMILIES:
        for seed in seeds:
            run = null_run_index.get((family, seed))
            info = source_info_from_run(run)
            cache_hits = [item for item in cache_index.get((family, seed), []) if item["bits_exists"]]
            max_cache_bits = max([item["required_bits"] for item in cache_hits], default=0)
            rows.append(
                {
                    "source_kind": "null_surrogate",
                    "family": family,
                    "seed": seed,
                    **info,
                    "source_cache_status": "hit-sufficient" if max_cache_bits >= required_bits else "hit-insufficient" if max_cache_bits else "missing",
                    "cache_bits_available": max_cache_bits,
                    "derivable": (
                        family in SUPPORTED_NULLS
                        and info["phase1_run_exists"]
                        and info["input_struct_exists"]
                        and info["source_length_bits"] >= required_bits
                    ),
                }
            )
    return rows


def prepare_missing_cache(
    *,
    null_run_index: dict[tuple[str, int], dict[str, Any]],
    seeds: list[int],
    required_bits: int,
    source_inventory: list[dict[str, Any]],
    quiet: bool,
) -> list[dict[str, Any]]:
    rows = []
    needs_cache = {
        (str(row["family"]), int(row["seed"]))
        for row in source_inventory
        if row["source_kind"] == "null_surrogate"
        and str(row["source_cache_status"]) != "hit-sufficient"
    }
    for family in NULL_FAMILIES:
        for seed in seeds:
            key = (family, seed)
            if key not in needs_cache:
                rows.append(
                    {
                        "family": family,
                        "seed": seed,
                        "action": "cache-already-sufficient",
                        "required_bits": required_bits,
                        "source_cache_status": "hit-sufficient",
                        "source_cache_path": "",
                    }
                )
                continue
            run = null_run_index.get(key)
            if run is None:
                rows.append(
                    {
                        "family": family,
                        "seed": seed,
                        "action": "missing-phase1-run",
                        "required_bits": required_bits,
                        "source_cache_status": "missing",
                        "source_cache_path": "",
                    }
                )
                continue
            phase_print(
                "Preparing P4-SLS source cache",
                f"{family} seed={seed} | required={required_bits:,} bits",
                quiet=quiet,
            )
            _, source_info = load_run_source_bits(
                run["dataset"],
                required_bits=required_bits,
                show_progress=not quiet,
                progress_label=f"P4-SLS {family} seed {seed}",
            )
            rows.append(
                {
                    "family": family,
                    "seed": seed,
                    "action": "cache-prepared",
                    "required_bits": required_bits,
                    "source_cache_status": source_info.get("source_cache_status", ""),
                    "source_cache_path": source_info.get("source_cache_path", ""),
                    "source_cache_metadata_path": source_info.get("source_cache_metadata_path", ""),
                    "source_cache_total_bits": source_info.get("source_cache_total_bits", ""),
                }
            )
    return rows


def source_info_from_run(run: dict[str, Any] | None) -> dict[str, Any]:
    if run is None:
        return {
            "phase1_run_exists": False,
            "run_dir": "",
            "dataset_path": "",
            "dataset_sha256": "",
            "input_struct_path": "",
            "input_struct_exists": False,
            "input_metadata_path": "",
            "input_metadata_exists": False,
            "source_length_bits": 0,
        }
    dataset = run["dataset"]
    config = dataset.get("config", {})
    source_metadata = dataset.get("source_metadata", {})
    struct_path = Path(str(config.get("input_struct_path", "")))
    metadata_path = Path(str(config.get("input_metadata_path", "")))
    source_length = int(
        source_metadata.get("sequence_length")
        or deep_get(source_metadata, ["format_info", "length"])
        or 0
    )
    dataset_path = Path(run["_dataset_path"])
    return {
        "phase1_run_exists": True,
        "run_dir": run["_run_dir"],
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "input_struct_path": str(struct_path),
        "input_struct_exists": struct_path.is_file(),
        "input_metadata_path": str(metadata_path),
        "input_metadata_exists": metadata_path.is_file(),
        "source_length_bits": source_length,
    }


def build_window_inventory(observed_run: dict[str, Any] | None, source_length_bits: int | None) -> list[dict[str, Any]]:
    if observed_run is not None:
        source_length_bits = source_info_from_run(observed_run)["source_length_bits"]
    rows = []
    for window in WINDOWS:
        lagged_starts = [window["start_bits"] + lag for lag in LAGS]
        lagged_ends = [window["start_bits"] + lag + WINDOW_BITS for lag in LAGS]
        min_required = min([window["start_bits"], *lagged_starts])
        max_required = max([window["start_bits"] + WINDOW_BITS, *lagged_ends])
        rows.append(
            {
                **window,
                "end_bits": window["start_bits"] + WINDOW_BITS,
                "min_lagged_start_bits": min(lagged_starts),
                "max_lagged_end_bits": max(lagged_ends),
                "min_required_start_bits": min_required,
                "max_required_end_bits": max_required,
                "in_source_bounds": (
                    source_length_bits is not None
                    and min_required >= 0
                    and max_required <= int(source_length_bits)
                ),
            }
        )
    return rows


def build_verdict(
    *,
    contract: Path,
    source_manifests: dict[str, dict[str, Any]],
    seed_inventory: dict[str, Any],
    source_inventory: list[dict[str, Any]],
    window_inventory: list[dict[str, Any]],
) -> dict[str, Any]:
    failures = []
    if not contract.is_file():
        failures.append("contract file missing")
    elif sha256_file(contract) != CONTRACT_SHA256:
        failures.append("contract sha256 mismatch")
    for label, manifest in source_manifests.items():
        if not manifest["exists"]:
            failures.append(f"{label} manifest missing")
    if not seed_inventory["seed_sets_consistent"]:
        failures.append("manifest-derived seed sets are missing or inconsistent")
    for row in source_inventory:
        if not bool(row["derivable"]):
            failures.append(f"{row['source_kind']} {row['family']} seed={row['seed']} is not derivable")
    for row in window_inventory:
        if not bool(row["in_source_bounds"]):
            failures.append(f"{row['band']} required interval outside source bounds")
    return {
        "status": "inputs-not-derivable" if failures else "inputs-derivable",
        "failures": failures,
        "stage1_authorized": False,
        "note": "Stage 0 only; no probe/readout/figure was computed.",
    }


def build_manifest(summary: dict[str, Any], source_manifests: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "run_id": summary["run_id"],
        "timestamp_utc": summary["timestamp_utc"],
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "summary_sha256": None,
        "contract": summary["contract"],
        "source_manifests": {
            key: {name: value for name, value in item.items() if name != "payload"}
            for key, item in source_manifests.items()
        },
        "block_entropy_policy": BLOCK_ENTROPY_POLICY,
        "stage_boundary": {
            "stage0_only": True,
            "stage1_authorized": False,
            "probes_computed": False,
            "figures_created": False,
        },
        "outputs": {
            "summary": "summary.json",
            "manifest": "manifest.json",
            "report": "report.md",
            "window_inventory": "window_inventory.csv",
            "source_inventory": "source_inventory.csv",
            "seed_inventory": "seed_inventory.csv",
            "cache_preparation": "cache_preparation.csv",
        },
    }


def render_report(summary: dict[str, Any], source_rows: list[dict[str, Any]], window_rows: list[dict[str, Any]]) -> str:
    cache_counts = defaultdict(int)
    for row in source_rows:
        cache_counts[str(row["source_cache_status"])] += 1
    lines = [
        "# D-0148 P4-SLS Stage 0 inventory",
        "",
        f"- Run: `{summary['run_id']}`",
        f"- Verdict: `{summary['stage0_verdict']}`",
        f"- Contract SHA-256: `{summary['contract']['sha256']}`",
        f"- Block-entropy policy: `{BLOCK_ENTROPY_POLICY}`",
        "",
        "## Boundary",
        "",
        "This run performs inventory only. It computes no Stage 1 probe, no hard-null gap, no p-value, no qualitative readout, and no figure.",
        "",
        "## Inventory result",
        "",
        f"- Windows checked: {len(window_rows)}",
        f"- Null families checked: `{', '.join(NULL_FAMILIES)}`",
        f"- Manifest-derived seeds: `{','.join(str(seed) for seed in summary['seed_inventory']['seeds'])}`",
        f"- Required prefix for lag-conditioned windows: {summary['source_requirements']['max_required_prefix_bits']:,} bits",
        f"- Conservative per-stream uint8 memory: {summary['source_requirements']['max_required_prefix_gib_uint8']} GiB",
        f"- Source cache status counts: {dict(cache_counts)}",
        f"- Cache preparation requested: {summary['cache_preparation']['requested']}",
        "",
        "## Verdict details",
        "",
    ]
    failures = summary["verdict"]["failures"]
    if failures:
        lines.extend(f"- {failure}" for failure in failures)
    else:
        lines.append("- All frozen manifests exist.")
        lines.append("- Seed sets are manifest-derived and consistent.")
        lines.append("- Observed B and both generative null families are derivable for all declared windows.")
        lines.append("- All lag-conditioned intervals remain inside the B@20 source length.")
    return "\n".join(lines) + "\n"


def render_console_summary(summary: dict[str, Any]) -> str:
    lines = [
        "D-0148 P4-SLS Stage 0 inventory",
        "-" * 72,
        f"status: {summary['status']}",
        f"verdict: {summary['stage0_verdict']}",
        f"windows: {summary['counts']['windows']}",
        f"null families: {', '.join(summary['parameters']['null_families'])}",
        f"seeds: {','.join(str(seed) for seed in summary['seed_inventory']['seeds'])}",
        f"required prefix bits: {summary['source_requirements']['max_required_prefix_bits']:,}",
        f"elapsed: {summary['elapsed_seconds']:.3f}s",
    ]
    if summary["verdict"]["failures"]:
        lines.append("failures:")
        lines.extend(f"  - {failure}" for failure in summary["verdict"]["failures"])
    return "\n".join(lines)


def print_saved_paths(run_dir: Path) -> None:
    print(f"\nSaved summary to: {run_dir / 'summary.json'}")
    print(f"Saved report to: {run_dir / 'report.md'}")
    print(f"Saved manifest to: {run_dir / 'manifest.json'}")


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sha256_file(path: Path) -> str:
    digest = __import__("hashlib").sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def parse_seed_string(raw: Any) -> list[int]:
    if raw is None:
        return []
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def parse_models(raw: Any) -> list[str]:
    if raw is None:
        return []
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def command_value(command: Any, flag: str) -> str | None:
    if not isinstance(command, list):
        return None
    for index, token in enumerate(command):
        text = str(token)
        if text == flag and index + 1 < len(command):
            return str(command[index + 1])
        prefix = f"{flag}="
        if text.startswith(prefix):
            return text[len(prefix) :]
    return None


def deep_get(payload: Any, path: list[str]) -> Any:
    value = payload
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
