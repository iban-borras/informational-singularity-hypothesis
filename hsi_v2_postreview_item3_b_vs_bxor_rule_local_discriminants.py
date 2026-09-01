#!/usr/bin/env python3
"""D-0160 Item 3: B vs B_XOR rule-local discriminants."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

try:
    import psutil
except ImportError:  # pragma: no cover - execution gate rejects this environment.
    psutil = None  # type: ignore[assignment]


REPO_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = REPO_DIR.parent
sys.path.insert(0, str(REPO_DIR))

from v2.postreview.item2b_growth_shells import (  # noqa: E402
    RunLogger,
    sha256_file,
    sha256_file_progress,
    verify_compiled_kernels,
    write_csv_atomic,
    write_json_atomic,
)
from v2.postreview.item3_rule_local_discriminants import (  # noqa: E402
    NULL_FAMILIES,
    NULL_IMPLEMENTATION_SHA256,
    PREREG_SHA256,
    SEEDS,
    SITE_FRACTIONS,
    SOURCE_SHA256,
    build_item3_readout,
    build_profile_job,
    count_observable_bits,
    derive_site_specs,
    extract_observed_sites,
    frozen_protocol,
    generate_null_cache_job,
    null_cache_identity,
    profile_job_identity,
    validate_identity_cache,
    valid_fit_metadata,
)


PREREG_PATH = (
    PROJECT_ROOT / "prereg_postreview_item3_b_vs_bxor_rule_local_discriminants_v1.0.md"
)
NULL_MODELS_PATH = REPO_DIR / "v2" / "common" / "null_models.py"
DEFAULT_OUTPUT = "results/hsi_v2/postreview_item3_b_vs_bxor_rule_local_discriminants"
RAM_BUDGET_BYTES = 32 * 1024**3
STAGE0_RSS_LIMIT_BYTES = 4 * 1024**3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("stage0", "stage1"), required=True)
    parser.add_argument("--results-base", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--run-dir", default=None, help="Explicit run directory for strict resume."
    )
    parser.add_argument("--raw-chunk-mib", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--stage0-manifest", default=None)
    parser.add_argument("--stage0-manifest-sha256", default=None)
    parser.add_argument("--stage1-authorization-sha256", default=None)
    parser.add_argument("--code-audit-sha256", default=None)
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> int:
    configure_stdout()
    load_dotenv()
    args = build_parser().parse_args()
    verify_preregistration()
    validate_args(args)
    results_base = resolve_results_base(args.results_base)
    output_root = resolve_results_path(args.output_dir, results_base)
    run_dir = resolve_run_dir(args, output_root)
    if not args.execute:
        print_plan(args, results_base, run_dir)
        return 0
    if psutil is None:
        raise SystemExit(
            "psutil is required for contract-level RSS telemetry and worker admission."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(run_dir / "run.log")
    logger.emit(
        "run-start",
        stage=args.stage,
        run_dir=str(run_dir),
        preregistration_sha256=PREREG_SHA256,
        code_audit_sha256=args.code_audit_sha256.upper(),
    )
    try:
        if args.stage == "stage0":
            return execute_stage0(args, results_base, run_dir, logger)
        return execute_stage1(args, run_dir, logger)
    except KeyboardInterrupt:
        logger.emit("run-interrupted", stage=args.stage)
        return 130
    except Exception as exc:
        logger.emit("run-failed", stage=args.stage, error=repr(exc))
        write_failure(run_dir, args.stage, args.code_audit_sha256, exc)
        raise


def execute_stage0(
    args: argparse.Namespace,
    results_base: Path,
    run_dir: Path,
    logger: RunLogger,
) -> int:
    started = time.monotonic()
    kernel_state = verify_compiled_kernels()
    logger.emit("compiled-fixtures-passed", **kernel_state)
    code_hashes = current_code_hashes()
    if code_hashes["null_models"] != NULL_IMPLEMENTATION_SHA256:
        raise RuntimeError(
            "Execution-time null implementation differs from the frozen reference hash."
        )
    sources = source_paths(results_base)
    source_state_path = run_dir / "source_state.json"
    source_state = load_json(source_state_path, default={"sources": {}})
    observed_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []

    for source in ("B", "BXOR"):
        path = sources[source]
        if not path.exists():
            raise FileNotFoundError(f"Frozen source not found: {path}")
        logger.emit("source-hash-start", source=source, path=str(path))
        actual_sha256 = sha256_file_progress(path, logger, label=f"source-{source}")
        if actual_sha256 != SOURCE_SHA256[source]:
            raise RuntimeError(
                f"Source hash mismatch for {source}: expected {SOURCE_SHA256[source]}, got {actual_sha256}."
            )
        reusable = validate_source_state(
            source_state.get("sources", {}).get(source), actual_sha256
        )
        if reusable is not None:
            logger.emit(
                "source-state-reused", source=source, windows=len(reusable["windows"])
            )
            source_rows.append(reusable["source"])
            observed_rows.extend(reusable["windows"])
            continue

        last_count_progress = 0.0

        def count_progress(current: int, total: int) -> None:
            nonlocal last_count_progress
            now = time.monotonic()
            if now - last_count_progress < 30.0:
                return
            last_count_progress = now
            logger.emit(
                "progress",
                force=True,
                phase="stage0-count-observable",
                source=source,
                packed_bytes=f"{current}/{total}",
                percent=round(100.0 * current / max(1, total), 3),
            )

        count_telemetry = count_observable_bits(
            path,
            raw_chunk_bytes=args.raw_chunk_mib * 1024**2,
            progress=count_progress,
        )
        rows, extraction = extract_observed_sites(
            path,
            source=source,
            source_sha256=actual_sha256,
            count_telemetry=count_telemetry,
            cache_dir=run_dir / "caches",
            raw_chunk_bytes=args.raw_chunk_mib * 1024**2,
            logger=logger,
        )
        source_row = {
            "source": source,
            "path": str(path.resolve()),
            "sha256": actual_sha256,
            "sequence_length": count_telemetry["sequence_length"],
            "observable_length": count_telemetry["observable_length"],
            "telemetry": extraction,
        }
        source_rows.append(source_row)
        observed_rows.extend(rows)
        source_state.setdefault("sources", {})[source] = {
            "source": source_row,
            "windows": rows,
        }
        write_json_atomic(source_state_path, source_state)
        logger.emit("source-sites-frozen", source=source, windows=len(rows))

    if len(observed_rows) != 6:
        raise RuntimeError(f"Expected six observed caches, got {len(observed_rows)}.")
    null_specs = build_null_specs(observed_rows, run_dir)
    records, preflight = run_cache_jobs(null_specs, args.workers, logger)
    if len(records) != 72:
        raise RuntimeError(f"Expected 72 null cache closures, got {len(records)}.")
    null_rows = [record["sidecar"] for record in records]
    if any(record.get("status") != "completed" for record in records):
        raise RuntimeError("At least one mandatory null cache did not complete.")
    peak_rss = max(
        [int(row["telemetry"]["decode_peak_rss_bytes"]) for row in source_rows]
        + [int(record["peak_rss_bytes"]) for record in records]
    )
    if peak_rss > STAGE0_RSS_LIMIT_BYTES:
        raise RuntimeError(f"Stage 0 peak RSS exceeded 4 GiB: {peak_rss}.")

    report_path = run_dir / "stage0_report.md"
    write_stage0_report(
        report_path, source_rows, observed_rows, null_rows, preflight, peak_rss
    )
    manifest_path = run_dir / "stage0_manifest.json"
    manifest = {
        "status": "completed",
        "stage": "stage0",
        "outcome": "inputs-derivable",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "preregistration": {"path": str(PREREG_PATH), "sha256": PREREG_SHA256},
        "code_audit_sha256": args.code_audit_sha256.upper(),
        "code_hashes": code_hashes,
        "kernel_state": kernel_state,
        "protocol": frozen_protocol(),
        "sources": source_rows,
        "observed_caches": observed_rows,
        "null_caches": null_rows,
        "null_cache_jobs": records,
        "resource_preflight": preflight,
        "peak_rss_bytes": peak_rss,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "scientific_values_emitted": [],
    }
    manifest["artifact_hashes"] = hash_artifacts(
        run_dir, exclude={"run.log", "stage0_manifest.json", "stage0_manifest.sha256"}
    )
    write_json_atomic(manifest_path, manifest)
    manifest_sha256 = sha256_file(manifest_path)
    (run_dir / "stage0_manifest.sha256").write_text(
        f"{manifest_sha256}  stage0_manifest.json\n", encoding="ascii"
    )
    logger.emit(
        "stage0-completed", outcome="inputs-derivable", manifest_sha256=manifest_sha256
    )
    print("Stage 0 outcome: inputs-derivable")
    print(f"Stage 0 manifest: {manifest_path}")
    print(f"Stage 0 manifest SHA-256: {manifest_sha256}")
    print(
        "STOP: Stage 1 requires independent manifest review and explicit authorization."
    )
    return 0


def execute_stage1(args: argparse.Namespace, run_dir: Path, logger: RunLogger) -> int:
    started = time.monotonic()
    kernel_state = verify_compiled_kernels()
    logger.emit("compiled-fixtures-passed", **kernel_state)
    stage0_path = Path(args.stage0_manifest).expanduser().resolve()
    stage0_sha256 = sha256_file(stage0_path)
    if stage0_sha256 != args.stage0_manifest_sha256.upper():
        raise RuntimeError(
            f"Stage 0 manifest hash mismatch: expected {args.stage0_manifest_sha256}, got {stage0_sha256}."
        )
    stage0 = json.loads(stage0_path.read_text(encoding="utf-8"))
    validate_stage0_manifest(stage0, stage0_path.parent)
    code_hashes = current_code_hashes()
    if stage0.get("code_hashes") != code_hashes:
        raise RuntimeError("Stage 0 and Stage 1 implementation hashes differ.")
    jobs = build_profile_specs(stage0, stage0_sha256, code_hashes, run_dir)
    manifest_path = run_dir / "manifest.json"
    manifest: dict[str, Any] = {
        "status": "running",
        "stage": "stage1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "preregistration": {"path": str(PREREG_PATH), "sha256": PREREG_SHA256},
        "code_audit_sha256": args.code_audit_sha256.upper(),
        "stage1_authorization_sha256": args.stage1_authorization_sha256.upper(),
        "stage0_manifest": str(stage0_path),
        "stage0_manifest_sha256": stage0_sha256,
        "code_hashes": code_hashes,
        "kernel_state": kernel_state,
        "protocol": frozen_protocol(),
        "requested_workers": args.workers,
        "jobs": [],
    }
    write_json_atomic(manifest_path, manifest)

    preflight_ids = {
        "b_q1of10__observed",
        "b_q1of10__phase-matched-lz__seed-17",
    }
    preflight_specs = [spec for spec in jobs if spec["job_id"] in preflight_ids]
    preflight_records = run_parallel_jobs(
        preflight_specs,
        build_profile_job,
        workers=1,
        logger=logger,
        phase="stage1-profile-preflight",
    )
    max_preflight_rss = max(
        int(record["peak_rss_bytes"]) for record in preflight_records
    )
    admission = max(1, int(RAM_BUDGET_BYTES // max(1, int(1.5 * max_preflight_rss))))
    effective_workers = max(1, min(4, args.workers, admission))
    preflight = {
        "job_ids": sorted(preflight_ids),
        "max_peak_rss_bytes": max_preflight_rss,
        "ram_budget_bytes": RAM_BUDGET_BYTES,
        "safety_factor": 1.5,
        "effective_workers": effective_workers,
    }
    logger.emit("stage1-preflight-completed", **preflight)
    record_map = {record["job_id"]: record for record in preflight_records}
    manifest["jobs"] = ordered_records(jobs, record_map)
    manifest["resource_preflight"] = preflight
    write_json_atomic(manifest_path, manifest)

    remaining = [spec for spec in jobs if spec["job_id"] not in record_map]

    def checkpoint(record: dict[str, Any]) -> None:
        record_map[record["job_id"]] = record
        manifest["jobs"] = ordered_records(jobs, record_map)
        write_json_atomic(manifest_path, manifest)

    completed = run_parallel_jobs(
        remaining,
        build_profile_job,
        workers=effective_workers,
        logger=logger,
        phase="stage1-profile-jobs",
        on_record=checkpoint,
        initial_completed=len(record_map),
        total_override=len(jobs),
    )
    for record in completed:
        record_map[record["job_id"]] = record
    records = ordered_records(jobs, record_map)
    if len(records) != 78 or any(
        record.get("status") != "completed" for record in records
    ):
        raise RuntimeError("Stage 1 did not complete all 78 frozen profile jobs.")
    manifest["jobs"] = records
    write_json_atomic(manifest_path, manifest)

    readout = build_item3_readout(records)
    summary = {
        **readout["summary"],
        "sources": stage0["sources"],
        "source_sites": stage0["observed_caches"],
        "per_segment": readout["per_segment"],
        "protocol": frozen_protocol(),
        "preregistration_sha256": PREREG_SHA256,
        "code_audit_sha256": args.code_audit_sha256.upper(),
        "stage0_manifest_sha256": stage0_sha256,
        "implementation_hashes": code_hashes,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "worker_count": effective_workers,
        "peak_ram_estimate_bytes": int(1.5 * max_preflight_rss * effective_workers),
        "resume_history": {
            "reused_jobs": sum(bool(record.get("reused")) for record in records),
            "fresh_jobs": sum(not bool(record.get("reused")) for record in records),
        },
    }
    write_stage1_outputs(run_dir, summary, readout, records)
    summary["paper_facing_artifact_sha256"] = {
        name: sha256_file(run_dir / name)
        for name in required_output_names()
        if name not in {"summary.json", "manifest.json"}
    }
    write_json_atomic(run_dir / "summary.json", summary)
    manifest.update(
        {
            "status": "completed",
            "outcome": summary["outcome"],
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "elapsed_seconds": summary["elapsed_seconds"],
            "jobs": records,
            "outputs": {
                name: str((run_dir / name).resolve())
                for name in required_output_names()
            },
        }
    )
    manifest["artifact_hashes"] = hash_artifacts(
        run_dir, exclude={"run.log", "manifest.json", "manifest.sha256"}
    )
    write_json_atomic(manifest_path, manifest)
    manifest_sha256 = sha256_file(manifest_path)
    (run_dir / "manifest.sha256").write_text(
        f"{manifest_sha256}  manifest.json\n", encoding="ascii"
    )
    logger.emit(
        "stage1-completed", outcome=summary["outcome"], manifest_sha256=manifest_sha256
    )
    print("D-0160 Item 3 Stage 1 completed.")
    print(f"Outcome: {summary['outcome']}")
    print(f"Output: {run_dir}")
    print(f"Manifest SHA-256: {manifest_sha256}")
    return 0


def source_paths(results_base: Path) -> dict[str, Path]:
    return {
        "B": results_base
        / "level0"
        / "phi_snapshots"
        / "var_B"
        / "phi_iter20.struct.gz",
        "BXOR": (
            results_base
            / "hsi_v2"
            / "q2b_bprime_counterfactual"
            / "q2b-bxor-counterfactual__iter-20__seg-3x1M__20260520T003124"
            / "level0"
            / "phi_snapshots"
            / "var_BXOR"
            / "phi_iter20.struct.gz"
        ),
    }


def validate_source_state(payload: Any, source_sha256: str) -> dict[str, Any] | None:
    try:
        if (
            not isinstance(payload, dict)
            or payload.get("source", {}).get("sha256") != source_sha256
        ):
            return None
        windows = payload.get("windows")
        if not isinstance(windows, list) or len(windows) != 3:
            return None
        source = payload["source"]
        expected_specs = derive_site_specs(
            source["source"], int(source["observable_length"])
        )
        expected_geometry = {
            spec.q: (spec.start, spec.end, spec.observable_length)
            for spec in expected_specs
        }
        for row in windows:
            if row.get("site_q") not in expected_geometry or expected_geometry[
                row["site_q"]
            ] != (
                row.get("site_start"),
                row.get("site_end"),
                row.get("source_observable_length"),
            ):
                return None
            identity = {
                key: row[key]
                for key in (
                    "cache_kind",
                    "source",
                    "site_q",
                    "site_start",
                    "site_end",
                    "source_path",
                    "source_sha256",
                    "source_observable_length",
                    "preregistration_sha256",
                )
            }
            if (
                validate_identity_cache(Path(row["cache_path"]), identity=identity)
                is None
            ):
                return None
        return payload
    except (KeyError, TypeError, ValueError):
        return None


def build_null_specs(
    observed_rows: list[dict[str, Any]], run_dir: Path
) -> list[dict[str, Any]]:
    specs = []
    for row in sorted(
        observed_rows, key=lambda item: (item["source"], site_order(item["site_q"]))
    ):
        for family in NULL_FAMILIES:
            for seed in SEEDS:
                job_id = (
                    f"{row['source'].lower()}_q{row['site_q'].replace('/', 'of')}__"
                    f"{family}__seed-{seed}"
                )
                specs.append(
                    {
                        "job_id": job_id,
                        "source": row["source"],
                        "q": row["site_q"],
                        "site_start": row["site_start"],
                        "family": family,
                        "seed": seed,
                        "observed_cache_path": row["cache_path"],
                        "observed_cache_sha256": row["cache_sha256"],
                        "cache_path": str(
                            (
                                run_dir
                                / "caches"
                                / "nulls"
                                / row["source"].lower()
                                / f"q{row['site_q'].replace('/', 'of')}"
                                / family
                                / f"seed-{seed}.npy"
                            ).resolve()
                        ),
                    }
                )
    return specs


def run_cache_jobs(
    specs: list[dict[str, Any]], workers: int, logger: RunLogger
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    preflight_ids = {
        "b_q1of10__markov1__seed-17",
        "b_q1of10__phase-matched-lz__seed-17",
    }
    preflight_specs = [spec for spec in specs if spec["job_id"] in preflight_ids]
    preflight_records = run_parallel_jobs(
        preflight_specs,
        generate_null_cache_job,
        workers=1,
        logger=logger,
        phase="stage0-null-preflight",
    )
    max_rss = max(int(record["peak_rss_bytes"]) for record in preflight_records)
    admission = max(1, int(RAM_BUDGET_BYTES // max(1, int(1.5 * max_rss))))
    effective_workers = max(1, min(4, workers, admission))
    preflight = {
        "job_ids": sorted(preflight_ids),
        "max_peak_rss_bytes": max_rss,
        "ram_budget_bytes": RAM_BUDGET_BYTES,
        "safety_factor": 1.5,
        "effective_workers": effective_workers,
    }
    logger.emit("stage0-null-preflight-completed", **preflight)
    records = {record["job_id"]: record for record in preflight_records}
    remaining = [spec for spec in specs if spec["job_id"] not in records]
    for record in run_parallel_jobs(
        remaining,
        generate_null_cache_job,
        workers=effective_workers,
        logger=logger,
        phase="stage0-null-caches",
        initial_completed=len(records),
        total_override=len(specs),
    ):
        records[record["job_id"]] = record
    return [records[spec["job_id"]] for spec in specs], preflight


def build_profile_specs(
    stage0: dict[str, Any],
    stage0_sha256: str,
    code_hashes: dict[str, str],
    run_dir: Path,
) -> list[dict[str, Any]]:
    specs = []
    cache_rows = []
    for row in stage0["observed_caches"]:
        cache_rows.append(
            {
                "source": row["source"],
                "q": row["site_q"],
                "kind": "observed",
                "family": None,
                "seed": None,
                "cache_path": row["cache_path"],
                "cache_sha256": row["cache_sha256"],
            }
        )
    for row in stage0["null_caches"]:
        cache_rows.append(
            {
                "source": row["source"],
                "q": row["site_q"],
                "kind": "null",
                "family": row["null_family"],
                "seed": row["null_seed"],
                "cache_path": row["cache_path"],
                "cache_sha256": row["cache_sha256"],
            }
        )
    for row in sorted(
        cache_rows,
        key=lambda item: (
            item["source"],
            site_order(item["q"]),
            0 if item["kind"] == "observed" else 1,
            item["family"] or "",
            item["seed"] or -1,
        ),
    ):
        job_id = (
            f"{row['source'].lower()}_q{row['q'].replace('/', 'of')}__{row['kind']}"
        )
        if row["kind"] == "null":
            job_id = f"{row['source'].lower()}_q{row['q'].replace('/', 'of')}__{row['family']}__seed-{row['seed']}"
        spec = {**row, "job_id": job_id}
        spec["identity"] = profile_job_identity(
            spec, code_hashes=code_hashes, stage0_sha256=stage0_sha256
        )
        spec["job_dir"] = str((run_dir / "profile_jobs" / job_id).resolve())
        specs.append(spec)
    if len(specs) != 78:
        raise RuntimeError(f"Expected 78 Stage 1 jobs, got {len(specs)}.")
    return specs


def run_parallel_jobs(
    specs: list[dict[str, Any]],
    worker: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    workers: int,
    logger: RunLogger,
    phase: str,
    on_record: Callable[[dict[str, Any]], None] | None = None,
    initial_completed: int = 0,
    total_override: int | None = None,
) -> list[dict[str, Any]]:
    if not specs:
        return []
    total = total_override or (initial_completed + len(specs))
    started = time.monotonic()
    records = []
    executor = ProcessPoolExecutor(max_workers=workers)
    futures = {executor.submit(worker, spec): spec for spec in specs}
    try:
        while futures:
            done, _ = wait(futures, timeout=1.0, return_when=FIRST_COMPLETED)
            for future in done:
                spec = futures.pop(future)
                record = future.result()
                if record.get("status") != "completed":
                    raise RuntimeError(f"Worker failed: {spec['job_id']}")
                records.append(record)
                if on_record is not None:
                    on_record(record)
                logger.emit(
                    "job-completed",
                    job_id=record["job_id"],
                    jobs_completed=f"{initial_completed + len(records)}/{total}",
                    reused=record.get("reused", False),
                )
            elapsed = time.monotonic() - started
            completed = initial_completed + len(records)
            rate = len(records) / elapsed if elapsed > 0 else 0.0
            eta = len(futures) / rate if rate > 0 else None
            logger.emit(
                "progress",
                force=False,
                phase=phase,
                jobs_completed=f"{completed}/{total}",
                workers_active=min(workers, len(futures)),
                elapsed_seconds=round(elapsed, 3),
                rate_jobs_per_second=round(rate, 6),
                eta_seconds=None if eta is None else round(eta, 3),
            )
    except BaseException:
        for future in futures:
            future.cancel()
        for process in getattr(executor, "_processes", {}).values():
            if process.is_alive():
                process.terminate()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    executor.shutdown(wait=True)
    return records


def validate_stage0_manifest(payload: dict[str, Any], stage0_dir: Path) -> None:
    if (
        payload.get("status") != "completed"
        or payload.get("outcome") != "inputs-derivable"
    ):
        raise RuntimeError(
            "Stage 0 manifest is not a completed inputs-derivable closure."
        )
    if payload.get("preregistration", {}).get("sha256") != PREREG_SHA256:
        raise RuntimeError("Stage 0 manifest points to a different preregistration.")
    if payload.get("scientific_values_emitted") != []:
        raise RuntimeError("Stage 0 anti-observation declaration is not empty.")
    for relative, expected in payload.get("artifact_hashes", {}).items():
        path = stage0_dir / relative
        if not path.exists() or sha256_file(path) != expected:
            raise RuntimeError(f"Stage 0 artifact hash mismatch: {relative}")
    if (
        len(payload.get("observed_caches", [])) != 6
        or len(payload.get("null_caches", [])) != 72
    ):
        raise RuntimeError(
            "Stage 0 cache cardinality differs from the frozen 6 + 72 matrix."
        )
    for row in payload["observed_caches"]:
        identity = {
            key: row[key]
            for key in (
                "cache_kind",
                "source",
                "site_q",
                "site_start",
                "site_end",
                "source_path",
                "source_sha256",
                "source_observable_length",
                "preregistration_sha256",
            )
        }
        if validate_identity_cache(Path(row["cache_path"]), identity=identity) is None:
            raise RuntimeError(f"Observed cache identity mismatch: {row['cache_path']}")
    for row in payload["null_caches"]:
        spec = {
            "source": row["source"],
            "q": row["site_q"],
            "site_start": row["site_start"],
            "family": row["null_family"],
            "seed": row["null_seed"],
            "observed_cache_sha256": row["observed_cache_sha256"],
        }
        sidecar = validate_identity_cache(
            Path(row["cache_path"]), identity=null_cache_identity(spec)
        )
        if sidecar is None or not valid_fit_metadata(sidecar, row["null_family"]):
            raise RuntimeError(f"Null cache identity mismatch: {row['cache_path']}")


def ordered_records(
    jobs: list[dict[str, Any]], records: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    return [records[spec["job_id"]] for spec in jobs if spec["job_id"] in records]


def write_stage1_outputs(
    run_dir: Path,
    summary: dict[str, Any],
    readout: dict[str, Any],
    records: list[dict[str, Any]],
) -> None:
    write_json_atomic(run_dir / "summary.json", summary)
    write_report(run_dir / "report.md", summary)
    write_csv_atomic(
        run_dir / "per_site_probe_readout.csv",
        flatten_site_rows(readout["per_site"]),
        [
            "q",
            "probe1_D_obs_median",
            "probe1_D_obs_ci95_low",
            "probe1_D_obs_ci95_high",
            "probe1_G_median",
            "probe1_G_ci95_low",
            "probe1_G_ci95_high",
            "probe1_raw_positive",
            "probe1_hard_null_positive",
            "probe2_D_obs_median",
            "probe2_D_obs_ci95_low",
            "probe2_D_obs_ci95_high",
            "probe2_G_median",
            "probe2_G_ci95_low",
            "probe2_G_ci95_high",
            "probe2_raw_positive",
            "probe2_hard_null_positive",
            "D3_median",
        ],
    )
    segment_rows = []
    envelope_rows = []
    for row in readout["per_segment"]:
        segment_rows.append(
            {
                key: value
                for key, value in row.items()
                if not key.endswith("_null_pairs")
            }
        )
        for probe in (1, 2):
            for label, value in sorted(row[f"D{probe}_null_pairs"].items()):
                family, seed = label.rsplit("__seed-", 1)
                envelope_rows.append(
                    {
                        "q": row["q"],
                        "segment_index": row["segment_index"],
                        "probe": probe,
                        "family": family,
                        "seed": int(seed),
                        "D_null_pair": value,
                        "E": row[f"E{probe}"],
                        "D_obs": row[f"D{probe}_obs"],
                        "G": row[f"G{probe}"],
                    }
                )
    write_csv_atomic(
        run_dir / "per_segment_probe_readout.csv",
        segment_rows,
        [
            "site_index",
            "q",
            "segment_index",
            "D1_obs",
            "D2_obs",
            "D3",
            "E1",
            "G1",
            "E2",
            "G2",
        ],
    )
    write_csv_atomic(
        run_dir / "null_envelope_readout.csv",
        envelope_rows,
        [
            "q",
            "segment_index",
            "probe",
            "family",
            "seed",
            "D_null_pair",
            "E",
            "D_obs",
            "G",
        ],
    )
    write_json_atomic(run_dir / "fiber_kernel_profiles.json", readout["fiber_archive"])
    write_json_atomic(
        run_dir / "extension_motif_profiles.json", readout["motif_archive"]
    )
    write_json_atomic(
        run_dir / "routing_residual_profiles.json", readout["routing_archive"]
    )
    write_json_atomic(run_dir / "job_status.json", records)


def required_output_names() -> tuple[str, ...]:
    return (
        "manifest.json",
        "summary.json",
        "report.md",
        "per_site_probe_readout.csv",
        "per_segment_probe_readout.csv",
        "fiber_kernel_profiles.json",
        "extension_motif_profiles.json",
        "routing_residual_profiles.json",
        "null_envelope_readout.csv",
        "job_status.json",
    )


def flatten_site_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        flat: dict[str, Any] = {"q": row["q"], "D3_median": row["D3_median"]}
        for probe in (1, 2):
            values = row[f"probe{probe}"]
            flat.update(
                {
                    f"probe{probe}_D_obs_median": values["D_obs_median"],
                    f"probe{probe}_D_obs_ci95_low": values["D_obs_ci95"][0],
                    f"probe{probe}_D_obs_ci95_high": values["D_obs_ci95"][1],
                    f"probe{probe}_G_median": values["G_median"],
                    f"probe{probe}_G_ci95_low": values["G_ci95"][0],
                    f"probe{probe}_G_ci95_high": values["G_ci95"][1],
                    f"probe{probe}_raw_positive": values["raw_positive"],
                    f"probe{probe}_hard_null_positive": values["hard_null_positive"],
                }
            )
        output.append(flat)
    return output


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# D-0160 Item 3 Rule-Local Discriminant Readout",
        "",
        f"- Status: `{summary['status']}`",
        f"- Outcome: `{summary['outcome']}`",
        f"- Probe 1 region-positive: `{summary['region_positive']['probe1']}`",
        f"- Probe 2 region-positive: `{summary['region_positive']['probe2']}`",
        "",
        "## Site Readout",
        "",
        "| q | P1 Dobs | P1 G | P1 hard+ | P2 Dobs | P2 G | P2 hard+ | D3 |",
        "| ---: | ---: | ---: | --- | ---: | ---: | --- | ---: |",
    ]
    for row in summary["per_site"]:
        p1 = row["probe1"]
        p2 = row["probe2"]
        lines.append(
            f"| {row['q']} | {p1['D_obs_median']:.6f} | {p1['G_median']:.6f} | "
            f"{p1['hard_null_positive']} | {p2['D_obs_median']:.6f} | {p2['G_median']:.6f} | "
            f"{p2['hard_null_positive']} | {row['D3_median']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            summary["bootstrap_boundary"],
            "",
            summary["wording_boundary"],
            "",
            "This finite-depth readout does not establish P3 uniquely, rank B above B_XOR, "
            "or authorize Gate 2, geometry, or a universal law.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_stage0_report(
    path: Path,
    sources: list[dict[str, Any]],
    observed: list[dict[str, Any]],
    nulls: list[dict[str, Any]],
    preflight: dict[str, Any],
    peak_rss: int,
) -> None:
    lines = [
        "# D-0160 Item 3 Stage 0",
        "",
        "- Outcome: `inputs-derivable`",
        f"- Frozen sources verified: `{len(sources)}/2`",
        f"- Observed caches frozen: `{len(observed)}/6`",
        f"- Mandatory null caches frozen: `{len(nulls)}/72`",
        f"- Effective workers after preflight: `{preflight['effective_workers']}`",
        f"- Peak RSS: `{peak_rss}` bytes",
        "- Scientific values emitted: `none`",
        "",
        "Stage 1 remains blocked pending independent manifest review and explicit authorization.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def hash_artifacts(root: Path, *, exclude: set[str]) -> dict[str, str]:
    hashes = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in exclude or any(
            part == "quarantine" for part in path.relative_to(root).parts
        ):
            continue
        hashes[relative] = sha256_file(path)
    return hashes


def current_code_hashes() -> dict[str, str]:
    paths = {
        "runner": Path(__file__).resolve(),
        "item3_core": REPO_DIR
        / "v2"
        / "postreview"
        / "item3_rule_local_discriminants.py",
        "item2b_streaming_core": REPO_DIR
        / "v2"
        / "postreview"
        / "item2b_growth_shells.py",
        "io": REPO_DIR / "v2" / "common" / "io.py",
        "null_models": NULL_MODELS_PATH,
        "phase1_tower": REPO_DIR / "v2" / "phase1" / "tower.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def write_failure(run_dir: Path, stage: str, audit_sha256: str, exc: Exception) -> None:
    name = "stage0_manifest.json" if stage == "stage0" else "manifest.json"
    path = run_dir / name
    payload = load_json(path, default={})
    payload.update(
        {
            "status": "failed",
            "stage": stage,
            "outcome": "inputs-or-run-not-derivable",
            "error": repr(exc),
            "preregistration_sha256": PREREG_SHA256,
            "code_audit_sha256": audit_sha256.upper(),
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    write_json_atomic(path, payload)
    (run_dir / "failure_report.md").write_text(
        f"# D-0160 Item 3 Failure\n\n- Stage: `{stage}`\n- Error: `{exc!r}`\n",
        encoding="utf-8",
    )


def load_json(path: Path, *, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return default


def site_order(q: str) -> int:
    order = {item[0]: index for index, item in enumerate(SITE_FRACTIONS)}
    return order[q]


def verify_preregistration() -> None:
    if not PREREG_PATH.exists():
        raise SystemExit(f"Frozen preregistration not found: {PREREG_PATH}")
    actual = sha256_file(PREREG_PATH)
    if actual != PREREG_SHA256:
        raise SystemExit(
            f"Frozen preregistration hash mismatch: expected {PREREG_SHA256}, got {actual}."
        )


def validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.raw_chunk_mib <= 16:
        raise SystemExit("--raw-chunk-mib must be in [1,16].")
    if not 1 <= args.workers <= 4:
        raise SystemExit("--workers must be in [1,4].")
    validate_sha256_arg("--code-audit-sha256", args.code_audit_sha256)
    if args.stage == "stage0":
        if (
            args.stage0_manifest
            or args.stage0_manifest_sha256
            or args.stage1_authorization_sha256
        ):
            raise SystemExit("Stage 0 does not accept Stage 1 gate arguments.")
    else:
        if not args.stage0_manifest:
            raise SystemExit("Stage 1 requires --stage0-manifest.")
        validate_sha256_arg("--stage0-manifest-sha256", args.stage0_manifest_sha256)
        validate_sha256_arg(
            "--stage1-authorization-sha256", args.stage1_authorization_sha256
        )


def validate_sha256_arg(name: str, value: str | None) -> None:
    if (
        value is None
        or len(value) != 64
        or any(char not in "0123456789abcdefABCDEF" for char in value)
    ):
        raise SystemExit(f"{name} must be a 64-character SHA-256 value.")


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)


def load_dotenv() -> None:
    path = REPO_DIR / ".env"
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"'))


def resolve_results_base(raw: str | None) -> Path:
    value = raw or os.environ.get("HSI_RESULTS_BASE_DIR")
    if not value:
        return (REPO_DIR / "results").resolve()
    return Path(value).expanduser().resolve()


def resolve_results_path(raw: str, results_base: Path) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    normalized = path.as_posix()
    if normalized == "results":
        return results_base
    if normalized.startswith("results/"):
        return (results_base / normalized[len("results/") :]).resolve()
    return (REPO_DIR / path).resolve()


def resolve_run_dir(args: argparse.Namespace, output_root: Path) -> Path:
    if args.run_dir:
        return Path(args.run_dir).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    return output_root / f"item3-rule-local-{args.stage}__{timestamp}"


def print_plan(args: argparse.Namespace, results_base: Path, run_dir: Path) -> None:
    print("D-0160 Item 3 dry run")
    print(f"Stage: {args.stage}")
    print(f"Results base: {results_base}")
    print(f"Run directory: {run_dir}")
    print(f"Workers requested: {args.workers}")
    print(
        "No source window was read. Add --execute only after the applicable gate is anchored."
    )


if __name__ == "__main__":
    raise SystemExit(main())
