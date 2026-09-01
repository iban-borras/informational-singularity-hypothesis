"""Run frozen D-0161 Item 4 Stage 0/1 with bounded parallelism and resume."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

try:
    import psutil
except ImportError:  # pragma: no cover - execution rejects this environment.
    psutil = None  # type: ignore[assignment]

from v2.postreview.item2b_growth_shells import (
    RunLogger,
    sha256_file,
    sha256_file_progress,
    verify_compiled_kernels,
    write_csv_atomic,
    write_json_atomic,
)
from v2.postreview.item4_multiseed_monte_carlo import (
    BASELINE_DISPLAY,
    CLEAN_BASELINE_FULL,
    CLEAN_BASELINE_PROVENANCE,
    NULL_FAMILIES,
    NULL_IMPLEMENTATION_SHA256,
    PREREG_SHA256,
    SEEDS,
    SOURCE_SPECS,
    build_profile_job,
    extract_observed_prefix,
    frozen_protocol,
    generate_null_cache_job,
    null_cache_identity,
    observed_cache_identity,
    profile_job_identity,
    source_paths,
    valid_fit_metadata,
    validate_identity_cache,
)
from v2.postreview.item4_statistics import (
    ALPHA,
    COORDINATES,
    classify_outcome,
    holm_stepdown,
    summarize_source_family,
)


REPO_DIR = Path(__file__).resolve().parent
PROJECT_DIR = REPO_DIR.parent
PREREG_PATH = (
    PROJECT_DIR / "prereg_postreview_item4_gate1_multiseed_monte_carlo_v1.0.md"
)
NULL_MODELS_PATH = REPO_DIR / "v2" / "common" / "null_models.py"
OUTPUT_RELATIVE = Path("hsi_v2") / "postreview_item4_gate1_multiseed_monte_carlo"
RAM_BUDGET_BYTES = 32 * 1024**3
MAX_WORKERS = 4
COMPLETION_LOG_EVERY = 25
MANIFEST_REFRESH_JOBS = 25
MANIFEST_REFRESH_SECONDS = 30.0


class Item4RunLogger(RunLogger):
    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self._last_human_progress = 0.0

    def emit(self, event: str, *, force: bool = True, **fields: Any) -> None:
        super().emit(event, force=force, **fields)
        if event != "progress":
            return
        now = time.monotonic()
        if not force and now - self._last_human_progress < 30.0:
            return
        self._last_human_progress = now
        completed = fields.get("jobs_completed") or fields.get("caches_completed")
        if completed is None:
            completed = fields.get("percent", "n/a")
            completed = (
                f"{completed}%" if isinstance(completed, (int, float)) else completed
            )
        rss_gib = (
            float(
                fields.get("aggregate_child_rss_bytes")
                or fields.get("peak_rss_bytes", 0)
            )
            / 1024**3
        )
        eta = fields.get("eta_seconds")
        eta_text = "n/a" if eta is None else format_duration(float(eta))
        line = (
            f"[progress] {fields.get('phase', 'work')} | {completed} | "
            f"active={fields.get('source') or fields.get('snapshot') or 'n/a'} "
            f"{fields.get('family') or ''} seed={fields.get('seed')} | "
            f"workers={fields.get('workers_active', 'n/a')} | rss={rss_gib:.2f} GiB | "
            f"elapsed={format_duration(float(fields.get('elapsed_seconds', 0.0)))} | eta={eta_text}"
        )
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write("# " + line + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("stage0", "stage1"), required=True)
    parser.add_argument("--results-base", default=None)
    parser.add_argument("--output-dir", default=str(Path("results") / OUTPUT_RELATIVE))
    parser.add_argument(
        "--run-dir", default=None, help="Existing/new run directory for strict resume."
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
        raise SystemExit("psutil is required for RSS telemetry and worker admission.")
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = Item4RunLogger(run_dir / "run.log")
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
    args: argparse.Namespace, results_base: Path, run_dir: Path, logger: RunLogger
) -> int:
    started = time.monotonic()
    kernel_state = verify_compiled_kernels()
    logger.emit("compiled-fixtures-passed", **kernel_state)
    code_hashes = current_code_hashes()
    if code_hashes["null_models"] != NULL_IMPLEMENTATION_SHA256:
        raise RuntimeError(
            "Null implementation differs from the frozen reference hash."
        )

    source_rows: list[dict[str, Any]] = []
    observed_rows: list[dict[str, Any]] = []
    paths = source_paths(results_base)
    for source in SOURCE_SPECS:
        spec = SOURCE_SPECS[source]
        path = paths[source]
        if not path.exists():
            raise FileNotFoundError(f"Frozen source not found: {path}")
        logger.emit("source-hash-start", source=source, path=str(path))
        actual_sha256 = sha256_file_progress(path, logger, label=f"source-{source}")
        if actual_sha256 != spec.sha256:
            raise RuntimeError(
                f"Source hash mismatch for {source}: expected {spec.sha256}, got {actual_sha256}."
            )
        cache_path = run_dir / "caches" / "observed" / f"{source.lower()}_prefix3m.npy"
        sidecar, telemetry = extract_observed_prefix(
            path,
            spec=spec,
            source_sha256=actual_sha256,
            cache_path=cache_path,
            raw_chunk_bytes=args.raw_chunk_mib * 1024**2,
            logger=logger,
        )
        source_rows.append(
            {
                "source": source,
                "iteration": spec.iteration,
                "path": str(path.resolve()),
                "sha256": actual_sha256,
                "observable_bits": spec.observable_bits,
                "sequence_length": telemetry.get("sequence_length"),
                "telemetry": telemetry,
            }
        )
        observed_rows.append(sidecar)
        logger.emit(
            "observed-cache-frozen", source=source, reused=sidecar.get("reused", False)
        )

    write_json_atomic(run_dir / "source_identity.json", source_rows)
    write_csv_atomic(
        run_dir / "observed_cache_inventory.csv",
        observed_rows,
        [
            "source",
            "iteration",
            "source_sha256",
            "source_observable_length",
            "window_start",
            "window_end",
            "observable_length",
            "cache_path",
            "cache_sha256",
            "reused",
        ],
    )
    null_specs = build_null_specs(observed_rows, run_dir)
    records, preflight = run_cache_jobs(null_specs, args.workers, logger)
    if len(records) != 6_132 or any(
        row.get("status") != "completed" for row in records
    ):
        raise RuntimeError("Stage 0 did not complete all 6,132 frozen null caches.")
    null_rows = [record["sidecar"] for record in records]
    write_csv_atomic(
        run_dir / "null_cache_inventory.csv",
        null_rows,
        [
            "source",
            "iteration",
            "null_family",
            "null_seed",
            "observed_cache_sha256",
            "observable_length",
            "cache_path",
            "cache_sha256",
        ],
    )
    telemetry_rows = [
        {
            "stage": "stage0-source",
            "job_id": f"source-{row['source']}",
            "elapsed_seconds": row["telemetry"].get("elapsed_seconds", 0.0),
            "peak_rss_bytes": row["telemetry"].get("peak_rss_bytes", 0),
            "bytes_written": 0,
            "reused": row["telemetry"].get("reused", False),
        }
        for row in source_rows
    ] + [
        {
            "stage": "stage0",
            "job_id": record["job_id"],
            "elapsed_seconds": record["elapsed_seconds"],
            "peak_rss_bytes": record["peak_rss_bytes"],
            "bytes_written": record.get("bytes_written", 0),
            "reused": record.get("reused", False),
        }
        for record in records
    ]
    write_csv_atomic(
        run_dir / "resource_telemetry.csv",
        telemetry_rows,
        [
            "stage",
            "job_id",
            "elapsed_seconds",
            "peak_rss_bytes",
            "bytes_written",
            "reused",
        ],
    )
    write_stage0_report(
        run_dir / "stage0_report.md", source_rows, observed_rows, null_rows, preflight
    )

    manifest_path = run_dir / "stage0_manifest.json"
    manifest = {
        "status": "completed",
        "stage": "stage0",
        "outcome": "inputs-derivable",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir.resolve()),
        "preregistration": {"path": str(PREREG_PATH), "sha256": PREREG_SHA256},
        "code_audit_sha256": args.code_audit_sha256.upper(),
        "code_hashes": code_hashes,
        "kernel_state": kernel_state,
        "protocol": frozen_protocol(),
        "sources": source_rows,
        "observed_caches": observed_rows,
        "null_caches": null_rows,
        "resource_preflight": preflight,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "scientific_values_emitted": [],
    }
    manifest["artifact_hashes"] = stage0_artifact_hashes(
        run_dir, observed_rows, null_rows
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
            "Stage 0 manifest hash differs from the authorization input."
        )
    stage0 = json.loads(stage0_path.read_text(encoding="utf-8"))
    validate_stage0_manifest(stage0, stage0_path.parent)
    code_hashes = current_code_hashes()
    if stage0.get("code_hashes") != code_hashes:
        raise RuntimeError("Stage 0 and Stage 1 implementation hashes differ.")

    jobs = build_profile_specs(stage0, stage0_sha256, code_hashes, run_dir)
    manifest_path = run_dir / "manifest.json"
    manifest = {
        "status": "running",
        "stage": "stage1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir.resolve()),
        "preregistration": {"path": str(PREREG_PATH), "sha256": PREREG_SHA256},
        "code_audit_sha256": args.code_audit_sha256.upper(),
        "stage1_authorization_sha256": args.stage1_authorization_sha256.upper(),
        "stage0_manifest": str(stage0_path),
        "stage0_manifest_sha256": stage0_sha256,
        "code_hashes": code_hashes,
        "kernel_state": kernel_state,
        "protocol": frozen_protocol(),
        "requested_workers": args.workers,
        "jobs_completed": 0,
        "job_status_root": str((run_dir / "profile_jobs").resolve()),
    }
    write_json_atomic(manifest_path, manifest)

    preflight_ids = {
        "b__observed",
        "b__markov1__seed-1",
        "b__matched-lz__seed-1",
        "b__phase-matched-lz__seed-1",
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
    effective_workers = max(1, min(MAX_WORKERS, args.workers, admission))
    preflight = {
        "job_ids": sorted(preflight_ids),
        "max_peak_rss_bytes": max_preflight_rss,
        "ram_budget_bytes": RAM_BUDGET_BYTES,
        "safety_factor": 1.5,
        "effective_workers": effective_workers,
    }
    logger.emit("stage1-preflight-completed", **preflight)
    records = {record["job_id"]: record for record in preflight_records}
    checkpoint = ManifestCheckpoint(manifest_path, manifest, total=len(jobs))
    checkpoint.update(records, force=True)
    remaining = [spec for spec in jobs if spec["job_id"] not in records]

    def on_record(record: dict[str, Any]) -> None:
        records[record["job_id"]] = record
        checkpoint.update(records)

    for record in run_parallel_jobs(
        remaining,
        build_profile_job,
        workers=effective_workers,
        logger=logger,
        phase="stage1-profile-jobs",
        on_record=on_record,
        initial_completed=len(records),
        total_override=len(jobs),
    ):
        records[record["job_id"]] = record
    ordered = [records[spec["job_id"]] for spec in jobs if spec["job_id"] in records]
    if len(ordered) != 6_136 or any(
        row.get("status") != "completed" for row in ordered
    ):
        raise RuntimeError("Stage 1 did not complete all 6,136 frozen profile jobs.")

    readout = build_readout(ordered)
    write_stage1_outputs(run_dir, readout, ordered)
    telemetry_rows = [
        {
            "stage": "stage1",
            "job_id": row["job_id"],
            "elapsed_seconds": row["elapsed_seconds"],
            "peak_rss_bytes": row["peak_rss_bytes"],
            "reused": row.get("reused", False),
        }
        for row in ordered
    ]
    write_csv_atomic(
        run_dir / "resource_telemetry.csv",
        telemetry_rows,
        ["stage", "job_id", "elapsed_seconds", "peak_rss_bytes", "reused"],
    )
    summary = {
        **readout["summary"],
        "protocol": frozen_protocol(),
        "sources": stage0["sources"],
        "preregistration_sha256": PREREG_SHA256,
        "code_audit_sha256": args.code_audit_sha256.upper(),
        "stage0_manifest_sha256": stage0_sha256,
        "stage1_authorization_sha256": args.stage1_authorization_sha256.upper(),
        "implementation_hashes": code_hashes,
        "resource_preflight": preflight,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "resume_history": {
            "reused_jobs": sum(bool(row.get("reused")) for row in ordered),
            "fresh_jobs": sum(not bool(row.get("reused")) for row in ordered),
        },
    }
    write_json_atomic(run_dir / "summary.json", summary)
    write_report(run_dir / "report.md", summary)
    final_manifest = {
        **manifest,
        "status": "completed",
        "outcome": summary["outcome"],
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": summary["elapsed_seconds"],
        "resource_preflight": preflight,
        "jobs_completed": len(ordered),
        "jobs": ordered,
    }
    final_manifest["artifact_hashes"] = hash_artifacts(
        run_dir, exclude={"run.log", "manifest.json", "manifest.sha256"}
    )
    write_json_atomic(manifest_path, final_manifest)
    manifest_sha256 = sha256_file(manifest_path)
    (run_dir / "manifest.sha256").write_text(
        f"{manifest_sha256}  manifest.json\n", encoding="ascii"
    )
    logger.emit(
        "stage1-completed", outcome=summary["outcome"], manifest_sha256=manifest_sha256
    )
    print("D-0161 Item 4 Stage 1 completed.")
    print(f"Outcome: {summary['outcome']}")
    print(f"Output: {run_dir}")
    print(f"Manifest SHA-256: {manifest_sha256}")
    return 0


class ManifestCheckpoint:
    def __init__(self, path: Path, payload: dict[str, Any], *, total: int) -> None:
        self.path = path
        self.payload = payload
        self.total = total
        self.last_count = 0
        self.last_write = time.monotonic()

    def update(
        self, records: dict[str, dict[str, Any]], *, force: bool = False
    ) -> None:
        count = len(records)
        now = time.monotonic()
        if (
            not force
            and count - self.last_count < MANIFEST_REFRESH_JOBS
            and now - self.last_write < MANIFEST_REFRESH_SECONDS
        ):
            return
        self.payload["jobs_completed"] = count
        self.payload["jobs_total"] = self.total
        self.payload["last_completed_job_id"] = next(reversed(records), None)
        self.payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
        write_json_atomic(self.path, self.payload)
        self.last_count = count
        self.last_write = now


def build_null_specs(
    observed_rows: list[dict[str, Any]], run_dir: Path
) -> list[dict[str, Any]]:
    by_source = {row["source"]: row for row in observed_rows}
    specs: list[dict[str, Any]] = []
    for source, source_spec in SOURCE_SPECS.items():
        observed = by_source[source]
        for family in NULL_FAMILIES:
            for seed in SEEDS:
                job_id = f"{source.lower()}__{family}__seed-{seed}"
                specs.append(
                    {
                        "job_id": job_id,
                        "source": source,
                        "iteration": source_spec.iteration,
                        "family": family,
                        "seed": seed,
                        "observed_cache_path": observed["cache_path"],
                        "observed_cache_sha256": observed["cache_sha256"],
                        "cache_path": str(
                            (
                                run_dir
                                / "caches"
                                / "nulls"
                                / source.lower()
                                / family
                                / f"seed-{seed}.npy"
                            ).resolve()
                        ),
                    }
                )
    if len(specs) != 6_132:
        raise RuntimeError(f"Expected 6,132 null cache specs, got {len(specs)}.")
    return specs


def run_cache_jobs(
    specs: list[dict[str, Any]], workers: int, logger: RunLogger
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    preflight_ids = {
        "b__markov1__seed-1",
        "b__matched-lz__seed-1",
        "b__phase-matched-lz__seed-1",
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
    effective_workers = max(1, min(MAX_WORKERS, workers, admission))
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
    cache_rows: list[dict[str, Any]] = []
    for row in stage0["observed_caches"]:
        cache_rows.append(
            {
                "source": row["source"],
                "iteration": int(row["iteration"]),
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
                "iteration": int(row["iteration"]),
                "kind": "null",
                "family": row["null_family"],
                "seed": int(row["null_seed"]),
                "cache_path": row["cache_path"],
                "cache_sha256": row["cache_sha256"],
            }
        )
    specs = []
    for row in sorted(
        cache_rows,
        key=lambda item: (
            tuple(SOURCE_SPECS).index(item["source"]),
            0 if item["kind"] == "observed" else 1,
            "" if item["family"] is None else NULL_FAMILIES.index(item["family"]),
            -1 if item["seed"] is None else item["seed"],
        ),
    ):
        job_id = f"{row['source'].lower()}__observed"
        if row["kind"] == "null":
            job_id = f"{row['source'].lower()}__{row['family']}__seed-{row['seed']}"
        spec = {**row, "job_id": job_id}
        spec["identity"] = profile_job_identity(
            spec, code_hashes=code_hashes, stage0_sha256=stage0_sha256
        )
        spec["job_dir"] = str((run_dir / "profile_jobs" / job_id).resolve())
        specs.append(spec)
    if len(specs) != 6_136:
        raise RuntimeError(f"Expected 6,136 Stage 1 specs, got {len(specs)}.")
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
    total = total_override or initial_completed + len(specs)
    started = time.monotonic()
    pending_index = 0
    records: list[dict[str, Any]] = []
    bytes_written_total = 0
    futures: dict[Future, dict[str, Any]] = {}
    executor = ProcessPoolExecutor(max_workers=workers)

    def submit_available() -> None:
        nonlocal pending_index
        while pending_index < len(specs) and len(futures) < workers:
            child_rss = aggregate_child_rss_bytes()
            if futures and child_rss >= RAM_BUDGET_BYTES:
                break
            spec = specs[pending_index]
            pending_index += 1
            futures[executor.submit(worker, spec)] = spec

    submit_available()
    try:
        while futures:
            done, _ = wait(futures, timeout=1.0, return_when=FIRST_COMPLETED)
            for future in done:
                spec = futures.pop(future)
                record = future.result()
                if record.get("status") != "completed":
                    raise RuntimeError(f"Worker failed: {spec['job_id']}")
                records.append(record)
                bytes_written_total += int(record.get("bytes_written", 0))
                if on_record is not None:
                    on_record(record)
                completed_count = initial_completed + len(records)
                if (
                    len(records) <= workers
                    or completed_count % COMPLETION_LOG_EVERY == 0
                    or completed_count == total
                ):
                    logger.emit(
                        "job-batch-completed",
                        job_id=record["job_id"],
                        source=record.get("source", spec.get("source")),
                        family=record.get("family", spec.get("family")),
                        seed=record.get("seed", spec.get("seed")),
                        jobs_completed=f"{completed_count}/{total}",
                        reused=record.get("reused", False),
                    )
            submit_available()
            elapsed = max(time.monotonic() - started, 1e-9)
            rate = len(records) / elapsed
            unfinished = len(specs) - len(records)
            active_spec = next(iter(futures.values()), records[-1] if records else {})
            logger.emit(
                "progress",
                force=False,
                phase=phase,
                source=active_spec.get("source"),
                family=active_spec.get("family"),
                seed=active_spec.get("seed"),
                scale_or_subphase=(
                    "phase1-profile" if phase.startswith("stage1") else "null-cache"
                ),
                caches_completed=(
                    f"{initial_completed + len(records)}/{total}"
                    if phase.startswith("stage0")
                    else None
                ),
                jobs_completed=f"{initial_completed + len(records)}/{total}",
                workers_active=min(workers, len(futures)),
                aggregate_child_rss_bytes=aggregate_child_rss_bytes(),
                bytes_written=bytes_written_total,
                elapsed_seconds=round(elapsed, 3),
                rate_jobs_per_second=round(rate, 6),
                eta_seconds=None if rate <= 0 else round(unfinished / rate, 3),
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


def aggregate_child_rss_bytes() -> int:
    if psutil is None:
        return 0
    process = psutil.Process(os.getpid())
    total = 0
    for child in process.children(recursive=True):
        try:
            total += int(child.memory_info().rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


def build_readout(records: list[dict[str, Any]]) -> dict[str, Any]:
    observed = {
        row["source"]: row["tail_vector"]
        for row in records
        if row["kind"] == "observed"
    }
    nulls: dict[tuple[str, str], list[dict[str, Any]]] = {
        (source, family): [] for source in SOURCE_SPECS for family in NULL_FAMILIES
    }
    for row in records:
        if row["kind"] == "null":
            nulls[(row["source"], row["family"])].append(
                {"seed": row["seed"], **row["tail_vector"]}
            )
    degenerate = len(observed) != len(SOURCE_SPECS)
    baseline_rows = []
    for source in SOURCE_SPECS:
        vector = observed.get(source)
        displays = (
            {}
            if vector is None
            else {key: format(float(vector[key]), ".4f") for key in COORDINATES}
        )
        expected = BASELINE_DISPLAY[source]
        clean = CLEAN_BASELINE_FULL[source]
        baseline_rows.append(
            {
                "source": source,
                **({key: None for key in COORDINATES} if vector is None else vector),
                "clean_reference_R": clean["R"],
                "clean_reference_S": clean["S"],
                "clean_reference_A": clean["A"],
                "delta_R": None if vector is None else float(vector["R"] - clean["R"]),
                "delta_S": None if vector is None else float(vector["S"] - clean["S"]),
                "delta_A": None if vector is None else float(vector["A"] - clean["A"]),
                "display_R": displays.get("R"),
                "display_S": displays.get("S"),
                "display_A": displays.get("A"),
                "expected_R": expected["R"],
                "expected_S": expected["S"],
                "expected_A": expected["A"],
                "reproduced": displays == expected,
            }
        )
    baseline_reproduced = all(row["reproduced"] for row in baseline_rows)
    pair_rows: list[dict[str, Any]] = []
    if not degenerate:
        try:
            for source in SOURCE_SPECS:
                for family in NULL_FAMILIES:
                    pair_rows.append(
                        summarize_source_family(
                            source, family, observed[source], nulls[(source, family)]
                        )
                    )
        except (KeyError, TypeError, ValueError):
            degenerate = True
    holm = [] if degenerate else holm_stepdown(pair_rows, alpha=ALPHA)
    decision = classify_outcome(
        derivable=True,
        degenerate=degenerate,
        baseline_reproduced=baseline_reproduced,
        holm_trace=holm,
        sources=SOURCE_SPECS,
        families=NULL_FAMILIES,
    )
    summary = {
        "status": "completed",
        **decision,
        "baseline_reproduced": baseline_reproduced,
        "observed_tail_vectors": baseline_rows,
        "clean_baseline_provenance": CLEAN_BASELINE_PROVENANCE,
        "source_family_tests": pair_rows,
        "holm_stepdown_trace": holm,
        "conditional_inference_boundary": (
            "The p-values are one-sided plug-in Monte Carlo exceedance values conditional "
            "on each fitted generator and finite source window; they are not exact permutation "
            "p-values and do not integrate parameter-estimation uncertainty."
        ),
    }
    return {
        "summary": summary,
        "observed": baseline_rows,
        "nulls": nulls,
        "pairs": pair_rows,
        "holm": holm,
    }


def write_stage1_outputs(
    run_dir: Path, readout: dict[str, Any], records: list[dict[str, Any]]
) -> None:
    write_csv_atomic(
        run_dir / "observed_tail_vectors.csv",
        readout["observed"],
        [
            "source",
            "R",
            "S",
            "A",
            "clean_reference_R",
            "clean_reference_S",
            "clean_reference_A",
            "delta_R",
            "delta_S",
            "delta_A",
            "display_R",
            "display_S",
            "display_A",
            "expected_R",
            "expected_S",
            "expected_A",
            "reproduced",
        ],
    )
    null_rows = []
    for (source, family), rows in readout["nulls"].items():
        null_rows.extend({"source": source, "family": family, **row} for row in rows)
    write_csv_atomic(
        run_dir / "per_null_tail_vectors.csv",
        null_rows,
        ["source", "family", "seed", "R", "S", "A"],
    )
    pair_rows = []
    dispersion_rows = []
    for row in readout["pairs"]:
        pair_rows.append(
            {
                "source": row["source"],
                "family": row["family"],
                "null_count": row["null_count"],
                "C": row["pareto_exceedance_count"],
                "p_value": row["p_value"],
                "rank_resolution": row["rank_resolution"],
                "cp95_low": row["clopper_pearson_95"]["lower"],
                "cp95_high": row["clopper_pearson_95"]["upper"],
                "historical_C": row["historical_six_seed_subset"][
                    "pareto_exceedance_count"
                ],
                "historical_p": row["historical_six_seed_subset"]["p_value"],
                "percentile_R": row["observed_percentile_ranks"]["R"],
                "percentile_S": row["observed_percentile_ranks"]["S"],
                "percentile_A": row["observed_percentile_ranks"]["A"],
            }
        )
        for coordinate in COORDINATES:
            dispersion_rows.append(
                {
                    "source": row["source"],
                    "family": row["family"],
                    "coordinate": coordinate,
                    **row["coordinate_dispersion"][coordinate],
                }
            )
    write_csv_atomic(
        run_dir / "source_family_exceedance.csv",
        pair_rows,
        [
            "source",
            "family",
            "null_count",
            "C",
            "p_value",
            "rank_resolution",
            "cp95_low",
            "cp95_high",
            "historical_C",
            "historical_p",
            "percentile_R",
            "percentile_S",
            "percentile_A",
        ],
    )
    write_csv_atomic(
        run_dir / "null_coordinate_dispersion.csv",
        dispersion_rows,
        [
            "source",
            "family",
            "coordinate",
            "mean",
            "sample_variance",
            "median",
            "mad",
            "minimum",
            "maximum",
            "quantile_2_5",
            "quantile_97_5",
        ],
    )
    write_csv_atomic(
        run_dir / "holm_stepdown_trace.csv",
        readout["holm"],
        [
            "rank",
            "source",
            "family",
            "raw_p_value",
            "holm_threshold",
            "holm_adjusted_p_value",
            "reject",
        ],
    )


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# D-0161 Item 4 Multi-seed Gate 1 Readout",
        "",
        f"- Status: `{summary['status']}`",
        f"- Outcome: `{summary['outcome']}`",
        f"- Baseline reproduced: `{summary['baseline_reproduced']}`",
        "",
        "## Source-family tests",
        "",
        "| Source | Family | C/511 | p | Holm reject |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    decisions = {
        (row["source"], row["family"]): row["reject"]
        for row in summary["holm_stepdown_trace"]
    }
    for row in summary["source_family_tests"]:
        lines.append(
            f"| {row['source']} | {row['family']} | {row['pareto_exceedance_count']}/511 | "
            f"{row['p_value']:.8f} | {decisions.get((row['source'], row['family']), False)} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            summary["conditional_inference_boundary"],
            "",
            "The result is restricted to the tested finite sources, prefix windows, metrics, "
            "and fitted null families. It does not establish asymptotic robustness or a universal class law.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def validate_stage0_manifest(payload: dict[str, Any], stage0_dir: Path) -> None:
    if (
        payload.get("status") != "completed"
        or payload.get("outcome") != "inputs-derivable"
    ):
        raise RuntimeError("Stage 0 is not a completed inputs-derivable closure.")
    if payload.get("preregistration", {}).get("sha256") != PREREG_SHA256:
        raise RuntimeError("Stage 0 points to a different preregistration.")
    if payload.get("scientific_values_emitted") != []:
        raise RuntimeError("Stage 0 anti-observation declaration is not empty.")
    if (
        len(payload.get("observed_caches", [])) != 4
        or len(payload.get("null_caches", [])) != 6_132
    ):
        raise RuntimeError("Stage 0 cache inventory is incomplete.")
    for relative, expected in payload.get("artifact_hashes", {}).items():
        path = stage0_dir / relative
        if not path.exists():
            raise RuntimeError(f"Missing Stage 0 artifact: {path}")
        if relative.endswith(".npy"):
            continue
        if sha256_file(path) != expected:
            raise RuntimeError(f"Stage 0 artifact hash mismatch: {path}")
    for row in payload["observed_caches"]:
        source_spec = SOURCE_SPECS[row["source"]]
        identity = observed_cache_identity(
            source_spec, Path(row["source_path"]), row["source_sha256"]
        )
        sidecar = validate_identity_cache(Path(row["cache_path"]), identity=identity)
        if sidecar is None or sidecar["cache_sha256"] != row["cache_sha256"]:
            raise RuntimeError(f"Observed cache identity mismatch: {row['cache_path']}")
    for row in payload["null_caches"]:
        spec = {
            "source": row["source"],
            "iteration": row["iteration"],
            "observed_cache_sha256": row["observed_cache_sha256"],
            "family": row["null_family"],
            "seed": row["null_seed"],
        }
        sidecar = validate_identity_cache(
            Path(row["cache_path"]), identity=null_cache_identity(spec)
        )
        if sidecar is None or not valid_fit_metadata(sidecar, row["null_family"]):
            raise RuntimeError(f"Null cache identity mismatch: {row['cache_path']}")


def stage0_artifact_hashes(
    run_dir: Path, observed_rows: list[dict[str, Any]], null_rows: list[dict[str, Any]]
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for row in observed_rows + null_rows:
        cache = Path(row["cache_path"])
        hashes[cache.relative_to(run_dir).as_posix()] = row["cache_sha256"]
        sidecar = cache.with_suffix(".json")
        hashes[sidecar.relative_to(run_dir).as_posix()] = sha256_file(sidecar)
    for name in (
        "stage0_report.md",
        "source_identity.json",
        "observed_cache_inventory.csv",
        "null_cache_inventory.csv",
        "resource_telemetry.csv",
    ):
        hashes[name] = sha256_file(run_dir / name)
    return hashes


def hash_artifacts(root: Path, *, exclude: set[str]) -> dict[str, str]:
    hashes = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in exclude or "quarantine" in path.relative_to(root).parts:
            continue
        hashes[relative] = sha256_file(path)
    return hashes


def write_stage0_report(
    path: Path,
    sources: list[dict[str, Any]],
    observed: list[dict[str, Any]],
    nulls: list[dict[str, Any]],
    preflight: dict[str, Any],
) -> None:
    lines = [
        "# D-0161 Item 4 Stage 0",
        "",
        "- Outcome: `inputs-derivable`",
        f"- Frozen sources verified: `{len(sources)}/4`",
        f"- Observed caches frozen: `{len(observed)}/4`",
        f"- Mandatory null caches frozen: `{len(nulls)}/6132`",
        f"- Effective workers after preflight: `{preflight['effective_workers']}`",
        "- Scientific values emitted: `none`",
        "",
        "Stage 1 remains blocked pending independent manifest review and explicit authorization.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def current_code_hashes() -> dict[str, str]:
    paths = {
        "runner": Path(__file__).resolve(),
        "item4_core": REPO_DIR / "v2" / "postreview" / "item4_multiseed_monte_carlo.py",
        "item4_statistics": REPO_DIR / "v2" / "postreview" / "item4_statistics.py",
        "streaming_core": REPO_DIR / "v2" / "postreview" / "item2b_growth_shells.py",
        "io": REPO_DIR / "v2" / "common" / "io.py",
        "null_models": NULL_MODELS_PATH,
        "phase1_runner": REPO_DIR / "hsi_v2_phase1_run.py",
        "phase1_tower": REPO_DIR / "v2" / "phase1" / "tower.py",
        "phase1_report": REPO_DIR / "v2" / "phase1" / "report.py",
        "phase1_coherence": REPO_DIR / "v2" / "phase1" / "coherence.py",
        "phase1_transport": REPO_DIR / "v2" / "phase1" / "transport.py",
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
        f"# D-0161 Item 4 Failure\n\n- Stage: `{stage}`\n- Error: `{exc!r}`\n",
        encoding="utf-8",
    )


def load_json(path: Path, *, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return default


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
    if not 1 <= args.workers <= MAX_WORKERS:
        raise SystemExit(f"--workers must be in [1,{MAX_WORKERS}].")
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
    return (
        (REPO_DIR / "results").resolve()
        if not value
        else Path(value).expanduser().resolve()
    )


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
    return output_root / f"item4-gate1-multiseed-{args.stage}__{timestamp}"


def print_plan(args: argparse.Namespace, results_base: Path, run_dir: Path) -> None:
    print("D-0161 Item 4 dry run")
    print(f"Stage: {args.stage}")
    print(f"Results base: {results_base}")
    print(f"Run directory: {run_dir}")
    print(f"Workers requested: {args.workers}")
    print(
        "No source or target cache was opened. Add --execute only after the applicable gate."
    )


def format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}h {minutes:02d}m {secs:02d}s"


if __name__ == "__main__":
    raise SystemExit(main())
