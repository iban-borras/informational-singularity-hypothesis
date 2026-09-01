#!/usr/bin/env python3
"""D-0158 Item 2b: B-markov1 growth-shell gate-plane experiment."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import psutil
except ImportError:  # pragma: no cover - execution gate rejects this environment.
    psutil = None  # type: ignore[assignment]


REPO_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = REPO_DIR.parent
sys.path.insert(0, str(REPO_DIR))

from v2.postreview.item2b_growth_shells import (  # noqa: E402
    FROZEN_WINDOWS,
    POLICIES,
    PREREG_SHA256,
    SCALES,
    SEEDS,
    SOURCE_SPECS,
    RunLogger,
    build_growth_shell_readout,
    freeze_or_reuse_cache,
    scan_nested_pair,
    sha256_file,
    sha256_file_progress,
    source_sequence_length,
    verify_compiled_kernels,
    write_csv_atomic,
    write_json_atomic,
)
from v2.common.io import load_observable_cache  # noqa: E402


PREREG_PATH = PROJECT_ROOT / "prereg_postreview_item2b_bmarkov1_growth_shells_v1.0.md"
DEFAULT_OUTPUT = "results/hsi_v2/postreview_item2b_bmarkov1_growth_shells"
RAM_BUDGET_BYTES = 32 * 1024**3
STAGE0_RSS_LIMIT_BYTES = 4 * 1024**3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("stage0", "stage1"), required=True)
    parser.add_argument("--results-base", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--run-dir", default=None, help="Explicit run directory for strict resume.")
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
        raise SystemExit("psutil is required for contract-level RSS telemetry and worker admission.")
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
        failure_name = "stage0_manifest.json" if args.stage == "stage0" else "manifest.json"
        failure_path = run_dir / failure_name
        failure_payload: dict[str, Any] = {}
        if failure_path.exists():
            try:
                failure_payload = json.loads(failure_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                failure_payload = {}
        failure_payload.update(
            {
                "status": "failed",
                "outcome": "inputs-or-shells-not-derivable",
                "error": repr(exc),
                "stage": args.stage,
                "preregistration_sha256": PREREG_SHA256,
                "code_audit_sha256": args.code_audit_sha256.upper(),
                "finished_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        write_json_atomic(
            failure_path,
            failure_payload,
        )
        write_failure_report(run_dir, args.stage, exc)
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
    sources: dict[str, dict[str, Any]] = {}

    for depth, (expected_sha256, expected_observable_bits) in SOURCE_SPECS.items():
        path = source_path(results_base, depth)
        if not path.exists():
            raise RuntimeError(f"Missing frozen source B@{depth}: {path}")
        actual_sha256 = sha256_file_progress(path, logger, label=f"B@{depth}")
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"Source hash mismatch for B@{depth}: expected {expected_sha256}, got {actual_sha256}."
            )
        sources[str(depth)] = {
            "variant": "B",
            "iteration": depth,
            "path": str(path),
            "sha256": actual_sha256,
            "file_bytes": path.stat().st_size,
            "sequence_length": source_sequence_length(path),
            "expected_observable_bits": expected_observable_bits,
        }
        logger.emit("source-verified", **sources[str(depth)])

    windows_dir = run_dir / "windows"
    window_rows: list[dict[str, Any]] = []
    pair_telemetry: list[dict[str, Any]] = []
    for parent_depth, child_depth in ((19, 20), (23, 24)):
        specs = tuple(spec for spec in FROZEN_WINDOWS if spec.depth == child_depth)
        extracted, telemetry = scan_nested_pair(
            Path(sources[str(parent_depth)]["path"]),
            Path(sources[str(child_depth)]["path"]),
            parent_sequence_length=int(sources[str(parent_depth)]["sequence_length"]),
            child_sequence_length=int(sources[str(child_depth)]["sequence_length"]),
            parent_observable_bits=int(sources[str(parent_depth)]["expected_observable_bits"]),
            child_observable_bits=int(sources[str(child_depth)]["expected_observable_bits"]),
            windows=specs,
            raw_chunk_bytes=args.raw_chunk_mib * 1024 * 1024,
            logger=logger,
        )
        telemetry.update({"parent_depth": parent_depth, "child_depth": child_depth})
        pair_telemetry.append(telemetry)
        if telemetry["peak_rss_bytes"] >= STAGE0_RSS_LIMIT_BYTES:
            raise RuntimeError(
                f"Stage 0 RSS limit exceeded: {telemetry['peak_rss_bytes']} >= {STAGE0_RSS_LIMIT_BYTES}."
            )

        for spec in specs:
            cache_path = windows_dir / f"B_{spec.slug}.npy"
            sidecar = freeze_or_reuse_cache(
                cache_path,
                extracted[spec.slug],
                metadata={
                    "stage": "item2b_stage0_window_cache",
                    "variant": "B",
                    "iteration": spec.depth,
                    "q": spec.q,
                    "observable_start": spec.start,
                    "observable_end_exclusive": spec.end,
                    "source_path": sources[str(spec.depth)]["path"],
                    "source_sha256": sources[str(spec.depth)]["sha256"],
                    "parent_iteration": parent_depth,
                    "parent_source_sha256": sources[str(parent_depth)]["sha256"],
                    "preregistration_sha256": PREREG_SHA256,
                    "extraction_code_sha256": code_hashes["item2b_internal"],
                },
            )
            row = {
                "depth": spec.depth,
                "q": spec.q,
                "start": spec.start,
                "end_exclusive": spec.end,
                "observable_length": sidecar["observable_length"],
                "cache_path": sidecar["cache_path"],
                "cache_sha256": sidecar["cache_sha256"],
                "metadata_path": str(cache_path.with_suffix(".json").resolve()),
                "source_sha256": sidecar["source_sha256"],
                "reused": sidecar["reused"],
            }
            window_rows.append(row)
            logger.emit("cache-frozen", **row)

    source_identity_path = run_dir / "source_identity.json"
    inventory_path = run_dir / "window_inventory.csv"
    report_path = run_dir / "stage0_report.md"
    manifest_path = run_dir / "stage0_manifest.json"
    write_json_atomic(
        source_identity_path,
        {
            "sources": sources,
            "prefix_checks": pair_telemetry,
            "preregistration_sha256": PREREG_SHA256,
        },
    )
    write_csv_atomic(
        inventory_path,
        window_rows,
        [
            "depth",
            "q",
            "start",
            "end_exclusive",
            "observable_length",
            "cache_path",
            "cache_sha256",
            "metadata_path",
            "source_sha256",
            "reused",
        ],
    )
    write_stage0_report(report_path, sources, window_rows, pair_telemetry)
    artifact_hashes = hash_artifacts(
        run_dir,
        exclude={"run.log", "stage0_manifest.json", "stage0_manifest.sha256"},
    )
    manifest = {
        "status": "completed",
        "outcome": "inputs-derivable",
        "stage": "stage0",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "run_dir": str(run_dir),
        "preregistration": {"path": str(PREREG_PATH), "sha256": PREREG_SHA256},
        "code_audit_sha256": args.code_audit_sha256.upper(),
        "code_hashes": code_hashes,
        "kernel_state": kernel_state,
        "sources": sources,
        "prefix_checks": pair_telemetry,
        "windows": window_rows,
        "peak_rss_bytes": max(item["peak_rss_bytes"] for item in pair_telemetry),
        "artifact_hashes": artifact_hashes,
        "boundary": "Stage 0 contains no pattern, null, gate-plane, or scientific metric readout.",
    }
    write_json_atomic(manifest_path, manifest)
    manifest_sha256 = sha256_file(manifest_path)
    (run_dir / "stage0_manifest.sha256").write_text(
        f"{manifest_sha256}  stage0_manifest.json\n", encoding="ascii"
    )
    logger.emit("stage0-completed", outcome="inputs-derivable", manifest_sha256=manifest_sha256)
    print(f"Stage 0 outcome: inputs-derivable")
    print(f"Stage 0 manifest: {manifest_path}")
    print(f"Stage 0 manifest SHA-256: {manifest_sha256}")
    print("STOP: Stage 1 requires independent manifest review and explicit authorization.")
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
    caches = validate_stage0_caches(stage0)
    jobs = build_jobs(caches, stage0_sha256, code_hashes)
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

    preflight_specs = [
        next(item for item in jobs if item["depth"] == 20 and item["q"] == "1/10" and item["seed"] is None),
        next(item for item in jobs if item["depth"] == 20 and item["q"] == "1/10" and item["seed"] == 17),
    ]
    records: dict[str, dict[str, Any]] = {}
    for spec in preflight_specs:
        record = run_single_job(spec, run_dir, logger, completed=len(records), total=len(jobs))
        records[spec["job_id"]] = record
        manifest["jobs"] = ordered_records(jobs, records)
        write_json_atomic(manifest_path, manifest)
        if record["status"] != "completed":
            raise RuntimeError(f"Resource preflight job failed: {spec['job_id']}")

    max_preflight_rss = max(int(records[item["job_id"]]["peak_rss_bytes"]) for item in preflight_specs)
    admission = max(1, int(RAM_BUDGET_BYTES // max(1, int(1.5 * max_preflight_rss))))
    effective_workers = max(1, min(4, args.workers, admission))
    manifest["resource_preflight"] = {
        "observed_job": preflight_specs[0]["job_id"],
        "markov1_job": preflight_specs[1]["job_id"],
        "max_peak_rss_bytes": max_preflight_rss,
        "ram_budget_bytes": RAM_BUDGET_BYTES,
        "safety_factor": 1.5,
        "effective_workers": effective_workers,
    }
    write_json_atomic(manifest_path, manifest)
    logger.emit("stage1-preflight-completed", **manifest["resource_preflight"])

    remaining = [item for item in jobs if item["job_id"] not in records]
    concurrent_records = run_job_queue(
        remaining,
        run_dir,
        logger,
        initial_completed=len(records),
        total=len(jobs),
        workers=effective_workers,
        initial_durations=[float(records[item["job_id"]]["elapsed_seconds"]) for item in preflight_specs],
        stage_started=started,
    )
    records.update({item["job_id"]: item for item in concurrent_records})
    manifest["jobs"] = ordered_records(jobs, records)
    write_json_atomic(manifest_path, manifest)
    if len(records) != 42 or any(item["status"] != "completed" for item in records.values()):
        raise RuntimeError("Stage 1 did not complete all 42 frozen jobs.")

    summary, per_window, per_seed = build_growth_shell_readout(list(records.values()))
    summary.update(
        {
            "preregistration_sha256": PREREG_SHA256,
            "stage0_manifest_sha256": stage0_sha256,
            "run_dir": str(run_dir),
        }
    )
    summary_path = run_dir / "summary.json"
    report_path = run_dir / "report.md"
    per_window_path = run_dir / "per_window_gate_plane.csv"
    per_seed_path = run_dir / "per_seed_markov1_gate_plane.csv"
    telemetry_path = run_dir / "resource_telemetry.csv"
    write_json_atomic(summary_path, summary)
    write_stage1_report(report_path, summary)
    write_csv_atomic(
        per_window_path,
        per_window,
        [
            "depth",
            "q",
            "retention_B",
            "active_mean_tail_B",
            "max_retention_markov1",
            "G",
            "D",
            "S",
            "Delta_G",
            "Delta_D",
            "observed_run_dir",
        ],
    )
    write_csv_atomic(
        per_seed_path,
        per_seed,
        ["depth", "q", "seed", "retention_markov1", "active_mean_tail_markov1", "run_dir"],
    )
    telemetry_rows = [
        {
            "job_id": item["job_id"],
            "depth": item["depth"],
            "q": item["q"],
            "kind": item["kind"],
            "seed": item["seed"],
            "elapsed_seconds": item["elapsed_seconds"],
            "peak_rss_bytes": item["peak_rss_bytes"],
            "reused": item["reused"],
        }
        for item in manifest["jobs"]
    ]
    write_csv_atomic(
        telemetry_path,
        telemetry_rows,
        ["job_id", "depth", "q", "kind", "seed", "elapsed_seconds", "peak_rss_bytes", "reused"],
    )
    manifest.update(
        {
            "status": "completed",
            "outcome": summary["outcome"],
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "outputs": {
                "summary": str(summary_path),
                "report": str(report_path),
                "per_window_gate_plane": str(per_window_path),
                "per_seed_markov1_gate_plane": str(per_seed_path),
                "resource_telemetry": str(telemetry_path),
            },
        }
    )
    manifest["artifact_hashes"] = hash_artifacts(
        run_dir,
        exclude={"run.log", "manifest.json", "manifest.sha256"},
    )
    write_json_atomic(manifest_path, manifest)
    manifest_sha256 = sha256_file(manifest_path)
    (run_dir / "manifest.sha256").write_text(f"{manifest_sha256}  manifest.json\n", encoding="ascii")
    logger.emit("stage1-completed", outcome=summary["outcome"], manifest_sha256=manifest_sha256)
    print("D-0158 Item 2b Stage 1 completed.")
    print(f"Outcome: {summary['outcome']}")
    print(f"Output: {run_dir}")
    print(f"Manifest SHA-256: {manifest_sha256}")
    return 0


def build_jobs(
    caches: dict[tuple[int, str], dict[str, Any]],
    stage0_sha256: str,
    code_hashes: dict[str, str],
) -> list[dict[str, Any]]:
    jobs = []
    for spec in FROZEN_WINDOWS:
        cache = caches[(spec.depth, spec.q)]
        for seed in (None, *SEEDS):
            kind = "observed" if seed is None else "markov1"
            job_id = f"d{spec.depth}_q{spec.q.replace('/', 'of')}__{kind}"
            if seed is not None:
                job_id += f"__seed-{seed}"
            identity = {
                "job_id": job_id,
                "depth": spec.depth,
                "q": spec.q,
                "kind": kind,
                "seed": seed,
                "cache_path": cache["cache_path"],
                "cache_sha256": cache["cache_sha256"],
                "stage0_manifest_sha256": stage0_sha256,
                "preregistration_sha256": PREREG_SHA256,
                "protocol": frozen_protocol(),
                "code_hashes": code_hashes,
            }
            jobs.append({**identity, "identity": identity})
    return jobs


def run_single_job(
    spec: dict[str, Any],
    run_dir: Path,
    logger: RunLogger,
    *,
    completed: int,
    total: int,
) -> dict[str, Any]:
    existing = validated_job_record(spec, run_dir)
    if existing is not None:
        logger.emit("job-reused", job_id=spec["job_id"], jobs_completed=f"{completed + 1}/{total}")
        return existing
    state = launch_job(spec, run_dir)
    while state["process"].poll() is None:
        update_running_state(state)
        emit_stage1_progress(
            logger,
            [state],
            completed,
            total,
            stage_started=state["started_monotonic"],
            completed_durations=[],
            effective_workers=1,
        )
        time.sleep(1.0)
    return finalize_job(state, spec, run_dir)


def run_job_queue(
    specs: list[dict[str, Any]],
    run_dir: Path,
    logger: RunLogger,
    *,
    initial_completed: int,
    total: int,
    workers: int,
    initial_durations: list[float],
    stage_started: float,
) -> list[dict[str, Any]]:
    pending = list(specs)
    active: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    try:
        while pending or active:
            for state in list(active):
                update_running_state(state)
                if state["process"].poll() is not None:
                    active.remove(state)
                    record = finalize_job(state, state["spec"], run_dir)
                    records.append(record)
                    if record["status"] != "completed":
                        raise RuntimeError(f"Phase 1 child failed: {record['job_id']}")

            completed = initial_completed + len(records)
            aggregate_rss = sum(int(item["current_rss_bytes"]) for item in active)
            while pending and len(active) < workers and aggregate_rss < RAM_BUDGET_BYTES:
                spec = pending.pop(0)
                existing = validated_job_record(spec, run_dir)
                if existing is not None:
                    records.append(existing)
                    completed += 1
                    logger.emit("job-reused", job_id=spec["job_id"], jobs_completed=f"{completed}/{total}")
                    continue
                state = launch_job(spec, run_dir)
                active.append(state)
                update_running_state(state)
                aggregate_rss = sum(int(item["current_rss_bytes"]) for item in active)

            emit_stage1_progress(
                logger,
                active,
                completed,
                total,
                stage_started=stage_started,
                completed_durations=initial_durations + [float(item["elapsed_seconds"]) for item in records],
                effective_workers=workers,
            )
            if active:
                time.sleep(1.0)
            elif pending:
                raise RuntimeError("Stage 1 queue stalled without an active worker.")
        return records
    except BaseException:
        for state in active:
            terminate_process(state["process"])
            state["handle"].close()
        raise


def launch_job(spec: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    job_dir = run_dir / "phase1_jobs" / spec["job_id"]
    if job_dir.exists():
        quarantine_job(job_dir)
    output_dir = job_dir / "phase1_output"
    job_dir.mkdir(parents=True, exist_ok=True)
    command = phase1_command(spec, output_dir)
    log_path = job_dir / "child.log"
    handle = open(log_path, "a", encoding="utf-8", buffering=1)
    process = subprocess.Popen(
        command,
        cwd=REPO_DIR,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return {
        "spec": spec,
        "process": process,
        "handle": handle,
        "job_dir": job_dir,
        "output_dir": output_dir,
        "command": command,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "started_monotonic": time.monotonic(),
        "current_rss_bytes": 0,
        "peak_rss_bytes": 0,
        "last_subphase": "starting",
    }


def update_running_state(state: dict[str, Any]) -> None:
    rss = process_tree_rss(state["process"].pid)
    state["current_rss_bytes"] = rss
    state["peak_rss_bytes"] = max(int(state["peak_rss_bytes"]), rss)
    state["last_subphase"] = tail_last_status(Path(state["job_dir"]) / "child.log")


def finalize_job(
    state: dict[str, Any],
    spec: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    state["handle"].close()
    return_code = int(state["process"].returncode)
    record = {
        "job_id": spec["job_id"],
        "depth": spec["depth"],
        "q": spec["q"],
        "kind": spec["kind"],
        "seed": spec["seed"],
        "status": "failed" if return_code else "completed",
        "return_code": return_code,
        "started_at": state["started_at"],
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": round(time.monotonic() - state["started_monotonic"], 3),
        "peak_rss_bytes": int(state["peak_rss_bytes"]),
        "reused": False,
        "identity": spec["identity"],
        "command": state["command"],
    }
    if return_code == 0:
        phase1_run = find_phase1_run(state["output_dir"])
        validate_phase1_output(phase1_run, spec)
        record["run_dir"] = str(phase1_run.resolve())
        record["artifact_hashes"] = hash_artifacts(phase1_run, exclude=set())
    write_json_atomic(state["job_dir"] / "job_status.json", record)
    return record


def validated_job_record(spec: dict[str, Any], run_dir: Path) -> dict[str, Any] | None:
    job_dir = run_dir / "phase1_jobs" / spec["job_id"]
    status_path = job_dir / "job_status.json"
    if not status_path.exists():
        return None
    try:
        record = json.loads(status_path.read_text(encoding="utf-8"))
        if record.get("status") != "completed" or record.get("identity") != spec["identity"]:
            raise ValueError("job identity mismatch")
        run_path = Path(record["run_dir"])
        validate_phase1_output(run_path, spec)
        for relative, expected in record["artifact_hashes"].items():
            if sha256_file(run_path / relative) != expected:
                raise ValueError(f"artifact mismatch: {relative}")
        return {**record, "reused": True}
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        quarantine_job(job_dir)
        return None


def phase1_command(spec: dict[str, Any], output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(REPO_DIR / "hsi_v2_phase1_run.py"),
        "--variant",
        "B",
        "--iteration",
        str(spec["depth"]),
        "--observable-cache",
        spec["cache_path"],
        "--segment-bits",
        "1000000",
        "--num-segments",
        "3",
        "--scales",
        ",".join(str(value) for value in SCALES),
        "--policies",
        ",".join(POLICIES),
        "--min-count-floor",
        "16",
        "--min-count-rate",
        "1e-6",
        "--min-segment-support",
        "2",
        "--cv-max",
        "1.5",
        "--output-dir",
        str(output_dir),
    ]
    if spec["seed"] is not None:
        command.extend(["--null-model", "markov1", "--null-seed", str(spec["seed"])])
    return command


def validate_phase1_output(run_dir: Path, spec: dict[str, Any]) -> None:
    dataset_path = run_dir / "dataset.json"
    summary_path = run_dir / "phase1_summary.json"
    manifest_path = run_dir / "manifest.json"
    for path in (dataset_path, summary_path, manifest_path):
        if not path.exists():
            raise ValueError(f"Missing Phase 1 artifact: {path}")
    config = json.loads(dataset_path.read_text(encoding="utf-8"))["config"]
    expected = frozen_protocol()
    checks = {
        "source_variant": config.get("source_variant") == "B",
        "source_iteration": config.get("source_iteration") == spec["depth"],
        "input_mode": config.get("input_mode") == "observable_cache",
        "cache_path": Path(config.get("input_observable_cache_path", "")).resolve() == Path(spec["cache_path"]).resolve(),
        "cache_sha256": config.get("input_observable_cache_sha256") == spec["cache_sha256"],
        "segment_bits": config.get("segment_bits") == expected["segment_bits"],
        "num_segments": config.get("num_segments") == expected["num_segments"],
        "scales": config.get("scales") == expected["scales"],
        "policies": config.get("policies") == expected["policies"],
        "min_count_floor": config.get("min_count_floor") == expected["min_count_floor"],
        "min_count_rate": float(config.get("min_count_rate")) == expected["min_count_rate"],
        "min_segment_support": config.get("min_segment_support_requested") == expected["min_segment_support"],
        "cv_max": float(config.get("cv_max")) == expected["cv_max"],
        "null_model": config.get("null_model") == ("markov1" if spec["seed"] is not None else None),
        "null_seed": config.get("null_seed") == spec["seed"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Phase 1 output identity mismatch: {', '.join(failed)}")


def validate_stage0_manifest(payload: dict[str, Any], stage0_dir: Path) -> None:
    if payload.get("status") != "completed" or payload.get("outcome") != "inputs-derivable":
        raise RuntimeError("Stage 0 manifest is not a completed inputs-derivable closure.")
    if payload.get("preregistration", {}).get("sha256") != PREREG_SHA256:
        raise RuntimeError("Stage 0 manifest points to a different preregistration.")
    for relative, expected in payload.get("artifact_hashes", {}).items():
        path = stage0_dir / relative
        if not path.exists() or sha256_file(path) != expected:
            raise RuntimeError(f"Stage 0 artifact hash mismatch: {relative}")


def validate_stage0_caches(payload: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    caches = {}
    for row in payload.get("windows", []):
        path = Path(row["cache_path"])
        _, sidecar, _ = load_observable_cache(path, expected_bits=3_000_000)
        if sidecar.get("source_sha256") != row["source_sha256"]:
            raise RuntimeError(f"Cache source identity mismatch: {path}")
        caches[(int(row["depth"]), str(row["q"]))] = row
    required = {(spec.depth, spec.q) for spec in FROZEN_WINDOWS}
    if set(caches) != required:
        raise RuntimeError("Stage 0 manifest does not contain exactly the six frozen caches.")
    return caches


def frozen_protocol() -> dict[str, Any]:
    return {
        "variant": "B",
        "null_model": "markov1",
        "null_seeds": list(SEEDS),
        "segment_bits": 1_000_000,
        "num_segments": 3,
        "segment_offset_bits": 0,
        "scales": list(SCALES),
        "policies": list(POLICIES),
        "min_count_floor": 16,
        "min_count_rate": 1e-6,
        "min_segment_support": 2,
        "cv_max": 1.5,
        "main_transport_policy": "prefix",
        "control_transport_policy": "suffix",
        "tail_edges": 2,
    }


def emit_stage1_progress(
    logger: RunLogger,
    active: list[dict[str, Any]],
    completed: int,
    total: int,
    *,
    stage_started: float,
    completed_durations: list[float],
    effective_workers: int,
) -> None:
    elapsed = max(time.monotonic() - stage_started, 1e-9)
    remaining = total - completed
    eta = None
    if completed_durations:
        eta = (sum(completed_durations) / len(completed_durations)) * remaining / max(1, effective_workers)
    current = active[0]["spec"] if active else {}
    logger.emit(
        "progress",
        force=False,
        phase="stage1-phase1-jobs",
        depth=current.get("depth"),
        q=current.get("q"),
        observed_or_null=current.get("kind"),
        seed=current.get("seed"),
        scale_or_subphase=active[0].get("last_subphase", "phase1-child-process") if active else "queue-idle",
        jobs_completed=f"{completed}/{total}",
        workers_active=len(active),
        aggregate_child_rss_bytes=sum(int(item["current_rss_bytes"]) for item in active),
        elapsed_seconds=round(elapsed, 3),
        eta_seconds=None if eta is None else round(eta, 3),
    )


def tail_last_status(path: Path, *, max_bytes: int = 16_384) -> str:
    try:
        with open(path, "rb") as handle:
            size = handle.seek(0, os.SEEK_END)
            handle.seek(max(0, size - max_bytes))
            text = handle.read().decode("utf-8", errors="replace")
        candidates = [item.strip() for item in text.replace("\r", "\n").splitlines() if item.strip()]
        return candidates[-1][-240:] if candidates else "phase1-child-process"
    except OSError:
        return "phase1-child-process"


def process_tree_rss(pid: int) -> int:
    assert psutil is not None
    try:
        process = psutil.Process(pid)
        total = process.memory_info().rss
        for child in process.children(recursive=True):
            try:
                total += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return int(total)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0


def terminate_process(process: subprocess.Popen) -> None:
    assert psutil is not None
    if process.poll() is not None:
        return
    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
        for child in children:
            child.terminate()
        parent.terminate()
        _, alive = psutil.wait_procs([parent, *children], timeout=5)
        for item in alive:
            item.kill()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        process.kill()


def find_phase1_run(output_dir: Path) -> Path:
    candidates = sorted(path.parent for path in output_dir.rglob("dataset.json"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one Phase 1 run in {output_dir}, found {len(candidates)}.")
    return candidates[0]


def quarantine_job(job_dir: Path) -> None:
    if not job_dir.exists():
        return
    quarantine = job_dir.parent / "quarantine"
    quarantine.mkdir(parents=True, exist_ok=True)
    destination = quarantine / f"{job_dir.name}__{datetime.now().strftime('%Y%m%dT%H%M%S%f')}"
    shutil.move(str(job_dir), str(destination))


def ordered_records(jobs: list[dict[str, Any]], records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [records[item["job_id"]] for item in jobs if item["job_id"] in records]


def write_stage0_report(
    path: Path,
    sources: dict[str, dict[str, Any]],
    windows: list[dict[str, Any]],
    telemetry: list[dict[str, Any]],
) -> None:
    lines = [
        "# D-0158 Item 2b Stage 0",
        "",
        "- Outcome: `inputs-derivable`",
        f"- Preregistration SHA-256: `{PREREG_SHA256}`",
        "- Exact observable-prefix checks: `B@19 in B@20`, `B@23 in B@24`",
        "- Scientific metrics computed: `none`",
        "",
        "## Sources",
        "",
    ]
    for depth, row in sources.items():
        lines.append(f"- B@{depth}: `{row['sha256']}`; observable bits `{row['expected_observable_bits']}`")
    lines.extend(["", "## Frozen Windows", "", "| depth | q | interval | cache SHA-256 |", "| ---: | ---: | --- | --- |"]) 
    for row in windows:
        lines.append(
            f"| {row['depth']} | {row['q']} | [{row['start']}, {row['end_exclusive']}) | `{row['cache_sha256']}` |"
        )
    lines.extend(["", "## Resource Boundary", ""])
    lines.append(f"- Peak RSS: `{max(item['peak_rss_bytes'] for item in telemetry)}` bytes (< 4 GiB).")
    lines.append("- Stage 1 was not started.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_stage1_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# D-0158 Item 2b Growth-Shell Readout",
        "",
        f"- Status: `{summary['status']}`",
        f"- Outcome: `{summary['outcome']}`",
        f"- C(20): `{summary['C20']}`",
        f"- C(24): `{summary['C24']}`",
        f"- Preregistration SHA-256: `{summary['preregistration_sha256']}`",
        f"- Stage 0 manifest SHA-256: `{summary['stage0_manifest_sha256']}`",
        "",
        "## Per-Window Gate Plane",
        "",
        "| depth | q | G | D | S |",
        "| ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary["per_window"]:
        lines.append(f"| {row['depth']} | {row['q']} | {row['G']:.6f} | {row['D']:.6f} | {row['S']} |")
    lines.extend(["", "## Depth Deltas", "", "| q | Delta_G | Delta_D |", "| ---: | ---: | ---: |"]) 
    for row in summary["depth_deltas"]:
        lines.append(f"| {row['q']} | {row['Delta_G']:.6f} | {row['Delta_D']:.6f} |")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            summary["n3_boundary"],
            "",
            "This is a finite-depth growth-shell Markov-1 comparison. It does not authorize asymptotic, stationarity, Gate 2, or geometric claims.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_failure_report(run_dir: Path, stage: str, exc: Exception) -> None:
    name = "stage0_report.md" if stage == "stage0" else "report.md"
    (run_dir / name).write_text(
        f"# D-0158 Item 2b {stage} failure\n\n"
        "- Status: `failed`\n"
        "- Outcome: `inputs-or-shells-not-derivable`\n"
        f"- Error: `{repr(exc)}`\n"
        f"- Preregistration SHA-256: `{PREREG_SHA256}`\n\n"
        "No source, window, seed, offset, threshold, or protocol substitution was attempted.\n",
        encoding="utf-8",
    )


def current_code_hashes() -> dict[str, str]:
    from v2.postreview import item2b_growth_shells

    return {
        "public_runner": sha256_file(Path(__file__)),
        "item2b_internal": sha256_file(Path(item2b_growth_shells.__file__)),
        "phase1_runner": sha256_file(REPO_DIR / "hsi_v2_phase1_run.py"),
        "phase1_tower": sha256_file(REPO_DIR / "v2" / "phase1" / "tower.py"),
        "null_models": sha256_file(REPO_DIR / "v2" / "common" / "null_models.py"),
        "common_io": sha256_file(REPO_DIR / "v2" / "common" / "io.py"),
    }


def hash_artifacts(root: Path, *, exclude: set[str]) -> dict[str, str]:
    hashes = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if (
            relative in exclude
            or path.name.endswith(".tmp")
            or path.name in {"child.log", "manifest.sha256", "stage0_manifest.sha256"}
            or "quarantine" in path.parts
        ):
            continue
        hashes[relative] = sha256_file(path)
    return hashes


def source_path(results_base: Path, depth: int) -> Path:
    return results_base / "level0" / "phi_snapshots" / "var_B" / f"phi_iter{depth}.struct.gz"


def verify_preregistration() -> None:
    if not PREREG_PATH.exists():
        raise SystemExit(f"Missing frozen preregistration: {PREREG_PATH}")
    actual = sha256_file(PREREG_PATH)
    if actual != PREREG_SHA256:
        raise SystemExit(f"Preregistration hash mismatch: expected {PREREG_SHA256}, got {actual}")


def validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.raw_chunk_mib <= 16:
        raise SystemExit("--raw-chunk-mib must be in [1,16].")
    if not 1 <= args.workers <= 4:
        raise SystemExit("--workers must be in [1,4].")
    if args.execute:
        validate_sha256_arg("--code-audit-sha256", args.code_audit_sha256)
    if args.stage == "stage1" and args.execute:
        if not args.stage0_manifest:
            raise SystemExit("Stage 1 requires --stage0-manifest.")
        validate_sha256_arg("--stage0-manifest-sha256", args.stage0_manifest_sha256)
        validate_sha256_arg("--stage1-authorization-sha256", args.stage1_authorization_sha256)


def validate_sha256_arg(name: str, value: str | None) -> None:
    if value is None or len(value) != 64 or any(char not in "0123456789abcdefABCDEF" for char in value):
        raise SystemExit(f"{name} must be a 64-character hexadecimal SHA-256.")


def configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace", line_buffering=True)


def load_dotenv() -> None:
    path = REPO_DIR / ".env"
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() and key.strip() not in os.environ:
            os.environ[key.strip()] = value.strip().strip('"')


def resolve_results_base(raw: str | None) -> Path:
    value = raw or os.environ.get("HSI_RESULTS_BASE_DIR") or os.environ.get("HSI_V1_RESULTS_BASE_DIR")
    return Path(value).expanduser().resolve() if value else (REPO_DIR / "results").resolve()


def resolve_results_path(raw: str, results_base: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    if path.parts and path.parts[0].lower() == "results":
        return (results_base / Path(*path.parts[1:])).resolve()
    return (REPO_DIR / path).resolve()


def resolve_run_dir(args: argparse.Namespace, output_root: Path) -> Path:
    if args.run_dir:
        return Path(args.run_dir).expanduser().resolve()
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    return output_root / f"item2b-growth-shells-{args.stage}__{stamp}"


def print_plan(args: argparse.Namespace, results_base: Path, run_dir: Path) -> None:
    print("D-0158 Item 2b plan (dry-run; no target source opened)")
    print("-" * 78)
    print(f"stage: {args.stage}")
    print(f"run_dir: {run_dir}")
    print(f"preregistration_sha256: {PREREG_SHA256}")
    if args.stage == "stage0":
        for depth in SOURCE_SPECS:
            print(f"source B@{depth}: {source_path(results_base, depth)}")
        print("stage0 result: six immutable caches; mandatory stop")
    else:
        print(f"stage0_manifest: {args.stage0_manifest}")
        print("stage1 jobs: 42 (6 windows x [observed + 6 Markov-1])")
    print("Add --execute plus the required audit/authorization hashes to run.")


if __name__ == "__main__":
    raise SystemExit(main())
