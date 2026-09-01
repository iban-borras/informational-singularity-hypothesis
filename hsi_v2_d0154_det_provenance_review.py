#!/usr/bin/env python3
"""
D-0154 DET provenance-review batch.

Recomputes recurrence determinism (DET) over clean snapshots using the
executed legacy representative-subsample protocol: build the full medium
sample and then thin it to at most 10k evenly spaced bits.

This is a read-only reconciliation harness. It does not alter the RQA metric.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_DIR = Path(__file__).resolve().parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from level1_deep_analysis import recurrence_analysis  # noqa: E402
from utils.streaming_phi_loader import StreamingPhiLoader  # noqa: E402

DEFAULT_TARGETS = "E:24,I:23,D:20,G:20,F:20"
DEFAULT_OUTPUT_ROOT = "results/hsi_v2/d0154_det_provenance_review"
PAPER_DET: dict[str, float | None] = {
    "E": 0.55,
    "I": 0.55,
    "F": 0.56,
    "D": 0.55,
    "G": 0.55,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="D-0154 DET provenance review over clean snapshots.")
    parser.add_argument("--targets", default=DEFAULT_TARGETS)
    parser.add_argument("--results-base", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--det-bits", type=int, default=10_000)
    parser.add_argument("--medium-divisor", type=int, default=1_000_000)
    parser.add_argument("--medium-min-step", type=int, default=100)
    parser.add_argument("--threshold-seed", type=int, default=42)
    parser.add_argument("--chunk-bytes", type=int, default=50_000_000)
    parser.add_argument("--skip-source-hash", action="store_true")
    parser.add_argument(
        "--resume-run",
        default=None,
        help="Resume an interrupted run directory at completed-variant boundaries.",
    )
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> int:
    configure_stdout()
    load_dotenv()
    args = build_parser().parse_args()

    if args.resume_run and not args.execute:
        raise SystemExit("--resume-run requires --execute.")
    if args.resume_run and args.skip_source_hash:
        raise SystemExit("--resume-run requires SHA-256 source identity; remove --skip-source-hash.")

    targets = parse_targets(args.targets)
    results_base = resolve_results_base(args.results_base)
    output_root = resolve_results_path(args.output_dir, results_base)
    previous_manifest: dict[str, Any] | None = None
    if args.resume_run:
        run_dir = Path(args.resume_run).expanduser().resolve()
        manifest_path = run_dir / "manifest.json"
        if not run_dir.is_dir() or not manifest_path.is_file():
            raise SystemExit(f"Resume run is missing its manifest: {manifest_path}")
        previous_manifest = read_json(manifest_path)
        generated_at = str(previous_manifest.get("generated_at", ""))
        if not generated_at:
            raise RuntimeError("Resume manifest has no generated_at timestamp.")
    else:
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        run_dir = output_root / f"d0154-det-provenance-review__{timestamp}"
        generated_at = datetime.now().isoformat(timespec="seconds")

    code_hashes = {
        "runner": sha256_file(Path(__file__)),
        "medium_sampler": sha256_file(REPO_DIR / "utils" / "streaming_subsample.py"),
        "rqa_metric": sha256_file(REPO_DIR / "level1_deep_analysis.py"),
    }

    plan = {
        "status": "planned" if not args.execute else "running",
        "generated_at": generated_at,
        "script": Path(__file__).name,
        "decision_log_context": "D-0154 Amendment 10 DET provenance-review",
        "results_base": str(results_base),
        "run_dir": str(run_dir),
        "targets": targets,
        "protocol": {
            "legacy_sample": "full_medium_sample_then_dynamic_thinning",
            "det_bits": args.det_bits,
            "medium_divisor": args.medium_divisor,
            "medium_min_step": args.medium_min_step,
            "representative_step_formula": "max(1, len(medium_samples) // det_bits)",
            "threshold_seed": args.threshold_seed,
            "chunk_bytes": args.chunk_bytes,
            "source_hash_policy": "skipped" if args.skip_source_hash else "sha256",
        },
        "implementation_provenance": {
            "medium_sampler": "utils/streaming_subsample.py::streaming_multiscale_subsample",
            "medium_sampler_sha256": code_hashes["medium_sampler"],
            "rqa_metric": "level1_deep_analysis.py::recurrence_analysis",
            "rqa_metric_sha256": code_hashes["rqa_metric"],
        },
        "code_hashes": code_hashes,
        "resume_protocol": {
            "schema_version": 1,
            "boundary": "completed_variant_artifact",
            "source_identity": "sha256" if not args.skip_source_hash else "unverified",
            "atomic_variant_artifacts": True,
            "resume_eligible": not args.skip_source_hash,
        },
    }

    if not args.execute:
        print_plan(plan)
        return 0

    log_path = run_dir / "batch.log"
    manifest_path = run_dir / "manifest.json"
    if previous_manifest is None:
        run_dir.mkdir(parents=True, exist_ok=False)
        resume_count = 0
        write_json(manifest_path, {**plan, "resume_count": resume_count})
        write_log_header(log_path, plan)
    else:
        validate_resume_plan(previous_manifest, plan)
        resume_count = int(previous_manifest.get("resume_count", 0)) + 1
        write_json(
            manifest_path,
            {
                **previous_manifest,
                **plan,
                "status": "running",
                "resume_count": resume_count,
                "last_resumed_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
        log(log_path, f"[resume] Variant-boundary resume #{resume_count}: {run_dir}")

    log(log_path, "[phase] Discovering clean sources")
    sources = discover_sources(targets, results_base, args.skip_source_hash)
    if previous_manifest is not None and previous_manifest.get("sources") is not None:
        validate_resume_sources(previous_manifest["sources"], sources)
    running_manifest = {
        **plan,
        "status": "running",
        "resume_count": resume_count,
        "sources": sources,
    }
    if previous_manifest is not None:
        running_manifest["last_resumed_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(manifest_path, running_manifest)

    summary: dict[str, Any] = {
        "status": "running",
        "generated_at": generated_at,
        "run_dir": str(run_dir),
        "targets": targets,
        "protocol": plan["protocol"],
        "implementation_provenance": plan["implementation_provenance"],
        "sources": sources,
        "rows": [],
        "resumed": previous_manifest is not None,
        "resume_count": resume_count,
        "reused_variants": [],
        "review_required": False,
        "notes": [
            "Read-only DET provenance review over clean D-0154 snapshots.",
            "Legacy DET sample reproduces the executed two-stage code path: full medium sampling, then dynamic thinning to 10k bits.",
            "The historical comment says every 100th medium bit, but the executable step is derived from the realized medium sample count.",
            "Prefix-contiguous DET is reported only as a decomposition control, not as the legacy value.",
        ],
    }

    details: dict[str, Any] = {}
    try:
        for variant, iteration in targets.items():
            record = sources[variant]
            struct_path = Path(record["struct_path"])
            identity = build_variant_identity(plan, variant, iteration, record)
            artifact_path = run_dir / f"{variant}_det.json"
            cached = load_completed_variant(artifact_path, identity)
            if cached is not None:
                log(log_path, f"[resume] Reusing completed variant {variant}@{iteration}")
                row = cached["row"]
                detail = cached["detail"]
                summary["reused_variants"].append(variant)
            else:
                log(log_path, f"[variant] {variant}@{iteration}")
                legacy_bits, legacy_sample = build_legacy_det_bits(
                    struct_path,
                    int(record["sequence_length"]),
                    args.det_bits,
                    args.medium_divisor,
                    args.medium_min_step,
                    args.chunk_bytes,
                    log_path,
                )
                prefix_bits = build_prefix_det_bits(struct_path, args.det_bits)

                legacy_det = compute_det(legacy_bits, args.threshold_seed)
                prefix_det = compute_det(prefix_bits, args.threshold_seed)
                row, detail = build_variant_result(
                    variant,
                    iteration,
                    record,
                    legacy_bits,
                    prefix_bits,
                    legacy_sample,
                    legacy_det,
                    prefix_det,
                )
                write_json(
                    artifact_path,
                    {
                        "status": "completed",
                        "completed_at": datetime.now().isoformat(timespec="seconds"),
                        "execution_identity": identity,
                        "row": row,
                        "detail": detail,
                    },
                )

            summary["rows"].append(row)
            details[variant] = detail
            write_json(run_dir / "summary.json", summary)
            write_csv(run_dir / "det_reconciliation.csv", summary["rows"])
    except (Exception, KeyboardInterrupt) as exc:
        write_json(
            manifest_path,
            {
                **running_manifest,
                "status": "interrupted",
                "interrupted_at": datetime.now().isoformat(timespec="seconds"),
                "failure": {"type": type(exc).__name__, "message": str(exc)},
            },
        )
        log(log_path, f"[interrupted] {type(exc).__name__}: {exc}")
        raise

    summary["review_required"] = any(row["legacy_status"] != "match" for row in summary["rows"])
    summary["status"] = "completed"
    summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(run_dir / "raw_det_details.json", details)
    write_json(run_dir / "summary.json", summary)
    write_csv(run_dir / "det_reconciliation.csv", summary["rows"])
    write_report(run_dir / "report.md", summary)

    manifest = {
        **plan,
        "status": "completed",
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "script_sha256": code_hashes["runner"],
        "resume_count": resume_count,
        "reused_variants": summary["reused_variants"],
        "sources": sources,
        "outputs": {},
    }
    write_json(manifest_path, manifest)
    manifest["outputs"] = hash_outputs(run_dir)
    write_json(manifest_path, manifest)
    manifest_sha = sha256_file(manifest_path)
    atomic_write_text(run_dir / "manifest.sha256", f"{manifest_sha}  manifest.json\n")

    print("D-0154 DET provenance review completed.")
    print(f"Output: {run_dir}")
    print(f"Manifest SHA-256: {manifest_sha}")
    print(f"Review required: {summary['review_required']}")
    return 0


def validate_resume_plan(saved: dict[str, Any], candidate: dict[str, Any]) -> None:
    if saved.get("status") == "completed":
        raise RuntimeError("Completed DET runs cannot be resumed.")
    keys = (
        "script",
        "decision_log_context",
        "results_base",
        "run_dir",
        "targets",
        "protocol",
        "implementation_provenance",
        "code_hashes",
        "resume_protocol",
    )
    for key in keys:
        if saved.get(key) != candidate.get(key):
            raise RuntimeError(f"Resume rejected: plan field {key!r} changed.")


def validate_resume_sources(
    saved: dict[str, dict[str, Any]], candidate: dict[str, dict[str, Any]]
) -> None:
    if saved.keys() != candidate.keys():
        raise RuntimeError("Resume rejected: source variants changed.")
    for variant in saved:
        if saved[variant] != candidate[variant]:
            raise RuntimeError(f"Resume rejected: source identity changed for {variant}.")


def build_variant_identity(
    plan: dict[str, Any],
    variant: str,
    iteration: int,
    source: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "decision_log_context": plan["decision_log_context"],
        "targets": plan["targets"],
        "variant": variant,
        "iteration": iteration,
        "source": source,
        "protocol": plan["protocol"],
        "implementation_provenance": plan["implementation_provenance"],
        "code_hashes": plan["code_hashes"],
    }


def load_completed_variant(
    path: Path, expected_identity: dict[str, Any]
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    artifact = read_json(path)
    if artifact.get("status") != "completed":
        raise RuntimeError(f"Resume rejected: incomplete variant artifact {path.name}.")
    if artifact.get("execution_identity") != expected_identity:
        raise RuntimeError(f"Resume rejected: identity mismatch in {path.name}.")
    if not isinstance(artifact.get("row"), dict) or not isinstance(
        artifact.get("detail"), dict
    ):
        raise RuntimeError(f"Resume rejected: malformed variant artifact {path.name}.")
    return artifact


def build_variant_result(
    variant: str,
    iteration: int,
    source: dict[str, Any],
    legacy_bits: str,
    prefix_bits: str,
    legacy_sample: dict[str, Any],
    legacy_det: dict[str, Any],
    prefix_det: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    paper_value = PAPER_DET.get(variant)
    row = {
        "variant": variant,
        "iteration": iteration,
        "paper_det": paper_value,
        "paper_display": display_det(paper_value),
        "legacy_det": legacy_det.get("determinism"),
        "legacy_display": display_det(legacy_det.get("determinism")),
        "prefix_det": prefix_det.get("determinism"),
        "prefix_display": display_det(prefix_det.get("determinism")),
        "legacy_status": compare_display(paper_value, legacy_det.get("determinism")),
        "prefix_status": compare_display(paper_value, prefix_det.get("determinism")),
        "legacy_sample_bits": len(legacy_bits),
        "prefix_bits": len(prefix_bits),
        "medium_step": legacy_sample["medium_step"],
        "medium_sample_count": legacy_sample["medium_sample_count"],
        "representative_step": legacy_sample["representative_step"],
        "legacy_observable_interval": legacy_sample["legacy_observable_interval"],
        "observable_bits_scanned": legacy_sample["observable_bits_scanned"],
        "legacy_sample_sha256": legacy_sample["sample_sha256"],
    }
    detail = {
        "variant": variant,
        "iteration": iteration,
        "source": source,
        "legacy_sample": legacy_sample,
        "legacy_det": legacy_det,
        "prefix_det": prefix_det,
    }
    return row, detail


def build_legacy_det_bits(
    struct_path: Path,
    total_chars_hint: int,
    det_bits: int,
    medium_divisor: int,
    medium_min_step: int,
    chunk_bytes: int,
    log_path: Path,
) -> tuple[str, dict[str, Any]]:
    medium_step = (
        max(medium_min_step, total_chars_hint // medium_divisor)
        if total_chars_hint > 0
        else medium_min_step
    )
    medium_samples = bytearray()
    observable_index = 0
    next_capture = 0
    started = time.time()
    loader = StreamingPhiLoader(str(struct_path))
    log(
        log_path,
        "  DET legacy sample: "
        f"medium_step={medium_step:,}, target={det_bits:,}; scanning full observable stream",
    )
    for chunk in loader.iter_bits_numpy(chunk_bytes=chunk_bytes):
        chunk_len = len(chunk)
        chunk_start = observable_index
        chunk_end = chunk_start + chunk_len
        if next_capture < chunk_end:
            if next_capture < chunk_start:
                missed = (chunk_start - next_capture + medium_step - 1) // medium_step
                next_capture += missed * medium_step
            if next_capture < chunk_end:
                start_offset = next_capture - chunk_start
                indices = np.arange(start_offset, chunk_len, medium_step)
                if len(indices):
                    medium_samples.extend(chunk[indices].tobytes())
                    next_capture = chunk_start + int(indices[-1]) + medium_step
        observable_index = chunk_end
        if observable_index and observable_index % 1_000_000_000 < chunk_len:
            elapsed = max(time.time() - started, 1e-9)
            log(
                log_path,
                f"  DET medium progress: observable={observable_index:,}; "
                f"medium={len(medium_samples):,}; rate={observable_index / elapsed / 1e6:.2f} Mobs/s",
            )

    representative_step = max(1, len(medium_samples) // det_bits)
    representative = medium_samples[::representative_step][:det_bits]
    if len(representative) != det_bits:
        raise RuntimeError(
            f"Legacy DET sample underfilled: expected {det_bits:,}, got {len(representative):,} "
            f"from {len(medium_samples):,} medium samples."
        )
    bits = bytes(representative).translate(bytes.maketrans(b"\x00\x01", b"01")).decode("ascii")
    observable_interval = medium_step * representative_step
    return bits, {
        "sample_protocol": "full_medium_sample_then_dynamic_thinning",
        "medium_step": medium_step,
        "total_chars_hint": total_chars_hint,
        "medium_sample_count": len(medium_samples),
        "representative_step": representative_step,
        "legacy_observable_interval": observable_interval,
        "target_bits": det_bits,
        "sample_bits": len(bits),
        "sample_sha256": hashlib.sha256(bits.encode("ascii")).hexdigest().upper(),
        "observable_bits_scanned": observable_index,
        "elapsed_seconds": round(time.time() - started, 3),
    }


def build_prefix_det_bits(struct_path: Path, det_bits: int) -> str:
    loader = StreamingPhiLoader(str(struct_path))
    chunks: list[str] = []
    consumed = 0
    for chunk in loader.iter_bits_numpy(chunk_bytes=1_000_000):
        need = det_bits - consumed
        if need <= 0:
            break
        selected = chunk[:need]
        chunks.append("".join("1" if bit else "0" for bit in selected))
        consumed += len(selected)
    return "".join(chunks)


def compute_det(bits: str, seed: int) -> dict[str, Any]:
    np.random.seed(seed)
    result = recurrence_analysis(bits, max_bits=len(bits), verbose=False)
    result["threshold_seed"] = seed
    result["input_bits"] = len(bits)
    return result


def discover_sources(targets: dict[str, int], results_base: Path, skip_hash: bool) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    snapshot_root = results_base / "level0" / "phi_snapshots"
    report_root = results_base / "level0" / "reports"
    for variant, iteration in targets.items():
        struct_path = snapshot_root / f"var_{variant}" / f"phi_iter{iteration}.struct.gz"
        metadata_path = snapshot_root / f"var_{variant}" / f"phi_iter{iteration}.json"
        report_path = latest_report(report_root, variant, iteration)
        if not struct_path.exists():
            raise FileNotFoundError(f"Missing clean snapshot for {variant}@{iteration}: {struct_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata for {variant}@{iteration}: {metadata_path}")
        metadata = read_json(metadata_path)
        report = read_json(report_path) if report_path is not None else {}
        if metadata.get("sequence_length"):
            sequence_length = int(metadata["sequence_length"])
            sampling_length_source = "metadata.sequence_length"
        else:
            fallback_lengths = (
                ("metadata.observable_len", metadata.get("observable_len")),
                ("metadata.phi_length", metadata.get("phi_length")),
                ("metadata.final_length", metadata.get("final_length")),
                ("report.sequence_length", report.get("sequence_length")),
                ("report.observable_len", report.get("observable_len")),
                ("report.final_length", report.get("final_length")),
            )
            sampling_length_source, raw_length = next(
                ((name, value) for name, value in fallback_lengths if value),
                ("missing", 0),
            )
            sequence_length = int(raw_length)
        if sequence_length <= 0:
            raise ValueError(f"No sampling-length hint is derivable for {variant}@{iteration}.")
        records[variant] = {
            "variant": variant,
            "iteration": iteration,
            "struct_path": str(struct_path),
            "metadata_path": str(metadata_path),
            "report_path": None if report_path is None else str(report_path),
            "struct_bytes": struct_path.stat().st_size,
            "sequence_length": sequence_length,
            "sampling_length_source": sampling_length_source,
            "metadata_sha256": sha256_file(metadata_path),
            "report_sha256": None if report_path is None else sha256_file(report_path),
            "struct_sha256": None if skip_hash else sha256_file(struct_path),
        }
    return records


def latest_report(report_root: Path, variant: str, iteration: int) -> Path | None:
    candidates = sorted(report_root.glob(f"variant_{variant}_{iteration}_*.json"), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def load_dotenv() -> None:
    env_path = REPO_DIR / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def resolve_results_base(raw: str | None) -> Path:
    if raw:
        return Path(raw).expanduser().resolve()
    env_value = os.environ.get("HSI_RESULTS_BASE_DIR") or os.environ.get("HSI_V1_RESULTS_BASE_DIR")
    if env_value:
        return Path(env_value).expanduser().resolve()
    return (REPO_DIR / "results").resolve()


def resolve_results_path(raw: str, results_base: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    parts = path.parts
    if parts and parts[0].lower() == "results":
        return (results_base / Path(*parts[1:])).resolve()
    return (REPO_DIR / path).resolve()


def parse_targets(raw: str) -> dict[str, int]:
    targets: dict[str, int] = {}
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        variant, iteration = token.split(":", 1)
        targets[variant.strip().upper()] = int(iteration)
    if not targets:
        raise SystemExit("No targets selected.")
    return targets


def compare_display(paper_value: float | None, clean_value: Any) -> str:
    if paper_value is None and clean_value is None:
        return "not-applicable"
    if paper_value is None and clean_value is not None:
        return "new-value-not-in-paper-cell"
    if paper_value is not None and clean_value is None:
        return "missing-new-value"
    return "match" if round(float(paper_value), 2) == round(float(clean_value), 2) else "mismatch"


def display_det(value: Any) -> str:
    if value is None:
        return "---"
    return f"{float(value):.2f}"


def write_report(path: Path, summary: dict[str, Any]) -> None:
    rows = summary["rows"]
    det_bits = int(summary["protocol"]["det_bits"])
    review_count = sum(1 for row in rows if row["legacy_status"] != "match")
    lines = [
        "# D-0154 DET Provenance Review",
        "",
        f"- Status: `{summary['status']}`",
        f"- Run directory: `{summary['run_dir']}`",
        f"- Review required: `{summary['review_required']}`",
        f"- Legacy DET cells requiring review: `{review_count}`",
        "",
        "## Protocol",
        "",
        "- Read-only over clean D-0154 Level 0 snapshots.",
        f"- Legacy DET sample: full medium sample, then dynamic thinning with `max(1, len(medium)//{det_bits})`, capped at {det_bits:,} bits.",
        "- The historical source comment says every 100th medium bit; the report records the realized executable step instead of assuming it.",
        "- Prefix-contiguous DET is reported only as decomposition control.",
        "- RQA implementation is `level1_deep_analysis.py::recurrence_analysis`; threshold RNG seed is fixed.",
        "- The manifest anchors both legacy implementation files by SHA-256.",
        "",
        "## DET Reconciliation",
        "",
        "| Variant | Iter | Paper DET | Net legacy DET | Net prefix DET | Legacy status | Prefix status | Medium step | Medium count | Repr. step | Observable interval |",
        "|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['iteration']} | {row['paper_display']} | "
            f"{row['legacy_display']} | {row['prefix_display']} | `{row['legacy_status']}` | "
            f"`{row['prefix_status']}` | {row['medium_step']:,} | {row['medium_sample_count']:,} | "
            f"{row['representative_step']:,} | {row['legacy_observable_interval']:,} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- If the net legacy DET reproduces the paper display, the table cell closes without manuscript patch.",
            "- If the net legacy DET differs, escalation should use the decomposition columns before interpretation.",
        ]
    )
    atomic_write_text(path, "\n".join(lines) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, handle.getvalue())


def write_json(path: Path, payload: Any) -> None:
    serialized = json.dumps(json_safe(payload), indent=2) + "\n"
    atomic_write_text(path, serialized)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(temp_path, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
    return value


def log(log_path: Path, message: str) -> None:
    print(message, flush=True)
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def write_log_header(log_path: Path, plan: dict[str, Any]) -> None:
    text = "# D-0154 DET provenance review\n\n"
    text += json.dumps(json_safe(plan), indent=2)
    text += "\n\n"
    atomic_write_text(log_path, text)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def hash_outputs(run_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name in {"manifest.json", "manifest.sha256", "batch.log"}:
            continue
        if path.name.startswith(".") and path.name.endswith(".tmp"):
            continue
        hashes[path.relative_to(run_dir).as_posix()] = sha256_file(path)
    return hashes


def print_plan(plan: dict[str, Any]) -> None:
    print("D-0154 DET provenance-review plan (dry-run)")
    print("-" * 76)
    print(f"run_dir: {plan['run_dir']}")
    print(f"targets: {plan['targets']}")
    print("\nAdd --execute to run.")


if __name__ == "__main__":
    raise SystemExit(main())
