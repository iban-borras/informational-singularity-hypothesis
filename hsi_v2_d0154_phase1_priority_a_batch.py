#!/usr/bin/env python3
"""
D-0154 Priority-A Phase 1 rederivation batch.

Runs the canonical Phase 1 tower derivation for clean post-patch variants and
then emits compact Phase 1 and transport reports plus a batch manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_DIR = Path(__file__).resolve().parent
DEFAULT_TARGETS = "E:24,I:23,D:20,G:20,F:20"
DEFAULT_SCALES = "8,12,16,20,24,28,32"
DEFAULT_POLICIES = "prefix,suffix"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run D-0154 Priority-A clean Phase 1 summaries for sanitized variants."
    )
    parser.add_argument(
        "--targets",
        default=DEFAULT_TARGETS,
        help="Comma-separated target map, e.g. E:24,I:23,D:20,G:20,F:20.",
    )
    parser.add_argument("--segment-bits", type=int, default=1_000_000)
    parser.add_argument("--num-segments", type=int, default=3)
    parser.add_argument("--segment-offset-bits", type=int, default=0)
    parser.add_argument("--scales", default=DEFAULT_SCALES)
    parser.add_argument("--policies", default=DEFAULT_POLICIES)
    parser.add_argument("--min-count-floor", type=int, default=16)
    parser.add_argument("--min-count-rate", type=float, default=1e-6)
    parser.add_argument("--min-segment-support", type=int, default=None)
    parser.add_argument("--cv-max", type=float, default=1.5)
    parser.add_argument(
        "--results-base",
        default=None,
        help="Override results base. Defaults to HSI_RESULTS_BASE_DIR/.env or hsi_agents_project/results.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/hsi_v2/d0154_priority_a_phase1",
        help="Batch output root. Relative results/... paths are rooted at HSI_RESULTS_BASE_DIR.",
    )
    parser.add_argument("--tail-edges", type=int, default=2)
    parser.add_argument("--skip-reports", action="store_true")
    parser.add_argument("--with-hints", action="store_true")
    parser.add_argument("--quiet-child", action="store_true", help="Pass --quiet to child Phase 1 runs.")
    parser.add_argument("--execute", action="store_true", help="Actually run the batch. Omit for dry-run.")
    return parser


def main() -> int:
    configure_stdout()
    args = build_parser().parse_args()
    load_dotenv()

    targets = parse_targets(args.targets)
    variants = list(targets)
    results_base = resolve_results_base(args.results_base)
    output_root = resolve_results_path(args.output_dir, results_base)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    batch_dir = output_root / f"d0154-priority-a-phase1__{timestamp}"
    phase1_dir = batch_dir / "phase1_runs"
    report_dir = batch_dir / "reports"
    manifest_path = batch_dir / "manifest.json"
    log_path = batch_dir / "batch.log"

    run_commands = [
        build_phase1_command(variant, iteration, phase1_dir, args)
        for variant, iteration in targets.items()
    ]
    report_commands = build_report_commands(variants, phase1_dir, report_dir, args)

    manifest: dict[str, Any] = {
        "status": "planned" if not args.execute else "running",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": "hsi_v2_d0154_phase1_priority_a_batch.py",
        "decision_log_context": "D-0154 Priority A Phase 1 clean summaries",
        "results_base": str(results_base),
        "batch_dir": str(batch_dir),
        "phase1_dir": str(phase1_dir),
        "report_dir": str(report_dir),
        "targets": targets,
        "protocol": {
            "segment_bits": args.segment_bits,
            "num_segments": args.num_segments,
            "segment_offset_bits": args.segment_offset_bits,
            "scales": parse_int_list(args.scales),
            "policies": parse_str_list(args.policies),
            "min_count_floor": args.min_count_floor,
            "min_count_rate": args.min_count_rate,
            "min_segment_support": args.min_segment_support,
            "cv_max": args.cv_max,
            "tail_edges": args.tail_edges,
        },
        "commands": {
            "phase1_runs": run_commands,
            "reports": report_commands,
        },
        "runs": [],
        "reports": [],
    }

    if not args.execute:
        print_plan(manifest)
        return 0

    batch_dir.mkdir(parents=True, exist_ok=True)
    phase1_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, manifest)
    write_log_header(log_path, manifest)

    for variant, command in zip(variants, run_commands):
        record: dict[str, Any] = {
            "variant": variant,
            "iteration": targets[variant],
            "command": command,
            "started_at": datetime.now().isoformat(timespec="seconds"),
        }
        started = time.time()
        code = run_command(command, log_path)
        record["return_code"] = code
        record["elapsed_seconds"] = round(time.time() - started, 3)
        record["finished_at"] = datetime.now().isoformat(timespec="seconds")
        record["status"] = "completed" if code == 0 else "failed"
        record["run_dir"] = str(find_latest_run_dir(phase1_dir, variant, targets[variant]))
        manifest["runs"].append(record)
        write_json(manifest_path, manifest)
        if code != 0:
            manifest["status"] = "failed"
            manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
            write_json(manifest_path, manifest)
            return code

    if not args.skip_reports:
        for command in report_commands:
            record = {
                "command": command,
                "started_at": datetime.now().isoformat(timespec="seconds"),
            }
            started = time.time()
            code = run_command(command, log_path)
            record["return_code"] = code
            record["elapsed_seconds"] = round(time.time() - started, 3)
            record["finished_at"] = datetime.now().isoformat(timespec="seconds")
            record["status"] = "completed" if code == 0 else "failed"
            manifest["reports"].append(record)
            write_json(manifest_path, manifest)
            if code != 0:
                manifest["status"] = "failed"
                manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
                write_json(manifest_path, manifest)
                return code

    manifest["artifact_hashes"] = hash_batch_artifacts(batch_dir)
    manifest["status"] = "completed"
    manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(manifest_path, manifest)
    manifest_hash = sha256_file(manifest_path)
    (batch_dir / "manifest.sha256").write_text(f"{manifest_hash}  manifest.json\n", encoding="utf-8")
    print(f"D-0154 Priority-A Phase 1 batch completed.")
    print(f"Manifest: {manifest_path}")
    print(f"Manifest SHA-256: {manifest_hash}")
    return 0


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
    for item in raw.split(","):
        if not item.strip():
            continue
        variant, iteration = item.split(":", 1)
        variant = variant.strip().upper()
        if not variant:
            raise SystemExit(f"Invalid target entry: {item}")
        targets[variant] = int(iteration)
    if not targets:
        raise SystemExit("No targets selected.")
    return targets


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_str_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def build_phase1_command(
    variant: str,
    iteration: int,
    phase1_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        sys.executable,
        str(REPO_DIR / "hsi_v2_phase1_run.py"),
        "--variant",
        variant,
        "--iteration",
        str(iteration),
        "--segment-bits",
        str(args.segment_bits),
        "--num-segments",
        str(args.num_segments),
        "--segment-offset-bits",
        str(args.segment_offset_bits),
        "--scales",
        args.scales,
        "--policies",
        args.policies,
        "--min-count-floor",
        str(args.min_count_floor),
        "--min-count-rate",
        str(args.min_count_rate),
        "--cv-max",
        str(args.cv_max),
        "--output-dir",
        str(phase1_dir),
    ]
    if args.min_segment_support is not None:
        command.extend(["--min-segment-support", str(args.min_segment_support)])
    if args.quiet_child:
        command.append("--quiet")
    return command


def build_report_commands(
    variants: list[str],
    phase1_dir: Path,
    report_dir: Path,
    args: argparse.Namespace,
) -> list[list[str]]:
    common = [
        "--phase1-dir",
        str(phase1_dir),
        "--report-dir",
        str(report_dir),
        "--variants",
        ",".join(variants),
        "--segment-bits",
        str(args.segment_bits),
        "--num-segments",
        str(args.num_segments),
        "--segment-offset-bits",
        str(args.segment_offset_bits),
        "--scales",
        args.scales,
        "--policies",
        args.policies,
        "--no-family-inference",
    ]
    if args.with_hints:
        common.append("--with-hints")
    return [
        [sys.executable, str(REPO_DIR / "hsi_v2_phase1_report.py"), *common],
        [
            sys.executable,
            str(REPO_DIR / "hsi_v2_phase1_transport_report.py"),
            *common,
            "--tail-edges",
            str(args.tail_edges),
        ],
    ]


def run_command(command: list[str], log_path: Path) -> int:
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("HSI_PROGRESS_MODE", "log")
    print("[run] " + " ".join(command), flush=True)
    with open(log_path, "a", encoding="utf-8") as log:
        log.write("\n\n## RUN\n\n")
        log.write(" ".join(command))
        log.write("\n\n```text\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=str(REPO_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        code = process.wait()
        log.write("\n```\n")
        log.write(f"\nReturn code: {code}\n")
    return code


def find_latest_run_dir(phase1_dir: Path, variant: str, iteration: int) -> str | None:
    prefix = f"phase1-tower__var-{variant}__iter-{iteration}__"
    candidates = [path for path in phase1_dir.glob(prefix + "*") if path.is_dir()]
    if not candidates:
        return None
    return str(max(candidates, key=lambda path: path.stat().st_mtime).resolve())


def hash_batch_artifacts(batch_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(batch_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name in {"batch.log", "manifest.json", "manifest.sha256"}:
            continue
        relative = path.relative_to(batch_dir).as_posix()
        hashes[relative] = sha256_file(path)
    return hashes


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_log_header(log_path: Path, manifest: dict[str, Any]) -> None:
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("# D-0154 Priority-A Phase 1 batch\n\n")
        log.write(json.dumps(manifest, indent=2))
        log.write("\n")


def print_plan(manifest: dict[str, Any]) -> None:
    print("D-0154 Priority-A Phase 1 batch plan (dry-run)")
    print("-" * 76)
    print(f"batch_dir: {manifest['batch_dir']}")
    print(f"targets: {manifest['targets']}")
    print("phase1 commands:")
    for command in manifest["commands"]["phase1_runs"]:
        print("  " + " ".join(command))
    print("report commands:")
    for command in manifest["commands"]["reports"]:
        print("  " + " ".join(command))
    print("\nAdd --execute to run.")


if __name__ == "__main__":
    raise SystemExit(main())
