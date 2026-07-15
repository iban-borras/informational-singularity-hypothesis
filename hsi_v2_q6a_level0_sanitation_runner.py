#!/usr/bin/env python3
"""
Q6-A Level-0 sanitation runner for pre-patch non-B HSI variants.

The script quarantines legacy Level-0 artifacts for selected variants and then
runs clean post-patch generation from iteration 0. It is intentionally small and
boring: move old artifacts aside, run the canonical generator, write a manifest.
"""

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


DEFAULT_TARGETS = {
    "I": 23,
    "D": 20,
    "G": 20,
    "F": 20,
}


VISUALIZATION_PATTERNS = (
    "autocorr_{variant}*",
    "fft_{variant}_*",
    "fft_amp_{variant}_*",
    "growth_time_{variant}*",
    "hilbert_{variant}*",
    "raster2d_{variant}*",
    "spectrum_beta_{variant}*",
    "spectrum_enhanced_{variant}*",
)


REPO_DIR = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Quarantine pre-patch Level-0 artifacts and regenerate clean non-B variants."
    )
    parser.add_argument(
        "--variants",
        default="I,D,G,F",
        help="Comma-separated variants to sanitize. Defaults to I,D,G,F.",
    )
    parser.add_argument(
        "--targets",
        default=",".join(f"{variant}:{iteration}" for variant, iteration in DEFAULT_TARGETS.items()),
        help="Comma-separated target map, e.g. I:23,D:20,G:20,F:20.",
    )
    parser.add_argument(
        "--results-base",
        default=None,
        help="Override results base. Defaults to HSI_RESULTS_BASE_DIR/.env or hsi_agents_project/results.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/hsi_v2/q6a_level0_sanitation",
        help="Manifest/log output directory, relative to results base when rooted at results/.",
    )
    parser.add_argument("--execute", action="store_true", help="Actually move files and run generation.")
    parser.add_argument(
        "--quarantine-existing",
        action="store_true",
        help="Move existing canonical artifacts to legacy_invalidated before generation.",
    )
    parser.add_argument(
        "--force-quarantine-postpatch",
        action="store_true",
        help="Allow quarantining dirs that already contain post-patch state checkpoints.",
    )
    parser.add_argument("--no-plots", action="store_true", default=True)
    parser.add_argument("--force-compress", action="store_true", default=True)
    parser.add_argument(
        "--results-ready-timeout-seconds",
        type=int,
        default=600,
        help="Seconds to wait for the results drive to become writable before failing.",
    )
    parser.add_argument(
        "--results-ready-check-interval-seconds",
        type=float,
        default=5.0,
        help="Polling interval while waiting for the results drive.",
    )
    parser.add_argument(
        "--transient-retries",
        type=int,
        default=1,
        help="Retry a failed generation this many times after rechecking the results drive.",
    )
    parser.add_argument(
        "--no-retry-resume",
        action="store_true",
        help="Keep --no-resume on retry. By default retries resume from clean post-patch checkpoints.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    load_dotenv()

    results_base = resolve_results_base(args.results_base)
    wait_for_results_base(
        results_base,
        timeout_seconds=args.results_ready_timeout_seconds,
        interval_seconds=args.results_ready_check_interval_seconds,
    )
    output_root = resolve_results_path(args.output_dir, results_base)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = output_root / f"q6a-level0-sanitation__{timestamp}"
    log_path = run_dir / "sanitation_runner.log"
    manifest_path = run_dir / "manifest.json"

    variants = parse_variants(args.variants)
    targets = parse_targets(args.targets)
    for variant in variants:
        if variant not in targets:
            raise SystemExit(f"Missing target iteration for variant {variant}")

    plan = {
        "status": "planned" if not args.execute else "running",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": "hsi_v2_q6a_level0_sanitation_runner.py",
        "results_base": str(results_base),
        "variants": variants,
        "targets": {variant: targets[variant] for variant in variants},
        "execute": args.execute,
        "quarantine_existing": args.quarantine_existing,
        "transient_retries": args.transient_retries,
        "retry_resume_enabled": not args.no_retry_resume,
        "results_ready_timeout_seconds": args.results_ready_timeout_seconds,
        "commands": [
            build_generation_command(variant, targets[variant], args)
            for variant in variants
        ],
        "runs": [],
    }

    if not args.execute:
        print_plan(plan)
        return 0

    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, plan)
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("# Q6-A Level-0 sanitation runner\n\n")
        log.write(json.dumps(plan, indent=2))
        log.write("\n\n")

    for variant in variants:
        target_iteration = targets[variant]
        record: dict[str, Any] = {
            "variant": variant,
            "target_iteration": target_iteration,
            "started_at": datetime.now().isoformat(timespec="seconds"),
        }
        try:
            quarantine_dir = None
            resume_existing_postpatch = False
            if args.quarantine_existing:
                variant_dir = results_base / "level0" / "phi_snapshots" / f"var_{variant}"
                if (
                    variant_dir.exists()
                    and contains_state_checkpoints(variant_dir)
                    and not args.force_quarantine_postpatch
                ):
                    resume_existing_postpatch = True
                    print(
                        f"[resume] {variant_dir} contains post-patch checkpoints; "
                        "skipping quarantine and resuming this clean run.",
                        flush=True,
                    )
                else:
                    quarantine_dir = quarantine_variant(
                        results_base,
                        variant,
                        timestamp,
                        allow_postpatch=args.force_quarantine_postpatch,
                    )
            else:
                assert_no_existing_variant_dir(results_base, variant)

            record["quarantine_dir"] = str(quarantine_dir) if quarantine_dir else None
            record["resume_existing_postpatch"] = resume_existing_postpatch
            command = build_generation_command(variant, target_iteration, args)
            if resume_existing_postpatch:
                command = without_flag(command, "--no-resume")
            record["command"] = command
            code = run_command_with_retries(command, log_path, results_base, args)
            record["return_code"] = code
            record["finished_at"] = datetime.now().isoformat(timespec="seconds")
            record["status"] = "completed" if code == 0 else "failed"
            plan["runs"].append(record)
            write_json(manifest_path, plan)
            if code != 0:
                plan["status"] = "failed"
                write_json(manifest_path, plan)
                return code
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = str(exc)
            record["finished_at"] = datetime.now().isoformat(timespec="seconds")
            plan["runs"].append(record)
            plan["status"] = "failed"
            write_json(manifest_path, plan)
            raise

    plan["status"] = "completed"
    plan["finished_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(manifest_path, plan)
    print(f"Sanitation completed. Manifest: {manifest_path}")
    return 0


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


def wait_for_results_base(
    results_base: Path,
    *,
    timeout_seconds: int,
    interval_seconds: float,
) -> None:
    """Wait until the results drive is present and writable.

    Windows can return before an external drive has fully woken from hibernation.
    A tiny write/delete probe is safer than trusting that the path merely exists.
    """
    deadline = time.time() + max(1, timeout_seconds)
    interval = max(1.0, interval_seconds)
    last_error: Exception | None = None
    warned = False

    while True:
        try:
            results_base.mkdir(parents=True, exist_ok=True)
            probe = results_base / ".hsi_write_probe.tmp"
            with open(probe, "w", encoding="utf-8") as handle:
                handle.write(datetime.now().isoformat(timespec="seconds"))
            try:
                probe.unlink()
            except FileNotFoundError:
                pass
            if warned:
                print(f"[drive] Results base is writable again: {results_base}", flush=True)
            return
        except OSError as exc:
            last_error = exc
            if time.time() >= deadline:
                raise RuntimeError(
                    f"Results base did not become writable within {timeout_seconds}s: "
                    f"{results_base} ({last_error})"
                ) from exc
            if not warned:
                print(
                    f"[drive] Waiting for writable results base: {results_base} ({exc})",
                    flush=True,
                )
                warned = True
            time.sleep(interval)


def parse_variants(raw: str) -> list[str]:
    variants = []
    for item in raw.split(","):
        variant = item.strip().upper()
        if variant:
            variants.append(variant)
    if not variants:
        raise SystemExit("No variants selected")
    return variants


def parse_targets(raw: str) -> dict[str, int]:
    targets: dict[str, int] = {}
    for item in raw.split(","):
        if not item.strip():
            continue
        variant, iteration = item.split(":", 1)
        targets[variant.strip().upper()] = int(iteration)
    return targets


def build_generation_command(variant: str, target_iteration: int, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(REPO_DIR / "level0_generate.py"),
        "--variant",
        variant,
        "--iterations",
        str(target_iteration),
        "--no-resume",
    ]
    if args.no_plots:
        command.append("--no-plots")
    if args.force_compress:
        command.append("--force-compress")
    return command


def assert_no_existing_variant_dir(results_base: Path, variant: str) -> None:
    variant_dir = results_base / "level0" / "phi_snapshots" / f"var_{variant}"
    if variant_dir.exists():
        raise RuntimeError(
            f"{variant_dir} exists. Use --quarantine-existing to move legacy artifacts first."
        )


def quarantine_variant(
    results_base: Path,
    variant: str,
    timestamp: str,
    *,
    allow_postpatch: bool,
) -> Path:
    level0 = results_base / "level0"
    quarantine_root = level0 / "legacy_invalidated" / f"q6a_prepatch_{variant}_legacy_{timestamp}"
    quarantine_root.mkdir(parents=True, exist_ok=True)

    variant_dir = level0 / "phi_snapshots" / f"var_{variant}"
    if variant_dir.exists():
        if contains_state_checkpoints(variant_dir) and not allow_postpatch:
            raise RuntimeError(
                f"{variant_dir} contains post-patch state checkpoints. Refusing to quarantine it. "
                "If this is intentional, pass --force-quarantine-postpatch."
            )
        move_path(variant_dir, quarantine_root / "phi_snapshots" / f"var_{variant}")

    reports_dir = level0 / "reports"
    quarantine_reports = quarantine_root / "reports"
    if reports_dir.exists():
        for report_path in sorted(reports_dir.glob(f"variant_{variant}*")):
            move_path(report_path, quarantine_reports / report_path.name)

    visualizations_dir = level0 / "visualizations"
    quarantine_visualizations = quarantine_root / "visualizations"
    if visualizations_dir.exists():
        for pattern in VISUALIZATION_PATTERNS:
            for viz_path in sorted(visualizations_dir.glob(pattern.format(variant=variant))):
                move_path(viz_path, quarantine_visualizations / viz_path.name)

    return quarantine_root


def contains_state_checkpoints(variant_dir: Path) -> bool:
    return any(variant_dir.glob("phi_iter*.state.txt.gz"))


def move_path(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise RuntimeError(f"Refusing to overwrite quarantine destination: {destination}")
    shutil.move(str(source), str(destination))
    print(f"[quarantine] {source} -> {destination}", flush=True)


def run_command(command: list[str], log_path: Path) -> int:
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("HSI_PROGRESS_MODE", "log")
    env.setdefault("HSI_STREAM_CHILD_LOG", "1")
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


def run_command_with_retries(
    command: list[str],
    log_path: Path,
    results_base: Path,
    args: argparse.Namespace,
) -> int:
    max_retries = max(0, int(args.transient_retries))
    current_command = list(command)

    for attempt in range(max_retries + 1):
        wait_for_results_base(
            results_base,
            timeout_seconds=args.results_ready_timeout_seconds,
            interval_seconds=args.results_ready_check_interval_seconds,
        )
        code = run_command(current_command, log_path)
        if code == 0 or attempt >= max_retries:
            return code

        retry_number = attempt + 1
        if not args.no_retry_resume:
            current_command = without_flag(command, "--no-resume")
        message = (
            f"[retry] Generation failed with exit code {code}; "
            f"retry {retry_number}/{max_retries} after drive recheck. "
            f"resume_enabled={not args.no_retry_resume}"
        )
        print(message, flush=True)
        with open(log_path, "a", encoding="utf-8") as log:
            log.write("\n" + message + "\n")

    return code


def without_flag(command: list[str], flag: str) -> list[str]:
    return [part for part in command if part != flag]


def print_plan(plan: dict[str, Any]) -> None:
    print("Q6-A Level-0 sanitation plan (dry run)")
    print("-" * 72)
    print(f"results_base: {plan['results_base']}")
    print(f"variants: {', '.join(plan['variants'])}")
    print(f"targets: {plan['targets']}")
    print("commands:")
    for command in plan["commands"]:
        print("  " + " ".join(command))
    print("\nAdd --execute --quarantine-existing to run.")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())
