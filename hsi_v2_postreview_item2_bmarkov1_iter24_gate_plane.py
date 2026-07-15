#!/usr/bin/env python3
"""
Post-review Item 2: B vs B-markov1 gate-plane depth check.

Executes the frozen D-0155 v1.0 contract:
- observed B@20 and B@24
- B-markov1@20 and B-markov1@24 for six canonical seeds
- same Phase 1 protocol for every run
- repo-facing outcome based on G(20/24), D(20/24), Delta_G, Delta_D
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = REPO_DIR.parent
PREREG_PATH = PROJECT_ROOT / "prereg_postreview_item2_bmarkov1_iter24_gate_plane_v1.0.md"
PREREG_SHA256 = "3181DD505E072FB26BF94083C03B8C7BC23FC034B6B0759633CD04CED21F2D5D"

DEFAULT_SEEDS = "17,101,211,307,401,503"
DEFAULT_DEPTHS = "20,24"
DEFAULT_SCALES = "8,12,16,20,24,28,32,40,48"
DEFAULT_POLICIES = "prefix,suffix"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute D-0155 Item 2 B-markov1@24 gate-plane contract."
    )
    parser.add_argument("--depths", default=DEFAULT_DEPTHS)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--segment-bits", type=int, default=1_000_000)
    parser.add_argument("--num-segments", type=int, default=3)
    parser.add_argument("--segment-offset-bits", type=int, default=0)
    parser.add_argument("--scales", default=DEFAULT_SCALES)
    parser.add_argument("--policies", default=DEFAULT_POLICIES)
    parser.add_argument("--min-count-floor", type=int, default=16)
    parser.add_argument("--min-count-rate", type=float, default=1e-6)
    parser.add_argument("--min-segment-support", type=int, default=None)
    parser.add_argument("--cv-max", type=float, default=1.5)
    parser.add_argument("--tail-edges", type=int, default=2)
    parser.add_argument(
        "--output-dir",
        default="results/hsi_v2/postreview_item2_bmarkov1_iter24_gate_plane",
    )
    parser.add_argument(
        "--results-base",
        default=None,
        help="Override results base. Defaults to HSI_RESULTS_BASE_DIR/.env or hsi_agents_project/results.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse exact matching run dirs inside this output root if present.",
    )
    parser.add_argument("--quiet-child", action="store_true", help="Pass --quiet to Phase 1 child runs.")
    parser.add_argument("--execute", action="store_true", help="Actually execute. Omit for dry-run.")
    return parser


def main() -> int:
    configure_stdout()
    args = build_parser().parse_args()
    load_dotenv()
    verify_prereg()

    depths = parse_int_list(args.depths)
    seeds = parse_int_list(args.seeds)
    if depths != [20, 24]:
        raise SystemExit("Frozen contract requires --depths 20,24.")
    if seeds != [17, 101, 211, 307, 401, 503]:
        raise SystemExit("Frozen contract requires --seeds 17,101,211,307,401,503.")

    results_base = resolve_results_base(args.results_base)
    output_root = resolve_results_path(args.output_dir, results_base)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = output_root / f"postreview-item2-bmarkov1-gate-plane__{timestamp}"
    phase1_dir = run_dir / "phase1_runs"
    log_path = run_dir / "run.log"
    manifest_path = run_dir / "manifest.json"

    observed_commands = [
        build_phase1_command(depth, phase1_dir, args)
        for depth in depths
    ]
    null_commands = [
        build_phase1_command(depth, phase1_dir, args, null_seed=seed)
        for depth in depths
        for seed in seeds
    ]
    manifest: dict[str, Any] = {
        "status": "planned" if not args.execute else "running",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": "hsi_v2_postreview_item2_bmarkov1_iter24_gate_plane.py",
        "preregistration": {
            "path": str(PREREG_PATH),
            "sha256": PREREG_SHA256,
        },
        "results_base": str(results_base),
        "run_dir": str(run_dir),
        "phase1_dir": str(phase1_dir),
        "protocol": protocol_dict(args, depths, seeds),
        "commands": {
            "observed": observed_commands,
            "markov1": null_commands,
        },
        "phase1_runs": [],
        "outputs": {},
    }

    if not args.execute:
        print_plan(manifest)
        return 0

    run_dir.mkdir(parents=True, exist_ok=True)
    phase1_dir.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, manifest)
    write_log_header(log_path, manifest)

    try:
        for depth, command in zip(depths, observed_commands):
            record = execute_or_reuse(
                command,
                phase1_dir,
                log_path,
                depth=depth,
                seed=None,
                args=args,
            )
            manifest["phase1_runs"].append(record)
            write_json(manifest_path, manifest)
            if record["status"] != "completed":
                return fail_manifest(manifest, manifest_path, "phase1 observed run failed")

        for depth in depths:
            for seed in seeds:
                command = build_phase1_command(depth, phase1_dir, args, null_seed=seed)
                record = execute_or_reuse(
                    command,
                    phase1_dir,
                    log_path,
                    depth=depth,
                    seed=seed,
                    args=args,
                )
                manifest["phase1_runs"].append(record)
                write_json(manifest_path, manifest)
                if record["status"] != "completed":
                    return fail_manifest(manifest, manifest_path, "phase1 markov1 run failed")

        readout = build_readout(run_dir, phase1_dir, depths, seeds, args)
        summary_path = run_dir / "summary.json"
        report_path = run_dir / "report.md"
        per_depth_path = run_dir / "per_depth_gate_plane.csv"
        per_seed_path = run_dir / "per_seed_markov1_gate_plane.csv"
        write_json(summary_path, readout["summary"])
        write_report(report_path, readout["summary"])
        write_per_depth_csv(per_depth_path, readout["per_depth_rows"])
        write_per_seed_csv(per_seed_path, readout["per_seed_rows"])

        manifest["outputs"] = {
            "summary": str(summary_path),
            "report": str(report_path),
            "per_depth_gate_plane": str(per_depth_path),
            "per_seed_markov1_gate_plane": str(per_seed_path),
        }
        manifest["source_hashes"] = source_hashes(results_base, depths)
        manifest["status"] = "completed"
        manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
        write_json(manifest_path, manifest)
        manifest["artifact_hashes"] = hash_primary_artifacts(run_dir)
        write_json(manifest_path, manifest)
        manifest_hash = sha256_file(manifest_path)
        (run_dir / "manifest.sha256").write_text(f"{manifest_hash}  manifest.json\n", encoding="utf-8")

        print("D-0155 Item 2 completed.")
        print(f"Outcome: {readout['summary']['outcome']}")
        print(f"Summary: {summary_path}")
        print(f"Report: {report_path}")
        print(f"Manifest: {manifest_path}")
        print(f"Manifest SHA-256: {manifest_hash}")
        return 0
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
        write_json(manifest_path, manifest)
        raise


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


def verify_prereg() -> None:
    if not PREREG_PATH.exists():
        raise SystemExit(f"Missing frozen preregistration: {PREREG_PATH}")
    actual = sha256_file(PREREG_PATH)
    if actual != PREREG_SHA256:
        raise SystemExit(
            f"Frozen preregistration hash mismatch: expected {PREREG_SHA256}, got {actual}"
        )


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


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def protocol_dict(args: argparse.Namespace, depths: list[int], seeds: list[int]) -> dict[str, Any]:
    return {
        "variant": "B",
        "depths": depths,
        "null_model": "markov1",
        "null_seeds": seeds,
        "segment_bits": args.segment_bits,
        "num_segments": args.num_segments,
        "segment_offset_bits": args.segment_offset_bits,
        "scales": parse_int_list(args.scales),
        "policies": [item.strip() for item in args.policies.split(",") if item.strip()],
        "min_count_floor": args.min_count_floor,
        "min_count_rate": args.min_count_rate,
        "min_segment_support": args.min_segment_support,
        "cv_max": args.cv_max,
        "tail_edges": args.tail_edges,
        "G_threshold": 0.30,
        "D_threshold": 0.20,
        "Delta_G_stability_band": 0.05,
    }


def build_phase1_command(
    depth: int,
    phase1_dir: Path,
    args: argparse.Namespace,
    *,
    null_seed: int | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(REPO_DIR / "hsi_v2_phase1_run.py"),
        "--variant",
        "B",
        "--iteration",
        str(depth),
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
    if null_seed is not None:
        command.extend(["--null-model", "markov1", "--null-seed", str(null_seed)])
    if args.quiet_child:
        command.append("--quiet")
    return command


def execute_or_reuse(
    command: list[str],
    phase1_dir: Path,
    log_path: Path,
    *,
    depth: int,
    seed: int | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    existing = find_matching_run(phase1_dir, depth=depth, seed=seed, args=args)
    if args.reuse_existing and existing is not None:
        return {
            "depth": depth,
            "seed": seed,
            "kind": "markov1" if seed is not None else "observed",
            "status": "completed",
            "reused": True,
            "run_dir": str(existing),
        }

    record: dict[str, Any] = {
        "depth": depth,
        "seed": seed,
        "kind": "markov1" if seed is not None else "observed",
        "command": command,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "reused": False,
    }
    started = time.time()
    code = run_command(command, log_path)
    record["return_code"] = code
    record["elapsed_seconds"] = round(time.time() - started, 3)
    record["finished_at"] = datetime.now().isoformat(timespec="seconds")
    record["status"] = "completed" if code == 0 else "failed"
    found = find_matching_run(phase1_dir, depth=depth, seed=seed, args=args)
    record["run_dir"] = str(found) if found is not None else None
    return record


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


def find_matching_run(
    phase1_dir: Path,
    *,
    depth: int,
    seed: int | None,
    args: argparse.Namespace,
) -> Path | None:
    root = phase1_dir / "nulls" if seed is not None else phase1_dir
    if not root.exists():
        return None
    matches: list[Path] = []
    for dataset_path in root.glob("*/dataset.json"):
        try:
            dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not dataset_matches(dataset, depth=depth, seed=seed, args=args):
            continue
        if not (dataset_path.parent / "phase1_summary.json").exists():
            continue
        matches.append(dataset_path.parent)
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime).resolve()


def dataset_matches(dataset: dict[str, Any], *, depth: int, seed: int | None, args: argparse.Namespace) -> bool:
    config = dataset.get("config", {})
    if str(config.get("source_variant", config.get("variant", ""))).upper() != "B":
        return False
    if config.get("source_iteration", config.get("iteration")) != depth:
        return False
    if config.get("segment_bits") != args.segment_bits:
        return False
    if config.get("num_segments") != args.num_segments:
        return False
    if config.get("segment_offset_bits", 0) != args.segment_offset_bits:
        return False
    if list(config.get("scales", [])) != parse_int_list(args.scales):
        return False
    if list(config.get("policies", [])) != [item.strip() for item in args.policies.split(",") if item.strip()]:
        return False
    if config.get("min_count_floor") != args.min_count_floor:
        return False
    if float(config.get("min_count_rate")) != float(args.min_count_rate):
        return False
    if config.get("min_segment_support_requested") != args.min_segment_support:
        return False
    if float(config.get("cv_max")) != float(args.cv_max):
        return False
    if seed is None:
        return config.get("null_model") is None
    return config.get("null_model") == "markov1" and config.get("null_seed") == seed


def build_readout(
    run_dir: Path,
    phase1_dir: Path,
    depths: list[int],
    seeds: list[int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    rows_by_depth: dict[int, dict[str, Any]] = {}
    per_seed_rows: list[dict[str, Any]] = []
    per_depth_rows: list[dict[str, Any]] = []

    for depth in depths:
        observed_run = require_matching_run(phase1_dir, depth=depth, seed=None, args=args)
        observed_point = gate_point(observed_run, tail_edges=args.tail_edges)
        seed_points = []
        for seed in seeds:
            null_run = require_matching_run(phase1_dir, depth=depth, seed=seed, args=args)
            point = gate_point(null_run, tail_edges=args.tail_edges)
            point["seed"] = seed
            seed_points.append(point)
            per_seed_rows.append(
                {
                    "depth": depth,
                    "seed": seed,
                    "retention_markov1": point["retention_last"],
                    "active_mean_tail_markov1": point["active_mean_tail"],
                    "run_dir": point["run_dir"],
                }
            )

        max_retention = max(item["retention_last"] for item in seed_points)
        nearest_distance = min(
            euclidean(
                observed_point["retention_last"],
                observed_point["active_mean_tail"],
                item["retention_last"],
                item["active_mean_tail"],
            )
            for item in seed_points
        )
        G = observed_point["retention_last"] - max_retention
        D = nearest_distance
        depth_row = {
            "depth": depth,
            "retention_B": observed_point["retention_last"],
            "active_mean_tail_B": observed_point["active_mean_tail"],
            "max_retention_markov1": max_retention,
            "nearest_distance_markov1": nearest_distance,
            "G": G,
            "D": D,
            "observed_run_dir": observed_point["run_dir"],
        }
        rows_by_depth[depth] = depth_row
        per_depth_rows.append(depth_row)

    G20 = rows_by_depth[20]["G"]
    D20 = rows_by_depth[20]["D"]
    G24 = rows_by_depth[24]["G"]
    D24 = rows_by_depth[24]["D"]
    delta_G = G24 - G20
    delta_D = D24 - D20
    baseline_established = G20 >= 0.30 and D20 >= 0.20
    if not baseline_established:
        outcome = "markov1-baseline-not-established"
    elif G24 >= 0.30 and D24 >= 0.20 and delta_G >= 0.05:
        outcome = "markov1-separation-expands-at-iter24"
    elif G24 >= 0.30 and D24 >= 0.20 and -0.05 < delta_G < 0.05:
        outcome = "markov1-separation-stable-at-iter24"
    else:
        outcome = "markov1-separation-contracts-at-iter24"

    summary = {
        "status": "completed",
        "outcome": outcome,
        "G20": G20,
        "D20": D20,
        "G24": G24,
        "D24": D24,
        "Delta_G": delta_G,
        "Delta_D": delta_D,
        "baseline_control": {
            "baseline_separation_established": baseline_established,
            "rule": "G(20) >= 0.30 AND D(20) >= 0.20",
            "blocked_relative_comparison": not baseline_established,
        },
        "thresholds": {
            "G": 0.30,
            "D": 0.20,
            "Delta_G_expand": 0.05,
            "Delta_G_contract": -0.05,
        },
        "depths": per_depth_rows,
        "markov1_seeds": per_seed_rows,
        "run_dir": str(run_dir),
        "preregistration_sha256": PREREG_SHA256,
    }
    return {
        "summary": summary,
        "per_depth_rows": per_depth_rows,
        "per_seed_rows": per_seed_rows,
    }


def require_matching_run(phase1_dir: Path, *, depth: int, seed: int | None, args: argparse.Namespace) -> Path:
    run_dir = find_matching_run(phase1_dir, depth=depth, seed=seed, args=args)
    if run_dir is None:
        label = f"B-markov1@{depth} seed={seed}" if seed is not None else f"B@{depth}"
        raise RuntimeError(f"Missing exact Phase 1 run for {label}")
    return run_dir


def gate_point(run_dir: Path, *, tail_edges: int) -> dict[str, Any]:
    from v2.phase1.report import build_report_rows
    from v2.phase1.transport import build_transport_rows

    run = load_phase1_run(run_dir)
    report_row = build_report_rows([run])[0]
    transport_row = build_transport_rows([run], tail_edges=tail_edges)[0]
    return {
        "retention_last": float(report_row["retention_last"]),
        "active_mean_tail": float(transport_row["tail_transport_active_mean"]),
        "run_dir": str(run_dir),
    }


def load_phase1_run(run_dir: Path) -> dict[str, Any]:
    dataset_path = run_dir / "dataset.json"
    summary_path = run_dir / "phase1_summary.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "dataset": dataset,
        "summary": summary,
        "moments_prefix": load_optional_json(run_dir / "moments" / "moments_prefix.json"),
        "moments_suffix": load_optional_json(run_dir / "moments" / "moments_suffix.json"),
        "_dataset_path": str(dataset_path.resolve()),
        "_summary_path": str(summary_path.resolve()),
        "_run_dir": str(run_dir.resolve()),
        "_generated_at_dt": datetime.fromisoformat(dataset["generated_at"]),
        "generated_at": dataset["generated_at"],
    }


def load_optional_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def euclidean(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def source_hashes(results_base: Path, depths: list[int]) -> dict[str, dict[str, str]]:
    hashes = {}
    for depth in depths:
        path = results_base / "level0" / "phi_snapshots" / "var_B" / f"phi_iter{depth}.struct.gz"
        if not path.exists():
            raise RuntimeError(f"Missing B@{depth} source for hashing: {path}")
        hashes[f"B@{depth}"] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }
    return hashes


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Post-review Item 2 - B-markov1@24 Gate-Plane Readout",
        "",
        f"- Status: `{summary['status']}`",
        f"- Outcome: `{summary['outcome']}`",
        f"- Preregistration SHA-256: `{summary['preregistration_sha256']}`",
        "",
        "## Primary Metrics",
        "",
        f"- `G20`: `{summary['G20']:.6f}`",
        f"- `D20`: `{summary['D20']:.6f}`",
        f"- `G24`: `{summary['G24']:.6f}`",
        f"- `D24`: `{summary['D24']:.6f}`",
        f"- `Delta_G`: `{summary['Delta_G']:.6f}`",
        f"- `Delta_D`: `{summary['Delta_D']:.6f}`",
        "",
        "## Baseline Control",
        "",
        f"- Rule: `{summary['baseline_control']['rule']}`",
        f"- Established: `{summary['baseline_control']['baseline_separation_established']}`",
        f"- Relative comparison blocked: `{summary['baseline_control']['blocked_relative_comparison']}`",
        "",
        "## Per-Depth Gate Plane",
        "",
        "| depth | R_B | A_B | max R_markov1 | D_nearest | G | D |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["depths"]:
        lines.append(
            f"| {row['depth']} | {row['retention_B']:.6f} | {row['active_mean_tail_B']:.6f} | "
            f"{row['max_retention_markov1']:.6f} | {row['nearest_distance_markov1']:.6f} | "
            f"{row['G']:.6f} | {row['D']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This readout is repo-facing. It does not authorize manuscript edits without Grace audit, Sofia editorial review, a Decision Log entry, and explicit Iban authorization.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_per_depth_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "depth",
        "retention_B",
        "active_mean_tail_B",
        "max_retention_markov1",
        "nearest_distance_markov1",
        "G",
        "D",
        "observed_run_dir",
    ]
    write_csv(path, rows, fieldnames)


def write_per_seed_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "depth",
        "seed",
        "retention_markov1",
        "active_mean_tail_markov1",
        "run_dir",
    ]
    write_csv(path, rows, fieldnames)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def hash_primary_artifacts(run_dir: Path) -> dict[str, str]:
    names = [
        "summary.json",
        "report.md",
        "per_depth_gate_plane.csv",
        "per_seed_markov1_gate_plane.csv",
    ]
    hashes = {name: sha256_file(run_dir / name) for name in names}
    for manifest_path in sorted((run_dir / "phase1_runs").rglob("manifest.json")):
        hashes[manifest_path.relative_to(run_dir).as_posix()] = sha256_file(manifest_path)
    return hashes


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def fail_manifest(manifest: dict[str, Any], manifest_path: Path, reason: str) -> int:
    manifest["status"] = "failed"
    manifest["error"] = reason
    manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(manifest_path, manifest)
    return 1


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_log_header(log_path: Path, manifest: dict[str, Any]) -> None:
    log_path.write_text(
        "# D-0155 Item 2 B-markov1@24 gate-plane run\n\n"
        + json.dumps(manifest, indent=2)
        + "\n",
        encoding="utf-8",
    )


def print_plan(manifest: dict[str, Any]) -> None:
    print("D-0155 Item 2 B-markov1@24 gate-plane plan (dry-run)")
    print("-" * 78)
    print(f"run_dir: {manifest['run_dir']}")
    print(f"prereg_sha256: {manifest['preregistration']['sha256']}")
    print("observed commands:")
    for command in manifest["commands"]["observed"]:
        print("  " + " ".join(command))
    print("markov1 commands:")
    for command in manifest["commands"]["markov1"]:
        print("  " + " ".join(command))
    print("\nAdd --execute to run.")


if __name__ == "__main__":
    raise SystemExit(main())
