#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from level1_deep_analysis import lempel_ziv_complexity, test_multiple_constants
from level2_transfer_entropy import StreamingTEAccumulator
from utils.streaming_phi_loader import StreamingPhiLoader


CONTRACT_ID = "D-0089"
CONTRACT_DOC = "docs/HSI_v2_Table5_OP_Signature_Derivation_Proposal_v0_1.md"
CONTRACT_SHA256 = "ADEEE804972A15AB0BD3DA2AD031E5B5BC904B5A64D995D96C63780A0E9BA35D"
D0087_SHA256 = "66785B2E8368E7900A12A9FD5ACE0CB60DFA01AF24D62A575541A10C029F29A0"
EXPECTED_SNAPSHOT_SHA256 = {
    "O": "850F447BD02841DD0FD49FC2A4362460BEB16E441537D48A45411AB7DF88ED2E",
    "P": "8F2F495A37FBA301C4D9B5A72C4FD1C62732E22E81E4F1822C38D754BEBA1FA7",
}
LZ_ALERT_BAND = (0.60, 0.64)
TE_ALERT_THRESHOLD = 2.0


def parse_csv_tokens(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_int_csv(raw: str) -> list[int]:
    values = []
    for token in parse_csv_tokens(raw):
        try:
            values.append(int(token))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid integer token: {token}") from exc
    return values


def resolve_project_path(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_stamp() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def log_phase(message: str, quiet: bool = False) -> None:
    if not quiet:
        print(message, flush=True)


def load_metadata(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_snapshot(
    *,
    variant: str,
    source_root: Path,
) -> dict[str, Any]:
    struct_path = source_root / f"var_{variant}" / "phi_iter20.struct.gz"
    metadata_path = source_root / f"var_{variant}" / "phi_iter20.json"

    if not struct_path.exists():
        raise RuntimeError(f"{variant}: missing structural snapshot: {struct_path}")
    if not metadata_path.exists():
        raise RuntimeError(f"{variant}: missing metadata snapshot: {metadata_path}")

    metadata = load_metadata(metadata_path)
    actual_sha = sha256_file(struct_path)
    expected_sha = EXPECTED_SNAPSHOT_SHA256.get(variant)
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"{variant}: sha256_struct_gz mismatch; expected {expected_sha}, got {actual_sha}"
        )
    if metadata.get("sha256_struct_gz") != expected_sha:
        raise RuntimeError(
            f"{variant}: metadata sha256_struct_gz mismatch; expected {expected_sha}, "
            f"got {metadata.get('sha256_struct_gz')}"
        )
    if metadata.get("pre_registration_sha256") != D0087_SHA256:
        raise RuntimeError(
            f"{variant}: metadata pre_registration_sha256 mismatch; expected {D0087_SHA256}, "
            f"got {metadata.get('pre_registration_sha256')}"
        )
    if int(metadata.get("iteration", -1)) != 20:
        raise RuntimeError(f"{variant}: expected iteration 20, got {metadata.get('iteration')}")
    for key in ("freq_zero", "freq_one"):
        if not math.isclose(float(metadata.get(key, -1.0)), 0.5, abs_tol=1e-12):
            raise RuntimeError(f"{variant}: expected {key}=0.5, got {metadata.get(key)}")
    if float(metadata.get("terminalization_ratio", 1.0)) > 0.05:
        raise RuntimeError(
            f"{variant}: expected terminalization_ratio <= 0.05, "
            f"got {metadata.get('terminalization_ratio')}"
        )

    return {
        "variant": variant,
        "struct_path": str(struct_path.resolve()),
        "metadata_path": str(metadata_path.resolve()),
        "sha256_struct_gz": actual_sha,
        "metadata": metadata,
    }


def load_observable_bits_limited(struct_path: Path, expected_bits: int) -> tuple[str, dict[str, int]]:
    loader = StreamingPhiLoader(str(struct_path))
    chunks: list[bytes] = []
    consumed = 0
    yielded = 0
    chunk_count = 0

    for bits in loader.iter_bits_numpy():
        chunk_count += 1
        yielded += int(len(bits))
        remaining = expected_bits - consumed
        if remaining <= 0:
            continue
        if len(bits) > remaining:
            bits = bits[:remaining]
        chunks.append((bits + ord("0")).astype(np.uint8).tobytes())
        consumed += int(len(bits))

    if consumed != expected_bits:
        raise RuntimeError(
            f"Observable bit count mismatch while loading {struct_path}: "
            f"expected {expected_bits}, got {consumed}"
        )

    return b"".join(chunks).decode("ascii"), {
        "loader_chunks": chunk_count,
        "observable_bits_yielded_before_trim": yielded,
        "observable_bits_consumed": consumed,
        "trimmed_padding_bits": max(0, yielded - consumed),
    }


def compute_lz_signature(bits: str, scales: list[int], quiet: bool) -> dict[str, Any]:
    if not scales:
        raise RuntimeError("No LZ scales provided")
    if max(scales) > len(bits):
        raise RuntimeError(
            f"LZ max scale {max(scales)} exceeds available bits {len(bits)}"
        )

    by_scale: dict[str, dict[str, float | int]] = {}
    for index, scale in enumerate(scales, start=1):
        log_phase(f"        LZ scale {index}/{len(scales)}: {scale:,} bits", quiet)
        segment = bits[:scale]
        raw_lz = lempel_ziv_complexity(segment)
        normalized = raw_lz / (scale / math.log2(scale)) if scale > 1 else 0.0
        by_scale[str(scale)] = {
            "raw_lz": int(raw_lz),
            "normalized_lz": float(normalized),
        }

    lz_ratios: list[float] = []
    for left, right in zip(scales, scales[1:]):
        left_norm = float(by_scale[str(left)]["normalized_lz"])
        right_norm = float(by_scale[str(right)]["normalized_lz"])
        if left_norm > 0:
            lz_ratios.append(float(right_norm / left_norm))

    constants_test = test_multiple_constants(lz_ratios, verbose=False)
    mean_ratio = float(constants_test["mean_ratio"])
    return {
        "scales": scales,
        "by_scale": by_scale,
        "scale_ratios": [float(right / left) for left, right in zip(scales, scales[1:])],
        "lz_ratios": lz_ratios,
        "constants_test": constants_test,
        "mean_ratio": mean_ratio,
        "alert": LZ_ALERT_BAND[0] <= mean_ratio <= LZ_ALERT_BAND[1],
        "alert_band": list(LZ_ALERT_BAND),
    }


def compute_te_signature(
    *,
    struct_path: Path,
    expected_bits: int,
    scales: list[int],
    history: int,
    chunk_bits: int,
    shuffle_seed: int,
    quiet: bool,
) -> dict[str, Any]:
    loader = StreamingPhiLoader(str(struct_path))
    accumulator = StreamingTEAccumulator(scales, k=history, n_bins=2)
    shuffle_accumulator = StreamingTEAccumulator(scales, k=history, n_bins=2)
    rng = np.random.default_rng(seed=shuffle_seed)

    consumed = 0
    yielded = 0
    chunk_count = 0
    chunk_bytes = chunk_bits // 2

    for bits in loader.iter_bits_numpy(chunk_bytes):
        chunk_count += 1
        yielded += int(len(bits))
        remaining = expected_bits - consumed
        if remaining <= 0:
            continue
        if len(bits) > remaining:
            bits = bits[:remaining]
        consumed += int(len(bits))
        accumulator.process_chunk(bits, method="density")
        shuffled = bits.copy()
        rng.shuffle(shuffled)
        shuffle_accumulator.process_chunk(shuffled, method="density")
        log_phase(
            f"        TE chunk {chunk_count}: processed {consumed:,}/{expected_bits:,} bits (+shuffle)",
            quiet,
        )

    if consumed != expected_bits:
        raise RuntimeError(
            f"TE observable bit count mismatch while loading {struct_path}: "
            f"expected {expected_bits}, got {consumed}"
        )

    observed = accumulator.compute_all_metrics()
    shuffled_metrics = shuffle_accumulator.compute_all_metrics()
    te_matrix = observed["te_matrix"]
    shuffle_te_matrix = shuffled_metrics["te_matrix"]

    n_scales = len(scales)
    bottom_up = []
    top_down = []
    shuffle_bottom_up = []
    shuffle_top_down = []
    for i in range(n_scales):
        for j in range(n_scales):
            if i < j:
                bottom_up.append(te_matrix[i, j])
                shuffle_bottom_up.append(shuffle_te_matrix[i, j])
            elif i > j:
                top_down.append(te_matrix[i, j])
                shuffle_top_down.append(shuffle_te_matrix[i, j])

    avg_bottom_up = float(np.mean(bottom_up)) if bottom_up else 0.0
    avg_top_down = float(np.mean(top_down)) if top_down else 0.0
    avg_observed_te = (avg_bottom_up + avg_top_down) / 2
    avg_shuffle_bottom_up = float(np.mean(shuffle_bottom_up)) if shuffle_bottom_up else 0.0
    avg_shuffle_top_down = float(np.mean(shuffle_top_down)) if shuffle_top_down else 0.0
    avg_shuffle_te = (avg_shuffle_bottom_up + avg_shuffle_top_down) / 2
    if avg_shuffle_te > 1e-10:
        ratio = avg_observed_te / avg_shuffle_te
    else:
        ratio = float("inf") if avg_observed_te > 1e-10 else 1.0

    return {
        "scales": scales,
        "history": history,
        "chunk_bits": chunk_bits,
        "chunk_bytes": chunk_bytes,
        "chunks_processed": chunk_count,
        "effective_shuffle_cardinality": f"{chunk_count} chunk(s) x seed {shuffle_seed}",
        "shuffle_seed": shuffle_seed,
        "observable_bits_yielded_before_trim": yielded,
        "observable_bits_consumed": consumed,
        "trimmed_padding_bits": max(0, yielded - consumed),
        "te_matrix": te_matrix.tolist(),
        "shuffle_te_matrix": shuffle_te_matrix.tolist(),
        "avg_bottom_up": avg_bottom_up,
        "avg_top_down": avg_top_down,
        "avg_observed_te": float(avg_observed_te),
        "avg_shuffle_bottom_up": avg_shuffle_bottom_up,
        "avg_shuffle_top_down": avg_shuffle_top_down,
        "avg_shuffle_te": float(avg_shuffle_te),
        "te_ratio_observed_vs_shuffle": float(ratio),
        "alert": ratio > TE_ALERT_THRESHOLD,
        "alert_threshold": TE_ALERT_THRESHOLD,
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2)
        handle.write("\n")


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# D-0089 Table 5 O/P signature derivation",
        "",
        f"- Status: `{summary['status']}`",
        f"- Contract: `{summary['contract']['id']}`",
        f"- Contract SHA256: `{summary['contract']['sha256']}`",
        f"- Run directory: `{summary['run_dir']}`",
        "",
        "## Protocol",
        "",
        "- Read-only derivation over D-0087 snapshots.",
        "- No generator was imported or executed.",
        "- LZ uses legacy arbitrary scales over the complete authorized snapshot.",
        "- TE uses streaming density mode with per-chunk shuffle control.",
        "- `D_f` and DET remain `---` for O/P by external-control row discipline.",
        "",
        "## Effective Sample",
        "",
    ]
    for row in summary["rows"]:
        lines.extend(
            [
                f"### {row['variant']}",
                "",
                f"- Observable bits processed: `{row['observable_bits']}`",
                f"- Snapshot SHA256: `{row['sha256_struct_gz']}`",
                f"- LZ mean ratio: `{row['lz_mean_ratio']:.6f}`",
                f"- TE ratio observed/shuffle: `{row['te_ratio']:.6f}`",
                f"- TE chunk count: `{row['te_chunks_processed']}`",
                f"- Effective shuffle cardinality: `{row['effective_shuffle_cardinality']}`",
                f"- cbar policy: `{row['cbar']}`",
                f"- D_f policy: `{row['df']}`",
                f"- DET policy: `{row['det']}`",
                f"- Kill triggered: `{row['kill_triggered']}`",
                "",
            ]
        )
    if summary["kill_triggers"]:
        lines.extend(["## Kill Triggers", ""])
        for trigger in summary["kill_triggers"]:
            lines.append(f"- {trigger}")
        lines.append("")
        lines.append(
            "Per D-0089, a kill trigger means the manuscript table must not be patched silently."
        )
    else:
        lines.extend(
            [
                "## Kill Triggers",
                "",
                "None.",
            ]
        )
    lines.extend(
        [
            "",
            "## Table 5 Candidate Rows",
            "",
            "| Var. | Type | LZ | cbar | TE ratio | D_f | DET |",
            "|---|---|---:|---|---:|---|---|",
        ]
    )
    for row in summary["rows"]:
        lines.append(
            f"| {row['variant']} | External recursive | {row['lz_mean_ratio']:.4f} | "
            f"{row['cbar']} | x{row['te_ratio']:.2f} | {row['df']} | {row['det']} |"
        )
    lines.append("")
    lines.append(
        "Caption note required: O/P rows are derived from D-0087 snapshots "
        f"({summary['rows'][0]['observable_bits']:,} and {summary['rows'][1]['observable_bits']:,} bits), "
        "not from regenerated 1G legacy streams."
    )
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "variant",
        "type",
        "observable_bits",
        "lz_mean_ratio",
        "te_ratio",
        "cbar",
        "df",
        "det",
        "te_chunks_processed",
        "effective_shuffle_cardinality",
        "kill_triggered",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="D-0089 read-only Table 5 signature derivation for O/P controls."
    )
    parser.add_argument("--source-root", default="results/level0/phi_snapshots")
    parser.add_argument("--output-dir", default="results/hsi_v2/table5_op_signature_derivation")
    parser.add_argument("--variants", default="O,P")
    parser.add_argument("--iteration", type=int, default=20)
    parser.add_argument("--lz-scales", type=parse_int_csv, default=parse_int_csv("1000,2000,5000,10000,20000,50000"))
    parser.add_argument("--te-scales", type=parse_int_csv, default=parse_int_csv("4,8,16,32,64,128"))
    parser.add_argument("--te-history", type=int, default=3)
    parser.add_argument("--te-chunk-bits", type=int, default=10_000_000)
    parser.add_argument("--te-shuffle-seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    variants = [token.upper() for token in parse_csv_tokens(args.variants)]
    if variants != ["O", "P"]:
        raise SystemExit("D-0089 contract only authorizes variants O,P in that order.")
    if args.iteration != 20:
        raise SystemExit("D-0089 contract only authorizes iteration 20.")

    source_root = resolve_project_path(args.source_root)
    output_root = resolve_project_path(args.output_dir)
    stamp = run_stamp()
    run_dir = output_root / f"table5-op-signature__var-O-P__iter-20__{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    command = [Path(sys.executable).name, Path(__file__).name, *(argv if argv is not None else sys.argv[1:])]
    started_at = now_utc()
    t0 = time.perf_counter()

    log_phase("D-0089 Table 5 O/P signature derivation", args.quiet)
    log_phase("-" * 96, args.quiet)
    log_phase(f"[Phase] Validating D-0087 snapshots\n        source_root={source_root}", args.quiet)

    snapshot_records = [validate_snapshot(variant=variant, source_root=source_root) for variant in variants]
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    kill_triggers: list[str] = []

    for record in snapshot_records:
        variant = record["variant"]
        metadata = record["metadata"]
        struct_path = Path(record["struct_path"])
        expected_bits = int(metadata["final_observable_bits"])
        log_phase(
            f"[Case] {variant}: loading {expected_bits:,} authorized observable bits",
            args.quiet,
        )
        bits, load_stats = load_observable_bits_limited(struct_path, expected_bits)

        log_phase(f"[Case] {variant}: LZ derivation", args.quiet)
        lz = compute_lz_signature(bits, args.lz_scales, args.quiet)

        log_phase(f"[Case] {variant}: TE derivation", args.quiet)
        te = compute_te_signature(
            struct_path=struct_path,
            expected_bits=expected_bits,
            scales=args.te_scales,
            history=args.te_history,
            chunk_bits=args.te_chunk_bits,
            shuffle_seed=args.te_shuffle_seed,
            quiet=args.quiet,
        )

        row_kills = []
        if lz["alert"]:
            row_kills.append(
                f"{variant}: LZ mean ratio {lz['mean_ratio']:.6f} inside alert band {LZ_ALERT_BAND}"
            )
        if te["alert"]:
            row_kills.append(
                f"{variant}: TE ratio {te['te_ratio_observed_vs_shuffle']:.6f} exceeds {TE_ALERT_THRESHOLD}"
            )
        kill_triggers.extend(row_kills)

        row = {
            "variant": variant,
            "type": "External recursive",
            "observable_bits": expected_bits,
            "sha256_struct_gz": record["sha256_struct_gz"],
            "lz_mean_ratio": float(lz["mean_ratio"]),
            "te_ratio": float(te["te_ratio_observed_vs_shuffle"]),
            "cbar": "timeout/N.A.",
            "df": "---",
            "det": "---",
            "te_chunks_processed": int(te["chunks_processed"]),
            "effective_shuffle_cardinality": te["effective_shuffle_cardinality"],
            "kill_triggered": bool(row_kills),
        }
        rows.append(row)
        details[variant] = {
            "source": record,
            "load_stats": load_stats,
            "lz": lz,
            "te": te,
        }

    status = "kill-triggered" if kill_triggers else "accepted-readout"
    summary = {
        "status": status,
        "contract": {
            "id": CONTRACT_ID,
            "doc": CONTRACT_DOC,
            "sha256": CONTRACT_SHA256,
            "d0087_pre_registration_sha256": D0087_SHA256,
        },
        "run_dir": str(run_dir.resolve()),
        "started_at_utc": started_at,
        "finished_at_utc": now_utc(),
        "elapsed_seconds": time.perf_counter() - t0,
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "command": command,
        "cwd": str(Path.cwd().resolve()),
        "generator_imported_or_executed": False,
        "source_root": str(source_root.resolve()),
        "parameters": {
            "variants": variants,
            "iteration": args.iteration,
            "lz_scales": args.lz_scales,
            "te_scales": args.te_scales,
            "te_history": args.te_history,
            "te_chunk_bits": args.te_chunk_bits,
            "te_shuffle_seed": args.te_shuffle_seed,
            "lz_alert_band": list(LZ_ALERT_BAND),
            "te_alert_threshold": TE_ALERT_THRESHOLD,
        },
        "rows": rows,
        "details": details,
        "kill_triggers": kill_triggers,
    }

    summary_path = run_dir / "summary.json"
    report_path = run_dir / "report.md"
    table_path = run_dir / "table.csv"
    manifest_path = run_dir / "manifest.json"
    write_json(summary_path, summary)
    report_path.write_text(render_report(summary) + "\n", encoding="utf-8")
    write_csv(table_path, rows)
    manifest = {
        "status": status,
        "contract": summary["contract"],
        "run_dir": summary["run_dir"],
        "artifacts": {
            "summary": str(summary_path.resolve()),
            "report": str(report_path.resolve()),
            "table": str(table_path.resolve()),
            "manifest": str(manifest_path.resolve()),
        },
        "source_snapshots": {
            row["variant"]: {
                "sha256_struct_gz": row["sha256_struct_gz"],
                "observable_bits": row["observable_bits"],
            }
            for row in rows
        },
        "command": command,
        "script_sha256": summary["script_sha256"],
        "created_at_utc": now_utc(),
    }
    write_json(manifest_path, manifest)

    print("\nD-0089 Table 5 O/P signature derivation")
    print("-" * 96)
    print(f"{'var':<4} {'bits':>10} {'LZ':>10} {'TE':>10} {'chunks':>7} {'kill':>6}")
    for row in rows:
        print(
            f"{row['variant']:<4} {row['observable_bits']:>10} "
            f"{row['lz_mean_ratio']:>10.4f} {row['te_ratio']:>10.4f} "
            f"{row['te_chunks_processed']:>7} {str(row['kill_triggered']):>6}"
        )
    if kill_triggers:
        print("\nKill triggers:")
        for trigger in kill_triggers:
            print(f"- {trigger}")
    print(f"\nSaved summary to: {summary_path}")
    print(f"Saved report to: {report_path}")
    print(f"Saved CSV table to: {table_path}")
    print(f"Saved manifest to: {manifest_path}")

    return 2 if kill_triggers else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
