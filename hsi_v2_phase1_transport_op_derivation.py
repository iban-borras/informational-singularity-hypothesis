#!/usr/bin/env python3
"""D-0092 read-only Phase 1 transport derivation for O/P controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v2.phase1.report import split_rows_by_role
from v2.phase1.transport import (
    build_selection,
    build_transport_rows,
    render_console_summary,
    render_markdown_report,
)

CONTRACT_ID = "D-0092"
D0087_SHA256 = "66785B2E8368E7900A12A9FD5ACE0CB60DFA01AF24D62A575541A10C029F29A0"
DEFAULT_SOURCE_RUNS = {
    "O": "results/hsi_v2/phase1_external_recursive_controls/"
    "phase1-tower__var-O__iter-20__seg-3x1M__m-8-12-16-20-24-28-32__proj-prefix-suffix__20260506T235738",
    "P": "results/hsi_v2/phase1_external_recursive_controls/"
    "phase1-tower__var-P__iter-20__seg-3x1M__m-8-12-16-20-24-28-32__proj-prefix-suffix__20260506T235740",
}
TAIL_EDGES = 2
HSI_LIKE_ACTIVE_MEAN_TAIL = 0.30


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="D-0092 read-only transport report for existing O/P Phase 1 towers."
    )
    parser.add_argument("--output-dir", default="results/hsi_v2/phase1_external_recursive_controls/transport_op")
    parser.add_argument("--source-o", default=DEFAULT_SOURCE_RUNS["O"])
    parser.add_argument("--source-p", default=DEFAULT_SOURCE_RUNS["P"])
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    output_root = resolve_project_path(args.output_dir)
    run_slug = f"phase1-transport__var-O-P__d0092__{timestamp}"
    run_dir = output_root / run_slug
    run_dir.mkdir(parents=True, exist_ok=False)

    started = time.perf_counter()
    command = [Path(sys.executable).name, Path(__file__).name, *(argv if argv is not None else sys.argv[1:])]
    log("D-0092 Phase 1 transport O/P derivation", args.quiet)
    log("-" * 92, args.quiet)

    try:
        source_paths = {
            "O": resolve_project_path(args.source_o),
            "P": resolve_project_path(args.source_p),
        }
        log("[Phase] Validating source towers", args.quiet)
        runs = [load_source_run(variant, source_paths[variant]) for variant in ("O", "P")]

        log("[Phase] Running canonical phase1-transport derivation", args.quiet)
        rows = build_transport_rows(
            runs,
            main_policy="prefix",
            control_policy="suffix",
            tail_edges=TAIL_EDGES,
            include_hints=False,
        )
        selection = build_selection(
            rows,
            family_inferred=False,
            tail_edges=TAIL_EDGES,
            main_policy="prefix",
            control_policy="suffix",
        )
        selection = canonical_report_selection(selection)
        output_rows = [canonical_report_row(row) for row in rows]
        summary = {
            "generated_at": now_utc(),
            "status": "completed",
            "contract": {
                "id": CONTRACT_ID,
                "d0087_pre_registration_sha256": D0087_SHA256,
                "source_decision_log": "docs/HSI_v2_Decision_Log.md#D-0092",
            },
            "selection": selection,
            "rows": output_rows,
            "grouped_rows": split_rows_by_role(output_rows),
            "d0092_readout": build_d0092_readout(output_rows),
            "source_towers": [run["_source_verification"] for run in runs],
            "command": command,
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "run_dir": str(run_dir.resolve()),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        kills = build_kill_triggers(summary["d0092_readout"])
        summary["kill_triggers"] = kills
        if kills:
            summary["status"] = "kill-triggered"
            write_escalation_note(run_dir / "escalation_note.md", summary)

        write_json(run_dir / "summary.json", summary)
        report = render_report(summary)
        (run_dir / "report.md").write_text(report + "\n", encoding="utf-8")
        manifest = build_manifest(summary, run_dir)
        write_json(run_dir / "manifest.json", manifest)

        if not args.quiet:
            print(render_console_summary(output_rows))
            print("")
            print(render_d0092_console(summary))
            print(f"\nSaved summary to: {run_dir / 'summary.json'}")
            print(f"Saved report to: {run_dir / 'report.md'}")
            if kills:
                print(f"Saved escalation note to: {run_dir / 'escalation_note.md'}")
            print(f"Saved manifest to: {run_dir / 'manifest.json'}")
        return 2 if kills else 0
    except Exception as exc:
        failure = {
            "generated_at": now_utc(),
            "status": "failed",
            "contract": {"id": CONTRACT_ID, "d0087_pre_registration_sha256": D0087_SHA256},
            "error": str(exc),
            "command": command,
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "run_dir": str(run_dir.resolve()),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        write_json(run_dir / "summary.json", failure)
        write_escalation_note(run_dir / "escalation_note.md", failure)
        write_json(run_dir / "manifest.json", build_manifest(failure, run_dir))
        print(f"[x] D-0092 derivation failed: {exc}", file=sys.stderr)
        print(f"Saved failure summary to: {run_dir / 'summary.json'}", file=sys.stderr)
        return 2


def load_source_run(variant: str, run_dir: Path) -> dict[str, Any]:
    if not run_dir.exists():
        raise RuntimeError(f"{variant}: source run directory does not exist: {run_dir}")
    dataset_path = run_dir / "dataset.json"
    summary_path = run_dir / "phase1_summary.json"
    manifest_path = run_dir / "manifest.json"
    for path in (dataset_path, summary_path, manifest_path):
        if not path.exists():
            raise RuntimeError(f"{variant}: missing required source artifact: {path}")
    for subdir in ("pattern_spaces", "fibers", "moments"):
        path = run_dir / subdir
        if not path.is_dir() or not any(path.iterdir()):
            raise RuntimeError(f"{variant}: missing or empty source subdirectory: {path}")

    dataset = load_json(dataset_path)
    summary = load_json(summary_path)
    manifest = load_json(manifest_path)
    config = dataset.get("config", {})
    source_metadata = dataset.get("source_metadata", {})
    if dataset.get("stage") != "phase1_tower":
        raise RuntimeError(f"{variant}: expected stage phase1_tower, got {dataset.get('stage')}")
    if config.get("variant") != variant:
        raise RuntimeError(f"{variant}: dataset variant mismatch: {config.get('variant')}")
    if config.get("iteration") != 20:
        raise RuntimeError(f"{variant}: expected iteration 20, got {config.get('iteration')}")
    if list(config.get("scales", [])) != [8, 12, 16, 20, 24, 28, 32]:
        raise RuntimeError(f"{variant}: scale ladder mismatch: {config.get('scales')}")
    if list(config.get("policies", [])) != ["prefix", "suffix"]:
        raise RuntimeError(f"{variant}: policy set mismatch: {config.get('policies')}")
    if source_metadata.get("pre_registration_sha256") != D0087_SHA256:
        raise RuntimeError(
            f"{variant}: D-0087 anchor mismatch: {source_metadata.get('pre_registration_sha256')}"
        )
    verify_manifest_paths(variant, run_dir, manifest)

    return {
        "dataset": dataset,
        "summary": summary,
        "moments_prefix": load_json(run_dir / "moments" / "moments_prefix.json"),
        "moments_suffix": load_json(run_dir / "moments" / "moments_suffix.json"),
        "_dataset_path": str(dataset_path.resolve()),
        "_summary_path": str(summary_path.resolve()),
        "_run_dir": str(run_dir.resolve()),
        "_generated_at_dt": parse_timestamp(dataset.get("generated_at")),
        "generated_at": dataset.get("generated_at"),
        "_source_verification": {
            "variant": variant,
            "run_dir": str(run_dir.resolve()),
            "dataset_path": str(dataset_path.resolve()),
            "summary_path": str(summary_path.resolve()),
            "manifest_path": str(manifest_path.resolve()),
            "manifest_paths_verified": True,
            "source_manifest_contains_artifact_hashes": False,
            "computed_tower_sha256": sha256_tree(run_dir),
            "component_sha256": component_hashes(run_dir),
            "d0087_pre_registration_sha256": source_metadata.get("pre_registration_sha256"),
            "snapshot_sha256_struct_gz": source_metadata.get("sha256_struct_gz"),
        },
    }


def verify_manifest_paths(variant: str, run_dir: Path, manifest: dict[str, Any]) -> None:
    outputs = manifest.get("outputs") or {}
    expected = {
        "dataset": run_dir / "dataset.json",
        "pattern_spaces": run_dir / "pattern_spaces",
        "fibers": run_dir / "fibers",
        "moments": run_dir / "moments",
        "summary": run_dir / "phase1_summary.json",
        "manifest": run_dir / "manifest.json",
    }
    for key, expected_path in expected.items():
        raw = outputs.get(key)
        if raw is None:
            raise RuntimeError(f"{variant}: manifest missing outputs.{key}")
        if Path(raw).resolve() != expected_path.resolve():
            raise RuntimeError(
                f"{variant}: manifest outputs.{key} mismatch; expected {expected_path}, got {raw}"
            )


def build_d0092_readout(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    readout = []
    for row in rows:
        active_mean_tail = float(row["tail_transport_active_mean"])
        readout.append(
            {
                "variant": row["variant"],
                "breadth_ratio": float(row["breadth_ratio_last_to_first"]),
                "H_trans": float(row["tail_transport_entropy_mean"]),
                "mu_act": active_mean_tail,
                "delta_mu_sgn": float(row["tail_transport_active_gap_signed"]),
                "delta_mu_rel": float(row["tail_transport_active_gap_relative"]),
                "active_mean_tail": active_mean_tail,
                "active_mean_tail_alert_threshold": HSI_LIKE_ACTIVE_MEAN_TAIL,
                "kill_triggered": active_mean_tail >= HSI_LIKE_ACTIVE_MEAN_TAIL,
            }
        )
    return readout


def canonical_report_row(row: dict[str, Any]) -> dict[str, Any]:
    """Preserve the D-0092 reference report row schema while using current code."""

    output = dict(row)
    output.pop("segment_offset_bits", None)
    return output


def canonical_report_selection(selection: dict[str, Any]) -> dict[str, Any]:
    output = dict(selection)
    output.pop("segment_offset_bits", None)
    return output


def build_kill_triggers(readout: list[dict[str, Any]]) -> list[str]:
    return [
        (
            f"{row['variant']}: active_mean_tail={row['active_mean_tail']:.6f} "
            f">= {HSI_LIKE_ACTIVE_MEAN_TAIL:.2f}"
        )
        for row in readout
        if row["kill_triggered"]
    ]


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# D-0092 Phase 1 Transport O/P Derivation",
        "",
        f"- Status: `{summary['status']}`",
        f"- Contract: `{summary['contract']['id']}`",
        f"- Run directory: `{summary['run_dir']}`",
        "",
        "## Protocol",
        "",
        "- Read-only derivation over existing D-0087 O/P Phase 1 towers.",
        "- Canonical derivation path: `v2.phase1.transport.build_transport_rows`.",
        "- Main/control policies: `prefix / suffix`.",
        "- Tail edges: `2` (`28 -> 24` and `32 -> 28`).",
        "- No tower, snapshot, pattern-space, fiber, or moments regeneration.",
        "",
        "## Transport Rows",
        "",
        "| Var. | Breadth ratio | H_trans | mu_act | Delta mu sgn | Delta mu rel | active_mean_tail | Kill |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary.get("d0092_readout", []):
        lines.append(
            f"| {row['variant']} | {row['breadth_ratio']:.4f} | {row['H_trans']:.4f} | "
            f"{row['mu_act']:.4f} | {row['delta_mu_sgn']:+.4f} | {row['delta_mu_rel']:+.4f} | "
            f"{row['active_mean_tail']:.4f} | {row['kill_triggered']} |"
        )
    lines.extend(["", "## Source Towers", ""])
    for source in summary.get("source_towers", []):
        lines.extend(
            [
                f"### {source['variant']}",
                "",
                f"- Run dir: `{source['run_dir']}`",
                f"- Computed tower SHA256: `{source['computed_tower_sha256']}`",
                f"- D-0087 anchor: `{source['d0087_pre_registration_sha256']}`",
                f"- Snapshot SHA256: `{source['snapshot_sha256_struct_gz']}`",
                "",
            ]
        )
    if summary.get("kill_triggers"):
        lines.extend(["## Kill / Escalation", ""])
        lines.extend(f"- {item}" for item in summary["kill_triggers"])
        lines.append("")
        lines.append("Per D-0092, no manuscript patch is authorized while this alert is active.")
    else:
        lines.extend(["## Kill / Escalation", "", "None."])
    return "\n".join(lines)


def render_d0092_console(summary: dict[str, Any]) -> str:
    lines = [
        "D-0092 transport table readout",
        "-" * 104,
        f"{'var':<4} {'breadth':>9} {'H_trans':>9} {'mu_act':>9} {'dmu_sgn':>9} {'dmu_rel':>9} {'tail':>9} {'kill':>6}",
    ]
    for row in summary.get("d0092_readout", []):
        lines.append(
            f"{row['variant']:<4} {row['breadth_ratio']:>9.4f} {row['H_trans']:>9.4f} "
            f"{row['mu_act']:>9.4f} {row['delta_mu_sgn']:>+9.4f} "
            f"{row['delta_mu_rel']:>+9.4f} {row['active_mean_tail']:>9.4f} "
            f"{str(row['kill_triggered']):>6}"
        )
    if summary.get("kill_triggers"):
        lines.append("")
        lines.append("Kill triggers:")
        lines.extend(f"- {item}" for item in summary["kill_triggers"])
    return "\n".join(lines)


def write_escalation_note(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# D-0092 Escalation Note",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Generated at: `{summary.get('generated_at', now_utc())}`",
        f"- Run directory: `{summary.get('run_dir')}`",
        "",
    ]
    if summary.get("error"):
        lines.extend(["## Error", "", f"`{summary['error']}`", ""])
    if summary.get("kill_triggers"):
        lines.extend(["## Kill Triggers", ""])
        lines.extend(f"- {item}" for item in summary["kill_triggers"])
        lines.append("")
        lines.append("Per D-0092, the manuscript must not be patched silently.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_manifest(summary: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    return {
        "status": summary.get("status"),
        "contract": summary.get("contract"),
        "script": summary.get("script"),
        "script_sha256": summary.get("script_sha256"),
        "command": summary.get("command"),
        "source_towers": summary.get("source_towers", []),
        "outputs": {
            "summary": str((run_dir / "summary.json").resolve()),
            "report": str((run_dir / "report.md").resolve()),
            "manifest": str((run_dir / "manifest.json").resolve()),
            "escalation_note": str((run_dir / "escalation_note.md").resolve())
            if (run_dir / "escalation_note.md").exists()
            else None,
        },
        "kill_triggers": summary.get("kill_triggers", []),
    }


def resolve_project_path(raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_timestamp(raw: str | None) -> datetime:
    if not raw:
        return datetime.fromtimestamp(0, tz=timezone.utc)
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def sha256_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest().upper()


def component_hashes(root: Path) -> dict[str, str]:
    components = {
        "dataset": root / "dataset.json",
        "summary": root / "phase1_summary.json",
        "manifest": root / "manifest.json",
        "pattern_spaces": root / "pattern_spaces",
        "fibers": root / "fibers",
        "moments": root / "moments",
    }
    hashes = {}
    for key, path in components.items():
        hashes[key] = sha256_tree(path) if path.is_dir() else sha256_file(path)
    return hashes


def log(message: str, quiet: bool) -> None:
    if not quiet:
        print(message, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
