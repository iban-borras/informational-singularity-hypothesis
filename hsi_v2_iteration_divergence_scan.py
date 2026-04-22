#!/usr/bin/env python3
"""
HSI v2 Iteration Divergence Scan

Find the first observable-bit divergence between two Level 0 structural snapshots
without materializing giant prefixes in memory.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from v2.common.io import (
    companion_metadata_path,
    load_struct_metadata,
    normalize_variant,
    resolve_struct_path,
)
from v2.common.observable_divergence import scan_first_divergence
from utils.progress import ProgressIndicator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find the first observable-bit divergence between two HSI Level 0 snapshots."
    )
    parser.add_argument("--variant", type=str, required=True, help="Primary variant code, e.g. B.")
    parser.add_argument("--iteration-a", type=int, required=True, help="First iteration to compare.")
    parser.add_argument("--iteration-b", type=int, required=True, help="Second iteration to compare.")
    parser.add_argument(
        "--variant-b",
        type=str,
        default=None,
        help="Optional second variant code. Defaults to --variant.",
    )
    parser.add_argument(
        "--level0-root",
        type=str,
        default=None,
        help="Optional override for the Level 0 snapshots root directory.",
    )
    parser.add_argument(
        "--block-bits",
        type=int,
        default=1_000_000,
        help="Observable bits compared per aligned block.",
    )
    parser.add_argument(
        "--chunk-bytes",
        type=int,
        default=8_000_000,
        help="Raw compressed bytes read per streaming chunk.",
    )
    parser.add_argument(
        "--max-observable-bits",
        type=int,
        default=None,
        help="Optional hard cap on observable bits to scan before stopping.",
    )
    parser.add_argument(
        "--context-bits",
        type=int,
        default=64,
        help="Context bits to save on each side of the first difference.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/hsi_v2/iteration_divergence",
        help="Output directory, relative to hsi_agents_project or absolute.",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce console output.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.block_bits <= 0:
        parser.error("--block-bits must be positive.")
    if args.chunk_bytes <= 0:
        parser.error("--chunk-bytes must be positive.")
    if args.max_observable_bits is not None and args.max_observable_bits <= 0:
        parser.error("--max-observable-bits must be positive when provided.")
    if args.context_bits < 0:
        parser.error("--context-bits cannot be negative.")

    variant_a = normalize_variant(args.variant)
    variant_b = normalize_variant(args.variant_b or args.variant)

    try:
        struct_path_a = resolve_struct_path(
            variant=variant_a,
            iteration=args.iteration_a,
            level0_root=args.level0_root,
        )
        struct_path_b = resolve_struct_path(
            variant=variant_b,
            iteration=args.iteration_b,
            level0_root=args.level0_root,
        )
    except FileNotFoundError as exc:
        print(f"\n[Error] {exc}", file=sys.stderr)
        return 2

    metadata_a = load_struct_metadata(struct_path_a)
    metadata_b = load_struct_metadata(struct_path_b)

    phase_print(
        "Resolved comparison inputs",
        (
            f"A={variant_a}@{args.iteration_a} | B={variant_b}@{args.iteration_b}"
            f" | block_bits={args.block_bits:,}"
        ),
        quiet=args.quiet,
    )
    phase_print(
        "Scanning observable divergence",
        (
            f"{struct_path_a.name} vs {struct_path_b.name}"
            + (
                f" | max_bits={args.max_observable_bits:,}"
                if args.max_observable_bits is not None
                else " | max_bits=until-divergence-or-eof"
            )
        ),
        quiet=args.quiet,
    )

    progress_total = args.max_observable_bits if args.max_observable_bits is not None else None
    scan_progress = ProgressIndicator("Observable divergence scan", total=progress_total) if not args.quiet else None
    if scan_progress is not None:
        with scan_progress:
            divergence = scan_first_divergence(
                struct_path_a,
                struct_path_b,
                block_bits=args.block_bits,
                raw_chunk_bytes=args.chunk_bytes,
                max_observable_bits=args.max_observable_bits,
                context_bits=args.context_bits,
                progress=scan_progress,
            )
    else:
        divergence = scan_first_divergence(
            struct_path_a,
            struct_path_b,
            block_bits=args.block_bits,
            raw_chunk_bytes=args.chunk_bytes,
            max_observable_bits=args.max_observable_bits,
            context_bits=args.context_bits,
        )

    output_dir = resolve_output_dir(args.output_dir)
    slug = build_slug(
        variant_a=variant_a,
        variant_b=variant_b,
        iteration_a=args.iteration_a,
        iteration_b=args.iteration_b,
        max_observable_bits=args.max_observable_bits,
    )
    run_dir = output_dir / slug
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_payload = {
        "stage": "iteration_divergence_scan",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "variant_a": variant_a,
            "variant_b": variant_b,
            "iteration_a": args.iteration_a,
            "iteration_b": args.iteration_b,
            "block_bits": args.block_bits,
            "chunk_bytes": args.chunk_bytes,
            "max_observable_bits": args.max_observable_bits,
            "context_bits": args.context_bits,
        },
        "inputs": {
            "struct_path_a": str(struct_path_a),
            "struct_path_b": str(struct_path_b),
            "metadata_path_a": str(companion_metadata_path(struct_path_a)),
            "metadata_path_b": str(companion_metadata_path(struct_path_b)),
        },
        "source_metadata": {
            "a": metadata_a,
            "b": metadata_b,
        },
        "divergence": divergence,
    }

    manifest = {
        "run_slug": slug,
        "generated_at": summary_payload["generated_at"],
        "script": "hsi_v2_iteration_divergence_scan.py",
        "cwd": str(Path.cwd()),
        "outputs": {
            "summary": str(run_dir / "divergence_summary.json"),
            "report": str(run_dir / "divergence_report.md"),
            "manifest": str(run_dir / "manifest.json"),
        },
        "inputs": summary_payload["inputs"],
        "arguments": vars(args),
    }

    phase_print("Writing artifacts", str(run_dir), quiet=args.quiet)
    save_json(run_dir / "divergence_summary.json", summary_payload)
    save_text(run_dir / "divergence_report.md", render_markdown_report(summary_payload))
    save_json(run_dir / "manifest.json", manifest)

    if not args.quiet:
        print_summary(summary_payload)
        print(f"\nSaved summary to: {run_dir / 'divergence_summary.json'}")
        print(f"Saved report to: {run_dir / 'divergence_report.md'}")
        print(f"Saved manifest to: {run_dir / 'manifest.json'}")

    return 0


def build_slug(
    *,
    variant_a: str,
    variant_b: str,
    iteration_a: int,
    iteration_b: int,
    max_observable_bits: int | None,
) -> str:
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    max_part = "all" if max_observable_bits is None else compact_int(max_observable_bits)
    variant_part = variant_a if variant_a == variant_b else f"{variant_a}-vs-{variant_b}"
    return (
        f"iter-divergence__var-{variant_part}__iter-{iteration_a}-vs-{iteration_b}"
        f"__max-{max_part}__{timestamp}"
    )


def compact_int(value: int) -> str:
    if value % 1_000_000_000 == 0:
        return f"{value // 1_000_000_000}G"
    if value % 1_000_000 == 0:
        return f"{value // 1_000_000}M"
    if value % 1_000 == 0:
        return f"{value // 1_000}K"
    return str(value)


def render_markdown_report(payload: dict) -> str:
    config = payload["config"]
    divergence = payload["divergence"]
    lines = [
        "# HSI v2 Iteration Divergence Scan",
        "",
        "## Selection",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Variant A: `{config['variant_a']}@{config['iteration_a']}`",
        f"- Variant B: `{config['variant_b']}@{config['iteration_b']}`",
        f"- Block bits: `{config['block_bits']}`",
        f"- Raw chunk bytes: `{config['chunk_bytes']}`",
        f"- Max observable bits: `{config['max_observable_bits']}`",
        f"- Context bits: `{config['context_bits']}`",
        "",
        "## Result",
        "",
        f"- Difference found: `{divergence['difference_found']}`",
        f"- Difference kind: `{divergence['difference_kind']}`",
        f"- First difference bit: `{divergence['first_difference_bit']}`",
        f"- Identical prefix bits: `{divergence['identical_prefix_bits']}`",
        f"- Compared bits: `{divergence['compared_bits']}`",
        f"- Scan stopped by: `{divergence['scan_stopped_by']}`",
        f"- Bit A: `{divergence['bit_a']}`",
        f"- Bit B: `{divergence['bit_b']}`",
        "",
        "## Local Context",
        "",
        f"- Left context bits: `{divergence['left_context_bits']}`",
        f"- Right context A bits: `{divergence['right_context_a_bits']}`",
        f"- Right context B bits: `{divergence['right_context_b_bits']}`",
        "",
        "## Run Paths",
        "",
        f"- A: `{payload['inputs']['struct_path_a']}`",
        f"- B: `{payload['inputs']['struct_path_b']}`",
    ]
    return "\n".join(lines)


def print_summary(payload: dict) -> None:
    config = payload["config"]
    divergence = payload["divergence"]
    print("\nIteration divergence summary")
    print("-" * 94)
    print(f"{'A':>8}: {config['variant_a']}@{config['iteration_a']}")
    print(f"{'B':>8}: {config['variant_b']}@{config['iteration_b']}")
    print(f"{'block':>8}: {config['block_bits']:,} bits")
    print(f"{'max':>8}: {config['max_observable_bits'] if config['max_observable_bits'] is not None else 'all'}")
    print(f"{'diff?':>8}: {divergence['difference_found']}")
    print(f"{'kind':>8}: {divergence['difference_kind']}")
    print(f"{'first':>8}: {divergence['first_difference_bit']}")
    print(f"{'equal':>8}: {divergence['identical_prefix_bits']:,}")
    print(f"{'compared':>8}: {divergence['compared_bits']:,}")


def resolve_output_dir(raw_output_dir: str) -> Path:
    output_path = Path(raw_output_dir)
    if not output_path.is_absolute():
        output_path = Path(__file__).parent / output_path
    return output_path.resolve()


def save_json(path: Path, payload: dict | list) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_text(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")


def phase_print(title: str, detail: str = "", quiet: bool = False) -> None:
    if quiet:
        return
    print(f"\n[Phase] {title}")
    if detail:
        print(f"        {detail}")


if __name__ == "__main__":
    raise SystemExit(main())
