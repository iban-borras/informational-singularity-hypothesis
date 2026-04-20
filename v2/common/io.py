from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from utils.streaming_phi_loader import StreamingPhiLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEVEL0_ROOT = PROJECT_ROOT / "results" / "level0" / "phi_snapshots"


def normalize_variant(variant: str) -> str:
    variant = variant.strip().upper()
    if not variant:
        raise ValueError("Variant cannot be empty.")
    return variant


def list_available_iterations(var_dir: Path) -> list[int]:
    pattern = re.compile(r"phi_iter(\d+)\.struct\.gz$")
    iterations = []
    for path in var_dir.glob("phi_iter*.struct.gz"):
        match = pattern.match(path.name)
        if match:
            iterations.append(int(match.group(1)))
    return sorted(iterations)


def format_available_iterations(iterations: list[int]) -> str:
    if not iterations:
        return "none"

    if iterations == list(range(iterations[0], iterations[-1] + 1)):
        if len(iterations) == 1:
            return str(iterations[0])
        return f"{iterations[0]}..{iterations[-1]}"

    preview = ", ".join(str(value) for value in iterations[:10])
    if len(iterations) > 10:
        preview += ", ..."
    return preview


def detect_latest_iteration(var_dir: Path) -> int:
    iterations = list_available_iterations(var_dir)
    if not iterations:
        raise FileNotFoundError(f"No structural snapshots found in {var_dir}")

    return max(iterations)


def resolve_struct_path(
    *,
    input_path: Optional[str] = None,
    variant: Optional[str] = None,
    iteration: Optional[int] = None,
    level0_root: Optional[str] = None,
) -> Path:
    if input_path:
        path = Path(input_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        return path

    if not variant:
        raise ValueError("Either --input or --variant must be provided.")

    variant_code = normalize_variant(variant)
    base_dir = Path(level0_root).expanduser().resolve() if level0_root else DEFAULT_LEVEL0_ROOT
    var_dir = base_dir / f"var_{variant_code}"
    if not var_dir.exists():
        raise FileNotFoundError(
            f"Variant {variant_code} not found under {base_dir}."
        )

    if iteration is None:
        iteration = detect_latest_iteration(var_dir)

    path = var_dir / f"phi_iter{iteration}.struct.gz"
    if not path.exists():
        available_iterations = list_available_iterations(var_dir)
        available_summary = format_available_iterations(available_iterations)
        latest_hint = (
            f" Try --iteration {available_iterations[-1]}."
            if available_iterations
            else ""
        )
        raise FileNotFoundError(
            f"Structural snapshot not found for variant {variant_code} iteration {iteration}. "
            f"Available iterations: {available_summary}.{latest_hint}"
        )

    return path


def companion_metadata_path(struct_path: Path) -> Path:
    return struct_path.with_suffix("").with_suffix(".json")


def load_struct_metadata(struct_path: Path) -> Dict:
    metadata_path = companion_metadata_path(struct_path)
    if not metadata_path.exists():
        return {}
    with open(metadata_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_observable_prefix_bits(
    struct_path: Path,
    total_bits: int,
    chunk_bytes: Optional[int] = None,
    progress=None,
) -> np.ndarray:
    if total_bits <= 0:
        raise ValueError("total_bits must be positive.")

    effective_chunk_bytes = chunk_bytes or max(262_144, total_bits // 2)
    loader = StreamingPhiLoader(str(struct_path), chunk_bytes=effective_chunk_bytes)
    chunks = []
    remaining = total_bits
    loaded = 0

    for chunk in loader.iter_bits_numpy(chunk_bytes=effective_chunk_bytes):
        if len(chunk) >= remaining:
            final_chunk = chunk[:remaining]
            chunks.append(final_chunk)
            loaded += len(final_chunk)
            if progress is not None:
                progress.update(loaded, message=f"{loaded:,}/{total_bits:,} bits")
            remaining = 0
            break
        chunks.append(chunk)
        remaining -= len(chunk)
        loaded += len(chunk)
        if progress is not None:
            progress.update(loaded, message=f"{loaded:,}/{total_bits:,} bits")

    if remaining > 0:
        raise ValueError(
            f"Requested {total_bits} observable bits but only found {total_bits - remaining} in {struct_path}"
        )

    bits = np.concatenate(chunks).astype(np.uint8, copy=False)
    return np.ascontiguousarray(bits)


def split_into_segments(
    bits: np.ndarray,
    segment_bits: int,
    num_segments: int,
    start_offset_bits: int = 0,
) -> list[np.ndarray]:
    if start_offset_bits < 0:
        raise ValueError("start_offset_bits cannot be negative.")

    total_required = start_offset_bits + (segment_bits * num_segments)
    if bits.size < total_required:
        raise ValueError(
            f"Need {total_required} bits for offset {start_offset_bits} and {num_segments} segments of size {segment_bits}, got {bits.size}."
        )

    segments = []
    for index in range(num_segments):
        start = start_offset_bits + (index * segment_bits)
        end = start + segment_bits
        segments.append(np.ascontiguousarray(bits[start:end]))
    return segments
