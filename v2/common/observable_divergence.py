from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from utils.streaming_phi_loader import StreamingPhiLoader


def iter_observable_blocks(
    struct_path: Path,
    *,
    block_bits: int,
    raw_chunk_bytes: int,
    max_observable_bits: int | None = None,
) -> Iterator[np.ndarray]:
    if block_bits <= 0:
        raise ValueError("block_bits must be positive.")
    if raw_chunk_bytes <= 0:
        raise ValueError("raw_chunk_bytes must be positive.")
    if max_observable_bits is not None and max_observable_bits <= 0:
        raise ValueError("max_observable_bits must be positive when provided.")

    loader = StreamingPhiLoader(str(struct_path), chunk_bytes=raw_chunk_bytes)
    buffer = np.empty(0, dtype=np.uint8)
    emitted = 0

    for chunk in loader.iter_bits_numpy(chunk_bytes=raw_chunk_bytes):
        if max_observable_bits is not None:
            remaining_total = max_observable_bits - emitted - buffer.size
            if remaining_total <= 0:
                break
            if chunk.size > remaining_total:
                chunk = chunk[:remaining_total]

        if chunk.size == 0:
            continue

        if buffer.size == 0:
            buffer = chunk
        else:
            buffer = np.concatenate((buffer, chunk))

        while buffer.size >= block_bits:
            yield np.ascontiguousarray(buffer[:block_bits])
            emitted += block_bits
            buffer = np.ascontiguousarray(buffer[block_bits:])

    if buffer.size > 0:
        yield np.ascontiguousarray(buffer)


def scan_first_divergence(
    struct_path_a: Path,
    struct_path_b: Path,
    *,
    block_bits: int = 1_000_000,
    raw_chunk_bytes: int = 8_000_000,
    max_observable_bits: int | None = None,
    context_bits: int = 64,
    progress=None,
) -> dict:
    if context_bits < 0:
        raise ValueError("context_bits cannot be negative.")

    blocks_a = iter_observable_blocks(
        struct_path_a,
        block_bits=block_bits,
        raw_chunk_bytes=raw_chunk_bytes,
        max_observable_bits=max_observable_bits,
    )
    blocks_b = iter_observable_blocks(
        struct_path_b,
        block_bits=block_bits,
        raw_chunk_bytes=raw_chunk_bytes,
        max_observable_bits=max_observable_bits,
    )

    compared_bits = 0
    prev_tail = np.empty(0, dtype=np.uint8)
    block_index = 0

    while True:
        block_a = next(blocks_a, None)
        block_b = next(blocks_b, None)

        if block_a is None and block_b is None:
            return {
                "difference_found": False,
                "difference_kind": None,
                "first_difference_bit": None,
                "identical_prefix_bits": compared_bits,
                "compared_bits": compared_bits,
                "scan_stopped_by": "eof",
                "bit_a": None,
                "bit_b": None,
                "left_context_bits": _bits_to_string(prev_tail),
                "right_context_a_bits": "",
                "right_context_b_bits": "",
            }

        if block_a is None or block_b is None:
            tail_a = block_a if block_a is not None else np.empty(0, dtype=np.uint8)
            tail_b = block_b if block_b is not None else np.empty(0, dtype=np.uint8)
            return {
                "difference_found": True,
                "difference_kind": "length_mismatch",
                "first_difference_bit": compared_bits,
                "identical_prefix_bits": compared_bits,
                "compared_bits": compared_bits,
                "scan_stopped_by": "length_mismatch",
                "bit_a": None if block_a is None else int(tail_a[0]),
                "bit_b": None if block_b is None else int(tail_b[0]),
                "left_context_bits": _bits_to_string(prev_tail),
                "right_context_a_bits": _bits_to_string(tail_a[1 : 1 + context_bits]),
                "right_context_b_bits": _bits_to_string(tail_b[1 : 1 + context_bits]),
            }

        min_size = min(block_a.size, block_b.size)
        if min_size > 0:
            diff_indices = np.flatnonzero(block_a[:min_size] != block_b[:min_size])
            if diff_indices.size > 0:
                local_idx = int(diff_indices[0])
                left_context = _tail_concat(prev_tail, block_a[:local_idx], context_bits)
                return {
                    "difference_found": True,
                    "difference_kind": "bit_mismatch",
                    "first_difference_bit": compared_bits + local_idx,
                    "identical_prefix_bits": compared_bits + local_idx,
                    "compared_bits": compared_bits + local_idx + 1,
                    "scan_stopped_by": "first_difference",
                    "block_index": block_index,
                    "bit_a": int(block_a[local_idx]),
                    "bit_b": int(block_b[local_idx]),
                    "left_context_bits": _bits_to_string(left_context),
                    "right_context_a_bits": _bits_to_string(block_a[local_idx + 1 : local_idx + 1 + context_bits]),
                    "right_context_b_bits": _bits_to_string(block_b[local_idx + 1 : local_idx + 1 + context_bits]),
                }

        if block_a.size != block_b.size:
            mismatch_index = compared_bits + min_size
            left_context = _tail_concat(prev_tail, block_a[:min_size], context_bits)
            return {
                "difference_found": True,
                "difference_kind": "length_mismatch",
                "first_difference_bit": mismatch_index,
                "identical_prefix_bits": mismatch_index,
                "compared_bits": mismatch_index,
                "scan_stopped_by": "length_mismatch",
                "block_index": block_index,
                "bit_a": int(block_a[min_size]) if block_a.size > min_size else None,
                "bit_b": int(block_b[min_size]) if block_b.size > min_size else None,
                "left_context_bits": _bits_to_string(left_context),
                "right_context_a_bits": _bits_to_string(block_a[min_size + 1 : min_size + 1 + context_bits]),
                "right_context_b_bits": _bits_to_string(block_b[min_size + 1 : min_size + 1 + context_bits]),
            }

        compared_bits += min_size
        prev_tail = _tail_concat(prev_tail, block_a[:min_size], context_bits)
        block_index += 1
        if progress is not None:
            progress.update(compared_bits, message=f"{compared_bits:,} bits compared")

        if max_observable_bits is not None and compared_bits >= max_observable_bits:
            return {
                "difference_found": False,
                "difference_kind": None,
                "first_difference_bit": None,
                "identical_prefix_bits": compared_bits,
                "compared_bits": compared_bits,
                "scan_stopped_by": "max_observable_bits",
                "bit_a": None,
                "bit_b": None,
                "left_context_bits": _bits_to_string(prev_tail),
                "right_context_a_bits": "",
                "right_context_b_bits": "",
            }


def _tail_concat(left: np.ndarray, right: np.ndarray, context_bits: int) -> np.ndarray:
    if context_bits <= 0:
        return np.empty(0, dtype=np.uint8)
    if left.size == 0:
        return np.ascontiguousarray(right[-context_bits:])
    if right.size == 0:
        return np.ascontiguousarray(left[-context_bits:])
    merged = np.concatenate((left, right))
    return np.ascontiguousarray(merged[-context_bits:])


def _bits_to_string(bits: np.ndarray) -> str:
    if bits.size == 0:
        return ""
    return "".join("1" if int(bit) else "0" for bit in bits)
