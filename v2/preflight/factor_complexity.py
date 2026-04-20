from __future__ import annotations

import math
from typing import Iterable

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - fallback for environments without numba
    def njit(*args, **kwargs):  # type: ignore
        def decorator(func):
            return func
        return decorator


@njit(cache=True)
def rolling_codes_uint64(bits: np.ndarray, m: int) -> np.ndarray:
    n = bits.size
    window_count = n - m + 1
    codes = np.empty(window_count, dtype=np.uint64)

    code = np.uint64(0)
    for index in range(m):
        code = (code << np.uint64(1)) | np.uint64(bits[index])
    codes[0] = code

    mask = (np.uint64(1) << np.uint64(m)) - np.uint64(1)
    for index in range(1, window_count):
        code = ((code << np.uint64(1)) & mask) | np.uint64(bits[index + m - 1])
        codes[index] = code

    return codes


@njit(cache=True)
def count_unique_sorted(sorted_codes: np.ndarray) -> int:
    if sorted_codes.size == 0:
        return 0

    unique_count = 1
    previous = sorted_codes[0]
    for index in range(1, sorted_codes.size):
        current = sorted_codes[index]
        if current != previous:
            unique_count += 1
            previous = current
    return unique_count


@njit(cache=True)
def branching_histogram_from_sorted(sorted_codes: np.ndarray, sorted_bits: np.ndarray) -> tuple[int, np.ndarray]:
    histogram = np.zeros(3, dtype=np.int64)
    if sorted_codes.size == 0:
        return 0, histogram

    unique_count = 0
    current_code = sorted_codes[0]
    extension_mask = np.int64(1 << int(sorted_bits[0]))

    for index in range(1, sorted_codes.size):
        code = sorted_codes[index]
        if code != current_code:
            unique_count += 1
            histogram[1 if extension_mask in (1, 2) else 2] += 1
            current_code = code
            extension_mask = np.int64(1 << int(sorted_bits[index]))
        else:
            extension_mask |= np.int64(1 << int(sorted_bits[index]))

    unique_count += 1
    histogram[1 if extension_mask in (1, 2) else 2] += 1
    return unique_count, histogram


def analyze_factor_complexity(
    bits: np.ndarray,
    m_values: Iterable[int],
    progress=None,
    progress_base: int = 0,
    progress_message_prefix: str = "",
) -> dict:
    bit_count = int(bits.size)
    if bit_count == 0:
        raise ValueError("Cannot analyze an empty bit sequence.")

    m_values = list(m_values)
    if not m_values:
        raise ValueError("At least one m value is required.")

    results = []
    p_values: dict[int, int] = {}

    for offset, m in enumerate(m_values, start=1):
        if m <= 0:
            raise ValueError("All m values must be positive.")
        if m >= 63:
            raise ValueError("This implementation supports m <= 62.")
        if m > bit_count:
            raise ValueError(f"m={m} exceeds segment length {bit_count}.")

        codes = rolling_codes_uint64(bits, m)
        order_full = np.argsort(codes, kind="mergesort")
        sorted_codes = codes[order_full]
        p_m = count_unique_sorted(sorted_codes)
        p_values[m] = p_m

        scale_entry = {
            "m": int(m),
            "p_m": int(p_m),
            "full_space_fraction": float(p_m / (2 ** m)),
            "h_eff": float(math.log2(max(p_m, 1)) / m),
        }

        if codes.size > 1:
            codes_right = codes[:-1]
            next_bits = bits[m:]
            order_right = np.argsort(codes_right, kind="mergesort")
            right_unique, right_hist = branching_histogram_from_sorted(
                codes_right[order_right],
                next_bits[order_right],
            )

            codes_left = codes[1:]
            prev_bits = bits[:-m]
            order_left = np.argsort(codes_left, kind="mergesort")
            left_unique, left_hist = branching_histogram_from_sorted(
                codes_left[order_left],
                prev_bits[order_left],
            )

            scale_entry["right_extensions"] = _serialize_branching(right_unique, right_hist, p_m)
            scale_entry["left_extensions"] = _serialize_branching(left_unique, left_hist, p_m)
        else:
            scale_entry["right_extensions"] = _serialize_branching(0, np.zeros(3, dtype=np.int64), p_m)
            scale_entry["left_extensions"] = _serialize_branching(0, np.zeros(3, dtype=np.int64), p_m)

        results.append(scale_entry)

        if progress is not None:
            message = f"{progress_message_prefix}m={m}" if progress_message_prefix else f"m={m}"
            progress.update(progress_base + offset, message=message)

    for index, entry in enumerate(results[:-1]):
        next_entry = results[index + 1]
        entry["rho_to_next"] = float(next_entry["p_m"] / max(entry["p_m"], 1))
    results[-1]["rho_to_next"] = None

    tail_window = min(5, len(results))
    tail = results[-tail_window:]
    rho_values = [item["rho_to_next"] for item in tail if item["rho_to_next"] is not None]

    return {
        "bit_count": bit_count,
        "density": float(bits.mean()),
        "pattern_space_mode": "observed_factors_L_m",
        "scales": results,
        "tail_summary": {
            "tail_window": tail_window,
            "mean_h_eff_tail": float(np.mean([item["h_eff"] for item in tail])),
            "mean_right_branching_tail": float(
                np.mean([item["right_extensions"]["mean_extensions"] for item in tail])
            ),
            "mean_left_branching_tail": float(
                np.mean([item["left_extensions"]["mean_extensions"] for item in tail])
            ),
            "mean_rho_tail": float(np.mean(rho_values)) if rho_values else None,
        },
    }


def _serialize_branching(unique_count: int, histogram: np.ndarray, p_m: int) -> dict:
    one_extension = int(histogram[1])
    two_extensions = int(histogram[2])
    mean_extensions = 0.0
    if unique_count:
        mean_extensions = float((one_extension + 2 * two_extensions) / unique_count)

    return {
        "eligible_pattern_count": int(unique_count),
        "eligible_fraction_of_p_m": float(unique_count / max(p_m, 1)),
        "one_extension_count": one_extension,
        "two_extension_count": two_extensions,
        "one_extension_fraction": float(one_extension / unique_count) if unique_count else 0.0,
        "two_extension_fraction": float(two_extensions / unique_count) if unique_count else 0.0,
        "mean_extensions": mean_extensions,
    }
