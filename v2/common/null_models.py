from __future__ import annotations

from contextlib import nullcontext
import math
from typing import Iterable

import numpy as np
from utils.progress import ProgressIndicator

try:
    from numba import njit
except Exception:  # pragma: no cover - optional acceleration only.
    njit = None


SUPPORTED_NULLS = (
    "shuffled",
    "same-density",
    "markov1",
    "matched-lz",
    "phase-matched-lz",
    "block-entropy",
)
LZ_CALIBRATION_WINDOW_BITS = 32_768
LZ_MAX_WINDOWS = 3
MATCHED_LZ_BLOCK_BITS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1_024, 2_048, 4_096)
PHASE_MATCHED_LZ_PERIOD_BLOCKS = (2, 3, 5, 7)
MARKOV1_PROGRESS_CHUNK_BITS = 1_000_000


if njit is not None:
    @njit(cache=True)
    def _lempel_ziv_complexity_numba(arr: np.ndarray) -> int:
        n = len(arr)
        if n == 0:
            return 0

        complexity = 1
        prefix_len = 1
        while prefix_len < n:
            max_match = 0
            for start in range(prefix_len):
                match_len = 0
                while prefix_len + match_len < n and arr[start + match_len] == arr[prefix_len + match_len]:
                    match_len += 1
                    if start + match_len >= prefix_len:
                        break
                if match_len > max_match:
                    max_match = match_len

            prefix_len += max(1, max_match + 1)
            complexity += 1

        return complexity
else:
    def _lempel_ziv_complexity_numba(arr: np.ndarray) -> int:
        n = len(arr)
        if n == 0:
            return 0

        complexity = 1
        prefix_len = 1
        while prefix_len < n:
            max_match = 0
            for start in range(prefix_len):
                match_len = 0
                while prefix_len + match_len < n and arr[start + match_len] == arr[prefix_len + match_len]:
                    match_len += 1
                    if start + match_len >= prefix_len:
                        break
                if match_len > max_match:
                    max_match = match_len

            prefix_len += max(1, max_match + 1)
            complexity += 1

        return complexity


def parse_nulls(raw: str) -> list[str]:
    if not raw.strip():
        return []
    parsed = [item.strip().lower() for item in raw.split(",") if item.strip()]
    invalid = [item for item in parsed if item not in SUPPORTED_NULLS]
    if invalid:
        supported = ", ".join(SUPPORTED_NULLS)
        raise ValueError(f"Unsupported null model(s): {invalid}. Supported: {supported}")
    return parsed


def generate_controls(bits: np.ndarray, nulls: Iterable[str], seed: int) -> dict[str, np.ndarray]:
    nulls = list(nulls)
    if not nulls:
        return {}

    seed_sequence = np.random.SeedSequence(seed)
    child_sequences = seed_sequence.spawn(len(nulls))
    controls: dict[str, np.ndarray] = {}

    for null_name, child_seed in zip(nulls, child_sequences):
        child_value = int(child_seed.generate_state(1, dtype=np.uint64)[0])
        generated, _ = generate_control(bits, null_name, child_value)
        controls[null_name] = generated

    return controls


def generate_control(
    bits: np.ndarray,
    null_name: str,
    seed: int,
    *,
    show_progress: bool = False,
    progress_label: str | None = None,
) -> tuple[np.ndarray, dict[str, object] | None]:
    rng = np.random.default_rng(seed)
    if null_name == "shuffled":
        generated = bits.copy()
        rng.shuffle(generated)
        return np.ascontiguousarray(generated, dtype=np.uint8), None
    if null_name == "same-density":
        density = float(bits.mean())
        generated = (rng.random(bits.size) < density).astype(np.uint8)
        return np.ascontiguousarray(generated, dtype=np.uint8), None
    if null_name == "markov1":
        generated = generate_markov1(
            bits,
            rng,
            show_progress=show_progress,
            progress_label=progress_label,
        )
        return np.ascontiguousarray(generated, dtype=np.uint8), None
    if null_name == "matched-lz":
        generated, metadata = generate_matched_lz(
            bits,
            rng,
            show_progress=show_progress,
            progress_label=progress_label,
        )
        return np.ascontiguousarray(generated, dtype=np.uint8), metadata
    if null_name == "phase-matched-lz":
        generated, metadata = generate_phase_matched_lz(
            bits,
            rng,
            show_progress=show_progress,
            progress_label=progress_label,
        )
        return np.ascontiguousarray(generated, dtype=np.uint8), metadata
    if null_name == "block-entropy":
        generated, metadata = generate_block_entropy(
            bits,
            rng,
            show_progress=show_progress,
            progress_label=progress_label,
        )
        return np.ascontiguousarray(generated, dtype=np.uint8), metadata
    raise ValueError(f"Unsupported null model: {null_name}")


def generate_markov1(
    bits: np.ndarray,
    rng: np.random.Generator,
    *,
    show_progress: bool = False,
    progress_label: str | None = None,
) -> np.ndarray:
    if bits.size == 0:
        return np.array([], dtype=np.uint8)

    density = float(bits.mean())
    pair_count = max(0, bits.size - 1)
    counts = np.zeros((2, 2), dtype=np.int64)
    chunk_bits = max(1, min(MARKOV1_PROGRESS_CHUNK_BITS, pair_count or 1))

    progress = (
        ProgressIndicator(
            _null_progress_name("markov1", progress_label),
            total=max(1, pair_count * 2),
        )
        if show_progress
        else None
    )
    with progress or nullcontext():
        for start in range(0, pair_count, chunk_bits):
            end = min(start + chunk_bits, pair_count)
            pair_codes = (bits[start:end] << 1) | bits[start + 1 : end + 1]
            counts += np.bincount(pair_codes, minlength=4).reshape(2, 2)
            if progress is not None:
                progress.update(
                    end,
                    message=f"phase=count transitions | {end:,}/{pair_count:,} pairs",
                )

        probs = np.zeros((2, 2), dtype=np.float64)
        for state in (0, 1):
            total = counts[state].sum()
            if total == 0:
                probs[state, 1] = density
                probs[state, 0] = 1.0 - density
            else:
                probs[state] = counts[state] / total

        generated = np.empty(bits.size, dtype=np.uint8)
        generated[0] = np.uint8(1 if rng.random() < density else 0)

        for start in range(1, bits.size, chunk_bits):
            end = min(start + chunk_bits, bits.size)
            random_values = rng.random(end - start)
            generated[start:end] = _generate_markov1_chunk(
                random_values,
                int(generated[start - 1]),
                float(probs[0, 1]),
                float(probs[1, 1]),
            )
            if progress is not None:
                produced = end - 1
                progress.update(
                    pair_count + produced,
                    message=f"phase=sample chain | {produced:,}/{pair_count:,} bits",
                )

    return generated


def generate_matched_lz(
    bits: np.ndarray,
    rng: np.random.Generator,
    *,
    show_progress: bool = False,
    progress_label: str | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    if bits.size == 0:
        return np.array([], dtype=np.uint8), {
            "method": "block_shuffle_lz_match",
            "window_bits": 0,
            "window_starts": [],
            "target_density": 0.0,
            "target_lz_normalized": 0.0,
            "achieved_lz_normalized": 0.0,
            "selected_block_bits": 1,
        }

    effective_window_bits = min(int(LZ_CALIBRATION_WINDOW_BITS), int(bits.size))
    calibration_starts = calibration_window_starts(bits.size, effective_window_bits, LZ_MAX_WINDOWS)
    candidate_block_bits = [
        block_bits
        for block_bits in MATCHED_LZ_BLOCK_BITS
        if block_bits == 1 or (block_bits < bits.size and bits.size // block_bits >= 8)
    ]
    if not candidate_block_bits:
        candidate_block_bits = [1]

    total_units = len(calibration_starts) + (len(candidate_block_bits) * (1 + len(calibration_starts)))
    progress = (
        ProgressIndicator(
            _null_progress_name("matched-lz", progress_label),
            total=max(1, total_units),
        )
        if show_progress
        else None
    )

    best_bits: np.ndarray | None = None
    best_profile: dict[str, object] | None = None
    best_block_bits = candidate_block_bits[0]
    best_error = float("inf")

    progress_units = 0
    with progress or nullcontext():
        target_profile, progress_units = _estimate_lz_profile_with_progress(
            bits,
            window_bits=effective_window_bits,
            starts=calibration_starts,
            progress=progress,
            progress_current=progress_units,
            message_prefix=f"phase=target profile | windows={len(calibration_starts)}",
        )

        for candidate_index, block_bits in enumerate(candidate_block_bits, start=1):
            candidate = shuffled_block_surrogate(bits, block_bits, rng)
            progress_units += 1
            if progress is not None:
                progress.update(
                    progress_units,
                    message=(
                        f"phase=evaluate candidates | candidate {candidate_index}/{len(candidate_block_bits)} "
                        f"| block={block_bits} | best_error={best_error:.6f}"
                    ),
                )
            candidate_profile, progress_units = _estimate_lz_profile_with_progress(
                candidate,
                window_bits=int(target_profile["window_bits"]),
                starts=[int(value) for value in target_profile["window_starts"]],
                progress=progress,
                progress_current=progress_units,
                message_prefix=(
                    f"phase=evaluate candidates | candidate {candidate_index}/{len(candidate_block_bits)} "
                    f"| block={block_bits}"
                ),
            )
            error = abs(float(candidate_profile["mean_normalized_lz"]) - float(target_profile["mean_normalized_lz"]))
            if error < best_error:
                best_error = error
                best_bits = candidate
                best_profile = candidate_profile
                best_block_bits = block_bits

    assert best_bits is not None
    assert best_profile is not None
    metadata = {
        "method": "block_shuffle_lz_match",
        "window_bits": int(target_profile["window_bits"]),
        "window_starts": [int(value) for value in target_profile["window_starts"]],
        "target_density": float(bits.mean()),
        "target_lz_normalized": float(target_profile["mean_normalized_lz"]),
        "achieved_lz_normalized": float(best_profile["mean_normalized_lz"]),
        "lz_abs_error": float(best_error),
        "selected_block_bits": int(best_block_bits),
        "candidate_block_bits": candidate_block_bits,
        "preserves": ["exact_density", "within_block_patterns", "approx_global_lz"],
        "breaks": ["global_block_order", "long_range_arrangement"],
    }
    return best_bits, metadata


def generate_phase_matched_lz(
    bits: np.ndarray,
    rng: np.random.Generator,
    *,
    show_progress: bool = False,
    progress_label: str | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """LZ-matched block surrogate that preserves block phase classes.

    This is deliberately close to matched-LZ, but it shuffles blocks only
    within modulo-period lanes. It is therefore a stronger null for phase-like
    routing claims: if the readout only needs coarse phase lanes, this null has
    a fairer chance than a global block shuffle.
    """
    if bits.size == 0:
        return np.array([], dtype=np.uint8), {
            "method": "phase_class_block_shuffle_lz_match",
            "window_bits": 0,
            "window_starts": [],
            "target_density": 0.0,
            "target_lz_normalized": 0.0,
            "achieved_lz_normalized": 0.0,
            "selected_block_bits": 1,
            "selected_phase_period_blocks": 2,
        }

    effective_window_bits = min(int(LZ_CALIBRATION_WINDOW_BITS), int(bits.size))
    calibration_starts = calibration_window_starts(bits.size, effective_window_bits, LZ_MAX_WINDOWS)
    candidate_block_bits = [
        block_bits
        for block_bits in MATCHED_LZ_BLOCK_BITS
        if block_bits == 1 or (block_bits < bits.size and bits.size // block_bits >= 8)
    ] or [1]
    candidate_specs = [
        (block_bits, period_blocks)
        for block_bits in candidate_block_bits
        for period_blocks in PHASE_MATCHED_LZ_PERIOD_BLOCKS
        if bits.size // max(1, block_bits) >= period_blocks
    ]
    if not candidate_specs:
        candidate_specs = [(candidate_block_bits[0], 2)]

    total_units = len(calibration_starts) + (len(candidate_specs) * (1 + len(calibration_starts)))
    progress = (
        ProgressIndicator(
            _null_progress_name("phase-matched-lz", progress_label),
            total=max(1, total_units),
        )
        if show_progress
        else None
    )

    best_bits: np.ndarray | None = None
    best_profile: dict[str, object] | None = None
    best_block_bits = candidate_specs[0][0]
    best_period_blocks = candidate_specs[0][1]
    best_error = float("inf")

    progress_units = 0
    with progress or nullcontext():
        target_profile, progress_units = _estimate_lz_profile_with_progress(
            bits,
            window_bits=effective_window_bits,
            starts=calibration_starts,
            progress=progress,
            progress_current=progress_units,
            message_prefix=f"phase=target profile | windows={len(calibration_starts)}",
        )

        for candidate_index, (block_bits, period_blocks) in enumerate(candidate_specs, start=1):
            candidate = phase_class_block_surrogate(bits, block_bits, period_blocks, rng)
            progress_units += 1
            if progress is not None:
                progress.update(
                    progress_units,
                    message=(
                        f"phase=evaluate candidates | candidate {candidate_index}/{len(candidate_specs)} "
                        f"| block={block_bits} | period={period_blocks} | best_error={best_error:.6f}"
                    ),
                )
            candidate_profile, progress_units = _estimate_lz_profile_with_progress(
                candidate,
                window_bits=int(target_profile["window_bits"]),
                starts=[int(value) for value in target_profile["window_starts"]],
                progress=progress,
                progress_current=progress_units,
                message_prefix=(
                    f"phase=evaluate candidates | candidate {candidate_index}/{len(candidate_specs)} "
                    f"| block={block_bits} | period={period_blocks}"
                ),
            )
            error = abs(float(candidate_profile["mean_normalized_lz"]) - float(target_profile["mean_normalized_lz"]))
            if error < best_error:
                best_error = error
                best_bits = candidate
                best_profile = candidate_profile
                best_block_bits = block_bits
                best_period_blocks = period_blocks

    assert best_bits is not None
    assert best_profile is not None
    metadata = {
        "method": "phase_class_block_shuffle_lz_match",
        "window_bits": int(target_profile["window_bits"]),
        "window_starts": [int(value) for value in target_profile["window_starts"]],
        "target_density": float(bits.mean()),
        "target_lz_normalized": float(target_profile["mean_normalized_lz"]),
        "achieved_lz_normalized": float(best_profile["mean_normalized_lz"]),
        "lz_abs_error": float(best_error),
        "selected_block_bits": int(best_block_bits),
        "selected_phase_period_blocks": int(best_period_blocks),
        "candidate_block_bits": candidate_block_bits,
        "candidate_phase_period_blocks": list(PHASE_MATCHED_LZ_PERIOD_BLOCKS),
        "preserves": [
            "exact_density",
            "within_block_patterns",
            "approx_global_lz",
            "block_phase_class",
        ],
        "breaks": [
            "within_phase_block_order",
            "global_block_order",
            "exact_long_range_arrangement",
        ],
    }
    return best_bits, metadata


def generate_block_entropy(
    bits: np.ndarray,
    rng: np.random.Generator,
    *,
    show_progress: bool = False,
    progress_label: str | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Block-shuffle surrogate selected by local transition entropy, not LZ."""
    if bits.size == 0:
        return np.array([], dtype=np.uint8), {
            "method": "block_shuffle_transition_entropy_match",
            "window_bits": 0,
            "window_starts": [],
            "target_density": 0.0,
            "target_transition_entropy": 0.0,
            "achieved_transition_entropy": 0.0,
            "selected_block_bits": 1,
        }

    effective_window_bits = min(int(LZ_CALIBRATION_WINDOW_BITS), int(bits.size))
    calibration_starts = calibration_window_starts(bits.size, effective_window_bits, LZ_MAX_WINDOWS)
    candidate_block_bits = [
        block_bits
        for block_bits in MATCHED_LZ_BLOCK_BITS
        if block_bits == 1 or (block_bits < bits.size and bits.size // block_bits >= 8)
    ] or [1]

    total_units = len(calibration_starts) + (len(candidate_block_bits) * (1 + len(calibration_starts)))
    progress = (
        ProgressIndicator(
            _null_progress_name("block-entropy", progress_label),
            total=max(1, total_units),
        )
        if show_progress
        else None
    )

    best_bits: np.ndarray | None = None
    best_profile: dict[str, object] | None = None
    best_block_bits = candidate_block_bits[0]
    best_error = float("inf")

    progress_units = 0
    with progress or nullcontext():
        target_profile, progress_units = _estimate_transition_entropy_profile_with_progress(
            bits,
            window_bits=effective_window_bits,
            starts=calibration_starts,
            progress=progress,
            progress_current=progress_units,
            message_prefix=f"phase=target entropy profile | windows={len(calibration_starts)}",
        )

        for candidate_index, block_bits in enumerate(candidate_block_bits, start=1):
            candidate = shuffled_block_surrogate(bits, block_bits, rng)
            progress_units += 1
            if progress is not None:
                progress.update(
                    progress_units,
                    message=(
                        f"phase=evaluate entropy candidates | candidate {candidate_index}/{len(candidate_block_bits)} "
                        f"| block={block_bits} | best_error={best_error:.6f}"
                    ),
                )
            candidate_profile, progress_units = _estimate_transition_entropy_profile_with_progress(
                candidate,
                window_bits=int(target_profile["window_bits"]),
                starts=[int(value) for value in target_profile["window_starts"]],
                progress=progress,
                progress_current=progress_units,
                message_prefix=(
                    f"phase=evaluate entropy candidates | candidate {candidate_index}/{len(candidate_block_bits)} "
                    f"| block={block_bits}"
                ),
            )
            error = abs(
                float(candidate_profile["mean_transition_entropy"])
                - float(target_profile["mean_transition_entropy"])
            )
            if error < best_error:
                best_error = error
                best_bits = candidate
                best_profile = candidate_profile
                best_block_bits = block_bits

    assert best_bits is not None
    assert best_profile is not None
    metadata = {
        "method": "block_shuffle_transition_entropy_match",
        "window_bits": int(target_profile["window_bits"]),
        "window_starts": [int(value) for value in target_profile["window_starts"]],
        "target_density": float(bits.mean()),
        "target_transition_entropy": float(target_profile["mean_transition_entropy"]),
        "achieved_transition_entropy": float(best_profile["mean_transition_entropy"]),
        "transition_entropy_abs_error": float(best_error),
        "selected_block_bits": int(best_block_bits),
        "candidate_block_bits": candidate_block_bits,
        "preserves": ["exact_density", "within_block_patterns", "approx_transition_entropy"],
        "breaks": ["global_block_order", "long_range_arrangement"],
    }
    return best_bits, metadata


def shuffled_block_surrogate(bits: np.ndarray, block_bits: int, rng: np.random.Generator) -> np.ndarray:
    if bits.size == 0:
        return np.array([], dtype=np.uint8)
    if block_bits <= 1 or bits.size < block_bits * 2:
        generated = bits.copy()
        rng.shuffle(generated)
        return generated

    full_block_count = bits.size // block_bits
    head_size = full_block_count * block_bits
    blocks = bits[:head_size].reshape(full_block_count, block_bits)
    permutation = rng.permutation(full_block_count)

    generated = np.empty(bits.size, dtype=np.uint8)
    generated[:head_size] = blocks[permutation].reshape(head_size)
    if head_size < bits.size:
        generated[head_size:] = bits[head_size:]
    return generated


def phase_class_block_surrogate(
    bits: np.ndarray,
    block_bits: int,
    phase_period_blocks: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if bits.size == 0:
        return np.array([], dtype=np.uint8)
    if phase_period_blocks <= 1:
        return shuffled_block_surrogate(bits, block_bits, rng)
    if block_bits <= 0:
        raise ValueError("block_bits must be positive")
    if bits.size < block_bits * 2:
        generated = bits.copy()
        rng.shuffle(generated)
        return generated

    full_block_count = bits.size // block_bits
    head_size = full_block_count * block_bits
    blocks = bits[:head_size].reshape(full_block_count, block_bits)
    generated_blocks = np.empty_like(blocks)

    for phase in range(phase_period_blocks):
        indices = np.arange(phase, full_block_count, phase_period_blocks, dtype=np.int64)
        if indices.size == 0:
            continue
        if indices.size == 1:
            generated_blocks[indices] = blocks[indices]
            continue
        generated_blocks[indices] = blocks[rng.permutation(indices)]

    generated = np.empty(bits.size, dtype=np.uint8)
    generated[:head_size] = generated_blocks.reshape(head_size)
    if head_size < bits.size:
        generated[head_size:] = bits[head_size:]
    return generated


def estimate_lz_profile(
    bits: np.ndarray,
    *,
    window_bits: int = LZ_CALIBRATION_WINDOW_BITS,
    starts: list[int] | None = None,
    max_windows: int = LZ_MAX_WINDOWS,
) -> dict[str, object]:
    profile, _ = _estimate_lz_profile_with_progress(
        bits,
        window_bits=window_bits,
        starts=starts,
        max_windows=max_windows,
    )
    return profile


def estimate_transition_entropy_profile(
    bits: np.ndarray,
    *,
    window_bits: int = LZ_CALIBRATION_WINDOW_BITS,
    starts: list[int] | None = None,
    max_windows: int = LZ_MAX_WINDOWS,
) -> dict[str, object]:
    profile, _ = _estimate_transition_entropy_profile_with_progress(
        bits,
        window_bits=window_bits,
        starts=starts,
        max_windows=max_windows,
    )
    return profile


def _estimate_transition_entropy_profile_with_progress(
    bits: np.ndarray,
    *,
    window_bits: int = LZ_CALIBRATION_WINDOW_BITS,
    starts: list[int] | None = None,
    max_windows: int = LZ_MAX_WINDOWS,
    progress: ProgressIndicator | None = None,
    progress_current: int = 0,
    message_prefix: str = "",
) -> tuple[dict[str, object], int]:
    if bits.size == 0:
        return {
            "window_bits": 0,
            "window_starts": [],
            "values": [],
            "mean_transition_entropy": 0.0,
        }, progress_current

    effective_window_bits = min(int(window_bits), int(bits.size))
    calibration_starts = starts if starts is not None else calibration_window_starts(bits.size, effective_window_bits, max_windows)
    values = []
    for window_index, start in enumerate(calibration_starts, start=1):
        window = np.ascontiguousarray(bits[start : start + effective_window_bits], dtype=np.uint8)
        values.append(transition_conditional_entropy(window))
        progress_current += 1
        if progress is not None:
            message = message_prefix or "phase=transition entropy profile"
            progress.update(
                progress_current,
                message=f"{message} | window {window_index}/{len(calibration_starts)}",
            )

    return {
        "window_bits": effective_window_bits,
        "window_starts": calibration_starts,
        "values": values,
        "mean_transition_entropy": float(np.mean(values)),
    }, progress_current


def _estimate_lz_profile_with_progress(
    bits: np.ndarray,
    *,
    window_bits: int = LZ_CALIBRATION_WINDOW_BITS,
    starts: list[int] | None = None,
    max_windows: int = LZ_MAX_WINDOWS,
    progress: ProgressIndicator | None = None,
    progress_current: int = 0,
    message_prefix: str = "",
) -> tuple[dict[str, object], int]:
    if bits.size == 0:
        return {
            "window_bits": 0,
            "window_starts": [],
            "values": [],
            "mean_normalized_lz": 0.0,
        }, progress_current

    effective_window_bits = min(int(window_bits), int(bits.size))
    calibration_starts = starts if starts is not None else calibration_window_starts(bits.size, effective_window_bits, max_windows)
    values = []
    for window_index, start in enumerate(calibration_starts, start=1):
        window = np.ascontiguousarray(bits[start : start + effective_window_bits], dtype=np.uint8)
        values.append(normalized_lz_complexity(window))
        progress_current += 1
        if progress is not None:
            message = message_prefix or "phase=lz profile"
            progress.update(
                progress_current,
                message=f"{message} | window {window_index}/{len(calibration_starts)}",
            )

    return {
        "window_bits": effective_window_bits,
        "window_starts": calibration_starts,
        "values": values,
        "mean_normalized_lz": float(np.mean(values)),
    }, progress_current


def calibration_window_starts(total_bits: int, window_bits: int, max_windows: int) -> list[int]:
    if total_bits <= window_bits or max_windows <= 1:
        return [0]

    max_start = total_bits - window_bits
    anchors = [0, max_start // 2, max_start]
    starts: list[int] = []
    for anchor in anchors[:max_windows]:
        anchor = int(anchor)
        if anchor not in starts:
            starts.append(anchor)
    return starts


def normalized_lz_complexity(bits: np.ndarray) -> float:
    n = int(bits.size)
    if n < 2:
        return 0.0
    complexity = _lempel_ziv_complexity_numba(np.ascontiguousarray(bits, dtype=np.uint8))
    return float(complexity / (n / math.log2(n)))


def transition_conditional_entropy(bits: np.ndarray) -> float:
    if bits.size < 2:
        return 0.0
    pair_codes = (bits[:-1] << 1) | bits[1:]
    counts = np.bincount(pair_codes, minlength=4).reshape(2, 2).astype(np.float64)
    state_counts = counts.sum(axis=1)
    total_pairs = float(state_counts.sum())
    if total_pairs <= 0.0:
        return 0.0

    entropy = 0.0
    for state in (0, 1):
        state_total = float(state_counts[state])
        if state_total <= 0.0:
            continue
        probabilities = counts[state] / state_total
        state_entropy = 0.0
        for probability in probabilities:
            if probability > 0.0:
                state_entropy -= float(probability) * math.log2(float(probability))
        entropy += (state_total / total_pairs) * state_entropy
    return float(entropy)


if njit is not None:
    @njit(cache=True)
    def _generate_markov1_chunk(
        random_values: np.ndarray,
        start_prev: int,
        p01: float,
        p11: float,
    ) -> np.ndarray:
        generated = np.empty(random_values.size, dtype=np.uint8)
        prev = start_prev
        for index in range(random_values.size):
            prob1 = p11 if prev == 1 else p01
            bit = 1 if random_values[index] < prob1 else 0
            generated[index] = bit
            prev = bit
        return generated
else:
    def _generate_markov1_chunk(
        random_values: np.ndarray,
        start_prev: int,
        p01: float,
        p11: float,
    ) -> np.ndarray:
        generated = np.empty(random_values.size, dtype=np.uint8)
        prev = start_prev
        for index, random_value in enumerate(random_values):
            prob1 = p11 if prev == 1 else p01
            bit = 1 if random_value < prob1 else 0
            generated[index] = bit
            prev = bit
        return generated


def _null_progress_name(null_name: str, progress_label: str | None) -> str:
    if progress_label:
        return f"Build null surrogate ({progress_label})"
    return f"Build null surrogate ({null_name})"
