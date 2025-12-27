#!/usr/bin/env python3
"""
Numba-optimized kernels for Level 1 HSI analysis.

These functions provide 10-100x speedup over pure Python implementations
by compiling to native machine code using Numba JIT.
"""

import numpy as np
from numba import njit, prange
from typing import Tuple, Dict, List

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def str_to_bytes(s: str) -> np.ndarray:
    """Convert string to numpy byte array for Numba processing."""
    return np.frombuffer(s.encode('ascii'), dtype=np.uint8)


def bytes_to_str(arr: np.ndarray) -> str:
    """Convert numpy byte array back to string."""
    return arr.tobytes().decode('ascii')


# ============================================================================
# PATTERN DETECTION KERNELS
# ============================================================================

@njit(cache=True)
def hash_pattern(arr: np.ndarray) -> np.int64:
    """Fast hash for a byte array pattern."""
    h = np.int64(0)
    for i in range(len(arr)):
        h = h * 31 + arr[i]
    return h


@njit(parallel=True, cache=True)
def sliding_window_hashes(sequence: np.ndarray, 
                          min_len: int, 
                          max_len: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute hashes for all sliding window patterns.
    
    Returns:
        hashes: Array of pattern hashes
        starts: Array of start positions
        lengths: Array of pattern lengths
    """
    n = len(sequence)
    
    # Calculate total number of patterns
    total = 0
    for length in range(min_len, max_len + 1):
        total += max(0, n - length + 1)
    
    hashes = np.zeros(total, dtype=np.int64)
    starts = np.zeros(total, dtype=np.int32)
    lengths = np.zeros(total, dtype=np.int32)
    
    idx = 0
    for length in range(min_len, max_len + 1):
        for i in prange(n - length + 1):
            h = np.int64(0)
            for j in range(length):
                h = h * 31 + sequence[i + j]
            local_idx = idx + i
            hashes[local_idx] = h
            starts[local_idx] = i
            lengths[local_idx] = length
        idx += n - length + 1
    
    return hashes, starts, lengths


@njit(parallel=True, cache=True)
def count_pattern_occurrences(sequence: np.ndarray,
                               pattern_hashes: np.ndarray,
                               pattern_starts: np.ndarray, 
                               pattern_lengths: np.ndarray) -> np.ndarray:
    """Count occurrences of each unique pattern hash."""
    n = len(pattern_hashes)
    counts = np.zeros(n, dtype=np.int32)
    
    for i in prange(n):
        h = pattern_hashes[i]
        count = 0
        for j in range(n):
            if pattern_hashes[j] == h:
                count += 1
        counts[i] = count
    
    return counts


@njit(parallel=True, cache=True)
def block_similarities(blocks: np.ndarray) -> np.ndarray:
    """
    Compute similarity matrix between blocks.
    
    Args:
        blocks: 2D array of shape (n_blocks, block_size)
    
    Returns:
        similarities: 2D array of shape (n_blocks, n_blocks)
    """
    n = blocks.shape[0]
    block_size = blocks.shape[1]
    similarities = np.zeros((n, n), dtype=np.float64)
    
    for i in prange(n):
        for j in range(i, n):
            if i == j:
                similarities[i, j] = 1.0
            else:
                matches = 0
                for k in range(block_size):
                    if blocks[i, k] == blocks[j, k]:
                        matches += 1
                sim = matches / block_size
                similarities[i, j] = sim
                similarities[j, i] = sim
    
    return similarities


# ============================================================================
# MARKOV TRANSITION KERNELS  
# ============================================================================

@njit(cache=True)
def count_transitions_order1(sequence: np.ndarray) -> np.ndarray:
    """Count first-order Markov transitions (single char context)."""
    # For binary sequences: 2 possible contexts, 2 possible next states
    # transitions[context][next] = count
    transitions = np.zeros((256, 256), dtype=np.int64)
    
    for i in range(len(sequence) - 1):
        ctx = sequence[i]
        nxt = sequence[i + 1]
        transitions[ctx, nxt] += 1
    
    return transitions


@njit(cache=True)
def count_transitions_order2(sequence: np.ndarray) -> np.ndarray:
    """Count second-order Markov transitions (2-char context)."""
    # 256*256 possible 2-char contexts, 256 possible next states
    # Using hash: ctx_hash = c1 * 256 + c2
    transitions = np.zeros((65536, 256), dtype=np.int64)

    for i in range(len(sequence) - 2):
        ctx_hash = sequence[i] * 256 + sequence[i + 1]
        nxt = sequence[i + 2]
        transitions[ctx_hash, nxt] += 1

    return transitions


@njit(cache=True)
def count_transitions_order3(sequence: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Count third-order Markov transitions using arrays."""
    n = len(sequence)
    if n < 4:
        # Must match types with normal return: int64, uint8, int64
        return (np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.uint8),
                np.zeros(0, dtype=np.int64))

    # Store as (hash, next_char, count) tuples
    max_entries = n - 3
    ctx_hashes = np.zeros(max_entries, dtype=np.int64)
    next_chars = np.zeros(max_entries, dtype=np.uint8)

    for i in range(n - 3):
        h = sequence[i] * 65536 + sequence[i+1] * 256 + sequence[i+2]
        ctx_hashes[i] = h
        next_chars[i] = sequence[i + 3]

    # Count unique combinations
    counts = np.ones(max_entries, dtype=np.int64)

    return ctx_hashes, next_chars, counts


# ============================================================================
# VALIDATION KERNELS
# ============================================================================

@njit(cache=True)
def validate_markov_rule(sequence: np.ndarray,
                         context: np.ndarray,
                         expected_next: np.uint8) -> Tuple[int, int]:
    """
    Validate a Markov rule by counting matches.

    Returns:
        (correct_predictions, total_occurrences)
    """
    n = len(sequence)
    ctx_len = len(context)
    correct = 0
    total = 0

    for i in range(n - ctx_len):
        # Check if context matches
        match = True
        for j in range(ctx_len):
            if sequence[i + j] != context[j]:
                match = False
                break

        if match:
            total += 1
            if i + ctx_len < n and sequence[i + ctx_len] == expected_next:
                correct += 1

    return correct, total


@njit(cache=True)
def find_pattern_positions(sequence: np.ndarray,
                           pattern: np.ndarray) -> np.ndarray:
    """Find all positions where pattern occurs in sequence."""
    n = len(sequence)
    p_len = len(pattern)

    # First pass: count occurrences
    count = 0
    for i in range(n - p_len + 1):
        match = True
        for j in range(p_len):
            if sequence[i + j] != pattern[j]:
                match = False
                break
        if match:
            count += 1

    # Second pass: store positions
    positions = np.zeros(count, dtype=np.int32)
    idx = 0
    for i in range(n - p_len + 1):
        match = True
        for j in range(p_len):
            if sequence[i + j] != pattern[j]:
                match = False
                break
        if match:
            positions[idx] = i
            idx += 1

    return positions


@njit(cache=True)
def validate_periodicity(positions: np.ndarray, expected_period: int) -> Tuple[int, int]:
    """Validate periodicity rule by checking distances between occurrences."""
    if len(positions) < 2:
        return 0, 0

    correct = 0
    total = len(positions) - 1

    for i in range(len(positions) - 1):
        distance = positions[i + 1] - positions[i]
        if distance == expected_period:
            correct += 1

    return correct, total


# ============================================================================
# CONTEXT ANALYSIS KERNELS
# ============================================================================

@njit(cache=True)
def extract_contexts_around_pattern(sequence: np.ndarray,
                                    pattern: np.ndarray,
                                    window: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract context windows before and after each pattern occurrence.

    Returns:
        contexts_before: 2D array of shape (n_occurrences, window)
        contexts_after: 2D array of shape (n_occurrences, window)
    """
    positions = find_pattern_positions(sequence, pattern)
    n_pos = len(positions)
    p_len = len(pattern)
    n = len(sequence)

    contexts_before = np.zeros((n_pos, window), dtype=np.uint8)
    contexts_after = np.zeros((n_pos, window), dtype=np.uint8)

    for i in range(n_pos):
        pos = positions[i]

        # Context before
        start = max(0, pos - window)
        for j in range(window):
            idx = start + j
            if idx < pos and idx >= 0:
                contexts_before[i, j] = sequence[idx]

        # Context after
        after_start = pos + p_len
        for j in range(window):
            idx = after_start + j
            if idx < n:
                contexts_after[i, j] = sequence[idx]

    return contexts_before, contexts_after


# ============================================================================
# AUTOSIMILARITY KERNEL
# ============================================================================

@njit(cache=True)
def compute_autosimilarity(pattern: np.ndarray) -> float:
    """Compute autosimilarity of a pattern."""
    n = len(pattern)
    if n < 4:
        return 0.0

    half = n // 2
    first_half = pattern[:half]
    second_half = pattern[half:half*2]

    matches = 0
    for i in range(half):
        if first_half[i] == second_half[i]:
            matches += 1

    return matches / half

