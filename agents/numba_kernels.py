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


# ============================================================================
# STRUCTURAL PATTERN DETECTION KERNELS
# ============================================================================

# Character codes for fast comparison
CHAR_OPEN = np.uint8(ord('('))
CHAR_CLOSE = np.uint8(ord(')'))
CHAR_ZERO = np.uint8(ord('0'))
CHAR_ONE = np.uint8(ord('1'))


@njit(cache=True)
def _analyze_nesting_numba(seq: np.ndarray) -> Tuple[np.int64, np.int64, np.ndarray, np.ndarray]:
    """
    Numba-optimized nesting analysis.

    Returns:
        max_depth: Maximum nesting depth encountered
        total_parens: Total number of parentheses
        depth_dist: Array of counts per depth level (index = depth)
        transitions: Flattened array of transition counts (from_depth * MAX + to_depth)
    """
    n = len(seq)
    # MAX_DEPTH = 256: Verified sufficient for all HSI variants (Dec 2025)
    # Empirical max_depth: F=9, B=9, E=13, H=9, A=0 (see scripts/check_depth.py)
    MAX_DEPTH = 256

    depth = np.int64(0)
    max_depth = np.int64(0)
    prev_depth = np.int64(0)
    total_parens = np.int64(0)

    depth_dist = np.zeros(MAX_DEPTH, dtype=np.int64)
    # Transitions stored as flat array: transitions[from * MAX_DEPTH + to]
    transitions = np.zeros(MAX_DEPTH * MAX_DEPTH, dtype=np.int64)

    for i in range(n):
        char = seq[i]

        if char == CHAR_OPEN:
            depth += 1
            if depth > max_depth:
                max_depth = depth
            if depth < MAX_DEPTH:
                depth_dist[depth] += 1
            total_parens += 1

            # Track transition
            if total_parens > 1:
                if prev_depth < MAX_DEPTH and depth < MAX_DEPTH:
                    transitions[prev_depth * MAX_DEPTH + depth] += 1
            prev_depth = depth

        elif char == CHAR_CLOSE:
            total_parens += 1
            # Track transition before decreasing
            if prev_depth < MAX_DEPTH and depth < MAX_DEPTH:
                transitions[prev_depth * MAX_DEPTH + depth] += 1
            prev_depth = depth
            depth -= 1

    return max_depth, total_parens, depth_dist, transitions


@njit(cache=True)
def _analyze_nesting_numba_chunk(seq: np.ndarray,
                                  initial_depth: np.int64,
                                  initial_prev_depth: np.int64,
                                  initial_total_parens: np.int64,
                                  depth_dist: np.ndarray,
                                  transitions: np.ndarray) -> Tuple[np.int64, np.int64, np.int64, np.int64]:
    """
    Numba-optimized nesting analysis for a CHUNK of data.

    This version takes mutable arrays (depth_dist, transitions) and updates them in-place.
    State (depth, prev_depth, total_parens) is passed in and returned for chaining chunks.

    Args:
        seq: Chunk of sequence bytes
        initial_depth: Current nesting depth from previous chunk
        initial_prev_depth: Previous depth for transition tracking
        initial_total_parens: Total parentheses count so far
        depth_dist: Array to update with depth distribution (modified in-place)
        transitions: Array to update with transitions (modified in-place)

    Returns:
        max_depth: Maximum depth seen in this chunk
        final_depth: Depth at end of chunk (for next chunk)
        final_prev_depth: Prev depth at end of chunk
        final_total_parens: Total parens including this chunk
    """
    n = len(seq)
    MAX_DEPTH = 256

    depth = initial_depth
    max_depth = initial_depth
    prev_depth = initial_prev_depth
    total_parens = initial_total_parens

    for i in range(n):
        char = seq[i]

        if char == CHAR_OPEN:
            depth += 1
            if depth > max_depth:
                max_depth = depth
            if depth < MAX_DEPTH:
                depth_dist[depth] += 1
            total_parens += 1

            if total_parens > 1:
                if prev_depth < MAX_DEPTH and depth < MAX_DEPTH:
                    transitions[prev_depth * MAX_DEPTH + depth] += 1
            prev_depth = depth

        elif char == CHAR_CLOSE:
            total_parens += 1
            if prev_depth < MAX_DEPTH and depth < MAX_DEPTH:
                transitions[prev_depth * MAX_DEPTH + depth] += 1
            prev_depth = depth
            depth -= 1

    return max_depth, depth, prev_depth, total_parens


@njit(cache=True)
def _analyze_containment_numba(seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.int64, np.float64, np.float64]:
    """
    Numba-optimized containment analysis - single pass.

    Key insight: Instead of storing chars in lists, we track positions and
    compute lengths/entropy directly using marker arrays.

    Returns:
        length_counts: Array where length_counts[len] = count of absolutes with that length
        content_stats: Array of [zeros_count, ones_count] per absolute (flattened)
        num_absolutes: Total number of absolutes found
        sum_lengths: Sum of all lengths (for mean calculation)
        sum_sq_lengths: Sum of squared lengths (for std calculation)
    """
    n = len(seq)
    MAX_LENGTH = 10000  # Max length to track in distribution
    MAX_DEPTH = 256

    # Stack to track open parentheses positions
    # stack[i] = (start_pos, zeros_so_far, ones_so_far)
    stack_starts = np.zeros(MAX_DEPTH, dtype=np.int64)
    stack_zeros = np.zeros(MAX_DEPTH, dtype=np.int64)
    stack_ones = np.zeros(MAX_DEPTH, dtype=np.int64)
    stack_ptr = np.int64(0)  # Stack pointer

    # Results
    length_counts = np.zeros(MAX_LENGTH, dtype=np.int64)
    num_absolutes = np.int64(0)

    # Welford's algorithm accumulators for entropy
    entropy_count = np.int64(0)
    entropy_mean = np.float64(0.0)
    entropy_m2 = np.float64(0.0)

    # Length statistics
    sum_lengths = np.float64(0.0)
    sum_sq_lengths = np.float64(0.0)

    for i in range(n):
        char = seq[i]

        if char == CHAR_OPEN:
            # Push new frame to stack
            if stack_ptr < MAX_DEPTH:
                stack_starts[stack_ptr] = i
                stack_zeros[stack_ptr] = 0
                stack_ones[stack_ptr] = 0
                stack_ptr += 1

        elif char == CHAR_CLOSE:
            if stack_ptr > 0:
                stack_ptr -= 1
                # Pop and analyze
                zeros = stack_zeros[stack_ptr]
                ones = stack_ones[stack_ptr]
                content_len = zeros + ones

                # Update length distribution
                if content_len < MAX_LENGTH:
                    length_counts[content_len] += 1

                # Update length stats (Welford-style)
                sum_lengths += content_len
                sum_sq_lengths += content_len * content_len

                # Calculate entropy and update accumulator
                if content_len > 0:
                    p0 = zeros / content_len
                    p1 = ones / content_len
                    ent = np.float64(0.0)
                    if p0 > 0:
                        ent -= p0 * np.log2(p0)
                    if p1 > 0:
                        ent -= p1 * np.log2(p1)

                    # Welford update for entropy
                    entropy_count += 1
                    delta = ent - entropy_mean
                    entropy_mean += delta / entropy_count
                    delta2 = ent - entropy_mean
                    entropy_m2 += delta * delta2

                num_absolutes += 1

        elif char == CHAR_ZERO:
            # Add to all open absolutes
            for j in range(stack_ptr):
                stack_zeros[j] += 1

        elif char == CHAR_ONE:
            # Add to all open absolutes
            for j in range(stack_ptr):
                stack_ones[j] += 1

    # Pack entropy stats into a small array
    entropy_stats = np.array([
        np.float64(entropy_count),
        entropy_mean,
        entropy_m2
    ], dtype=np.float64)

    return length_counts, entropy_stats, num_absolutes, sum_lengths, sum_sq_lengths


@njit(cache=True)
def _analyze_stratified_numba(seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba-optimized stratified order analysis.

    Returns:
        depth_zeros: Array of zero counts per depth
        depth_ones: Array of one counts per depth
        depth_seq_counts: Array of sequence counts per depth
    """
    n = len(seq)
    MAX_DEPTH = 256

    depth = np.int64(0)
    current_zeros = np.int64(0)
    current_ones = np.int64(0)

    depth_zeros = np.zeros(MAX_DEPTH, dtype=np.int64)
    depth_ones = np.zeros(MAX_DEPTH, dtype=np.int64)
    depth_seq_counts = np.zeros(MAX_DEPTH, dtype=np.int64)

    for i in range(n):
        char = seq[i]

        if char == CHAR_OPEN:
            # Flush current sequence to current depth
            if (current_zeros > 0 or current_ones > 0) and depth > 0 and depth < MAX_DEPTH:
                depth_zeros[depth] += current_zeros
                depth_ones[depth] += current_ones
                depth_seq_counts[depth] += 1
            current_zeros = 0
            current_ones = 0
            depth += 1

        elif char == CHAR_CLOSE:
            # Flush current sequence to current depth
            if (current_zeros > 0 or current_ones > 0) and depth > 0 and depth < MAX_DEPTH:
                depth_zeros[depth] += current_zeros
                depth_ones[depth] += current_ones
                depth_seq_counts[depth] += 1
            current_zeros = 0
            current_ones = 0
            depth -= 1

        elif char == CHAR_ZERO:
            current_zeros += 1

        elif char == CHAR_ONE:
            current_ones += 1

    return depth_zeros, depth_ones, depth_seq_counts


@njit(cache=True)
def _analyze_containment_numba_chunk(seq: np.ndarray,
                                      stack_starts: np.ndarray,
                                      stack_zeros: np.ndarray,
                                      stack_ones: np.ndarray,
                                      initial_stack_ptr: np.int64,
                                      length_counts: np.ndarray,
                                      entropy_stats: np.ndarray,
                                      sum_lengths_in: np.float64,
                                      sum_sq_lengths_in: np.float64,
                                      num_absolutes_in: np.int64) -> Tuple[np.int64, np.int64, np.float64, np.float64]:
    """
    Numba-optimized containment analysis for a CHUNK.

    Updates arrays in-place, returns state for next chunk.
    """
    n = len(seq)
    MAX_DEPTH = 256
    MAX_LENGTH = 10000

    stack_ptr = initial_stack_ptr
    num_absolutes = num_absolutes_in
    sum_lengths = sum_lengths_in
    sum_sq_lengths = sum_sq_lengths_in

    # Unpack entropy stats
    entropy_count = np.int64(entropy_stats[0])
    entropy_mean = entropy_stats[1]
    entropy_m2 = entropy_stats[2]

    for i in range(n):
        char = seq[i]

        if char == CHAR_OPEN:
            if stack_ptr < MAX_DEPTH:
                stack_starts[stack_ptr] = i
                stack_zeros[stack_ptr] = 0
                stack_ones[stack_ptr] = 0
                stack_ptr += 1

        elif char == CHAR_CLOSE:
            if stack_ptr > 0:
                stack_ptr -= 1
                zeros = stack_zeros[stack_ptr]
                ones = stack_ones[stack_ptr]
                content_len = zeros + ones

                if content_len < MAX_LENGTH:
                    length_counts[content_len] += 1

                sum_lengths += content_len
                sum_sq_lengths += content_len * content_len

                if content_len > 0:
                    p0 = zeros / content_len
                    p1 = ones / content_len
                    ent = np.float64(0.0)
                    if p0 > 0:
                        ent -= p0 * np.log2(p0)
                    if p1 > 0:
                        ent -= p1 * np.log2(p1)

                    entropy_count += 1
                    delta = ent - entropy_mean
                    entropy_mean += delta / entropy_count
                    delta2 = ent - entropy_mean
                    entropy_m2 += delta * delta2

                num_absolutes += 1

        elif char == CHAR_ZERO:
            for j in range(stack_ptr):
                stack_zeros[j] += 1

        elif char == CHAR_ONE:
            for j in range(stack_ptr):
                stack_ones[j] += 1

    # Pack entropy stats back
    entropy_stats[0] = np.float64(entropy_count)
    entropy_stats[1] = entropy_mean
    entropy_stats[2] = entropy_m2

    return stack_ptr, num_absolutes, sum_lengths, sum_sq_lengths


@njit(cache=True)
def _analyze_stratified_numba_chunk(seq: np.ndarray,
                                     initial_depth: np.int64,
                                     initial_zeros: np.int64,
                                     initial_ones: np.int64,
                                     depth_zeros: np.ndarray,
                                     depth_ones: np.ndarray,
                                     depth_seq_counts: np.ndarray) -> Tuple[np.int64, np.int64, np.int64]:
    """
    Numba-optimized stratified analysis for a CHUNK.

    Updates arrays in-place, returns state for next chunk.
    """
    n = len(seq)
    MAX_DEPTH = 256

    depth = initial_depth
    current_zeros = initial_zeros
    current_ones = initial_ones

    for i in range(n):
        char = seq[i]

        if char == CHAR_OPEN:
            if (current_zeros > 0 or current_ones > 0) and depth > 0 and depth < MAX_DEPTH:
                depth_zeros[depth] += current_zeros
                depth_ones[depth] += current_ones
                depth_seq_counts[depth] += 1
            current_zeros = 0
            current_ones = 0
            depth += 1

        elif char == CHAR_CLOSE:
            if (current_zeros > 0 or current_ones > 0) and depth > 0 and depth < MAX_DEPTH:
                depth_zeros[depth] += current_zeros
                depth_ones[depth] += current_ones
                depth_seq_counts[depth] += 1
            current_zeros = 0
            current_ones = 0
            depth -= 1

        elif char == CHAR_ZERO:
            current_zeros += 1

        elif char == CHAR_ONE:
            current_ones += 1

    return depth, current_zeros, current_ones
