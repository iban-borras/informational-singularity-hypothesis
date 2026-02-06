#!/usr/bin/env python3
"""
🔀 HSI Level 2 - Transfer Entropy Multi-Scale Analysis

Implements Antigravity's Proposal 2: Measure information flow between
different scales of the Φ sequence to detect causal relationships.

Transfer Entropy (TE) measures the directed information flow from X → Y:
    TE(X→Y) = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)

If TE(X→Y) > 0, then knowing X's past helps predict Y's future beyond
knowing Y's past alone.

Multi-Scale Application:
- Coarse-grain the sequence at different scales (2^k blocks)
- Measure TE between scales: Does micro-structure predict macro-structure?
- If TE(micro→macro) > TE(macro→micro): Bottom-up emergence
- If TE(macro→micro) > TE(micro→macro): Top-down causation

Scientific Value:
This can reveal if HSI has genuine multi-scale causal structure
(like living systems) vs pure noise.

Requirements:
    pip install pyinform  # or use our simplified implementation

Usage:
    python level2_transfer_entropy.py --variant B --iteration 18
    python level2_transfer_entropy.py --variants B D E --iteration 18 --compare
    python level2_transfer_entropy.py --variant B --iteration 18 --scales 1 2 4 8 16

Author: Sophia (for Iban & Antigravity)
Date: December 2025
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

import numpy as np

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable

# Try to import numba for JIT compilation
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Fallback decorators
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    def prange(*args):
        return range(*args)

# Try to import pyinform (specialized TE library)
try:
    import pyinform
    from pyinform import transfer_entropy as te
    HAS_PYINFORM = True
except ImportError:
    HAS_PYINFORM = False

# Local imports
from metrics.emergence_index import load_phi_sequence

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

RESULTS_DIR = Path("results")
TE_DIR = RESULTS_DIR / "level2" / "transfer_entropy"

# Default parameters
# Scales chosen to capture fractal structures while maintaining entropy variability
# Too large scales (>64) often collapse to uniform density in binary discretization
DEFAULT_SCALES = [4, 8, 16, 32, 64, 128]  # Block sizes for coarse-graining
DEFAULT_HISTORY = 3  # k parameter (history length)
DEFAULT_MAX_SAMPLES = 100_000  # Limit for TE computation

# ═══════════════════════════════════════════════════════════════════════════════
# COARSE-GRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def coarse_grain(sequence: np.ndarray, block_size: int, method: str = 'binary') -> np.ndarray:
    """
    Coarse-grain a binary sequence at a given block size.

    Methods:
    - 'binary': Majority vote → 0 or 1 (standard approach)
    - 'density': Proportion of 1s → continuous [0, 1] (preserves gradients)

    Args:
        sequence: Binary sequence as numpy array
        block_size: Size of blocks to aggregate
        method: 'binary' or 'density'

    Returns:
        Coarse-grained sequence (int8 for binary, float32 for density)
    """
    n = len(sequence)
    n_blocks = n // block_size

    if n_blocks == 0:
        return sequence

    # Reshape into blocks
    truncated = sequence[:n_blocks * block_size]
    blocks = truncated.reshape(n_blocks, block_size)

    if method == 'density':
        # Proportion of 1s in each block → continuous value [0, 1]
        coarse = blocks.mean(axis=1).astype(np.float32)
    else:
        # Majority vote: 1 if sum > half, else 0
        coarse = (blocks.sum(axis=1) > block_size / 2).astype(np.int8)

    return coarse


def fast_string_to_array(sequence: str) -> np.ndarray:
    """
    Convert binary string to numpy array efficiently.
    Uses frombuffer for ~10x speedup over list comprehension.
    """
    # Convert string to bytes and then to numpy
    byte_array = np.frombuffer(sequence.encode('ascii'), dtype=np.uint8)
    # '0' is ASCII 48, '1' is ASCII 49
    return (byte_array - ord('0')).astype(np.int8)


def create_multiscale_representation(sequence: str,
                                      scales: List[int],
                                      max_samples: int,
                                      method: str = 'binary',
                                      verbose: bool = True) -> Dict[int, np.ndarray]:
    """
    Create coarse-grained representations at multiple scales.

    Args:
        sequence: Binary string
        scales: List of block sizes
        max_samples: Maximum samples per scale
        method: 'binary' (majority vote) or 'density' (proportion of 1s)
        verbose: Print progress

    Returns:
        Dictionary mapping scale → coarse-grained array
    """
    log = print if verbose else lambda *a, **k: None

    method_desc = "majority vote → binary" if method == 'binary' else "density → continuous"
    log(f"\n   📏 Creating multi-scale representations...")
    log(f"      Scales: {scales}")
    log(f"      Method: {method_desc}")

    # Fast conversion to numpy
    t0 = time.perf_counter()
    seq_array = fast_string_to_array(sequence)
    log(f"      String→array: {time.perf_counter() - t0:.2f}s")

    representations = {}

    for scale in tqdm(scales, desc="Coarse-graining", disable=not HAS_TQDM or not verbose):
        coarse = coarse_grain(seq_array, scale, method=method)

        # Limit length if needed (take from beginning to preserve temporal order)
        if len(coarse) > max_samples:
            coarse = coarse[:max_samples]

        representations[scale] = coarse
        log(f"      Scale {scale:3d}: {len(coarse):,} samples")

    return representations


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMING TRANSFER ENTROPY (for large files)
# ═══════════════════════════════════════════════════════════════════════════════

class StreamingTEAccumulator:
    """
    Accumulates Transfer Entropy counts across chunks for memory-efficient
    processing of large files.

    Instead of loading entire sequences into RAM, we:
    1. Process chunks of bits
    2. Coarse-grain each chunk
    3. Accumulate counts in dictionaries (small: max 16k entries)
    4. Compute final TE from accumulated counts

    Also computes:
    - Mutual Information (MI) between scales
    - Scale Entropy for each scale
    - Auto-similarity correlation between scale pairs
    """

    def __init__(self, scales: List[int], k: int = 3, n_bins: int = 2):
        """
        Initialize accumulator for given scales.

        Args:
            scales: List of coarse-graining block sizes
            k: History length for TE
            n_bins: Number of bins for discretization (default 2 = binary)
        """
        self.scales = scales
        self.k = k
        self.n_bins = n_bins  # 2 for binary

        # Create counter dictionaries for each pair of scales (for TE)
        self.counters = {}
        for i in range(len(scales)):
            for j in range(len(scales)):
                if i != j:
                    self.counters[(i, j)] = {
                        'joint': {},      # (y_past, x_past, y_future) -> count
                        'y_past': {},     # y_past -> count
                        'yx_past': {},    # (y_past, x_past) -> count
                        'y_future_given_past': {},  # (y_past, y_future) -> count
                    }

        self.total_samples = {(i, j): 0 for i in range(len(scales))
                              for j in range(len(scales)) if i != j}
        self.chunks_processed = 0
        self.total_bits_processed = 0

        # === MUTUAL INFORMATION counters ===
        # For MI(X,Y) = H(X) + H(Y) - H(X,Y), we need counts for each scale
        # and joint counts for each pair
        self.mi_counters = {}
        for i in range(len(scales)):
            self.mi_counters[i] = {}  # value -> count (for H(X_i))
        self.mi_joint_counters = {}  # (i,j) -> {(x,y) -> count}
        for i in range(len(scales)):
            for j in range(i + 1, len(scales)):  # Only upper triangle (symmetric)
                self.mi_joint_counters[(i, j)] = {}
        self.mi_samples = {i: 0 for i in range(len(scales))}
        self.mi_joint_samples = {(i, j): 0 for i in range(len(scales))
                                  for j in range(i + 1, len(scales))}

        # === AUTO-SIMILARITY counters (Pearson correlation) ===
        # For r = (sum(xy) - n*mean_x*mean_y) / (n * std_x * std_y)
        # We accumulate: sum_x, sum_y, sum_xy, sum_x2, sum_y2, n
        self.corr_accumulators = {}
        for i in range(len(scales)):
            for j in range(i + 1, len(scales)):  # Only compare with larger scales
                self.corr_accumulators[(i, j)] = {
                    'sum_x': 0.0, 'sum_y': 0.0, 'sum_xy': 0.0,
                    'sum_x2': 0.0, 'sum_y2': 0.0, 'n': 0
                }

    def _discretize(self, data: np.ndarray) -> np.ndarray:
        """
        Discretize continuous [0,1] data to binary (0/1).

        Binary discretization is consistent across all scales and
        well-studied in TE literature. With k=3, gives 8 combinations (2³).
        """
        return (data >= 0.5).astype(np.int8)

    def process_chunk(self, chunk_bits, method: str = 'density'):
        """
        Process a chunk of bits and accumulate counts.

        Args:
            chunk_bits: Binary string OR numpy int8 array (only 0s and 1s)
            method: 'density' or 'binary'
        """
        # Handle both string and numpy array input
        if isinstance(chunk_bits, np.ndarray):
            seq_array = chunk_bits
        else:
            if len(chunk_bits) < max(self.scales) * (self.k + 1):
                return  # Chunk too small
            byte_array = np.frombuffer(chunk_bits.encode('ascii'), dtype=np.uint8)
            seq_array = (byte_array - ord('0')).astype(np.int8)

        if len(seq_array) < max(self.scales) * (self.k + 1):
            return  # Chunk too small

        # Coarse-grain at each scale
        coarse_grained = {}
        for idx, scale in enumerate(self.scales):
            n_blocks = len(seq_array) // scale
            if n_blocks < self.k + 1:
                continue

            truncated = seq_array[:n_blocks * scale]
            blocks = truncated.reshape(n_blocks, scale)

            if method == 'density':
                coarse = blocks.mean(axis=1).astype(np.float32)
                coarse = self._discretize(coarse)
            else:
                coarse = (blocks.sum(axis=1) > scale / 2).astype(np.int8)

            coarse_grained[idx] = coarse

        # Accumulate counts for each pair using VECTORIZED encoding
        # Key insight: with n_bins=4 and k=3, we can encode tuples as integers
        # y_past (3 values 0-3) -> 6 bits, x_past -> 6 bits, y_future -> 2 bits = 14 bits total
        # This allows using numpy bincount instead of Python dict iteration

        for (i, j), counters in self.counters.items():
            if i not in coarse_grained or j not in coarse_grained:
                continue

            source = coarse_grained[i].astype(np.int32)  # X
            target = coarse_grained[j].astype(np.int32)  # Y

            n = min(len(source), len(target))
            if n <= self.k:
                continue

            # Vectorized: create sliding windows using stride tricks
            # Encode y_past as integer: y[t-k]*16 + y[t-k+1]*4 + y[t-k+2]
            # For k=3, n_bins=4: max value = 63

            # Build encoded arrays
            y_past_encoded = np.zeros(n - self.k, dtype=np.int32)
            x_past_encoded = np.zeros(n - self.k, dtype=np.int32)

            for offset in range(self.k):
                multiplier = self.n_bins ** (self.k - 1 - offset)
                y_past_encoded += target[offset:n - self.k + offset] * multiplier
                x_past_encoded += source[offset:n - self.k + offset] * multiplier

            y_future = target[self.k:n]

            # Combined key: y_past * 256 + x_past * 4 + y_future
            # Max: 63 * 256 + 63 * 4 + 3 = 16128 + 252 + 3 = 16383 (fits in int16)
            n_y_past = self.n_bins ** self.k  # 64
            joint_keys = y_past_encoded * (n_y_past * self.n_bins) + x_past_encoded * self.n_bins + y_future

            # Use bincount for fast counting
            joint_counts = np.bincount(joint_keys, minlength=n_y_past * n_y_past * self.n_bins)
            y_past_counts = np.bincount(y_past_encoded, minlength=n_y_past)
            yx_past_keys = y_past_encoded * n_y_past + x_past_encoded
            yx_past_counts = np.bincount(yx_past_keys, minlength=n_y_past * n_y_past)
            yf_keys = y_past_encoded * self.n_bins + y_future
            yf_counts = np.bincount(yf_keys, minlength=n_y_past * self.n_bins)

            # Merge into existing counters (convert to dict format for compatibility)
            for key, count in enumerate(joint_counts):
                if count > 0:
                    counters['joint'][key] = counters['joint'].get(key, 0) + int(count)
            for key, count in enumerate(y_past_counts):
                if count > 0:
                    counters['y_past'][key] = counters['y_past'].get(key, 0) + int(count)
            for key, count in enumerate(yx_past_counts):
                if count > 0:
                    counters['yx_past'][key] = counters['yx_past'].get(key, 0) + int(count)
            for key, count in enumerate(yf_counts):
                if count > 0:
                    counters['y_future_given_past'][key] = counters['y_future_given_past'].get(key, 0) + int(count)

            self.total_samples[(i, j)] += n - self.k

        # === ACCUMULATE MI AND CORRELATION COUNTERS ===
        # For each scale, count binary values (for Scale Entropy and MI)
        for idx in coarse_grained:
            data = coarse_grained[idx]
            counts = np.bincount(data.astype(np.int32), minlength=self.n_bins)
            for val, cnt in enumerate(counts):
                if cnt > 0:
                    self.mi_counters[idx][val] = self.mi_counters[idx].get(val, 0) + int(cnt)
            self.mi_samples[idx] += len(data)

        # For each pair, count joint values (for MI) and accumulate correlation sums
        for i in range(len(self.scales)):
            for j in range(i + 1, len(self.scales)):
                if i not in coarse_grained or j not in coarse_grained:
                    continue

                x = coarse_grained[i].astype(np.float64)
                y = coarse_grained[j].astype(np.float64)
                n = min(len(x), len(y))
                if n < 2:
                    continue

                x, y = x[:n], y[:n]

                # Joint counts for MI (binary: 0,1 x 0,1 = 4 combos)
                joint_key = x.astype(np.int32) * self.n_bins + y.astype(np.int32)
                joint_counts = np.bincount(joint_key, minlength=self.n_bins * self.n_bins)
                for key, cnt in enumerate(joint_counts):
                    if cnt > 0:
                        self.mi_joint_counters[(i, j)][key] = \
                            self.mi_joint_counters[(i, j)].get(key, 0) + int(cnt)
                self.mi_joint_samples[(i, j)] += n

                # Correlation sums (for auto-similarity)
                acc = self.corr_accumulators[(i, j)]
                acc['sum_x'] += float(np.sum(x))
                acc['sum_y'] += float(np.sum(y))
                acc['sum_xy'] += float(np.sum(x * y))
                acc['sum_x2'] += float(np.sum(x * x))
                acc['sum_y2'] += float(np.sum(y * y))
                acc['n'] += n

        self.chunks_processed += 1
        self.total_bits_processed += len(seq_array)

    def compute_te_matrix(self) -> Tuple[np.ndarray, Dict]:
        """
        Compute final TE matrix from accumulated counts.

        Keys are now integers encoded as:
        - joint: y_past * (n_y_past * n_bins) + x_past * n_bins + y_future
        - y_past: just the y_past code (0 to n_bins^k - 1)
        - yx_past: y_past * n_y_past + x_past
        - y_future_given_past: y_past * n_bins + y_future

        Returns:
            (te_matrix, metadata)
        """
        n_scales = len(self.scales)
        te_matrix = np.zeros((n_scales, n_scales))
        n_y_past = self.n_bins ** self.k  # 64 for k=3, n_bins=4

        for (i, j), counters in self.counters.items():
            total = self.total_samples[(i, j)]
            if total == 0:
                continue

            te_value = 0.0
            joint = counters['joint']
            y_past_counts = counters['y_past']
            yx_past_counts = counters['yx_past']
            y_future_given_past = counters['y_future_given_past']

            # Iterate over joint counts (integer keys)
            for joint_key, count in joint.items():
                # Decode: joint_key = y_past * (n_y_past * n_bins) + x_past * n_bins + y_future
                y_future = joint_key % self.n_bins
                remainder = joint_key // self.n_bins
                x_past = remainder % n_y_past
                y_past = remainder // n_y_past

                p_joint = count / total

                # yx_past key = y_past * n_y_past + x_past
                yx_key = y_past * n_y_past + x_past
                if yx_key not in yx_past_counts or yx_past_counts[yx_key] == 0:
                    continue
                p_future_given_both = count / yx_past_counts[yx_key]

                # yf key = y_past * n_bins + y_future
                yf_key = y_past * self.n_bins + y_future
                if yf_key in y_future_given_past and y_past in y_past_counts:
                    if y_past_counts[y_past] > 0:
                        p_future_given_y = y_future_given_past[yf_key] / y_past_counts[y_past]
                        if p_future_given_y > 0:
                            ratio = p_future_given_both / p_future_given_y
                            te_value += p_joint * np.log2(ratio)

            te_matrix[i, j] = max(0.0, te_value)

        metadata = {
            'chunks_processed': self.chunks_processed,
            'total_bits': self.total_bits_processed,
            'samples_per_pair': {f"{i}->{j}": self.total_samples[(i,j)]
                                 for (i,j) in self.total_samples}
        }

        return te_matrix, metadata

    def compute_scale_entropy(self) -> Dict[int, float]:
        """
        Compute Shannon entropy H(X) for each scale.

        Returns:
            Dict mapping scale index to entropy in bits
        """
        entropies = {}
        for idx in range(len(self.scales)):
            total = self.mi_samples[idx]
            if total == 0:
                entropies[idx] = 0.0
                continue

            h = 0.0
            for val, count in self.mi_counters[idx].items():
                if count > 0:
                    p = count / total
                    h -= p * np.log2(p)
            entropies[idx] = h

        return entropies

    def compute_mi_matrix(self) -> np.ndarray:
        """
        Compute Mutual Information matrix between scales.

        MI(X,Y) = H(X) + H(Y) - H(X,Y)

        Returns:
            Symmetric MI matrix (n_scales x n_scales)
        """
        n_scales = len(self.scales)
        mi_matrix = np.zeros((n_scales, n_scales))

        # First compute individual entropies
        scale_h = self.compute_scale_entropy()

        # Diagonal: MI(X,X) = H(X)
        for i in range(n_scales):
            mi_matrix[i, i] = scale_h[i]

        # Off-diagonal: MI(X,Y) = H(X) + H(Y) - H(X,Y)
        for i in range(n_scales):
            for j in range(i + 1, n_scales):
                total = self.mi_joint_samples.get((i, j), 0)
                if total == 0:
                    continue

                # Compute H(X,Y)
                h_joint = 0.0
                for key, count in self.mi_joint_counters[(i, j)].items():
                    if count > 0:
                        p = count / total
                        h_joint -= p * np.log2(p)

                mi = scale_h[i] + scale_h[j] - h_joint
                mi = max(0.0, mi)  # Ensure non-negative
                mi_matrix[i, j] = mi
                mi_matrix[j, i] = mi  # Symmetric

        return mi_matrix

    def compute_auto_similarity(self) -> Dict[Tuple[int, int], float]:
        """
        Compute Pearson correlation between scale pairs.

        This measures auto-similarity: do patterns at scale S
        correlate with patterns at scale 2S, 4S, etc.?

        Returns:
            Dict mapping (i, j) to correlation coefficient r
        """
        correlations = {}

        for (i, j), acc in self.corr_accumulators.items():
            n = acc['n']
            if n < 2:
                correlations[(i, j)] = 0.0
                continue

            sum_x = acc['sum_x']
            sum_y = acc['sum_y']
            sum_xy = acc['sum_xy']
            sum_x2 = acc['sum_x2']
            sum_y2 = acc['sum_y2']

            # Pearson r = (n*sum_xy - sum_x*sum_y) / sqrt((n*sum_x2 - sum_x^2)(n*sum_y2 - sum_y^2))
            numerator = n * sum_xy - sum_x * sum_y
            denom_x = n * sum_x2 - sum_x ** 2
            denom_y = n * sum_y2 - sum_y ** 2

            if denom_x <= 0 or denom_y <= 0:
                correlations[(i, j)] = 0.0
            else:
                r = numerator / np.sqrt(denom_x * denom_y)
                correlations[(i, j)] = float(np.clip(r, -1.0, 1.0))

        return correlations

    def compute_all_metrics(self) -> Dict[str, Any]:
        """
        Compute all metrics: TE, MI, Scale Entropy, Auto-similarity.

        Returns:
            Dict with all computed metrics
        """
        te_matrix, te_meta = self.compute_te_matrix()
        mi_matrix = self.compute_mi_matrix()
        scale_entropy = self.compute_scale_entropy()
        auto_sim = self.compute_auto_similarity()

        return {
            'te_matrix': te_matrix,
            'te_metadata': te_meta,
            'mi_matrix': mi_matrix,
            'scale_entropy': scale_entropy,
            'auto_similarity': auto_sim,
            'scales': self.scales,
            'total_bits': self.total_bits_processed,
            'chunks_processed': self.chunks_processed
        }


def analyze_variant_te_streaming(variant: str, iteration: int,
                                  scales: List[int] = DEFAULT_SCALES,
                                  k: int = DEFAULT_HISTORY,
                                  method: str = 'density',
                                  chunk_size: int = 10_000_000,
                                  verbose: bool = True,
                                  with_shuffle_control: bool = False) -> Dict[str, Any]:
    """
    Analyze Transfer Entropy using streaming for large files.

    Processes the ENTIRE file without loading it all into RAM.

    Args:
        variant: Variant code (A, B, etc.)
        iteration: Iteration number
        scales: Block sizes for coarse-graining
        k: History length
        method: 'density' or 'binary'
        chunk_size: Bits per chunk (default 10M)
        verbose: Print progress
        with_shuffle_control: If True, also compute TE for shuffled chunks
                              to validate that observed TE is significant.
                              (Added 2026-02-06 per Antigra Phase 3 review)

    Returns:
        Results dictionary with optional 'shuffle_control' section
    """
    import sys
    from pathlib import Path

    log = print if verbose else lambda *a, **k: None
    t_start = time.perf_counter()
    timings = {}

    # Find the file
    results_dir = Path(__file__).parent / "results"
    base_path = results_dir / "level0" / "phi_snapshots" / f"var_{variant}"
    struct_path = base_path / f"phi_iter{iteration}.struct.gz"

    if not struct_path.exists():
        return {'error': f'File not found: {struct_path}'}

    file_size_mb = struct_path.stat().st_size / (1024 * 1024)

    log(f"\n{'═'*70}")
    log(f"🔀 HSI TRANSFER ENTROPY ANALYSIS (STREAMING MODE)")
    log(f"{'═'*70}")
    log(f"   Variant: {variant}")
    log(f"   Iteration: {iteration}")
    log(f"   File: {struct_path.name} ({file_size_mb:.1f} MB)")
    log(f"   Scales: {scales}")
    log(f"   History k: {k}")
    log(f"   Chunk size: {chunk_size:,} bits")
    log(f"   Method: {method}")
    log(f"{'─'*70}")
    sys.stdout.flush()

    # Initialize streaming loader
    try:
        from utils.streaming_phi_loader import StreamingPhiLoader
        loader = StreamingPhiLoader(str(struct_path))
    except ImportError as e:
        return {'error': f'StreamingPhiLoader not available: {e}'}

    # Initialize accumulator (n_bins=2 for binary discretization)
    accumulator = StreamingTEAccumulator(scales, k=k, n_bins=2)

    # Initialize shuffle control accumulator if requested
    shuffle_accumulator = None
    if with_shuffle_control:
        shuffle_accumulator = StreamingTEAccumulator(scales, k=k, n_bins=2)
        log(f"   🎲 Shuffle control ENABLED (will compute TE for randomized chunks)")

    # Process chunks using FAST numpy-based iterator
    log(f"\n   📂 Processing file in streaming mode (vectorized)...")
    sys.stdout.flush()
    t0 = time.perf_counter()

    chunk_count = 0
    # chunk_size is in bits, but iter_bits_numpy takes bytes
    # ~2 bits per byte on average (filtering out structure)
    chunk_bytes = chunk_size // 2

    # Random generator for shuffle (fixed seed for reproducibility)
    rng = np.random.default_rng(seed=42) if with_shuffle_control else None

    for bits_array in loader.iter_bits_numpy(chunk_bytes):
        chunk_count += 1
        accumulator.process_chunk(bits_array, method=method)

        # Process shuffled version for control
        if shuffle_accumulator is not None:
            shuffled_bits = bits_array.copy()
            rng.shuffle(shuffled_bits)  # In-place shuffle
            shuffle_accumulator.process_chunk(shuffled_bits, method=method)

        # Progress every 10 chunks
        if verbose and chunk_count % 10 == 0:
            bits_so_far = accumulator.total_bits_processed
            suffix = " (+shuffle)" if with_shuffle_control else ""
            print(f"\r      Chunks: {chunk_count}, Bits: {bits_so_far:,}{suffix}", end='', flush=True)

    if verbose:
        print()  # New line after progress

    timings['streaming'] = time.perf_counter() - t0
    log(f"      Processed {accumulator.total_bits_processed:,} bits in {chunk_count} chunks ({timings['streaming']:.2f}s)")
    sys.stdout.flush()

    # Compute ALL metrics (TE, MI, Scale Entropy, Auto-similarity)
    log(f"\n   🔀 Computing multi-scale metrics...")
    t0 = time.perf_counter()
    all_metrics = accumulator.compute_all_metrics()
    te_matrix = all_metrics['te_matrix']
    mi_matrix = all_metrics['mi_matrix']
    scale_entropy = all_metrics['scale_entropy']
    auto_sim = all_metrics['auto_similarity']
    timings['metrics_compute'] = time.perf_counter() - t0

    # Analyze TE results
    n_scales = len(scales)
    bottom_up_te = []
    top_down_te = []

    for i in range(n_scales):
        for j in range(n_scales):
            if i < j:  # fine → coarse
                bottom_up_te.append(te_matrix[i, j])
            elif i > j:  # coarse → fine
                top_down_te.append(te_matrix[i, j])

    avg_bottom_up = np.mean(bottom_up_te) if bottom_up_te else 0
    avg_top_down = np.mean(top_down_te) if top_down_te else 0
    asymmetry = avg_bottom_up - avg_top_down

    # TE Interpretation
    if abs(asymmetry) < 0.001:
        direction = "symmetric"
        interpretation = "↔️ SYMMETRIC: No clear directional information flow"
    elif asymmetry > 0:
        direction = "bottom_up"
        interpretation = "⬆️ BOTTOM-UP: Fine scales causally influence coarse scales"
    else:
        direction = "top_down"
        interpretation = "⬇️ TOP-DOWN: Coarse scales causally influence fine scales"

    # MI analysis (average off-diagonal)
    mi_offdiag = [mi_matrix[i, j] for i in range(n_scales)
                  for j in range(n_scales) if i != j]
    avg_mi = np.mean(mi_offdiag) if mi_offdiag else 0

    # Auto-similarity analysis
    # Calculate fractal index (average correlation across scale pairs)
    if auto_sim:
        correlations = list(auto_sim.values())
        fractal_index = np.mean(correlations) if correlations else 0
    else:
        fractal_index = 0

    # Scale entropy analysis (entropy decay rate)
    scale_h_values = [scale_entropy.get(i, 0) for i in range(n_scales)]
    if len(scale_h_values) >= 2 and scale_h_values[0] > 0:
        # Entropy ratio (last/first scale)
        entropy_ratio = scale_h_values[-1] / scale_h_values[0] if scale_h_values[0] > 0 else 0
        # Decay per octave (log2 of scale ratio)
        scale_ratio = np.log2(scales[-1] / scales[0]) if scales[0] > 0 else 1
        entropy_decay = (scale_h_values[0] - scale_h_values[-1]) / scale_ratio if scale_ratio > 0 else 0
    else:
        entropy_ratio = 0
        entropy_decay = 0

    # === SHUFFLE CONTROL ANALYSIS (if enabled) ===
    shuffle_control_results = None
    if shuffle_accumulator is not None:
        log(f"\n   🎲 Computing shuffle control metrics...")
        t0_shuffle = time.perf_counter()
        shuffle_metrics = shuffle_accumulator.compute_all_metrics()
        shuffle_te_matrix = shuffle_metrics['te_matrix']

        # Calculate shuffle TE averages
        shuffle_bottom_up = []
        shuffle_top_down = []
        for i in range(n_scales):
            for j in range(n_scales):
                if i < j:
                    shuffle_bottom_up.append(shuffle_te_matrix[i, j])
                elif i > j:
                    shuffle_top_down.append(shuffle_te_matrix[i, j])

        avg_shuffle_bottom_up = np.mean(shuffle_bottom_up) if shuffle_bottom_up else 0
        avg_shuffle_top_down = np.mean(shuffle_top_down) if shuffle_top_down else 0
        avg_shuffle_te = (avg_shuffle_bottom_up + avg_shuffle_top_down) / 2
        avg_observed_te = (avg_bottom_up + avg_top_down) / 2

        # Calculate significance ratio
        if avg_shuffle_te > 1e-10:
            te_ratio = avg_observed_te / avg_shuffle_te
        else:
            te_ratio = float('inf') if avg_observed_te > 1e-10 else 1.0

        # Interpretation
        if te_ratio > 2.0:
            shuffle_interpretation = "✅ SIGNIFICANT: Observed TE is >2x shuffle baseline"
        elif te_ratio > 1.5:
            shuffle_interpretation = "⚠️ MARGINAL: Observed TE is 1.5-2x shuffle baseline"
        elif te_ratio > 1.1:
            shuffle_interpretation = "❓ WEAK: Observed TE is only slightly above shuffle"
        else:
            shuffle_interpretation = "❌ NOT SIGNIFICANT: Observed TE ≈ shuffle (no real structure)"

        timings['shuffle_compute'] = time.perf_counter() - t0_shuffle

        shuffle_control_results = {
            'shuffle_te_matrix': shuffle_te_matrix.tolist(),
            'avg_shuffle_bottom_up': float(avg_shuffle_bottom_up),
            'avg_shuffle_top_down': float(avg_shuffle_top_down),
            'avg_shuffle_te': float(avg_shuffle_te),
            'avg_observed_te': float(avg_observed_te),
            'te_ratio_observed_vs_shuffle': float(te_ratio),
            'interpretation': shuffle_interpretation,
            'method': 'per_chunk_shuffle',
            'seed': 42
        }

        log(f"      Observed TE: {avg_observed_te:.6f}")
        log(f"      Shuffle TE:  {avg_shuffle_te:.6f}")
        log(f"      Ratio:       {te_ratio:.2f}x")
        log(f"      {shuffle_interpretation}")

    te_results = {
        'te_matrix': te_matrix.tolist(),
        'scales': scales,
        'avg_bottom_up': float(avg_bottom_up),
        'avg_top_down': float(avg_top_down),
        'asymmetry': float(asymmetry),
        'direction': direction,
        'interpretation': interpretation,
        'streaming_metadata': all_metrics['te_metadata']
    }

    mi_results = {
        'mi_matrix': mi_matrix.tolist(),
        'avg_mi': float(avg_mi),
        'scale_entropy': {str(scales[i]): float(v) for i, v in scale_entropy.items()},
        'entropy_ratio': float(entropy_ratio),
        'entropy_decay_per_octave': float(entropy_decay)
    }

    similarity_results = {
        'auto_similarity': {f"{scales[i]}->{scales[j]}": float(v)
                           for (i, j), v in auto_sim.items()},
        'fractal_index': float(fractal_index)
    }

    # Print comprehensive summary
    log(f"\n   {'═'*60}")
    log(f"   📊 MULTI-SCALE ANALYSIS RESULTS")
    log(f"   {'═'*60}")

    log(f"\n   🔀 TRANSFER ENTROPY (directional information flow):")
    log(f"      TE (fine → coarse): {avg_bottom_up:.8f} bits")
    log(f"      TE (coarse → fine): {avg_top_down:.8f} bits")
    log(f"      Asymmetry: {asymmetry:+.8f} bits")
    log(f"      {interpretation}")

    log(f"\n   🔗 MUTUAL INFORMATION (total correlation):")
    log(f"      Avg MI between scales: {avg_mi:.6f} bits")

    log(f"\n   📈 SCALE ENTROPY (structure at each scale):")
    for i, s in enumerate(scales):
        h = scale_entropy.get(i, 0)
        log(f"      H(scale={s:>4}): {h:.6f} bits")
    log(f"      Entropy ratio (last/first): {entropy_ratio:.4f}")
    log(f"      Entropy decay: {entropy_decay:.6f} bits/octave")

    log(f"\n   🔄 AUTO-SIMILARITY (pattern correlation across scales):")
    for (i, j), r in sorted(auto_sim.items()):
        log(f"      {scales[i]:>4} → {scales[j]:>4}: r = {r:+.4f}")
    log(f"      Fractal Index (avg correlation): {fractal_index:.4f}")
    log(f"   {'═'*60}")

    total_time = time.perf_counter() - t_start
    timings['total'] = total_time

    # Compile comprehensive results
    results = {
        'variant': variant,
        'iteration': iteration,
        'timestamp': datetime.now().isoformat(),
        'total_bits_processed': accumulator.total_bits_processed,
        'te_analysis': te_results,
        'mi_analysis': mi_results,
        'similarity_analysis': similarity_results,
        'total_time_s': total_time,
        'timings': timings,
        'config': {
            'scales': scales,
            'k': k,
            'chunk_size': chunk_size,
            'method': method,
            'mode': 'streaming',
            'with_shuffle_control': with_shuffle_control
        }
    }

    # Add shuffle control results if computed
    if shuffle_control_results is not None:
        results['shuffle_control'] = shuffle_control_results

    # Save results
    TE_DIR.mkdir(parents=True, exist_ok=True)
    method_suffix = "_density" if method == 'density' else ""
    output_path = TE_DIR / f"multiscale_{variant}_iter{iteration}{method_suffix}.json"

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)

    log(f"\n   ⏱️ TIMING BREAKDOWN:")
    log(f"      Streaming:       {timings.get('streaming', 0):.2f}s")
    log(f"      Metrics compute: {timings.get('metrics_compute', 0):.2f}s")
    log(f"      ─────────────────────")
    log(f"      TOTAL:           {total_time:.2f}s")
    log(f"\n   💾 Results saved to: {output_path}")
    sys.stdout.flush()

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSFER ENTROPY COMPUTATION (Numba-optimized)
# ═══════════════════════════════════════════════════════════════════════════════

# Numba-accelerated core function
if HAS_NUMBA:
    @njit(cache=True)
    def _te_core_numba(source: np.ndarray, target: np.ndarray, k: int) -> float:
        """
        Numba-optimized Transfer Entropy computation.
        Uses integer encoding of states for fast hashing.
        """
        n = min(len(source), len(target))
        if n <= k + 1:
            return 0.0

        # For binary sequences with k history, max states = 2^k
        # We encode states as integers: state = sum(bit[i] * 2^i)
        n_states = 1 << k  # 2^k
        n_joint = n_states * n_states * 2  # y_past * x_past * y_future

        # Count arrays (flat indexing)
        joint_counts = np.zeros(n_joint, dtype=np.int32)
        y_past_counts = np.zeros(n_states, dtype=np.int32)
        yx_past_counts = np.zeros(n_states * n_states, dtype=np.int32)
        y_future_given_past = np.zeros(n_states * 2, dtype=np.int32)

        # Build counts
        for t in range(k, n):
            # Encode y_past as integer
            y_past_idx = 0
            for i in range(k):
                y_past_idx += target[t - k + i] << i

            # Encode x_past as integer
            x_past_idx = 0
            for i in range(k):
                x_past_idx += source[t - k + i] << i

            y_future = target[t]

            # Update counts
            joint_idx = y_past_idx * n_states * 2 + x_past_idx * 2 + y_future
            joint_counts[joint_idx] += 1
            y_past_counts[y_past_idx] += 1
            yx_past_counts[y_past_idx * n_states + x_past_idx] += 1
            y_future_given_past[y_past_idx * 2 + y_future] += 1

        total = n - k

        # Compute TE
        te_value = 0.0
        log2_inv = 1.0 / np.log(2.0)

        for y_past_idx in range(n_states):
            if y_past_counts[y_past_idx] == 0:
                continue
            for x_past_idx in range(n_states):
                yx_idx = y_past_idx * n_states + x_past_idx
                if yx_past_counts[yx_idx] == 0:
                    continue
                for y_future in range(2):
                    joint_idx = y_past_idx * n_states * 2 + x_past_idx * 2 + y_future
                    count = joint_counts[joint_idx]
                    if count == 0:
                        continue

                    p_joint = count / total
                    p_future_given_both = count / yx_past_counts[yx_idx]

                    yf_idx = y_past_idx * 2 + y_future
                    p_future_given_y = y_future_given_past[yf_idx] / y_past_counts[y_past_idx]

                    if p_future_given_y > 0:
                        te_value += p_joint * np.log(p_future_given_both / p_future_given_y) * log2_inv

        return max(0.0, te_value)


def compute_transfer_entropy_simple(source: np.ndarray,
                                     target: np.ndarray,
                                     k: int = 3,
                                     max_te_samples: int = 500000) -> float:
    """
    Compute Transfer Entropy from source → target.
    Uses Numba if available (for binary data), otherwise falls back to Python.

    Args:
        max_te_samples: Max samples for TE calculation (Python fallback is O(n) with high memory)
    """
    n = min(len(source), len(target))
    if n <= k + 1:
        return 0.0

    # Check if data is binary (0/1 only) - Numba version only works for binary
    is_binary = (np.unique(source).size <= 2) and (np.unique(target).size <= 2)

    # Use Numba version if available AND data is binary
    if HAS_NUMBA and is_binary:
        return _te_core_numba(source[:n].astype(np.int32),
                              target[:n].astype(np.int32), k)

    # Fallback: Python implementation with Counter
    # Limit samples to avoid memory issues with large continuous data
    if n > max_te_samples:
        # Sample from middle for consistency
        start = (n - max_te_samples) // 2
        source = source[start:start + max_te_samples]
        target = target[start:start + max_te_samples]
        n = max_te_samples

    from collections import Counter

    source = source[:n]
    target = target[:n]

    joint_counts = Counter()
    y_past_counts = Counter()
    yx_past_counts = Counter()
    y_future_given_past = Counter()

    for t in range(k, n):
        y_past = tuple(target[t-k:t])
        x_past = tuple(source[t-k:t])
        y_future = target[t]

        joint_counts[(y_past, x_past, y_future)] += 1
        y_past_counts[y_past] += 1
        yx_past_counts[(y_past, x_past)] += 1
        y_future_given_past[(y_past, y_future)] += 1

    total = n - k
    te_value = 0.0

    for (y_past, x_past, y_future), count in joint_counts.items():
        p_joint = count / total
        p_future_given_both = count / yx_past_counts[(y_past, x_past)]
        p_future_given_y = y_future_given_past[(y_past, y_future)] / y_past_counts[y_past]

        if p_future_given_y > 0:
            te_value += p_joint * np.log2(p_future_given_both / p_future_given_y)

    return max(0.0, te_value)


def discretize_continuous(data: np.ndarray, n_bins: int = 4) -> np.ndarray:
    """
    Discretize continuous data into bins for TE computation.

    Args:
        data: Continuous values (e.g., density in [0, 1])
        n_bins: Number of bins (default 4: low, mid-low, mid-high, high)

    Returns:
        Discretized integer array
    """
    # Use fixed bins for [0, 1] range (density values)
    bins = np.linspace(0, 1, n_bins + 1)
    # np.digitize returns 1-indexed, subtract 1 for 0-indexed
    discretized = np.digitize(data, bins[1:-1]).astype(np.int32)
    return discretized


def compute_transfer_entropy(source: np.ndarray,
                              target: np.ndarray,
                              k: int = 3,
                              n_bins: int = 4,
                              verbose: bool = False) -> Dict[str, float]:
    """
    Compute Transfer Entropy using best available method.
    Supports both binary and continuous (density) data.

    Args:
        source: Source time series (binary or continuous)
        target: Target time series (binary or continuous)
        k: History length
        n_bins: Number of bins for discretizing continuous data
        verbose: Print details

    Returns:
        Dictionary with TE value and metadata
    """
    log = print if verbose else lambda *a, **k: None

    # Check if data is continuous (float) or binary (int)
    is_continuous = source.dtype in [np.float32, np.float64]

    result = {
        'method': 'simple_histogram',
        'k': k,
        'source_length': len(source),
        'target_length': len(target),
        'is_continuous': is_continuous,
        'te_value': 0.0
    }

    # Discretize if continuous
    if is_continuous:
        source_disc = discretize_continuous(source, n_bins)
        target_disc = discretize_continuous(target, n_bins)
        result['n_bins'] = n_bins
    else:
        source_disc = source.astype(np.int32)
        target_disc = target.astype(np.int32)

    if HAS_PYINFORM and not is_continuous:
        # pyinform only works well with binary data
        log(f"      Using: pyinform")
        result['method'] = 'pyinform'
        try:
            te_value = te.transfer_entropy(source_disc.tolist(), target_disc.tolist(), k)
            result['te_value'] = float(te_value)
        except Exception as e:
            log(f"      ⚠️ pyinform error: {e}, falling back to simple method")
            result['te_value'] = compute_transfer_entropy_simple(source_disc, target_disc, k)
    else:
        method_name = f"histogram ({n_bins} bins)" if is_continuous else "histogram (binary)"
        log(f"      Using: {method_name}")
        result['te_value'] = compute_transfer_entropy_simple(source_disc, target_disc, k)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-SCALE ANALYSIS (Parallelized)
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_te_pair(args: Tuple) -> Tuple[int, int, float]:
    """Helper for parallel TE computation."""
    import sys

    i, j, source, target, k = args

    try:
        te_result = compute_transfer_entropy(source, target, k, verbose=False)
        te_val = te_result['te_value']
        # Flush stdout to prevent buffering deadlock on Windows
        sys.stdout.flush()
        return (i, j, te_val)
    except Exception as e:
        print(f"\n   ⚠️ Error computing TE pair ({i},{j}): {type(e).__name__}: {e}", flush=True)
        return (i, j, 0.0)


def analyze_multiscale_te(representations: Dict[int, np.ndarray],
                          k: int = 3,
                          verbose: bool = True,
                          n_workers: int = None) -> Dict[str, Any]:
    """
    Compute Transfer Entropy between all pairs of scales.
    Uses parallel processing for speedup.

    Key questions:
    - TE(fine → coarse) vs TE(coarse → fine): Direction of information flow
    - If TE(micro → macro) > TE(macro → micro): Bottom-up emergence

    Args:
        representations: Scale → array mapping
        k: History length for TE
        verbose: Print progress
        n_workers: Number of parallel workers (default: CPU count)

    Returns:
        Dictionary with TE matrix and interpretations
    """
    log = print if verbose else lambda *a, **k: None

    scales = sorted(representations.keys())
    n_scales = len(scales)

    if n_workers is None:
        n_workers = min(os.cpu_count() or 4, 8)  # Cap at 8 workers

    log(f"\n   🔀 Computing Transfer Entropy between scales...")
    log(f"      History length k={k}")
    log(f"      Workers: {n_workers}")

    # Create TE matrix
    te_matrix = np.zeros((n_scales, n_scales))

    # Prepare pairs with aligned data
    pairs_data = []
    for i in range(n_scales):
        for j in range(n_scales):
            if i != j:
                scale_i = scales[i]
                scale_j = scales[j]
                source = representations[scale_i]
                target = representations[scale_j]
                min_len = min(len(source), len(target))
                pairs_data.append((i, j, source[:min_len], target[:min_len], k))

    n_pairs = len(pairs_data)
    log(f"      Pairs to compute: {n_pairs}")

    t0 = time.perf_counter()
    import sys

    # Parallel computation
    if n_workers > 1 and n_pairs > 4:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_compute_te_pair, args): args[:2] for args in pairs_data}

            completed = 0
            errors = 0
            for future in as_completed(futures):
                i, j = futures[future]
                try:
                    _, _, te_val = future.result(timeout=300)  # 5 min timeout per pair
                    te_matrix[i, j] = te_val
                except Exception as e:
                    errors += 1
                    print(f"\n   ⚠️ Error on pair ({i},{j}): {type(e).__name__}: {e}", flush=True)
                completed += 1
                if verbose and completed % 5 == 0:
                    print(f"\r      Progress: {completed}/{n_pairs} pairs", end='', flush=True)
                # Flush to prevent buffering deadlock on Windows
                sys.stdout.flush()

            if errors > 0:
                print(f"\n   ⚠️ {errors} pairs failed")

        if verbose:
            print()  # New line after progress
    else:
        # Sequential fallback
        for args in tqdm(pairs_data, desc="Computing TE", disable=not HAS_TQDM or not verbose):
            i, j, te_val = _compute_te_pair(args)
            te_matrix[i, j] = te_val

    log(f"      TE computation: {time.perf_counter() - t0:.2f}s")

    # Analyze results
    log(f"\n   📊 TRANSFER ENTROPY MATRIX (bits):")
    log(f"      {'':>8}", end='')
    for s in scales:
        log(f" →{s:>5}", end='')
    log()

    for i, s_from in enumerate(scales):
        log(f"      {s_from:>5} →", end='')
        for j in range(n_scales):
            if i == j:
                log(f"   ---", end='')
            else:
                log(f" {te_matrix[i,j]:>5.3f}", end='')
        log()

    # Compute directional asymmetries
    bottom_up = 0.0  # TE from fine to coarse
    top_down = 0.0   # TE from coarse to fine
    n_pairs = 0

    for i in range(n_scales):
        for j in range(i + 1, n_scales):
            # i is finer scale, j is coarser
            bottom_up += te_matrix[i, j]  # fine → coarse
            top_down += te_matrix[j, i]   # coarse → fine
            n_pairs += 1

    if n_pairs > 0:
        avg_bottom_up = bottom_up / n_pairs
        avg_top_down = top_down / n_pairs
    else:
        avg_bottom_up = avg_top_down = 0.0

    # Interpretation
    asymmetry = avg_bottom_up - avg_top_down

    log(f"\n   🔮 DIRECTIONAL ANALYSIS:")
    log(f"      Avg TE (fine → coarse): {avg_bottom_up:.4f} bits")
    log(f"      Avg TE (coarse → fine): {avg_top_down:.4f} bits")
    log(f"      Asymmetry: {asymmetry:+.4f} bits")

    if asymmetry > 0.01:
        interpretation = "🔼 BOTTOM-UP EMERGENCE: Fine scales causally influence coarse scales"
        log(f"\n      {interpretation}")
        log(f"         This suggests genuine emergent behavior (micro → macro causation)")
    elif asymmetry < -0.01:
        interpretation = "🔽 TOP-DOWN CAUSATION: Coarse scales causally influence fine scales"
        log(f"\n      {interpretation}")
        log(f"         This suggests constraint-driven dynamics (macro → micro)")
    else:
        interpretation = "↔️ SYMMETRIC: No clear directional information flow"
        log(f"\n      {interpretation}")
        log(f"         Information flows equally in both directions")

    results = {
        'scales': scales,
        'te_matrix': te_matrix.tolist(),
        'avg_bottom_up': avg_bottom_up,
        'avg_top_down': avg_top_down,
        'asymmetry': asymmetry,
        'interpretation': interpretation,
        'k': k
    }

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_variant_te(variant: str,
                       iteration: int,
                       scales: List[int] = None,
                       k: int = DEFAULT_HISTORY,
                       max_samples: int = DEFAULT_MAX_SAMPLES,
                       method: str = 'binary',
                       verbose: bool = True,
                       n_workers: int = None) -> Dict[str, Any]:
    """
    Perform complete multi-scale Transfer Entropy analysis on a variant.

    Args:
        variant: Variant letter
        iteration: Iteration number
        scales: List of coarse-graining scales
        k: History length for TE
        max_samples: Maximum samples per scale
        method: 'binary' (majority vote) or 'density' (proportion of 1s)
        verbose: Print progress
        n_workers: Number of parallel workers

    Returns:
        Complete analysis results
    """
    if scales is None:
        scales = DEFAULT_SCALES.copy()

    n_cpus = os.cpu_count() or 4
    method_desc = "majority vote → binary" if method == 'binary' else "density → continuous (4 bins)"

    print(f"\n{'═'*70}")
    print(f"🔀 HSI TRANSFER ENTROPY ANALYSIS")
    print(f"{'═'*70}")
    print(f"   Variant: {variant}")
    print(f"   Iteration: {iteration}")
    print(f"   Scales: {scales}")
    print(f"   History k: {k}")
    print(f"   Max samples: {max_samples:,}")
    print(f"   Method: {method_desc}")
    print(f"   Library: {'pyinform' if HAS_PYINFORM else ('numba-optimized' if HAS_NUMBA else 'pure Python')}")
    print(f"   CPUs available: {n_cpus}")
    print(f"   Numba JIT: {'✅ enabled' if HAS_NUMBA else '❌ disabled'}")
    print(f"{'─'*70}")

    t_start = time.perf_counter()
    timings = {}

    # Load sequence
    print(f"\n   📂 Loading Φ sequence...")
    t0 = time.perf_counter()
    sequence = load_phi_sequence(variant, iteration)
    timings['load'] = time.perf_counter() - t0

    if not sequence:
        print(f"   ❌ Failed to load sequence")
        return {'error': 'failed to load sequence'}

    print(f"      Loaded {len(sequence):,} bits ({timings['load']:.2f}s)")

    # Create multi-scale representations
    t0 = time.perf_counter()
    representations = create_multiscale_representation(
        sequence, scales, max_samples, method=method, verbose=verbose
    )
    timings['coarse_grain'] = time.perf_counter() - t0
    print(f"      Coarse-graining: {timings['coarse_grain']:.2f}s")

    # Compute TE between scales
    t0 = time.perf_counter()
    te_results = analyze_multiscale_te(representations, k=k, verbose=verbose, n_workers=n_workers)
    timings['te_compute'] = time.perf_counter() - t0

    total_time = time.perf_counter() - t_start
    timings['total'] = total_time

    # Compile results
    results = {
        'variant': variant,
        'iteration': iteration,
        'timestamp': datetime.now().isoformat(),
        'sequence_length': len(sequence),
        'te_analysis': te_results,
        'total_time_s': total_time,
        'timings': timings,
        'config': {
            'scales': scales,
            'k': k,
            'max_samples': max_samples,
            'method': method,
            'library': 'pyinform' if HAS_PYINFORM else ('numba' if HAS_NUMBA else 'python'),
            'numba_enabled': HAS_NUMBA
        }
    }

    # Save results (include method and samples in filename)
    TE_DIR.mkdir(parents=True, exist_ok=True)
    method_suffix = "_density" if method == 'density' else ""
    samples_suffix = f"_{max_samples//1000}k" if max_samples < 1000000 else f"_{max_samples//1000000}M"
    output_path = TE_DIR / f"te_{variant}_iter{iteration}{method_suffix}{samples_suffix}.json"

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)

    # Summary timing
    print(f"\n   ⏱️ TIMING BREAKDOWN:")
    print(f"      Load:          {timings.get('load', 0):.2f}s")
    print(f"      Coarse-grain:  {timings.get('coarse_grain', 0):.2f}s")
    print(f"      TE compute:    {timings.get('te_compute', 0):.2f}s")
    print(f"      ─────────────────────")
    print(f"      TOTAL:         {total_time:.2f}s")

    print(f"\n   💾 Results saved to: {output_path}")
    print(f"{'═'*70}\n")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_te_heatmap(results: Dict[str, Any], output_dir: Path) -> Optional[Path]:
    """Generate heatmap of the TE matrix."""
    import sys

    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend (thread-safe)
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    try:
        te_analysis = results.get('te_analysis', {})
        te_matrix = np.array(te_analysis.get('te_matrix', []))
        scales = te_analysis.get('scales', [])

        if te_matrix.size == 0:
            return None

        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(te_matrix, cmap='YlOrRd', aspect='auto')

        # Labels
        ax.set_xticks(range(len(scales)))
        ax.set_yticks(range(len(scales)))
        ax.set_xticklabels([f'→{s}' for s in scales])
        ax.set_yticklabels([f'{s}→' for s in scales])

        ax.set_xlabel('Target Scale', fontsize=12)
        ax.set_ylabel('Source Scale', fontsize=12)
        ax.set_title(f"Transfer Entropy Matrix - Variant {results.get('variant', '?')}\n"
                     f"(Iteration {results.get('iteration', '?')})", fontsize=14)

        # Add values
        for i in range(len(scales)):
            for j in range(len(scales)):
                if i != j:
                    text = ax.text(j, i, f'{te_matrix[i, j]:.3f}',
                                  ha='center', va='center', fontsize=9)

        plt.colorbar(im, label='Transfer Entropy (bits)')

        # Add asymmetry annotation
        asymmetry = te_analysis.get('asymmetry', 0)
        interpretation = te_analysis.get('interpretation', '')
        ax.text(0.02, -0.12, f"Asymmetry: {asymmetry:+.4f} bits\n{interpretation}",
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        output_dir.mkdir(parents=True, exist_ok=True)
        config = results.get('config', {})
        method_suffix = "_density" if config.get('method') == 'density' else ""
        max_samples = config.get('max_samples', 50000)
        samples_suffix = f"_{max_samples//1000}k" if max_samples < 1000000 else f"_{max_samples//1000000}M"
        plot_path = output_dir / f"te_heatmap_{results.get('variant', 'X')}_iter{results.get('iteration', 0)}{method_suffix}{samples_suffix}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        sys.stdout.flush()  # Prevent buffering issues
        return plot_path

    except Exception as e:
        print(f"   ⚠️ Error generating heatmap: {e}", flush=True)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND LINE INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="🔀 HSI Level 2 - Transfer Entropy Multi-Scale Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python level2_transfer_entropy.py --variant B --iteration 18
  python level2_transfer_entropy.py --variant B --iteration 18 --scales 1 2 4 8 16 32
  python level2_transfer_entropy.py --variants B D E --iteration 18 --compare
  python level2_transfer_entropy.py --variant B --iteration 18 --plot
        """
    )

    parser.add_argument('--variant', '-v', type=str,
                       help='Single variant to analyze (A-I)')
    parser.add_argument('--variants', type=str, nargs='+',
                       help='Multiple variants to compare')
    parser.add_argument('--iteration', '-i', type=int, required=True,
                       help='Iteration number')
    parser.add_argument('--scales', type=int, nargs='+', default=DEFAULT_SCALES,
                       help=f'Coarse-graining scales (default: {DEFAULT_SCALES})')
    parser.add_argument('--k', type=int, default=DEFAULT_HISTORY,
                       help=f'History length for TE (default: {DEFAULT_HISTORY})')
    parser.add_argument('--max-samples', type=int, default=DEFAULT_MAX_SAMPLES,
                       help=f'Max samples per scale (default: {DEFAULT_MAX_SAMPLES:,})')
    parser.add_argument('--method', '-m', type=str, default='binary',
                       choices=['binary', 'density'],
                       help='Coarse-graining method: binary (majority vote) or density (proportion)')
    parser.add_argument('--workers', '-w', type=int, default=None,
                       help='Number of parallel workers (default: auto)')
    parser.add_argument('--plot', action='store_true',
                       help='Generate TE heatmap')
    parser.add_argument('--compare', action='store_true',
                       help='Compare multiple variants')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Minimal output')
    parser.add_argument('--streaming', '-S', action='store_true',
                       help='Use streaming mode for large files (processes entire file)')
    parser.add_argument('--chunk-size', type=int, default=10_000_000,
                       help='Chunk size in bits for streaming mode (default: 10M)')
    parser.add_argument('--shuffle-control', action='store_true',
                       help='Compute TE for shuffled chunks as baseline control '
                            '(validates that observed TE is significant)')

    args = parser.parse_args()

    variants = args.variants or ([args.variant] if args.variant else ['B'])
    verbose = not args.quiet

    all_results = {}
    import sys

    for var in variants:
        if args.streaming:
            # Use streaming mode for large files
            result = analyze_variant_te_streaming(
                var, args.iteration,
                scales=args.scales,
                k=args.k,
                method=args.method,
                chunk_size=args.chunk_size,
                verbose=verbose,
                with_shuffle_control=args.shuffle_control
            )
        else:
            # Original mode (loads sample into RAM)
            result = analyze_variant_te(
                var, args.iteration,
                scales=args.scales,
                k=args.k,
                max_samples=args.max_samples,
                method=args.method,
                verbose=verbose,
                n_workers=args.workers
            )
        all_results[var] = result

        if args.plot:
            plot_path = generate_te_heatmap(result, TE_DIR)
            if plot_path:
                print(f"   📊 Heatmap saved to: {plot_path}")

        # Flush after each variant to prevent buffering issues
        sys.stdout.flush()

    # Summary comparison
    if len(variants) > 1:
        print(f"\n📊 COMPARISON SUMMARY")
        print(f"{'═'*70}")
        print(f"   {'Variant':<10} {'Bottom-Up':<12} {'Top-Down':<12} {'Asymmetry':<12} {'Direction':<20}")
        print(f"   {'─'*66}")

        for var, res in all_results.items():
            te = res.get('te_analysis', {})
            bu = te.get('avg_bottom_up', 0)
            td = te.get('avg_top_down', 0)
            asym = te.get('asymmetry', 0)

            if asym > 0.01:
                direction = '🔼 Bottom-up'
            elif asym < -0.01:
                direction = '🔽 Top-down'
            else:
                direction = '↔️ Symmetric'

            print(f"   {var:<10} {bu:<12.4f} {td:<12.4f} {asym:<+12.4f} {direction:<20}")

        print(f"{'═'*70}\n")


if __name__ == "__main__":
    main()
