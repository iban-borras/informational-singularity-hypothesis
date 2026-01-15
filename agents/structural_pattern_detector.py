"""
🔍 Structural Pattern Detector — HSI v33.1 (Optimized)

This module detects patterns in the STRUCTURAL information of Φ sequences.
Unlike the standard pattern detector (which works on observable bits),
this detector analyzes the parentheses structure to find emergent order.

Key patterns detected:
1. Absolute Nesting Patterns: How Absolutes nest within each other
2. Absolute Containment Patterns: What each Absolute contains
3. Stratified Order Patterns: Entropy and information at each depth level

Optimizations v33.1:
- Progress logging every 5%
- list.append() + join() instead of string concatenation
- Incremental statistics (Welford's algorithm) to avoid storing all values
- Configurable memory limits
- Sampling mode for very large sequences (optional)

Author: Iban Borràs with Augment Agent collaboration
Date: January 2025 / Optimized December 2025
"""

import numpy as np
import json
import time
import sys
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from collections import defaultdict, Counter
import hashlib
from scipy.stats import entropy

# Numba-optimized kernels
try:
    from .numba_kernels import (
        str_to_bytes,
        _analyze_nesting_numba,
        _analyze_nesting_numba_chunk,
        _analyze_containment_numba,
        _analyze_containment_numba_chunk,
        _analyze_stratified_numba,
        _analyze_stratified_numba_chunk
    )
    NUMBA_STRUCTURAL_AVAILABLE = True
except ImportError:
    NUMBA_STRUCTURAL_AVAILABLE = False

# Progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class WelfordAccumulator:
    """
    Welford's online algorithm for computing mean and variance.
    Memory-efficient: O(1) instead of O(n).
    """

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0  # Sum of squared differences

    def update(self, value: float):
        """Add a new value to the accumulator."""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.M2 += delta * delta2

    @property
    def variance(self) -> float:
        """Return sample variance."""
        if self.count < 2:
            return 0.0
        return self.M2 / (self.count - 1)

    @property
    def std(self) -> float:
        """Return sample standard deviation."""
        return np.sqrt(self.variance)


class StructuralPatternDetector:
    """
    Detector for structural patterns in Φ sequences with parentheses.

    Analyzes the stratified containment structure encoded in parentheses
    to detect emergent order patterns.

    v33.1: Optimized for large sequences (100M+ characters).
    """

    # Default thresholds
    DEFAULT_MAX_MEMORY_GB = 30.0
    DEFAULT_SAMPLE_SIZE = 5_000_000  # 5M chars per sample
    DEFAULT_MAX_ABSOLUTES = 10_000_000  # Max absolutes to process fully

    def __init__(self,
                 min_depth: int = 1,
                 max_depth: int = 50,
                 min_occurrences: int = 2,
                 max_memory_gb: float = 30.0,
                 enable_sampling: bool = False,
                 verbose: bool = True):
        """
        Initialize the structural pattern detector.

        Args:
            min_depth: Minimum nesting depth to analyze
            max_depth: Maximum nesting depth to analyze
            min_occurrences: Minimum occurrences for a pattern
            max_memory_gb: Maximum RAM to use (default 30GB)
            enable_sampling: If True, use sampling for very large sequences
            verbose: If True, print progress logs
        """
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.min_occurrences = min_occurrences
        self.max_memory_gb = max_memory_gb
        self.enable_sampling = enable_sampling
        self.verbose = verbose

        self.detected_patterns = []
        self._start_time = None
    
    def _log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)
            sys.stdout.flush()

    def _format_time(self, seconds: float) -> str:
        """Format seconds as human-readable time."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"

    def _estimate_memory_mb(self, phi_structural: str) -> float:
        """Estimate memory usage for processing this sequence."""
        n = len(phi_structural)
        # Realistic estimate based on actual processing:
        # - Original string: 1 byte per char
        # - Numba processing: converts to bytes array (1 byte per char)
        # - Depth tracking: O(max_depth) = negligible
        # - Result dictionaries: O(1000s of entries) = negligible
        # Total: ~3 bytes per char (with some overhead)
        return n * 3 / (1024 * 1024)

    def detect_structural_patterns(self, phi_structural: str) -> List[Dict[str, Any]]:
        """
        Detect structural patterns in Φ with parentheses.

        Args:
            phi_structural: Φ string with structure (e.g., "((01)1)")

        Returns:
            List of detected structural patterns with metrics
        """
        self._start_time = time.time()
        original_n = len(phi_structural)
        n = original_n

        self._log(f"🔍 Detecting structural patterns in Φ (length: {n:,})")

        # Estimate memory (conservative: ~3 bytes per char)
        est_mem = self._estimate_memory_mb(phi_structural)
        limit_mb = self.max_memory_gb * 1024
        self._log(f"   📊 Estimated memory: {est_mem:.0f}MB (limit: {limit_mb:.0f}MB)")

        # Check current system RAM if psutil available
        try:
            import psutil
            ram = psutil.virtual_memory()
            available_mb = ram.available / (1024 * 1024)
            self._log(f"   📊 Available RAM: {available_mb:.0f}MB ({100-ram.percent:.1f}% free)")
        except ImportError:
            available_mb = None

        # Only use sampling as absolute last resort (>95% of limit AND low available RAM)
        use_sampling = False
        if est_mem > limit_mb * 0.95:
            if available_mb is not None and available_mb < est_mem * 1.5:
                # Really need to sample - not enough RAM
                sample_size = int(n * (limit_mb / est_mem) * 0.7)
                sample_size = max(50_000_000, (sample_size // 1_000_000) * 1_000_000)
                use_sampling = True
                self._log(f"   ⚠️ Low available RAM! Using sampling: {sample_size:,} chars ({100*sample_size/n:.1f}%)")

                # Sample from beginning, middle, and end for representativeness
                chunk_size = sample_size // 3
                start_chunk = phi_structural[:chunk_size]
                mid_start = (n - chunk_size) // 2
                mid_chunk = phi_structural[mid_start:mid_start + chunk_size]
                end_chunk = phi_structural[-(sample_size - 2*chunk_size):]

                phi_structural = start_chunk + mid_chunk + end_chunk
                n = len(phi_structural)
                self._log(f"   📊 Sampled sequence length: {n:,} chars")
            else:
                self._log(f"   ✅ Sufficient RAM available, proceeding with full analysis")

        # Method 1: Nesting depth patterns (O(n), fast)
        self._log(f"\n   📐 Phase 1/3: Nesting patterns...")
        nesting_patterns = self._detect_nesting_patterns(phi_structural)
        elapsed = time.time() - self._start_time
        self._log(f"   ✅ Phase 1 complete ({self._format_time(elapsed)})")

        # Method 2: Containment patterns (optimized)
        self._log(f"\n   📦 Phase 2/3: Containment patterns...")
        containment_patterns = self._detect_containment_patterns(phi_structural)
        elapsed = time.time() - self._start_time
        self._log(f"   ✅ Phase 2 complete ({self._format_time(elapsed)})")

        # Method 3: Stratified order patterns
        self._log(f"\n   📊 Phase 3/3: Stratified order patterns...")
        stratified_patterns = self._detect_stratified_order_patterns(phi_structural)
        elapsed = time.time() - self._start_time
        self._log(f"   ✅ Phase 3 complete ({self._format_time(elapsed)})")

        # Combine all patterns
        all_patterns = nesting_patterns + containment_patterns + stratified_patterns

        # Add sampling metadata if we used sampling
        if use_sampling:
            sampling_meta = {
                'type': 'sampling_metadata',
                'original_length': original_n,
                'sampled_length': n,
                'sample_coverage': n / original_n,
                'sampling_strategy': 'start+middle+end',
                'warning': 'Results based on sampled data due to memory constraints'
            }
            all_patterns.insert(0, sampling_meta)
            self._log(f"   ⚠️ Note: Results based on {100*n/original_n:.1f}% sample")

        total_time = time.time() - self._start_time
        self._log(f"\n   🎉 Structural analysis complete: {len(all_patterns)} patterns in {self._format_time(total_time)}")

        self.detected_patterns = all_patterns
        return all_patterns
    
    def _detect_nesting_patterns(self, phi_structural: str) -> List[Dict[str, Any]]:
        """
        Detect patterns in how Absolutes nest within each other.

        Analyzes:
        - Maximum nesting depth
        - Depth distribution
        - Depth transitions (calculated on-the-fly, O(1) memory)

        v33.2: Uses Numba JIT for 5-10x speedup on large sequences.
        """
        patterns = []
        n = len(phi_structural)

        # Try Numba-optimized version if available
        # For very large arrays, use chunked processing to avoid crashes
        NUMBA_CHUNK_SIZE = 100_000_000  # 100M elements per chunk
        use_numba = NUMBA_STRUCTURAL_AVAILABLE and n > 100_000
        numba_success = False

        if use_numba:
            try:
                if n <= NUMBA_CHUNK_SIZE:
                    # Small enough for single pass
                    self._log(f"      [nesting] Using Numba acceleration...")
                    sys.stdout.flush()
                    seq_bytes = str_to_bytes(phi_structural)
                    max_depth, total_parens, depth_dist, trans_flat = _analyze_nesting_numba(seq_bytes)
                else:
                    # Use chunked processing for large sequences
                    num_chunks = (n + NUMBA_CHUNK_SIZE - 1) // NUMBA_CHUNK_SIZE
                    self._log(f"      [nesting] Using Numba chunked ({num_chunks} chunks of {NUMBA_CHUNK_SIZE:,})...")
                    sys.stdout.flush()

                    # Initialize accumulators
                    MAX_DEPTH = 256
                    depth_dist = np.zeros(MAX_DEPTH, dtype=np.int64)
                    trans_flat = np.zeros(MAX_DEPTH * MAX_DEPTH, dtype=np.int64)
                    max_depth = np.int64(0)
                    current_depth = np.int64(0)
                    prev_depth = np.int64(0)
                    total_parens = np.int64(0)

                    # Use tqdm for chunk progress if available
                    if TQDM_AVAILABLE:
                        chunk_iter = tqdm(range(num_chunks), desc="      [nesting]",
                                         unit="chunk", leave=True, ncols=100,
                                         bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
                    else:
                        chunk_iter = range(num_chunks)

                    for chunk_idx in chunk_iter:
                        start = chunk_idx * NUMBA_CHUNK_SIZE
                        end = min(start + NUMBA_CHUNK_SIZE, n)
                        chunk_str = phi_structural[start:end]
                        chunk_bytes = str_to_bytes(chunk_str)

                        chunk_max, current_depth, prev_depth, total_parens = _analyze_nesting_numba_chunk(
                            chunk_bytes, current_depth, prev_depth, total_parens,
                            depth_dist, trans_flat
                        )
                        max_depth = max(max_depth, chunk_max)

                        if TQDM_AVAILABLE:
                            chunk_iter.set_postfix({'max_depth': int(max_depth)})
                        else:
                            pct = 100 * (chunk_idx + 1) // num_chunks
                            self._log(f"      [nesting] Chunk {chunk_idx+1}/{num_chunks} ({pct}%) - max_depth={max_depth}")
                            sys.stdout.flush()

                # Convert numpy arrays back to dicts
                depth_distribution = {int(d): int(c) for d, c in enumerate(depth_dist) if c > 0}

                # Reconstruct transitions dict from flattened array
                MAX_DEPTH = 256
                transitions = {}
                for from_d in range(MAX_DEPTH):
                    for to_d in range(MAX_DEPTH):
                        count = trans_flat[from_d * MAX_DEPTH + to_d]
                        if count > 0:
                            transitions[f"({from_d},{to_d})"] = int(count)

                self._log(f"      [nesting] Complete: max_depth={max_depth}, parens={total_parens:,}")
                numba_success = True

            except Exception as e:
                self._log(f"      ❌ [nesting] Numba error: {e}")
                self._log(f"      [nesting] Falling back to Python...")

        # Fallback to pure Python
        if not numba_success:
            log_interval = max(1, n // 20)
            depth = 0
            max_depth = 0
            prev_depth = 0
            depth_distribution = defaultdict(int)
            transitions = defaultdict(int)
            total_parens = 0

            if TQDM_AVAILABLE and n > 1_000_000:
                iterator = tqdm(enumerate(phi_structural), total=n,
                               desc="      [nesting]", unit="char", leave=True, ncols=100)
            else:
                iterator = enumerate(phi_structural)

            for i, char in iterator:
                if not TQDM_AVAILABLE and i % log_interval == 0 and i > 0:
                    pct = 100 * i // n
                    self._log(f"      [nesting] {i:,}/{n:,} ({pct}%) - max_depth={max_depth}")

                if char == '(':
                    depth += 1
                    max_depth = max(max_depth, depth)
                    depth_distribution[depth] += 1
                    total_parens += 1
                    if total_parens > 1:
                        transitions[f"({prev_depth},{depth})"] += 1
                    prev_depth = depth

                elif char == ')':
                    total_parens += 1
                    transitions[f"({prev_depth},{depth})"] += 1
                    prev_depth = depth
                    depth -= 1

            depth_distribution = dict(depth_distribution)
            transitions = dict(transitions)

        # Check if this is a flat sequence (no parentheses)
        if total_parens == 0:
            self._log(f"      [nesting] No parentheses found - skipping structural analysis")
            patterns.append({
                'pattern_id': 'nesting_depth_distribution',
                'pattern_type': 'nesting',
                'max_depth': 0,
                'depth_distribution': {},
                'total_parentheses': 0,
                'is_flat_sequence': True,
                'note': 'Sequence has no structural parentheses - flat binary data',
                'method': 'nesting_analysis'
            })
            return patterns

        # Pattern 1: Depth distribution
        patterns.append({
            'pattern_id': 'nesting_depth_distribution',
            'pattern_type': 'nesting',
            'max_depth': int(max_depth),
            'depth_distribution': depth_distribution,
            'total_parentheses': int(total_parens),
            'method': 'nesting_analysis' + ('_numba' if numba_success else '')
        })

        # Pattern 2: Depth transitions (LIMITED to top 1000 most common)
        if transitions:
            # Sort by frequency and keep only top transitions
            MAX_TRANSITIONS = 1000
            if len(transitions) > MAX_TRANSITIONS:
                sorted_trans = sorted(transitions.items(), key=lambda x: -x[1])
                top_transitions = dict(sorted_trans[:MAX_TRANSITIONS])
                total_trans = sum(transitions.values())
                covered_trans = sum(top_transitions.values())
                self._log(f"      [nesting] Limiting transitions: {len(transitions):,} → {MAX_TRANSITIONS} (covers {100*covered_trans/total_trans:.1f}%)")
            else:
                top_transitions = transitions
                total_trans = sum(transitions.values())

            patterns.append({
                'pattern_id': 'nesting_depth_transitions',
                'pattern_type': 'nesting',
                'transitions': top_transitions,
                'num_unique_transitions': len(transitions),
                'num_stored_transitions': len(top_transitions),
                'total_transitions': total_trans,
                'method': 'transition_analysis'
            })

        return patterns
    
    def _detect_containment_patterns(self, phi_structural: str) -> List[Dict[str, Any]]:
        """
        Detect what binary sequences are contained within each Absolute.

        v33.2: Uses Numba JIT for 5-10x speedup.
        Note: Numba version doesn't track recurring content patterns (trade-off for speed).

        Analyzes:
        - Content length distribution
        - Content complexity (entropy) - computed incrementally
        - Recurring content patterns (Python fallback only)
        """
        patterns = []
        n = len(phi_structural)

        # Quick check: does this sequence have parentheses?
        # Sample first 10000 chars to detect flat sequences efficiently
        sample = phi_structural[:min(10000, n)]
        has_parens = '(' in sample or ')' in sample
        if not has_parens:
            self._log(f"      [containment] No parentheses found - skipping (flat sequence)")
            patterns.append({
                'pattern_id': 'containment_length_distribution',
                'pattern_type': 'containment',
                'num_absolutes': 0,
                'is_flat_sequence': True,
                'note': 'Sequence has no structural parentheses - flat binary data',
                'method': 'containment_analysis'
            })
            return patterns

        # Chunked Numba for large sequences
        NUMBA_CHUNK_SIZE = 100_000_000
        use_numba = NUMBA_STRUCTURAL_AVAILABLE and n > 100_000
        numba_success = False

        if use_numba:
            try:
                MAX_DEPTH = 256
                MAX_LENGTH = 10000

                if n <= NUMBA_CHUNK_SIZE:
                    # Single pass
                    self._log(f"      [containment] Using Numba acceleration ({n:,} chars)...")
                    seq_bytes = str_to_bytes(phi_structural)
                    length_counts, entropy_stats, num_absolutes, sum_lengths, sum_sq_lengths = \
                        _analyze_containment_numba(seq_bytes)
                else:
                    # Chunked processing
                    num_chunks = (n + NUMBA_CHUNK_SIZE - 1) // NUMBA_CHUNK_SIZE
                    self._log(f"      [containment] Using Numba chunked ({num_chunks} chunks)...")

                    # Initialize state
                    stack_starts = np.zeros(MAX_DEPTH, dtype=np.int64)
                    stack_zeros = np.zeros(MAX_DEPTH, dtype=np.int64)
                    stack_ones = np.zeros(MAX_DEPTH, dtype=np.int64)
                    stack_ptr = np.int64(0)
                    length_counts = np.zeros(MAX_LENGTH, dtype=np.int64)
                    entropy_stats = np.zeros(3, dtype=np.float64)
                    sum_lengths = np.float64(0.0)
                    sum_sq_lengths = np.float64(0.0)
                    num_absolutes = np.int64(0)

                    for chunk_idx in range(num_chunks):
                        start = chunk_idx * NUMBA_CHUNK_SIZE
                        end = min(start + NUMBA_CHUNK_SIZE, n)
                        chunk_bytes = str_to_bytes(phi_structural[start:end])

                        stack_ptr, num_absolutes, sum_lengths, sum_sq_lengths = \
                            _analyze_containment_numba_chunk(
                                chunk_bytes, stack_starts, stack_zeros, stack_ones,
                                stack_ptr, length_counts, entropy_stats,
                                sum_lengths, sum_sq_lengths, num_absolutes
                            )

                        pct = 100 * (chunk_idx + 1) // num_chunks
                        self._log(f"      [containment] Chunk {chunk_idx+1}/{num_chunks} ({pct}%)")

                numba_success = True
            except Exception as e:
                self._log(f"      ❌ [containment] Numba error: {e}, falling back to Python...")

        if numba_success:

            # Convert length_counts array to dict
            length_distribution = {int(length): int(count)
                                  for length, count in enumerate(length_counts) if count > 0}

            # Calculate mean and std from sums
            if num_absolutes > 0:
                mean_length = sum_lengths / num_absolutes
                if num_absolutes > 1:
                    variance = (sum_sq_lengths - sum_lengths * sum_lengths / num_absolutes) / (num_absolutes - 1)
                    std_length = np.sqrt(max(0, variance))
                else:
                    std_length = 0.0
            else:
                mean_length = 0.0
                std_length = 0.0

            # Extract entropy stats
            entropy_count = int(entropy_stats[0])
            entropy_mean = float(entropy_stats[1])
            entropy_m2 = float(entropy_stats[2])
            if entropy_count > 1:
                entropy_std = np.sqrt(entropy_m2 / (entropy_count - 1))
            else:
                entropy_std = 0.0

            self._log(f"      [containment] Complete: {num_absolutes:,} absolutes (Numba)")

            # Build patterns
            patterns.append({
                'pattern_id': 'containment_length_distribution',
                'pattern_type': 'containment',
                'num_absolutes': int(num_absolutes),
                'length_distribution': length_distribution,
                'mean_length': mean_length,
                'std_length': std_length,
                'method': 'containment_analysis_numba'
            })

            if entropy_count > 0:
                patterns.append({
                    'pattern_id': 'containment_entropy_distribution',
                    'pattern_type': 'containment',
                    'mean_entropy': entropy_mean,
                    'std_entropy': entropy_std,
                    'num_samples': entropy_count,
                    'method': 'entropy_analysis_numba'
                })

            # Note: Recurring content patterns not available in Numba version
            # (would require hash maps which Numba doesn't support efficiently)

            return patterns

        # Fallback to pure Python with progress bar
        log_interval = max(1, n // 20)
        stack = []
        depth = 0
        length_distribution = Counter()
        length_accumulator = WelfordAccumulator()
        entropy_accumulator = WelfordAccumulator()
        content_counts = Counter()
        num_absolutes = 0

        self._log(f"      [containment] Processing {n:,} characters (Python)...")

        if TQDM_AVAILABLE and n > 1_000_000:
            iterator = tqdm(enumerate(phi_structural), total=n,
                           desc="      [containment]", unit="char", leave=True, ncols=100)
        else:
            iterator = enumerate(phi_structural)

        for i, char in iterator:
            if not TQDM_AVAILABLE and i % log_interval == 0 and i > 0:
                pct = 100 * i // n
                self._log(f"      [containment] {i:,}/{n:,} ({pct}%) - absolutes={num_absolutes:,}")

            if char == '(':
                depth += 1
                stack.append({'start': i, 'depth': depth, 'chars': []})
            elif char == ')':
                if stack:
                    absolute = stack.pop()
                    observable_content = ''.join(absolute['chars'])
                    content_len = len(observable_content)
                    length_distribution[content_len] += 1
                    length_accumulator.update(content_len)
                    if content_len > 0:
                        counts = Counter(observable_content)
                        probs = np.array([counts.get(c, 0) / content_len for c in '01'])
                        probs = probs[probs > 0]
                        ent = entropy(probs, base=2) if len(probs) > 0 else 0
                        entropy_accumulator.update(ent)
                    if content_len <= 100:
                        content_counts[observable_content] += 1
                    num_absolutes += 1
                depth -= 1
            else:
                for abs_info in stack:
                    abs_info['chars'].append(char)

        self._log(f"      [containment] Complete: {num_absolutes:,} absolutes processed")

        if num_absolutes == 0:
            return patterns

        patterns.append({
            'pattern_id': 'containment_length_distribution',
            'pattern_type': 'containment',
            'num_absolutes': num_absolutes,
            'length_distribution': dict(length_distribution),
            'mean_length': length_accumulator.mean,
            'std_length': length_accumulator.std,
            'method': 'containment_analysis'
        })

        if entropy_accumulator.count > 0:
            patterns.append({
                'pattern_id': 'containment_entropy_distribution',
                'pattern_type': 'containment',
                'mean_entropy': entropy_accumulator.mean,
                'std_entropy': entropy_accumulator.std,
                'num_samples': entropy_accumulator.count,
                'method': 'entropy_analysis'
            })

        recurring_contents = {content: count for content, count in content_counts.items()
                            if count >= self.min_occurrences and len(content) > 0}
        if recurring_contents:
            top_recurring = dict(Counter(recurring_contents).most_common(1000))
            patterns.append({
                'pattern_id': 'containment_recurring_contents',
                'pattern_type': 'containment',
                'recurring_contents': top_recurring,
                'num_unique_recurring': len(recurring_contents),
                'total_unique_tracked': len(content_counts),
                'method': 'recurrence_analysis'
            })

        return patterns

    def _detect_stratified_order_patterns(self, phi_structural: str) -> List[Dict[str, Any]]:
        """
        Detect how order emerges through stratified containment.

        v33.2: Uses Numba JIT for 3-5x speedup.

        Analyzes:
        - Entropy at each nesting level
        - Information density at each level
        - Correlation between levels
        """
        patterns = []
        n = len(phi_structural)

        # Quick check: does this sequence have parentheses?
        sample = phi_structural[:min(10000, n)]
        has_parens = '(' in sample or ')' in sample
        if not has_parens:
            self._log(f"      [stratified] No parentheses found - skipping (flat sequence)")
            patterns.append({
                'pattern_id': 'stratified_order_metrics',
                'pattern_type': 'stratified_order',
                'num_depths': 0,
                'is_flat_sequence': True,
                'note': 'Sequence has no structural parentheses - flat binary data',
                'method': 'stratified_analysis'
            })
            return patterns

        # Chunked Numba for large sequences
        NUMBA_CHUNK_SIZE = 100_000_000
        MAX_DEPTH = 256
        use_numba = NUMBA_STRUCTURAL_AVAILABLE and n > 100_000
        numba_success = False

        if use_numba:
            try:
                if n <= NUMBA_CHUNK_SIZE:
                    # Single pass
                    self._log(f"      [stratified] Using Numba acceleration ({n:,} chars)...")
                    seq_bytes = str_to_bytes(phi_structural)
                    depth_zeros, depth_ones, depth_seq_counts = _analyze_stratified_numba(seq_bytes)
                else:
                    # Chunked processing
                    num_chunks = (n + NUMBA_CHUNK_SIZE - 1) // NUMBA_CHUNK_SIZE
                    self._log(f"      [stratified] Using Numba chunked ({num_chunks} chunks)...")

                    depth_zeros = np.zeros(MAX_DEPTH, dtype=np.int64)
                    depth_ones = np.zeros(MAX_DEPTH, dtype=np.int64)
                    depth_seq_counts = np.zeros(MAX_DEPTH, dtype=np.int64)
                    current_depth = np.int64(0)
                    current_zeros = np.int64(0)
                    current_ones = np.int64(0)

                    for chunk_idx in range(num_chunks):
                        start = chunk_idx * NUMBA_CHUNK_SIZE
                        end = min(start + NUMBA_CHUNK_SIZE, n)
                        chunk_bytes = str_to_bytes(phi_structural[start:end])

                        current_depth, current_zeros, current_ones = _analyze_stratified_numba_chunk(
                            chunk_bytes, current_depth, current_zeros, current_ones,
                            depth_zeros, depth_ones, depth_seq_counts
                        )

                        pct = 100 * (chunk_idx + 1) // num_chunks
                        self._log(f"      [stratified] Chunk {chunk_idx+1}/{num_chunks} ({pct}%)")

                # Build depth_stats from numpy arrays
                depth_stats = {}
                for d in range(len(depth_zeros)):
                    zeros = int(depth_zeros[d])
                    ones = int(depth_ones[d])
                    seq_count = int(depth_seq_counts[d])
                    if zeros > 0 or ones > 0:
                        depth_stats[d] = {
                            'zeros': zeros,
                            'ones': ones,
                            'sequence_count': seq_count,
                            'total_chars': zeros + ones
                        }

                self._log(f"      [stratified] Complete: {len(depth_stats)} depths (Numba)")
                numba_success = True

            except Exception as e:
                self._log(f"      ❌ [stratified] Numba error: {e}, falling back to Python...")

        if not numba_success:
            # Fallback to pure Python with progress bar
            log_interval = max(1, n // 20)
            depth_stats = defaultdict(lambda: {'zeros': 0, 'ones': 0, 'sequence_count': 0, 'total_chars': 0})
            depth = 0
            current_chars = []

            self._log(f"      [stratified] Processing {n:,} characters (Python)...")

            if TQDM_AVAILABLE and n > 1_000_000:
                iterator = tqdm(enumerate(phi_structural), total=n,
                               desc="      [stratified]", unit="char", leave=True, ncols=100)
            else:
                iterator = enumerate(phi_structural)

            for i, char in iterator:
                if not TQDM_AVAILABLE and i % log_interval == 0 and i > 0:
                    pct = 100 * i // n
                    self._log(f"      [stratified] {i:,}/{n:,} ({pct}%)")

                if char == '(':
                    if current_chars and depth > 0:
                        seq = ''.join(current_chars)
                        depth_stats[depth]['zeros'] += seq.count('0')
                        depth_stats[depth]['ones'] += seq.count('1')
                        depth_stats[depth]['total_chars'] += len(seq)
                        depth_stats[depth]['sequence_count'] += 1
                        current_chars = []
                    depth += 1
                elif char == ')':
                    if current_chars and depth > 0:
                        seq = ''.join(current_chars)
                        depth_stats[depth]['zeros'] += seq.count('0')
                        depth_stats[depth]['ones'] += seq.count('1')
                        depth_stats[depth]['total_chars'] += len(seq)
                        depth_stats[depth]['sequence_count'] += 1
                        current_chars = []
                    depth -= 1
                else:
                    current_chars.append(char)

            if current_chars and depth > 0:
                seq = ''.join(current_chars)
                depth_stats[depth]['zeros'] += seq.count('0')
                depth_stats[depth]['ones'] += seq.count('1')
                depth_stats[depth]['total_chars'] += len(seq)
                depth_stats[depth]['sequence_count'] += 1

            depth_stats = dict(depth_stats)

        if not depth_stats:
            return patterns

        # Calculate metrics from accumulated stats
        depth_metrics = {}
        for d, stats in sorted(depth_stats.items()):
            total = stats['total_chars']
            if total == 0:
                continue

            zeros = stats['zeros']
            ones = stats['ones']
            if zeros + ones > 0:
                probs = np.array([zeros / (zeros + ones), ones / (zeros + ones)])
                probs = probs[probs > 0]
                ent = entropy(probs, base=2) if len(probs) > 0 else 0
            else:
                ent = 0

            density = total / n if n > 0 else 0

            depth_metrics[d] = {
                'entropy': float(ent),
                'density': float(density),
                'sequence_count': stats['sequence_count'],
                'total_length': total,
                'zero_count': zeros,
                'one_count': ones
            }

        method = 'stratified_analysis_numba' if NUMBA_STRUCTURAL_AVAILABLE and n > 100_000 else 'stratified_analysis'
        patterns.append({
            'pattern_id': 'stratified_order_metrics',
            'pattern_type': 'stratified_order',
            'depth_metrics': depth_metrics,
            'num_depths': len(depth_metrics),
            'method': method
        })

        self._log(f"      [stratified] Complete: {len(depth_metrics)} depth levels analyzed")

        return patterns

    def _extract_absolutes(self, phi_structural: str) -> List[Dict[str, Any]]:
        """
        Extract all parenthesized groups (Absolutes) from Φ.

        ⚠️ DEPRECATED in v33.1 - Use _detect_containment_patterns() instead,
        which processes absolutes on-the-fly without storing them all.

        This method is kept for backwards compatibility but should not be
        used for large sequences.

        Returns list of dictionaries with:
        - observable_content: Content without parentheses
        - start_position: Start index in phi_structural
        - end_position: End index in phi_structural
        - depth: Nesting depth
        """
        self._log("      ⚠️ Warning: _extract_absolutes() is deprecated for large sequences")

        absolutes = []
        stack = []  # Each entry: {'start': int, 'depth': int, 'chars': list}
        depth = 0
        n = len(phi_structural)
        log_interval = max(1, n // 20)
        max_absolutes = self.DEFAULT_MAX_ABSOLUTES

        for i, char in enumerate(phi_structural):
            # Progress logging
            if i % log_interval == 0 and i > 0:
                pct = 100 * i // n
                self._log(f"      [extract] {i:,}/{n:,} ({pct}%) - absolutes={len(absolutes):,}")

            # Safety limit
            if len(absolutes) >= max_absolutes:
                self._log(f"      ⚠️ Reached {max_absolutes:,} absolutes limit, stopping extraction")
                break

            if char == '(':
                depth += 1
                stack.append({'start': i, 'depth': depth, 'chars': []})
            elif char == ')':
                if stack:
                    absolute = stack.pop()
                    observable_content = ''.join(absolute['chars'])

                    absolutes.append({
                        'observable_content': observable_content,
                        'start_position': absolute['start'],
                        'end_position': i,
                        'depth': absolute['depth']
                    })
                depth -= 1
            else:
                # Add character to all open absolutes using O(1) list append
                for abs_info in stack:
                    abs_info['chars'].append(char)

        return absolutes

    def _extract_sequences_by_depth(self, phi_structural: str) -> Dict[int, List[str]]:
        """
        Extract all character sequences at each nesting depth.

        ⚠️ DEPRECATED in v33.1 - Use _detect_stratified_order_patterns() instead,
        which computes statistics on-the-fly without storing all sequences.

        Returns dictionary mapping depth -> list of sequences at that depth
        """
        self._log("      ⚠️ Warning: _extract_sequences_by_depth() is deprecated for large sequences")

        depth_sequences = defaultdict(list)
        depth = 0
        current_chars = []  # Use list for O(1) append
        n = len(phi_structural)
        log_interval = max(1, n // 20)

        for i, char in enumerate(phi_structural):
            if i % log_interval == 0 and i > 0:
                pct = 100 * i // n
                self._log(f"      [sequences] {i:,}/{n:,} ({pct}%)")

            if char == '(':
                if current_chars and depth > 0:
                    depth_sequences[depth].append(''.join(current_chars))
                    current_chars = []
                depth += 1
            elif char == ')':
                if current_chars:
                    depth_sequences[depth].append(''.join(current_chars))
                    current_chars = []
                depth -= 1
            else:
                current_chars.append(char)

        # Add final sequence if any
        if current_chars and depth > 0:
            depth_sequences[depth].append(''.join(current_chars))

        return dict(depth_sequences)

    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get summary statistics of detected structural patterns."""
        if not self.detected_patterns:
            return {
                'total_patterns': 0,
                'pattern_types': {}
            }

        pattern_types = Counter([p['pattern_type'] for p in self.detected_patterns])

        return {
            'total_patterns': len(self.detected_patterns),
            'pattern_types': dict(pattern_types),
            'patterns': self.detected_patterns
        }

    def save_patterns(self, output_path: str):
        """Save detected patterns to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': self.get_pattern_summary(),
                'patterns': self.detected_patterns
            }, f, indent=2, ensure_ascii=False)

        print(f"✅ Structural patterns saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    print("🔍 Structural Pattern Detector — HSI v33.1 (Optimized)")
    print("=" * 60)

    # Test 1: Small example
    print("\n📌 Test 1: Small example")
    example_phi = "((01)1)"

    detector = StructuralPatternDetector(
        min_depth=1,
        max_depth=10,
        min_occurrences=1,
        verbose=True
    )
    patterns = detector.detect_structural_patterns(example_phi)

    print(f"\nDetected {len(patterns)} structural patterns:")
    for pattern in patterns:
        print(f"\n  Pattern ID: {pattern['pattern_id']}")
        print(f"  Type: {pattern['pattern_type']}")
        print(f"  Method: {pattern['method']}")

    summary = detector.get_pattern_summary()
    print(f"\nSummary: {summary['total_patterns']} patterns")
    print(f"Pattern types: {summary['pattern_types']}")

    # Test 2: Performance test with larger sequence
    print("\n" + "=" * 60)
    print("📌 Test 2: Performance test (1M characters)")

    # Generate a larger test sequence
    import random
    random.seed(42)
    large_phi = ""
    depth = 0
    for _ in range(250_000):
        if random.random() < 0.4 and depth < 20:
            large_phi += "("
            depth += 1
        elif random.random() < 0.4 and depth > 0:
            large_phi += ")"
            depth -= 1
        else:
            large_phi += random.choice("01")
    # Close remaining parentheses
    large_phi += ")" * depth

    print(f"Generated test sequence: {len(large_phi):,} characters")

    detector2 = StructuralPatternDetector(
        min_depth=1,
        max_depth=50,
        min_occurrences=2,
        verbose=True
    )

    import time
    start = time.time()
    patterns2 = detector2.detect_structural_patterns(large_phi)
    elapsed = time.time() - start

    print(f"\n✅ Completed in {elapsed:.2f}s")
    print(f"   Patterns detected: {len(patterns2)}")

    summary2 = detector2.get_pattern_summary()
    print(f"   Pattern types: {summary2['pattern_types']}")


