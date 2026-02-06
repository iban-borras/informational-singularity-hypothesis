#!/usr/bin/env python3
"""
Level 1 — Deep Analysis Tools (Phase 2)

Unified script for advanced pattern discovery including:
1. Wavelet φ-Band Detection — Find φ signatures in frequency domain
2. Recurrence Plot Analysis — Detect temporal dynamics and periodicity
3. Lempel-Ziv Complexity — Measure algorithmic complexity at multiple scales

Usage:
    python level1_deep_analysis.py --variant B --iteration 15
    python level1_deep_analysis.py --variant B --iteration 15 --analysis wavelet
    python level1_deep_analysis.py --variant B --iteration 15 --analysis recurrence
    python level1_deep_analysis.py --variant B --iteration 15 --compare F
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None

# Numba for JIT compilation
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback: decorator that does nothing
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

sys.path.insert(0, str(Path(__file__).parent))
from level1.data_loader import load_phi_for_level1, load_metadata
from utils.streaming_phi_loader import load_phi_for_agents

PHI = (1 + math.sqrt(5)) / 2


# =============================================================================
# Multi-Scale Subsampling (for 100G+ bit datasets)
# =============================================================================

def multiscale_subsample(bits: str, verbose: bool = True) -> Dict[str, np.ndarray]:
    """
    Create multi-scale subsamples from a large bit string.

    Based on Nyquist theorem: to detect patterns of scale N, we only need
    to sample at frequency 2/N. This allows analyzing 100G+ bits using
    only ~12M values in memory.

    Sampling strategy:
    - Fine scale (1-1k patterns):     every bit,    max 100k samples
    - Medium scale (1k-1M patterns):  every 100,    max 1M samples
    - Large scale (1M-1G patterns):   every 10k,    max 10M samples
    - Huge scale (1G+ patterns):      every 1M,     max 100k samples

    Args:
        bits: Binary string (can be 100G+ characters)
        verbose: Print progress info

    Returns:
        Dict with 'fine', 'medium', 'large', 'huge' numpy arrays
        and metadata about sampling
    """
    total_bits = len(bits)

    if verbose:
        print(f"\n📐 Multi-Scale Subsampling (Nyquist-based)")
        print(f"   Total bits: {total_bits:,}")

    samples = {}
    metadata = {
        'total_bits': total_bits,
        'levels': {}
    }

    # Level 1: Fine scale - every bit, max 100k
    fine_step = 1
    fine_max = min(100_000, total_bits)
    fine_data = bits[:fine_max]
    samples['fine'] = np.frombuffer(fine_data.encode('ascii'), dtype=np.uint8) - ord('0')
    metadata['levels']['fine'] = {
        'step': fine_step,
        'count': len(samples['fine']),
        'covers_range': f"0-{fine_max:,}",
        'detects_patterns': "1-1k bits"
    }

    # Level 2: Medium scale - every 100 bits, max 1M samples
    medium_step = 100
    medium_max = min(100_000_000, total_bits)  # covers up to 100M bits
    medium_indices = range(0, medium_max, medium_step)
    medium_data = ''.join(bits[i] for i in medium_indices if i < total_bits)
    samples['medium'] = np.frombuffer(medium_data.encode('ascii'), dtype=np.uint8) - ord('0')
    metadata['levels']['medium'] = {
        'step': medium_step,
        'count': len(samples['medium']),
        'covers_range': f"0-{medium_max:,}",
        'detects_patterns': "1k-1M bits"
    }

    # Level 3: Large scale - every 10k bits, max 10M samples
    large_step = 10_000
    large_max = min(100_000_000_000, total_bits)  # covers up to 100G bits
    large_indices = range(0, large_max, large_step)
    large_data = ''.join(bits[i] for i in large_indices if i < total_bits)
    samples['large'] = np.frombuffer(large_data.encode('ascii'), dtype=np.uint8) - ord('0')
    metadata['levels']['large'] = {
        'step': large_step,
        'count': len(samples['large']),
        'covers_range': f"0-{large_max:,}",
        'detects_patterns': "1M-1G bits"
    }

    # Level 4: Huge scale - every 1M bits, max 100k samples
    huge_step = 1_000_000
    huge_max = total_bits
    huge_indices = range(0, huge_max, huge_step)
    huge_data = ''.join(bits[i] for i in huge_indices if i < total_bits)
    if huge_data:
        samples['huge'] = np.frombuffer(huge_data.encode('ascii'), dtype=np.uint8) - ord('0')
    else:
        samples['huge'] = np.array([], dtype=np.uint8)
    metadata['levels']['huge'] = {
        'step': huge_step,
        'count': len(samples['huge']),
        'covers_range': f"0-{huge_max:,}",
        'detects_patterns': "1G+ bits"
    }

    # Calculate memory usage
    total_samples = sum(len(s) for s in samples.values())
    memory_bytes = sum(s.nbytes for s in samples.values())
    metadata['total_samples'] = total_samples
    metadata['memory_bytes'] = memory_bytes

    if verbose:
        print(f"   ┌─────────────┬──────────┬─────────────┬──────────────────┐")
        print(f"   │ Level       │ Step     │ Samples     │ Detects patterns │")
        print(f"   ├─────────────┼──────────┼─────────────┼──────────────────┤")
        for level_name in ['fine', 'medium', 'large', 'huge']:
            info = metadata['levels'][level_name]
            print(f"   │ {level_name:<11} │ {info['step']:>8,} │ {info['count']:>11,} │ {info['detects_patterns']:<16} │")
        print(f"   └─────────────┴──────────┴─────────────┴──────────────────┘")
        print(f"   Total: {total_samples:,} samples ({memory_bytes / 1_000_000:.1f} MB)")
        print(f"   Compression: {total_bits / max(total_samples, 1):.1f}x")
        print(f"   ℹ️  Nyquist: patterns of scale N detectable with samples at 2/N frequency")

    return {'samples': samples, 'metadata': metadata}


# =============================================================================
# Wavelet φ-Band Detection
# =============================================================================

def wavelet_analysis(bits: str, max_bits: int = 100000, verbose: bool = True) -> Dict:
    """
    Detect φ-related frequency bands using wavelet-like multi-scale analysis.

    Instead of full CWT (computationally expensive), we use a simplified
    multi-scale approach: analyze entropy at scales that form φ-ratios.

    Args:
        bits: Binary string
        max_bits: Maximum bits to analyze
        verbose: Print progress

    Returns:
        Analysis results with φ-band detection
    """
    if verbose:
        print(f"\n📊 Wavelet φ-Band Detection")
        print(f"   Data: {len(bits):,} bits (analyzing {min(len(bits), max_bits):,})")

    data = bits[:max_bits]

    # Convert to numpy array - vectorized (much faster than list comprehension)
    if verbose:
        print(f"   Converting to array...", end=" ", flush=True)
    # Use frombuffer for O(n) vectorized conversion instead of O(n) Python loop
    data_arr = np.frombuffer(data.encode('ascii'), dtype=np.uint8) - ord('0')
    data_arr = data_arr.astype(np.float32)
    if verbose:
        print(f"✓")

    # Generate φ-related scales: 2, 2φ, 2φ², 2φ³, ...
    phi_scales = []
    scale = 2
    while scale < len(data) // 10:
        phi_scales.append(int(scale))
        scale *= PHI

    # Also generate power-of-2 scales for comparison
    pow2_scales = [2**i for i in range(1, 15) if 2**i < len(data) // 10]

    if verbose:
        print(f"   φ-scales: {len(phi_scales)} scales ({phi_scales[:5]}...)")
        print(f"   2^n-scales: {len(pow2_scales)} scales ({pow2_scales[:5]}...)")

    def calc_scale_entropy(arr: np.ndarray, scale: int) -> float:
        """Calculate entropy at given scale by averaging blocks."""
        n_blocks = len(arr) // scale
        if n_blocks < 2:
            return 0.0
        blocks = arr[:n_blocks * scale].reshape(n_blocks, scale)
        densities = blocks.mean(axis=1)
        # Bin densities and calculate entropy
        hist, _ = np.histogram(densities, bins=10, range=(0, 1))
        hist = hist[hist > 0] / hist.sum()
        return float(-np.sum(hist * np.log2(hist + 1e-10)))

    phi_entropies = []
    pow2_entropies = []

    # Calculate φ-scale entropies with progress bar
    if verbose and TQDM_AVAILABLE:
        phi_iter = tqdm(phi_scales, desc="   φ-scales", unit="scale",
                        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    else:
        phi_iter = phi_scales
        if verbose:
            print(f"   Calculating φ-scale entropies...", end=" ", flush=True)

    for scale in phi_iter:
        phi_entropies.append(calc_scale_entropy(data_arr, scale))

    if verbose and not TQDM_AVAILABLE:
        print(f"✓")

    # Calculate 2^n-scale entropies with progress bar
    if verbose and TQDM_AVAILABLE:
        pow2_iter = tqdm(pow2_scales, desc="   2^n-scales", unit="scale",
                         bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    else:
        pow2_iter = pow2_scales
        if verbose:
            print(f"   Calculating 2^n-scale entropies...", end=" ", flush=True)

    for scale in pow2_iter:
        pow2_entropies.append(calc_scale_entropy(data_arr, scale))

    if verbose and not TQDM_AVAILABLE:
        print(f"✓")
    
    # Calculate smoothness (lower = more structured at those scales)
    phi_variance = float(np.var(phi_entropies)) if phi_entropies else 0
    pow2_variance = float(np.var(pow2_entropies)) if pow2_entropies else 0
    
    # φ-band strength: if φ-scales show more consistent entropy, φ is present
    phi_band_strength = 1.0 / (1.0 + phi_variance) if phi_variance > 0 else 0
    pow2_band_strength = 1.0 / (1.0 + pow2_variance) if pow2_variance > 0 else 0
    
    # Ratio indicates φ-preference
    phi_preference = phi_band_strength / (pow2_band_strength + 0.001)
    
    return {
        'phi_scales': phi_scales,
        'pow2_scales': pow2_scales,
        'phi_entropies': phi_entropies,
        'pow2_entropies': pow2_entropies,
        'phi_variance': phi_variance,
        'pow2_variance': pow2_variance,
        'phi_band_strength': phi_band_strength,
        'pow2_band_strength': pow2_band_strength,
        'phi_preference': phi_preference,
        'phi_detected': phi_preference > 1.2  # φ-scales more consistent than 2^n
    }


def wavelet_analysis_multiscale(subsamples: Dict, verbose: bool = True) -> Dict:
    """
    Wavelet φ-Band Detection using multi-scale subsamples.

    Analyzes patterns at ALL scales from 1 bit to 100G+ bits by using
    pre-computed subsamples at different resolutions.

    Args:
        subsamples: Output from multiscale_subsample()
        verbose: Print progress

    Returns:
        Analysis results with φ-band detection at all scales
    """
    samples = subsamples['samples']
    metadata = subsamples['metadata']

    if verbose:
        print(f"\n📊 Wavelet φ-Band Detection (Multi-Scale)")
        print(f"   Total bits represented: {metadata['total_bits']:,}")
        print(f"   Samples in memory: {metadata['total_samples']:,}")

    def calc_scale_entropy(arr: np.ndarray, scale: int) -> float:
        """Calculate entropy at given scale by averaging blocks."""
        n_blocks = len(arr) // scale
        if n_blocks < 2:
            return 0.0
        blocks = arr[:n_blocks * scale].reshape(n_blocks, scale).astype(np.float32)
        densities = blocks.mean(axis=1)
        hist, _ = np.histogram(densities, bins=10, range=(0, 1))
        hist = hist[hist > 0] / hist.sum()
        return float(-np.sum(hist * np.log2(hist + 1e-10)))

    all_phi_entropies = []
    all_pow2_entropies = []
    all_phi_scales = []
    all_pow2_scales = []

    # Map levels to real bit scales
    level_multipliers = {
        'fine': 1,       # each sample = 1 bit
        'medium': 100,   # each sample = 100 bits
        'large': 10_000, # each sample = 10k bits
        'huge': 1_000_000 # each sample = 1M bits
    }

    for level_name in ['fine', 'medium', 'large', 'huge']:
        arr = samples[level_name]
        multiplier = level_multipliers[level_name]

        if len(arr) < 20:
            continue

        if verbose:
            print(f"   Analyzing {level_name} scale ({len(arr):,} samples)...", end=" ", flush=True)

        # Generate φ-related scales for this level
        phi_scales = []
        scale = 2
        max_scale = len(arr) // 10
        while scale < max_scale:
            phi_scales.append(int(scale))
            scale *= PHI

        # Generate power-of-2 scales
        pow2_scales = [2**i for i in range(1, 20) if 2**i < max_scale]

        # Calculate entropies
        for s in phi_scales:
            ent = calc_scale_entropy(arr, s)
            all_phi_entropies.append(ent)
            all_phi_scales.append(s * multiplier)  # Real bit scale

        for s in pow2_scales:
            ent = calc_scale_entropy(arr, s)
            all_pow2_entropies.append(ent)
            all_pow2_scales.append(s * multiplier)  # Real bit scale

        if verbose:
            print(f"✓ ({len(phi_scales)} φ, {len(pow2_scales)} 2^n)")

    # Calculate overall φ-preference
    phi_variance = float(np.var(all_phi_entropies)) if all_phi_entropies else 0
    pow2_variance = float(np.var(all_pow2_entropies)) if all_pow2_entropies else 0

    phi_band_strength = 1.0 / (1.0 + phi_variance) if phi_variance > 0 else 0
    pow2_band_strength = 1.0 / (1.0 + pow2_variance) if pow2_variance > 0 else 0

    phi_preference = phi_band_strength / (pow2_band_strength + 0.001)

    # Scale range covered
    min_scale = min(all_phi_scales) if all_phi_scales else 0
    max_scale = max(all_phi_scales) if all_phi_scales else 0

    if verbose:
        print(f"   Scale range: {min_scale:,} - {max_scale:,} bits")
        print(f"   φ-preference: {phi_preference:.3f}")

    return {
        'phi_scales': all_phi_scales,
        'pow2_scales': all_pow2_scales,
        'phi_entropies': all_phi_entropies,
        'pow2_entropies': all_pow2_entropies,
        'phi_variance': phi_variance,
        'pow2_variance': pow2_variance,
        'phi_band_strength': phi_band_strength,
        'pow2_band_strength': pow2_band_strength,
        'phi_preference': phi_preference,
        'phi_detected': phi_preference > 1.2,
        'min_scale_bits': min_scale,
        'max_scale_bits': max_scale,
        'multiscale': True
    }


# =============================================================================
# Recurrence Plot Analysis
# =============================================================================

def recurrence_analysis(bits: str, max_bits: int = 10000,
                        embedding_dim: int = 3, delay: int = 1,
                        threshold: float = None, verbose: bool = True) -> Dict:
    """
    Recurrence Plot Analysis — detect temporal dynamics.

    Args:
        bits: Binary string
        max_bits: Maximum bits (recurrence is O(n²), keep small!)
        embedding_dim: Embedding dimension for phase space
        delay: Time delay for embedding
        threshold: Distance threshold for recurrence (auto-calculated if None)
        verbose: Print progress

    Returns:
        Recurrence metrics including determinism and laminarity

    Note on threshold:
        For binary data embedded in R^d, typical distances are:
        - Identical vectors: 0
        - One bit different: 1
        - Random: ~sqrt(d/2) ≈ 0.87 for d=3

        Auto-threshold uses 10% of the median pairwise distance,
        which is statistically meaningful for any data distribution.
    """
    if verbose:
        print(f"\n🔄 Recurrence Plot Analysis")
        print(f"   Data: {len(bits):,} bits (analyzing {min(len(bits), max_bits):,})")
        print(f"   Embedding: dim={embedding_dim}, delay={delay}")
    
    data = bits[:max_bits]
    data_arr = np.array([int(b) for b in data], dtype=np.float32)

    # Create embedded vectors
    n = len(data_arr) - (embedding_dim - 1) * delay
    if n < 10:
        return {'error': 'Insufficient data for embedding'}

    if verbose:
        print(f"   Creating {n} embedded vectors...", end=" ", flush=True)

    embedded = np.zeros((n, embedding_dim))
    for i in range(embedding_dim):
        embedded[:, i] = data_arr[i * delay:i * delay + n]

    if verbose:
        print(f"✓")
        print(f"   Building recurrence matrix...", end=" ", flush=True)

    # Calculate recurrence matrix (O(n²) - that's why we limit max_bits)
    # Use vectorized approach for speed
    diff = embedded[:, np.newaxis, :] - embedded[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=2))

    # Auto-calculate threshold if not provided
    # Use 10% of median distance (statistically meaningful)
    if threshold is None:
        # Sample distances to avoid computing full median (memory efficient)
        sample_size = min(1000, n)
        sample_idx = np.random.choice(n, sample_size, replace=False)
        sample_distances = distances[sample_idx][:, sample_idx]
        # Exclude diagonal (self-distances = 0)
        upper_tri = sample_distances[np.triu_indices(sample_size, k=1)]
        threshold = float(np.median(upper_tri) * 0.1)
        if verbose:
            print(f"(auto-threshold={threshold:.3f})", end=" ", flush=True)

    recurrence_matrix = (distances < threshold).astype(np.int8)
    
    if verbose:
        print(f"✓ ({n}×{n} matrix)")
    
    # Calculate RQA metrics
    total_points = n * n
    recurrence_rate = float(np.sum(recurrence_matrix)) / total_points
    
    # Determinism: ratio of recurrence points forming diagonal lines
    if verbose:
        print(f"   Calculating determinism...", end=" ", flush=True)
    
    diag_lengths = []
    for k in range(-n + 1, n):
        diag = np.diag(recurrence_matrix, k)
        # Find runs of 1s
        runs = np.diff(np.where(np.concatenate(([0], diag, [0])) != 1)[0])[::2]
        diag_lengths.extend(runs[runs >= 2].tolist())
    
    if diag_lengths:
        determinism = sum(diag_lengths) / max(1, np.sum(recurrence_matrix))
        avg_diag_length = float(np.mean(diag_lengths))
        max_diag_length = int(max(diag_lengths))
    else:
        determinism = 0
        avg_diag_length = 0
        max_diag_length = 0
    
    if verbose:
        print(f"✓")
    
    return {
        'recurrence_rate': recurrence_rate,
        'determinism': determinism,
        'avg_diagonal_length': avg_diag_length,
        'max_diagonal_length': max_diag_length,
        'embedding_dim': embedding_dim,
        'delay': delay,
        'threshold': threshold,
        'matrix_size': n,
        'is_deterministic': determinism > 0.5
    }


# =============================================================================
# Multi-Scale Lempel-Ziv Complexity
# =============================================================================

@njit(cache=True)
def _lempel_ziv_complexity_numba(arr: np.ndarray) -> int:
    """Calculate Lempel-Ziv complexity (LZ76) - Numba JIT version.

    Takes a numpy array of 0/1 values for maximum performance.
    """
    n = len(arr)
    if n == 0:
        return 0

    complexity = 1
    prefix_len = 1

    while prefix_len < n:
        # Find longest match
        max_match = 0
        for start in range(prefix_len):
            match_len = 0
            while (prefix_len + match_len < n and
                   arr[start + match_len] == arr[prefix_len + match_len]):
                match_len += 1
                if start + match_len >= prefix_len:
                    break
            if match_len > max_match:
                max_match = match_len

        prefix_len += max(1, max_match + 1)
        complexity += 1

    return complexity


def lempel_ziv_complexity(s: str) -> int:
    """Calculate Lempel-Ziv complexity (LZ76).

    Wrapper that converts string to numpy array and calls JIT version.
    """
    if len(s) == 0:
        return 0
    # Fast vectorized conversion
    arr = np.frombuffer(s.encode('ascii'), dtype=np.uint8) - ord('0')
    return _lempel_ziv_complexity_numba(arr)


def _lz_worker(segment: str, result_queue) -> None:
    """Worker function for LZ computation in separate process.

    Must be at module level for multiprocessing pickle to work on Windows.
    """
    try:
        lz = lempel_ziv_complexity(segment)
        result_queue.put(('ok', lz))
    except Exception as e:
        result_queue.put(('error', str(e)))


def multiscale_lz_analysis(bits: str, max_bits: int = 100000,
                           verbose: bool = True) -> Dict:
    """
    Analyze Lempel-Ziv complexity at multiple scales.

    Uses THREE scale types to test for φ-scaling:
    1. Geometric φ-scales: 1000, 1000×φ, 1000×φ², ... (tests if φ is preferred)
    2. Power-of-2 scales: 1024, 2048, 4096, ... (standard control)
    3. Arbitrary scales: 1000, 2000, 5000, ... (robustness check)

    If φ is hidden in the structure, LZ complexity ratios at φ-scales
    should show lower variance than at other scales.

    Args:
        bits: Binary string
        max_bits: Maximum bits to analyze
        verbose: Print progress

    Returns:
        Multi-scale LZ analysis results with φ vs 2^n comparison
    """
    if verbose:
        print(f"\n🧬 Multi-Scale Lempel-Ziv Complexity")
        print(f"   Data: {len(bits):,} bits (analyzing {min(len(bits), max_bits):,})")

    data = bits[:max_bits]
    n = len(data)

    # Generate three types of scales
    # 1. Geometric φ-scales: 1000 × φ^k
    phi_scales = []
    scale = 1000
    while scale <= n and scale <= max_bits:
        phi_scales.append(int(scale))
        scale *= PHI

    # 2. Power-of-2 scales
    pow2_scales = [2**k for k in range(10, 20) if 2**k <= n and 2**k <= max_bits]

    # 3. Arbitrary scales (original)
    arb_scales = [1000, 2000, 5000, 10000, 20000, 50000]
    arb_scales = [s for s in arb_scales if s <= n]

    if verbose:
        print(f"   φ-scales ({len(phi_scales)}): {phi_scales[:5]}...")
        print(f"   2^n-scales ({len(pow2_scales)}): {pow2_scales[:5]}...")
        print(f"   Arbitrary ({len(arb_scales)}): {arb_scales}")

    # Adaptive timeout based on scale size (aggressive to avoid eternal waits)
    def get_timeout_for_scale(scale: int) -> int:
        """Return timeout in seconds based on scale size."""
        if scale < 1_000_000:       # <1M bits
            return 15 * 60          # 15 minutes
        elif scale < 10_000_000:    # 1M-10M bits
            return 10 * 60          # 10 minutes
        else:                       # >10M bits
            return 5 * 60           # 5 minutes

    def analyze_scales(scales: List[int], name: str, show_progress: bool = False) -> Dict:
        """Analyze LZ at a set of scales with optional progress bar and timeout."""
        import multiprocessing as mp

        results = {}
        completed_scales = 0
        total_scales = len(scales)
        timed_out_scales = []

        # Progress display
        if show_progress:
            print(f"   Analyzing {name} ({total_scales} scales)...")

        for idx, scale in enumerate(scales):
            scale_start = time.time()
            scale_timeout = get_timeout_for_scale(scale)

            # Show which scale we're working on
            if show_progress and scale > 100000:  # Only for large scales
                timeout_min = scale_timeout // 60
                print(f"      Scale {idx+1}/{total_scales}: {scale:,} bits (max {timeout_min}min)...", end=" ", flush=True)

            try:
                segment = data[:scale]

                # Use multiprocessing.Process directly - can be killed on Windows
                # _lz_worker is at module level so it can be pickled
                result_queue = mp.Queue()
                proc = mp.Process(target=_lz_worker, args=(segment, result_queue))
                proc.start()
                proc.join(timeout=scale_timeout)

                if proc.is_alive():
                    # Timeout - kill the process forcefully
                    proc.terminate()
                    proc.join(timeout=5)  # Wait up to 5s for termination
                    if proc.is_alive():
                        proc.kill()  # Force kill if still alive
                        proc.join(timeout=2)

                    elapsed = time.time() - scale_start
                    timed_out_scales.append({
                        'scale': scale,
                        'index': idx,
                        'elapsed': elapsed
                    })
                    if show_progress:
                        print(f"⏱️ TIMEOUT after {elapsed/60:.1f}min (scale {scale:,})")
                else:
                    # Process completed - get result
                    try:
                        status, value = result_queue.get_nowait()
                        if status == 'ok':
                            lz = value
                            normalized = lz / (scale / math.log2(scale)) if scale > 1 else 0
                            results[scale] = {'raw_lz': lz, 'normalized_lz': float(normalized)}
                            completed_scales += 1

                            elapsed = time.time() - scale_start
                            if show_progress and scale > 100000:
                                print(f"✓ ({elapsed:.1f}s)")
                        else:
                            if show_progress:
                                print(f"❌ Error: {value}")
                    except Exception:
                        if show_progress:
                            print(f"❌ No result from process")

            except Exception as e:
                if show_progress:
                    print(f"❌ Error at scale {scale}: {e}")
                continue

        # Calculate ratios
        sorted_scales = sorted(results.keys())
        lz_ratios = []
        for i in range(len(sorted_scales) - 1):
            s1, s2 = sorted_scales[i], sorted_scales[i + 1]
            if results[s1]['normalized_lz'] > 0:
                lz_ratios.append(results[s2]['normalized_lz'] / results[s1]['normalized_lz'])

        # φ-distances
        phi_distances = [min(abs(r - PHI), abs(r - 1/PHI)) for r in lz_ratios]
        mean_dist = float(np.mean(phi_distances)) if phi_distances else float('nan')

        # Calculate completion percentage
        completion_pct = 100.0 * completed_scales / total_scales if total_scales > 0 else 0

        # Summary
        if show_progress and timed_out_scales:
            print(f"   ⚠️ {name}: {completed_scales}/{total_scales} scales completed ({completion_pct:.0f}%)")
            print(f"      Timed out: {len(timed_out_scales)} scales")

        return {
            'by_scale': results,
            'lz_ratios': lz_ratios,
            'phi_distances': phi_distances,
            'mean_phi_distance': mean_dist,
            'ratio_variance': float(np.var(lz_ratios)) if lz_ratios else 0.0,
            # Completion metadata
            'completion': {
                'total_scales': total_scales,
                'completed_scales': completed_scales,
                'completion_pct': completion_pct,
                'timed_out': timed_out_scales,
                'is_complete': len(timed_out_scales) == 0
            }
        }

    # Analyze each scale type with progress bars
    phi_results = analyze_scales(phi_scales, "φ-scales", show_progress=verbose)
    phi_complete = phi_results.get('completion', {}).get('is_complete', True)
    if verbose:
        status = "✓" if phi_complete else "⚠️ partial"
        print(f"   φ-scales: {status} (φ-dist={phi_results['mean_phi_distance']:.3f})")

    pow2_results = analyze_scales(pow2_scales, "2^n-scales", show_progress=verbose)
    pow2_complete = pow2_results.get('completion', {}).get('is_complete', True)
    if verbose:
        status = "✓" if pow2_complete else "⚠️ partial"
        print(f"   2^n-scales: {status} (φ-dist={pow2_results['mean_phi_distance']:.3f})")

    arb_results = analyze_scales(arb_scales, "arbitrary", show_progress=verbose)
    arb_complete = arb_results.get('completion', {}).get('is_complete', True)
    if verbose:
        status = "✓" if arb_complete else "⚠️ partial"
        print(f"   arbitrary: {status} (φ-dist={arb_results['mean_phi_distance']:.3f})")

    # Compare: does φ-scale show better φ-alignment?
    # Lower variance in lz_ratios at φ-scales suggests φ is "natural"
    phi_preference = 1.0
    if pow2_results['ratio_variance'] > 0:
        phi_preference = pow2_results['ratio_variance'] / (phi_results['ratio_variance'] + 1e-10)

    # Check overall completion status
    all_complete = phi_complete and pow2_complete and arb_complete

    # Main result uses arbitrary scales (for backward compatibility)
    # but includes comparison data
    return {
        'by_scale': arb_results['by_scale'],
        'scale_ratios': [arb_scales[i+1]/arb_scales[i] for i in range(len(arb_scales)-1)],
        'lz_ratios': arb_results['lz_ratios'],
        'phi_distances': arb_results['phi_distances'],
        'mean_phi_distance': arb_results['mean_phi_distance'],
        'phi_detected': arb_results['mean_phi_distance'] < 0.3 if not math.isnan(arb_results['mean_phi_distance']) else False,
        # New: comparison between scale types
        'scale_comparison': {
            'phi_scales': {
                'scales': phi_scales,
                'mean_phi_distance': phi_results['mean_phi_distance'],
                'ratio_variance': phi_results['ratio_variance'],
                'completion': phi_results.get('completion', {})
            },
            'pow2_scales': {
                'scales': pow2_scales,
                'mean_phi_distance': pow2_results['mean_phi_distance'],
                'ratio_variance': pow2_results['ratio_variance'],
                'completion': pow2_results.get('completion', {})
            },
            'arbitrary_scales': {
                'completion': arb_results.get('completion', {})
            },
            'phi_preference': phi_preference,  # >1 means φ-scales more consistent
            'interpretation': 'φ-scales more consistent' if phi_preference > 1.2 else 'No φ-preference detected'
        },
        # Overall completion status
        'analysis_complete': all_complete,
        'analysis_note': None if all_complete else 'Some scales timed out (adaptive 5-15min limits). Results based on completed scales only.'
    }


def bootstrap_lz_confidence_interval(bits: str, max_bits: int = 100000,
                                      n_bootstrap: int = 100,
                                      confidence: float = 0.95,
                                      block_size: int = 1000,
                                      verbose: bool = True) -> Dict:
    """
    Calculate bootstrap confidence intervals for LZ φ-distance.

    Uses BLOCK BOOTSTRAP to preserve local structure in the sequence.
    This is critical because LZ complexity depends on sequential patterns,
    and naive resampling destroys the structure we're trying to measure.

    Args:
        bits: Binary string
        max_bits: Maximum bits to analyze
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default 95%)
        block_size: Size of contiguous blocks for resampling (preserves structure)
        verbose: Print progress

    Returns:
        Bootstrap statistics with CI for mean_phi_distance
    """
    if verbose:
        print(f"\n📊 Block Bootstrap CI (n={n_bootstrap}, {confidence*100:.0f}%, block={block_size})")

    data = bits[:max_bits]
    n = len(data)

    # Scales to analyze
    scales = [1000, 2000, 5000, 10000, 20000, 50000]
    scales = [s for s in scales if s <= n]

    # Prepare blocks for block bootstrap
    n_blocks = n // block_size
    if n_blocks < 10:
        # Fall back to smaller blocks if data is small
        block_size = max(100, n // 20)
        n_blocks = n // block_size

    if verbose:
        print(f"   Data: {n:,} bits → {n_blocks} blocks of {block_size} bits")

    phi_distances_samples = []

    iterator = range(n_bootstrap)
    if verbose and TQDM_AVAILABLE:
        iterator = tqdm(iterator, desc="   Bootstrap", unit="sample")

    for _ in iterator:
        # BLOCK BOOTSTRAP: resample contiguous blocks (preserves local structure)
        block_indices = np.random.choice(n_blocks, size=n_blocks, replace=True)
        sample_parts = [data[i*block_size:(i+1)*block_size] for i in block_indices]
        sample = ''.join(sample_parts)

        # Calculate LZ at each scale
        lz_values = {}
        for scale in scales:
            segment = sample[:scale]
            lz = lempel_ziv_complexity(segment)
            normalized = lz / (scale / math.log2(scale)) if scale > 1 else 0
            lz_values[scale] = normalized

        # Calculate ratios
        sorted_scales = sorted(lz_values.keys())
        lz_ratios = []
        for i in range(len(sorted_scales) - 1):
            s1, s2 = sorted_scales[i], sorted_scales[i + 1]
            if lz_values[s1] > 0:
                lz_ratios.append(lz_values[s2] / lz_values[s1])

        # Calculate phi distance
        if lz_ratios:
            phi_dists = [min(abs(r - PHI), abs(r - 1/PHI)) for r in lz_ratios]
            phi_distances_samples.append(np.mean(phi_dists))

    # Calculate CI
    alpha = 1 - confidence
    lower = np.percentile(phi_distances_samples, alpha/2 * 100)
    upper = np.percentile(phi_distances_samples, (1 - alpha/2) * 100)
    mean = np.mean(phi_distances_samples)
    std = np.std(phi_distances_samples)

    if verbose:
        print(f"   Mean φ-distance: {mean:.4f} ± {std:.4f}")
        print(f"   {confidence*100:.0f}% CI: [{lower:.4f}, {upper:.4f}]")

    return {
        'mean_phi_distance': float(mean),
        'std': float(std),
        'ci_lower': float(lower),
        'ci_upper': float(upper),
        'confidence_level': confidence,
        'n_bootstrap': n_bootstrap,
        'block_size': block_size,
        'n_blocks': n_blocks,
        'samples': phi_distances_samples
    }


def test_multiple_constants(lz_ratios: List[float], verbose: bool = True) -> Dict:
    """
    Test LZ ratios against multiple mathematical constants.

    Honestly reports which constant is closest to the observed ratios.
    Includes z-score and p-value for statistical significance.

    Args:
        lz_ratios: List of LZ complexity ratios between scales
        verbose: Print results

    Returns:
        Comparison with multiple constants with statistical tests
    """
    if not lz_ratios:
        return {'error': 'No LZ ratios provided'}

    # Constants to test
    constants = {
        'φ (Golden Ratio)': PHI,
        '1/φ': 1/PHI,
        '√2': math.sqrt(2),
        '1/√2': 1/math.sqrt(2),
        '√3': math.sqrt(3),
        '1/√3': 1/math.sqrt(3),
        'e': math.e,
        '1/e': 1/math.e,
        'π/3': math.pi/3,
        'ln(2)': math.log(2),
        '2': 2.0,
        '0.5': 0.5,
        '1.0 (random)': 1.0
    }

    # Calculate statistics
    mean_ratio = float(np.mean(lz_ratios))
    std_ratio = float(np.std(lz_ratios, ddof=1)) if len(lz_ratios) > 1 else 0.0
    n_samples = len(lz_ratios)
    se_ratio = std_ratio / math.sqrt(n_samples) if n_samples > 0 else 0.0

    # Test each constant with z-score
    results = []
    for name, value in constants.items():
        distance = abs(mean_ratio - value)

        # Z-score: how many standard errors away is this constant?
        z_score = distance / se_ratio if se_ratio > 0 else float('inf')

        # Two-tailed p-value (approximate using normal distribution)
        # p = 2 * (1 - Φ(|z|)) where Φ is standard normal CDF
        # Use approximation: p ≈ exp(-0.5 * z²) * 0.4 for large z
        if z_score < 6:
            # More accurate for small z
            from math import erfc
            p_value = erfc(z_score / math.sqrt(2))
        else:
            p_value = 0.0  # Effectively zero

        results.append({
            'constant': name,
            'value': value,
            'distance': distance,
            'z_score': z_score,
            'p_value': p_value
        })

    # Sort by distance
    results.sort(key=lambda x: x['distance'])

    # Find 1.0 (random) distance for comparison
    random_distance = next((r['distance'] for r in results if r['constant'] == '1.0 (random)'), None)

    # Effect size: how much closer to best match vs random?
    best_distance = results[0]['distance']
    effect_size = (random_distance - best_distance) / std_ratio if std_ratio > 0 and random_distance else 0.0

    if verbose:
        print(f"\n🔬 Multiple Constants Test")
        print(f"   Mean LZ ratio: {mean_ratio:.4f} ± {std_ratio:.4f} (n={n_samples})")
        print(f"   Closest matches:")
        for i, r in enumerate(results[:5]):
            marker = "🎯" if i == 0 else "  "
            sig = "***" if r['p_value'] < 0.001 else "**" if r['p_value'] < 0.01 else "*" if r['p_value'] < 0.05 else ""
            print(f"   {marker} {r['constant']}: {r['value']:.4f} (dist={r['distance']:.4f}, z={r['z_score']:.1f}{sig})")

        # Statistical interpretation
        if results[0]['p_value'] < 0.05:
            print(f"   📊 Best match is statistically significant (p<0.05)")
        if random_distance and best_distance < random_distance * 0.5:
            print(f"   📊 Effect size: {effect_size:.2f}σ closer to {results[0]['constant']} than random")

    return {
        'mean_ratio': mean_ratio,
        'std_ratio': std_ratio,
        'n_samples': n_samples,
        'rankings': results,
        'best_match': results[0]['constant'],
        'best_match_value': results[0]['value'],
        'best_match_distance': results[0]['distance'],
        'best_match_p_value': results[0]['p_value'],
        'effect_size_vs_random': effect_size,
        'phi_rank': next((i+1 for i, r in enumerate(results) if 'φ' in r['constant']), None),
        'statistically_significant': results[0]['p_value'] < 0.05
    }


def multiscale_lz_analysis_subsampled(subsamples: Dict, verbose: bool = True) -> Dict:
    """
    Analyze Lempel-Ziv complexity using multi-scale subsamples.

    Calculates LZ at each subsample level, representing patterns
    at different real bit scales (1 bit to 1M bits per sample).

    Args:
        subsamples: Output from multiscale_subsample()
        verbose: Print progress

    Returns:
        Multi-scale LZ analysis results covering all scales
    """
    samples = subsamples['samples']
    metadata = subsamples['metadata']

    if verbose:
        print(f"\n🧬 Multi-Scale Lempel-Ziv Complexity (Subsampled)")
        print(f"   Total bits represented: {metadata['total_bits']:,}")

    level_multipliers = {
        'fine': 1,
        'medium': 100,
        'large': 10_000,
        'huge': 1_000_000
    }

    all_results = {}
    all_normalized = []
    all_real_scales = []

    for level_name in ['fine', 'medium', 'large', 'huge']:
        arr = samples[level_name]
        multiplier = level_multipliers[level_name]

        if len(arr) < 100:
            continue

        if verbose:
            print(f"   {level_name}: {len(arr):,} samples...", end=" ", flush=True)

        # Calculate LZ on subsets of increasing size
        subset_sizes = [100, 500, 1000, 5000, 10000, 50000]
        subset_sizes = [s for s in subset_sizes if s <= len(arr)]

        level_results = {}
        for size in subset_sizes:
            # Convert numpy array to string for LZ function
            segment_str = ''.join(str(b) for b in arr[:size])
            lz = lempel_ziv_complexity(segment_str)
            normalized = lz / (size / math.log2(size)) if size > 1 else 0

            real_scale = size * multiplier
            level_results[real_scale] = {
                'raw_lz': lz,
                'normalized_lz': float(normalized),
                'samples': size,
                'level': level_name
            }
            all_normalized.append(float(normalized))
            all_real_scales.append(real_scale)

        all_results.update(level_results)

        if verbose:
            print(f"✓ ({len(subset_sizes)} scales)")

    # Calculate scale ratios
    sorted_scales = sorted(all_results.keys())
    scale_ratios = []
    lz_ratios = []

    for i in range(len(sorted_scales) - 1):
        s1, s2 = sorted_scales[i], sorted_scales[i + 1]
        scale_ratios.append(s2 / s1)
        if all_results[s1]['normalized_lz'] > 0:
            lz_ratios.append(all_results[s2]['normalized_lz'] / all_results[s1]['normalized_lz'])

    # Check if LZ ratios are close to φ or 1/φ
    phi_distances = [min(abs(r - PHI), abs(r - 1/PHI)) for r in lz_ratios]
    mean_phi_distance = float(np.mean(phi_distances)) if phi_distances else float('nan')

    min_scale = min(sorted_scales) if sorted_scales else 0
    max_scale = max(sorted_scales) if sorted_scales else 0

    if verbose:
        print(f"   Scale range: {min_scale:,} - {max_scale:,} bits")
        print(f"   Mean φ-distance: {mean_phi_distance:.3f}")

    return {
        'by_scale': all_results,
        'scale_ratios': scale_ratios,
        'lz_ratios': lz_ratios,
        'phi_distances': phi_distances,
        'mean_phi_distance': mean_phi_distance,
        'phi_detected': mean_phi_distance < 0.3 if not math.isnan(mean_phi_distance) else False,
        'min_scale_bits': min_scale,
        'max_scale_bits': max_scale,
        'multiscale': True
    }


# =============================================================================
# Main Analysis Runner
# =============================================================================

def run_deep_analysis(
    variant: str,
    iteration: int,
    analyses: Optional[List[str]] = None,
    max_bits: int = 1_000_000_000,
    verbose: bool = True,
    output_path: Optional[str] = None,
    use_multiscale: bool = False,
    n_bootstrap: int = 100
) -> Dict:
    """
    Run deep analysis on a variant.

    Args:
        variant: Variant code (B, F, etc.)
        iteration: Iteration number
        analyses: List of analyses to run ('wavelet', 'recurrence', 'lz', 'all')
        max_bits: Maximum bits to analyze (default 1G for publication)
        verbose: Print progress
        output_path: Optional path to save JSON results
        use_multiscale: If True, use Nyquist-based subsampling to analyze
                       the FULL dataset (100G+ bits) with ~12M samples in RAM
        n_bootstrap: Number of bootstrap samples for CI (default 100 for publication)

    Returns:
        Combined analysis results
    """
    import time
    start_time = time.time()

    if analyses is None or 'all' in analyses:
        analyses = ['wavelet', 'recurrence', 'lz']

    # Load data
    data_dir = f"results/level0/phi_snapshots/var_{variant}"
    if not os.path.exists(data_dir):
        data_dir = f"results/level0/var_{variant}"
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found for variant {variant}")

    if verbose:
        print(f"\n{'='*60}")
        print(f"🔬 DEEP ANALYSIS — Variant {variant}, Iteration {iteration}")
        print(f"{'='*60}")
        print(f"   📁 Data directory: {data_dir}")
        print(f"   📊 Analyses: {', '.join(analyses)}")
        if use_multiscale:
            print(f"   🌐 Mode: MULTISCALE (Nyquist subsampling for 100G+ bits)")
        else:
            print(f"   🔢 Max bits: {max_bits:,}")
        print(f"   ⚡ JIT acceleration: {'Numba' if NUMBA_AVAILABLE else 'Not available'}")
        print(f"   📊 Progress bars: {'tqdm' if TQDM_AVAILABLE else 'Basic'}")
        print(f"")
        print(f"📥 Loading Φ data...", flush=True)

    load_start = time.time()

    # Try streaming loader with progress bar (for v33 format)
    struct_path = Path(data_dir) / f"phi_iter{iteration}.struct.gz"
    if struct_path.exists():
        # Load metadata first to get accurate size for progress bar
        metadata = load_metadata(data_dir, iteration)
        # Use sequence_length as safe upper bound (structural length >= observable length)
        total_chars_hint = metadata.get('sequence_length')

        # Determine how many chars to load:
        # - In multiscale mode: use STREAMING (don't load full sequence!)
        # - In standard mode: only load max_bits (saves memory!)
        if use_multiscale:
            # STREAMING MODE: Don't load sequence into RAM
            # We'll use streaming_multiscale_subsample later
            phi_observable = ""  # Placeholder - not used in streaming mode
            load_max_chars = 0
        else:
            # Load slightly more than max_bits to account for structural overhead
            # (parentheses in structural format don't count as observable bits)
            load_max_chars = max_bits * 2  # Safety margin for parentheses

            # Use streaming loader with progress bar
            phi_observable, _ = load_phi_for_agents(
                str(struct_path),
                max_chars=load_max_chars,
                observable_only=True,
                show_progress=verbose,
                total_chars_hint=load_max_chars
            )
    else:
        # Fallback to standard loader (v32 format or other)
        _, phi_observable, metadata = load_phi_for_level1(
            data_dir, iteration,
            return_structural=False,
            return_observable=True,
            return_metadata=True
        )

    load_elapsed = time.time() - load_start

    # Prepare subsamples if using multiscale mode (STREAMING - before other output)
    subsamples = None
    total_bits = 0
    if use_multiscale:
        if verbose:
            print(f"\n{'─'*60}")
        # Use STREAMING subsampling - doesn't load full sequence into RAM
        from utils.streaming_subsample import streaming_multiscale_subsample
        subsamples = streaming_multiscale_subsample(
            str(struct_path),
            verbose=verbose,
            total_chars_hint=total_chars_hint
        )
        total_bits = subsamples['metadata']['total_bits']
        phi_observable = ""  # Keep empty - we don't need it
    else:
        total_bits = len(phi_observable)
        if verbose:
            if total_bits >= 1_000_000_000:
                size_str = f"{total_bits / 1_000_000_000:.2f}G"
            elif total_bits >= 1_000_000:
                size_str = f"{total_bits / 1_000_000:.2f}M"
            elif total_bits >= 1_000:
                size_str = f"{total_bits / 1_000:.2f}k"
            else:
                size_str = str(total_bits)
            print(f"   ✓ Loaded {size_str} bits in {load_elapsed:.1f}s")

            # Show metadata info
            if metadata:
                fmt = metadata.get('format', 'unknown')
                print(f"   📋 Format: {fmt}")

    results = {
        'variant': variant,
        'iteration': iteration,
        'total_bits': total_bits,
        'max_bits_analyzed': min(max_bits, total_bits) if not use_multiscale else total_bits,
        'phi_target': PHI,
        'mode': 'multiscale_streaming' if use_multiscale else 'standard'
    }

    if use_multiscale and subsamples:
        results['subsampling_metadata'] = subsamples['metadata']

    # Run requested analyses with timing
    if 'wavelet' in analyses:
        if verbose:
            print(f"\n{'─'*60}")
        analysis_start = time.time()

        if use_multiscale and subsamples:
            results['wavelet'] = wavelet_analysis_multiscale(subsamples, verbose)
        else:
            results['wavelet'] = wavelet_analysis(phi_observable, max_bits, verbose)

        if verbose:
            elapsed = time.time() - analysis_start
            status = "✅ DETECTED" if results['wavelet'].get('phi_detected') else "❌ Not detected"
            print(f"   → φ-band {status} (pref={results['wavelet']['phi_preference']:.3f}) [{elapsed:.1f}s]")
            if results['wavelet'].get('multiscale'):
                print(f"   → Scale range: {results['wavelet']['min_scale_bits']:,} - {results['wavelet']['max_scale_bits']:,} bits")

    if 'recurrence' in analyses:
        if verbose:
            print(f"\n{'─'*60}")
        analysis_start = time.time()
        # Recurrence is O(n²), always use small sample
        recurrence_bits = min(10000, max_bits // 10 if not use_multiscale else 10000)
        # In streaming mode, use MEDIUM samples (every 100th bit) for representative coverage
        # This captures dynamics across the ENTIRE sequence, not just the first 10k bits
        if use_multiscale and subsamples:
            medium_samples = subsamples['samples']['medium']
            # Take evenly spaced samples from medium to get ~10k representative bits
            step = max(1, len(medium_samples) // recurrence_bits)
            representative_samples = medium_samples[::step][:recurrence_bits]
            phi_for_recurrence = ''.join(str(b) for b in representative_samples)
            if verbose:
                print(f"   Using {len(representative_samples):,} representative samples from full sequence")
        else:
            phi_for_recurrence = phi_observable
        results['recurrence'] = recurrence_analysis(phi_for_recurrence, recurrence_bits, verbose=verbose)
        if verbose:
            elapsed = time.time() - analysis_start
            r = results['recurrence']
            if 'error' not in r:
                status = "✅ DETERMINISTIC" if r.get('is_deterministic') else "❌ Stochastic"
                print(f"   → {status} (det={r['determinism']:.3f}) [{elapsed:.1f}s]")
            else:
                print(f"   → ⚠️ {r['error']} [{elapsed:.1f}s]")

    if 'lz' in analyses:
        if verbose:
            print(f"\n{'─'*60}")
        analysis_start = time.time()

        if use_multiscale and subsamples:
            results['lz'] = multiscale_lz_analysis_subsampled(subsamples, verbose)
        else:
            results['lz'] = multiscale_lz_analysis(phi_observable, max_bits, verbose)

        # Add multiple constants test for rigor (fast)
        lz_ratios = results['lz'].get('lz_ratios', [])
        if lz_ratios:
            results['lz']['constants_test'] = test_multiple_constants(lz_ratios, verbose)

        # Bootstrap confidence intervals if requested
        if n_bootstrap > 0 and not use_multiscale:
            if verbose:
                print(f"\n   Running bootstrap ({n_bootstrap} samples)...")
            bootstrap_results = bootstrap_lz_confidence_interval(
                phi_observable, max_bits, n_bootstrap, verbose=verbose
            )
            results['lz']['bootstrap'] = bootstrap_results

            # Add interpretation
            ci_lower = bootstrap_results['ci_lower']
            ci_upper = bootstrap_results['ci_upper']
            if verbose:
                # Check if φ-distance CI excludes random (≈0.38 for random)
                random_phi_dist = abs(1.0 - PHI)  # ≈ 0.382
                if ci_upper < random_phi_dist * 0.8:
                    print(f"   📊 CI excludes random: [{ci_lower:.3f}, {ci_upper:.3f}] << {random_phi_dist:.3f}")

        if verbose:
            elapsed = time.time() - analysis_start
            lz = results['lz']
            status = "✅ DETECTED" if lz.get('phi_detected') else "❌ Not detected"
            dist = lz.get('mean_phi_distance', float('nan'))
            print(f"   → φ-scaling {status} (φ-dist={dist:.3f}) [{elapsed:.1f}s]")
            if lz.get('multiscale'):
                print(f"   → Scale range: {lz['min_scale_bits']:,} - {lz['max_scale_bits']:,} bits")

    # Summary
    total_elapsed = time.time() - start_time
    if verbose:
        print(f"\n{'='*60}")
        print(f"📊 SUMMARY — Variant {variant}, Iteration {iteration}")
        print(f"{'='*60}")
        print(f"   Total time: {total_elapsed:.1f}s")
        if use_multiscale:
            meta = results.get('subsampling_metadata', {})
            print(f"   Mode: MULTISCALE STREAMING (Nyquist-based subsampling)")
            print(f"   Total bits: {meta.get('total_bits', total_bits):,}")
            print(f"   Samples in RAM: {meta.get('total_samples', 'N/A'):,} ({meta.get('memory_bytes', 0) / 1_000_000:.1f} MB)")
            print(f"   ℹ️  Nyquist theorem: patterns of scale N require sampling at 2/N frequency")
        else:
            print(f"   Bits analyzed: {min(max_bits, len(phi_observable)):,} / {len(phi_observable):,}")
        print(f"")

        if 'wavelet' in results:
            w = results['wavelet']
            status = "✅ DETECTED" if w.get('phi_detected') else "❌ Not detected"
            print(f"   Wavelet φ-band: {status} (preference={w['phi_preference']:.3f})")

        if 'recurrence' in results:
            r = results['recurrence']
            if 'error' not in r:
                status = "✅ DETERMINISTIC" if r.get('is_deterministic') else "❌ Stochastic"
                print(f"   Recurrence: {status} (det={r['determinism']:.3f})")
            else:
                print(f"   Recurrence: ⚠️ {r['error']}")

        if 'lz' in results:
            lz = results['lz']
            status = "✅ DETECTED" if lz.get('phi_detected') else "❌ Not detected"
            dist = lz.get('mean_phi_distance', float('nan'))
            print(f"   LZ φ-scaling: {status} (φ-dist={dist:.3f})")

    if output_path:
        from datetime import datetime

        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(k): convert(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert(x) for x in obj]
            elif hasattr(obj, '__class__') and obj.__class__.__module__ == 'numpy':
                # Catch any other numpy type
                return str(obj)
            return obj

        # Add metadata for traceability
        results['metadata'] = {
            'script': 'level1_deep_analysis.py',
            'generated_at': datetime.now().isoformat(),
            'max_bits_analyzed': max_bits,
            'analyses_run': analyses if analyses else ['all']
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(convert(results), f, indent=2, default=str)
        if verbose:
            print(f"\n📁 Results saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Deep Analysis Tools — Wavelet, Recurrence, LZ complexity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single variant:
    python level1_deep_analysis.py -v B -i 15 --max-bits 100000

  Multiple variants (comparative analysis):
    python level1_deep_analysis.py --variants A,B,F -i 15

  Quick mode for development (1M bits, no bootstrap):
    python level1_deep_analysis.py -v B -i 15 --quick

  Multiscale mode (analyze ALL bits using Nyquist subsampling):
    python level1_deep_analysis.py -v B -i 15 --multiscale

  The --multiscale flag uses Nyquist-based subsampling to analyze patterns
  at ALL scales (from 1 bit to 100G+ bits) using only ~12M samples in RAM.
        """
    )
    parser.add_argument("--variant", "-v", default="B", help="Single variant to analyze")
    parser.add_argument("--variants", type=str, default=None,
                        help="Multiple variants (comma-separated, e.g., A,B,F,J,K,L)")
    parser.add_argument("--iteration", "-i", type=int, default=15, help="Iteration number")
    parser.add_argument("--max-bits", "-m", type=int, default=1_000_000_000,
                        help="Max bits to analyze (default: 1G for publication)")
    parser.add_argument("--analysis", "-a", type=str, default="all",
                        help="Analysis type: wavelet, recurrence, lz, all (comma-separated)")
    parser.add_argument("--multiscale", "--ms", action="store_true",
                        help="Use Nyquist-based subsampling to analyze ALL bits (100G+ capable)")
    parser.add_argument("--bootstrap", "-b", type=int, default=100,
                        help="Number of bootstrap samples for CI (default: 100 for publication)")
    parser.add_argument("--quick", "-q", action="store_true",
                        help="Quick mode: 1M bits, no bootstrap (for development)")
    parser.add_argument("--compare", "-c", type=str, default=None,
                        help="Compare with another variant (e.g., --compare F)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output JSON file (default: auto-generated in results/level1/analysis/)")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save results to file")

    args = parser.parse_args()

    # Quick mode overrides for development
    if args.quick:
        args.max_bits = 1_000_000  # 1M instead of 1G
        args.bootstrap = 0         # No bootstrap

    analyses = args.analysis.split(",") if args.analysis != "all" else ["all"]

    # Determine variants to analyze
    if args.variants:
        variants = [v.strip().upper() for v in args.variants.split(",")]
    else:
        variants = [args.variant.upper()]

    # Helper to format bits nicely
    def format_bits(n):
        if n >= 1_000_000_000:
            return f"{n // 1_000_000_000}G"
        elif n >= 1_000_000:
            return f"{n // 1_000_000}M"
        elif n >= 1_000:
            return f"{n // 1_000}k"
        return str(n)

    sample_info = "multiscale" if args.multiscale else format_bits(args.max_bits)

    # Multi-variant mode
    if len(variants) > 1:
        print(f"\n{'='*70}")
        print(f"🔬 COMPARATIVE DEEP ANALYSIS — {len(variants)} variants")
        print(f"{'='*70}")
        print(f"   Variants: {', '.join(variants)}")
        print(f"   Iteration: {args.iteration}")
        print(f"   Max bits: {args.max_bits:,}")
        print(f"   Analyses: {', '.join(analyses if 'all' not in analyses else ['wavelet', 'recurrence', 'lz'])}")
        print()

        all_results = {}
        for variant in variants:
            try:
                # Generate output path for each variant
                if args.no_save:
                    output_path = None
                else:
                    from utils.file_saver import get_output_path
                    filename = f"deep_analysis_var_{variant}_iter{args.iteration}_{sample_info}.json"
                    output_path = str(get_output_path(1, "analysis", filename))

                results = run_deep_analysis(
                    variant, args.iteration,
                    analyses=analyses,
                    max_bits=args.max_bits,
                    output_path=output_path,
                    use_multiscale=args.multiscale,
                    n_bootstrap=args.bootstrap
                )
                all_results[variant] = results
            except FileNotFoundError as e:
                print(f"⚠️  Variant {variant}: {e}")
                all_results[variant] = {'error': str(e)}

        # Comparative summary table
        print(f"\n{'='*70}")
        print(f"📊 COMPARATIVE SUMMARY — Iteration {args.iteration}")
        print(f"{'='*70}")

        # Header
        header = "   Variant  │ Type     │ LZ Ratio │ Best Match      │ φ-dist"
        print(header)
        print(f"   {'─'*60}")

        for variant in variants:
            r = all_results.get(variant, {})
            if 'error' in r:
                print(f"   {variant:^7}  │ {'ERROR':^8} │ {'─':^8} │ {'─':^15} │ {'─':^6}")
                continue

            # Determine type
            vtype = "Control" if variant in ('A', 'J', 'K', 'L') else "HSI"

            lz = r.get('lz', {})
            ct = lz.get('constants_test', {})
            lz_ratio = ct.get('mean_ratio', float('nan'))
            best_match = ct.get('best_match', '─')[:15]
            phi_dist = lz.get('mean_phi_distance', float('nan'))

            # Emoji based on phi detection
            marker = "✅" if lz.get('phi_detected') else "  "

            print(f"   {variant:^7}  │ {vtype:^8} │ {lz_ratio:^8.3f} │ {best_match:^15} │ {phi_dist:^6.3f} {marker}")

        print()

        # Statistical comparison if we have both HSI and control variants
        hsi_variants = [v for v in variants if v not in ('A', 'J', 'K', 'L') and 'error' not in all_results.get(v, {})]
        ctrl_variants = [v for v in variants if v in ('A', 'J', 'K', 'L') and 'error' not in all_results.get(v, {})]

        if hsi_variants and ctrl_variants:
            hsi_ratios = [all_results[v].get('lz', {}).get('constants_test', {}).get('mean_ratio', 0) for v in hsi_variants]
            ctrl_ratios = [all_results[v].get('lz', {}).get('constants_test', {}).get('mean_ratio', 0) for v in ctrl_variants]

            hsi_mean = np.mean(hsi_ratios) if hsi_ratios else 0
            ctrl_mean = np.mean(ctrl_ratios) if ctrl_ratios else 0

            print(f"   📈 HSI mean LZ ratio:     {hsi_mean:.4f} (n={len(hsi_variants)})")
            print(f"   📉 Control mean LZ ratio: {ctrl_mean:.4f} (n={len(ctrl_variants)})")
            print(f"   📊 Difference:            {abs(hsi_mean - ctrl_mean):.4f}")

            # Simple effect size (Cohen's d approximation)
            if len(hsi_ratios) > 1 and len(ctrl_ratios) > 1:
                pooled_std = np.sqrt((np.var(hsi_ratios) + np.var(ctrl_ratios)) / 2)
                if pooled_std > 0:
                    cohens_d = (hsi_mean - ctrl_mean) / pooled_std
                    print(f"   📐 Effect size (Cohen's d): {cohens_d:.2f}")

        # Save comparative report
        if not args.no_save:
            from utils.file_saver import get_output_path
            report = {
                'variants': variants,
                'iteration': args.iteration,
                'max_bits': args.max_bits,
                'analyses': analyses,
                'results': {v: all_results[v] for v in variants}
            }
            report_path = get_output_path(1, "analysis", f"comparative_iter{args.iteration}_{sample_info}.json")
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\n   💾 Comparative report saved: {report_path}")

    else:
        # Single variant mode (original behavior)
        variant = variants[0]

        if args.output:
            output_path = args.output
        elif args.no_save:
            output_path = None
        else:
            from utils.file_saver import get_output_path
            filename = f"deep_analysis_var_{variant}_iter{args.iteration}_{sample_info}.json"
            output_path = str(get_output_path(1, "analysis", filename))

        try:
            results = run_deep_analysis(
                variant, args.iteration,
                analyses=analyses,
                max_bits=args.max_bits,
                output_path=output_path,
                use_multiscale=args.multiscale,
                n_bootstrap=args.bootstrap
            )

            if args.compare:
                print(f"\n{'='*60}")
                print(f"📊 COMPARISON: {variant} vs {args.compare}")
                print(f"{'='*60}")

                results_compare = run_deep_analysis(
                    args.compare, args.iteration,
                    analyses=analyses,
                    max_bits=args.max_bits,
                    use_multiscale=args.multiscale,
                    n_bootstrap=args.bootstrap
                )

                print(f"\n   Metric                    {variant:>10}  {args.compare:>10}")
                print(f"   {'-'*45}")

                if 'wavelet' in results and 'wavelet' in results_compare:
                    print(f"   φ-preference              {results['wavelet']['phi_preference']:>10.3f}  {results_compare['wavelet']['phi_preference']:>10.3f}")

                if 'recurrence' in results and 'recurrence' in results_compare:
                    print(f"   Determinism               {results['recurrence']['determinism']:>10.3f}  {results_compare['recurrence']['determinism']:>10.3f}")

                if 'lz' in results and 'lz' in results_compare:
                    print(f"   LZ φ-distance             {results['lz']['mean_phi_distance']:>10.3f}  {results_compare['lz']['mean_phi_distance']:>10.3f}")

        except FileNotFoundError as e:
            print(f"❌ {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()

