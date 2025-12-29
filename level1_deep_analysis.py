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
                        threshold: float = 0.1, verbose: bool = True) -> Dict:
    """
    Recurrence Plot Analysis — detect temporal dynamics.
    
    Args:
        bits: Binary string
        max_bits: Maximum bits (recurrence is O(n²), keep small!)
        embedding_dim: Embedding dimension for phase space
        delay: Time delay for embedding
        threshold: Distance threshold for recurrence
        verbose: Print progress
    
    Returns:
        Recurrence metrics including determinism and laminarity
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


def multiscale_lz_analysis(bits: str, max_bits: int = 100000,
                           verbose: bool = True) -> Dict:
    """
    Analyze Lempel-Ziv complexity at multiple scales.

    If φ is hidden in the structure, LZ complexity ratios between
    φ-related scales might show special properties.

    Args:
        bits: Binary string
        max_bits: Maximum bits to analyze
        verbose: Print progress

    Returns:
        Multi-scale LZ analysis results
    """
    if verbose:
        print(f"\n🧬 Multi-Scale Lempel-Ziv Complexity")
        print(f"   Data: {len(bits):,} bits (analyzing {min(len(bits), max_bits):,})")

    data = bits[:max_bits]

    # Scales to analyze
    scales = [1000, 2000, 5000, 10000, 20000, 50000]
    scales = [s for s in scales if s <= len(data)]

    if verbose:
        print(f"   Scales: {scales}")

    results = {}

    # Use tqdm for progress bar if available
    if verbose and TQDM_AVAILABLE:
        scale_iter = tqdm(scales, desc="   LZ scales", unit="scale",
                          bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    else:
        scale_iter = scales

    for scale in scale_iter:
        segment = data[:scale]
        lz = lempel_ziv_complexity(segment)

        # Normalized LZ: LZ / (n / log2(n))
        normalized = lz / (scale / math.log2(scale)) if scale > 1 else 0

        results[scale] = {
            'raw_lz': lz,
            'normalized_lz': float(normalized)
        }

        # Update progress bar description with current result
        if verbose and TQDM_AVAILABLE:
            scale_iter.set_postfix({'scale': scale, 'LZ': lz, 'norm': f'{normalized:.3f}'})

    # Calculate scale ratios
    sorted_scales = sorted(results.keys())
    scale_ratios = []
    lz_ratios = []

    for i in range(len(sorted_scales) - 1):
        s1, s2 = sorted_scales[i], sorted_scales[i + 1]
        scale_ratios.append(s2 / s1)
        if results[s1]['normalized_lz'] > 0:
            lz_ratios.append(results[s2]['normalized_lz'] / results[s1]['normalized_lz'])

    # Check if LZ ratios are close to φ or 1/φ
    phi_distances = [min(abs(r - PHI), abs(r - 1/PHI)) for r in lz_ratios]
    mean_phi_distance = float(np.mean(phi_distances)) if phi_distances else float('nan')

    return {
        'by_scale': results,
        'scale_ratios': scale_ratios,
        'lz_ratios': lz_ratios,
        'phi_distances': phi_distances,
        'mean_phi_distance': mean_phi_distance,
        'phi_detected': mean_phi_distance < 0.3 if not math.isnan(mean_phi_distance) else False
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
    max_bits: int = 100000,
    verbose: bool = True,
    output_path: Optional[str] = None,
    use_multiscale: bool = False
) -> Dict:
    """
    Run deep analysis on a variant.

    Args:
        variant: Variant code (B, F, etc.)
        iteration: Iteration number
        analyses: List of analyses to run ('wavelet', 'recurrence', 'lz', 'all')
        max_bits: Maximum bits to analyze (ignored if use_multiscale=True)
        verbose: Print progress
        output_path: Optional path to save JSON results
        use_multiscale: If True, use Nyquist-based subsampling to analyze
                       the FULL dataset (100G+ bits) with ~12M samples in RAM

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
        # - In multiscale mode: load all (Nyquist subsampling handles memory)
        # - In standard mode: only load max_bits (saves memory!)
        if use_multiscale:
            load_max_chars = None  # Load everything
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
            total_chars_hint=total_chars_hint if use_multiscale else load_max_chars
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

    if verbose:
        bits_size = len(phi_observable)
        if bits_size >= 1_000_000_000:
            size_str = f"{bits_size / 1_000_000_000:.2f}G"
        elif bits_size >= 1_000_000:
            size_str = f"{bits_size / 1_000_000:.2f}M"
        elif bits_size >= 1_000:
            size_str = f"{bits_size / 1_000:.2f}k"
        else:
            size_str = str(bits_size)
        print(f"   ✓ Loaded {size_str} bits in {load_elapsed:.1f}s")

        # Show metadata info
        if metadata:
            fmt = metadata.get('format', 'unknown')
            print(f"   📋 Format: {fmt}")

    results = {
        'variant': variant,
        'iteration': iteration,
        'total_bits': len(phi_observable),
        'max_bits_analyzed': min(max_bits, len(phi_observable)) if not use_multiscale else len(phi_observable),
        'phi_target': PHI,
        'mode': 'multiscale' if use_multiscale else 'standard'
    }

    # Prepare subsamples if using multiscale mode
    subsamples = None
    if use_multiscale:
        if verbose:
            print(f"\n{'─'*60}")
        subsamples = multiscale_subsample(phi_observable, verbose)
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
        # Recurrence is O(n²), always use small sample (not affected by multiscale)
        recurrence_bits = min(10000, max_bits // 10 if not use_multiscale else 10000)
        results['recurrence'] = recurrence_analysis(phi_observable, recurrence_bits, verbose=verbose)
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
            print(f"   Mode: MULTISCALE (Nyquist-based subsampling)")
            print(f"   Total bits: {len(phi_observable):,}")
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
  Standard mode (first N bits):
    python level1_deep_analysis.py -v B -i 20 --max-bits 100000

  Multiscale mode (analyze ALL bits using Nyquist subsampling):
    python level1_deep_analysis.py -v B -i 20 --multiscale

  The --multiscale flag uses Nyquist-based subsampling to analyze patterns
  at ALL scales (from 1 bit to 100G+ bits) using only ~12M samples in RAM.
  This is scientifically valid because detecting patterns of scale N only
  requires sampling at frequency 2/N (Nyquist theorem).
        """
    )
    parser.add_argument("--variant", "-v", default="B", help="Variant (B, D, E, F, G, H, I)")
    parser.add_argument("--iteration", "-i", type=int, default=15, help="Iteration number")
    parser.add_argument("--max-bits", "-m", type=int, default=100000,
                        help="Max bits to analyze (ignored if --multiscale)")
    parser.add_argument("--analysis", "-a", type=str, default="all",
                        help="Analysis type: wavelet, recurrence, lz, all (comma-separated)")
    parser.add_argument("--multiscale", "--ms", action="store_true",
                        help="Use Nyquist-based subsampling to analyze ALL bits (100G+ capable)")
    parser.add_argument("--compare", "-c", type=str, default=None,
                        help="Compare with another variant (e.g., --compare F)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output JSON file (default: auto-generated in results/level1/analysis/)")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save results to file")

    args = parser.parse_args()

    analyses = args.analysis.split(",") if args.analysis != "all" else ["all"]

    # Generate default output path if not specified
    if args.output:
        output_path = args.output
    elif args.no_save:
        output_path = None
    else:
        # Auto-generate descriptive filename
        from utils.file_saver import get_output_path
        if args.multiscale:
            sample_info = "multiscale"
        else:
            # Format max_bits nicely (e.g., 100000 -> 100k, 1000000 -> 1M)
            mb = args.max_bits
            if mb >= 1_000_000_000:
                sample_info = f"{mb // 1_000_000_000}G"
            elif mb >= 1_000_000:
                sample_info = f"{mb // 1_000_000}M"
            elif mb >= 1_000:
                sample_info = f"{mb // 1_000}k"
            else:
                sample_info = str(mb)

        filename = f"deep_analysis_var_{args.variant}_iter{args.iteration}_{sample_info}.json"
        output_path = str(get_output_path(1, "analysis", filename))

    try:
        results = run_deep_analysis(
            args.variant, args.iteration,
            analyses=analyses,
            max_bits=args.max_bits,
            output_path=output_path,
            use_multiscale=args.multiscale
        )

        if args.compare:
            print(f"\n{'='*60}")
            print(f"📊 COMPARISON: {args.variant} vs {args.compare}")
            print(f"{'='*60}")

            results_compare = run_deep_analysis(
                args.compare, args.iteration,
                analyses=analyses,
                max_bits=args.max_bits,
                use_multiscale=args.multiscale
            )

            # Compare key metrics
            print(f"\n   Metric                    {args.variant:>10}  {args.compare:>10}")
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

