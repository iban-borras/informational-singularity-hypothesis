"""
🌌 Emergence Index — Metric for Level 2 Potential in HSI

This module calculates the "Emergence Index", a composite metric that estimates
the potential of a Φ sequence to exhibit emergent physical-like behavior.

The index combines three fundamental indicators:
1. Criticality (1/f spectrum): Systems at the "edge of chaos" show 1/f noise
2. Lempel-Ziv Complexity: Neither pure order nor pure chaos = maximum potential
3. Long-Range Coherence: Mutual Information between distant regions

A higher Emergence Index suggests the variant has more potential for generating
complex emergent structures at Level 2 (physical law-like behavior).

Theoretical basis:
- 1/f noise indicates self-organized criticality (Bak et al., 1987)
- Intermediate complexity indicates optimal computational capacity (Langton, 1990)
- Long-range correlations indicate coherent global structure

Author: Iban Borràs with Augment Agent collaboration
Date: December 2024
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from scipy.fft import fft
from scipy.stats import linregress
import warnings


def calculate_power_spectrum_slope(sequence: str, min_freq_idx: int = 1) -> Tuple[float, float]:
    """
    Calculate the power spectrum slope to detect 1/f criticality.
    
    A slope of -1 indicates perfect 1/f noise (criticality).
    A slope of 0 indicates white noise (pure randomness).
    A slope of -2 indicates Brownian motion (over-correlated).
    
    Args:
        sequence: Binary string (e.g., "01101001...")
        min_freq_idx: Minimum frequency index to avoid DC component
        
    Returns:
        Tuple of (slope, r_squared) where slope ~ -1 indicates criticality
    """
    # Convert to numpy array
    bits = np.array([int(b) for b in sequence], dtype=np.float64)
    
    # Subtract mean to remove DC component
    bits = bits - np.mean(bits)
    
    # Apply window to reduce spectral leakage
    window = np.hanning(len(bits))
    bits_windowed = bits * window
    
    # Compute FFT
    spectrum = np.abs(fft(bits_windowed)[:len(bits)//2])**2
    
    # Frequency bins (skip DC)
    freqs = np.arange(min_freq_idx, len(spectrum))
    power = spectrum[min_freq_idx:]
    
    # Filter zero/negative values for log
    mask = power > 0
    if np.sum(mask) < 10:
        return 0.0, 0.0
    
    log_freq = np.log10(freqs[mask])
    log_power = np.log10(power[mask])
    
    # Linear regression in log-log space
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        slope, intercept, r_value, p_value, std_err = linregress(log_freq, log_power)
    
    return slope, r_value**2


def _get_lz76_numba_kernel():
    """
    Get the Numba-compiled LZ76 kernel. Compiled on first call.

    Optimized version using:
    1. Pre-computed position lists for short patterns (length 1-8)
    2. Direct byte comparison for longer patterns
    """
    try:
        from numba import njit
        from numba.typed import Dict
        from numba.core import types

        @njit(cache=True)
        def lz76_kernel_fast(seq, dummy):
            """
            Optimized LZ76 algorithm.

            For random sequences, most matches are very short (1-3 bits).
            We optimize by tracking positions of each byte value for O(1) lookup.

            Args:
                seq: numpy array of uint8 (0 or 1)
                dummy: unused (kept for API compatibility)

            Returns:
                (complexity, 0)
            """
            n = len(seq)
            if n == 0:
                return 0, 0

            complexity = 0
            i = 0

            # For each position, track where we last saw 0 and 1
            # This gives O(1) lookup for length-1 patterns
            last_zero = -1
            last_one = -1

            while i < n:
                if i == 0:
                    complexity += 1
                    if seq[0] == 0:
                        last_zero = 0
                    else:
                        last_one = 0
                    i += 1
                    continue

                # Find longest match using extension from known positions
                match_len = 0
                current_bit = seq[i]

                # Check if current bit exists in history
                start_pos = last_zero if current_bit == 0 else last_one

                if start_pos >= 0:
                    # Found at least length 1, now extend
                    match_len = 1
                    max_possible = min(n - i, i)

                    # Try to extend from all positions that start with current_bit
                    # Scan history for all matching start positions
                    for j in range(i):
                        if seq[j] != current_bit:
                            continue

                        # How far can we extend from position j?
                        ext = 1
                        max_ext = min(i - j, n - i)
                        while ext < max_ext and seq[j + ext] == seq[i + ext]:
                            ext += 1

                        if ext > match_len:
                            match_len = ext
                            if match_len >= max_possible:
                                break

                complexity += 1

                # Update position trackers before moving i
                for k in range(match_len + 1):
                    if i + k < n:
                        if seq[i + k] == 0:
                            last_zero = i + k
                        else:
                            last_one = i + k

                i += match_len + 1

            return complexity, 0

        return lz76_kernel_fast

    except ImportError:
        return None


# Cache the compiled kernel
_LZ76_KERNEL = None


def calculate_lempel_ziv_complexity(sequence: str, max_sample: int = 500_000, verbose: bool = True) -> float:
    """
    Calculate normalized Lempel-Ziv complexity using optimized LZ76 algorithm.

    Returns a value between 0 and 1:
    - 0 = perfectly predictable (e.g., "000000...")
    - 1 = maximally complex (incompressible random)
    - ~0.5 = optimal complexity for emergence

    This implementation uses Numba JIT compilation for 10-100x speedup.
    Falls back to pure Python if Numba is not available.

    Reference: Lempel, A., & Ziv, J. (1976). "On the Complexity of Finite Sequences"

    Args:
        sequence: Binary string
        max_sample: Maximum bits to analyze (default 500K)
        verbose: Print progress updates

    Returns:
        Normalized LZ complexity (0-1)
    """
    global _LZ76_KERNEL

    n = len(sequence)
    if n == 0:
        return 0.0

    # Sample if sequence is too long
    if n > max_sample:
        start = (n - max_sample) // 2
        sequence = sequence[start:start + max_sample]
        n = max_sample
        if verbose:
            print(f"      (Sampling {max_sample:,} bits for LZ complexity)")

    # Try Numba-accelerated version
    if _LZ76_KERNEL is None:
        _LZ76_KERNEL = _get_lz76_numba_kernel()

    if _LZ76_KERNEL is not None:
        # Convert to numpy array for Numba
        seq_array = np.array([ord(c) - ord('0') for c in sequence], dtype=np.uint8)

        if verbose:
            print("      (Using Numba-accelerated LZ76)")

        # Run compiled kernel
        import time
        t0 = time.time()
        complexity, _ = _LZ76_KERNEL(seq_array, n // 10)

        if verbose:
            elapsed = time.time() - t0
            print(f"      LZ computed in {elapsed:.1f}s")
    else:
        # Fallback to pure Python (slower)
        if verbose:
            print("      (Numba not available, using pure Python - this will be slow)")
        complexity = _lz76_pure_python(sequence, verbose)

    # Normalize by theoretical maximum for random sequence
    # For random binary: C_max ≈ n / log2(n) (Lempel-Ziv 1976)
    if n > 1:
        c_max = n / np.log2(n)
        normalized = complexity / c_max
    else:
        normalized = 0.0

    return min(1.0, normalized)


def _lz76_pure_python(sequence: str, verbose: bool = True) -> int:
    """Pure Python fallback for LZ76 when Numba is not available."""
    n = len(sequence)
    seq = sequence.encode('ascii')

    complexity = 0
    i = 0
    last_progress_pct = 0

    while i < n:
        if verbose:
            pct = int((i / n) * 100)
            if pct >= last_progress_pct + 10:
                print(f"      LZ progress: {pct}%")
                last_progress_pct = pct

        if i == 0:
            complexity += 1
            i += 1
            continue

        match_len = 0
        history = seq[0:i]
        lo, hi = 1, min(n - i, i)

        while lo <= hi:
            mid = (lo + hi) // 2
            pattern = seq[i:i + mid]
            if pattern in history:
                match_len = mid
                lo = mid + 1
            else:
                hi = mid - 1

        complexity += 1
        i += match_len + 1

    return complexity


def calculate_long_range_mutual_info(sequence: str, 
                                      block_size: int = 100,
                                      max_distance: int = 10000) -> float:
    """
    Calculate mutual information between distant blocks.
    
    High MI at long distances indicates global coherence.
    
    Args:
        sequence: Binary string
        block_size: Size of blocks to compare
        max_distance: Maximum distance between blocks
        
    Returns:
        Average MI ratio (actual MI / max possible MI)
    """
    n = len(sequence)
    if n < block_size * 2 + max_distance:
        max_distance = (n - block_size * 2) // 2
    
    if max_distance < block_size:
        return 0.0
    
    # Sample multiple block pairs at various distances
    distances = np.linspace(block_size, max_distance, min(20, max_distance // block_size))
    mi_values = []
    
    for dist in distances:
        dist = int(dist)
        # Sample multiple pairs at this distance
        n_samples = min(10, (n - block_size - dist) // (block_size * 2))
        
        for _ in range(max(1, n_samples)):
            start1 = np.random.randint(0, n - block_size - dist)
            start2 = start1 + dist
            
            block1 = sequence[start1:start1 + block_size]
            block2 = sequence[start2:start2 + block_size]
            
            # Calculate MI via joint and marginal probabilities
            mi = _calculate_block_mi(block1, block2)
            mi_values.append(mi)
    
    return np.mean(mi_values) if mi_values else 0.0


def _calculate_block_mi(block1: str, block2: str) -> float:
    """Calculate mutual information between two binary blocks."""
    # Count joint occurrences of bit pairs
    joint = np.zeros((2, 2))
    for b1, b2 in zip(block1, block2):
        joint[int(b1), int(b2)] += 1

    joint /= len(block1)  # Normalize to probabilities

    # Marginal probabilities
    p1 = joint.sum(axis=1)
    p2 = joint.sum(axis=0)

    # Mutual Information: I(X;Y) = sum p(x,y) * log(p(x,y) / (p(x)*p(y)))
    mi = 0.0
    for i in range(2):
        for j in range(2):
            if joint[i, j] > 0 and p1[i] > 0 and p2[j] > 0:
                mi += joint[i, j] * np.log2(joint[i, j] / (p1[i] * p2[j]))

    # Normalize by maximum possible MI (1 bit for binary)
    return max(0.0, mi)


def calculate_dfa(sequence: str, min_box: int = 4, max_box: int = None) -> Dict[str, Any]:
    """
    Detrended Fluctuation Analysis (DFA) to compute Hurst exponent.

    DFA measures long-range correlations in non-stationary time series.
    The Hurst exponent H indicates:
    - H = 0.5: Uncorrelated (white noise)
    - H < 0.5: Anti-persistent (mean-reverting)
    - H > 0.5: Persistent (trending, long-range correlations)
    - H ≈ 1.0: 1/f noise (critical, optimal for emergence)

    Args:
        sequence: Binary string
        min_box: Minimum box size for fluctuation calculation
        max_box: Maximum box size (default: len/4)

    Returns:
        Dictionary with hurst_exponent, r_squared, interpretation
    """
    # Convert to numeric array and compute cumulative sum (profile)
    y = np.array([int(b) for b in sequence], dtype=np.float64)
    y = y - y.mean()  # Center
    profile = np.cumsum(y)  # Integrated series

    n = len(profile)
    if max_box is None:
        max_box = n // 4

    # Generate box sizes (logarithmically spaced)
    box_sizes = np.unique(np.logspace(
        np.log10(min_box),
        np.log10(max_box),
        num=20
    ).astype(int))

    fluctuations = []

    for box_size in box_sizes:
        # Number of complete boxes
        n_boxes = n // box_size
        if n_boxes < 2:
            continue

        # Calculate fluctuation for each box
        box_flucts = []
        for i in range(n_boxes):
            start = i * box_size
            end = start + box_size
            segment = profile[start:end]

            # Fit linear trend and compute residual
            x = np.arange(box_size)
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)
            residual = segment - trend

            # RMS of residual
            rms = np.sqrt(np.mean(residual ** 2))
            box_flucts.append(rms)

        # Average fluctuation for this box size
        fluctuations.append((box_size, np.mean(box_flucts)))

    if len(fluctuations) < 3:
        return {
            'hurst_exponent': np.nan,
            'r_squared': 0.0,
            'interpretation': 'Insufficient data for DFA'
        }

    # Linear regression in log-log space: F(n) ~ n^H
    log_n = np.log10([f[0] for f in fluctuations])
    log_f = np.log10([f[1] for f in fluctuations])

    slope, intercept, r_value, _, _ = linregress(log_n, log_f)
    hurst = slope
    r_squared = r_value ** 2

    return {
        'hurst_exponent': hurst,
        'r_squared': r_squared,
        'box_sizes': [f[0] for f in fluctuations],
        'fluctuations': [f[1] for f in fluctuations],
        'interpretation': _interpret_hurst(hurst)
    }


def _interpret_hurst(h: float) -> str:
    """Interpret Hurst exponent from DFA."""
    if np.isnan(h):
        return "Could not calculate"
    elif h < 0.4:
        return "Anti-persistent (mean-reverting, unusual)"
    elif h < 0.6:
        return "Uncorrelated (white noise)"
    elif h < 0.85:
        return "Persistent (long-range correlations)"
    elif h <= 1.1:
        return "Critical (1/f noise, optimal for emergence)"
    else:
        return "Non-stationary (unbounded drift)"


def calculate_multiscale_entropy(sequence: str,
                                  max_scale: int = 20,
                                  m: int = 2,
                                  r_factor: float = 0.15) -> Dict[str, Any]:
    """
    Multi-Scale Entropy (MSE) analysis.

    Calculates Sample Entropy at multiple coarse-graining scales.
    Key insight from Costa et al. (2002):
    - White noise: entropy drops sharply with scale
    - Complex signals: entropy remains high across scales
    - Periodic: low entropy at all scales

    Args:
        sequence: Binary string
        max_scale: Maximum coarse-graining scale
        m: Embedding dimension (pattern length)
        r_factor: Tolerance factor (r = r_factor * std)

    Returns:
        Dictionary with mse_values, complexity_index, interpretation
    """
    # Convert to numeric
    data = np.array([int(b) for b in sequence], dtype=np.float64)
    n = len(data)

    mse_values = []
    scales = []

    for scale in range(1, min(max_scale + 1, n // 100)):
        # Coarse-grain: average consecutive non-overlapping windows
        if scale == 1:
            coarse = data
        else:
            # Truncate to multiple of scale
            truncated = data[:n // scale * scale]
            coarse = truncated.reshape(-1, scale).mean(axis=1)

        # Calculate Sample Entropy (fast version with sampling)
        se = _sample_entropy_fast(coarse, m, r_factor)
        if not np.isnan(se) and se < 10:
            mse_values.append(se)
            scales.append(scale)

    if len(mse_values) < 3:
        return {
            'mse_values': [],
            'scales': [],
            'complexity_index': 0.0,
            'mse_slope': 0.0,
            'interpretation': 'Insufficient data for MSE'
        }

    # Complexity Index: area under MSE curve (sum of entropies)
    complexity_index = np.sum(mse_values)

    # MSE slope: how entropy changes with scale
    # Negative slope = entropy drops with scale = noise-like
    # Zero/positive slope = complexity maintained = structured
    slope, _, r_value, _, _ = linregress(scales, mse_values)

    return {
        'mse_values': mse_values,
        'scales': scales,
        'complexity_index': complexity_index,
        'mse_slope': slope,
        'mean_entropy': np.mean(mse_values),
        'interpretation': _interpret_mse(slope, np.mean(mse_values))
    }


def _sample_entropy_fast(data: np.ndarray, m: int, r_factor: float,
                          max_samples: int = 1000) -> float:
    """
    Fast Sample Entropy using sampling for large sequences.

    For n > max_samples, we sample random pairs instead of checking all O(n²) pairs.
    """
    n = len(data)
    if n < m + 2:
        return np.nan

    # Tolerance
    r = r_factor * np.std(data)
    if r == 0:
        return 0.0  # Constant signal = zero entropy

    # If small enough, do exact calculation
    if n <= max_samples:
        return _sample_entropy_exact(data, m, r)

    # Sample-based approximation for large sequences
    n_samples = min(max_samples * max_samples // 2, 50000)

    count_m = 0
    count_m1 = 0

    for _ in range(n_samples):
        i = np.random.randint(0, n - m - 1)
        j = np.random.randint(0, n - m - 1)
        if i == j:
            continue

        # Check m-length match
        if np.max(np.abs(data[i:i+m] - data[j:j+m])) <= r:
            count_m += 1
            # Check if m+1 also matches
            if np.max(np.abs(data[i:i+m+1] - data[j:j+m+1])) <= r:
                count_m1 += 1

    if count_m == 0:
        return np.nan
    if count_m1 == 0:
        return 2.5  # High entropy cap

    return -np.log(count_m1 / count_m)


def _sample_entropy_exact(data: np.ndarray, m: int, r: float) -> float:
    """Exact Sample Entropy for small sequences."""
    n = len(data)

    def count_matches(template_len):
        count = 0
        for i in range(n - template_len):
            for j in range(i + 1, n - template_len):
                if np.max(np.abs(data[i:i+template_len] - data[j:j+template_len])) <= r:
                    count += 1
        return count

    A = count_matches(m + 1)
    B = count_matches(m)

    if B == 0:
        return np.nan
    if A == 0:
        return 2.5  # High entropy cap

    return -np.log(A / B)


def _interpret_mse(slope: float, mean_entropy: float) -> str:
    """Interpret MSE results."""
    if slope < -0.05:
        structure = "noise-like (entropy drops with scale)"
    elif slope > 0.02:
        structure = "complex (entropy increases with scale)"
    else:
        structure = "structured (entropy stable across scales)"

    if mean_entropy < 0.5:
        level = "low entropy (ordered)"
    elif mean_entropy < 1.5:
        level = "moderate entropy"
    else:
        level = "high entropy"

    return f"{structure}, {level}"


def calculate_hierarchical_block_entropy(sequence: str,
                                          max_block_power: int = 10,
                                          verbose: bool = False) -> Dict[str, Any]:
    """
    Calculate entropy at multiple block scales to detect hierarchical structure.

    For random sequences, entropy is maximum and constant across all scales.
    For structured sequences, entropy varies with scale (hierarchy signature).

    Args:
        sequence: Binary string
        max_block_power: Maximum block size as power of 2 (e.g., 10 = 1024 bits)
        verbose: Print progress

    Returns:
        Dictionary with:
        - entropies: Dict mapping block_size -> entropy value
        - hierarchy_score: Variance/Mean ratio (high = structured)
        - entropy_profile: Normalized entropy at each scale
    """
    n = len(sequence)
    entropies = {}

    # Calculate entropy at each scale (block sizes: 1, 2, 4, 8, ..., 2^max_block_power)
    for k in range(0, max_block_power + 1):
        block_size = 2 ** k

        if block_size > n // 4:  # Need at least 4 blocks for statistics
            break

        # Count frequency of each unique block pattern
        n_blocks = n // block_size
        block_counts = {}

        for i in range(n_blocks):
            block = sequence[i * block_size:(i + 1) * block_size]
            block_counts[block] = block_counts.get(block, 0) + 1

        # Calculate Shannon entropy of block distribution
        total = sum(block_counts.values())
        entropy = 0.0
        for count in block_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)

        # Normalize by maximum possible entropy for this block size
        # Max entropy = log2(number of possible patterns) = block_size for binary
        # But limited by number of blocks (can't have more patterns than blocks)
        max_patterns = min(2 ** block_size, n_blocks)
        max_entropy = np.log2(max_patterns) if max_patterns > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        entropies[block_size] = {
            'raw': entropy,
            'normalized': normalized_entropy,
            'n_unique_patterns': len(block_counts),
            'max_possible_patterns': max_patterns
        }

        if verbose:
            print(f"      Block {block_size:5d}: H={entropy:.3f}, "
                  f"H_norm={normalized_entropy:.3f}, "
                  f"patterns={len(block_counts):,}/{max_patterns:,}")

    # Calculate hierarchy score: how much entropy varies across scales
    if len(entropies) >= 2:
        norm_vals = [e['normalized'] for e in entropies.values()]
        mean_entropy = np.mean(norm_vals)
        var_entropy = np.var(norm_vals)

        # Coefficient of variation (high = variable = structured)
        hierarchy_score = np.std(norm_vals) / mean_entropy if mean_entropy > 0 else 0.0

        # Also compute entropy "slope" across scales
        scales = list(range(len(norm_vals)))
        if len(scales) >= 2:
            slope, _, r_value, _, _ = linregress(scales, norm_vals)
            entropy_slope = slope
            entropy_r2 = r_value ** 2
        else:
            entropy_slope = 0.0
            entropy_r2 = 0.0
    else:
        hierarchy_score = 0.0
        entropy_slope = 0.0
        entropy_r2 = 0.0
        mean_entropy = 0.0
        var_entropy = 0.0

    return {
        'entropies': entropies,
        'hierarchy_score': hierarchy_score,
        'mean_normalized_entropy': mean_entropy,
        'entropy_variance': var_entropy,
        'entropy_slope': entropy_slope,  # Negative = entropy decreases with scale (structured)
        'entropy_r2': entropy_r2,
        'interpretation': _interpret_hierarchy(hierarchy_score, entropy_slope)
    }


def _interpret_hierarchy(hierarchy_score: float, entropy_slope: float) -> str:
    """Interpret hierarchy score and entropy slope."""
    if hierarchy_score < 0.05:
        structure = "flat (no hierarchy)"
    elif hierarchy_score < 0.15:
        structure = "weak hierarchy"
    elif hierarchy_score < 0.30:
        structure = "moderate hierarchy"
    else:
        structure = "strong hierarchy"

    if entropy_slope > 0.02:
        trend = "entropy increases with scale (unusual)"
    elif entropy_slope < -0.02:
        trend = "entropy decreases with scale (typical of structure)"
    else:
        trend = "entropy stable across scales"

    return f"{structure}, {trend}"


def calculate_emergence_index(sequence: str,
                               sample_size: int = 1_000_000,
                               verbose: bool = True) -> Dict[str, Any]:
    """
    Calculate the composite Emergence Index.

    The index combines five components (Tier 1 + Tier 2 per Manus report):
    - Criticality score (1/f power spectrum)
    - Complexity score (LZ compressibility, penalizes only noise)
    - Coherence score (long-range mutual information)
    - Hierarchy score (multi-scale entropy structure)
    - DFA score (Hurst exponent, long-range correlations)

    Args:
        sequence: Binary Φ sequence
        sample_size: Maximum sample size for analysis (for large sequences)
        verbose: Print progress messages

    Returns:
        Dictionary with:
        - emergence_index: Composite score (0-1, higher = more potential)
        - criticality: Dict with slope, r_squared, score
        - complexity: Dict with lz_complexity, score
        - coherence: Dict with mi_ratio, score
        - hierarchy: Dict with hierarchy_score, entropy profile
        - interpretation: Human-readable interpretation
    """
    import time

    def log(msg):
        if verbose:
            print(f"   {msg}", flush=True)

    # Sample if sequence is too large
    if len(sequence) > sample_size:
        # Take middle portion for representative sample
        start = (len(sequence) - sample_size) // 2
        seq = sequence[start:start + sample_size]
    else:
        seq = sequence

    results = {
        'sequence_length': len(sequence),
        'sample_size': len(seq)
    }

    # 1. Criticality (1/f spectrum)
    log("⚡ [1/5] Calculating power spectrum (FFT)...")
    t0 = time.time()
    slope, r_sq = calculate_power_spectrum_slope(seq)
    log(f"   Done in {time.time()-t0:.1f}s - slope={slope:.3f}")
    # Perfect 1/f has slope = -1; score based on distance from -1
    criticality_score = max(0, 1 - abs(slope + 1) / 2)  # 0-1 scale
    results['criticality'] = {
        'slope': slope,
        'r_squared': r_sq,
        'target': -1.0,
        'score': criticality_score,
        'interpretation': _interpret_slope(slope)
    }

    # 2. Lempel-Ziv Complexity (Compressibility)
    log("🧩 [2/5] Calculating LZ complexity...")
    t0 = time.time()
    lz = calculate_lempel_ziv_complexity(seq, verbose=verbose)
    log(f"   Done in {time.time()-t0:.1f}s - LZ={lz:.3f}")
    # NEW SCORING: Penalize only empty randomness (LZ > 0.7), not order
    # Physical systems are compressible (have laws), so LZ < 0.7 is good
    # LZ ~ 1.0 means incompressible noise = bad for emergence
    # LZ in [0.3, 0.7] is ideal (structured complexity per Manus report)
    if lz <= 0.3:
        # Highly ordered - still good, just very compressible
        complexity_score = 0.7 + (lz / 0.3) * 0.3  # 0.7-1.0
    elif lz <= 0.7:
        # Ideal range: structured complexity
        complexity_score = 1.0
    else:
        # Too random - penalize linearly
        complexity_score = max(0, 1.0 - (lz - 0.7) / 0.3)  # 1.0 -> 0

    results['complexity'] = {
        'lz_normalized': lz,
        'ideal_range': '0.3-0.7',
        'score': complexity_score,
        'interpretation': _interpret_lz(lz)
    }

    # 3. Long-Range Coherence
    log("🔗 [3/5] Calculating long-range MI...")
    t0 = time.time()
    mi_ratio = calculate_long_range_mutual_info(seq)
    log(f"   Done in {time.time()-t0:.1f}s - MI={mi_ratio:.4f}")
    # Higher MI = more coherence = better
    coherence_score = min(1.0, mi_ratio * 10)  # Scale to 0-1
    results['coherence'] = {
        'mi_ratio': mi_ratio,
        'score': coherence_score,
        'interpretation': _interpret_mi(mi_ratio)
    }

    # 4. Hierarchical Block Entropy
    log("📊 [4/5] Calculating hierarchical entropy...")
    t0 = time.time()
    hbe = calculate_hierarchical_block_entropy(seq, max_block_power=12, verbose=False)
    log(f"   Done in {time.time()-t0:.1f}s - HierScore={hbe['hierarchy_score']:.3f}")

    # Score: Higher hierarchy_score = more structure = better for physical systems
    # But we need non-trivial hierarchy (not just repetition)
    # Optimal: hierarchy_score > 0.2 AND mean_entropy > 0.3
    raw_hier = hbe['hierarchy_score']
    mean_ent = hbe['mean_normalized_entropy']

    # Penalize if too ordered (mean entropy < 0.3) or too random (mean entropy > 0.9)
    entropy_penalty = 1.0
    if mean_ent < 0.3:
        entropy_penalty = mean_ent / 0.3  # Penalize over-order
    elif mean_ent > 0.9:
        entropy_penalty = (1.0 - mean_ent) / 0.1  # Penalize near-random

    # Hierarchy score capped at 1.0, scaled
    hierarchy_score = min(1.0, raw_hier * 2) * entropy_penalty

    results['hierarchy'] = {
        'raw_hierarchy_score': raw_hier,
        'mean_entropy': mean_ent,
        'entropy_slope': hbe['entropy_slope'],
        'score': hierarchy_score,
        'interpretation': hbe['interpretation']
    }

    # 5. DFA - Hurst Exponent (NEW)
    log("📈 [5/5] Calculating DFA (Hurst exponent)...")
    t0 = time.time()
    dfa_result = calculate_dfa(seq)
    hurst = dfa_result['hurst_exponent']
    log(f"   Done in {time.time()-t0:.1f}s - H={hurst:.3f}")

    # Score: H ≈ 1.0 is optimal (1/f criticality)
    # H = 0.5 is white noise, H > 1.1 is non-stationary
    if np.isnan(hurst):
        dfa_score = 0.0
    elif 0.85 <= hurst <= 1.1:
        # Optimal range: 1/f criticality
        dfa_score = 1.0
    elif 0.6 <= hurst < 0.85:
        # Good: persistent correlations
        dfa_score = 0.6 + (hurst - 0.6) / 0.25 * 0.4  # 0.6 -> 1.0
    elif hurst < 0.6:
        # Poor: too random or anti-persistent
        dfa_score = max(0, hurst / 0.6 * 0.6)
    else:  # hurst > 1.1
        # Non-stationary: penalize
        dfa_score = max(0, 1.0 - (hurst - 1.1) / 0.4)

    results['dfa'] = {
        'hurst_exponent': hurst,
        'r_squared': dfa_result['r_squared'],
        'score': dfa_score,
        'interpretation': dfa_result['interpretation']
    }

    # Composite Emergence Index (weighted average)
    # 5 metrics now, balanced weights
    weights = {
        'criticality': 0.20,  # 1/f spectrum (FFT)
        'complexity': 0.15,   # LZ complexity (compressibility)
        'coherence': 0.20,    # Long-range MI
        'hierarchy': 0.20,    # Multi-scale structure (HBE)
        'dfa': 0.25           # Hurst exponent (long-range correlations)
    }

    emergence_index = (
        weights['criticality'] * criticality_score +
        weights['complexity'] * complexity_score +
        weights['coherence'] * coherence_score +
        weights['hierarchy'] * hierarchy_score +
        weights['dfa'] * dfa_score
    )

    results['emergence_index'] = emergence_index
    results['component_weights'] = weights
    results['interpretation'] = _interpret_emergence(emergence_index)

    log(f"✅ Emergence Index: {emergence_index:.4f}")

    return results


def _interpret_slope(slope: float) -> str:
    """Interpret power spectrum slope."""
    if -1.2 <= slope <= -0.8:
        return "1/f noise (critical, high emergence potential)"
    elif slope > -0.3:
        return "White noise (random, low structure)"
    elif slope < -1.5:
        return "Brownian (over-correlated, low dynamics)"
    else:
        return f"Intermediate ({slope:.2f})"


def _interpret_lz(lz: float) -> str:
    """Interpret Lempel-Ziv complexity."""
    if lz < 0.1:
        return "Very low complexity (highly ordered, periodic)"
    elif lz < 0.3:
        return "Low complexity (ordered, compressible - good for physics)"
    elif lz <= 0.7:
        return "Structured complexity (ideal range for emergence)"
    else:
        return "High complexity (near-random, low structure - noise)"


def _interpret_mi(mi: float) -> str:
    """Interpret mutual information ratio."""
    if mi < 0.01:
        return "No long-range correlation (local structure only)"
    elif mi < 0.05:
        return "Weak coherence (some global structure)"
    elif mi < 0.1:
        return "Moderate coherence (significant global patterns)"
    else:
        return "Strong coherence (highly correlated distant regions)"


def _interpret_emergence(index: float) -> str:
    """Interpret composite emergence index."""
    if index < 0.3:
        return "Low emergence potential (random or trivially ordered)"
    elif index < 0.5:
        return "Moderate emergence potential"
    elif index < 0.7:
        return "High emergence potential (good candidate for Level 2)"
    else:
        return "Very high emergence potential (strong Level 2 candidate)"


# =============================================================================
# DATA LOADING UTILITIES
# =============================================================================

def load_phi_sequence(variant: str, iteration: int,
                       max_bits: int = 2_000_000) -> Optional[str]:
    """
    Load a SAMPLE of Φ sequence from phi_snapshots for a given variant and iteration.

    To avoid memory issues with large sequences (2GB+), we only load a sample
    from the middle portion of the sequence.

    Args:
        variant: Variant code (A, B, D, E, F, G, H)
        iteration: Iteration number
        max_bits: Maximum bits to load (default 2M for fast analysis)

    Returns:
        Binary string sample or None if not found
    """
    import gzip
    from pathlib import Path

    # Try v33 format first (bitarray in .struct.gz)
    base_path = Path(__file__).parent.parent / "results" / "phi_snapshots" / f"var_{variant}"
    struct_path = base_path / f"phi_iter{iteration}.struct.gz"

    if struct_path.exists():
        try:
            from bitarray import bitarray
            with gzip.open(struct_path, 'rb') as f:
                # Read file to get total size first
                data = f.read()
            ba = bitarray()
            ba.frombytes(data)
            total_bits = len(ba)

            # Sample from middle for representative data
            if total_bits > max_bits:
                start = (total_bits - max_bits) // 2
                sample = ba[start:start + max_bits].to01()
                print(f"   Sampled {max_bits:,} bits from middle of {total_bits:,} total bits")
                return sample
            else:
                return ba.to01()
        except Exception as e:
            print(f"Warning: Could not load {struct_path}: {e}")

    # Try bin.gz format
    bin_path = base_path / f"phi_iter{iteration}.bin.gz"
    if bin_path.exists():
        try:
            from bitarray import bitarray
            with gzip.open(bin_path, 'rb') as f:
                data = f.read()
            ba = bitarray()
            ba.frombytes(data)
            total_bits = len(ba)

            if total_bits > max_bits:
                start = (total_bits - max_bits) // 2
                sample = ba[start:start + max_bits].to01()
                print(f"   Sampled {max_bits:,} bits from middle of {total_bits:,} total bits")
                return sample
            else:
                return ba.to01()
        except Exception as e:
            print(f"Warning: Could not load {bin_path}: {e}")

    return None


def compare_variants_emergence(variants: list, iteration: int) -> Dict[str, Any]:
    """
    Compare emergence index across multiple variants.

    Args:
        variants: List of variant codes (e.g., ['A', 'B', 'D'])
        iteration: Iteration number to compare

    Returns:
        Dictionary with comparison results and ranking
    """
    results = {}

    for variant in variants:
        print(f"   Analyzing variant {variant}...", flush=True)
        seq = load_phi_sequence(variant, iteration)

        if seq is None:
            print(f"   ⚠️ Could not load variant {variant} iter {iteration}")
            results[variant] = None
            continue

        emergence = calculate_emergence_index(seq)
        results[variant] = emergence

        print(f"   ✓ Variant {variant}: EI = {emergence['emergence_index']:.3f}")

    # Rank variants by emergence index
    valid_results = {k: v for k, v in results.items() if v is not None}
    ranking = sorted(valid_results.keys(),
                     key=lambda v: valid_results[v]['emergence_index'],
                     reverse=True)

    return {
        'iteration': iteration,
        'variants': results,
        'ranking': ranking,
        'best_variant': ranking[0] if ranking else None,
        'best_score': valid_results[ranking[0]]['emergence_index'] if ranking else 0
    }


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Command-line interface for Emergence Index calculation."""
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description='Calculate Emergence Index for HSI Φ sequences.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python emergence_index.py --variant B --iteration 18
  python emergence_index.py --variants A B D E F --iteration 18 --compare
  python emergence_index.py --variant B --iteration 18 --output results/emergence_B.json
        """
    )

    parser.add_argument('--variant', '-v', type=str,
                        help='Single variant to analyze (A, B, D, E, F, G, H)')
    parser.add_argument('--variants', nargs='+',
                        help='Multiple variants for comparison')
    parser.add_argument('--iteration', '-i', type=int, required=True,
                        help='Iteration number')
    parser.add_argument('--compare', action='store_true',
                        help='Compare multiple variants and rank them')
    parser.add_argument('--output', '-o', type=str,
                        help='Output JSON file path')
    parser.add_argument('--sample-size', type=int, default=1_000_000,
                        help='Sample size for large sequences (default: 1M)')

    args = parser.parse_args()

    # Validate arguments
    if not args.variant and not args.variants:
        parser.error("Must specify --variant or --variants")

    print("\n🌌 HSI Emergence Index Calculator")
    print("=" * 50)

    if args.compare or args.variants:
        # Multi-variant comparison mode
        variants = args.variants or [args.variant]
        print(f"\n📊 Comparing variants: {', '.join(variants)}")
        print(f"   Iteration: {args.iteration}\n")

        results = compare_variants_emergence(variants, args.iteration)

        # Print summary
        print("\n" + "=" * 50)
        print("📈 EMERGENCE INDEX RANKING")
        print("=" * 50)

        for rank, variant in enumerate(results['ranking'], 1):
            data = results['variants'][variant]
            ei = data['emergence_index']
            interp = data['interpretation']
            print(f"   {rank}. Variant {variant}: {ei:.4f} - {interp}")

        if results['best_variant']:
            print(f"\n🏆 Best candidate for Level 2: Variant {results['best_variant']}")

    else:
        # Single variant mode
        print(f"\n📊 Analyzing variant {args.variant}, iteration {args.iteration}\n")

        seq = load_phi_sequence(args.variant, args.iteration)
        if seq is None:
            print(f"❌ Could not load variant {args.variant} iteration {args.iteration}")
            return

        print(f"   Loaded {len(seq):,} bits")
        results = calculate_emergence_index(seq, args.sample_size)

        # Print detailed results
        print("\n" + "=" * 50)
        print("🌌 EMERGENCE INDEX ANALYSIS")
        print("=" * 50)

        print(f"\n📏 Sample: {results['sample_size']:,} / {results['sequence_length']:,} bits")

        c = results['criticality']
        print(f"\n⚡ CRITICALITY (1/f spectrum)")
        print(f"   Slope: {c['slope']:.3f} (target: {c['target']})")
        print(f"   R²: {c['r_squared']:.3f}")
        print(f"   Score: {c['score']:.3f}")
        print(f"   → {c['interpretation']}")

        x = results['complexity']
        print(f"\n🧩 COMPLEXITY (Lempel-Ziv)")
        print(f"   LZ normalized: {x['lz_normalized']:.3f} (ideal: {x['ideal_range']})")
        print(f"   Score: {x['score']:.3f}")
        print(f"   → {x['interpretation']}")

        h = results['coherence']
        print(f"\n🔗 COHERENCE (Long-range MI)")
        print(f"   MI ratio: {h['mi_ratio']:.4f}")
        print(f"   Score: {h['score']:.3f}")
        print(f"   → {h['interpretation']}")

        # Hierarchy
        if 'hierarchy' in results:
            hr = results['hierarchy']
            print(f"\n📊 HIERARCHY (Multi-scale entropy)")
            print(f"   Hierarchy score: {hr['raw_hierarchy_score']:.4f}")
            print(f"   Mean entropy: {hr['mean_entropy']:.3f}")
            print(f"   Score: {hr['score']:.3f}")
            print(f"   → {hr['interpretation']}")

        # DFA (Hurst exponent)
        if 'dfa' in results:
            dfa = results['dfa']
            print(f"\n📈 DFA (Hurst exponent)")
            print(f"   Hurst H: {dfa['hurst_exponent']:.3f}")
            print(f"   R²: {dfa['r_squared']:.3f}")
            print(f"   Score: {dfa['score']:.3f}")
            print(f"   → {dfa['interpretation']}")

        print(f"\n{'=' * 50}")
        print(f"🌌 EMERGENCE INDEX: {results['emergence_index']:.4f}")
        print(f"   → {results['interpretation']}")
        print("=" * 50)

    # Save results (auto-generate filename if not specified)
    if args.output:
        output_path = Path(args.output)
    else:
        # Auto-generate filename based on variant(s) and iteration
        script_dir = Path(__file__).parent.parent
        results_dir = script_dir / "results"
        if args.compare and args.variants:
            variants_str = "_".join(args.variants)
            filename = f"emergence_vars_{variants_str}_iter{args.iteration}.json"
        else:
            variant = args.variant or (args.variants[0] if args.variants else "unknown")
            filename = f"emergence_var{variant}_iter{args.iteration}.json"
        output_path = results_dir / filename

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to {output_path}")


if __name__ == '__main__':
    main()

