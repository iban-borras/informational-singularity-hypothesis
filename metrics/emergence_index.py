"""
🌌 Structural Emergence Index (SEI) — Metric for Level 2 Potential in HSI

This module calculates the "Structural Emergence Index" (SEI), a composite metric
designed to detect the potential for emergence of PHYSICAL LAWS in Φ sequences.

IMPORTANT: Unlike "Edge of Chaos" metrics (Langton 1990), SEI is specifically
designed for HSI theory where we seek emergence of stable physical laws, not
maximal computational capacity at the chaos boundary.

The SEI combines four fundamental indicators:

1. ORDER (30%) — Compressibility via Lempel-Ziv
   Physical laws are compressible (can be described with short equations).
   A universe with stable laws has LOW LZ complexity, not intermediate.
   Score = 1 - LZ (order is rewarded, randomness penalized)

2. HIERARCHY (30%) — Multi-scale structure via Hierarchical Block Entropy
   Physical laws operate across scales (quantum → classical → cosmological).
   High hierarchy indicates potential for scale-invariant emergent physics.

3. COHERENCE (20%) — Long-range correlations via Mutual Information
   Physical constants are the same everywhere in the universe.
   High coherence indicates law constancy across distant regions.

4. NON-RANDOMNESS (20%) — Combined DFA (Hurst) + Spectrum analysis
   Random sequences cannot sustain stable physical laws.
   Penalizes systems with H ≈ 0.5 (white noise) or flat spectrum.

Scientific basis:
- Compressibility ↔ law existence (Chaitin, 1987; Solomonoff, 1964)
- Multi-scale structure ↔ renormalization group (Wilson, 1971)
- Long-range order ↔ phase coherence (Anderson, 1972)
- Non-randomness ↔ deterministic dynamics (Peng et al., 1994)

Author: Iban Borràs with Augment Agent collaboration
Date: December 2024, reformulated December 2025
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from scipy.fft import fft
from scipy.stats import linregress
import warnings

# RAM monitoring for adaptive parallelism
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def _wait_for_ram(target_percent: float = 70.0, max_wait: int = 300,
                  check_interval: int = 5, verbose: bool = False) -> bool:
    """
    Wait until RAM usage drops below target percentage.

    Args:
        target_percent: Target RAM usage percentage to wait for
        max_wait: Maximum seconds to wait before giving up
        check_interval: Seconds between checks
        verbose: Print waiting messages

    Returns:
        True if RAM dropped below target, False if timeout
    """
    if not HAS_PSUTIL:
        return True  # Can't check, assume OK

    import time
    waited = 0
    initial_ram = psutil.virtual_memory().percent

    if initial_ram <= target_percent:
        return True

    if verbose:
        print(f"   ⏳ RAM at {initial_ram:.1f}% - waiting for <{target_percent}%...", flush=True)

    while waited < max_wait:
        time.sleep(check_interval)
        waited += check_interval
        current_ram = psutil.virtual_memory().percent

        if current_ram <= target_percent:
            if verbose:
                print(f"   ✅ RAM dropped to {current_ram:.1f}% after {waited}s", flush=True)
            return True

    if verbose:
        print(f"   ⚠️ RAM still at {psutil.virtual_memory().percent:.1f}% after {max_wait}s timeout", flush=True)
    return False


def _get_adaptive_workers(base_workers: int, ram_threshold: float = 75.0,
                          min_workers: int = 2, verbose: bool = False) -> int:
    """
    Adjust number of workers based on current RAM usage.

    Args:
        base_workers: Maximum number of workers to use
        ram_threshold: RAM usage percentage above which to reduce workers (default: 75%)
        min_workers: Minimum number of workers to maintain
        verbose: Print adjustment messages

    Returns:
        Adjusted number of workers
    """
    if not HAS_PSUTIL:
        return base_workers

    ram_percent = psutil.virtual_memory().percent

    if ram_percent > ram_threshold:
        # More aggressive reduction: halve workers for every 10% above threshold
        # At 75% -> 100%, at 85% -> 50%, at 95% -> 25%
        excess = ram_percent - ram_threshold
        reduction_steps = int(excess / 10)  # Each 10% over threshold = halve again
        new_workers = base_workers
        for _ in range(reduction_steps + 1):
            new_workers = max(min_workers, new_workers // 2)

        if verbose and new_workers < base_workers:
            print(f"   ⚠️ RAM at {ram_percent:.1f}% - reducing workers: {base_workers} → {new_workers}", flush=True)

        return new_workers

    return base_workers


# =============================================================================
# CONFIGURABLE WEIGHTS
# =============================================================================

def _load_emergence_weights() -> Dict[str, float]:
    """
    Load Structural Emergence Index (SEI) component weights from config.json.

    The weights determine how each component contributes to the final index.
    See the comment block in calculate_emergence_index() for scientific justification.

    STRUCTURAL EMERGENCE INDEX (SEI) - Dec 2025 Reformulation:

    Unlike "Edge of Chaos" metrics (Langton 1990), SEI is designed to detect
    emergence of PHYSICAL LAWS, which have different characteristics:

    1. Physical laws = Compressibility: A universe with laws is highly compressible
       (can be described with short equations). Low LZ is GOOD, not bad.

    2. Multi-scale hierarchy: Physical laws operate across scales
       (quantum → classical → cosmological). Hierarchy is key.

    3. Randomness is the worst case: A random universe cannot have stable laws.

    Returns:
        Dictionary with weights for each component, summing to 1.0
    """
    from pathlib import Path
    import json

    # Default weights for STRUCTURAL EMERGENCE (not edge-of-chaos)
    # Scientific justification in calculate_emergence_index() docstring
    defaults = {
        'order': 0.30,       # (1 - LZ) Compressibility = physical laws
        'hierarchy': 0.30,   # Multi-scale structure = emergent physics
        'coherence': 0.20,   # Long-range MI = law constancy across space
        'non_randomness': 0.20  # Penalize random systems (DFA + criticality)
    }

    try:
        config_path = Path(__file__).parent.parent / 'config.json'
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            metrics_config = config.get('metrics', {})
            custom_weights = metrics_config.get('emergence_weights', {})

            if custom_weights:
                # Merge with defaults and normalize to sum=1.0
                merged = {**defaults, **custom_weights}
                total = sum(merged.values())
                return {k: v/total for k, v in merged.items()}
    except Exception:
        pass

    return defaults


# =============================================================================
# MEMORY-MAPPED FILE HELPERS FOR GIANT SEQUENCES
# =============================================================================

def _decompress_to_temp(gz_path, verbose: bool = False) -> Optional[str]:
    """
    Decompress a .gz file to a temporary file for memory-mapped access.

    Args:
        gz_path: Path to .gz compressed file
        verbose: Print progress messages

    Returns:
        Path to temporary file, or None if failed
    """
    import gzip
    import tempfile
    import os
    from pathlib import Path

    gz_path = Path(gz_path)
    if not gz_path.exists():
        return None

    # Create temp file in same directory (faster, same filesystem)
    temp_dir = gz_path.parent
    temp_path = temp_dir / f".tmp_{gz_path.stem}_mmap.bin"

    # Estimate decompressed size (gzip ratio ~7-10x for binary)
    compressed_size = gz_path.stat().st_size
    estimated_size = compressed_size * 8  # Conservative estimate

    if verbose:
        print(f"   📦 Decompressing to temp file for mmap access...", flush=True)
        print(f"      Source: {gz_path.name} ({compressed_size / 1e6:.1f} MB)", flush=True)

    try:
        import time
        from tqdm import tqdm
        t0 = time.time()

        with gzip.open(gz_path, 'rb') as f_in:
            with open(temp_path, 'wb') as f_out:
                # Stream in chunks to avoid loading all in RAM
                chunk_size = 64 * 1024 * 1024  # 64 MB chunks
                total_written = 0

                # Progress bar with estimated size
                pbar = tqdm(total=estimated_size, unit='B', unit_scale=True,
                           desc="      Decompressing", disable=not verbose,
                           leave=False)

                while True:
                    chunk = f_in.read(chunk_size)
                    if not chunk:
                        break
                    f_out.write(chunk)
                    total_written += len(chunk)
                    pbar.update(len(chunk))

                # Adjust total if we went over estimate
                if total_written > estimated_size:
                    pbar.total = total_written
                    pbar.refresh()
                pbar.close()

        if verbose:
            elapsed = time.time() - t0
            size_gb = temp_path.stat().st_size / 1e9
            print(f"   ✅ Temp file ready: {size_gb:.1f} GB in {elapsed:.1f}s", flush=True)

        return str(temp_path)

    except Exception as e:
        if verbose:
            print(f"   ❌ Decompression failed: {e}", flush=True)
        # Clean up partial file
        if temp_path.exists():
            os.remove(temp_path)
        return None


def _read_chunk_from_mmap(mmap_obj, start_byte: int, end_byte: int) -> str:
    """
    Read a chunk from memory-mapped file and convert to bit string.

    Args:
        mmap_obj: Memory-mapped file object
        start_byte: Starting byte position
        end_byte: Ending byte position (exclusive)

    Returns:
        Binary string representation of the bytes
    """
    from bitarray import bitarray

    # Read bytes from mmap
    chunk_bytes = mmap_obj[start_byte:end_byte]

    # Convert to bitarray then to string
    ba = bitarray()
    ba.frombytes(chunk_bytes)
    return ba.to01()


def _cleanup_temp_file(temp_path: str, verbose: bool = False):
    """Clean up temporary file after processing."""
    import os
    from pathlib import Path

    try:
        if temp_path and Path(temp_path).exists():
            os.remove(temp_path)
            if verbose:
                print(f"   🗑️ Cleaned up temp file", flush=True)
    except Exception as e:
        if verbose:
            print(f"   ⚠️ Could not clean temp file: {e}", flush=True)


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
    Optimized with numpy vectorization.

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

    # Convert to numpy array once (much faster for slicing)
    seq_array = np.frombuffer(sequence.encode('ascii'), dtype=np.uint8) - ord('0')

    # Sample multiple block pairs at various distances
    n_distances = min(20, max_distance // block_size)
    distances = np.linspace(block_size, max_distance, n_distances).astype(int)
    mi_values = []

    for dist in distances:
        # Sample multiple pairs at this distance
        n_samples = min(10, (n - block_size - dist) // (block_size * 2))

        for _ in range(max(1, n_samples)):
            start1 = np.random.randint(0, n - block_size - dist)
            start2 = start1 + dist

            block1 = seq_array[start1:start1 + block_size]
            block2 = seq_array[start2:start2 + block_size]

            # Calculate MI via joint and marginal probabilities (vectorized)
            mi = _calculate_block_mi_fast(block1, block2)
            mi_values.append(mi)

    return np.mean(mi_values) if mi_values else 0.0


def _calculate_block_mi_fast(block1: np.ndarray, block2: np.ndarray) -> float:
    """Calculate mutual information between two binary blocks (vectorized)."""
    n = len(block1)

    # Count joint occurrences vectorized: 2*b1 + b2 gives indices 0,1,2,3
    combined = 2 * block1 + block2
    counts = np.bincount(combined, minlength=4)
    joint = counts.reshape(2, 2) / n

    # Marginal probabilities
    p1 = joint.sum(axis=1)
    p2 = joint.sum(axis=0)

    # Outer product for denominator
    outer = np.outer(p1, p2)

    # Mutual Information with masking for valid entries
    valid = (joint > 0) & (outer > 0)
    if not np.any(valid):
        return 0.0

    mi = np.sum(joint[valid] * np.log2(joint[valid] / outer[valid]))

    return max(0.0, mi)


# Numba-accelerated DFA kernel
_DFA_KERNEL = None

def _get_dfa_numba_kernel():
    """Get Numba-compiled DFA fluctuation calculator."""
    try:
        from numba import njit, prange

        @njit(parallel=True, cache=True)
        def dfa_fluctuation_kernel(profile, box_size):
            """
            Calculate RMS fluctuation for all boxes of given size.
            Numba-accelerated with parallel execution.
            """
            n = len(profile)
            n_boxes = n // box_size

            if n_boxes < 2:
                return 0.0

            total_fluct = 0.0

            for i in prange(n_boxes):
                start = i * box_size

                # Linear detrending using least squares formula
                # For y = ax + b, optimal a = cov(x,y)/var(x), b = mean(y) - a*mean(x)
                sum_x = 0.0
                sum_y = 0.0
                sum_xy = 0.0
                sum_xx = 0.0

                for j in range(box_size):
                    x = float(j)
                    y = profile[start + j]
                    sum_x += x
                    sum_y += y
                    sum_xy += x * y
                    sum_xx += x * x

                n_f = float(box_size)
                mean_x = sum_x / n_f
                mean_y = sum_y / n_f

                var_x = sum_xx / n_f - mean_x * mean_x
                cov_xy = sum_xy / n_f - mean_x * mean_y

                if var_x > 1e-10:
                    a = cov_xy / var_x
                else:
                    a = 0.0
                b = mean_y - a * mean_x

                # Calculate RMS of residuals
                rms_sq = 0.0
                for j in range(box_size):
                    trend = a * float(j) + b
                    residual = profile[start + j] - trend
                    rms_sq += residual * residual

                rms = (rms_sq / n_f) ** 0.5
                total_fluct += rms

            return total_fluct / float(n_boxes)

        return dfa_fluctuation_kernel

    except ImportError:
        return None


def calculate_dfa(sequence: str, min_box: int = 4, max_box: int = None) -> Dict[str, Any]:
    """
    Detrended Fluctuation Analysis (DFA) to compute Hurst exponent.

    Uses Numba JIT compilation for 10-50x speedup on large sequences.

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
    global _DFA_KERNEL

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

    # Try to use Numba kernel
    if _DFA_KERNEL is None:
        _DFA_KERNEL = _get_dfa_numba_kernel()

    fluctuations = []

    if _DFA_KERNEL is not None:
        # Numba-accelerated path
        for box_size in box_sizes:
            n_boxes = n // box_size
            if n_boxes < 2:
                continue
            avg_fluct = _DFA_KERNEL(profile, int(box_size))
            if avg_fluct > 0:
                fluctuations.append((box_size, avg_fluct))
    else:
        # Fallback to pure Python/numpy
        for box_size in box_sizes:
            n_boxes = n // box_size
            if n_boxes < 2:
                continue

            box_flucts = []
            for i in range(n_boxes):
                start = i * box_size
                end = start + box_size
                segment = profile[start:end]

                x = np.arange(box_size)
                coeffs = np.polyfit(x, segment, 1)
                trend = np.polyval(coeffs, x)
                residual = segment - trend

                rms = np.sqrt(np.mean(residual ** 2))
                box_flucts.append(rms)

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

    # =========================================================================
    # STRUCTURAL EMERGENCE INDEX (SEI) - Dec 2025 Reformulation
    # =========================================================================
    #
    # SCIENTIFIC JUSTIFICATION:
    #
    # Unlike "Edge of Chaos" metrics (Langton 1990), the SEI is designed to
    # detect emergence of PHYSICAL LAWS, which have different characteristics:
    #
    # 1. ORDER (30%) - Physical laws = Compressibility
    #    A universe with stable physical laws is highly compressible because
    #    it can be described with short equations (Newton, Maxwell, Einstein).
    #    LZ complexity close to 1.0 means RANDOM = no laws = bad.
    #    LZ complexity close to 0.0 means ORDERED = compressible = good.
    #    Score = 1 - LZ (inverted: order is rewarded, not penalized)
    #
    # 2. HIERARCHY (30%) - Multi-scale structure
    #    Physical laws operate across scales (quantum → classical → cosmological).
    #    Hierarchical Block Entropy measures if structure exists at multiple
    #    scales. High hierarchy = emergent physics potential.
    #
    # 3. COHERENCE (20%) - Long-range correlations
    #    Physical constants are the same everywhere in the universe.
    #    Long-range Mutual Information measures if distant regions share
    #    structure. High coherence = law constancy across space.
    #
    # 4. NON-RANDOMNESS (20%) - Combined DFA + Criticality
    #    Random sequences (H ≈ 0.5, white noise spectrum) cannot sustain
    #    stable physical laws. This component penalizes randomness using:
    #    - DFA: Hurst exponent (H > 0.5 = persistent correlations)
    #    - Criticality: Spectrum slope (slope < 0 = structured)
    #    Score = average of both, rewarding deviation from pure randomness.
    #
    # These weights can be customized via config.json → metrics.emergence_weights
    # =========================================================================
    weights = _load_emergence_weights()

    # Calculate component scores for SEI
    order_score = 1.0 - lz  # Invert LZ: low LZ = high order = good

    # Non-randomness: combine DFA and criticality to penalize pure randomness
    # H = 0.5 is white noise → score 0; H > 0.6 is persistent → score increases
    # Slope near 0 is white noise → score 0; slope < -0.5 is structured → score increases
    dfa_nonrandom = max(0, (hurst - 0.5) * 2) if not np.isnan(hurst) else 0  # 0.5→0, 1.0→1
    crit_nonrandom = max(0, min(1, -slope)) if slope < 0 else 0  # slope -1→1, 0→0
    non_randomness_score = (dfa_nonrandom + crit_nonrandom) / 2

    emergence_index = (
        weights['order'] * order_score +
        weights['hierarchy'] * hierarchy_score +
        weights['coherence'] * coherence_score +
        weights['non_randomness'] * non_randomness_score
    )

    # Store detailed scores for analysis
    results['order'] = {
        'lz_normalized': lz,
        'order_score': order_score,
        'interpretation': f"Order = {order_score:.1%} (LZ={lz:.3f}, lower=more ordered)"
    }
    results['non_randomness'] = {
        'dfa_component': dfa_nonrandom,
        'criticality_component': crit_nonrandom,
        'score': non_randomness_score,
        'interpretation': f"Non-random = {non_randomness_score:.1%} (H={hurst:.2f}, slope={slope:.2f})"
    }

    # =========================================================================
    # EDGE OF CHAOS INDEX (ECI) - Original Langton-inspired metric
    # =========================================================================
    # Preserved for comparison: seeks the computational "sweet spot" between
    # order and chaos (Langton 1990, Kauffman 1993).
    #
    # Components (original weights):
    # - Criticality (25%): 1/f spectrum = self-organized criticality
    # - Complexity (25%): LZ ≈ 0.5 = edge of chaos (penalize both extremes)
    # - Coherence (20%): Long-range MI = global structure
    # - Hierarchy (15%): Multi-scale entropy
    # - DFA (15%): Hurst exponent ≈ 0.8-1.0 = 1/f correlations
    # =========================================================================

    # LZ score for ECI: peak at 0.5 (edge of chaos)
    if lz <= 0.3:
        eci_lz_score = 0.7 + (lz / 0.3) * 0.3  # 0→0.7, 0.3→1.0
    elif lz <= 0.7:
        eci_lz_score = 1.0  # Ideal range
    else:
        eci_lz_score = max(0, 1.0 - (lz - 0.7) / 0.3)  # 0.7→1.0, 1.0→0

    edge_of_chaos_index = (
        0.25 * criticality_score +
        0.25 * eci_lz_score +
        0.20 * coherence_score +
        0.15 * hierarchy_score +
        0.15 * dfa_score
    )

    results['edge_of_chaos_index'] = edge_of_chaos_index
    results['eci_interpretation'] = _interpret_eci(edge_of_chaos_index)

    # SEI is the main emergence_index
    results['emergence_index'] = emergence_index  # SEI
    results['structural_emergence_index'] = emergence_index  # Alias
    results['component_weights'] = weights
    results['interpretation'] = _interpret_emergence(emergence_index)

    log(f"✅ SEI (Structural): {emergence_index:.4f} | ECI (Edge of Chaos): {edge_of_chaos_index:.4f}")

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
    """Interpret Structural Emergence Index (SEI)."""
    if index < 0.3:
        return "Low structure (random, no stable laws possible)"
    elif index < 0.5:
        return "Moderate structure (some law-like patterns)"
    elif index < 0.7:
        return "High structure (good candidate for physical laws)"
    else:
        return "Very high structure (strong potential for emergent physics)"


def _interpret_eci(index: float) -> str:
    """Interpret Edge of Chaos Index (ECI)."""
    if index < 0.3:
        return "Far from criticality (too ordered or too chaotic)"
    elif index < 0.5:
        return "Approaching criticality (some computational potential)"
    elif index < 0.7:
        return "Near edge of chaos (good computational capacity)"
    else:
        return "At edge of chaos (optimal for complex emergence)"


# =============================================================================
# STREAMING ANALYSIS FOR LARGE SEQUENCES
# =============================================================================

# Numba-accelerated run-length encoding kernel
_RUN_LENGTH_KERNEL = None

def _get_run_length_numba_kernel():
    """Get Numba-compiled run-length encoder."""
    try:
        from numba import njit

        @njit(cache=True)
        def run_length_kernel(seq_array):
            """
            Calculate run lengths from binary array.
            Returns array of run lengths.
            """
            n = len(seq_array)
            if n == 0:
                return np.zeros(0, dtype=np.int32)

            # First pass: count runs
            n_runs = 1
            for i in range(1, n):
                if seq_array[i] != seq_array[i-1]:
                    n_runs += 1

            # Second pass: fill run lengths
            runs = np.zeros(n_runs, dtype=np.int32)
            run_idx = 0
            run_length = 1

            for i in range(1, n):
                if seq_array[i] == seq_array[i-1]:
                    run_length += 1
                else:
                    runs[run_idx] = run_length
                    run_idx += 1
                    run_length = 1
            runs[run_idx] = run_length

            return runs

        return run_length_kernel
    except ImportError:
        return None


def _calculate_phi_ratios_chunk(chunk_str: str, max_ratios: int = 1000) -> dict:
    """
    Calculate φ-related ratios for a chunk.
    Returns partial data that can be aggregated across chunks.

    IMPORTANT: Samples at most max_ratios to prevent memory explosion.
    Uses Numba-accelerated run-length encoding for 50-100x speedup.
    """
    global _RUN_LENGTH_KERNEL

    PHI = 1.618033988749895
    n = len(chunk_str)

    if n < 100:
        return {'run_ratios': [], 'density_ratios': [], 'phi_close_count': 0, 'total_runs': 0}

    # Convert to numpy array once
    seq_array = np.frombuffer(chunk_str.encode('ascii'), dtype=np.uint8) - ord('0')

    # Try Numba-accelerated version
    if _RUN_LENGTH_KERNEL is None:
        _RUN_LENGTH_KERNEL = _get_run_length_numba_kernel()

    if _RUN_LENGTH_KERNEL is not None:
        # Fast Numba path
        runs = _RUN_LENGTH_KERNEL(seq_array)
    else:
        # Fallback to numpy (still faster than pure Python)
        # Find transition points
        transitions = np.where(np.diff(seq_array) != 0)[0] + 1
        transitions = np.concatenate([[0], transitions, [n]])
        runs = np.diff(transitions).astype(np.int32)

    # Calculate ratios between consecutive runs - SAMPLED
    total_runs = len(runs)
    run_ratios = []
    phi_close_count = 0

    if total_runs >= 2:
        # Sample uniformly if too many runs
        step = max(1, (total_runs - 1) // max_ratios)

        # Vectorized ratio calculation for sampled indices
        indices = np.arange(0, total_runs - 1, step)
        if len(indices) > 0:
            numerators = runs[indices + 1].astype(np.float64)
            denominators = runs[indices].astype(np.float64)

            # Avoid division by zero
            valid = denominators > 0
            ratios = np.zeros(len(indices))
            ratios[valid] = numerators[valid] / denominators[valid]

            # Filter to reasonable range
            mask = (ratios > 0.1) & (ratios < 10)
            run_ratios = ratios[mask].tolist()

            # Count φ-proximity
            phi_close = np.abs(ratios[mask] - PHI) < 0.3
            inv_phi_close = np.abs(ratios[mask] - 1/PHI) < 0.2
            phi_close_count = int(np.sum(phi_close | inv_phi_close))

    # Method 2: Density ratios (already efficient with numpy)
    density_ratios = []
    phi_pos = int(n / PHI)
    if phi_pos > 10 and phi_pos < n - 10:
        left_ones = np.sum(seq_array[:phi_pos])
        right_ones = np.sum(seq_array[phi_pos:])
        left_density = left_ones / phi_pos
        right_density = right_ones / (n - phi_pos)
        if right_density > 0.01:
            density_ratios.append(left_density / right_density)

    return {
        'run_ratios': run_ratios,
        'density_ratios': density_ratios,
        'phi_close_count': phi_close_count,
        'total_runs': total_runs
    }


def _process_chunk(args):
    """Worker function for parallel chunk processing."""
    chunk_str, chunk_idx, n_chunks, verbose = args

    result = {
        'power_slope': None,
        'lz': None,
        'mi': None,
        'dfa_hurst': None,
        'hierarchy': None,
        'phi_data': None
    }

    chunk_len = len(chunk_str)
    if chunk_len < 10000:
        return result

    # 1. Power spectrum slope (fast with FFT)
    try:
        slope, _ = calculate_power_spectrum_slope(chunk_str)
        result['power_slope'] = slope
    except:
        pass

    # 2. LZ Complexity (Numba-accelerated)
    try:
        lz = calculate_lempel_ziv_complexity(chunk_str, max_sample=chunk_len, verbose=False)
        result['lz'] = lz
    except:
        pass

    # 3. Long-range MI
    try:
        mi = calculate_long_range_mutual_info(chunk_str)
        result['mi'] = mi
    except:
        pass

    # 4. DFA (Numba-accelerated)
    try:
        dfa_result = calculate_dfa(chunk_str)
        if not np.isnan(dfa_result['hurst_exponent']):
            result['dfa_hurst'] = dfa_result['hurst_exponent']
    except:
        pass

    # 5. Hierarchical entropy
    try:
        hier_result = calculate_hierarchical_block_entropy(chunk_str, verbose=False)
        result['hierarchy'] = hier_result['hierarchy_score']
    except:
        pass

    # 6. φ-tendency partial data
    try:
        result['phi_data'] = _calculate_phi_ratios_chunk(chunk_str)
    except:
        pass

    return result


def calculate_emergence_index_streaming(variant: str, iteration: int,
                                        chunk_size: int = 50_000_000,
                                        verbose: bool = True,
                                        max_cpu_percent: int = 50,
                                        force_mmap: bool = False,
                                        no_cache: bool = False) -> Dict[str, Any]:
    """
    Calculate Emergence Index by streaming through the ENTIRE sequence.

    This is the scientifically rigorous version that processes all data
    instead of sampling. Uses Numba-accelerated functions and parallel
    processing for speed.

    Args:
        variant: Variant code (A, B, D, E, F, G, H, I)
        iteration: Iteration number
        chunk_size: Bits per chunk (default 50M for memory efficiency)
        verbose: Print progress
        max_cpu_percent: Maximum CPU usage (default 50% = half of cores)
        force_mmap: Force MMAP mode even for small files (saves RAM)
        no_cache: If True, ignore and delete existing checkpoints

    Returns:
        Dictionary with emergence metrics computed over full sequence
    """
    import gzip
    import time
    import os
    from pathlib import Path
    from bitarray import bitarray
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm

    def log(msg):
        if verbose:
            print(f"   {msg}", flush=True)

    # Calculate number of workers (50% of CPU cores by default)
    total_cores = os.cpu_count() or 4
    n_workers = max(1, int(total_cores * max_cpu_percent / 100))

    log(f"🖥️ Using {n_workers} of {total_cores} CPU cores ({max_cpu_percent}% limit)")

    # Find the data file - check both new and old structure
    results_dir = Path(__file__).parent.parent / "results"

    # Standard structure: level0/phi_snapshots/var_{X}/
    base_path = results_dir / "level0" / "phi_snapshots" / f"var_{variant}"
    struct_path = base_path / f"phi_iter{iteration}.struct.gz"
    bin_path = base_path / f"phi_iter{iteration}.bin.gz"

    data_path = struct_path if struct_path.exists() else (bin_path if bin_path.exists() else None)

    if not data_path:
        return {'error': f'No data found for {variant}@{iteration}'}

    # Get file size for progress estimation
    file_size_mb = data_path.stat().st_size / (1024 * 1024)
    log(f"📂 File: {data_path.name} ({file_size_mb:.1f} MB compressed)")

    # Decide whether to use MMAP based on estimated decompressed size vs available RAM
    # Typical compression ratio for binary data: ~8-12x, use 10x as estimate
    estimated_decompressed_gb = (file_size_mb * 10) / 1024  # Estimated GB when decompressed
    available_ram_gb = psutil.virtual_memory().available / (1024 ** 3)

    # Use MMAP if:
    # 1. force_mmap is True (user override), OR
    # 2. Estimated decompressed size > 50% of available RAM
    ram_threshold_ratio = 0.5
    use_streaming_load = force_mmap or (estimated_decompressed_gb > available_ram_gb * ram_threshold_ratio)

    # Determine reason for MMAP mode
    auto_mmap = use_streaming_load and not force_mmap

    if auto_mmap:
        log(f"   📊 Estimated size: {estimated_decompressed_gb:.1f} GB > {available_ram_gb * ram_threshold_ratio:.1f} GB (50% of {available_ram_gb:.1f} GB available)")
        log(f"   ⚡ Auto-selecting MMAP mode to save RAM")

    t0 = time.time()

    if use_streaming_load:
        # MEMORY-EFFICIENT: Use mmap instead of loading all into RAM
        if auto_mmap:
            log(f"📦 Using MMAP streaming (large file detected)...")
        # Note: force_mmap message already printed in run_emergence_analysis()

        import mmap

        # Step 1: Decompress to temp file for mmap access
        temp_path = _decompress_to_temp(data_path, verbose=verbose)
        if not temp_path:
            raise RuntimeError(f"Failed to decompress {data_path} for mmap processing")

        try:
            # Step 2: Open temp file with mmap
            temp_file = open(temp_path, 'rb')
            temp_size = os.path.getsize(temp_path)
            mm = mmap.mmap(temp_file.fileno(), 0, access=mmap.ACCESS_READ)

            total_bits = temp_size * 8  # 8 bits per byte
            log(f"   Total: {total_bits:,} bits ({total_bits/1e9:.2f}B) via MMAP")

            # Chunk size in bytes (chunk_size is in bits)
            chunk_size_bytes = chunk_size // 8
            n_chunks = (temp_size + chunk_size_bytes - 1) // chunk_size_bytes

            # Process in batches to limit RAM usage
            batch_size = n_workers * 2  # Process 2 chunks per worker at a time
            log(f"⚡ Processing {n_chunks} chunks in batches of {batch_size} (MMAP mode)...")

            # Accumulators
            power_slopes = []
            lz_values = []
            mi_values = []
            dfa_hursts = []
            hierarchy_scores = []
            all_phi_run_ratios = []
            all_phi_density_ratios = []
            total_phi_close = 0  # Aggregate φ-proximity count
            total_runs = 0       # Aggregate total runs
            failed_chunks = []  # Track failed chunks for retry

            # Checkpoint configuration
            CHECKPOINT_INTERVAL = 1000  # Save every 1000 chunks
            temp_dir = Path(__file__).parent.parent / "results" / "temp"
            temp_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = temp_dir / f"emergence_checkpoint_{variant}_{iteration}.json"

            # Handle no_cache: delete existing checkpoint if requested
            if no_cache and checkpoint_path.exists():
                checkpoint_path.unlink()
                log(f"🗑️ Cache cleared (--no-cache)")

            # Try to resume from checkpoint (persistent cache)
            start_chunk = 0
            if checkpoint_path.exists():
                try:
                    import json as json_mod
                    with open(checkpoint_path, 'r') as f:
                        ckpt = json_mod.load(f)

                    # Check if checkpoint is complete (all chunks done)
                    ckpt_completed = ckpt.get('completed', 0)
                    ckpt_total = ckpt.get('n_chunks', 0)

                    if ckpt_completed >= ckpt_total and ckpt_total == n_chunks:
                        # Fully completed cache - reuse all data!
                        log(f"✅ Using cached results: {ckpt_completed}/{n_chunks} chunks (complete)")
                        power_slopes = ckpt.get('power_slopes', [])
                        lz_values = ckpt.get('lz_values', [])
                        mi_values = ckpt.get('mi_values', [])
                        dfa_hursts = ckpt.get('dfa_hursts', [])
                        hierarchy_scores = ckpt.get('hierarchy_scores', [])
                        # Skip processing entirely - go to aggregation
                        start_chunk = n_chunks  # This will skip the for loop
                    else:
                        # Partial checkpoint - resume from where we left off
                        start_chunk = ckpt_completed
                        power_slopes = ckpt.get('power_slopes', [])
                        lz_values = ckpt.get('lz_values', [])
                        mi_values = ckpt.get('mi_values', [])
                        dfa_hursts = ckpt.get('dfa_hursts', [])
                        hierarchy_scores = ckpt.get('hierarchy_scores', [])
                        log(f"📥 Resuming from checkpoint: {start_chunk}/{n_chunks} chunks")
                except Exception as e:
                    log(f"⚠️ Could not load checkpoint: {e}")

            completed = start_chunk
            base_workers = n_workers  # Store original worker count for adaptive scaling

            # Single unified progress bar for all chunks
            n_batches = max(1, (n_chunks - start_chunk + batch_size - 1) // batch_size)
            pbar_main = tqdm(total=n_chunks, initial=start_chunk,
                            desc=f"      Chunks (batch 1/{n_batches})", unit="chunk",
                            disable=not verbose, leave=True)

            for batch_idx, batch_start in enumerate(range(start_chunk, n_chunks, batch_size), 1):
                # Update description to show current batch
                pbar_main.set_description(f"      Chunks (batch {batch_idx}/{n_batches})")
                batch_end = min(batch_start + batch_size, n_chunks)

                # Adaptive worker adjustment based on RAM usage (threshold 65% for early reaction)
                current_workers = _get_adaptive_workers(base_workers, ram_threshold=65.0,
                                                         min_workers=2, verbose=verbose)

                # Wait for RAM to be available before starting batch
                _wait_for_ram(target_percent=60.0, max_wait=120, verbose=verbose)

                # Create only this batch of chunks - read from mmap (low RAM!)
                chunk_args = []
                for chunk_idx in range(batch_start, batch_end):
                    start_byte = chunk_idx * chunk_size_bytes
                    end_byte = min(start_byte + chunk_size_bytes, temp_size)
                    # Read chunk from mmap and convert to string
                    chunk_str = _read_chunk_from_mmap(mm, start_byte, end_byte)
                    chunk_args.append((chunk_str, chunk_idx, n_chunks, False))

                # Process batch in parallel with retry for failed chunks
                MAX_RETRIES = 3  # Increased from 2
                batch_failed = []

                with ProcessPoolExecutor(max_workers=current_workers) as executor:
                    futures = {executor.submit(_process_chunk, args): args for args in chunk_args}

                    # Process futures as they complete, update main progress bar
                    for future in as_completed(futures):
                        chunk_info = futures[future]
                        try:
                            result = future.result(timeout=300)  # 5 min timeout per chunk
                            if result['power_slope'] is not None:
                                power_slopes.append(result['power_slope'])
                            if result['lz'] is not None:
                                lz_values.append(result['lz'])
                            if result['mi'] is not None:
                                mi_values.append(result['mi'])
                            if result['dfa_hurst'] is not None:
                                dfa_hursts.append(result['dfa_hurst'])
                            if result['hierarchy'] is not None:
                                hierarchy_scores.append(result['hierarchy'])
                            if result['phi_data'] is not None:
                                all_phi_run_ratios.extend(result['phi_data']['run_ratios'])
                                all_phi_density_ratios.extend(result['phi_data']['density_ratios'])
                                total_phi_close += result['phi_data'].get('phi_close_count', 0)
                                total_runs += result['phi_data'].get('total_runs', 0)
                        except Exception as e:
                            # Track failed chunk for retry (capture ALL errors, not just specific ones)
                            error_msg = str(e)
                            batch_failed.append(chunk_info)
                            if verbose:
                                pbar_main.write(f"      ⚠️ Chunk {chunk_info[1]} error: {error_msg[:100]}")

                        # Update main progress bar per chunk (not per batch)
                        completed += 1
                        pbar_main.update(1)

                # Retry failed chunks with resilience system
                if batch_failed:
                    pbar_main.write(f"      🔄 Retrying {len(batch_failed)} failed chunks...")

                    for args in batch_failed:
                        chunk_success = False
                        for retry in range(MAX_RETRIES):
                            # Wait for resources before each retry attempt (silent)
                            _wait_for_ram(target_percent=55.0, max_wait=180, verbose=False)

                            try:
                                result = _process_chunk(args)
                                if result['power_slope'] is not None:
                                    power_slopes.append(result['power_slope'])
                                if result['lz'] is not None:
                                    lz_values.append(result['lz'])
                                if result['mi'] is not None:
                                    mi_values.append(result['mi'])
                                if result['dfa_hurst'] is not None:
                                    dfa_hursts.append(result['dfa_hurst'])
                                if result['hierarchy'] is not None:
                                    hierarchy_scores.append(result['hierarchy'])
                                if result['phi_data'] is not None:
                                    all_phi_run_ratios.extend(result['phi_data']['run_ratios'])
                                    all_phi_density_ratios.extend(result['phi_data']['density_ratios'])
                                    total_phi_close += result['phi_data'].get('phi_close_count', 0)
                                    total_runs += result['phi_data'].get('total_runs', 0)
                                chunk_success = True
                                pbar_main.write(f"      ✓ Chunk {args[1]} recovered")
                                break  # Success, no more retries
                            except Exception as e:
                                if retry == MAX_RETRIES - 1:
                                    failed_chunks.append(args[1])
                                    pbar_main.write(f"      ❌ Chunk {args[1]} failed permanently")

                # Force garbage collection between batches to free memory
                import gc
                gc.collect()

                # Save checkpoint every CHECKPOINT_INTERVAL chunks
                if completed % CHECKPOINT_INTERVAL < batch_size and completed > start_chunk:
                    try:
                        import json as json_mod
                        ckpt_data = {
                            'completed': completed,
                            'n_chunks': n_chunks,
                            'power_slopes': power_slopes,
                            'lz_values': lz_values,
                            'mi_values': mi_values,
                            'dfa_hursts': dfa_hursts,
                            'hierarchy_scores': hierarchy_scores,
                            'failed_chunks': failed_chunks
                        }
                        with open(checkpoint_path, 'w') as f:
                            json_mod.dump(ckpt_data, f)
                        # Use pbar.write() to avoid duplicating progress bar
                        pbar_main.write(f"      💾 Checkpoint saved: {completed}/{n_chunks}")
                    except Exception as e:
                        pbar_main.write(f"      ⚠️ Could not save checkpoint: {e}")

                # Clear chunk_args to free memory
                del chunk_args

            # Close main progress bar
            pbar_main.close()

            # Save final checkpoint as persistent cache (don't delete!)
            if completed >= n_chunks and not failed_chunks:
                try:
                    import json as json_mod
                    ckpt_data = {
                        'completed': n_chunks,
                        'n_chunks': n_chunks,
                        'power_slopes': power_slopes,
                        'lz_values': lz_values,
                        'mi_values': mi_values,
                        'dfa_hursts': dfa_hursts,
                        'hierarchy_scores': hierarchy_scores,
                        'failed_chunks': []
                    }
                    with open(checkpoint_path, 'w') as f:
                        json_mod.dump(ckpt_data, f)
                    log(f"💾 Results cached for future use")
                except Exception as e:
                    log(f"⚠️ Could not save cache: {e}")

        finally:
            # Cleanup mmap and temp file
            mm.close()
            temp_file.close()
            _cleanup_temp_file(temp_path, verbose=verbose)

    else:
        # FAST: Small file, load all at once
        log(f"   Loading into memory...")
        with gzip.open(data_path, 'rb') as f:
            data = f.read()
        ba = bitarray()
        ba.frombytes(data)
        total_bits = len(ba)

        log(f"   Loaded {total_bits:,} bits in {time.time()-t0:.1f}s")

        n_chunks = (total_bits + chunk_size - 1) // chunk_size
        log(f"⚡ Processing {n_chunks} chunks of {chunk_size//1_000_000}M bits each (parallel)...")

        # NOTE: We extract chunks on-demand per batch to avoid memory explosion
        # For iteration 27+ with 240B+ bits, pre-extracting all chunks would need ~240GB RAM

        # Accumulators
        power_slopes = []
        lz_values = []
        mi_values = []
        dfa_hursts = []
        hierarchy_scores = []
        all_phi_run_ratios = []
        all_phi_density_ratios = []
        total_phi_close = 0  # Aggregate φ-proximity count
        total_runs = 0       # Aggregate total runs
        failed_chunks = []

        # Process in adaptive batches with RAM monitoring
        completed = 0
        base_workers = n_workers
        batch_size = 50  # Process 50 chunks per batch for adaptive control
        n_batches = (n_chunks + batch_size - 1) // batch_size
        MAX_RETRIES = 3
        batch_failed_all = []  # Collect all failed chunks across batches

        for batch_idx, batch_start in enumerate(range(0, n_chunks, batch_size), 1):
            batch_end = min(batch_start + batch_size, n_chunks)

            # Extract chunks on-demand for this batch only (memory efficient)
            batch_args = []
            for chunk_idx in range(batch_start, batch_end):
                start = chunk_idx * chunk_size
                end = min(start + chunk_size, total_bits)
                chunk_str = ba[start:end].to01()
                batch_args.append((chunk_str, chunk_idx, n_chunks, False))

            # Adaptive worker adjustment based on RAM usage (threshold 65% for early reaction)
            current_workers = _get_adaptive_workers(base_workers, ram_threshold=65.0,
                                                     min_workers=2, verbose=verbose)

            # Wait for RAM to be available before starting batch
            _wait_for_ram(target_percent=60.0, max_wait=120, verbose=verbose)

            with ProcessPoolExecutor(max_workers=current_workers) as executor:
                futures = {executor.submit(_process_chunk, args): args for args in batch_args}

                # Progress bar for chunks - shows batch X/Y
                batch_desc = f"      Batch {batch_idx}/{n_batches}"
                pbar = tqdm(as_completed(futures), total=len(futures),
                            desc=batch_desc,
                            unit="chunk", leave=True, disable=not verbose)

                for future in pbar:
                    completed += 1
                    chunk_info = futures[future]

                    try:
                        result = future.result(timeout=300)
                        if result['power_slope'] is not None:
                            power_slopes.append(result['power_slope'])
                        if result['lz'] is not None:
                            lz_values.append(result['lz'])
                        if result['mi'] is not None:
                            mi_values.append(result['mi'])
                        if result['dfa_hurst'] is not None:
                            dfa_hursts.append(result['dfa_hurst'])
                        if result['hierarchy'] is not None:
                            hierarchy_scores.append(result['hierarchy'])
                        if result['phi_data'] is not None:
                            all_phi_run_ratios.extend(result['phi_data']['run_ratios'])
                            all_phi_density_ratios.extend(result['phi_data']['density_ratios'])
                            total_phi_close += result['phi_data'].get('phi_close_count', 0)
                            total_runs += result['phi_data'].get('total_runs', 0)
                    except Exception as e:
                        # Track ALL failed chunks for retry (not just specific errors)
                        batch_failed_all.append(chunk_info)
                        if verbose:
                            log(f"   ⚠️ Chunk {chunk_info[1]} error: {str(e)[:100]}")
                pbar.close()

            # Free batch strings immediately before next batch
            del batch_args
            import gc
            gc.collect()

        # Retry failed chunks with resilience system
        if batch_failed_all:
            log(f"   🔄 Retrying {len(batch_failed_all)} failed chunks with resilience mode...")

            for args in batch_failed_all:
                chunk_success = False
                for retry in range(MAX_RETRIES):
                    # Wait for resources before each retry attempt
                    log(f"      Retry {retry+1}/{MAX_RETRIES} for chunk {args[1]} (waiting for RAM <55%)...")
                    _wait_for_ram(target_percent=55.0, max_wait=180, verbose=False)

                    try:
                        result = _process_chunk(args)
                        if result['power_slope'] is not None:
                            power_slopes.append(result['power_slope'])
                        if result['lz'] is not None:
                            lz_values.append(result['lz'])
                        if result['mi'] is not None:
                            mi_values.append(result['mi'])
                        if result['dfa_hurst'] is not None:
                            dfa_hursts.append(result['dfa_hurst'])
                        if result['hierarchy'] is not None:
                            hierarchy_scores.append(result['hierarchy'])
                        if result['phi_data'] is not None:
                            all_phi_run_ratios.extend(result['phi_data']['run_ratios'])
                            all_phi_density_ratios.extend(result['phi_data']['density_ratios'])
                            total_phi_close += result['phi_data'].get('phi_close_count', 0)
                            total_runs += result['phi_data'].get('total_runs', 0)
                        chunk_success = True
                        log(f"      ✓ Chunk {args[1]} recovered successfully")
                        break  # Success, no more retries
                    except Exception as e:
                        if retry == MAX_RETRIES - 1:
                            failed_chunks.append(args[1])
                            log(f"   ❌ Chunk {args[1]} failed permanently after {MAX_RETRIES} retries: {e}")

    log(f"✅ Processed all {n_chunks} chunks in {(time.time()-t0)/60:.1f} min")

    # Aggregate results (weighted average by chunk)
    def safe_mean(values):
        return np.mean(values) if values else np.nan

    avg_slope = safe_mean(power_slopes)
    avg_lz = safe_mean(lz_values)
    avg_mi = safe_mean(mi_values)
    avg_hurst = safe_mean(dfa_hursts)
    avg_hierarchy = safe_mean(hierarchy_scores)

    # Compute φ-tendency from aggregated statistics (memory-efficient)
    PHI = 1.618033988749895
    phi_signals = []

    # Method 1: Use pre-counted φ-close ratios (from sampling)
    # This avoids creating a 35GB array!
    if total_runs > 0:
        # Estimate φ-proximity from sampled counts
        sampled_phi_ratio = total_phi_close / len(all_phi_run_ratios) if all_phi_run_ratios else 0
        phi_signals.append(sampled_phi_ratio)

    # Method 2: Analyze density ratios (already small - one per chunk)
    if all_phi_density_ratios:
        density_arr = np.array(all_phi_density_ratios)
        # Check if density ratios cluster around 1.0 (balanced) or φ
        density_mean = np.mean(np.abs(density_arr - 1.0))
        density_signal = max(0, 1 - density_mean) * 0.5  # Contribution
        phi_signals.append(density_signal)

    # Combined φ-strength
    phi_strength = np.mean(phi_signals) if phi_signals else 0.0
    phi_strength = min(1.0, phi_strength * 2)  # Scale to [0, 1]

    log(f"   🔮 φ-tendency calculated from {len(all_phi_run_ratios):,} sampled ratios ({total_runs:,} total runs)")

    # Compute scores (same logic as calculate_emergence_index)
    criticality_score = max(0, 1 - abs(avg_slope + 1) / 2) if not np.isnan(avg_slope) else 0

    # === SEI Component Scores (aligned with non-streaming mode) ===

    # ORDER: 1 - LZ (low LZ = high order = good for physical laws)
    order_score = 1.0 - avg_lz if not np.isnan(avg_lz) else 0

    # HIERARCHY: normalized to 0-1
    hierarchy_score_normalized = min(1.0, avg_hierarchy * 2) if not np.isnan(avg_hierarchy) else 0

    # COHERENCE: MI-based, scaled
    coherence_score = min(1.0, avg_mi * 10) if not np.isnan(avg_mi) else 0

    # NON-RANDOMNESS: penalize pure random (H~0.5, slope~0)
    dfa_nonrandom = max(0, (avg_hurst - 0.5) * 2) if not np.isnan(avg_hurst) else 0
    crit_nonrandom = max(0, min(1, -avg_slope)) if avg_slope < 0 else 0
    non_randomness_score = (dfa_nonrandom + crit_nonrandom) / 2

    # Composite Emergence Index - use centralized weights for consistency
    # Both normal and streaming modes now use the same weights from config
    weights = _load_emergence_weights()

    emergence_index = (
        weights['order'] * order_score +
        weights['hierarchy'] * hierarchy_score_normalized +
        weights['coherence'] * coherence_score +
        weights['non_randomness'] * non_randomness_score
    )

    return {
        'variant': variant,
        'iteration': iteration,
        'total_bits': total_bits,
        'chunks_processed': n_chunks,
        'analysis_mode': 'streaming_full',
        'emergence_index': emergence_index,
        'phi_tendency': phi_strength,
        'phi_data': {
            'run_ratios_count': len(all_phi_run_ratios),
            'density_ratios_count': len(all_phi_density_ratios),
            'phi_strength': phi_strength
        },
        # SEI components (new structure)
        'order': {
            'lz_normalized': avg_lz,
            'order_score': order_score,
            'n_samples': len(lz_values)
        },
        'hierarchy': {
            'avg_hierarchy': avg_hierarchy,
            'score': hierarchy_score_normalized,
            'n_samples': len(hierarchy_scores)
        },
        'coherence': {
            'avg_mi': avg_mi,
            'score': coherence_score,
            'n_samples': len(mi_values)
        },
        'non_randomness': {
            'dfa_component': dfa_nonrandom,
            'criticality_component': crit_nonrandom,
            'score': non_randomness_score,
        },
        # Raw metrics for reference
        'dfa': {
            'avg_hurst': avg_hurst,
            'n_samples': len(dfa_hursts)
        },
        'criticality': {
            'avg_slope': avg_slope,
            'n_samples': len(power_slopes)
        }
    }


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
    results_dir = Path(__file__).parent.parent / "results"

    # Standard structure: level0/phi_snapshots/var_{X}/
    base_path = results_dir / "level0" / "phi_snapshots" / f"var_{variant}"
    struct_path = base_path / f"phi_iter{iteration}.struct.gz"

    if struct_path.exists():
        try:
            from bitarray import bitarray
            import threading
            import time as time_module

            file_size_mb = struct_path.stat().st_size / (1024 * 1024)
            # Estimate: ~10-20 MB/s decompression speed
            est_seconds = int(file_size_mb / 15)
            print(f"   📂 File: {struct_path.name} ({file_size_mb:.1f} MB)", flush=True)
            print(f"   ⏳ Decompressing (~{est_seconds}s estimated)...", flush=True)

            # Progress indicator in background thread
            stop_spinner = threading.Event()
            def spinner():
                start = time_module.time()
                chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
                i = 0
                while not stop_spinner.is_set():
                    elapsed = int(time_module.time() - start)
                    print(f"\r      {chars[i % len(chars)]} {elapsed}s elapsed...", end="", flush=True)
                    time_module.sleep(0.2)
                    i += 1
                print(f"\r      ✓ Decompression complete ({elapsed}s)   ", flush=True)

            spinner_thread = threading.Thread(target=spinner, daemon=True)
            spinner_thread.start()

            with gzip.open(struct_path, 'rb') as f:
                data = f.read()

            stop_spinner.set()
            spinner_thread.join(timeout=1)

            ba = bitarray()
            ba.frombytes(data)
            total_bits = len(ba)

            # Sample from middle for representative data
            if total_bits > max_bits:
                start = (total_bits - max_bits) // 2
                sample = ba[start:start + max_bits].to01()
                print(f"   ✓ Sampled {max_bits:,} bits from middle of {total_bits:,} total", flush=True)
                return sample
            else:
                print(f"   ✓ Loaded {total_bits:,} bits", flush=True)
                return ba.to01()
        except Exception as e:
            print(f"Warning: Could not load {struct_path}: {e}")

    # Try bin.gz format
    bin_path = base_path / f"phi_iter{iteration}.bin.gz"
    if bin_path.exists():
        try:
            from bitarray import bitarray
            import threading
            import time as time_module

            file_size_mb = bin_path.stat().st_size / (1024 * 1024)
            est_seconds = int(file_size_mb / 15)
            print(f"   📂 File: {bin_path.name} ({file_size_mb:.1f} MB)", flush=True)
            print(f"   ⏳ Decompressing (~{est_seconds}s estimated)...", flush=True)

            stop_spinner = threading.Event()
            def spinner():
                start = time_module.time()
                chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
                i = 0
                while not stop_spinner.is_set():
                    elapsed = int(time_module.time() - start)
                    print(f"\r      {chars[i % len(chars)]} {elapsed}s elapsed...", end="", flush=True)
                    time_module.sleep(0.2)
                    i += 1
                print(f"\r      ✓ Decompression complete ({elapsed}s)   ", flush=True)

            spinner_thread = threading.Thread(target=spinner, daemon=True)
            spinner_thread.start()

            with gzip.open(bin_path, 'rb') as f:
                data = f.read()

            stop_spinner.set()
            spinner_thread.join(timeout=1)

            ba = bitarray()
            ba.frombytes(data)
            total_bits = len(ba)

            if total_bits > max_bits:
                start = (total_bits - max_bits) // 2
                sample = ba[start:start + max_bits].to01()
                print(f"   ✓ Sampled {max_bits:,} bits from middle of {total_bits:,} total", flush=True)
                return sample
            else:
                print(f"   ✓ Loaded {total_bits:,} bits", flush=True)
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
                        help='Sample size for large sequences (default: 1M, ignored if --streaming)')
    parser.add_argument('--streaming', '-s', action='store_true',
                        help='Process ENTIRE sequence (rigorous but slow, uses Numba + parallel)')
    parser.add_argument('--chunk-size', type=int, default=50_000_000,
                        help='Bits per chunk in streaming mode (default: 50M)')
    parser.add_argument('--max-cpu', type=int, default=50,
                        help='Maximum CPU usage percentage for parallel streaming (default: 50)')
    parser.add_argument('--force-mmap', action='store_true',
                        help='Force MMAP mode even for small files (saves RAM, useful for low-memory systems)')
    parser.add_argument('--no-cache', action='store_true',
                        help='Ignore and delete existing checkpoints/cache (force full recalculation)')
    parser.add_argument('--null-test', action='store_true',
                        help='Run statistical significance test against null model')
    parser.add_argument('--null-samples', type=int, default=100,
                        help='Number of null model samples for significance testing (default: 100)')

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
        mode_str = " [STREAMING - full sequence]" if args.streaming else ""
        print(f"\n📊 Analyzing variant {args.variant}, iteration {args.iteration}{mode_str}\n")

        if args.streaming:
            # STREAMING MODE: Process entire sequence with parallel processing
            results = calculate_emergence_index_streaming(
                args.variant, args.iteration,
                chunk_size=args.chunk_size,
                max_cpu_percent=args.max_cpu,
                force_mmap=args.force_mmap,
                no_cache=args.no_cache
            )
            if 'error' in results:
                print(f"❌ {results['error']}")
                return
        else:
            # SAMPLE MODE: Load sequence and sample
            print(f"   Loading Φ sequence...", flush=True)
            seq = load_phi_sequence(args.variant, args.iteration)
            if seq is None:
                print(f"❌ Could not load variant {args.variant} iteration {args.iteration}")
                return

            print(f"   ✓ Loaded {len(seq):,} bits", flush=True)
            print(f"\n   Calculating emergence metrics...", flush=True)
            results = calculate_emergence_index(seq, args.sample_size, verbose=True)

        # Print detailed results - STRUCTURAL EMERGENCE INDEX (SEI)
        print("\n" + "=" * 60)
        print("🌌 STRUCTURAL EMERGENCE INDEX (SEI) ANALYSIS")
        print("=" * 60)
        print("   [Designed to detect emergence of PHYSICAL LAWS, not edge-of-chaos]")

        if args.streaming:
            print(f"\n📏 Full sequence: {results.get('total_bits', 'N/A'):,} bits")
            print(f"   Chunks processed: {results.get('chunks_processed', 'N/A')}")
        else:
            print(f"\n📏 Sample: {results['sample_size']:,} / {results['sequence_length']:,} bits")

        # 1. ORDER (30%) - Compressibility = physical laws
        if 'order' in results:
            o = results['order']
            print(f"\n📐 ORDER (30%) — Compressibility = Physical Laws")
            print(f"   LZ normalized: {o['lz_normalized']:.3f}")
            print(f"   Order score: {o['order_score']:.3f} (1 - LZ)")
            print(f"   → Lower LZ = more ordered = potential for stable laws")
        else:
            x = results['complexity']
            order_score = 1.0 - x['lz_normalized']
            print(f"\n📐 ORDER (30%) — Compressibility = Physical Laws")
            print(f"   LZ normalized: {x['lz_normalized']:.3f}")
            print(f"   Order score: {order_score:.3f} (1 - LZ)")
            print(f"   → {x['interpretation']}")

        # 2. HIERARCHY (30%) - Multi-scale structure
        if 'hierarchy' in results:
            hr = results['hierarchy']
            print(f"\n📊 HIERARCHY (30%) — Multi-scale Structure")
            print(f"   Hierarchy score: {hr['raw_hierarchy_score']:.4f}")
            print(f"   Mean entropy: {hr['mean_entropy']:.3f}")
            print(f"   Score: {hr['score']:.3f}")
            print(f"   → {hr['interpretation']}")

        # 3. COHERENCE (20%) - Long-range correlations
        h = results['coherence']
        print(f"\n🔗 COHERENCE (20%) — Long-range Correlations")
        print(f"   MI ratio: {h['mi_ratio']:.4f}")
        print(f"   Score: {h['score']:.3f}")
        print(f"   → {h['interpretation']}")

        # 4. NON-RANDOMNESS (20%) - DFA + Criticality combined
        if 'non_randomness' in results:
            nr = results['non_randomness']
            print(f"\n🎲 NON-RANDOMNESS (20%) — Penalizes Pure Random")
            print(f"   DFA component: {nr['dfa_component']:.3f}")
            print(f"   Criticality component: {nr['criticality_component']:.3f}")
            print(f"   Score: {nr['score']:.3f}")
            print(f"   → {nr['interpretation']}")
        else:
            # Fallback to old format
            c = results['criticality']
            print(f"\n⚡ CRITICALITY (spectrum analysis)")
            print(f"   Slope: {c['slope']:.3f} (target: {c['target']})")
            print(f"   → {c['interpretation']}")
            if 'dfa' in results:
                dfa = results['dfa']
                print(f"\n📈 DFA (Hurst exponent)")
                print(f"   Hurst H: {dfa['hurst_exponent']:.3f}")
                print(f"   → {dfa['interpretation']}")

        # Final scores: SEI and ECI
        print(f"\n{'=' * 60}")
        print("📊 DUAL INDEX SUMMARY")
        print("=" * 60)

        sei = results['emergence_index']
        eci = results.get('edge_of_chaos_index', 0)

        # SEI interpretation
        if sei >= 0.7:
            sei_emoji = "🌟"
        elif sei >= 0.5:
            sei_emoji = "✨"
        elif sei >= 0.3:
            sei_emoji = "📐"
        else:
            sei_emoji = "⚠️"

        # ECI interpretation
        if eci >= 0.7:
            eci_emoji = "🔥"
        elif eci >= 0.5:
            eci_emoji = "⚡"
        elif eci >= 0.3:
            eci_emoji = "〰️"
        else:
            eci_emoji = "❄️"

        print(f"\n{sei_emoji} SEI (Structural Emergence): {sei:.4f}")
        print(f"   → {results['interpretation']}")
        print(f"   [Measures: order + hierarchy + coherence → physical laws potential]")

        print(f"\n{eci_emoji} ECI (Edge of Chaos): {eci:.4f}")
        print(f"   → {results.get('eci_interpretation', 'N/A')}")
        print(f"   [Measures: criticality + complexity balance → computational capacity]")

        # Combined interpretation - more descriptive and honest
        print(f"\n{'─' * 60}")
        print("📋 ANALYSIS:")

        # Get non-randomness score
        nr_score = results.get('non_randomness', {}).get('score', 0)
        order_score = results.get('order', {}).get('order_score', 1 - results['complexity']['lz_normalized'])

        # Describe structure quality
        if order_score > 0.9:
            print(f"   • Order: Very high (LZ={1-order_score:.3f}) — highly compressible")
        elif order_score > 0.7:
            print(f"   • Order: High — significant compressibility")
        elif order_score > 0.5:
            print(f"   • Order: Moderate — some structure")
        else:
            print(f"   • Order: Low (LZ={1-order_score:.3f}) — near-random, incompressible")

        # Describe randomness
        if nr_score < 0.05:
            hurst = results.get('dfa', {}).get('hurst_exponent', 0.5)
            if hurst < 0.4:
                print(f"   • Dynamics: Anti-persistent (H={hurst:.2f}) — fluctuations revert")
            elif hurst > 0.6:
                print(f"   • Dynamics: Persistent (H={hurst:.2f}) — trends continue")
            else:
                print(f"   • Dynamics: White noise (H={hurst:.2f}) — no memory")
        else:
            print(f"   • Dynamics: Non-random ({nr_score:.0%}) — deterministic patterns")

        # Final assessment
        print(f"\n{'─' * 60}")
        if sei >= 0.6 and eci >= 0.5 and nr_score > 0.05:
            print("🎯 PROFILE: Structured + Dynamic + Non-random")
            print("   Strong candidate for emergent physical laws")
        elif sei >= 0.6 and eci >= 0.5 and nr_score <= 0.05:
            print("📐 PROFILE: Structured + Dynamic, but determinism unclear")
            print("   Has order and complexity, but dynamics may be stochastic")
        elif sei >= 0.5 and eci >= 0.6 and order_score < 0.6:
            print("🌀 PROFILE: Edge-of-chaos dynamics, but low structure")
            print("   Computational capacity high, but may be organized noise")
        elif sei >= 0.7 and eci < 0.5:
            print("❄️ PROFILE: Highly ordered but static")
            print("   Strong structure, but possibly too rigid for emergence")
        elif sei < 0.5 and eci < 0.5:
            print("❓ PROFILE: Low structure and low dynamics")
            print("   Unlikely to generate meaningful emergence")
        else:
            print(f"📊 PROFILE: Mixed (SEI={sei:.2f}, ECI={eci:.2f})")
            print("   Requires further analysis to characterize")
        print("=" * 60)

        # Statistical significance test against null model
        if args.null_test and not args.streaming:
            print(f"\n🔬 STATISTICAL SIGNIFICANCE TEST")
            print(f"   Generating {args.null_samples} null samples (shuffled)...")

            from .statistical_tests import compare_to_null_model

            # Test emergence index
            ei_test = compare_to_null_model(
                phi_sequence=seq,
                metric_fn=lambda s: calculate_emergence_index(s, args.sample_size, verbose=False)['emergence_index'],
                n_samples=args.null_samples,
                null_type='shuffle',
                alternative='greater'
            )

            print(f"\n   📊 Emergence Index vs Null Model:")
            print(f"      Observed: {ei_test['observed']:.4f}")
            print(f"      Null mean: {ei_test['null_mean']:.4f} ± {ei_test['null_std']:.4f}")
            print(f"      z-score: {ei_test['z_score']:.2f}")
            print(f"      p-value: {ei_test['p_value']:.4f}")
            print(f"      → {ei_test['significance']}")

            # Add to results
            results['statistical_test'] = {
                'emergence_index': ei_test,
                'null_type': 'shuffle',
                'n_samples': args.null_samples
            }

    # Save results (auto-generate filename if not specified)
    if args.output:
        output_path = Path(args.output)
    else:
        # Auto-generate filename using unified structure
        from utils.file_saver import get_output_path, relative_path
        if args.compare and args.variants:
            variants_str = "_".join(args.variants)
            filename = f"emergence_{variants_str}_iter{args.iteration}.json"
        else:
            variant = args.variant or (args.variants[0] if args.variants else "unknown")
            filename = f"emergence_{variant}_iter{args.iteration}.json"
        output_path = get_output_path(1, "metrics", filename)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Log with relative path
    try:
        from utils.file_saver import relative_path
        print(f"\n💾 Saved: {relative_path(output_path)}")
    except ImportError:
        print(f"\n💾 Results saved to {output_path}")


if __name__ == '__main__':
    main()

