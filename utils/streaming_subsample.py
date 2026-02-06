#!/usr/bin/env python3
"""
Streaming Multi-Scale Subsampling for HSI Analysis
===================================================

Memory-efficient subsampling that reads .struct.gz files in streaming mode
without loading the entire sequence into RAM.

Sampling strategy (Uniform Stride - v2.0):
- Fine scale (1-1k patterns):     every bit,    first 100k bits
- Medium scale (1k-1M patterns):  stride = max(100, total_bits // 1_000_000) → ~1M samples across ENTIRE file
- Large scale (1M-1G patterns):   stride = max(10_000, total_bits // 10_000_000) → ~10M samples across ENTIRE file
- Huge scale (1G+ patterns):      stride = max(1_000_000, total_bits // 100_000) → ~100k samples across ENTIRE file

v2.0 Change (2026-02-06): Fixed "Frozen Start" bias detected by Antigra peer review.
    Previous version had fixed MAX_IDX limits that stopped sampling after first 100M bits.
    Now uses uniform stride calculated from total file size to cover ENTIRE sequence.

Author: Sophia (Augment Agent) with Iban Borràs
Date: February 2026
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


def streaming_multiscale_subsample(
    struct_gz_path: str,
    verbose: bool = True,
    total_chars_hint: Optional[int] = None
) -> Dict[str, Any]:
    """
    Create multi-scale subsamples from a .struct.gz file in streaming mode.
    
    Reads the file once, collecting samples at different intervals without
    loading the entire sequence into memory.
    
    Args:
        struct_gz_path: Path to .struct.gz file
        verbose: Print progress info
        total_chars_hint: Optional hint for total chars (for progress bar)
    
    Returns:
        Dict with 'fine', 'medium', 'large', 'huge' numpy arrays
        and 'metadata' with sampling info
    """
    from .streaming_phi_loader import StreamingPhiLoader
    
    loader = StreamingPhiLoader(struct_gz_path)
    meta = loader.metadata
    
    # Get total size estimate
    if total_chars_hint:
        total_bits_estimate = total_chars_hint
    else:
        total_bits_estimate = meta.get('observable_len') or meta.get('phi_length') or meta.get('sequence_length', 0)

    # === v2.0: UNIFORM STRIDE CALCULATION ===
    # Calculate strides based on total file size to cover ENTIRE sequence uniformly
    # This fixes the "Frozen Start" bias where previous version only sampled first 100M bits

    FINE_STEP = 1
    FINE_MAX = 100_000  # Fine always takes first 100k bits (no change needed)

    # Medium: target ~1M samples uniformly distributed
    # stride = max(100, total_bits / 1_000_000)
    if total_bits_estimate > 0:
        MEDIUM_STEP = max(100, total_bits_estimate // 1_000_000)
        LARGE_STEP = max(10_000, total_bits_estimate // 10_000_000)
        HUGE_STEP = max(1_000_000, total_bits_estimate // 100_000)
    else:
        # Fallback if we don't know the size (shouldn't happen)
        MEDIUM_STEP = 100
        LARGE_STEP = 10_000
        HUGE_STEP = 1_000_000

    if verbose:
        print(f"\n📐 Streaming Multi-Scale Subsampling (v2.0 - Uniform Stride)")
        print(f"   File: {Path(struct_gz_path).name}")
        print(f"   Estimated bits: {total_bits_estimate:,}")
        print(f"   Dynamic strides: medium={MEDIUM_STEP:,}, large={LARGE_STEP:,}, huge={HUGE_STEP:,}")
    
    # Collectors
    fine_samples = []
    medium_samples = []
    large_samples = []
    huge_samples = []
    
    # Counters (use Python int, not numpy, to avoid overflow)
    bit_index = 0
    next_medium = 0
    next_large = 0
    next_huge = 0
    
    # Progress bar
    pbar = None
    if verbose and TQDM_AVAILABLE and total_bits_estimate > 0:
        pbar = tqdm(total=total_bits_estimate, unit='bits', desc='   Streaming')

    # Stream through observable bits - FAST numpy version (100x faster)
    for chunk_arr in loader.iter_bits_numpy(chunk_bytes=50_000_000):  # 50MB raw = ~100M bits
        chunk_len = len(chunk_arr)
        chunk_start = bit_index
        chunk_end = bit_index + chunk_len

        # Fine: first 100k bits
        if chunk_start < FINE_MAX:
            fine_end = min(FINE_MAX - chunk_start, chunk_len)
            fine_samples.extend(chunk_arr[:fine_end].tolist())

        # Medium: uniform stride across ENTIRE file (v2.0 fix)
        if chunk_start <= next_medium < chunk_end:
            start_offset = next_medium - chunk_start
            indices = np.arange(start_offset, chunk_len, MEDIUM_STEP)
            if len(indices) > 0:
                medium_samples.extend(chunk_arr[indices].tolist())
                next_medium = chunk_start + int(indices[-1]) + MEDIUM_STEP
        elif next_medium >= chunk_end:
            pass  # Haven't reached next sample point yet
        else:
            # next_medium is before chunk_start, need to catch up
            while next_medium < chunk_start:
                next_medium += MEDIUM_STEP

        # Large: uniform stride across ENTIRE file (v2.0 fix)
        if chunk_start <= next_large < chunk_end:
            start_offset = next_large - chunk_start
            indices = np.arange(start_offset, chunk_len, LARGE_STEP)
            if len(indices) > 0:
                large_samples.extend(chunk_arr[indices].tolist())
                next_large = chunk_start + int(indices[-1]) + LARGE_STEP
        elif next_large >= chunk_end:
            pass  # Haven't reached next sample point yet
        else:
            while next_large < chunk_start:
                next_large += LARGE_STEP

        # Huge: uniform stride across ENTIRE file
        if chunk_start <= next_huge < chunk_end:
            start_offset = next_huge - chunk_start
            indices = np.arange(start_offset, chunk_len, HUGE_STEP)
            if len(indices) > 0:
                huge_samples.extend(chunk_arr[indices].tolist())
                next_huge = chunk_start + int(indices[-1]) + HUGE_STEP
        elif next_huge >= chunk_end:
            pass  # Haven't reached next sample point yet
        else:
            while next_huge < chunk_start:
                next_huge += HUGE_STEP

        bit_index = chunk_end

        if pbar:
            pbar.update(chunk_len)

    if pbar:
        pbar.close()
    
    total_bits = bit_index
    
    # Convert to numpy arrays
    samples = {
        'fine': np.array(fine_samples, dtype=np.uint8),
        'medium': np.array(medium_samples, dtype=np.uint8),
        'large': np.array(large_samples, dtype=np.uint8),
        'huge': np.array(huge_samples, dtype=np.uint8)
    }

    # Create metadata (v2.0: all levels now cover ENTIRE sequence)
    metadata = {
        'total_bits': total_bits,
        'sampling_version': '2.0_uniform_stride',
        'levels': {
            'fine': {
                'step': FINE_STEP,
                'count': len(samples['fine']),
                'covers_range': f"0-{min(FINE_MAX, total_bits):,}",
                'detects_patterns': "1-1k bits",
                'coverage': 'first_100k'
            },
            'medium': {
                'step': MEDIUM_STEP,
                'count': len(samples['medium']),
                'covers_range': f"0-{total_bits:,}",  # v2.0: now covers ENTIRE file
                'detects_patterns': "1k-1M bits",
                'coverage': 'uniform_entire_file'
            },
            'large': {
                'step': LARGE_STEP,
                'count': len(samples['large']),
                'covers_range': f"0-{total_bits:,}",  # v2.0: now covers ENTIRE file
                'detects_patterns': "1M-1G bits",
                'coverage': 'uniform_entire_file'
            },
            'huge': {
                'step': HUGE_STEP,
                'count': len(samples['huge']),
                'covers_range': f"0-{total_bits:,}",
                'detects_patterns': "1G+ bits",
                'coverage': 'uniform_entire_file'
            }
        }
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
            print(f"   │ {level_name:11} │ {info['step']:8,} │ {info['count']:11,} │ {info['detects_patterns']:16} │")
        print(f"   └─────────────┴──────────┴─────────────┴──────────────────┘")
        print(f"   Total: {total_bits:,} bits → {total_samples:,} samples ({memory_bytes / 1_000_000:.1f} MB)")

    return {
        'samples': samples,
        'metadata': metadata
    }

