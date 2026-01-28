"""
🔬 HSI Level 2 - Local TDA Analysis (Windowed)

Performs TDA on sliding windows with parallel processing.
Detects LOCAL topological features: cycles (β₁), connectivity (β₀).

Author: Sophia (AI Assistant)
Date: January 2026
"""

import numpy as np
import time
from typing import Dict, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

try:
    import ripser
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False


@njit(cache=True)
def _takens_embedding_window(seq_array: np.ndarray, start_idx: int,
                              window_size: int, embedding_dim: int,
                              time_delay: int, max_points: int) -> np.ndarray:
    """Create Takens embedding for a single window."""
    max_vectors = window_size - (embedding_dim - 1) * time_delay
    if max_vectors <= 0:
        return np.empty((0, embedding_dim), dtype=np.float32)

    step = max_vectors // max_points if max_vectors > max_points else 1
    n_points = max_points if max_vectors > max_points else max_vectors

    result = np.empty((n_points, embedding_dim), dtype=np.float32)
    for i in range(n_points):
        idx = start_idx + i * step
        for j in range(embedding_dim):
            result[i, j] = seq_array[idx + j * time_delay]
    return result


def analyze_single_window(args: Tuple) -> Dict[str, Any]:
    """Analyze a single window. Designed for ProcessPoolExecutor."""
    (window_idx, seq_array, start_idx, window_size,
     embedding_dim, time_delay, max_points) = args

    result = {'window_idx': window_idx, 'start_idx': start_idx,
              'window_size': window_size, 'betti': None, 'error': None}

    try:
        point_cloud = _takens_embedding_window(
            seq_array, start_idx, window_size,
            embedding_dim, time_delay, max_points)

        if point_cloud.shape[0] < 10:
            result['error'] = 'too few points'
            return result

        if HAS_RIPSER:
            diagrams = ripser.ripser(point_cloud, maxdim=2)['dgms']
            betti = {}
            for dim, dgm in enumerate(diagrams):
                if len(dgm) > 0:
                    finite = dgm[np.isfinite(dgm[:, 1])]
                    if len(finite) > 0:
                        lifetimes = finite[:, 1] - finite[:, 0]
                        threshold = np.median(lifetimes) if len(lifetimes) > 0 else 0
                        betti[f'beta_{dim}'] = int(np.sum(lifetimes > threshold))
                    else:
                        betti[f'beta_{dim}'] = 0
                else:
                    betti[f'beta_{dim}'] = 0
            result['betti'] = betti
            result['n_points'] = point_cloud.shape[0]
        else:
            result['error'] = 'ripser not available'
    except Exception as e:
        result['error'] = str(e)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LOCAL ANALYSIS FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def run_local_tda(sequence: np.ndarray,
                  n_windows: int = 10,
                  embedding_dim: int = 3,
                  time_delay: int = 1,
                  max_points_per_window: int = 5000,
                  n_workers: int = None,
                  verbose: bool = True) -> Dict[str, Any]:
    """
    Run windowed TDA analysis with parallel processing.

    Args:
        sequence: Binary sequence as numpy array (0s and 1s)
        n_windows: Number of windows to analyze
        embedding_dim: Dimension for Takens embedding
        time_delay: Time delay τ for embedding
        max_points_per_window: Max points per window
        n_workers: Number of parallel workers (None = auto)
        verbose: Print progress

    Returns:
        Dictionary with aggregated results and per-window details
    """
    log = print if verbose else lambda *a, **k: None

    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)

    n = len(sequence)
    window_size = n // n_windows

    log(f"\n   🔬 LOCAL TDA ANALYSIS (Windowed)")
    log(f"      Sequence length: {n:,}")
    log(f"      Windows: {n_windows}")
    log(f"      Window size: {window_size:,}")
    log(f"      Points per window: {max_points_per_window:,}")
    log(f"      Workers: {n_workers}")

    if sequence.dtype != np.float32:
        sequence = sequence.astype(np.float32)

    # Prepare window arguments
    window_args = []
    for i in range(n_windows):
        start_idx = i * window_size
        ws = window_size if i < n_windows - 1 else n - start_idx
        window_args.append((i, sequence, start_idx, ws,
                           embedding_dim, time_delay, max_points_per_window))

    t_start = time.perf_counter()
    results = []

    log(f"\n      Processing windows...")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(analyze_single_window, args): args[0]
                   for args in window_args}

        for future in tqdm(as_completed(futures), total=n_windows,
                          desc="      Windows", disable=not HAS_TQDM or not verbose):
            results.append(future.result())

    results.sort(key=lambda x: x['window_idx'])
    elapsed = time.perf_counter() - t_start

    # Aggregate statistics
    betti_0 = [r['betti']['beta_0'] for r in results if r['betti']]
    betti_1 = [r['betti']['beta_1'] for r in results if r['betti']]
    betti_2 = [r['betti'].get('beta_2', 0) for r in results if r['betti']]

    aggregated = {
        'n_windows': n_windows,
        'n_successful': len(betti_0),
        'n_failed': n_windows - len(betti_0),
        'window_size': window_size,
        'elapsed_s': elapsed,
        'beta_0': {'mean': float(np.mean(betti_0)), 'std': float(np.std(betti_0))} if betti_0 else None,
        'beta_1': {'mean': float(np.mean(betti_1)), 'std': float(np.std(betti_1))} if betti_1 else None,
        'beta_2': {'mean': float(np.mean(betti_2)), 'std': float(np.std(betti_2))} if betti_2 else None,
        'windows': results
    }

    log(f"\n      ✓ Completed in {elapsed:.2f}s")
    log(f"      Successful: {len(betti_0)}/{n_windows}")
    if betti_1:
        log(f"      β₁ (cycles): {np.mean(betti_1):.1f} ± {np.std(betti_1):.1f}")
    if betti_2:
        log(f"      β₂ (surfaces): {np.mean(betti_2):.1f} ± {np.std(betti_2):.1f}")

    return aggregated
