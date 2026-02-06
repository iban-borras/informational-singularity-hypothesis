"""
🌐 HSI Level 2 - Global TDA Analysis (Landmarks)

Performs Topological Data Analysis on the entire sequence using
intelligent landmark subsampling to preserve topological structure.

This detects GLOBAL topological features:
- Emergent surfaces (β₂) - indicates spacetime-like structure
- Large-scale cycles (β₁) - indicates orbital dynamics
- Global connectivity (β₀)

Supports three TDA backends for optimization:
- ripser: Fast, good for small-medium datasets
- giotto: collapse_edges optimization for faster computation
- gudhi: Most efficient for large datasets

Author: Sophia (AI Assistant)
Date: January 2026
"""

import numpy as np
import time
import sys
from typing import Dict, Any, Optional, List, Literal

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

# TDA backends
try:
    import ripser
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False

try:
    from gtda.homology import VietorisRipsPersistence
    HAS_GIOTTO = True
except ImportError:
    HAS_GIOTTO = False

try:
    import gudhi
    HAS_GUDHI = True
except ImportError:
    HAS_GUDHI = False


@njit(cache=True)
def _takens_embedding_global(seq_array: np.ndarray, indices: np.ndarray,
                              embedding_dim: int, time_delay: int) -> np.ndarray:
    """Create Takens embedding for selected landmark indices."""
    n_points = len(indices)
    result = np.empty((n_points, embedding_dim), dtype=np.float32)

    for i in range(n_points):
        idx = indices[i]
        for j in range(embedding_dim):
            result[i, j] = seq_array[idx + j * time_delay]

    return result


def select_landmarks_uniform(n_total: int, n_landmarks: int,
                              embedding_dim: int, time_delay: int) -> np.ndarray:
    """Select uniformly distributed landmarks."""
    max_idx = n_total - (embedding_dim - 1) * time_delay
    if max_idx <= n_landmarks:
        return np.arange(max_idx, dtype=np.int64)

    step = max_idx // n_landmarks
    return np.arange(0, max_idx, step, dtype=np.int64)[:n_landmarks]


def select_landmarks_random(n_total: int, n_landmarks: int,
                            embedding_dim: int, time_delay: int,
                            seed: int = 42) -> np.ndarray:
    """Select random landmarks (reproducible with seed)."""
    max_idx = n_total - (embedding_dim - 1) * time_delay
    if max_idx <= n_landmarks:
        return np.arange(max_idx, dtype=np.int64)

    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(max_idx, size=n_landmarks, replace=False)).astype(np.int64)


# ═══════════════════════════════════════════════════════════════════════════
# TDA BACKEND IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════════════════

def _compute_ph_ripser(point_cloud: np.ndarray, max_dim: int) -> Dict[str, Any]:
    """Compute persistent homology using ripser (default)."""
    diagrams = ripser.ripser(point_cloud, maxdim=max_dim)['dgms']
    return _extract_betti_from_diagrams(diagrams)


def _compute_ph_giotto(point_cloud: np.ndarray, max_dim: int) -> Dict[str, Any]:
    """Compute persistent homology using giotto-tda with collapse_edges."""
    # Giotto expects 3D array: (n_samples, n_points, n_dims)
    pc_3d = point_cloud.reshape(1, *point_cloud.shape)

    vr = VietorisRipsPersistence(
        homology_dimensions=list(range(max_dim + 1)),
        collapse_edges=True,  # Key optimization!
        n_jobs=-1
    )
    diagrams = vr.fit_transform(pc_3d)[0]

    # Convert giotto format to standard format
    result = {'betti': {}, 'persistence_stats': {}}
    for dim in range(max_dim + 1):
        dim_mask = diagrams[:, 2] == dim
        dgm = diagrams[dim_mask, :2]

        if len(dgm) > 0:
            finite = dgm[np.isfinite(dgm[:, 1])]
            if len(finite) > 0:
                lifetimes = finite[:, 1] - finite[:, 0]
                threshold = np.median(lifetimes) if len(lifetimes) > 0 else 0
                significant = int(np.sum(lifetimes > threshold))

                result['betti'][f'beta_{dim}'] = significant
                result['persistence_stats'][f'dim_{dim}'] = {
                    'total_features': len(dgm),
                    'significant_features': significant,
                    'max_lifetime': float(np.max(lifetimes)),
                    'mean_lifetime': float(np.mean(lifetimes)),
                    'std_lifetime': float(np.std(lifetimes))
                }
            else:
                result['betti'][f'beta_{dim}'] = 0
                result['persistence_stats'][f'dim_{dim}'] = {'total_features': 0}
        else:
            result['betti'][f'beta_{dim}'] = 0
            result['persistence_stats'][f'dim_{dim}'] = {'total_features': 0}

    return result


def _compute_ph_gudhi(point_cloud: np.ndarray, max_dim: int) -> Dict[str, Any]:
    """Compute persistent homology using GUDHI (most efficient for large data)."""
    # Create Rips complex
    rips = gudhi.RipsComplex(points=point_cloud, max_edge_length=2.0)
    simplex_tree = rips.create_simplex_tree(max_dimension=max_dim + 1)

    # Compute persistence
    simplex_tree.compute_persistence()

    result = {'betti': {}, 'persistence_stats': {}}
    for dim in range(max_dim + 1):
        intervals = simplex_tree.persistence_intervals_in_dimension(dim)

        if len(intervals) > 0:
            finite = intervals[np.isfinite(intervals[:, 1])]
            if len(finite) > 0:
                lifetimes = finite[:, 1] - finite[:, 0]
                threshold = np.median(lifetimes) if len(lifetimes) > 0 else 0
                significant = int(np.sum(lifetimes > threshold))

                result['betti'][f'beta_{dim}'] = significant
                result['persistence_stats'][f'dim_{dim}'] = {
                    'total_features': len(intervals),
                    'significant_features': significant,
                    'max_lifetime': float(np.max(lifetimes)),
                    'mean_lifetime': float(np.mean(lifetimes)),
                    'std_lifetime': float(np.std(lifetimes))
                }
            else:
                result['betti'][f'beta_{dim}'] = 0
                result['persistence_stats'][f'dim_{dim}'] = {'total_features': 0}
        else:
            result['betti'][f'beta_{dim}'] = 0
            result['persistence_stats'][f'dim_{dim}'] = {'total_features': 0}

    return result


# ═══════════════════════════════════════════════════════════════════════════
# SCALABLE TDA METHODS (for large datasets)
# ═══════════════════════════════════════════════════════════════════════════

def _compute_ph_witness(landmarks: np.ndarray, witnesses: np.ndarray,
                        max_dim: int = 2, max_alpha_square: float = 0.1,
                        verbose: bool = True) -> Dict[str, Any]:
    """
    Compute persistent homology using Witness Complex (GUDHI).

    Much more scalable than Rips: O(landmarks² × witnesses) vs O(n³)

    Args:
        landmarks: Small set of landmark points (500-2000)
        witnesses: Large set of witness points (10000-100000)
        max_dim: Maximum homology dimension
        max_alpha_square: Maximum alpha² for simplex tree (smaller = faster)
        verbose: Print progress

    Returns:
        Dictionary with Betti numbers and persistence stats
    """
    log = print if verbose else lambda *a, **k: None

    if not HAS_GUDHI:
        return {'betti': {}, 'persistence_stats': {}, 'error': 'GUDHI not installed'}

    # Create witness complex
    log(f"         Creating witness complex...")
    t0 = time.perf_counter()
    witness_complex = gudhi.EuclideanWitnessComplex(
        witnesses=witnesses,
        landmarks=landmarks
    )
    log(f"         ✓ Witness complex created in {time.perf_counter()-t0:.2f}s")

    # Create simplex tree with SMALL alpha (critical for binary data!)
    # Binary data in 3D: points are in [0,1]³, max distance = √3 ≈ 1.73
    # Using small alpha to avoid connecting everything
    log(f"         Creating simplex tree (max_alpha²={max_alpha_square})...")
    t0 = time.perf_counter()
    simplex_tree = witness_complex.create_simplex_tree(
        max_alpha_square=max_alpha_square,
        limit_dimension=max_dim + 1
    )
    n_simplices = simplex_tree.num_simplices()
    log(f"         ✓ Simplex tree: {n_simplices:,} simplices in {time.perf_counter()-t0:.2f}s")

    # Safety check: if too many simplices, abort
    if n_simplices > 10_000_000:
        return {
            'betti': {},
            'persistence_stats': {},
            'error': f'Too many simplices ({n_simplices:,}). Reduce max_alpha_square.',
            'n_simplices': n_simplices
        }

    # Compute persistence
    log(f"         Computing persistence...")
    t0 = time.perf_counter()
    simplex_tree.compute_persistence()
    log(f"         ✓ Persistence computed in {time.perf_counter()-t0:.2f}s")

    result = {'betti': {}, 'persistence_stats': {}, 'method': 'witness_complex'}
    result['n_simplices'] = n_simplices
    result['max_alpha_square'] = max_alpha_square

    for dim in range(max_dim + 1):
        intervals = simplex_tree.persistence_intervals_in_dimension(dim)

        if len(intervals) > 0:
            finite = intervals[np.isfinite(intervals[:, 1])]
            if len(finite) > 0:
                lifetimes = finite[:, 1] - finite[:, 0]
                threshold = np.median(lifetimes) if len(lifetimes) > 0 else 0
                significant = int(np.sum(lifetimes > threshold))

                result['betti'][f'beta_{dim}'] = significant
                result['persistence_stats'][f'dim_{dim}'] = {
                    'total_features': len(intervals),
                    'significant_features': significant,
                    'max_lifetime': float(np.max(lifetimes)),
                    'mean_lifetime': float(np.mean(lifetimes)),
                    'std_lifetime': float(np.std(lifetimes))
                }
            else:
                result['betti'][f'beta_{dim}'] = 0
                result['persistence_stats'][f'dim_{dim}'] = {'total_features': 0}
        else:
            result['betti'][f'beta_{dim}'] = 0
            result['persistence_stats'][f'dim_{dim}'] = {'total_features': 0}

    return result


def _compute_ph_cubical(image_2d: np.ndarray, max_dim: int = 2) -> Dict[str, Any]:
    """
    Compute persistent homology using Cubical Complex (GUDHI).

    Extremely fast: O(n) complexity for n pixels!
    Ideal for density/pattern analysis of sequences.

    Args:
        image_2d: 2D array representing the sequence as an image
        max_dim: Maximum homology dimension (usually 1 for 2D images)

    Returns:
        Dictionary with Betti numbers and persistence stats
    """
    if not HAS_GUDHI:
        return {'betti': {}, 'persistence_stats': {}, 'error': 'GUDHI not installed'}

    # Create cubical complex from the image
    cubical = gudhi.CubicalComplex(top_dimensional_cells=image_2d)

    # Compute persistence
    cubical.compute_persistence()

    result = {'betti': {}, 'persistence_stats': {}, 'method': 'cubical_complex'}
    result['image_shape'] = image_2d.shape

    # For 2D images, max meaningful dimension is 1
    effective_max_dim = min(max_dim, 1)

    # Calculate density range for adaptive threshold
    density_range = float(image_2d.max() - image_2d.min())
    result['density_range'] = density_range

    for dim in range(effective_max_dim + 1):
        intervals = cubical.persistence_intervals_in_dimension(dim)

        if len(intervals) > 0:
            finite = intervals[np.isfinite(intervals[:, 1])]
            if len(finite) > 0:
                lifetimes = finite[:, 1] - finite[:, 0]

                # Adaptive threshold: 5% of density range (filters noise)
                # If density_range is small (uniform), threshold filters more
                persistence_threshold = 0.05 * density_range
                significant = int(np.sum(lifetimes > persistence_threshold))

                # Also count "strong" features (>10% of range)
                strong_threshold = 0.10 * density_range
                strong_features = int(np.sum(lifetimes > strong_threshold))

                result['betti'][f'beta_{dim}'] = significant
                result['persistence_stats'][f'dim_{dim}'] = {
                    'total_features': len(intervals),
                    'significant_features': significant,
                    'strong_features': strong_features,
                    'persistence_threshold': float(persistence_threshold),
                    'max_lifetime': float(np.max(lifetimes)),
                    'mean_lifetime': float(np.mean(lifetimes)),
                    'median_lifetime': float(np.median(lifetimes)),
                    'std_lifetime': float(np.std(lifetimes))
                }
            else:
                result['betti'][f'beta_{dim}'] = 0
                result['persistence_stats'][f'dim_{dim}'] = {'total_features': 0}
        else:
            result['betti'][f'beta_{dim}'] = 0
            result['persistence_stats'][f'dim_{dim}'] = {'total_features': 0}

    return result


def sequence_to_image(sequence: np.ndarray, width: int = None) -> np.ndarray:
    """
    Convert 1D binary sequence to 2D image for cubical persistence (binary mode).

    Args:
        sequence: 1D binary sequence
        width: Width of the image (height = len/width). If None, uses sqrt.

    Returns:
        2D numpy array
    """
    n = len(sequence)
    if width is None:
        width = int(np.sqrt(n))

    height = n // width
    trimmed = sequence[:width * height]

    return trimmed.reshape(height, width).astype(np.float32)


@njit(cache=True)
def _hilbert_d2xy(n: int, d: int) -> tuple:
    """Convert Hilbert index d to (x, y) coordinates for n×n grid. Numba-optimized."""
    x = 0
    y = 0
    s = 1
    while s < n:
        rx = 1 & (d // 2)
        ry = 1 & (d ^ rx)
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        d //= 4
        s *= 2
    return x, y


def sequence_to_hilbert_image(sequence: np.ndarray, block_size: int = 64) -> np.ndarray:
    """
    Convert 1D binary sequence to 2D Hilbert curve image with density gradients.

    This creates a natural gradient by computing density (sum of 1s) in each cell,
    which is ideal for cubical persistence (needs gradients, not binary 0/1).

    Args:
        sequence: 1D binary sequence (0s and 1s)
        block_size: Number of bits per Hilbert cell (controls resolution)

    Returns:
        2D numpy array with density values (0.0 to 1.0)
    """
    n = len(sequence)
    n_cells = n // block_size

    # Find the Hilbert order that fits our cells
    order = max(1, int(np.ceil(np.log2(np.sqrt(n_cells)))))
    grid_size = 2 ** order
    total_cells = grid_size * grid_size

    # Adjust block_size if needed to use all data
    effective_block_size = n // total_cells if total_cells > 0 else block_size

    # Create density grid
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)

    for d in range(min(total_cells, n // effective_block_size)):
        x, y = _hilbert_d2xy(grid_size, d)
        start = d * effective_block_size
        end = start + effective_block_size
        if end <= n:
            # Density = proportion of 1s in this block
            grid[y, x] = np.mean(sequence[start:end])

    return grid


def _extract_betti_from_diagrams(diagrams: List[np.ndarray]) -> Dict[str, Any]:
    """Extract Betti numbers from ripser-style diagrams."""
    betti = {}
    persistence_stats = {}

    for dim, dgm in enumerate(diagrams):
        if len(dgm) > 0:
            finite = dgm[np.isfinite(dgm[:, 1])]
            if len(finite) > 0:
                lifetimes = finite[:, 1] - finite[:, 0]
                threshold = np.median(lifetimes)
                significant = int(np.sum(lifetimes > threshold))

                betti[f'beta_{dim}'] = significant
                persistence_stats[f'dim_{dim}'] = {
                    'total_features': len(dgm),
                    'significant_features': significant,
                    'max_lifetime': float(np.max(lifetimes)),
                    'mean_lifetime': float(np.mean(lifetimes)),
                    'std_lifetime': float(np.std(lifetimes))
                }
            else:
                betti[f'beta_{dim}'] = 0
                persistence_stats[f'dim_{dim}'] = {'total_features': 0}
        else:
            betti[f'beta_{dim}'] = 0
            persistence_stats[f'dim_{dim}'] = {'total_features': 0}

    return {'betti': betti, 'persistence_stats': persistence_stats}


def get_best_backend() -> str:
    """Return the best available TDA backend."""
    if HAS_GUDHI:
        return 'gudhi'
    elif HAS_GIOTTO:
        return 'giotto'
    elif HAS_RIPSER:
        return 'ripser'
    else:
        return 'none'


def run_global_tda(sequence: np.ndarray,
                   n_landmarks: int = 10000,
                   embedding_dim: int = 3,
                   time_delay: int = 1,
                   landmark_method: str = 'uniform',
                   max_dim: int = 2,
                   backend: str = 'auto',
                   return_point_cloud: bool = False,
                   verbose: bool = True) -> Dict[str, Any]:
    """
    Run global TDA analysis with landmark subsampling.

    Args:
        sequence: Binary sequence as numpy array
        n_landmarks: Number of landmark points to use
        embedding_dim: Dimension for Takens embedding
        time_delay: Time delay τ for embedding
        landmark_method: 'uniform' or 'random'
        max_dim: Maximum homology dimension to compute
        backend: TDA backend ('auto', 'ripser', 'giotto', 'gudhi')
        verbose: Print progress

    Returns:
        Dictionary with global TDA results
    """
    log = print if verbose else lambda *a, **k: None

    n = len(sequence)

    # Select backend
    if backend == 'auto':
        backend = get_best_backend()

    log(f"\n   🌐 GLOBAL TDA ANALYSIS (Landmarks)")
    log(f"      Sequence length: {n:,}")
    log(f"      Landmarks: {n_landmarks:,}")
    log(f"      Method: {landmark_method}")
    log(f"      Embedding dim: {embedding_dim}")
    log(f"      Max homology dim: {max_dim}")
    log(f"      Backend: {backend}")

    if sequence.dtype != np.float32:
        sequence = sequence.astype(np.float32)

    # Select landmarks
    log(f"\n      Selecting landmarks...")
    t_start = time.perf_counter()

    if landmark_method == 'random':
        indices = select_landmarks_random(n, n_landmarks, embedding_dim, time_delay)
    else:
        indices = select_landmarks_uniform(n, n_landmarks, embedding_dim, time_delay)

    log(f"      Selected {len(indices):,} landmarks")

    # Create Takens embedding
    log(f"      Creating Takens embedding...")
    point_cloud = _takens_embedding_global(sequence, indices, embedding_dim, time_delay)
    log(f"      Point cloud shape: {point_cloud.shape}")

    # Compute persistent homology
    log(f"      Computing persistent homology...")
    sys.stdout.flush()
    log(f"      Memory usage: {point_cloud.nbytes / 1024 / 1024:.1f} MB for point cloud")
    sys.stdout.flush()

    result = {
        'n_landmarks': len(indices),
        'landmark_method': landmark_method,
        'embedding_dim': embedding_dim,
        'time_delay': time_delay,
        'sequence_length': n,
        'backend': backend,
        'betti': None,
        'persistence_entropy': None,
        'error': None
    }

    # Check backend availability
    if backend == 'none':
        result['error'] = 'No TDA backend available (install ripser, giotto-tda, or gudhi)'
        log(f"      ❌ No TDA backend installed")
        return result

    try:
        if backend == 'gudhi':
            ph_result = _compute_ph_gudhi(point_cloud, max_dim)
        elif backend == 'giotto':
            ph_result = _compute_ph_giotto(point_cloud, max_dim)
        else:  # ripser
            ph_result = _compute_ph_ripser(point_cloud, max_dim)

        result['betti'] = ph_result['betti']
        result['persistence_stats'] = ph_result['persistence_stats']
        betti = ph_result['betti']

    except MemoryError:
        result['error'] = 'Out of memory - try fewer landmarks'
        log(f"      ❌ OUT OF MEMORY: Reduce landmarks (current: {len(indices)})")
        return result
    except Exception as e:
        import traceback
        result['error'] = f"{type(e).__name__}: {str(e)}"
        log(f"      ❌ Error: {type(e).__name__}: {e}")
        log(f"      {traceback.format_exc()}")
        return result

    elapsed = time.perf_counter() - t_start
    result['elapsed_s'] = elapsed

    # Print summary
    log(f"\n      ✓ Completed in {elapsed:.2f}s")
    log(f"      β₀ (components): {betti.get('beta_0', 0)}")
    log(f"      β₁ (cycles): {betti.get('beta_1', 0)}")
    log(f"      β₂ (surfaces): {betti.get('beta_2', 0)}")

    # Interpretation
    if betti.get('beta_2', 0) > 0:
        log(f"\n      🌟 SIGNIFICANT: β₂ > 0 indicates emergent 2D surface!")
    if betti.get('beta_1', 0) > 5:
        log(f"      🔄 Multiple cycles detected - orbital dynamics present")

    # Optionally include point cloud for visualization
    if return_point_cloud:
        result['point_cloud'] = point_cloud

    return result


# ═══════════════════════════════════════════════════════════════════════════
# BLOCK PROCESSING FOR LARGE SEQUENCES
# ═══════════════════════════════════════════════════════════════════════════

def run_global_tda_blocks(sequence: np.ndarray,
                          n_blocks: int = 10,
                          landmarks_per_block: int = 5000,
                          embedding_dim: int = 3,
                          time_delay: int = 1,
                          max_dim: int = 2,
                          backend: str = 'auto',
                          verbose: bool = True) -> Dict[str, Any]:
    """
    Run TDA on blocks of the sequence and aggregate results.

    This allows processing large sequences with more landmarks per block,
    and detects if topological features are consistent across the sequence.

    Args:
        sequence: Binary sequence as numpy array
        n_blocks: Number of blocks to divide the sequence into
        landmarks_per_block: Landmarks to use per block
        embedding_dim: Dimension for Takens embedding
        time_delay: Time delay τ for embedding
        max_dim: Maximum homology dimension to compute
        backend: TDA backend ('auto', 'ripser', 'giotto', 'gudhi')
        verbose: Print progress

    Returns:
        Dictionary with aggregated TDA results across blocks
    """
    log = print if verbose else lambda *a, **k: None

    n = len(sequence)
    block_size = n // n_blocks

    if backend == 'auto':
        backend = get_best_backend()

    log(f"\n   🧱 BLOCK TDA ANALYSIS")
    log(f"      Sequence length: {n:,}")
    log(f"      Blocks: {n_blocks}")
    log(f"      Block size: {block_size:,}")
    log(f"      Landmarks per block: {landmarks_per_block:,}")
    log(f"      Backend: {backend}")

    t_start = time.perf_counter()
    block_results = []

    for i in tqdm(range(n_blocks), desc="      Blocks", disable=not HAS_TQDM or not verbose):
        start_idx = i * block_size
        end_idx = start_idx + block_size if i < n_blocks - 1 else n

        block_seq = sequence[start_idx:end_idx]

        # Run TDA on this block (silent)
        block_result = run_global_tda(
            sequence=block_seq,
            n_landmarks=landmarks_per_block,
            embedding_dim=embedding_dim,
            time_delay=time_delay,
            max_dim=max_dim,
            backend=backend,
            verbose=False
        )

        block_results.append({
            'block_idx': i,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'betti': block_result.get('betti'),
            'elapsed_s': block_result.get('elapsed_s'),
            'error': block_result.get('error')
        })

    elapsed = time.perf_counter() - t_start

    # Aggregate results
    successful = [r for r in block_results if r['betti'] is not None]

    if successful:
        beta_0 = [r['betti'].get('beta_0', 0) for r in successful]
        beta_1 = [r['betti'].get('beta_1', 0) for r in successful]
        beta_2 = [r['betti'].get('beta_2', 0) for r in successful]

        aggregated = {
            'n_blocks': n_blocks,
            'n_successful': len(successful),
            'landmarks_per_block': landmarks_per_block,
            'backend': backend,
            'elapsed_s': elapsed,
            'beta_0': {'mean': float(np.mean(beta_0)), 'std': float(np.std(beta_0)),
                       'min': int(np.min(beta_0)), 'max': int(np.max(beta_0))},
            'beta_1': {'mean': float(np.mean(beta_1)), 'std': float(np.std(beta_1)),
                       'min': int(np.min(beta_1)), 'max': int(np.max(beta_1))},
            'beta_2': {'mean': float(np.mean(beta_2)), 'std': float(np.std(beta_2)),
                       'min': int(np.min(beta_2)), 'max': int(np.max(beta_2))},
            'blocks': block_results
        }

        log(f"\n      ✓ Completed {len(successful)}/{n_blocks} blocks in {elapsed:.2f}s")
        log(f"      β₀: {np.mean(beta_0):.1f} ± {np.std(beta_0):.1f}")
        log(f"      β₁: {np.mean(beta_1):.1f} ± {np.std(beta_1):.1f}")
        log(f"      β₂: {np.mean(beta_2):.1f} ± {np.std(beta_2):.1f}")

        # Consistency check
        if np.std(beta_1) < 0.5 * np.mean(beta_1) if np.mean(beta_1) > 0 else True:
            log(f"      ✓ β₁ is CONSISTENT across blocks")
        else:
            log(f"      ⚠️ β₁ varies significantly across blocks")

    else:
        aggregated = {
            'n_blocks': n_blocks,
            'n_successful': 0,
            'error': 'All blocks failed',
            'blocks': block_results
        }
        log(f"      ❌ All blocks failed")

    return aggregated


# ═══════════════════════════════════════════════════════════════════════════
# SCALABLE TDA ANALYSIS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def run_witness_tda(sequence: np.ndarray,
                    n_landmarks: int = 500,
                    n_witnesses: int = 10000,
                    embedding_dim: int = 3,
                    time_delay: int = 1,
                    max_dim: int = 2,
                    max_alpha_square: float = 0.01,
                    verbose: bool = True) -> Dict[str, Any]:
    """
    Run TDA using Witness Complex - scalable for large sequences.

    Witness Complex uses a small set of landmarks and a large set of witnesses
    to approximate the topology. Much more scalable than Rips.

    Complexity: O(landmarks² × witnesses) vs O(n³) for Rips

    Args:
        sequence: Binary sequence as numpy array
        n_landmarks: Number of landmark points (500-2000 recommended)
        n_witnesses: Number of witness points (10000-100000 recommended)
        embedding_dim: Dimension for Takens embedding
        time_delay: Time delay τ for embedding
        max_dim: Maximum homology dimension
        max_alpha_square: Max alpha² for witness complex (smaller = faster, 0.01 for binary data)
        verbose: Print progress

    Returns:
        Dictionary with TDA results and methodology notes
    """
    log = print if verbose else lambda *a, **k: None
    t_start = time.perf_counter()

    n = len(sequence)
    log(f"\n   👁️ WITNESS COMPLEX TDA")
    log(f"      Sequence length: {n:,}")
    log(f"      Landmarks: {n_landmarks:,}")
    log(f"      Witnesses: {n_witnesses:,}")
    log(f"      Coverage: {(n_landmarks + n_witnesses) / n * 100:.3f}%")

    result = {
        'method': 'witness_complex',
        'n_landmarks': n_landmarks,
        'n_witnesses': n_witnesses,
        'embedding_dim': embedding_dim,
        'sequence_length': n,
        'approximation_notes': (
            f"Witness complex with {n_landmarks} landmarks and {n_witnesses} witnesses. "
            "Detects global structures but may miss fine local details. "
            "Recommended for exploratory analysis."
        ),
        'confidence': 'exploratory',
        'betti': None,
        'error': None
    }

    if not HAS_GUDHI:
        result['error'] = 'GUDHI not installed'
        log(f"      ❌ GUDHI not installed")
        return result

    # Select landmark indices (uniform)
    log(f"      Selecting landmarks...")
    landmark_indices = select_landmarks_uniform(n, n_landmarks, embedding_dim, time_delay)

    # Select witness indices (random, different from landmarks)
    log(f"      Selecting witnesses...")
    max_idx = n - (embedding_dim - 1) * time_delay
    rng = np.random.default_rng(42)
    all_indices = np.arange(max_idx)
    witness_candidates = np.setdiff1d(all_indices, landmark_indices)
    witness_indices = rng.choice(witness_candidates,
                                  size=min(n_witnesses, len(witness_candidates)),
                                  replace=False)

    # Create Takens embeddings
    log(f"      Creating Takens embeddings...")
    seq_array = sequence.astype(np.float32)
    landmarks = _takens_embedding_global(seq_array, landmark_indices.astype(np.int64),
                                          embedding_dim, time_delay)
    witnesses = _takens_embedding_global(seq_array, witness_indices.astype(np.int64),
                                          embedding_dim, time_delay)

    log(f"      Landmarks shape: {landmarks.shape}")
    log(f"      Witnesses shape: {witnesses.shape}")

    # Compute persistent homology
    log(f"      Computing witness persistence (max_α²={max_alpha_square})...")
    try:
        ph_result = _compute_ph_witness(landmarks, witnesses, max_dim,
                                         max_alpha_square=max_alpha_square,
                                         verbose=verbose)

        # Check for errors from the computation
        if ph_result.get('error'):
            result['error'] = ph_result['error']
            result['n_simplices'] = ph_result.get('n_simplices', 0)
            log(f"      ❌ {ph_result['error']}")
            return result

        result['betti'] = ph_result['betti']
        result['persistence_stats'] = ph_result['persistence_stats']
        result['n_simplices'] = ph_result.get('n_simplices', 0)
        result['max_alpha_square'] = max_alpha_square
        betti = ph_result['betti']

    except Exception as e:
        result['error'] = f"{type(e).__name__}: {str(e)}"
        log(f"      ❌ Error: {e}")
        return result

    elapsed = time.perf_counter() - t_start
    result['elapsed_s'] = elapsed

    log(f"\n      ✓ Completed in {elapsed:.2f}s")
    log(f"      Simplices generated: {result['n_simplices']:,}")
    log(f"      β₀ (components): {betti.get('beta_0', 0)}")
    log(f"      β₁ (cycles): {betti.get('beta_1', 0)}")
    log(f"      β₂ (surfaces): {betti.get('beta_2', 0)}")

    if betti.get('beta_1', 0) > 0:
        log(f"\n      🔄 Cycles detected! Possible orbital structures.")
    if betti.get('beta_2', 0) > 0:
        log(f"      🌟 SURFACES detected! Emergent 2D geometry!")

    return result


def run_cubical_tda(sequence: np.ndarray,
                    image_width: int = None,
                    max_dim: int = 1,
                    cubical_mode: str = 'hilbert',
                    hilbert_block_size: int = 64,
                    save_hilbert: bool = False,
                    variant: str = None,
                    iteration: int = None,
                    verbose: bool = True) -> Dict[str, Any]:
    """
    Run TDA using Cubical Complex - extremely fast O(n) analysis.

    Converts the sequence to a 2D image and computes cubical persistence.
    Ideal for detecting density patterns and large-scale structure.

    Args:
        sequence: Binary sequence as numpy array
        image_width: Width of 2D image (only for binary mode)
        max_dim: Maximum homology dimension (1 for 2D images)
        cubical_mode: 'hilbert' (default, with density gradients) or 'binary' (raw 0/1)
        hilbert_block_size: Bits per cell for Hilbert mode (default: 64)
        save_hilbert: Save Hilbert image as PNG file
        variant: Variant name (for filename)
        iteration: Iteration number (for filename)
        verbose: Print progress

    Returns:
        Dictionary with TDA results
    """
    log = print if verbose else lambda *a, **k: None
    t_start = time.perf_counter()

    n = len(sequence)
    mode_label = "HILBERT" if cubical_mode == 'hilbert' else "BINARY"
    log(f"\n   🧊 CUBICAL COMPLEX TDA ({mode_label})")
    log(f"      Sequence length: {n:,}")

    result = {
        'method': 'cubical_complex',
        'cubical_mode': cubical_mode,
        'sequence_length': n,
        'approximation_notes': (
            f"Cubical persistence on {'Hilbert curve with density gradients' if cubical_mode == 'hilbert' else 'binary 2D image'}. "
            "Detects density patterns and connected regions. "
            "Very fast O(n)."
        ),
        'confidence': 'exploratory',
        'betti': None,
        'error': None
    }

    if not HAS_GUDHI:
        result['error'] = 'GUDHI not installed'
        log(f"      ❌ GUDHI not installed")
        return result

    # Convert to 2D image based on mode
    if cubical_mode == 'hilbert':
        log(f"      Converting to Hilbert curve image (block_size={hilbert_block_size})...")
        image = sequence_to_hilbert_image(sequence, block_size=hilbert_block_size)
        result['hilbert_block_size'] = hilbert_block_size
    else:
        log(f"      Converting to binary 2D image...")
        image = sequence_to_image(sequence, image_width)

    result['image_shape'] = image.shape
    log(f"      Image shape: {image.shape}")

    if cubical_mode == 'hilbert':
        log(f"      Density range: [{image.min():.3f}, {image.max():.3f}]")

    # Save Hilbert image if requested
    if save_hilbert and cubical_mode == 'hilbert':
        import os
        try:
            import matplotlib.pyplot as plt
            from scipy import ndimage

            output_dir = "results/level2/tda_analysis"
            os.makedirs(output_dir, exist_ok=True)

            var_str = variant if variant else "X"
            iter_str = iteration if iteration else 0
            density_range = image.max() - image.min()

            # === IMAGE 1: Enhanced contrast (histogram equalization + smoothing) ===
            # This reveals large-scale information currents above noise
            filename_enhanced = f"hilbert_{var_str}_iter{iter_str}_bs{hilbert_block_size}_enhanced.png"
            filepath_enhanced = os.path.join(output_dir, filename_enhanced)

            # Gaussian smoothing to filter high-frequency noise (sigma=2 pixels)
            image_smooth = ndimage.gaussian_filter(image, sigma=2)

            # Histogram equalization: stretch to full [0,1] range
            img_min, img_max = image_smooth.min(), image_smooth.max()
            if img_max > img_min:
                image_eq = (image_smooth - img_min) / (img_max - img_min)
            else:
                image_eq = image_smooth

            # Deviation from mean (shows positive/negative currents)
            mean_density = image_smooth.mean()
            image_deviation = image_smooth - mean_density

            # Create figure with 2 subplots
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))

            # Left: Equalized density (full contrast)
            im1 = axes[0].imshow(image_eq, cmap='viridis', origin='lower')
            axes[0].set_title(f'Equalized Density\n(range stretched to [0,1])')
            plt.colorbar(im1, ax=axes[0], label='Normalized density')

            # Right: Deviation from mean (divergent colormap)
            max_dev = max(abs(image_deviation.min()), abs(image_deviation.max()))
            im2 = axes[1].imshow(image_deviation, cmap='RdBu_r', origin='lower',
                                  vmin=-max_dev, vmax=max_dev)
            axes[1].set_title(f'Deviation from Mean ({mean_density:.3f})\n(blue=less 1s, red=more 1s)')
            plt.colorbar(im2, ax=axes[1], label='Deviation')

            fig.suptitle(f'Information Currents - {var_str}@{iter_str}\n'
                        f'(bs={hilbert_block_size}, raw range={density_range:.4f}, σ=2 smoothing)',
                        fontsize=12)
            plt.tight_layout()
            plt.savefig(filepath_enhanced, dpi=150, bbox_inches='tight')
            plt.close()

            result['hilbert_image_enhanced_path'] = filepath_enhanced
            log(f"      📸 Saved enhanced Hilbert: {filepath_enhanced}")

            # === IMAGE 2: Original (raw density, fixed [0,1] range) ===
            filename = f"hilbert_{var_str}_iter{iter_str}_bs{hilbert_block_size}.png"
            filepath = os.path.join(output_dir, filename)

            plt.figure(figsize=(10, 10))
            plt.imshow(image, cmap='viridis', origin='lower', vmin=0, vmax=1)
            plt.colorbar(label='Density (proportion of 1s)')
            plt.title(f'Hilbert Curve - {var_str}@{iter_str} (bs={hilbert_block_size}, range={density_range:.3f})')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()

            result['hilbert_image_path'] = filepath
            log(f"      📸 Saved raw Hilbert: {filepath}")
        except Exception as e:
            log(f"      ⚠️ Could not save Hilbert image: {e}")

    # Compute cubical persistence
    log(f"      Computing cubical persistence...")
    try:
        ph_result = _compute_ph_cubical(image, max_dim)
        result['betti'] = ph_result['betti']
        result['persistence_stats'] = ph_result['persistence_stats']
        betti = ph_result['betti']

    except Exception as e:
        result['error'] = f"{type(e).__name__}: {str(e)}"
        log(f"      ❌ Error: {e}")
        return result

    elapsed = time.perf_counter() - t_start
    result['elapsed_s'] = elapsed

    log(f"\n      ✓ Completed in {elapsed:.2f}s")
    log(f"      β₀ (connected regions): {betti.get('beta_0', 0):,}")
    log(f"      β₁ (holes/loops): {betti.get('beta_1', 0):,}")

    # Show persistence filter info
    if 'persistence_stats' in result and 'dim_1' in result['persistence_stats']:
        stats = result['persistence_stats']['dim_1']
        if 'strong_features' in stats:
            log(f"      β₁ strong (>10% range): {stats['strong_features']:,}")
            log(f"      Persistence threshold: {stats.get('persistence_threshold', 0):.4f}")

    if betti.get('beta_0', 0) == 1:
        log(f"\n      ✓ Single connected component - sequence is topologically connected")
    if betti.get('beta_1', 0) > 0:
        log(f"      🔄 Holes detected in density pattern!")

    return result
