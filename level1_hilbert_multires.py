#!/usr/bin/env python3
"""
Level 1 — Hilbert Multi-Resolution Comparison (HMRC)

Compares Hilbert curve visualizations at multiple resolutions to find
self-similar φ patterns across scales.

The hypothesis: Feature sizes may scale by φ across resolutions,
revealing hidden order in the apparent turbulence.

OPTIMIZED VERSION:
- Numba JIT compilation for 50-100x speedup on core functions
- Streaming data loading (only reads bits needed for each order)
- Multiprocessing for parallel order analysis
- Publication-quality figure generation

Usage:
    python level1_hilbert_multires.py --variant B --iteration 15
    python level1_hilbert_multires.py --variant B --iteration 15 --orders 10,11,12
    python level1_hilbert_multires.py -v B -i 23 --plot --max-cpu 50
"""

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Numba JIT compilation (with fallback)
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
    prange = range

sys.path.insert(0, str(Path(__file__).parent))

PHI = (1 + math.sqrt(5)) / 2


def safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate Pearson correlation coefficient safely.

    Returns 0.0 if either array has zero variance (all values equal),
    avoiding the numpy RuntimeWarning for division by zero.
    """
    a_flat = a.flatten()
    b_flat = b.flatten()

    # Check for zero variance (would cause divide by zero)
    if np.std(a_flat) == 0 or np.std(b_flat) == 0:
        return 0.0

    return float(np.corrcoef(a_flat, b_flat)[0, 1])


# =============================================================================
# NUMBA-OPTIMIZED HILBERT FUNCTIONS
# =============================================================================

@njit(cache=True)
def hilbert_d2xy_fast(n: int, d: int) -> Tuple[int, int]:
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


@njit(parallel=True, cache=True)
def bits_to_hilbert_grid_fast(bits_array: np.ndarray, order: int) -> np.ndarray:
    """
    Convert bit array to Hilbert curve grid. Numba-optimized with parallelization.

    Args:
        bits_array: 1D numpy array of uint8 (0 or 1)
        order: Hilbert curve order (grid will be 2^order × 2^order)

    Returns:
        2D numpy array with bit values mapped via Hilbert curve
    """
    n = 2 ** order
    grid = np.zeros((n, n), dtype=np.float32)
    num_bits = min(len(bits_array), n * n)

    for d in prange(num_bits):
        # Explicit int64 cast to avoid Numba type inference issues with prange
        d_int = np.int64(d)
        x, y = hilbert_d2xy_fast(n, d_int)
        grid[y, x] = bits_array[d]

    return grid


@njit(parallel=True, cache=True)
def map_bits_chunk_to_grid(grid: np.ndarray, bits_array: np.ndarray, start_d: int, n: int):
    """
    Map a chunk of bits to their Hilbert positions in an existing grid.

    Numba-optimized for processing large orders in chunks.

    Args:
        grid: 2D numpy array to write to (modified in-place)
        bits_array: 1D numpy array of uint8 (0 or 1)
        start_d: Starting Hilbert index for this chunk
        n: Grid dimension (2^order)
    """
    num_bits = len(bits_array)
    for i in prange(num_bits):
        d = np.int64(start_d + i)
        x, y = hilbert_d2xy_fast(n, d)
        grid[y, x] = bits_array[i]


@njit(parallel=True, cache=True)
def compute_hilbert_coords_batch(bits_array: np.ndarray, start_d: int, n: int) -> np.ndarray:
    """
    Compute Hilbert coordinates for a batch of bit positions.

    Returns array of (y, x) coordinates for each bit.
    Memory efficient: only stores coordinates, not full grid.

    Args:
        bits_array: 1D numpy array (used only for length)
        start_d: Starting Hilbert index
        n: Grid dimension (2^order)

    Returns:
        2D array of shape (num_bits, 2) with (y, x) coordinates
    """
    num_bits = len(bits_array)
    coords = np.zeros((num_bits, 2), dtype=np.int32)

    for i in prange(num_bits):
        d = np.int64(start_d + i)
        x, y = hilbert_d2xy_fast(n, d)
        coords[i, 0] = y
        coords[i, 1] = x

    return coords


# Legacy function for compatibility
def hilbert_d2xy(n: int, d: int) -> Tuple[int, int]:
    """Convert Hilbert index d to (x, y) coordinates for n×n grid."""
    return hilbert_d2xy_fast(n, d)


def bits_to_hilbert_grid(bits: str, order: int) -> np.ndarray:
    """
    Convert bit string to Hilbert curve grid. Uses Numba-optimized version.

    Args:
        bits: Binary string
        order: Hilbert curve order (grid will be 2^order × 2^order)

    Returns:
        2D numpy array with density values
    """
    n = 2 ** order
    required_bits = n * n

    # Convert string to numpy array for Numba
    if len(bits) < required_bits:
        # Pad with zeros
        bits_array = np.zeros(required_bits, dtype=np.uint8)
        for i, c in enumerate(bits):
            bits_array[i] = ord(c) - ord('0')
    else:
        # Use frombuffer for efficiency (works for ASCII 0/1)
        bits_array = np.frombuffer(bits[:required_bits].encode('ascii'), dtype=np.uint8) - ord('0')

    # Use Numba-optimized function
    return bits_to_hilbert_grid_fast(bits_array, order)


def calculate_grid_features(grid: np.ndarray, verbose: bool = False) -> Dict:
    """Calculate features from a Hilbert grid."""
    n = grid.shape[0]
    
    if verbose:
        print(f"      Calculating features for {n}×{n} grid...", end=" ", flush=True)
    
    # Basic stats
    density = float(np.mean(grid))
    
    # Block densities at different sub-scales
    block_sizes = [2, 4, 8, 16] if n >= 16 else [2, 4]
    block_densities = {}
    
    for bs in block_sizes:
        if n % bs != 0:
            continue
        n_blocks = n // bs
        blocks = grid.reshape(n_blocks, bs, n_blocks, bs).mean(axis=(1, 3))
        block_densities[bs] = {
            'mean': float(np.mean(blocks)),
            'std': float(np.std(blocks)),
            'max': float(np.max(blocks)),
            'min': float(np.min(blocks))
        }
    
    # Spatial autocorrelation (Moran's I approximation)
    grid_shifted_h = np.roll(grid, 1, axis=1)
    grid_shifted_v = np.roll(grid, 1, axis=0)
    spatial_corr_h = safe_corrcoef(grid, grid_shifted_h)
    spatial_corr_v = safe_corrcoef(grid, grid_shifted_v)
    
    if verbose:
        print("✓")
    
    return {
        'grid_size': n,
        'total_bits': n * n,
        'density': density,
        'block_densities': block_densities,
        'spatial_correlation': {
            'horizontal': spatial_corr_h,
            'vertical': spatial_corr_v,
            'mean': (spatial_corr_h + spatial_corr_v) / 2
        }
    }


def analyze_multi_resolution(
    bits: str,
    orders: Optional[List[int]] = None,
    verbose: bool = True
) -> Dict:
    """
    Analyze Hilbert representations at multiple resolutions.
    
    Args:
        bits: Observable bit string
        orders: List of Hilbert curve orders to analyze
        verbose: Print progress
    
    Returns:
        Analysis results including cross-scale ratios
    """
    if orders is None:
        # Default: orders that fit in available data
        max_order = int(math.log2(math.sqrt(len(bits))))
        orders = list(range(max(4, max_order - 3), max_order + 1))
    
    if verbose:
        print(f"\n🗺️  Hilbert Multi-Resolution Comparison (HMRC)")
        print(f"   Data size: {len(bits):,} bits")
        print(f"   Orders to analyze: {orders}")
        print(f"   Target: φ = {PHI:.6f}")
    
    results_by_order = {}
    total_orders = len(orders)
    
    for idx, order in enumerate(orders):
        required = 4 ** order
        if required > len(bits):
            if verbose:
                print(f"   [{idx+1}/{total_orders}] Order {order}: ✗ (needs {required:,} bits)")
            continue
        
        if verbose:
            print(f"   [{idx+1}/{total_orders}] Order {order} ({4**order:,} bits)...")
        
        grid = bits_to_hilbert_grid(bits, order)
        features = calculate_grid_features(grid, verbose=verbose)
        results_by_order[order] = features
    
    # Calculate cross-scale ratios
    if verbose:
        print(f"\n   Calculating cross-scale ratios...")
    
    cross_scale = calculate_cross_scale_ratios(results_by_order, orders)
    
    return {
        'orders': orders,
        'results_by_order': results_by_order,
        'cross_scale': cross_scale,
        'phi_target': PHI,
        'data_size': len(bits)
    }


def calculate_cross_scale_ratios(
    results_by_order: Dict[int, Dict],
    orders: List[int]
) -> Dict:
    """Calculate ratios between consecutive resolutions."""
    valid_orders = [o for o in orders if o in results_by_order]

    ratios = {
        'density_ratios': [],
        'std_ratios': [],
        'spatial_corr_ratios': [],
        'order_pairs': []
    }

    for i in range(len(valid_orders) - 1):
        o1, o2 = valid_orders[i], valid_orders[i + 1]
        r1, r2 = results_by_order[o1], results_by_order[o2]

        ratios['order_pairs'].append(f"{o2}/{o1}")

        # Density ratio (should be ~1 for balanced data)
        if r1['density'] > 0:
            ratios['density_ratios'].append(r2['density'] / r1['density'])

        # Block std ratio at common block size
        common_bs = 4  # 4×4 blocks
        if common_bs in r1.get('block_densities', {}) and common_bs in r2.get('block_densities', {}):
            std1 = r1['block_densities'][common_bs]['std']
            std2 = r2['block_densities'][common_bs]['std']
            if std1 > 0:
                ratios['std_ratios'].append(std2 / std1)

        # Spatial correlation ratio
        sc1 = r1['spatial_correlation']['mean']
        sc2 = r2['spatial_correlation']['mean']
        if abs(sc1) > 0.001:
            ratios['spatial_corr_ratios'].append(sc2 / sc1)

    # Calculate φ proximity
    all_ratios = ratios['density_ratios'] + ratios['std_ratios'] + ratios['spatial_corr_ratios']
    valid_ratios = [r for r in all_ratios if not math.isnan(r) and r > 0]

    if valid_ratios:
        distances = [abs(r - PHI) for r in valid_ratios]
        ratios['phi_distance_mean'] = float(np.mean(distances))
        ratios['phi_distance_min'] = float(min(distances))
        ratios['closest_ratio'] = valid_ratios[distances.index(min(distances))]
    else:
        ratios['phi_distance_mean'] = float('nan')
        ratios['phi_distance_min'] = float('nan')
        ratios['closest_ratio'] = float('nan')

    return ratios


# =============================================================================
# STREAMING DATA LOADER
# =============================================================================

def stream_bits_for_order(data_dir: str, iteration: int, order: int) -> Optional[str]:
    """
    Stream only the bits needed for a specific Hilbert order.

    Args:
        data_dir: Path to variant data directory
        iteration: Iteration number
        order: Hilbert curve order (needs 4^order bits)

    Returns:
        Bit string of exactly 4^order bits, or None if not enough data
    """
    required_bits = 4 ** order

    # Try to find the structural file
    struct_path = Path(data_dir) / f"phi_iter{iteration}.struct.gz"

    if not struct_path.exists():
        print(f"   [Order {order}] ⚠️ File not found", flush=True)
        return None

    try:
        from utils.bitarray_encoder import stream_phi_prefix_gz
        bits = stream_phi_prefix_gz(str(struct_path), required_bits, clean=True)
        if len(bits) >= required_bits:
            return bits[:required_bits]
        else:
            print(f"   [Order {order}] ⚠️ Only {len(bits):,}/{required_bits:,} bits available", flush=True)
            return None
    except Exception as e:
        print(f"   [Order {order}] ⚠️ Load failed: {e}", flush=True)
        return None


def estimate_memory_for_order(order: int) -> float:
    """Estimate memory required for a Hilbert order in GB."""
    n = 2 ** order
    grid_size = n * n
    # float32 grid + uint8 bits array + working memory
    bytes_needed = grid_size * 4 + grid_size + grid_size * 8  # grid + bits + ops
    return bytes_needed / (1024 ** 3)


def get_available_memory_gb() -> float:
    """Get available system memory in GB."""
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)
    except ImportError:
        # Fallback: assume 4GB available if psutil not installed
        return 4.0


# =============================================================================
# CHUNKED PROCESSING FOR LARGE ORDERS (Order 15+)
# =============================================================================

def bits_to_hilbert_grid_chunked(
    data_dir: str,
    iteration: int,
    order: int,
    chunk_size: int = 10_000_000,
    temp_dir: Optional[str] = None,
    verbose: bool = True
) -> Optional[np.ndarray]:
    """
    Convert bits to Hilbert grid using chunked processing and memory-mapped file.

    This allows processing Order 15+ without running out of RAM.

    Args:
        data_dir: Path to variant data directory
        iteration: Iteration number
        order: Hilbert curve order
        chunk_size: Bits to process per chunk (default 10M)
        temp_dir: Directory for temp mmap file (default: system temp)
        verbose: Show progress bar

    Returns:
        Memory-mapped numpy array with the Hilbert grid, or None if failed
    """
    import tempfile
    import gzip

    n = 2 ** order
    total_bits = n * n

    # Setup progress indicator (using project's standard class)
    try:
        from utils.progress import ProgressIndicator
        use_progress = True
    except ImportError:
        use_progress = False

    if temp_dir is None:
        temp_dir = tempfile.gettempdir()

    mem_for_grid = total_bits / (1024**3)  # uint8 = 1 byte per element

    # For Order 16+ (>2GB), use HDF5 instead of mmap (Windows mmap limit)
    use_hdf5 = order >= 16

    if use_hdf5:
        try:
            import h5py
        except ImportError:
            print(f"   [Order {order}] ❌ h5py not installed. Run: pip install h5py", flush=True)
            return None

        h5_path = Path(temp_dir) / f"hilbert_grid_order{order}_iter{iteration}.h5"
        if verbose:
            print(f"   [Order {order}] HDF5 mode: {n}×{n} grid ({mem_for_grid:.2f} GB)", flush=True)
    else:
        mmap_path = Path(temp_dir) / f"hilbert_grid_order{order}_iter{iteration}.mmap"
        if verbose:
            print(f"   [Order {order}] mmap mode: {n}×{n} grid ({mem_for_grid:.2f} GB)", flush=True)

    try:
        # Create storage backend
        if use_hdf5:
            h5_file = h5py.File(h5_path, 'w')
            grid = h5_file.create_dataset('grid', shape=(n, n), dtype='uint8',
                                          chunks=(min(1024, n), min(1024, n)))
        else:
            # Use uint8 instead of float32: reduces file size by 4x (1GB vs 4GB for order 15)
            # This avoids Windows WinError 87 for large files (>2GB mmap limit on some systems)
            grid = np.memmap(mmap_path, dtype=np.uint8, mode='w+', shape=(n, n))
            h5_file = None

        # Find the data file and stream bits using proper decoder
        struct_path = Path(data_dir) / f"phi_iter{iteration}.struct.gz"
        if not struct_path.exists():
            print(f"   [Order {order}] ❌ File not found: {struct_path}", flush=True)
            return None

        # Use stream_phi_prefix_gz which handles the 2-bit binary format correctly
        from utils.bitarray_encoder import stream_phi_prefix_gz

        # Stream all bits needed in one call (memory-efficient, reads only what's needed)
        if verbose:
            print(f"   [Order {order}] 📥 Decompressing {total_bits:,} bits...", flush=True)

        try:
            import time
            start_time = time.time()
            bits_string = stream_phi_prefix_gz(str(struct_path), total_bits, clean=True, verbose=verbose)
            elapsed = time.time() - start_time
            if verbose:
                print(f"   [Order {order}] ✅ Loaded {len(bits_string):,} bits ({elapsed:.1f}s)", flush=True)
        except Exception as e:
            print(f"   [Order {order}] ❌ Decompression failed: {e}", flush=True)
            return None

        if len(bits_string) < total_bits:
            if verbose:
                print(f"   [Order {order}] ⚠️ Partial: {len(bits_string):,}/{total_bits:,} bits", flush=True)
            # Continue with what we have
            total_bits = len(bits_string)

        # Process bits in chunks to avoid RAM issues
        bits_processed = 0
        progress = None

        if verbose and use_progress:
            progress = ProgressIndicator(f"Mapping {total_bits:,} bits", total=total_bits)
            progress.__enter__()
        elif verbose:
            print(f"         Processing {total_bits:,} bits in chunks of {chunk_size:,}...")

        while bits_processed < total_bits:
            # Get next chunk
            chunk_end = min(bits_processed + chunk_size, total_bits)
            bits_chunk = bits_string[bits_processed:chunk_end]

            if not bits_chunk:
                break

            # Convert to numpy array (bits_chunk is already clean '0'/'1' string)
            bits_array = np.frombuffer(bits_chunk.encode('ascii'), dtype=np.uint8) - ord('0')

            # Pre-compute all Hilbert coordinates using Numba (parallel)
            coords = compute_hilbert_coords_batch(bits_array, bits_processed, n)

            # Write to grid (works for both mmap and HDF5)
            ys = coords[:, 0]
            xs = coords[:, 1]

            if use_hdf5:
                # HDF5: group writes by row for efficiency (avoid element-by-element)
                # Sort by row to minimize h5py I/O operations
                sorted_idx = np.argsort(ys)
                ys_sorted = ys[sorted_idx]
                xs_sorted = xs[sorted_idx]
                vals_sorted = bits_array[sorted_idx]

                # Find unique rows and their boundaries
                unique_rows, row_starts = np.unique(ys_sorted, return_index=True)
                row_ends = np.append(row_starts[1:], len(ys_sorted))

                for row, start, end in zip(unique_rows, row_starts, row_ends):
                    row_data = grid[row, :]  # Read full row
                    if isinstance(row_data, np.ndarray):
                        row_data[xs_sorted[start:end]] = vals_sorted[start:end]
                    else:
                        row_data = np.array(row_data)
                        row_data[xs_sorted[start:end]] = vals_sorted[start:end]
                    grid[row, :] = row_data  # Write back
            else:
                grid[ys, xs] = bits_array  # mmap supports fancy indexing

            bits_processed += len(bits_array)

            if progress:
                progress.update(bits_processed)

        if progress:
            progress.__exit__(None, None, None)

        # Data integrity validation
        if bits_processed < total_bits:
            coverage_pct = 100 * bits_processed / total_bits
            print(f"   [Order {order}] ⚠️ Partial: {bits_processed:,}/{total_bits:,} bits ({coverage_pct:.1f}%)", flush=True)

        if verbose:
            print(f"   [Order {order}] ✅ Mapped {bits_processed:,} bits to {n}×{n} grid", flush=True)

        # Flush/close and return
        if use_hdf5:
            h5_file.flush()
            # Return wrapper with path for cleanup
            grid._h5_file = h5_file
            grid._h5_path = str(h5_path)
        else:
            grid.flush()
            grid._mmap_path = str(mmap_path)

        return grid

    except Exception as e:
        print(f"   [Order {order}] ❌ Error: {e}", flush=True)
        # Cleanup
        if use_hdf5:
            if 'h5_file' in dir() and h5_file:
                h5_file.close()
            if h5_path.exists():
                try:
                    os.remove(h5_path)
                except:
                    pass
        else:
            if 'mmap_path' in dir() and mmap_path.exists():
                try:
                    os.remove(mmap_path)
                except:
                    pass
        return None


def cleanup_grid(grid, verbose: bool = False) -> bool:
    """
    Clean up a grid file (mmap or HDF5) after processing.

    Call this when you're done with a grid returned by bits_to_hilbert_grid_chunked()
    to free disk space.

    Args:
        grid: The memory-mapped array or HDF5 dataset
        verbose: Print status messages

    Returns:
        True if cleanup succeeded, False otherwise
    """
    if grid is None:
        return False

    # Check for HDF5
    h5_path = getattr(grid, '_h5_path', None)
    h5_file = getattr(grid, '_h5_file', None)

    if h5_path is not None:
        try:
            if h5_file:
                h5_file.close()
            if os.path.exists(h5_path):
                os.remove(h5_path)
                if verbose:
                    print(f"   🗑️ Cleaned up HDF5: {h5_path}", flush=True)
                return True
        except Exception as e:
            if verbose:
                print(f"   ⚠️ Failed to cleanup HDF5: {e}", flush=True)
        return False

    # Check for mmap
    mmap_path = getattr(grid, '_mmap_path', None)
    if mmap_path is None:
        return False

    try:
        del grid
        if os.path.exists(mmap_path):
            os.remove(mmap_path)
            if verbose:
                print(f"   🗑️ Cleaned up mmap: {mmap_path}", flush=True)
            return True
    except Exception as e:
        if verbose:
            print(f"   ⚠️ Failed to cleanup mmap: {e}", flush=True)
    return False


# Alias for backwards compatibility
cleanup_mmap_grid = cleanup_grid


def calculate_grid_features_chunked(grid: np.ndarray, order: int, verbose: bool = False) -> Dict:
    """
    Calculate features from a large Hilbert grid using chunked operations.

    Optimized for memory-mapped arrays (avoids loading entire grid into RAM).
    """
    n = grid.shape[0]

    try:
        from utils.progress import ProgressIndicator
        use_progress = True
    except ImportError:
        use_progress = False

    if verbose:
        print(f"   [Order {order}] 📊 Computing features for {n}×{n} grid...", flush=True)

    # Basic stats - can be computed incrementally
    total_sum = 0
    total_count = n * n
    chunk_rows = min(1024, n)  # Process 1024 rows at a time
    progress = None

    if verbose and use_progress:
        progress = ProgressIndicator("Computing density", total=n)
        progress.__enter__()

    for start_row in range(0, n, chunk_rows):
        end_row = min(start_row + chunk_rows, n)
        chunk = grid[start_row:end_row, :]
        total_sum += np.sum(chunk)
        if progress:
            progress.update(end_row)

    if progress:
        progress.__exit__(None, None, None)

    density = float(total_sum / total_count)

    # Block densities - sample for large grids
    block_sizes = [2, 4, 8, 16, 32, 64] if n >= 64 else [2, 4, 8, 16]
    block_densities = {}

    for bs in block_sizes:
        if n % bs != 0:
            continue

        n_blocks = n // bs
        total_blocks = n_blocks * n_blocks
        # For very large grids, sample blocks instead of computing all
        if n_blocks > 1000:
            # Sample 2500 random blocks (provides ~2% margin of error at 95% CI)
            sample_size = min(2500, total_blocks)
            np.random.seed(42)  # Reproducible
            block_means = []
            for _ in range(sample_size):
                bi, bj = np.random.randint(0, n_blocks, 2)
                block = grid[bi*bs:(bi+1)*bs, bj*bs:(bj+1)*bs]
                block_means.append(np.mean(block))
            block_means = np.array(block_means)
            is_sampled = True
        else:
            # Compute all blocks
            block_means = []
            for bi in range(n_blocks):
                for bj in range(n_blocks):
                    block = grid[bi*bs:(bi+1)*bs, bj*bs:(bj+1)*bs]
                    block_means.append(np.mean(block))
            block_means = np.array(block_means)
            is_sampled = False

        # Standard error for sampled statistics
        se = float(np.std(block_means) / np.sqrt(len(block_means)))

        block_densities[bs] = {
            'mean': float(np.mean(block_means)),
            'std': float(np.std(block_means)),
            'std_error': se,
            'max': float(np.max(block_means)),
            'min': float(np.min(block_means)),
            'n_samples': len(block_means),
            'n_total': total_blocks,
            'is_sampled': is_sampled
        }

    # Spatial autocorrelation - stratified sampling for large grids
    is_corr_sampled = False
    if n > 4096:
        # Stratified sampling: sample from 9 regions (3x3 grid) for representativeness
        sample_size = 1024
        regions_corr_h = []
        regions_corr_v = []
        np.random.seed(42)  # Reproducible

        # Sample from 9 regions: corners, edges, and center
        positions = [
            (0, 0), (0, n//2 - sample_size//2), (0, n - sample_size),
            (n//2 - sample_size//2, 0), (n//2 - sample_size//2, n//2 - sample_size//2), (n//2 - sample_size//2, n - sample_size),
            (n - sample_size, 0), (n - sample_size, n//2 - sample_size//2), (n - sample_size, n - sample_size)
        ]

        for row_start, col_start in positions:
            region = grid[row_start:row_start+sample_size, col_start:col_start+sample_size]
            region_h = np.roll(region, 1, axis=1)
            region_v = np.roll(region, 1, axis=0)
            corr_h = safe_corrcoef(region, region_h)
            corr_v = safe_corrcoef(region, region_v)
            if corr_h != 0.0:  # safe_corrcoef returns 0.0 for invalid
                regions_corr_h.append(corr_h)
            if corr_v != 0.0:
                regions_corr_v.append(corr_v)

        spatial_corr_h = float(np.mean(regions_corr_h)) if regions_corr_h else 0.0
        spatial_corr_v = float(np.mean(regions_corr_v)) if regions_corr_v else 0.0
        spatial_corr_std = float(np.std(regions_corr_h + regions_corr_v)) if regions_corr_h else 0.0
        is_corr_sampled = True
    else:
        grid_shifted_h = np.roll(grid, 1, axis=1)
        grid_shifted_v = np.roll(grid, 1, axis=0)
        spatial_corr_h = safe_corrcoef(grid, grid_shifted_h)
        spatial_corr_v = safe_corrcoef(grid, grid_shifted_v)
        spatial_corr_std = 0.0  # Exact value, no sampling error

    if verbose:
        print(f"   ✅ Features computed", flush=True)

    return {
        'grid_size': n,
        'total_bits': n * n,
        'density': density,
        'block_densities': block_densities,
        'spatial_correlation': {
            'horizontal': spatial_corr_h,
            'vertical': spatial_corr_v,
            'mean': (spatial_corr_h + spatial_corr_v) / 2,
            'std': spatial_corr_std,
            'is_sampled': is_corr_sampled,
            'n_regions': 9 if is_corr_sampled else 1
        }
    }


def analyze_single_order(args: Tuple) -> Tuple[int, Optional[Dict]]:
    """
    Analyze a single Hilbert order. Designed for multiprocessing.

    Args:
        args: Tuple of (order, bits_or_path, is_path, data_dir, iteration)

    Returns:
        Tuple of (order, features_dict or None)
    """
    order, bits_or_path, is_streaming, data_dir, iteration = args

    try:
        mem_needed = estimate_memory_for_order(order)
        mem_available = get_available_memory_gb()

        # Use chunked mode for large orders (Order 14+ or when memory is tight)
        use_chunked = order >= 14 or mem_needed > mem_available * 0.5

        if use_chunked:
            # CHUNKED MODE: Uses memory-mapped file, processes bits incrementally
            pass  # Message will be printed by bits_to_hilbert_grid_chunked

            grid = bits_to_hilbert_grid_chunked(data_dir, iteration, order, verbose=True)
            if grid is None:
                return order, None

            features = calculate_grid_features_chunked(grid, order, verbose=True)
            features['grid'] = grid  # Memory-mapped, doesn't consume RAM
            features['mode'] = 'chunked'
            return order, features

        else:
            # STANDARD MODE: Load all bits into memory
            if is_streaming:
                bits = stream_bits_for_order(data_dir, iteration, order)
                if bits is None:
                    return order, None
            else:
                bits = bits_or_path

            required = 4 ** order
            if len(bits) < required:
                print(f"   [Order {order}] ⚠️ Insufficient: {len(bits):,}/{required:,} bits", flush=True)
                return order, None

            grid = bits_to_hilbert_grid(bits, order)
            features = calculate_grid_features(grid, verbose=False)
            features['grid'] = grid
            features['mode'] = 'standard'
            return order, features

    except MemoryError:
        print(f"   [Order {order}] ❌ Out of memory", flush=True)
        return order, None
    except OSError as e:
        if "WinError 87" in str(e) or "parameter is incorrect" in str(e).lower():
            print(f"   [Order {order}] ⚠️ Windows mmap limit — grid too large for this system", flush=True)
        else:
            print(f"   [Order {order}] ❌ OS error: {e}", flush=True)
        return order, None
    except Exception as e:
        print(f"   [Order {order}] ❌ Error: {e}", flush=True)
        return order, None


def analyze_multi_resolution_parallel(
    data_dir: str,
    iteration: int,
    orders: Optional[List[int]] = None,
    max_workers: int = 4,
    max_bits: Optional[int] = None,
    verbose: bool = True
) -> Dict:
    """
    Analyze Hilbert representations at multiple resolutions using parallel processing.

    Args:
        data_dir: Path to variant data directory
        iteration: Iteration number
        orders: List of Hilbert curve orders to analyze
        max_workers: Maximum parallel workers
        max_bits: Maximum bits to load (for non-streaming mode)
        verbose: Print progress

    Returns:
        Analysis results including cross-scale ratios and grids
    """
    # Determine available data size first
    struct_path = Path(data_dir) / f"phi_iter{iteration}.struct.gz"

    if not struct_path.exists():
        raise FileNotFoundError(f"Data file not found: {struct_path}")

    # Get approximate file size to estimate bits
    file_size = struct_path.stat().st_size
    # Structural format: ~2 bits per char, gzip ~10x compression
    estimated_bits = file_size * 10 * 4  # Very rough estimate

    if max_bits:
        available_bits = min(max_bits, estimated_bits)
    else:
        available_bits = estimated_bits

    # Determine orders to analyze
    if orders is None:
        max_order = int(math.log2(math.sqrt(available_bits)))
        orders = list(range(max(6, max_order - 4), max_order + 1))

    if verbose:
        print(f"\n🗺️  Hilbert Multi-Resolution Comparison (HMRC) — PARALLEL")
        print(f"   {'Numba JIT:':16} {'✅ ENABLED' if HAS_NUMBA else '❌ DISABLED (install numba for 50x speedup)'}")
        print(f"   {'Workers:':16} {max_workers}")
        print(f"   {'Orders:':16} {orders}")
        print(f"   {'Target φ:':16} {PHI:.6f}")

        # Memory warning for large orders
        mem_available = get_available_memory_gb()
        for order in orders:
            mem_needed = estimate_memory_for_order(order)
            if mem_needed > mem_available * 0.8:
                print(f"   ⚠️  Warning: Order {order} needs ~{mem_needed:.1f}GB (available: {mem_available:.1f}GB)")

    results_by_order = {}
    grids = {}

    # Use streaming mode: each worker loads only its needed bits
    if verbose:
        print(f"\n   Processing orders in parallel (streaming mode)...")

    # Prepare tasks
    tasks = [(order, None, True, data_dir, iteration) for order in orders]

    # Process in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(analyze_single_order, task): task[0] for task in tasks}

        for future in as_completed(futures):
            order = futures[future]
            try:
                result_order, features = future.result()
                if features:
                    grid = features.pop('grid', None)
                    results_by_order[result_order] = features
                    if grid is not None:
                        grids[result_order] = grid
                    if verbose:
                        print(f"      Order {result_order}: ✅ ({4**result_order:,} bits)")
                else:
                    if verbose:
                        print(f"      Order {result_order}: ❌ (insufficient data)")
            except Exception as e:
                if verbose:
                    print(f"      Order {order}: ❌ Error: {e}")

    # Calculate cross-scale ratios
    if verbose:
        print(f"\n   Calculating cross-scale ratios...")

    cross_scale = calculate_cross_scale_ratios(results_by_order, orders)

    # Calculate total data size from the largest order processed
    data_size = 0
    if results_by_order:
        max_order = max(results_by_order.keys())
        data_size = 4 ** max_order

    return {
        'orders': orders,
        'results_by_order': results_by_order,
        'cross_scale': cross_scale,
        'phi_target': PHI,
        'data_size': data_size,
        'grids': grids  # For visualization
    }


# =============================================================================
# AGGREGATED METRICS FOR VISUALIZATION (Level 0 approach)
# =============================================================================

def compute_local_entropy(density_grid: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Compute local Shannon entropy using sliding window.

    Args:
        density_grid: 2D array with density values (0-1)
        window: Size of sliding window

    Returns:
        2D array with local entropy values (0-1)
    """
    try:
        from scipy.ndimage import uniform_filter
    except ImportError:
        # Fallback: return uniform entropy if scipy unavailable
        return np.full_like(density_grid, 0.5)

    # Smooth the density with uniform filter
    p1_smooth = uniform_filter(density_grid.astype(float), size=window, mode='constant')

    # Shannon entropy: -p*log2(p) - (1-p)*log2(1-p)
    eps = 1e-10
    p1_safe = np.clip(p1_smooth, eps, 1 - eps)
    p0_safe = 1.0 - p1_safe
    entropy = -p1_safe * np.log2(p1_safe) - p0_safe * np.log2(p0_safe)

    return entropy


def compute_aggregated_metrics(
    grids: Dict[int, np.ndarray],
    target_resolution: int = 256
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Compute aggregated metrics for each Hilbert order.

    Applies the same approach as Level 0: rebinning + density/differential/entropy.
    All orders are rebinned to the same target resolution for comparability.

    Args:
        grids: Dictionary {order: grid_array} with raw bit grids
        target_resolution: Target grid size for all orders (default 256x256)

    Returns:
        Dictionary {order: {'density': array, 'differential': array, 'entropy': array}}
    """
    metrics = {}

    for order, grid in sorted(grids.items()):
        grid_size = grid.shape[0]

        # Rebinning factor
        pool = max(1, grid_size // target_resolution)

        if pool > 1:
            # Rebin to target resolution by averaging blocks
            new_size = grid_size // pool
            rebinned = grid[:new_size * pool, :new_size * pool].reshape(
                new_size, pool, new_size, pool
            ).mean(axis=(1, 3))
        else:
            rebinned = grid.astype(float)

        # Density is the rebinned mean (proportion of 1s)
        density = rebinned

        # Differential: deviation from 50% random
        differential = density - 0.5

        # Local entropy
        entropy = compute_local_entropy(density, window=5)

        # φ-alignment: how close each local region is to φ-related ratios
        # We compute this as the local variance scaled by φ
        local_std = np.zeros_like(density)
        try:
            from scipy.ndimage import generic_filter
            local_std = generic_filter(density, np.std, size=5, mode='constant')
        except ImportError:
            pass

        metrics[order] = {
            'density': density,
            'differential': differential,
            'entropy': entropy,
            'local_std': local_std,
            'bits_per_cell': pool * pool,
            'effective_resolution': rebinned.shape[0]
        }

    return metrics


def compute_phi_alignment_map(
    metrics: Dict[int, Dict[str, np.ndarray]]
) -> Dict[int, np.ndarray]:
    """
    Compute φ-alignment maps showing where structure scales by φ.

    Compares density patterns between consecutive orders to find
    regions where the ratio approximates φ.

    Args:
        metrics: Output from compute_aggregated_metrics()

    Returns:
        Dictionary {order: phi_alignment_array} for each order pair
    """
    orders = sorted(metrics.keys())
    phi_maps = {}

    for i in range(len(orders) - 1):
        o1, o2 = orders[i], orders[i + 1]

        # Get density maps (already at same resolution)
        d1 = metrics[o1]['density']
        d2 = metrics[o2]['density']

        # Ensure same shape
        min_size = min(d1.shape[0], d2.shape[0])
        d1 = d1[:min_size, :min_size]
        d2 = d2[:min_size, :min_size]

        # Compute local ratio (avoid division by zero)
        eps = 1e-6
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(np.abs(d1) > eps, d2 / d1, 1.0)

        # φ-distance: how far each local ratio is from φ or 1/φ
        phi_dist = np.minimum(np.abs(ratio - PHI), np.abs(ratio - 1/PHI))

        # Convert to alignment (1 = perfect φ, 0 = far from φ)
        # Use exponential decay: alignment = exp(-distance)
        phi_alignment = np.exp(-phi_dist * 2)

        phi_maps[o2] = phi_alignment

    return phi_maps


def compute_cross_scale_coherence(
    metrics: Dict[int, Dict[str, np.ndarray]]
) -> Tuple[np.ndarray, Dict]:
    """
    Compute coherence map showing where structure persists across scales.

    Args:
        metrics: Output from compute_aggregated_metrics()

    Returns:
        Tuple of (coherence_map, stats_dict)
    """
    orders = sorted(metrics.keys())

    if len(orders) < 2:
        return np.zeros((256, 256)), {'mean': 0, 'std': 0}

    # Get common resolution
    first_density = metrics[orders[0]]['density']
    size = first_density.shape[0]

    # Stack all differential maps
    differentials = []
    for order in orders:
        diff = metrics[order]['differential']
        # Resize if needed
        if diff.shape[0] != size:
            from scipy.ndimage import zoom
            factor = size / diff.shape[0]
            diff = zoom(diff, factor, order=1)
        differentials.append(diff)

    diffs_stack = np.stack(differentials, axis=0)

    # Coherence = 1 - normalized variance across scales
    # High coherence means the pattern is consistent across scales
    variance = np.var(diffs_stack, axis=0)
    max_var = np.max(variance) if np.max(variance) > 0 else 1.0
    coherence = 1.0 - (variance / max_var)

    stats = {
        'mean': float(np.mean(coherence)),
        'std': float(np.std(coherence)),
        'high_coherence_fraction': float(np.mean(coherence > 0.8))
    }

    return coherence, stats


# =============================================================================
# SCIENTIFICALLY MEANINGFUL FIGURE GENERATION
# =============================================================================

def generate_multiscale_comparison_figure(
    metrics: Dict[int, Dict[str, np.ndarray]],
    phi_maps: Dict[int, np.ndarray],
    variant: str,
    iteration: int,
    output_dir: Path,
    fmt: str = 'png',
    dpi: int = 300
) -> Optional[str]:
    """
    Generate Figure 1: Multi-scale comparison with 3×N panels.

    Row 1: Differential (deviation from random) per order
    Row 2: Local entropy per order
    Row 3: φ-alignment per order
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
    except ImportError:
        print("      ⚠️ matplotlib not available")
        return None

    orders = sorted(metrics.keys())
    n_orders = len(orders)

    if n_orders == 0:
        return None

    # Create figure with 3 rows × N columns
    fig, axes = plt.subplots(3, n_orders, figsize=(3 * n_orders, 9))

    # Handle single column case
    if n_orders == 1:
        axes = axes.reshape(3, 1)

    # Row 1: Differential maps
    for i, order in enumerate(orders):
        ax = axes[0, i]
        diff = metrics[order]['differential']
        vabs = max(abs(diff.min()), abs(diff.max()), 0.01)
        norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
        im = ax.imshow(diff, cmap='RdBu_r', origin='lower', norm=norm, aspect='auto')
        ax.set_title(f'Order {order}\n{metrics[order]["bits_per_cell"]} bits/cell', fontsize=9)
        ax.axis('off')
        if i == n_orders - 1:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Diff', fontsize=8)

    axes[0, 0].set_ylabel('Differential\n(red=1s, blue=0s)', fontsize=10)

    # Row 2: Entropy maps (adaptive scale to show subtle variations)
    all_ent = np.concatenate([metrics[o]['entropy'].flatten() for o in orders])
    ent_vmin, ent_vmax = np.min(all_ent), np.max(all_ent)
    # Ensure some range if all values are identical
    if ent_vmax - ent_vmin < 0.01:
        ent_vmin, ent_vmax = 0, 1
    for i, order in enumerate(orders):
        ax = axes[1, i]
        ent = metrics[order]['entropy']
        im = ax.imshow(ent, cmap='viridis_r', origin='lower', vmin=ent_vmin, vmax=ent_vmax, aspect='auto')
        ax.axis('off')
        if i == n_orders - 1:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('H', fontsize=8)

    axes[1, 0].set_ylabel('Local Entropy\n(dark=ordered)', fontsize=10)

    # Row 3: φ-alignment maps
    for i, order in enumerate(orders):
        ax = axes[2, i]
        if order in phi_maps:
            phi_align = phi_maps[order]
            im = ax.imshow(phi_align, cmap='hot', origin='lower', vmin=0, vmax=1, aspect='auto')
        else:
            # First order has no φ-map (no previous order to compare)
            ax.text(0.5, 0.5, 'N/A\n(first order)', ha='center', va='center',
                   transform=ax.transAxes, fontsize=8)
            ax.set_facecolor('#f0f0f0')
        ax.axis('off')
        if i == n_orders - 1 and order in phi_maps:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('φ', fontsize=8)

    axes[2, 0].set_ylabel('φ-Alignment\n(bright=φ scaling)', fontsize=10)

    fig.suptitle(f'Multi-Scale Hilbert Analysis — Variant {variant}, Iter {iteration}\n'
                 f'Aggregated metrics reveal emergent patterns (φ ≈ {PHI:.3f})',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    filename = f'hilbert_multiscale_{variant}_iter{iteration}.{fmt}'
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()

    return filename


def generate_coherence_figure(
    coherence_map: np.ndarray,
    coherence_stats: Dict,
    variant: str,
    iteration: int,
    output_dir: Path,
    fmt: str = 'png',
    dpi: int = 300
) -> Optional[str]:
    """
    Generate Figure 2: Cross-scale coherence analysis.

    Panel 1: Coherence map (where patterns persist across scales)
    Panel 2: Histogram of coherence distribution
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Coherence map
    ax1 = axes[0]
    im = ax1.imshow(coherence_map, cmap='plasma', origin='lower', vmin=0, vmax=1)
    ax1.set_title('Cross-Scale Coherence Map\n(bright = pattern persists across scales)', fontsize=11)
    ax1.axis('off')
    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('Coherence', fontsize=10)

    # Panel 2: Histogram
    ax2 = axes[1]
    values = coherence_map.flatten()
    ax2.hist(values, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    ax2.axvline(coherence_stats['mean'], color='red', linestyle='--', linewidth=2,
                label=f'Mean: {coherence_stats["mean"]:.3f}')
    ax2.axvline(0.8, color='green', linestyle=':', linewidth=2,
                label=f'High coherence (>0.8): {coherence_stats["high_coherence_fraction"]*100:.1f}%')
    ax2.set_xlabel('Coherence', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Coherence Distribution', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f'Cross-Scale Structure Persistence — Variant {variant}, Iter {iteration}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    filename = f'hilbert_coherence_{variant}_iter{iteration}.{fmt}'
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()

    return filename


def generate_statistics_summary_figure(
    metrics: Dict[int, Dict[str, np.ndarray]],
    phi_maps: Dict[int, np.ndarray],
    variant: str,
    iteration: int,
    output_dir: Path,
    fmt: str = 'png',
    dpi: int = 300
) -> Optional[str]:
    """
    Generate Figure 3: Statistical summary across scales.

    Shows how key metrics evolve across Hilbert orders.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    orders = sorted(metrics.keys())

    # Extract statistics per order
    mean_entropy = [float(np.mean(metrics[o]['entropy'])) for o in orders]
    std_density = [float(np.std(metrics[o]['density'])) for o in orders]
    mean_abs_diff = [float(np.mean(np.abs(metrics[o]['differential']))) for o in orders]

    # φ-alignment stats
    phi_orders = sorted(phi_maps.keys())
    phi_means = [float(np.mean(phi_maps[o])) for o in phi_orders]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Entropy vs Order
    ax1 = axes[0, 0]
    ax1.plot(orders, mean_entropy, 'o-', color='purple', linewidth=2, markersize=8)
    ax1.set_xlabel('Hilbert Order', fontsize=11)
    ax1.set_ylabel('Mean Local Entropy', fontsize=11)
    ax1.set_title('Entropy Evolution Across Scales', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    # Plot 2: Density Variation vs Order
    ax2 = axes[0, 1]
    ax2.plot(orders, std_density, 's-', color='teal', linewidth=2, markersize=8)
    ax2.set_xlabel('Hilbert Order', fontsize=11)
    ax2.set_ylabel('Std(Density)', fontsize=11)
    ax2.set_title('Density Variation Across Scales', fontsize=11)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Mean |Differential| vs Order
    ax3 = axes[1, 0]
    ax3.plot(orders, mean_abs_diff, '^-', color='crimson', linewidth=2, markersize=8)
    ax3.set_xlabel('Hilbert Order', fontsize=11)
    ax3.set_ylabel('Mean |Differential|', fontsize=11)
    ax3.set_title('Deviation from Random Across Scales', fontsize=11)
    ax3.grid(True, alpha=0.3)

    # Plot 4: φ-Alignment vs Order
    ax4 = axes[1, 1]
    if phi_means:
        ax4.plot(phi_orders, phi_means, 'D-', color='gold', linewidth=2, markersize=8,
                markeredgecolor='black')
        ax4.axhline(y=np.exp(-0.1 * 2), color='green', linestyle='--', alpha=0.7,
                    label='Strong φ threshold')
    ax4.set_xlabel('Hilbert Order', fontsize=11)
    ax4.set_ylabel('Mean φ-Alignment', fontsize=11)
    ax4.set_title('φ-Scaling Alignment Across Scales', fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1)
    ax4.legend(fontsize=9)

    fig.suptitle(f'Multi-Scale Statistics Summary — Variant {variant}, Iter {iteration}\n'
                 f'How structure properties change with Hilbert order',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    filename = f'hilbert_stats_{variant}_iter{iteration}.{fmt}'
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()

    return filename


# =============================================================================
# ORIGINAL FIGURE GENERATION (enhanced)
# =============================================================================

def generate_hilbert_figures(
    results: Dict,
    variant: str,
    iteration: int,
    output_dir: Path,
    fmt: str = 'png',
    dpi: int = 300
) -> List[str]:
    """
    Generate publication-quality Hilbert visualization figures.

    Args:
        results: Analysis results with grids
        variant: Variant code
        iteration: Iteration number
        output_dir: Output directory
        fmt: Image format (png, pdf, svg)
        dpi: Resolution

    Returns:
        List of generated filenames
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("      ⚠️ matplotlib not available, skipping figures")
        return []

    generated = []
    grids = results.get('grids', {})

    if not grids:
        print("      ⚠️ No grids available for visualization")
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # =================================================================
    # Scientifically meaningful aggregated visualizations
    # =================================================================
    print("      📊 Computing aggregated metrics for scientific visualization...", flush=True)

    try:
        # Compute aggregated metrics (density, differential, entropy per order)
        metrics = compute_aggregated_metrics(grids, target_resolution=256)

        # Compute φ-alignment maps
        phi_maps = compute_phi_alignment_map(metrics)

        # Compute cross-scale coherence
        coherence_map, coherence_stats = compute_cross_scale_coherence(metrics)

        # Figure 1: Multi-scale comparison (3×N panels)
        print("      🎨 Generating multi-scale comparison figure...", flush=True)
        f1 = generate_multiscale_comparison_figure(
            metrics, phi_maps, variant, iteration, output_dir, fmt, dpi
        )
        if f1:
            generated.append(f1)
            print(f"         ✅ {f1}", flush=True)

        # Figure 2: Cross-scale coherence
        print("      🔗 Generating coherence analysis figure...", flush=True)
        f2 = generate_coherence_figure(
            coherence_map, coherence_stats, variant, iteration, output_dir, fmt, dpi
        )
        if f2:
            generated.append(f2)
            print(f"         ✅ {f2}", flush=True)

        # Figure 3: Statistics summary
        print("      📈 Generating statistics summary figure...", flush=True)
        f3 = generate_statistics_summary_figure(
            metrics, phi_maps, variant, iteration, output_dir, fmt, dpi
        )
        if f3:
            generated.append(f3)
            print(f"         ✅ {f3}", flush=True)

        print(f"      ✅ Generated {len([f for f in [f1, f2, f3] if f])} new scientific figures", flush=True)

    except Exception as e:
        print(f"      ⚠️ Error generating scientific figures: {e}", flush=True)
        import traceback
        traceback.print_exc()

    return generated


def print_results(results: Dict) -> None:
    """Print analysis results."""
    print(f"\n{'='*60}")
    print(f"🗺️  HILBERT MULTI-RESOLUTION — Hidden φ Discovery")
    print(f"{'='*60}")
    print(f"   φ target: {results['phi_target']:.6f}")
    if 'data_size' in results and results['data_size'] > 0:
        print(f"   Data: {results['data_size']:,} bits")

    print(f"\n📐 Resolution Analysis:")
    for order, data in results['results_by_order'].items():
        n = data['grid_size']
        print(f"   Order {order} ({n}×{n} = {n*n:,} bits):")
        print(f"      Density: {data['density']:.4f}")
        print(f"      Spatial corr: {data['spatial_correlation']['mean']:.4f}")

    cs = results['cross_scale']
    print(f"\n📈 Cross-Scale Ratios (looking for φ ≈ 1.618):")
    print(f"   Order pairs: {cs['order_pairs']}")

    if cs['density_ratios']:
        print(f"   Density ratios: {[f'{r:.3f}' for r in cs['density_ratios']]}")
    if cs['std_ratios']:
        print(f"   Std ratios: {[f'{r:.3f}' for r in cs['std_ratios']]}")
    if cs['spatial_corr_ratios']:
        print(f"   Spatial corr ratios: {[f'{r:.3f}' for r in cs['spatial_corr_ratios']]}")

    print(f"\n🎯 φ-Alignment Assessment:")
    dist = cs['phi_distance_mean']
    if not math.isnan(dist):
        print(f"   Mean φ-distance: {dist:.4f}")
        print(f"   Min φ-distance: {cs['phi_distance_min']:.4f}")
        print(f"   Closest ratio to φ: {cs['closest_ratio']:.4f}")

        if dist < 0.1:
            print(f"   ✅ STRONG φ signal in multi-resolution structure!")
        elif dist < 0.3:
            print(f"   ⚠️  Moderate φ signal")
        else:
            print(f"   ❌ Weak φ signal")
    else:
        print(f"   ❌ Insufficient data for φ analysis")


def main():
    parser = argparse.ArgumentParser(
        description="Hilbert Multi-Resolution Comparison — Find hidden φ across scales",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python level1_hilbert_multires.py -v B -i 23 --plot
    python level1_hilbert_multires.py -v B -i 23 --orders 10,11,12,13 --plot
    python level1_hilbert_multires.py -v B -i 23 --parallel --max-cpu 50 --plot
        """
    )
    parser.add_argument("--variant", "-v", default="B", help="Variant (B, D, E, F, G, H, I)")
    parser.add_argument("--iteration", "-i", type=int, default=15, help="Iteration number")
    parser.add_argument("--max-bits", "-m", type=int, default=None,
                        help="Max bits to analyze (default: auto based on orders)")
    parser.add_argument("--orders", "-r", type=str, default=None,
                        help="Comma-separated Hilbert orders (e.g., '12,13,14,15'). "
                             "Order N creates a 2^N × 2^N grid requiring 4^N bits. "
                             "Recommended: 12-16 for large iterations (>100M bits), "
                             "10-14 for medium, 8-12 for small. Use consecutive orders "
                             "to detect φ-scaling between resolutions.")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output JSON file")
    parser.add_argument("--plot", "-p", action="store_true", help="Generate visualization figures")
    parser.add_argument("--parallel", action="store_true",
                        help="Use parallel processing (recommended for large data)")
    parser.add_argument("--max-cpu", type=int, default=50,
                        help="Maximum CPU usage %% for parallel mode (default: 50)")
    parser.add_argument("--format", "-f", choices=['png', 'pdf', 'svg'], default='png',
                        help="Image format for plots (default: png)")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for plots (default: 300)")
    parser.add_argument("--legacy", action="store_true",
                        help="Use legacy non-parallel mode (loads all data to RAM)")

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"🗺️  HILBERT MULTI-RESOLUTION COMPARISON")
    print(f"{'='*60}")
    print(f"   Variant: {args.variant}, Iteration: {args.iteration}")

    # Determine data directory
    data_dir = f"results/level0/phi_snapshots/var_{args.variant}"
    if not os.path.exists(data_dir):
        data_dir = f"results/level0/var_{args.variant}"
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found for variant {args.variant}")
        sys.exit(1)

    # Parse orders
    orders = None
    if args.orders:
        orders = [int(o) for o in args.orders.split(",")]

    # Determine output directory for figures
    from utils.file_saver import get_output_path, relative_path
    figures_dir = get_output_path(1, "figures", "hilbert")

    # Choose processing mode
    if args.legacy:
        # Legacy mode: load all data first
        print(f"\n🔬 Loading data (legacy mode)...")
        try:
            from level1.data_loader import load_phi_for_level1
            _, phi_observable, metadata = load_phi_for_level1(
                data_dir, args.iteration,
                return_structural=False,
                return_observable=True,
                return_metadata=True
            )
        except FileNotFoundError as e:
            print(f"❌ {e}")
            sys.exit(1)

        print(f"   Loaded {len(phi_observable):,} bits")

        if args.max_bits and args.max_bits < len(phi_observable):
            phi_observable = phi_observable[:args.max_bits]
            print(f"   Truncated to {args.max_bits:,} bits")

        results = analyze_multi_resolution(phi_observable, orders=orders)
    else:
        # Optimized parallel mode with streaming
        max_workers = max(1, int(os.cpu_count() * args.max_cpu / 100))
        print(f"\n🔬 Analyzing with parallel streaming...")

        try:
            results = analyze_multi_resolution_parallel(
                data_dir, args.iteration,
                orders=orders,
                max_workers=max_workers,
                max_bits=args.max_bits,
                verbose=True
            )
        except FileNotFoundError as e:
            print(f"❌ {e}")
            sys.exit(1)

    results['variant'] = args.variant
    results['iteration'] = args.iteration

    print_results(results)

    # Generate figures if requested
    if args.plot:
        print(f"\n📊 Generating figures...")
        generated = generate_hilbert_figures(
            results, args.variant, args.iteration,
            figures_dir, fmt=args.format, dpi=args.dpi
        )
        if generated:
            print(f"   📁 Output: {relative_path(figures_dir)}")
            for fname in generated:
                print(f"      • {fname}")
        else:
            print(f"   ⚠️ No figures generated")

    # Save JSON results (always, with auto-generated filename if not specified)
    if args.output:
        output_path = Path(args.output)
    else:
        filename = f"hilbert_analysis_{args.variant}_iter{args.iteration}.json"
        output_path = get_output_path(1, "metrics", filename)

    def convert_for_json(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(k): convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(x) for x in obj]
        return obj

    # Remove grids from JSON (too large)
    results_for_json = {k: v for k, v in results.items() if k != 'grids'}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(convert_for_json(results_for_json), f, indent=2)
    print(f"\n💾 Results saved to: {relative_path(output_path)}")

    print(f"\n{'='*60}")
    print(f"✅ Analysis complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

