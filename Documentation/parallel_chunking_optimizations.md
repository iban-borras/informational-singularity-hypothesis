# Parallel Chunking Optimizations

## Overview

This document describes the technical improvements applied to parallel chunk processing
in the HSI Agents Project to handle sequences of 19+ billion bits (and up to 425+ billion
for variant B) without memory exhaustion or resource errors.

**Key features:**
- RAM-aware adaptive worker management
- Batch processing with pool recreation
- MMAP streaming for giant files (400GB+)
- φ-ratio sampling to prevent memory explosion

## Problems Identified

### 1. Memory Explosion from String Copies
- Each 50M bit chunk converted to 50MB string via `ba[start:end].to01()`
- With 11 workers processing 50 chunks per batch: ~2.5GB just for strings
- `ProcessPoolExecutor` serializes data with pickle, duplicating memory
- **Total memory per batch**: 12-15GB

### 2. φ-Ratios Accumulation (35GB Array)
- Each 50M bit chunk generates ~25M run-length ratios
- 381 chunks × 25M ratios = 9.5 billion ratios
- Creating `np.array()` of 9.5B floats = **72GB memory**
- Actual error seen: "Unable to allocate 34.9 GiB for array with shape (4678340616,)"

### 3. Windows Resource Exhaustion (WinError 1450)
- "Insufficient system resources exist to complete the requested service"
- Caused by too many worker processes under memory pressure
- Workers fail when kernel handles are exhausted

## Solutions Implemented

### Solution 1: Adaptive Worker Management

```python
def _get_adaptive_workers(base_workers: int, ram_threshold: float = 75.0,
                          min_workers: int = 2, verbose: bool = False) -> int:
    """
    Dynamically reduce workers based on RAM pressure.
    More aggressive reduction: halves workers for every 10% above threshold.
    """
    if not HAS_PSUTIL:
        return base_workers
    
    ram_percent = psutil.virtual_memory().percent
    
    if ram_percent < ram_threshold:
        return base_workers
    
    # Aggressive reduction: halve for every 10% above threshold
    reduction_steps = int((ram_percent - ram_threshold) / 10) + 1
    new_workers = max(min_workers, base_workers // (2 ** reduction_steps))
    
    if verbose and new_workers < base_workers:
        print(f"   ⚠️ RAM at {ram_percent:.1f}% - reducing workers: {base_workers} → {new_workers}")
    
    return new_workers
```

**Key features:**
- Starts reducing at 75% RAM (not 80%)
- Halves workers aggressively (11 → 5 → 2)
- Never goes below 2 workers

### Solution 2: RAM Wait Before Retry

```python
def _wait_for_ram(target_percent: float = 70.0, max_wait: int = 300, 
                  check_interval: int = 5, verbose: bool = False) -> bool:
    """
    Wait until RAM usage drops below target percentage.
    Used before retrying failed chunks.
    """
    # ... waits up to max_wait seconds for RAM to drop
```

**Usage:**
- Called before retrying failed chunks
- Prevents cascade failures from resource exhaustion
- Default wait: up to 2 minutes

### Solution 3: Sampled φ-Ratios (Memory Reduction 99.996%)

```python
def _calculate_phi_ratios_chunk(chunk_str: str, max_ratios: int = 1000) -> dict:
    """
    IMPORTANT: Samples at most max_ratios to prevent memory explosion.
    With 381 chunks × 25M ratios = 9.5B ratios → 72GB memory!
    Instead: 381 chunks × 1000 ratios = 381K ratios → ~3MB memory.
    """
    # ... uniform sampling of run-length ratios
    step = max(1, (len(runs) - 1) // max_ratios)
    
    # Count φ-proximity during processing (not aggregation)
    if abs(ratio - PHI) < 0.3 or abs(ratio - 1/PHI) < 0.2:
        phi_close_count += 1
    
    return {
        'run_ratios': run_ratios,        # Sampled (max 1000)
        'density_ratios': density_ratios,
        'phi_close_count': phi_close_count,  # Pre-counted
        'total_runs': total_runs             # For statistics
    }
```

**Key features:**
- Samples 1000 ratios per chunk (uniform distribution)
- Pre-counts φ-proximity during chunk processing
- Aggregation uses counts, not full arrays

### Solution 4: Batch Processing with Pool Recreation

```python
batch_size = 50  # Process 50 chunks per batch

for batch_start in range(0, n_chunks, batch_size):
    # Adaptive worker adjustment per batch
    current_workers = _get_adaptive_workers(base_workers, ram_threshold=80.0)
    
    # Create FRESH pool for each batch (releases resources)
    with ProcessPoolExecutor(max_workers=current_workers) as executor:
        futures = {executor.submit(_process_chunk, args): args for args in batch_args}
        # ... process results
    
    # Pool is destroyed here, freeing handles
```

**Benefits:**
- Releases kernel handles between batches
- Allows RAM-based worker adjustment
- Prevents resource accumulation

### Solution 5: MMAP Streaming for Giant Files

For sequences >50GB (like variant B@23 with 425 billion bits), loading all data into RAM
is impossible. The solution uses memory-mapped files:

```python
def _decompress_to_temp(gz_path, verbose=False):
    """Decompress .gz to temp file for mmap access."""
    temp_path = f".tmp_{filename}_mmap.bin"
    with gzip.open(gz_path, 'rb') as f_in:
        with open(temp_path, 'wb') as f_out:
            while chunk := f_in.read(64 * 1024 * 1024):  # 64MB chunks
                f_out.write(chunk)
    return temp_path

# Usage in streaming mode:
temp_path = _decompress_to_temp(data_path)
with open(temp_path, 'rb') as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    for chunk_idx in range(n_chunks):
        start_byte = chunk_idx * chunk_size_bytes
        end_byte = min(start_byte + chunk_size_bytes, total_bytes)
        chunk_str = _read_chunk_from_mmap(mm, start_byte, end_byte)
        # Process chunk...

    mm.close()
_cleanup_temp_file(temp_path)
```

**Benefits:**
- Constant RAM usage (~500MB) regardless of file size
- OS manages page caching efficiently
- Full parallel processing capability maintained
- Temp file cleaned up automatically

**Trade-off:**
- Requires disk space for decompressed temp file (e.g., 53GB for B@23)
- Initial decompression takes time (~5-10 min for 500MB compressed)

## Results

| Metric | Before | After |
|--------|--------|-------|
| φ-ratios memory | 35-72 GB | ~3 MB |
| Failed chunks | Many (WinError 1450) | 0 |
| Processing time (E@23) | Failed | 39.5 min |
| φ-tendency accuracy | N/A | Validated (sampled) |

## Configuration Options

```powershell
# Reduce CPU usage if RAM issues persist
python level1_trend_analysis.py --variants E --iterations 23 --max-cpu 25
```

| `--max-cpu` | Workers (22 cores) | Use Case |
|-------------|-------------------|----------|
| 50 (default) | 11 | Normal operation |
| 25 | 5 | Memory-constrained systems |
| 20 | 4 | Very large sequences (>20GB) |

## Dependencies

- `psutil>=5.9.0` - For RAM monitoring (added to requirements.txt)

## Files Modified

- `metrics/emergence_index.py` - All optimizations + MMAP streaming for giant files
- `agents/pattern_detector.py` - RAM-aware parallel processing
- `requirements.txt` - Added psutil

---

# Scripts Status

## 1. `agents/pattern_detector.py` ✅ OPTIMIZED

**Status:** Fully optimized with:
- ✅ Adaptive worker reduction based on RAM (`_get_adaptive_workers`)
- ✅ RAM wait before retry (`_wait_for_ram`)
- ✅ Batch processing with pool recreation (50 chunks per batch)
- ✅ Error retry mechanism (MAX_RETRIES = 2)

**Location:** Lines 393-487 (`_detect_sliding_window_numba_parallel`)

**No further changes required.**

---

## 2. `level0/streaming_collapse_engine.py`

**Current state:** Processes chunks sequentially (not parallel), so no resource exhaustion risk.

**Required changes:** NONE

**Note:** This file uses chunked I/O for memory efficiency but processes sequentially.
No parallel workers = no WinError 1450 risk.

---

## 3. Future Considerations

For any new parallel processing code:

1. **Always use batched pool creation**
   ```python
   for batch in batches:
       with ProcessPoolExecutor(max_workers=current_workers) as executor:
           # Process batch
       # Pool destroyed, resources released
   ```

2. **Always add adaptive workers**
   ```python
   current_workers = _get_adaptive_workers(base, ram_threshold=80.0)
   ```

3. **Always implement retry with RAM wait**
   ```python
   if failed_chunks:
       _wait_for_ram(target_percent=70.0)
       for chunk in failed_chunks:
           # Retry sequentially
   ```

4. **Avoid accumulating large arrays**
   - Sample or aggregate during chunk processing
   - Don't create billion-element arrays in aggregation

