#!/usr/bin/env python3
"""
HSI Streaming Φ Loader
======================

Memory-efficient streaming loader for large Φ files in v33 structural format.
Reads .struct.gz files in chunks without loading entire file into RAM.

Supports:
- Streaming iteration over characters
- Chunk-based processing for pattern detection
- Separation of observable bits (0/1) from structure (parentheses)

2-bit encoding scheme:
    00 → '0' (state 0)
    01 → '1' (state 1)  
    10 → '(' (Absolute start)
    11 → ')' (Absolute end)

Author: Iban Borràs with collaboration from Augment Agent (Sophia)
Date: December 2025
Version: v33
"""

import gzip
from pathlib import Path
from typing import Iterator, Tuple, Optional, Dict, Any, Generator
import json

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Decoding mapping: 2-bit pattern → character
DECODING_MAP = {
    0b00: '0',
    0b01: '1',
    0b10: '(',
    0b11: ')'
}


class StreamingPhiLoader:
    """
    Memory-efficient streaming loader for Φ structural files.
    
    Reads .struct.gz files in chunks, yielding decoded characters
    without loading the entire file into memory.
    """
    
    def __init__(self, struct_gz_path: str, chunk_bytes: int = 10_000_000):
        """
        Initialize the streaming loader.
        
        Args:
            struct_gz_path: Path to .struct.gz file
            chunk_bytes: Bytes to read per chunk (default 10MB = ~40M characters)
        """
        self.path = Path(struct_gz_path)
        self.chunk_bytes = chunk_bytes
        self._metadata: Optional[Dict[str, Any]] = None
        
        if not self.path.exists():
            raise FileNotFoundError(f"Structural file not found: {self.path}")
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Load and cache metadata from companion .json file."""
        if self._metadata is None:
            json_path = self.path.with_suffix('').with_suffix('.json')
            if json_path.exists():
                with open(json_path, 'r') as f:
                    self._metadata = json.load(f)
            else:
                self._metadata = {}
        return self._metadata
    
    def iter_chars(self) -> Generator[str, None, None]:
        """
        Iterate over all characters in the file.
        
        Yields:
            Single character ('0', '1', '(', or ')')
        """
        with gzip.open(self.path, 'rb') as f:
            while True:
                chunk = f.read(self.chunk_bytes)
                if not chunk:
                    break
                
                # Decode bytes to characters (4 chars per byte)
                for byte in chunk:
                    # Extract 4 pairs of 2 bits from each byte
                    yield DECODING_MAP[(byte >> 6) & 0b11]
                    yield DECODING_MAP[(byte >> 4) & 0b11]
                    yield DECODING_MAP[(byte >> 2) & 0b11]
                    yield DECODING_MAP[byte & 0b11]
    
    def iter_chunks(self, chunk_chars: int = 1_000_000) -> Generator[str, None, None]:
        """
        Iterate over chunks of characters.
        
        Args:
            chunk_chars: Number of characters per chunk
            
        Yields:
            String chunks of the specified size
        """
        buffer = []
        for char in self.iter_chars():
            buffer.append(char)
            if len(buffer) >= chunk_chars:
                yield ''.join(buffer)
                buffer = []
        
        if buffer:
            yield ''.join(buffer)
    
    def iter_observable_chunks(self, chunk_chars: int = 1_000_000) -> Generator[str, None, None]:
        """
        Iterate over chunks containing ONLY observable bits (0/1).
        Filters out parentheses structure.
        
        Args:
            chunk_chars: Number of characters per chunk
            
        Yields:
            String chunks containing only '0' and '1'
        """
        buffer = []
        for char in self.iter_chars():
            if char in '01':
                buffer.append(char)
                if len(buffer) >= chunk_chars:
                    yield ''.join(buffer)
                    buffer = []
        
        if buffer:
            yield ''.join(buffer)
    
    def iter_windows(self, window_size: int, step: int = 1) -> Generator[str, None, None]:
        """
        Iterate with sliding windows over the sequence.
        Memory-efficient for pattern detection.

        Args:
            window_size: Size of each window
            step: Step size between windows

        Yields:
            String windows of the specified size
        """
        buffer = []
        position = 0

        for char in self.iter_chars():
            buffer.append(char)

            # Once buffer is full, start yielding windows
            if len(buffer) >= window_size:
                if (position - window_size + 1) % step == 0:
                    yield ''.join(buffer[-window_size:])
                position += 1

                # Keep buffer from growing too large
                if len(buffer) > window_size * 2:
                    buffer = buffer[-window_size:]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the file (requires full scan).

        Returns:
            Dictionary with counts and percentages
        """
        counts = {'0': 0, '1': 0, '(': 0, ')': 0}
        total = 0

        for char in self.iter_chars():
            counts[char] += 1
            total += 1

        observable = counts['0'] + counts['1']
        structure = counts['('] + counts[')']

        return {
            'total_chars': total,
            'observable_bits': observable,
            'structure_chars': structure,
            'zeros': counts['0'],
            'ones': counts['1'],
            'opens': counts['('],
            'closes': counts[')'],
            'zero_ratio': counts['0'] / observable if observable > 0 else 0,
            'structure_ratio': structure / total if total > 0 else 0
        }


def load_phi_for_agents(
    struct_gz_path: str,
    max_chars: Optional[int] = None,
    observable_only: bool = False,
    show_progress: bool = False,
    total_chars_hint: Optional[int] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Convenience function to load Φ for agent analysis.

    For small files or when full sequence is needed.
    For large files, use StreamingPhiLoader directly.

    Args:
        struct_gz_path: Path to .struct.gz file
        max_chars: Maximum characters to load (None = all)
        observable_only: If True, return only 0/1 bits (no parentheses)
        show_progress: If True, show loading progress bar
        total_chars_hint: Optional hint for total chars (for accurate progress bar)

    Returns:
        Tuple of (phi_string, metadata)
    """
    import time

    loader = StreamingPhiLoader(struct_gz_path)

    # Determine total size for progress (from hint, metadata, or file size estimate)
    total_chars = total_chars_hint
    if show_progress and not total_chars:
        meta = loader.metadata
        if observable_only:
            total_chars = (meta.get('observable_len') or
                           meta.get('phi_length') or
                           meta.get('sequence_length'))
        else:
            total_chars = (meta.get('structural_len') or
                           meta.get('sequence_length'))

        # If max_chars is set and smaller, use that as target
        if max_chars and total_chars:
            total_chars = min(max_chars, total_chars)
        elif max_chars:
            total_chars = max_chars

        # If we still don't know, estimate from compressed file size
        if not total_chars:
            import os
            compressed_size = os.path.getsize(struct_gz_path)
            # Conservative estimate: assume high compression (100x)
            total_chars = compressed_size * 100

    chars = []
    start_time = time.time()
    last_update = 0
    update_interval = 10_000_000  # Update every 10M chars

    # Use tqdm progress bar if available and progress is requested
    pbar = None
    if show_progress and TQDM_AVAILABLE and total_chars:
        pbar = tqdm(
            total=total_chars,
            desc="   Loading Φ",
            unit="char",
            unit_scale=True,
            unit_divisor=1000,
            bar_format="   {desc}: {percentage:3.0f}%|{bar}| {n:.2f}G/{total:.2f}G [{elapsed}<{remaining}, {rate_fmt}]",
            mininterval=0.5
        )
        # Adjust display to show in Gigachars
        pbar.unit = "G"
        pbar.unit_scale = False
        pbar.total = total_chars / 1e9
        pbar.n = 0

    for char in loader.iter_chars():
        if observable_only and char not in '01':
            continue
        chars.append(char)

        current_count = len(chars)

        # Progress update
        if show_progress and current_count - last_update >= update_interval:
            chars_added = current_count - last_update
            last_update = current_count

            if pbar:
                # Update tqdm bar (clamp to avoid >100% warning)
                pbar.n = min(current_count / 1e9, pbar.total)
                pbar.refresh()
            else:
                # Fallback to print-based progress
                elapsed = time.time() - start_time
                rate = current_count / elapsed if elapsed > 0 else 0

                if total_chars and total_chars > 0:
                    pct = min(100.0, current_count / total_chars * 100)
                    remaining = (total_chars - current_count) / rate if rate > 0 else 0
                    print(f"   Loading: {current_count:,} / {total_chars:,} chars "
                          f"({pct:.1f}%) - {rate/1e6:.1f}M/s - ETA: {remaining:.0f}s",
                          flush=True)
                else:
                    print(f"   Loading: {current_count:,} chars - {rate/1e6:.1f}M/s",
                          flush=True)

        if max_chars and current_count >= max_chars:
            break

    # Close progress bar
    if pbar:
        pbar.n = pbar.total  # Set to 100% on completion
        pbar.refresh()
        pbar.close()

    # Final progress message
    if show_progress:
        elapsed = time.time() - start_time
        print(f"   ✓ Loaded {len(chars):,} chars in {elapsed:.1f}s", flush=True)

    return ''.join(chars), loader.metadata


def iter_phi_from_variant(
    variant: str,
    iteration: int,
    base_path: Optional[str] = None
) -> StreamingPhiLoader:
    """
    Get a streaming loader for a specific variant and iteration.

    Args:
        variant: Variant letter (e.g., 'B', 'D', 'E')
        iteration: Iteration number
        base_path: Base path for results (default: auto-detect)

    Returns:
        StreamingPhiLoader instance
    """
    if base_path is None:
        base_path = Path(__file__).parent.parent / "results" / "level0" / "phi_snapshots"
    else:
        base_path = Path(base_path)

    struct_path = base_path / f"var_{variant}" / f"phi_iter{iteration}.struct.gz"
    return StreamingPhiLoader(str(struct_path))


def load_phi_sampled(
    struct_gz_path: str,
    max_memory_gb: float = 8.0,
    segment_size: int = 10_000_000,
    observable_only: bool = False,
    show_progress: bool = True,
    random_seed: int = 42
) -> Tuple[str, Dict[str, Any]]:
    """
    Load Φ with random sampling for very large sequences.

    Samples random segments distributed across the full sequence to maintain
    statistical representativity while fitting in available RAM.

    Args:
        struct_gz_path: Path to .struct.gz file
        max_memory_gb: Maximum memory to use in GB (default: 8GB)
        segment_size: Size of each sampled segment (default: 10M chars)
        observable_only: If True, return only 0/1 bits
        show_progress: Show progress bar
        random_seed: Seed for reproducible sampling

    Returns:
        Tuple of (sampled_phi_string, metadata_with_sampling_info)
    """
    import time
    import random
    import psutil

    loader = StreamingPhiLoader(struct_gz_path)
    meta = loader.metadata

    # Get total size - try multiple metadata fields
    if observable_only:
        total_chars = (meta.get('observable_len') or
                       meta.get('phi_length') or
                       meta.get('sequence_length', 0))
    else:
        total_chars = (meta.get('structural_len') or
                       meta.get('sequence_length', 0))

    if not total_chars:
        # Fallback: estimate from file size (conservative: assume high compression)
        import os
        compressed_size = os.path.getsize(struct_gz_path)
        # Use 100x ratio as conservative estimate (actual can be 400x+ for repetitive data)
        total_chars = compressed_size * 100

    # Calculate how many chars we can load
    # Rough estimate: 1 char = 1 byte in Python string (actually 1-4 but 1 is safe)
    max_chars = int(max_memory_gb * 1e9)

    # Check if we need sampling
    if total_chars <= max_chars:
        # No sampling needed - load normally
        if show_progress:
            print(f"   📊 Sequence fits in memory ({total_chars/1e9:.2f}G), loading fully...", flush=True)
        return load_phi_for_agents(struct_gz_path, observable_only=observable_only,
                                   show_progress=show_progress, total_chars_hint=total_chars)

    # Calculate sampling parameters
    n_segments = max(1, max_chars // segment_size)
    total_sampled = n_segments * segment_size
    sampling_rate = total_sampled / total_chars

    if show_progress:
        print(f"   📊 Sequence too large for memory ({total_chars/1e9:.2f}G > {max_memory_gb}G)", flush=True)
        print(f"   🎲 Random sampling: {n_segments} segments × {segment_size/1e6:.0f}M chars", flush=True)
        print(f"   📈 Sampling rate: {sampling_rate*100:.2f}% (seed={random_seed})", flush=True)

    # Generate random segment start positions
    random.seed(random_seed)
    max_start = total_chars - segment_size
    if max_start <= 0:
        segment_starts = [0]
    else:
        # Distribute segments evenly with some randomness within each region
        region_size = max_start // n_segments
        segment_starts = []
        for i in range(n_segments):
            region_start = i * region_size
            region_end = min(region_start + region_size, max_start)
            start = random.randint(region_start, region_end)
            segment_starts.append(start)
        segment_starts.sort()

    # Load sampled segments
    start_time = time.time()
    sampled_chars = []
    segment_info = []

    pbar = None
    if show_progress and TQDM_AVAILABLE:
        pbar = tqdm(total=n_segments, desc="   Sampling Φ",
                   unit="seg", leave=True, ncols=100)

    for seg_idx, seg_start in enumerate(segment_starts):
        seg_end = seg_start + segment_size
        segment_chars = []

        # Stream through file to reach segment
        char_idx = 0
        for char in loader.iter_chars():
            if observable_only and char not in '01':
                continue

            if char_idx >= seg_start:
                segment_chars.append(char)
                if len(segment_chars) >= segment_size:
                    break
            char_idx += 1

        sampled_chars.extend(segment_chars)
        segment_info.append({
            'start': seg_start,
            'end': seg_start + len(segment_chars),
            'length': len(segment_chars)
        })

        if pbar:
            pbar.update(1)
            pbar.set_postfix({'chars': f'{len(sampled_chars)/1e6:.0f}M'})

        # Reset loader for next segment
        loader = StreamingPhiLoader(struct_gz_path)

    if pbar:
        pbar.close()

    elapsed = time.time() - start_time
    if show_progress:
        print(f"   ✓ Sampled {len(sampled_chars):,} chars from {n_segments} segments in {elapsed:.1f}s", flush=True)

    # Add sampling info to metadata
    sampling_metadata = {
        **meta,
        'sampling': {
            'method': 'random_segments',
            'seed': random_seed,
            'n_segments': n_segments,
            'segment_size': segment_size,
            'total_sampled': len(sampled_chars),
            'total_original': total_chars,
            'sampling_rate': sampling_rate,
            'segments': segment_info
        }
    }

    return ''.join(sampled_chars), sampling_metadata


if __name__ == "__main__":
    import sys

    print("🔄 Streaming Φ Loader — HSI")
    print("=" * 50)

    # Example usage
    if len(sys.argv) > 1:
        path = sys.argv[1]
        loader = StreamingPhiLoader(path)

        print(f"Loading: {path}")
        print(f"Metadata: {loader.metadata.get('format', 'unknown')}")

        # Count first 1000 chars
        sample = []
        for i, char in enumerate(loader.iter_chars()):
            sample.append(char)
            if i >= 999:
                break

        print(f"First 50 chars: {''.join(sample[:50])}")
        print(f"Sample size: {len(sample)}")
    else:
        print("Usage: python streaming_phi_loader.py <path_to_struct.gz>")
        print("\nExample:")
        print("  python streaming_phi_loader.py results/level0/phi_snapshots/var_B/phi_iter23.struct.gz")

