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
    observable_only: bool = False
) -> Tuple[str, Dict[str, Any]]:
    """
    Convenience function to load Φ for agent analysis.

    For small files or when full sequence is needed.
    For large files, use StreamingPhiLoader directly.

    Args:
        struct_gz_path: Path to .struct.gz file
        max_chars: Maximum characters to load (None = all)
        observable_only: If True, return only 0/1 bits (no parentheses)

    Returns:
        Tuple of (phi_string, metadata)
    """
    loader = StreamingPhiLoader(struct_gz_path)

    chars = []
    for char in loader.iter_chars():
        if observable_only and char not in '01':
            continue
        chars.append(char)
        if max_chars and len(chars) >= max_chars:
            break

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

