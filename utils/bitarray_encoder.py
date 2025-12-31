#!/usr/bin/env python3
"""
HSI Structural Format Encoder/Decoder
==========================================

2-bit encoding scheme that preserves structural information (parentheses):
    00 → '0' (state 0)
    01 → '1' (state 1)
    10 → '(' (Absolute start - parenthesis open)
    11 → ')' (Absolute end - parenthesis close)

This encoding preserves the intrinsic recurrent order information that is
CRITICAL for Level 1 analysis (pattern detection and rule inference).

Author: Iban Borràs with collaboration from Augment Agent (Sophia)
Date: November 2025
Version: v33
"""

from bitarray import bitarray
from typing import Optional
import gzip

# Progress protocol for subprocess communication
try:
    from hsi_agents_project.utils.progress_protocol import ProgressReporter
    HAS_PROGRESS_PROTOCOL = True
except ImportError:
    HAS_PROGRESS_PROTOCOL = False


# Encoding mapping: character → 2-bit pattern
ENCODING_MAP = {
    '0': '00',
    '1': '01',
    '(': '10',
    ')': '11'
}

# Decoding mapping: 2-bit pattern → character
DECODING_MAP = {
    '00': '0',
    '01': '1',
    '10': '(',
    '11': ')'
}


def encode_phi_with_structure(phi_str: str, chunk_size: int = 10_000_000, silent: bool = False) -> bitarray:
    """
    Encode Φ string with structural information using 2-bit encoding.

    MEMORY OPTIMIZED: Processes in chunks to avoid MemoryError on large strings.

    Args:
        phi_str: String containing '0', '1', '(', ')' characters
        chunk_size: Number of characters to process at once (default: 10M)
        silent: If True, suppress progress messages (default: False)

    Returns:
        bitarray with 2 bits per character

    Example:
        >>> encode_phi_with_structure("(01)1")
        bitarray('1000011101')  # 10 00 01 11 01
    """
    if not phi_str:
        return bitarray()

    # For small strings, use fast path
    if len(phi_str) <= chunk_size:
        bit_string = ''.join(ENCODING_MAP[c] for c in phi_str)
        return bitarray(bit_string)

    # For large strings, process in chunks to avoid MemoryError
    result = bitarray()
    total_chars = len(phi_str)
    num_chunks = (total_chars + chunk_size - 1) // chunk_size

    # Use progress protocol if available and not silent
    show_progress = not silent and total_chars > 50_000_000 and HAS_PROGRESS_PROTOCOL
    size_str = f"{total_chars/1e9:.2f}B" if total_chars >= 1e9 else f"{total_chars/1e6:.1f}M"

    if show_progress:
        with ProgressReporter(num_chunks, f"Encoding Φ ({size_str})", update_percent=2) as progress:
            for i, start in enumerate(range(0, total_chars, chunk_size)):
                end = min(start + chunk_size, total_chars)
                chunk = phi_str[start:end]
                bit_string = ''.join(ENCODING_MAP[c] for c in chunk)
                result.extend(bitarray(bit_string))
                progress.update(i + 1)
    else:
        for start in range(0, total_chars, chunk_size):
            end = min(start + chunk_size, total_chars)
            chunk = phi_str[start:end]
            bit_string = ''.join(ENCODING_MAP[c] for c in chunk)
            result.extend(bitarray(bit_string))

    return result


def decode_phi_with_structure(bits: bitarray) -> str:
    """
    Decode bitarray back to Φ string with structural information.
    
    Args:
        bits: bitarray with 2 bits per character
        
    Returns:
        String containing '0', '1', '(', ')' characters
        
    Example:
        >>> bits = bitarray('1000011101')
        >>> decode_phi_with_structure(bits)
        '(01)1'
    """
    if not bits or len(bits) == 0:
        return ""
    
    # Ensure even length (2 bits per character)
    if len(bits) % 2 != 0:
        raise ValueError(f"Bitarray length must be even (got {len(bits)})")
    
    # Decode 2 bits at a time
    chars = []
    bit_str = bits.to01()
    for i in range(0, len(bit_str), 2):
        two_bits = bit_str[i:i+2]
        chars.append(DECODING_MAP[two_bits])
    
    return ''.join(chars)


def save_phi_structural_gz(phi_str: str, filepath: str, compresslevel: int = 6, silent: bool = False) -> int:
    """
    Encode and save Φ with structural information to gzip file.

    Args:
        phi_str: String containing '0', '1', '(', ')' characters
        filepath: Output file path (should end with .gz)
        compresslevel: gzip compression level (1-9, default 6 for speed/size balance)
        silent: If True, suppress progress bar (default: False)

    Returns:
        Number of bytes written

    Example:
        >>> save_phi_structural_gz("(01)1", "phi_iter1.struct.gz")
        42  # bytes written
    """
    import os

    bits = encode_phi_with_structure(phi_str, silent=silent)

    # For large files, show compression message
    data_size_mb = len(bits) / 8 / 1_000_000
    if data_size_mb > 50 and not silent:
        print(f"   Compressing {data_size_mb:.1f}MB (level={compresslevel})...", flush=True)

    with gzip.open(filepath, "wb", compresslevel=compresslevel) as f:
        bits.tofile(f)

    file_size = os.path.getsize(filepath)
    if data_size_mb > 50 and not silent:
        print(f"   ✅ Saved: {file_size / 1_000_000:.1f}MB", flush=True)

    return file_size


def save_phi_structural_gz_from_file(
    input_path: str,
    output_path: str,
    compresslevel: int = 9,
    chunk_size: int = 50_000_000
) -> int:
    """
    STREAMING VERSION: Encode and save Φ from file to gzip without loading to RAM.

    Args:
        input_path: Path to input file containing '0', '1', '(', ')' characters
        output_path: Output file path (should end with .gz)
        compresslevel: gzip compression level (1-9, default 9)
        chunk_size: Number of characters to process at once (default: 50M)

    Returns:
        Number of bytes written
    """
    import os
    from pathlib import Path

    input_path = Path(input_path)
    total_size = input_path.stat().st_size
    processed = 0
    num_chunks = (total_size + chunk_size - 1) // chunk_size

    # Use tqdm if available for large files
    show_progress = total_size > 100_000_000 and HAS_TQDM
    pbar = None
    if show_progress:
        pbar = tqdm(
            total=num_chunks,
            desc="   Encoding Φ (streaming)",
            unit="chunk",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed}<{remaining}]"
        )

    with open(input_path, 'r', encoding='utf-8') as in_f:
        with gzip.open(output_path, "wb", compresslevel=compresslevel) as out_f:
            while True:
                chunk = in_f.read(chunk_size)
                if not chunk:
                    break

                # Encode chunk to bits
                bit_string = ''.join(ENCODING_MAP[c] for c in chunk if c in ENCODING_MAP)
                chunk_bits = bitarray(bit_string)

                # Write to gzip
                chunk_bits.tofile(out_f)

                processed += len(chunk)
                if pbar:
                    pbar.update(1)

    if pbar:
        pbar.close()

    return os.path.getsize(output_path)


def load_phi_structural_gz(filepath: str) -> str:
    """
    Load and decode Φ with structural information from gzip file.

    Args:
        filepath: Input file path (should end with .gz)

    Returns:
        String containing '0', '1', '(', ')' characters

    Example:
        >>> load_phi_structural_gz("phi_iter1.struct.gz")
        '(01)1'
    """
    bits = bitarray()

    with gzip.open(filepath, "rb") as f:
        bits.fromfile(f)

    return decode_phi_with_structure(bits)


def stream_phi_prefix_gz(filepath: str, max_chars: int, clean: bool = True) -> str:
    """
    Stream and decode only a prefix of Φ from gzip file.

    Memory-efficient: reads only the bytes needed for max_chars characters.
    Each character uses 2 bits, so we need (max_chars * 2) / 8 = max_chars / 4 bytes.

    Args:
        filepath: Input file path (should end with .gz)
        max_chars: Maximum number of output characters to read
        clean: If True, remove structural characters '(' and ')' from output

    Returns:
        String containing decoded characters (length <= max_chars)

    Example:
        >>> stream_phi_prefix_gz("phi_iter23.struct.gz", 1000000, clean=True)
        '0110101...'  # only 0s and 1s
    """
    # Calculate bytes needed: 2 bits per char, 8 bits per byte
    # Add extra margin for clean mode (we need more input to get enough clean output)
    if clean:
        # Estimate: parentheses are ~10% of structural data, so read 20% more
        bytes_needed = (int(max_chars * 1.25) + 3) // 4
    else:
        bytes_needed = (max_chars + 3) // 4  # Round up

    # Read in chunks for memory efficiency (1MB at a time)
    chunk_size = 1_048_576
    result_chars = []
    bytes_read = 0

    with gzip.open(filepath, "rb") as f:
        while bytes_read < bytes_needed and len(result_chars) < max_chars:
            read_amount = min(chunk_size, bytes_needed - bytes_read)
            raw = f.read(read_amount)
            if not raw:
                break
            bytes_read += len(raw)

            # Decode this chunk
            bits = bitarray()
            bits.frombytes(raw)

            # Process 2 bits at a time
            bit_str = bits.to01()
            for i in range(0, len(bit_str) - 1, 2):
                two_bits = bit_str[i:i+2]
                char = DECODING_MAP.get(two_bits)
                if char:
                    if clean and char in '()':
                        continue  # Skip structural chars
                    result_chars.append(char)
                    if len(result_chars) >= max_chars:
                        break

    return ''.join(result_chars[:max_chars])


def get_format_info(phi_str: str) -> dict:
    """
    Get information about storage format for a given Φ string.

    MEMORY OPTIMIZED: Calculates statistics without creating intermediate strings.

    Args:
        phi_str: String containing '0', '1', '(', ')' characters

    Returns:
        Dictionary with format information
    """
    import sys

    length = len(phi_str)

    # Original text size (OPTIMIZED: assume 1 byte per char for ASCII)
    text_bytes = length

    # Structural 2-bit encoding size (2 bits per character)
    structural_bits = length * 2
    structural_bytes = structural_bits // 8 + (1 if structural_bits % 8 else 0)

    # Clean (only 0/1) bitarray size (OPTIMIZED: count without creating string)
    clean_bits = sum(1 for c in phi_str if c in '01')
    clean_bytes = clean_bits // 8 + (1 if clean_bits % 8 else 0)

    return {
        'length': length,
        'text_bytes': text_bytes,
        'structural_bytes': structural_bytes,
        'clean_bytes': clean_bytes,
        'structural_savings_vs_text': 1.0 - (structural_bytes / text_bytes) if text_bytes > 0 else 0.0,
        'structural_overhead_vs_clean': (structural_bytes / clean_bytes) if clean_bytes > 0 else 0.0
    }

