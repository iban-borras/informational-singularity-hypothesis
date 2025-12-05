#!/usr/bin/env python3
"""
HSI v33 Structural Format Encoder/Decoder
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

    for start in range(0, total_chars, chunk_size):
        end = min(start + chunk_size, total_chars)
        chunk = phi_str[start:end]

        # Encode chunk
        bit_string = ''.join(ENCODING_MAP[c] for c in chunk)
        chunk_bits = bitarray(bit_string)

        # Append to result
        result.extend(chunk_bits)

        # Progress indicator for very large strings (every 5% or at end)
        if not silent and total_chars > 50_000_000:
            progress = 100 * end / total_chars
            prev_progress = 100 * start / total_chars
            # Print only at 5% intervals or at the end
            if (int(progress / 5) > int(prev_progress / 5)) or end == total_chars:
                print(f"   [encode] Progress: {progress:.1f}% ({end:,}/{total_chars:,} chars)", flush=True)

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


def save_phi_structural_gz(phi_str: str, filepath: str, compresslevel: int = 9) -> int:
    """
    Encode and save Φ with structural information to gzip file.

    Args:
        phi_str: String containing '0', '1', '(', ')' characters
        filepath: Output file path (should end with .gz)
        compresslevel: gzip compression level (1-9, default 9)

    Returns:
        Number of bytes written

    Example:
        >>> save_phi_structural_gz("(01)1", "phi_iter1.struct.gz")
        42  # bytes written
    """
    bits = encode_phi_with_structure(phi_str)

    with gzip.open(filepath, "wb", compresslevel=compresslevel) as f:
        bits.tofile(f)

    # Return file size
    import os
    return os.path.getsize(filepath)


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
                if total_size > 100_000_000:  # Only log for large files
                    pct = (processed / total_size) * 100
                    print(f"   [encode] Progress: {pct:.1f}% ({processed:,}/{total_size:,} chars)")

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

