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
from pathlib import Path
from typing import Optional, List, Tuple
import gzip
import json
import os
import shutil
import time

try:
    import numpy as np
    HAS_NUMPY_ENCODER = True
except ImportError:
    np = None
    HAS_NUMPY_ENCODER = False

# Progress bar support
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    tqdm = None

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

if HAS_NUMPY_ENCODER:
    _STRUCTURAL_ASCII_LUT = np.full(256, 255, dtype=np.uint8)
    _STRUCTURAL_ASCII_LUT[ord('0')] = 0
    _STRUCTURAL_ASCII_LUT[ord('1')] = 1
    _STRUCTURAL_ASCII_LUT[ord('(')] = 2
    _STRUCTURAL_ASCII_LUT[ord(')')] = 3


def _pack_structural_ascii(
    raw: bytes,
    carry_codes: bytes = b"",
    *,
    final: bool = False,
) -> Tuple[bytes, bytes]:
    """Pack ASCII structural symbols into the canonical 2-bit byte stream."""
    if HAS_NUMPY_ENCODER:
        values = _STRUCTURAL_ASCII_LUT[np.frombuffer(raw, dtype=np.uint8)]
        if np.any(values == 255):
            values = values[values != 255]
        if carry_codes:
            values = np.concatenate(
                (np.frombuffer(carry_codes, dtype=np.uint8), values)
            )

        complete = (values.size // 4) * 4
        if complete:
            packed = (
                (values[:complete:4] << 6)
                | (values[1:complete:4] << 4)
                | (values[2:complete:4] << 2)
                | values[3:complete:4]
            ).tobytes()
        else:
            packed = b""
        tail = values[complete:].tobytes()
    else:
        code_map = {ord('0'): 0, ord('1'): 1, ord('('): 2, ord(')'): 3}
        values = bytearray(carry_codes)
        values.extend(code_map[value] for value in raw if value in code_map)
        complete = (len(values) // 4) * 4
        packed = bytes(
            (values[i] << 6)
            | (values[i + 1] << 4)
            | (values[i + 2] << 2)
            | values[i + 3]
            for i in range(0, complete, 4)
        )
        tail = bytes(values[complete:])

    if final and tail:
        padded = tail + bytes(4 - len(tail))
        packed += bytes([
            (padded[0] << 6)
            | (padded[1] << 4)
            | (padded[2] << 2)
            | padded[3]
        ])
        tail = b""

    return packed, tail


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_json(path: Path) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


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


def _save_phi_structural_gz_from_file_legacy(
    input_path: str,
    output_path: str,
    compresslevel: int = 9,
    chunk_size: int = 50_000_000,
    input_length: Optional[int] = None,
) -> int:
    """
    STREAMING VERSION: Encode and save Φ from file to gzip without loading to RAM.

    Args:
        input_path: Path to input file containing '0', '1', '(', ')' characters
        output_path: Output file path (should end with .gz)
        compresslevel: gzip compression level (1-9, default 9)
        chunk_size: Number of characters to process at once (default: 50M)
        input_length: Optional logical character count. Useful when input_path
            is itself gzip-compressed and Path.stat() is only physical size.

    Returns:
        Number of bytes written
    """
    import os
    from pathlib import Path

    input_path = Path(input_path)
    total_size = int(input_length) if input_length is not None else input_path.stat().st_size
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

    input_open = gzip.open if str(input_path).endswith('.gz') else open

    with input_open(input_path, 'rt', encoding='utf-8') as in_f:
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


def save_phi_structural_gz_from_file(
    input_path: str,
    output_path: str,
    compresslevel: int = 9,
    chunk_size: int = 50_000_000,
    input_length: Optional[int] = None,
    checkpoint_key: Optional[str] = None,
    checkpoint_chars: Optional[int] = None,
    _test_interrupt_after_parts: Optional[int] = None,
) -> int:
    """Encode a structural stream into atomic, restartable gzip members."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    total_size = int(input_length) if input_length is not None else input_path.stat().st_size
    checkpoint_chars = int(
        checkpoint_chars
        or int(os.environ.get("HSI_SNAPSHOT_CHECKPOINT_MB", "2000")) * 1_000_000
    )
    checkpoint_chars = max(chunk_size, checkpoint_chars)

    parts_dir = output_path.with_name(output_path.name + ".parts")
    checkpoint_path = parts_dir / "checkpoint.json"
    complete_path = output_path.with_name(output_path.name + ".complete.json")
    identity = {
        "path": str(input_path.resolve()),
        "input_chars": total_size,
        "checkpoint_key": checkpoint_key or "v33-structural-stream-v2",
    }

    completed = _load_json(complete_path)
    if (
        completed
        and completed.get("source_identity") == identity
        and output_path.exists()
        and output_path.stat().st_size == completed.get("output_physical_bytes")
    ):
        print(f"   [snapshot] Reusing completed encoder output: {output_path.name}", flush=True)
        return output_path.stat().st_size

    state = _load_json(checkpoint_path)
    if not state or state.get("source_identity") != identity:
        if parts_dir.exists():
            shutil.rmtree(parts_dir)
        parts_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "version": 1,
            "source_identity": identity,
            "input_chars": 0,
            "encoded_bytes": 0,
            "carry_codes": [],
            "parts": [],
        }
        _write_json_atomic(checkpoint_path, state)
    else:
        for part in state.get("parts", []):
            part_path = parts_dir / part["name"]
            if (
                not part_path.exists()
                or part_path.stat().st_size != int(part["physical_bytes"])
            ):
                raise RuntimeError(f"Snapshot checkpoint part is missing or truncated: {part_path}")

    processed = int(state["input_chars"])
    resume_offset = processed
    carry_codes = bytes(state.get("carry_codes", []))
    input_open = gzip.open if str(input_path).endswith('.gz') else open
    started_at = time.perf_counter()
    last_progress = 0.0

    with input_open(input_path, "rb") as in_f:
        to_skip = processed
        while to_skip:
            skipped = in_f.read(min(chunk_size, to_skip))
            if not skipped:
                raise RuntimeError(
                    f"Snapshot checkpoint offset {processed:,} exceeds the input"
                )
            to_skip -= len(skipped)
        if processed:
            print(
                f"   [snapshot] Resuming encoder at {processed:,}/{total_size:,} chars "
                f"from {len(state['parts'])} parts",
                flush=True,
            )

        while processed < total_size:
            part_number = len(state["parts"]) + 1
            part_path = parts_dir / f"part-{part_number:06d}.gz"
            part_tmp = parts_dir / f"part-{part_number:06d}.tmp.gz"
            part_input = 0
            part_encoded = 0

            with open(part_tmp, "wb") as raw_output:
                with gzip.GzipFile(
                    filename="",
                    mode="wb",
                    compresslevel=compresslevel,
                    fileobj=raw_output,
                    mtime=0,
                ) as encoded_output:
                    while part_input < checkpoint_chars and processed < total_size:
                        read_size = min(
                            chunk_size,
                            checkpoint_chars - part_input,
                            total_size - processed,
                        )
                        raw = in_f.read(read_size)
                        if not raw:
                            raise RuntimeError(
                                f"Structural source ended at {processed:,}/{total_size:,} chars"
                            )
                        processed += len(raw)
                        part_input += len(raw)
                        packed, carry_codes = _pack_structural_ascii(
                            raw,
                            carry_codes,
                            final=(processed == total_size),
                        )
                        encoded_output.write(packed)
                        part_encoded += len(packed)

                        now = time.perf_counter()
                        if now - last_progress >= 30.0:
                            elapsed = max(now - started_at, 1e-9)
                            session_chars = processed - resume_offset
                            rate = session_chars / elapsed
                            remaining = total_size - processed
                            eta = remaining / rate if rate else 0.0
                            print(
                                f"   [snapshot] {100 * processed / total_size:6.2f}% | "
                                f"{processed:,}/{total_size:,} chars | "
                                f"{rate / 1e6:.2f} Mchar/s | eta={eta / 3600:.1f}h",
                                flush=True,
                            )
                            last_progress = now

                raw_output.flush()
                os.fsync(raw_output.fileno())
            os.replace(part_tmp, part_path)

            state["input_chars"] = processed
            state["encoded_bytes"] = int(state["encoded_bytes"]) + part_encoded
            state["carry_codes"] = list(carry_codes)
            state["parts"].append(
                {
                    "name": part_path.name,
                    "input_chars": part_input,
                    "encoded_bytes": part_encoded,
                    "physical_bytes": part_path.stat().st_size,
                }
            )
            _write_json_atomic(checkpoint_path, state)
            print(
                f"   [snapshot] Checkpoint part {part_number}: "
                f"{processed:,}/{total_size:,} input chars",
                flush=True,
            )

            if (
                _test_interrupt_after_parts is not None
                and len(state["parts"]) >= _test_interrupt_after_parts
            ):
                raise RuntimeError("simulated snapshot interruption")

    if carry_codes:
        raise RuntimeError("Structural encoder ended with an unflushed bit carry")

    assembling = output_path.with_name(output_path.name + ".assembling")
    with open(assembling, "wb") as destination:
        for part in state["parts"]:
            with open(parts_dir / part["name"], "rb") as source:
                shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(assembling, output_path)

    completed = {
        "version": 1,
        "source_identity": identity,
        "output_physical_bytes": output_path.stat().st_size,
        "encoded_bytes": int(state["encoded_bytes"]),
    }
    _write_json_atomic(complete_path, completed)
    shutil.rmtree(parts_dir)
    print(
        f"   [snapshot] Encoding complete in {len(state['parts'])} restartable parts",
        flush=True,
    )
    return output_path.stat().st_size


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


def decode_phi_structural_gz_to_text_file(
    input_path: str,
    output_path: str,
    expected_chars: Optional[int] = None,
    compresslevel: int = 1,
    chunk_size: int = 4_194_304,
    verbose: bool = True
) -> int:
    """
    Decode a v33 structural snapshot to a text/gzip accumulation file.

    This is the resume-safe inverse of save_phi_structural_gz*_without loading
    the full decoded Phi string into RAM.
    """
    import os
    from pathlib import Path

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    byte_table = [
        ''.join(DECODING_MAP[f"{byte:08b}"[i:i + 2]] for i in range(0, 8, 2))
        for byte in range(256)
    ]

    output_is_gzip = str(output_path).endswith(".gz")
    chars_written = 0
    chunks = 0
    last_report = 0
    t0 = None
    if verbose:
        import time
        t0 = time.perf_counter()
        print(
            f"   [resume] Decoding structural snapshot to temp accumulation: "
            f"{input_path.name} -> {output_path.name}",
            flush=True
        )

    with gzip.open(input_path, "rb") as in_f:
        if output_is_gzip:
            out_ctx = gzip.open(output_path, "wt", encoding="utf-8", compresslevel=compresslevel)
        else:
            out_ctx = open(output_path, "w", encoding="utf-8")
        with out_ctx as out_f:
            while True:
                raw = in_f.read(chunk_size)
                if not raw:
                    break

                decoded = ''.join(byte_table[b] for b in raw)
                if expected_chars is not None:
                    remaining = expected_chars - chars_written
                    if remaining <= 0:
                        break
                    if len(decoded) > remaining:
                        decoded = decoded[:remaining]

                out_f.write(decoded)
                chars_written += len(decoded)
                chunks += 1

                if verbose and expected_chars:
                    pct = int((100 * chars_written) / expected_chars)
                    if pct >= last_report + 5:
                        last_report = pct
                        elapsed = time.perf_counter() - t0 if t0 is not None else 0.0
                        print(
                            f"   [resume] Decoded {pct:3d}% "
                            f"({chars_written:,}/{expected_chars:,} chars, {elapsed:.1f}s)",
                            flush=True
                        )

    if verbose:
        elapsed = time.perf_counter() - t0 if t0 is not None else 0.0
        print(f"   [resume] Decode complete: {chars_written:,} chars in {elapsed:.1f}s", flush=True)

    return chars_written


def stream_phi_prefix_gz(filepath: str, max_chars: int, clean: bool = True,
                         verbose: bool = False) -> str:
    """
    Stream and decode only a prefix of Φ from gzip file.

    Memory-efficient: reads in chunks until we have enough clean characters.
    Each character uses 2 bits, so we need (max_chars * 2) / 8 = max_chars / 4 bytes
    (more if clean=True due to skipped parentheses).

    Args:
        filepath: Input file path (should end with .gz)
        max_chars: Maximum number of output characters to read
        clean: If True, remove structural characters '(' and ')' from output
        verbose: If True, show progress during decompression

    Returns:
        String containing decoded characters (length <= max_chars)

    Example:
        >>> stream_phi_prefix_gz("phi_iter23.struct.gz", 1000000, clean=True)
        '0110101...'  # only 0s and 1s
    """
    # Read in chunks for memory efficiency (4MB at a time for better I/O)
    chunk_size = 4_194_304
    result_chars = []
    chunks_read = 0

    # Progress reporting interval (every 10 chunks = ~40MB)
    progress_interval = 10

    with gzip.open(filepath, "rb") as f:
        # Keep reading until we have enough chars or EOF
        while len(result_chars) < max_chars:
            raw = f.read(chunk_size)
            if not raw:
                break  # EOF

            chunks_read += 1
            if verbose and chunks_read % progress_interval == 0:
                pct = 100 * len(result_chars) / max_chars
                print(f"      ... {pct:.0f}% ({len(result_chars):,}/{max_chars:,} bits)", flush=True)

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


def stream_multi_segment_gz(
    filepath: str,
    segment_size: int,
    num_segments: int = 4,
    total_chars: Optional[int] = None,
    clean: bool = False,
    verbose: bool = False
) -> List[Tuple[int, str]]:
    """
    Stream gzip file and extract multiple segments from different positions.

    Reads the file once (streaming) and captures segments at evenly spaced
    positions for scientific validation of consistency across the sequence.

    Args:
        filepath: Input gzip file path
        segment_size: Size of each segment to capture (e.g., 1_000_000)
        num_segments: Number of segments to capture (default: 4)
        total_chars: Total chars in file (if known, improves progress reporting)
        clean: If True, remove structural characters '(' and ')'
        verbose: Show progress

    Returns:
        List of (position, segment_string) tuples

    Example:
        >>> segments = stream_multi_segment_gz("phi.struct.gz", 1_000_000, 4)
        >>> for pos, seg in segments:
        ...     print(f"Position {pos}: {len(seg)} chars")
    """
    import os

    # Estimate total if not provided (based on compressed size * ~4 decompression ratio)
    if total_chars is None:
        compressed_size = os.path.getsize(filepath)
        # Rough estimate: 2 bits per char, ~4x compression ratio
        total_chars = compressed_size * 4 * 4  # Very rough estimate

    # Calculate capture positions (evenly spaced)
    # We want segments at 0%, 25%, 50%, 75% (for 4 segments)
    capture_positions = []
    for i in range(num_segments):
        pos = int((i / num_segments) * (total_chars - segment_size))
        capture_positions.append(max(0, pos))

    # Stream through file, capturing segments
    segments = []
    current_segment_idx = 0
    current_segment_buffer = []
    capturing = False

    chunk_size = 4_194_304  # 4MB chunks
    char_position = 0

    # Setup progress bar if available
    pbar = None
    if verbose and HAS_TQDM:
        # Create progress bar for streaming to next capture point
        next_target = capture_positions[0] if capture_positions else total_chars
        pbar = tqdm(
            total=total_chars,
            desc="   Streaming",
            unit="chars",
            unit_scale=True,
            ncols=100,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        )
    elif verbose:
        print(f"   Streaming through {total_chars:,} chars to capture {num_segments} segments...")

    with gzip.open(filepath, "rb") as f:
        while current_segment_idx < num_segments:
            raw = f.read(chunk_size)
            if not raw:
                break  # EOF

            # Decode chunk
            bits = bitarray()
            bits.frombytes(raw)
            bit_str = bits.to01()

            chars_in_chunk = 0

            # Process 2 bits at a time
            for i in range(0, len(bit_str) - 1, 2):
                two_bits = bit_str[i:i+2]
                char = DECODING_MAP.get(two_bits)
                if char:
                    if clean and char in '()':
                        char_position += 1
                        chars_in_chunk += 1
                        continue

                    # Check if we should start capturing
                    if not capturing and current_segment_idx < num_segments:
                        if char_position >= capture_positions[current_segment_idx]:
                            capturing = True
                            if pbar:
                                pbar.set_description(f"   📍 Seg {current_segment_idx + 1}/{num_segments}")

                    # Capture if active
                    if capturing:
                        current_segment_buffer.append(char)

                        # Check if segment is complete
                        if len(current_segment_buffer) >= segment_size:
                            segment_str = ''.join(current_segment_buffer)
                            segments.append((capture_positions[current_segment_idx], segment_str))

                            if pbar:
                                pbar.set_description(f"   ✓ Seg {current_segment_idx + 1}/{num_segments}")
                            elif verbose:
                                print(f"   ✓ Segment {current_segment_idx + 1}/{num_segments} captured at position {capture_positions[current_segment_idx]:,}")

                            current_segment_buffer = []
                            capturing = False
                            current_segment_idx += 1

                            if current_segment_idx >= num_segments:
                                break

                    char_position += 1
                    chars_in_chunk += 1

            # Update progress bar
            if pbar:
                pbar.update(chars_in_chunk)

    # Close progress bar
    if pbar:
        pbar.close()
        if verbose:
            print(f"   ✅ All {len(segments)} segments captured successfully")

    # Capture any remaining partial segment
    if current_segment_buffer and current_segment_idx < num_segments:
        segment_str = ''.join(current_segment_buffer)
        segments.append((capture_positions[current_segment_idx], segment_str))
        if verbose:
            print(f"   ⚠️ Partial segment {current_segment_idx + 1} ({len(segment_str):,} chars)")

    return segments
