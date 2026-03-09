"""
Accumulation Manager — Disk-based storage for large accumulations

This module provides memory-efficient storage for the accumulation variable
in HSI Level 0 generator, using disk-based append-only files.

Supports optional gzip compression for space-constrained environments.

Author: Iban Borràs with Augment Agent (Sophia)
Date: November 2025
"""

import os
import gzip
from pathlib import Path
from typing import Optional, Union, IO


class AccumulationManager:
    """
    Manages accumulation as disk-based append-only storage.
    
    This class provides a memory-efficient alternative to storing the entire
    accumulation string in RAM. Instead, it appends to a file on disk and
    tracks metadata (length, clean bits count) in memory.
    
    Usage:
        manager = AccumulationManager("results", "B")
        manager.append("(01)1")
        manager.append("0")
        length = manager.get_length()  # Fast, no disk read
        data = manager.read_all()  # Slow, reads entire file
        manager.cleanup()  # Remove temporary file
    """
    
    def __init__(self, output_dir: str, variant: str, resume_from: Optional[str] = None,
                 compress: bool = False, compress_level: int = 1):
        """
        Initialize accumulation manager.

        Args:
            output_dir: Directory for temporary files
            variant: Variant code (e.g., "B")
            resume_from: Path to existing accumulation file (for checkpoint recovery)
            compress: Whether to use gzip compression for temp files
            compress_level: Gzip compression level (1=fast, 9=max compression)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.variant = variant
        self.compress = compress
        self.compress_level = compress_level

        # File extension depends on compression mode
        ext = ".tmp.gz" if compress else ".tmp"
        self.file_path = self.output_dir / f"accumulation_{variant}{ext}"

        # Metadata (tracked in memory for fast access)
        self.current_length = 0
        self.clean_bits_count = 0

        # Buffer for batched writes (optimization)
        self.buffer = []
        self.buffer_size = 10_000  # Flush every 10K appends

        # Clean up old files from previous runs (both compressed and uncompressed)
        # This prevents stale files from lingering when switching compression modes
        old_tmp = self.output_dir / f"accumulation_{variant}.tmp"
        old_gz = self.output_dir / f"accumulation_{variant}.tmp.gz"

        # Resume from existing file if provided
        if resume_from and Path(resume_from).exists():
            self._resume_from_file(resume_from)
        else:
            # Clean up any old files (both extensions)
            if old_tmp.exists():
                old_tmp.unlink()
            if old_gz.exists():
                old_gz.unlink()
    
    def _open_file(self, mode: str) -> IO:
        """Open file with appropriate method based on compression setting."""
        if self.compress:
            if 'r' in mode:
                return gzip.open(self.file_path, 'rt', encoding='utf-8')
            else:
                return gzip.open(self.file_path, 'at', encoding='utf-8', compresslevel=self.compress_level)
        else:
            return open(self.file_path, mode, encoding='utf-8')

    def _resume_from_file(self, file_path: str):
        """Resume from existing accumulation file."""
        import shutil
        shutil.copy(file_path, self.file_path)

        # Recalculate metadata
        print(f"   [AccumulationManager] Resuming from {file_path}...")
        with self._open_file('r') as f:
            content = f.read()
            self.current_length = len(content)
            self.clean_bits_count = sum(1 for c in content if c in '01')
        compress_note = " (compressed)" if self.compress else ""
        print(f"   [AccumulationManager] Loaded {self.current_length:,} chars, {self.clean_bits_count:,} clean bits{compress_note}")
    
    def append(self, state: str):
        """
        Append state to accumulation file.

        Args:
            state: String to append (e.g., "(01)1")
        """
        # Add to buffer
        self.buffer.append(state)

        # Update metadata
        self.current_length += len(state)
        self.clean_bits_count += sum(1 for c in state if c in '01')

        # Flush if buffer is full
        if len(self.buffer) >= self.buffer_size:
            self._flush()

    def append_from_file(self, source_path: Path, chunk_size: int = 50_000_000) -> int:
        """
        Append content from another file without loading to RAM.

        Args:
            source_path: Path to source file
            chunk_size: Bytes to read at a time (default 50 MB)

        Returns:
            Number of bytes appended
        """
        # Flush any pending buffer first
        self._flush()

        source_path = Path(source_path)
        if not source_path.exists():
            return 0

        total_appended = 0
        clean_bits_added = 0

        # Handle compressed source files
        source_is_compressed = str(source_path).endswith('.gz')
        if source_is_compressed:
            src_open = lambda: gzip.open(source_path, 'rt', encoding='utf-8')
        else:
            src_open = lambda: open(source_path, 'r', encoding='utf-8')

        with src_open() as src:
            with self._open_file('a') as dst:
                while True:
                    chunk = src.read(chunk_size)
                    if not chunk:
                        break
                    dst.write(chunk)
                    total_appended += len(chunk)
                    clean_bits_added += sum(1 for c in chunk if c in '01')

        # Update metadata
        self.current_length += total_appended
        self.clean_bits_count += clean_bits_added

        return total_appended

    def _flush(self):
        """Flush buffer to disk."""
        if self.buffer:
            with self._open_file('a') as f:
                f.write(''.join(self.buffer))
            self.buffer = []
    
    def get_length(self) -> int:
        """Get total length without loading file."""
        return self.current_length
    
    def get_clean_bits_count(self) -> int:
        """Get clean bits count without loading file."""
        return self.clean_bits_count

    def get_file_size(self) -> int:
        """Get file size in bytes without loading file."""
        self._flush()
        if self.file_path.exists():
            return self.file_path.stat().st_size
        return 0

    def read_all(self) -> str:
        """
        Read entire accumulation (use sparingly!).

        WARNING: This loads the entire file into memory.
        Only use when absolutely necessary.
        """
        # Flush buffer first
        self._flush()

        if not self.file_path.exists():
            return ""

        with self._open_file('r') as f:
            return f.read()

    def read_chunk(self, start: int, end: int) -> str:
        """
        Read specific chunk of accumulation.

        Note: For compressed files, seeking is inefficient. Consider reading all.

        Args:
            start: Start position (bytes/chars)
            end: End position (bytes/chars)

        Returns:
            Chunk of accumulation
        """
        # Flush buffer first
        self._flush()

        if not self.file_path.exists():
            return ""

        with self._open_file('r') as f:
            if self.compress:
                # For gzip, we must read from start (no efficient seek)
                content = f.read(end)
                return content[start:end]
            else:
                f.seek(start)
                return f.read(end - start)
    
    def cleanup(self):
        """Remove temporary accumulation file."""
        # Flush buffer first
        self._flush()
        
        if self.file_path.exists():
            self.file_path.unlink()
            print(f"   [AccumulationManager] Cleaned up temporary file: {self.file_path}")
    
    def build_decay_frame(self, absolute_token: str, output_path: Optional[Path] = None,
                          compress_output: Optional[bool] = None,
                          prefix_mode: bool = False) -> Path:
        """
        Build decay frame on disk without loading to RAM.

        Default:      ({accumulation}){absolute_token}   — variants B, D, E, F, G, H, I
        prefix_mode:  {absolute_token}({accumulation})   — variant N (Ontological Collapse)

        Args:
            absolute_token: Token to append/prepend (e.g., "1", "01", "10")
            output_path: Optional custom output path
            compress_output: Whether to compress output (default: same as self.compress)
            prefix_mode: If True, token goes BEFORE parentheses (Variant N)

        Returns:
            Path to the decay frame file
        """
        # Flush any pending writes
        self._flush()

        if compress_output is None:
            compress_output = self.compress

        if output_path is None:
            ext = ".tmp.gz" if compress_output else ".tmp"
            output_path = self.output_dir / f"decay_frame_{self.variant}{ext}"

        chunk_size = 10_000_000  # 10 MB chunks

        # Choose output file opener
        if compress_output:
            out_open = lambda: gzip.open(output_path, 'wt', encoding='utf-8', compresslevel=self.compress_level)
        else:
            out_open = lambda: open(output_path, 'w', encoding='utf-8')

        with out_open() as out_f:
            if prefix_mode:
                # Variant N: token(accumulation)
                out_f.write(absolute_token)
                out_f.write('(')
            else:
                # Standard: (accumulation)token
                out_f.write('(')

            # Stream accumulation content
            if self.file_path.exists():
                with self._open_file('r') as in_f:
                    while True:
                        chunk = in_f.read(chunk_size)
                        if not chunk:
                            break
                        out_f.write(chunk)

            if prefix_mode:
                out_f.write(')')
            else:
                out_f.write(')')
                out_f.write(absolute_token)

        return output_path

    def get_file_path(self) -> Path:
        """Return path to accumulation file."""
        self._flush()
        return self.file_path

    def __del__(self):
        """Cleanup on deletion (optional)."""
        # Flush buffer to ensure data is written
        self._flush()

