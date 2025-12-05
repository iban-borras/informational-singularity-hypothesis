"""
Accumulation Manager — Disk-based storage for large accumulations

This module provides memory-efficient storage for the accumulation variable
in HSI Level 0 generator, using disk-based append-only files.

Author: Iban Borràs with Augment Agent (Sophia)
Date: November 2025
"""

import os
from pathlib import Path
from typing import Optional


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
    
    def __init__(self, output_dir: str, variant: str, resume_from: Optional[str] = None):
        """
        Initialize accumulation manager.
        
        Args:
            output_dir: Directory for temporary files
            variant: Variant code (e.g., "B")
            resume_from: Path to existing accumulation file (for checkpoint recovery)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.variant = variant
        self.file_path = self.output_dir / f"accumulation_{variant}.tmp"
        
        # Metadata (tracked in memory for fast access)
        self.current_length = 0
        self.clean_bits_count = 0
        
        # Buffer for batched writes (optimization)
        self.buffer = []
        self.buffer_size = 10_000  # Flush every 10K appends
        
        # Resume from existing file if provided
        if resume_from and Path(resume_from).exists():
            self._resume_from_file(resume_from)
        elif self.file_path.exists():
            # Clean up old file
            self.file_path.unlink()
    
    def _resume_from_file(self, file_path: str):
        """Resume from existing accumulation file."""
        import shutil
        shutil.copy(file_path, self.file_path)
        
        # Recalculate metadata
        print(f"   [AccumulationManager] Resuming from {file_path}...")
        with open(self.file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            self.current_length = len(content)
            self.clean_bits_count = sum(1 for c in content if c in '01')
        print(f"   [AccumulationManager] Loaded {self.current_length:,} chars, {self.clean_bits_count:,} clean bits")
    
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

        with open(source_path, 'r', encoding='utf-8') as src:
            with open(self.file_path, 'a', encoding='utf-8') as dst:
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
            with open(self.file_path, 'a', encoding='utf-8') as f:
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
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def read_chunk(self, start: int, end: int) -> str:
        """
        Read specific chunk of accumulation.
        
        Args:
            start: Start position (bytes)
            end: End position (bytes)
        
        Returns:
            Chunk of accumulation
        """
        # Flush buffer first
        self._flush()
        
        if not self.file_path.exists():
            return ""
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            f.seek(start)
            return f.read(end - start)
    
    def cleanup(self):
        """Remove temporary accumulation file."""
        # Flush buffer first
        self._flush()
        
        if self.file_path.exists():
            self.file_path.unlink()
            print(f"   [AccumulationManager] Cleaned up temporary file: {self.file_path}")
    
    def build_decay_frame(self, absolute_token: str, output_path: Optional[Path] = None) -> Path:
        """
        Build decay frame on disk without loading to RAM.

        Creates file with content: ({accumulation}){absolute_token}

        Args:
            absolute_token: Token to append (e.g., "1", "01", "10")
            output_path: Optional custom output path

        Returns:
            Path to the decay frame file
        """
        # Flush any pending writes
        self._flush()

        if output_path is None:
            output_path = self.output_dir / f"decay_frame_{self.variant}.tmp"

        chunk_size = 10_000_000  # 10 MB chunks

        with open(output_path, 'w', encoding='utf-8') as out_f:
            # Write opening parenthesis
            out_f.write('(')

            # Stream accumulation content
            if self.file_path.exists():
                with open(self.file_path, 'r', encoding='utf-8') as in_f:
                    while True:
                        chunk = in_f.read(chunk_size)
                        if not chunk:
                            break
                        out_f.write(chunk)

            # Write closing parenthesis and token
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

