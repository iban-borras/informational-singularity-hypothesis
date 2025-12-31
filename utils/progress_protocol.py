"""
Progress Protocol for subprocess communication.

This module provides a simple protocol for subprocesses to report progress
to a parent process, which can then display a proper tqdm progress bar.

Protocol format:
    PROGRESS_START:<total>:<description>
    PROGRESS:<current>
    PROGRESS_END

Example:
    # In subprocess:
    progress_start(100, "Encoding Φ")
    for i in range(100):
        do_work()
        progress_update(i + 1)
    progress_end()
    
    # Parent process parses these and shows tqdm bar

Author: HSI Project
Date: December 2025
"""

import sys

# Prefix for progress messages (easy to parse)
PREFIX_START = "@@PROGRESS_START@@"
PREFIX_UPDATE = "@@PROGRESS@@"
PREFIX_END = "@@PROGRESS_END@@"


def progress_start(total: int, description: str = "Processing") -> None:
    """Signal start of a progress-tracked operation."""
    print(f"{PREFIX_START}:{total}:{description}", flush=True)


def progress_update(current: int) -> None:
    """Update progress (current out of total)."""
    print(f"{PREFIX_UPDATE}:{current}", flush=True)


def progress_end() -> None:
    """Signal end of progress-tracked operation."""
    print(f"{PREFIX_END}", flush=True)


class ProgressReporter:
    """
    Context manager for progress reporting.
    
    Usage:
        with ProgressReporter(total=100, desc="Encoding") as progress:
            for i in range(100):
                do_work()
                progress.update(i + 1)
    
    Or with automatic updates:
        with ProgressReporter(total=100, desc="Encoding") as progress:
            for i in progress.iter(range(100)):
                do_work()
    """
    
    def __init__(self, total: int, description: str = "Processing", 
                 update_percent: int = 1):
        """
        Args:
            total: Total number of items
            description: Description for progress bar
            update_percent: Only send update every N percent (reduces output)
        """
        self.total = total
        self.description = description
        self.update_percent = update_percent
        self.current = 0
        self._last_reported_pct = -1
    
    def __enter__(self):
        progress_start(self.total, self.description)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        progress_end()
        return False
    
    def update(self, current: int) -> None:
        """Update progress to current value."""
        self.current = current
        # Only report if percentage changed enough
        if self.total > 0:
            pct = (current * 100) // self.total
            if pct >= self._last_reported_pct + self.update_percent:
                progress_update(current)
                self._last_reported_pct = pct
    
    def iter(self, iterable):
        """Iterate with automatic progress updates."""
        for i, item in enumerate(iterable):
            yield item
            self.update(i + 1)


def parse_progress_line(line: str) -> dict | None:
    """
    Parse a progress protocol line.
    
    Returns:
        None if not a progress line
        {'type': 'start', 'total': int, 'desc': str}
        {'type': 'update', 'current': int}
        {'type': 'end'}
    """
    line = line.strip()
    
    if line.startswith(PREFIX_START):
        parts = line[len(PREFIX_START)+1:].split(":", 1)
        if len(parts) >= 2:
            return {'type': 'start', 'total': int(parts[0]), 'desc': parts[1]}
    
    elif line.startswith(PREFIX_UPDATE):
        parts = line[len(PREFIX_UPDATE)+1:].split(":")
        if parts:
            return {'type': 'update', 'current': int(parts[0])}
    
    elif line.startswith(PREFIX_END):
        return {'type': 'end'}
    
    return None

