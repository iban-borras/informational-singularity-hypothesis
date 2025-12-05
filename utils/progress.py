#!/usr/bin/env python3
"""
Progress indicator utilities for HSI long-running operations.
Uses \r overwrite for clean, single-line progress updates.
"""
from __future__ import annotations
import sys
import time
from typing import Optional

# Spinner frames for indeterminate progress
SPINNER = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']

def format_time(seconds: float) -> str:
    """Format seconds as human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s:02d}s"
    else:
        h, rem = divmod(int(seconds), 3600)
        m, s = divmod(rem, 60)
        return f"{h}h {m:02d}m"

def progress_print(message: str, overwrite: bool = True, end: str = ""):
    """Print with immediate flush, optionally overwriting the current line."""
    if overwrite:
        # Clear line and return to start
        sys.stdout.write(f"\r\033[K{message}{end}")
    else:
        sys.stdout.write(f"{message}{end}")
    sys.stdout.flush()

def progress_done(message: str):
    """Print final message and move to new line."""
    sys.stdout.write(f"\r\033[K{message}\n")
    sys.stdout.flush()


class ProgressIndicator:
    """Context manager for tracking progress of long operations."""
    
    def __init__(self, task_name: str, total: Optional[int] = None):
        self.task_name = task_name
        self.total = total
        self.current = 0
        self.start_time = None
        self.spinner_idx = 0
        self._last_update = 0
        self._update_interval = 0.1  # Update at most every 100ms
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        self._show()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.start_time
        if exc_type is None:
            progress_done(f"[✓ {self.task_name}] Completed in {format_time(elapsed)}")
        else:
            progress_done(f"[✗ {self.task_name}] Failed after {format_time(elapsed)}")
        return False
    
    def update(self, current: Optional[int] = None, message: str = ""):
        """Update progress. Call periodically during long operations."""
        now = time.perf_counter()
        if now - self._last_update < self._update_interval:
            return  # Throttle updates
        self._last_update = now
        
        if current is not None:
            self.current = current
        self._show(message)
    
    def tick(self, message: str = ""):
        """Increment progress by 1."""
        self.current += 1
        self.update(self.current, message)
    
    def _show(self, extra: str = ""):
        elapsed = time.perf_counter() - self.start_time if self.start_time else 0
        
        if self.total and self.total > 0:
            # Determinate progress
            pct = min(100, (self.current / self.total) * 100)
            eta = ""
            if pct > 0:
                eta_sec = (elapsed / pct) * (100 - pct)
                eta = f" | ETA ~{format_time(eta_sec)}"
            bar = self._make_bar(pct)
            msg = f"[⏳ {self.task_name}] {bar} {pct:5.1f}% | {format_time(elapsed)}{eta}"
        else:
            # Indeterminate progress (spinner)
            spinner = SPINNER[self.spinner_idx % len(SPINNER)]
            self.spinner_idx += 1
            msg = f"[{spinner} {self.task_name}] Processing... | {format_time(elapsed)}"
        
        if extra:
            msg += f" | {extra}"
        
        progress_print(msg)
    
    def _make_bar(self, pct: float, width: int = 20) -> str:
        """Create a simple progress bar."""
        filled = int(width * pct / 100)
        empty = width - filled
        return f"[{'█' * filled}{'░' * empty}]"


def heartbeat(task_name: str, interval: float = 5.0):
    """
    Create a simple heartbeat generator for long operations.
    Usage:
        hb = heartbeat("MyTask")
        for item in items:
            next(hb)  # Prints heartbeat every `interval` seconds
            process(item)
    """
    start = time.perf_counter()
    last_beat = start
    spinner_idx = 0
    
    while True:
        now = time.perf_counter()
        if now - last_beat >= interval:
            elapsed = now - start
            spinner = SPINNER[spinner_idx % len(SPINNER)]
            spinner_idx += 1
            progress_print(f"[{spinner} {task_name}] Still working... | {format_time(elapsed)}")
            last_beat = now
        yield

