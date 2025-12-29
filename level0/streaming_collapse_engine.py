"""
Streaming Collapse Engine — Memory-efficient collapse for massive files

This module implements streaming-based collapse operations that process
files chunk-by-chunk without loading entire content into RAM.

The key insight is that parenthesis-based collapse can be done incrementally:
- Track nesting depth with a stack
- Buffer content of innermost groups
- Collapse when closing parenthesis found
- Write results to output file

Author: Iban Borràs with Augment Agent (Sophia)
Date: November 2025
"""

import os
import warnings
from pathlib import Path
from typing import Callable, Optional, List

# Easter egg: Theoretical limit that should never be reached (empirically verified Dec 2025)
# HSI variants saturate at max_depth ~15-17, so 100MB buffer is science fiction 🚀
_BUFFER_WARNING_THRESHOLD = 100 * 1024 * 1024  # 100 MB
_buffer_warning_shown = False


def _simplify_and(seq: str) -> str:
    """AND collapse: returns '1' only if all bits are '1', else '0'."""
    if not seq:
        return ""
    clean = ''.join(c for c in seq if c in '01')
    if not clean:
        return ""
    return '1' if all(c == '1' for c in clean) else '0'


class StreamingCollapseEngine:
    """
    Processes collapse operations on massive files using streaming.
    
    Instead of loading entire file into RAM, processes chunk-by-chunk
    while maintaining parenthesis nesting state.
    
    Memory usage: O(max_nesting_depth * avg_group_size)
    Typically < 100 MB even for 100+ GB files.
    """
    
    def __init__(
        self,
        chunk_size: int = 10_000_000,  # 10 MB chunks
        simplify_fn: Optional[Callable[[str], str]] = None
    ):
        """
        Initialize streaming collapse engine.
        
        Args:
            chunk_size: Bytes to read per chunk (default 10 MB)
            simplify_fn: Collapse function (default: AND)
        """
        self.chunk_size = chunk_size
        self.simplify_fn = simplify_fn or _simplify_and
        
        # State for streaming processing
        self._stack: List[List[str]] = []  # Stack of buffers for each nesting level
        self._output_buffer: List[str] = []  # Buffer for level 0 output
        self._chars_processed = 0
        self._output_file = None
    
    def collapse_file(
        self,
        input_path: Path,
        output_path: Path,
        log_progress: bool = True
    ) -> int:
        """
        Collapse file using streaming - main entry point.
        
        Args:
            input_path: Path to input file with parenthesized content
            output_path: Path for collapsed output
            log_progress: Whether to print progress
            
        Returns:
            Length of collapsed output
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        file_size = input_path.stat().st_size
        
        # Reset state
        self._stack = []
        self._output_buffer = []
        self._chars_processed = 0
        
        with open(output_path, 'w', encoding='utf-8') as out_f:
            self._output_file = out_f
            
            with open(input_path, 'r', encoding='utf-8') as in_f:
                while True:
                    chunk = in_f.read(self.chunk_size)
                    if not chunk:
                        break
                    
                    self._process_chunk(chunk)
                    
                    if log_progress and file_size > self.chunk_size:
                        pct = (self._chars_processed / file_size) * 100
                        print(f"   [streaming] Progress: {pct:.1f}%", end='\r')
            
            # Flush remaining output buffer
            self._flush_output()
            
            # Process any remaining content in stack (shouldn't happen with valid input)
            self._finalize()
        
        if log_progress and file_size > self.chunk_size:
            print(f"   [streaming] Progress: 100.0% - Complete")
        
        # Return output length
        return output_path.stat().st_size
    
    def _process_chunk(self, chunk: str) -> None:
        """Process a chunk of input, updating state."""
        for char in chunk:
            self._chars_processed += 1
            self._process_char(char)
    
    def _process_char(self, char: str) -> None:
        """Process single character with parenthesis tracking."""
        if char == '(':
            # Start new nesting level
            self._stack.append([])
        
        elif char == ')':
            if self._stack:
                # Collapse current level
                current_buffer = ''.join(self._stack.pop())
                collapsed = self.simplify_fn(current_buffer)
                
                if self._stack:
                    # Add result to parent level
                    self._stack[-1].append(collapsed)
                else:
                    # At level 0, add to output
                    self._output_buffer.append(collapsed)
                    self._maybe_flush_output()
            # else: unmatched ')' - ignore or could raise
        
        elif char in '01':
            if self._stack:
                # Inside parentheses - add to current buffer
                self._stack[-1].append(char)
            else:
                # Outside parentheses - direct to output
                self._output_buffer.append(char)
                self._maybe_flush_output()
        
        # Ignore other characters (whitespace, etc.)

    def _maybe_flush_output(self) -> None:
        """Flush output buffer if large enough."""
        if len(self._output_buffer) >= 100_000:  # 100K chars
            self._flush_output()

    def _flush_output(self) -> None:
        """Write output buffer to file."""
        if self._output_buffer and self._output_file:
            self._output_file.write(''.join(self._output_buffer))
            self._output_buffer = []

    def _finalize(self) -> None:
        """Handle any remaining content in stack (edge cases)."""
        # If stack is not empty, collapse remaining levels
        while self._stack:
            current_buffer = ''.join(self._stack.pop())
            collapsed = self.simplify_fn(current_buffer)

            if self._stack:
                self._stack[-1].append(collapsed)
            else:
                self._output_buffer.append(collapsed)

        self._flush_output()


def collapse_file_streaming(
    input_path: Path,
    output_path: Path,
    simplify_fn: Optional[Callable[[str], str]] = None,
    chunk_size: int = 10_000_000,
    log_progress: bool = True
) -> int:
    """
    Convenience function to collapse a file using streaming.

    Args:
        input_path: Input file with parenthesized content
        output_path: Output file for collapsed result
        simplify_fn: Collapse function (default: AND)
        chunk_size: Chunk size in bytes
        log_progress: Whether to print progress

    Returns:
        Length of output
    """
    engine = StreamingCollapseEngine(
        chunk_size=chunk_size,
        simplify_fn=simplify_fn
    )
    return engine.collapse_file(input_path, output_path, log_progress)


class OneLevelCollapseEngine:
    """
    Collapses ONLY the innermost parentheses in a single pass.

    This maintains HSI semantics by allowing multiple passes with
    intermediate state accumulation between each pass.

    Example:
        Input:  "((01)1)0"
        Pass 1: "(01)0" (only innermost (01) collapsed to 0)
        Pass 2: "00" (remaining (0) collapsed to 0)
    """

    def __init__(
        self,
        chunk_size: int = 50_000_000,  # 50 MB chunks
        simplify_fn: Optional[Callable[[str], str]] = None
    ):
        self.chunk_size = chunk_size
        self.simplify_fn = simplify_fn or _simplify_and

    def collapse_one_level(
        self,
        input_path: Path,
        output_path: Path,
        log_progress: bool = False
    ) -> tuple:
        """
        Collapse only innermost parentheses (those with no nested parens).

        Args:
            input_path: Path to input file
            output_path: Path for output file
            log_progress: Whether to print progress

        Returns:
            Tuple (output_size, had_changes)
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        file_size = input_path.stat().st_size
        chars_processed = 0
        had_changes = False

        # Buffer for current innermost group
        current_group = []
        # Stack to track if current level contains nested parens
        # Each entry: (start_pos, has_nested_children)
        paren_stack = []
        # Output buffer
        output_buffer = []

        with open(output_path, 'w', encoding='utf-8') as out_f:
            with open(input_path, 'r', encoding='utf-8') as in_f:
                while True:
                    chunk = in_f.read(self.chunk_size)
                    if not chunk:
                        break

                    for char in chunk:
                        chars_processed += 1

                        if char == '(':
                            if paren_stack:
                                # Mark parent as having nested children
                                paren_stack[-1] = True
                                # Also add '(' to current content
                                current_group.append(char)
                            # Push new level (initially no nested children)
                            paren_stack.append(False)
                            current_group.append(char)

                        elif char == ')':
                            current_group.append(char)

                            if paren_stack:
                                has_nested = paren_stack.pop()

                                if not has_nested:
                                    # This is an innermost group - COLLAPSE IT
                                    # Extract content between ( and )
                                    group_str = ''.join(current_group)
                                    # Find the matching ( for this )
                                    depth = 0
                                    start_idx = len(group_str) - 1
                                    for i in range(len(group_str) - 1, -1, -1):
                                        if group_str[i] == ')':
                                            depth += 1
                                        elif group_str[i] == '(':
                                            depth -= 1
                                            if depth == 0:
                                                start_idx = i
                                                break

                                    # Split: before, innermost, (nothing after since we're at ')')
                                    before = group_str[:start_idx]
                                    innermost = group_str[start_idx+1:-1]  # Content without parens

                                    # Collapse the innermost
                                    collapsed = self.simplify_fn(innermost)
                                    had_changes = True

                                    # Rebuild current_group
                                    current_group = list(before + collapsed)
                                # else: has nested, keep as-is

                        elif char in '01':
                            current_group.append(char)

                        # Easter egg: If someone discovers a variant that reaches this,
                        # call us! HSI variants saturate at ~15-17 depth 🎉
                        global _buffer_warning_shown
                        if not _buffer_warning_shown and len(current_group) > _BUFFER_WARNING_THRESHOLD:
                            warnings.warn(
                                f"🚀 Congratulations! You've discovered a variant with buffer > 100MB! "
                                f"This should never happen (current_group={len(current_group):,} chars). "
                                f"Contact the HSI team!",
                                ResourceWarning
                            )
                            _buffer_warning_shown = True

                        # Flush periodically
                        if not paren_stack and current_group:
                            output_buffer.extend(current_group)
                            current_group = []

                            if len(output_buffer) >= 100_000:
                                out_f.write(''.join(output_buffer))
                                output_buffer = []

                    if log_progress and file_size > self.chunk_size:
                        pct = (chars_processed / file_size) * 100
                        print(f"   [collapse-level] Progress: {pct:.1f}%", end='\r')

                # Final flush
                if current_group:
                    output_buffer.extend(current_group)
                if output_buffer:
                    out_f.write(''.join(output_buffer))

        if log_progress and file_size > self.chunk_size:
            print(f"   [collapse-level] Progress: 100.0% - Complete")

        output_size = output_path.stat().st_size
        return output_size, had_changes


def collapse_one_level_streaming(
    input_path: Path,
    output_path: Path,
    simplify_fn: Optional[Callable[[str], str]] = None,
    chunk_size: int = 50_000_000,
    log_progress: bool = False
) -> tuple:
    """
    Convenience function for one-level collapse.

    Returns:
        Tuple (output_size, had_changes)
    """
    engine = OneLevelCollapseEngine(chunk_size=chunk_size, simplify_fn=simplify_fn)
    return engine.collapse_one_level(input_path, output_path, log_progress)

