"""
Hybrid Collapse Engine — RAM-based regex processing with block segmentation

This module implements an efficient hybrid approach:
1. Process files in large RAM blocks (configurable, e.g., 30 GB)
2. Use fast regex for collapse within each block
3. Handle block boundaries safely (cut at parenthesis depth 0)
4. Fallback to streaming for edge cases

This is MUCH faster than char-by-char streaming while still handling
files larger than available RAM.

Author: Iban Borràs with Augment Agent (Sophia)
Date: November 2025
"""

import re
from pathlib import Path
from typing import Callable, Optional, Tuple


def _simplify_and(seq: str) -> str:
    """AND collapse: returns '1' only if all bits are '1', else '0'."""
    if not seq:
        return ""
    clean = ''.join(c for c in seq if c in '01')
    if not clean:
        return ""
    return '1' if all(c == '1' for c in clean) else '0'


class HybridCollapseEngine:
    """
    Processes collapse using RAM blocks + regex for maximum speed.
    
    Strategy:
    1. Load block of max_ram_bytes into memory
    2. Find safe cut point (parenthesis depth = 0)
    3. Apply regex collapse repeatedly within block
    4. Write result, continue with next block
    """
    
    def __init__(
        self,
        max_ram_bytes: int = 30_000_000_000,  # 30 GB default
        simplify_fn: Optional[Callable[[str], str]] = None
    ):
        """
        Initialize hybrid collapse engine.
        
        Args:
            max_ram_bytes: Maximum RAM to use per block
            simplify_fn: Collapse function (default: AND)
        """
        self.max_ram_bytes = max_ram_bytes
        self.simplify_fn = simplify_fn or _simplify_and
        # Compile regex once for performance
        self._pattern = re.compile(r'\(([01]+)\)')
    
    def _find_safe_cut_point(self, data: str, max_pos: int) -> int:
        """
        Find the last position <= max_pos where parenthesis depth is 0.
        
        Scans backwards from max_pos to find a safe cut point.
        
        Args:
            data: The string to analyze
            max_pos: Maximum position to consider
            
        Returns:
            Safe cut position (depth 0), or 0 if not found
        """
        # Scan forward to build depth at each position
        # Then find last position with depth 0 before max_pos
        depth = 0
        last_zero_depth = 0
        
        for i, char in enumerate(data):
            if i > max_pos:
                break
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            
            if depth == 0:
                last_zero_depth = i + 1  # Position AFTER this char
        
        return last_zero_depth
    
    def _collapse_regex(self, data: str) -> Tuple[str, bool]:
        """
        Apply one pass of regex collapse on innermost parentheses.
        
        Returns:
            Tuple (collapsed_string, had_changes)
        """
        had_changes = [False]  # Use list to modify in lambda
        
        def replacer(m):
            had_changes[0] = True
            return self.simplify_fn(m.group(1))
        
        result = self._pattern.sub(replacer, data)
        return result, had_changes[0]
    
    def collapse_one_pass(
        self,
        input_path: Path,
        output_path: Path,
        log_progress: bool = True
    ) -> Tuple[int, bool]:
        """
        Collapse innermost parentheses in one pass, processing in RAM blocks.
        
        Args:
            input_path: Input file path
            output_path: Output file path
            log_progress: Whether to show progress
            
        Returns:
            Tuple (output_size, had_changes)
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        file_size = input_path.stat().st_size
        total_had_changes = False
        bytes_processed = 0
        
        with open(output_path, 'w', encoding='utf-8') as out_f:
            with open(input_path, 'r', encoding='utf-8') as in_f:
                carry_over = ""  # Content carried from previous block
                block_num = 0
                
                while True:
                    block_num += 1
                    # Read block (minus carry_over size to stay within RAM limit)
                    read_size = self.max_ram_bytes - len(carry_over)
                    chunk = in_f.read(read_size)
                    
                    if not chunk and not carry_over:
                        break  # Done
                    
                    # Combine carry_over with new chunk
                    data = carry_over + chunk
                    bytes_processed += len(chunk)
                    
                    # Check if this is the last block
                    is_last_block = len(chunk) < read_size

                    if is_last_block:
                        # Process all remaining data
                        process_data = data
                        carry_over = ""
                    else:
                        # Find safe cut point
                        safe_cut = self._find_safe_cut_point(data, len(data) - 1)
                        if safe_cut == 0:
                            # No safe cut found - entire block is inside parentheses
                            # This means we have a very deep nesting situation
                            # Use a fallback: process what we can
                            safe_cut = len(data)

                        process_data = data[:safe_cut]
                        carry_over = data[safe_cut:]

                    # Apply regex collapse to this block
                    collapsed, had_changes = self._collapse_regex(process_data)
                    if had_changes:
                        total_had_changes = True

                    # Write result
                    out_f.write(collapsed)

                    if log_progress:
                        pct = (bytes_processed / file_size) * 100
                        print(f"   [hybrid] Block {block_num}: {pct:.1f}% processed", end='\r')

                    if is_last_block:
                        break

        if log_progress:
            print(f"   [hybrid] Complete: {file_size:,} bytes in {block_num} blocks")

        output_size = output_path.stat().st_size
        return output_size, total_had_changes

    def collapse_fully(
        self,
        input_path: Path,
        output_path: Path,
        log_progress: bool = True
    ) -> int:
        """
        Collapse file completely (all levels) using multiple passes.

        This does NOT add intermediate states to accumulation.
        For HSI semantics, use collapse_with_intermediates instead.

        Args:
            input_path: Input file path
            output_path: Final output path
            log_progress: Whether to show progress

        Returns:
            Final output size
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        temp_dir = output_path.parent

        current_file = input_path
        pass_num = 0

        while True:
            pass_num += 1
            next_file = temp_dir / f"hybrid_pass_{pass_num}.tmp"

            output_size, had_changes = self.collapse_one_pass(
                current_file, next_file, log_progress=log_progress
            )

            # Clean up previous temp file
            if current_file != input_path and current_file.exists():
                current_file.unlink()

            if not had_changes or output_size <= 1:
                # Done - move to final output
                if next_file != output_path:
                    next_file.rename(output_path)
                if log_progress:
                    print(f"   [hybrid] Completed in {pass_num} passes")
                break

            current_file = next_file

        return output_path.stat().st_size


def collapse_hybrid(
    input_path: Path,
    output_path: Path,
    max_ram_bytes: int = 30_000_000_000,
    simplify_fn: Optional[Callable[[str], str]] = None,
    log_progress: bool = True
) -> Tuple[int, bool]:
    """
    Convenience function for one-pass hybrid collapse.

    Returns:
        Tuple (output_size, had_changes)
    """
    engine = HybridCollapseEngine(max_ram_bytes=max_ram_bytes, simplify_fn=simplify_fn)
    return engine.collapse_one_pass(input_path, output_path, log_progress)

