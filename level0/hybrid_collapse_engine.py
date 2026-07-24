"""
Hybrid Collapse Engine — RAM-based regex processing with block segmentation

This module implements an exact, restartable hybrid approach:
1. Read bounded text chunks instead of using the RAM budget as a read size
2. Preserve a possible cross-chunk regex suffix
3. Write independently valid checkpoint parts
4. Resume interrupted passes without recomputing completed parts

Supports optional gzip compression for space-constrained environments.

Regex work remains vectorized while memory stays bounded for inputs larger
than available RAM.

Author: Iban Borràs with Augment Agent (Sophia)
Date: November 2025
"""

import re
import gzip
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Callable, Optional, Tuple, IO


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
    Processes collapse using bounded chunks and vectorized regex.

    Checkpoint parts are atomic and can be replayed after process or host
    interruption. A completed pass is cached until its enclosing iteration
    finishes.
    """
    
    def __init__(
        self,
        max_ram_bytes: int = 30_000_000_000,  # 30 GB default
        simplify_fn: Optional[Callable[[str], str]] = None,
        compress: bool = False,
        compress_level: int = 1,
        stream_chunk_chars: Optional[int] = None,
        checkpoint_chars: Optional[int] = None,
    ):
        """
        Initialize hybrid collapse engine.

        Args:
            max_ram_bytes: Maximum RAM to use per block
            simplify_fn: Collapse function (default: AND)
            compress: Whether to use gzip compression for temp files
            compress_level: Gzip compression level (1=fast, 9=max)
        """
        self.max_ram_bytes = max_ram_bytes
        self.simplify_fn = simplify_fn or _simplify_and
        self.compress = compress
        self.compress_level = compress_level
        # Compile regex once for performance
        self._pattern = re.compile(r'\(([01]+)\)')
        self._pending_suffix = re.compile(r'\([01]*$')

        # A RAM "limit" cannot also be the read size: regex replacement keeps
        # input and output alive concurrently. Bound each read to a small
        # fraction of the budget and checkpoint several reads as one part.
        env_chunk_mb = int(os.environ.get("HSI_HYBRID_CHUNK_MB", "64"))
        env_checkpoint_mb = int(os.environ.get("HSI_HYBRID_CHECKPOINT_MB", "512"))
        default_chunk = max(1, env_chunk_mb * 1_000_000)
        self.stream_chunk_chars = (
            int(stream_chunk_chars)
            if stream_chunk_chars is not None
            else min(default_chunk, max(1, max_ram_bytes // 4))
        )
        self.checkpoint_chars = (
            int(checkpoint_chars)
            if checkpoint_chars is not None
            else max(self.stream_chunk_chars, env_checkpoint_mb * 1_000_000)
        )

    def _open_file(self, path: Path, mode: str) -> IO:
        """Open file with appropriate method based on compression setting."""
        is_compressed = str(path).endswith('.gz')
        if is_compressed:
            if 'r' in mode:
                return gzip.open(path, 'rt', encoding='utf-8')
            else:
                return gzip.open(path, 'wt', encoding='utf-8', compresslevel=self.compress_level)
        else:
            return open(path, mode, encoding='utf-8')
    
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
    
    def _collapse_regex(self, data: str, max_chunk: int = 500_000_000) -> Tuple[str, bool]:
        """
        Apply one pass of regex collapse on innermost parentheses.

        For very large data, processes in overlapping chunks to avoid MemoryError.

        Args:
            data: String to collapse
            max_chunk: Maximum chunk size for regex processing (default 500MB)

        Returns:
            Tuple (collapsed_string, had_changes)
        """
        # If data is small enough, process directly
        if len(data) <= max_chunk:
            had_changes = [False]

            def replacer(m):
                had_changes[0] = True
                return self.simplify_fn(m.group(1))

            result = self._pattern.sub(replacer, data)
            return result, had_changes[0]

        # Large data: process in chunks with overlap
        print(f"      [regex] Large data ({len(data)/1e9:.2f}GB), chunking...", flush=True)

        total_had_changes = False
        result_parts = []
        pos = 0
        chunk_num = 0
        overlap = 10000  # Overlap to handle patterns at boundaries

        while pos < len(data):
            chunk_num += 1
            end = min(pos + max_chunk, len(data))

            # Include overlap from previous chunk boundary
            chunk = data[pos:end]

            had_changes = [False]
            def replacer(m):
                had_changes[0] = True
                return self.simplify_fn(m.group(1))

            collapsed = self._pattern.sub(replacer, chunk)

            if had_changes[0]:
                total_had_changes = True

            # For non-first chunks, skip the overlap portion
            if pos > 0 and len(collapsed) > overlap:
                result_parts.append(collapsed[overlap:])
            else:
                result_parts.append(collapsed)

            # Progress
            if chunk_num % 10 == 0:
                pct = (end / len(data)) * 100
                print(f"      [regex] Chunk {chunk_num}: {pct:.1f}%", flush=True)

            pos = end - overlap if end < len(data) else end
            if pos <= 0:
                pos = end

        print(f"      [regex] Done: {chunk_num} chunks processed", flush=True)
        return ''.join(result_parts), total_had_changes
    
    def collapse_one_pass(
        self,
        input_path: Path,
        output_path: Path,
        log_progress: bool = True,
        *,
        checkpoint_key: Optional[str] = None,
        expected_input_chars: Optional[int] = None,
    ) -> Tuple[int, bool]:
        """
        Collapse innermost parentheses in one pass, processing in RAM blocks.

        Automatically handles compressed (.gz) files based on extension.

        Args:
            input_path: Input file path
            output_path: Output file path
            log_progress: Whether to show progress

        Returns:
            Tuple (output_size, had_changes)
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        physical_size = input_path.stat().st_size
        compressed_input = str(input_path).endswith('.gz')
        identity = self._source_identity(input_path, checkpoint_key)
        complete_path = self._complete_path(output_path)
        parts_dir = self._parts_dir(output_path)
        checkpoint_path = parts_dir / "checkpoint.json"
        carry_path = parts_dir / "carry.txt.gz"

        completed = self._load_json(complete_path)
        if (
            completed
            and output_path.exists()
            and completed.get("source_identity") == identity
            and output_path.stat().st_size
            == int(completed.get("output_physical_bytes", -1))
        ):
            if log_progress:
                print(
                    f"   [hybrid] Reusing completed pass checkpoint: {output_path.name}",
                    flush=True,
                )
            return int(completed["output_chars"]), bool(completed["had_changes"])

        state = self._load_json(checkpoint_path)
        if not state or state.get("source_identity") != identity:
            if parts_dir.exists():
                shutil.rmtree(parts_dir)
            parts_dir.mkdir(parents=True, exist_ok=True)
            state = {
                "version": 1,
                "source_identity": identity,
                "input_chars": 0,
                "output_chars": 0,
                "had_changes": False,
                "parts": [],
            }
            self._write_carry(carry_path, "")
            self._write_json_atomic(checkpoint_path, state)
        else:
            self._validate_parts(parts_dir, state)

        chars_processed = int(state["input_chars"])
        chars_written = int(state["output_chars"])
        total_had_changes = bool(state["had_changes"])
        carry_over = self._read_carry(carry_path)
        progress_mode = os.environ.get("HSI_PROGRESS_MODE", "auto").strip().lower()
        progress_interactive = (
            progress_mode not in {"log", "plain", "none"}
            and sys.stdout.isatty()
        )
        progress_last_log = 0.0

        started_at = time.perf_counter()
        with self._open_file(input_path, 'r') as in_f:
            if chars_processed:
                self._skip_input(
                    in_f,
                    chars_processed,
                    log_progress=log_progress,
                    progress_mode=progress_mode,
                )
                if log_progress:
                    print(
                        f"   [hybrid] Resuming pass at {chars_processed:,} input chars "
                        f"from {len(state['parts'])} completed parts",
                        flush=True,
                    )

            eof = False
            while not eof:
                part_number = len(state["parts"]) + 1
                part_suffix = ".gz" if self.compress else ".txt"
                part_path = parts_dir / f"part-{part_number:06d}{part_suffix}"
                part_tmp = parts_dir / f"part-{part_number:06d}.tmp{part_suffix}"
                part_input = 0
                part_output = 0
                part_changed = False

                with self._open_file(part_tmp, 'w') as part_f:
                    while part_input < self.checkpoint_chars:
                        chunk = in_f.read(self.stream_chunk_chars)
                        if not chunk:
                            eof = True
                            process_data = carry_over
                            carry_over = ""
                        else:
                            data = carry_over + chunk
                            carry_match = self._pending_suffix.search(data)
                            if carry_match:
                                process_data = data[:carry_match.start()]
                                carry_over = carry_match.group(0)
                            else:
                                process_data = data
                                carry_over = ""
                            chars_processed += len(chunk)
                            part_input += len(chunk)

                        collapsed, had_changes = self._collapse_regex(process_data)
                        part_f.write(collapsed)
                        part_output += len(collapsed)
                        part_changed = part_changed or had_changes

                        del process_data, collapsed
                        if chunk:
                            del chunk
                        if eof:
                            break

                        if log_progress and progress_mode != "none":
                            now = time.perf_counter()
                            if (
                                progress_interactive
                                or now - progress_last_log >= 30.0
                            ):
                                message = self._progress_message(
                                    chars_processed,
                                    expected_input_chars,
                                    started_at,
                                )
                                print(
                                    message,
                                    end='\r' if progress_interactive else '\n',
                                    flush=True,
                                )
                                progress_last_log = now

                with open(part_tmp, "rb+") as part_sync:
                    os.fsync(part_sync.fileno())
                os.replace(part_tmp, part_path)
                chars_written += part_output
                total_had_changes = total_had_changes or part_changed
                state["input_chars"] = chars_processed
                state["output_chars"] = chars_written
                state["had_changes"] = total_had_changes
                state["parts"].append(
                    {
                        "name": part_path.name,
                        "input_chars": part_input,
                        "output_chars": part_output,
                        "had_changes": part_changed,
                        "physical_bytes": part_path.stat().st_size,
                    }
                )
                self._write_carry(carry_path, carry_over)
                self._write_json_atomic(checkpoint_path, state)

                if log_progress:
                    print(
                        f"   [hybrid] Checkpoint part {part_number}: "
                        f"{chars_processed:,} input chars, {chars_written:,} output chars",
                        flush=True,
                    )

        assembling = output_path.with_name(output_path.name + ".assembling")
        with open(assembling, "wb") as dst:
            for part in state["parts"]:
                with open(parts_dir / part["name"], "rb") as src:
                    shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
            dst.flush()
            os.fsync(dst.fileno())
        os.replace(assembling, output_path)

        completed = {
            "version": 1,
            "source_identity": identity,
            "output_chars": chars_written,
            "had_changes": total_had_changes,
            "output_physical_bytes": output_path.stat().st_size,
        }
        self._write_json_atomic(complete_path, completed)
        shutil.rmtree(parts_dir)

        compress_note = " (compressed)" if self.compress else ""
        if log_progress and progress_interactive:
            print("", flush=True)
        if log_progress:
            print(
                f"   [hybrid] Complete: {physical_size:,} physical bytes "
                f"in {len(state['parts'])} checkpoint parts{compress_note}; "
                f"logical_out={chars_written:,}",
                flush=True,
            )

        return chars_written, total_had_changes

    @staticmethod
    def _parts_dir(output_path: Path) -> Path:
        return output_path.with_name(output_path.name + ".parts")

    @staticmethod
    def _complete_path(output_path: Path) -> Path:
        return output_path.with_name(output_path.name + ".complete.json")

    @staticmethod
    def _load_json(path: Path) -> Optional[dict]:
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    @staticmethod
    def _source_identity(input_path: Path, checkpoint_key: Optional[str]) -> dict:
        stat = input_path.stat()
        identity = {
            "path": str(input_path.resolve()),
            "physical_bytes": stat.st_size,
        }
        if checkpoint_key is not None:
            identity["checkpoint_key"] = checkpoint_key
        else:
            identity["mtime_ns"] = stat.st_mtime_ns
        return identity

    @staticmethod
    def _write_carry(path: Path, carry: str) -> None:
        tmp = path.with_name(path.name + ".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=1) as f:
            f.write(carry)
        with open(tmp, "rb+") as carry_sync:
            os.fsync(carry_sync.fileno())
        os.replace(tmp, path)

    @staticmethod
    def _read_carry(path: Path) -> str:
        if not path.exists():
            return ""
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def _validate_parts(parts_dir: Path, state: dict) -> None:
        expected = {part["name"]: part for part in state.get("parts", [])}
        for name, record in expected.items():
            path = parts_dir / name
            if not path.exists():
                raise RuntimeError(f"Hybrid checkpoint part missing: {name}")
            if path.stat().st_size != int(record["physical_bytes"]):
                raise RuntimeError(f"Hybrid checkpoint part size mismatch: {name}")
        for path in parts_dir.glob("part-*"):
            if path.name not in expected:
                path.unlink()

    def _skip_input(
        self,
        in_f: IO,
        chars_to_skip: int,
        *,
        log_progress: bool,
        progress_mode: str,
    ) -> None:
        remaining = chars_to_skip
        last_log = time.perf_counter()
        while remaining:
            chunk = in_f.read(min(self.stream_chunk_chars, remaining))
            if not chunk:
                raise RuntimeError(
                    f"Hybrid checkpoint offset {chars_to_skip:,} exceeds input length"
                )
            remaining -= len(chunk)
            if (
                log_progress
                and progress_mode != "none"
                and time.perf_counter() - last_log >= 30.0
            ):
                print(
                    f"   [hybrid] Recovery seek: {chars_to_skip - remaining:,}/"
                    f"{chars_to_skip:,} chars",
                    flush=True,
                )
                last_log = time.perf_counter()

    @staticmethod
    def _progress_message(
        chars_processed: int,
        expected_input_chars: Optional[int],
        started_at: float,
    ) -> str:
        elapsed = max(time.perf_counter() - started_at, 1e-9)
        rate = chars_processed / elapsed
        if expected_input_chars:
            pct = min(100.0, chars_processed * 100.0 / expected_input_chars)
            remaining = max(0, expected_input_chars - chars_processed)
            eta = remaining / rate if rate > 0 else 0
            return (
                f"   [hybrid] {pct:6.2f}% | {chars_processed:,}/"
                f"{expected_input_chars:,} chars | {rate/1e6:.2f} Mchar/s | "
                f"eta={eta/3600:.1f}h"
            )
        return (
            f"   [hybrid] {chars_processed:,} chars | "
            f"{rate/1e6:.2f} Mchar/s"
        )

    def cleanup_checkpoint_artifacts(self, output_path: Path) -> None:
        parts_dir = self._parts_dir(Path(output_path))
        complete_path = self._complete_path(Path(output_path))
        if parts_dir.exists():
            shutil.rmtree(parts_dir)
        if complete_path.exists():
            complete_path.unlink()

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

        # Use compressed temp files if compression is enabled
        ext = ".tmp.gz" if self.compress else ".tmp"

        while True:
            pass_num += 1
            next_file = temp_dir / f"hybrid_pass_{pass_num}{ext}"

            output_size, had_changes = self.collapse_one_pass(
                current_file, next_file, log_progress=log_progress
            )

            # Clean up previous temp file
            if current_file != input_path and current_file.exists():
                current_file.unlink()
                self.cleanup_checkpoint_artifacts(current_file)

            if not had_changes or output_size <= 1:
                # Done - move to final output
                if next_file != output_path:
                    next_file.rename(output_path)
                    self.cleanup_checkpoint_artifacts(next_file)
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
    log_progress: bool = True,
    compress: bool = False,
    compress_level: int = 1
) -> Tuple[int, bool]:
    """
    Convenience function for one-pass hybrid collapse.

    Args:
        input_path: Input file path
        output_path: Output file path
        max_ram_bytes: Maximum RAM to use per block
        simplify_fn: Collapse function (default: AND)
        log_progress: Whether to show progress
        compress: Whether to use gzip compression for temp files
        compress_level: Gzip compression level (1=fast, 9=max)

    Returns:
        Tuple (output_size, had_changes)
    """
    engine = HybridCollapseEngine(
        max_ram_bytes=max_ram_bytes,
        simplify_fn=simplify_fn,
        compress=compress,
        compress_level=compress_level
    )
    return engine.collapse_one_pass(input_path, output_path, log_progress)
