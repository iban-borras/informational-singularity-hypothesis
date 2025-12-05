"""
HSI Level 0 Generator — Basal‑Pure Dynamics

This module implements the basal‑pure iterative collapse between Nothingness (0)
and Absolute (1), generating the cumulative binary sequence Φ.

Author: Iban Borràs with collaboration from Augment Agent (Sophia)
Date: Jan 2025
"""

import re
import json
import time
import os
import numpy as np
from typing import Tuple, List, Callable, Optional, Dict, Any
from pathlib import Path
from .phi_snapshot_manager import PhiSnapshotManager
from .accumulation_manager import AccumulationManager
from .streaming_collapse_engine import StreamingCollapseEngine, OneLevelCollapseEngine
from .hybrid_collapse_engine import HybridCollapseEngine

# Default safety limits (can be overridden by config.json)
DEFAULT_MAX_RAM_GB = 30  # 30 GB default
DEFAULT_MAX_DISK_TB = 1.0  # 1 TB default

# Base path for outputs (anchored to project root)
BASE_PATH = Path(__file__).resolve().parent.parent  # hsi_agents_project/
RESULTS_DIR = BASE_PATH / "results"
REPORTS_DIR = BASE_PATH / "results" / "reports"
VISUALIZATIONS_DIR = BASE_PATH / "results" / "visualizations"
SNAPSHOT_DATA_DIR = BASE_PATH / "results" / "snapshots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOT_DATA_DIR.mkdir(parents=True, exist_ok=True)



def find_last_completed_iteration(output_dir: str, variant: str) -> int:
    """
    Find the last successfully completed iteration from saved snapshots.

    Args:
        output_dir: Directory where snapshots are saved
        variant: Variant code (e.g., "B")

    Returns:
        Last completed iteration number (0 if none found)
    """
    import json
    from pathlib import Path

    snapshot_dir = Path(output_dir) / "phi_snapshots" / f"var_{variant}_abs1"

    if not snapshot_dir.exists():
        return 0

    # Find all metadata files
    metadata_files = sorted(snapshot_dir.glob("phi_iter*.json"))

    if not metadata_files:
        return 0

    # Check each metadata file to find the last complete one
    last_complete = 0

    for meta_file in metadata_files:
        try:
            # Extract iteration number from filename
            iter_num = int(meta_file.stem.replace("phi_iter", ""))

            # Check if corresponding data file exists
            with open(meta_file, 'r') as f:
                meta = json.load(f)

            format_type = meta.get('format', 'v32')

            if format_type == 'v33_structural':
                data_file = meta_file.parent / f"phi_iter{iter_num}.struct.gz"
            else:
                data_file = meta_file.parent / f"phi_iter{iter_num}.bin.gz"

            # If both metadata and data exist, this iteration is complete
            if data_file.exists():
                last_complete = max(last_complete, iter_num)
        except (ValueError, json.JSONDecodeError, KeyError):
            continue

    return last_complete


def cleanup_incomplete_iteration(output_dir: str, variant: str, iteration: int):
    """
    Clean up any incomplete files from a failed iteration.

    Args:
        output_dir: Directory where snapshots are saved
        variant: Variant code (e.g., "B")
        iteration: Iteration number to clean up
    """
    from pathlib import Path

    snapshot_dir = Path(output_dir) / "phi_snapshots" / f"var_{variant}_abs1"

    if not snapshot_dir.exists():
        return

    # Remove all files for this iteration
    patterns = [
        f"phi_iter{iteration}.json",
        f"phi_iter{iteration}.struct.gz",
        f"phi_iter{iteration}.bin.gz"
    ]

    for pattern in patterns:
        file_path = snapshot_dir / pattern
        if file_path.exists():
            file_path.unlink()
            print(f"   [cleanup] Removed incomplete file: {file_path.name}")


def simulate_phi(
    max_iterations: int = 20,
    collapse_rule: str = "AND",
    collapse_depth: int = 1,
    simplify_fn: Optional[Callable] = None,
    save_snapshots: bool = True,
    output_dir: Optional[str] = None,
    use_compression: bool = True,
    memory_threshold: int = 10**8,
    variant: str = "B",  # Basal-pure variants: A (instant), B (stratified baseline)
    resume_from_checkpoint: bool = True,  # NEW: Enable checkpoint recovery
    config: Optional[Dict[str, Any]] = None  # Configuration dict for RAM/disk limits
) -> Tuple[str, List[str], Dict[str, Any]]:
    """
    Simula el procés iteratiu de la Hipòtesi de Singularitat Informacional.

    Args:
        max_iterations: Nombre màxim d'iteracions a executar
        collapse_rule: Regla de col·lapse ("AND", "ASYM", "CUSTOM")
        collapse_depth: Nivells de parèntesis a col·lapsar per iteració
        simplify_fn: Funció personalitzada de simplificació (si collapse_rule="CUSTOM")
        save_snapshots: Si guardar els estats intermedis
        output_dir: Directori on guardar els resultats
        use_compression: Si usar compressió automàtica per seqüències grans
        memory_threshold: Llindar de bits per activar compressió automàtica

    Returns:
        Tuple amb:
        - phi: seqüència final com a string de 0s i 1s
        - snapshots: llista amb l'estat de Φ a cada iteració (pot estar buida si usa compressió)
        - metadata: diccionari amb informació sobre l'execució
    """

    # Basal‑pure initialization (no params, no masks)
    # MEMORY OPTIMIZED: Use disk-based accumulation manager instead of string
    accumulation_manager = None  # Will be initialized after output_dir is set
    current_state = "0"  # R0 = No‑Res
    snapshots = []

    # Ensure config is a dict (for RAM/disk limit lookups)
    if config is None:
        config = {}

    # Normalize and store variant early (needed for snapshot pathing)
    variant = (variant or "B").upper()
    metadata = {
        "max_iterations": max_iterations,
        "iterations_completed": 0,
        "final_length": 0,
        "use_compression": use_compression,
        "memory_threshold": memory_threshold,
        "compression_activated": False,
        "snapshots_saved": [],
        "basal_pure": True,
        "variant": variant
    }

    # Absolute token parametrization ("1" | "10" | "01") via env HSI_ABSOLUTE_TOKEN
    absolute_token = os.environ.get("HSI_ABSOLUTE_TOKEN", "1")
    if absolute_token not in ("1", "10", "01"):
        absolute_token = "1"

    # Storage format: v32 (clean bits only) or v33 (structural 2-bit encoding)
    storage_format = os.environ.get("HSI_STORAGE_FORMAT", "v33").lower()
    if storage_format not in ("v32", "v33"):
        storage_format = "v33"  # Default to v33
    metadata["storage_format"] = storage_format

    # Configurar gestor de snapshots si cal
    snapshot_manager = None
    if use_compression and output_dir:
        # Save under results/phi_snapshots/var_{VARIANT}
        snapshot_dir = Path(output_dir) / "phi_snapshots" / f"var_{variant}"
        snapshot_manager = PhiSnapshotManager(
            data_dir=str(snapshot_dir),
            memory_threshold=memory_threshold
        )

    # Initialize AccumulationManager (disk-based storage for memory efficiency)
    if output_dir:
        accumulation_manager = AccumulationManager(
            output_dir=str(Path(output_dir) / "temp"),
            variant=variant
        )
    else:
        # Fallback to in-memory for small tests without output_dir
        accumulation_manager = None

    # Per-iteration logging control
    try:
        LOG_EVERY = int(os.environ.get("HSI_LOG_EVERY", "0"))  # 0 disables
    except Exception:
        LOG_EVERY = 0

    # ========== CHECKPOINT RECOVERY ==========
    start_iteration = 0

    if resume_from_checkpoint and output_dir:
        last_complete = find_last_completed_iteration(output_dir, variant)

        if last_complete > 0:
            print(f"\n{'='*60}")
            print(f"🔄 CHECKPOINT RECOVERY")
            print(f"{'='*60}")
            print(f"Found {last_complete} completed iterations")

            # Clean up any incomplete iteration
            cleanup_incomplete_iteration(output_dir, variant, last_complete + 1)

            # Load state from last completed iteration
            try:
                from level1.data_loader import load_phi_for_level1
                import sys
                import io

                # Suppress progress messages during checkpoint recovery
                print(f"Loading checkpoint data (this may take a moment)...", flush=True)
                old_stdout = sys.stdout
                sys.stdout = io.StringIO()  # Redirect stdout to suppress progress

                try:
                    phi_s, phi_o, meta = load_phi_for_level1(
                        output_dir,
                        last_complete,
                        return_structural=True,
                        return_observable=True,
                        return_metadata=True
                    )

                    # Restore state
                    current_state = phi_s if phi_s else phi_o

                    # Restore accumulation (reconstruct from all previous iterations)
                    # MEMORY OPTIMIZED: Use AccumulationManager instead of string concatenation
                    if accumulation_manager:
                        for i in range(1, last_complete + 1):
                            phi_s_i, phi_o_i, _ = load_phi_for_level1(
                                output_dir, i,
                                return_structural=True,
                                return_observable=True,
                                return_metadata=False
                            )
                            accumulation_manager.append(phi_s_i if phi_s_i else phi_o_i)
                finally:
                    # Restore stdout
                    sys.stdout = old_stdout

                start_iteration = last_complete

                print(f"✅ Restored state from iteration {last_complete}")
                print(f"   Current state length: {len(current_state):,}")
                print(f"   Accumulation length: {len(accumulation):,}")
                print(f"   Resuming from iteration {start_iteration + 1}")
                print(f"{'='*60}\n")

            except Exception as e:
                print(f"⚠️  WARNING: Could not restore checkpoint: {e}")
                print(f"   Starting from iteration 0")
                start_iteration = 0

    # Nota: collapse_rule/collapse_depth/simplify_fn are ignored in basal‑pure mode
    def _simplify_base(seq: str) -> str:
        """Basal AND-rule simplification: 01 and 10 annihilate; compress runs.
        This matches the original v0 behavior and enforces decay to No‑Res.
        """
        # Annihilate opposing pairs
        result = re.sub(r'(01|10)', '', seq)
        # Compress runs
        result = re.sub(r'0+', '0', result)
        result = re.sub(r'1+', '1', result)
        return result

    def _clean_sequence(seq: str, chunk_size: int = 10_000_000) -> str:
        """
        Remove structural parentheses; keep only 0/1 for observable Φ.

        MEMORY OPTIMIZED: Processes in chunks to avoid MemoryError on large strings.
        """
        if not seq:
            return ""

        # For small strings, use fast path
        if len(seq) <= chunk_size:
            return re.sub(r'[^01]', '', seq)

        # For large strings, process in chunks
        result = []
        total_chars = len(seq)

        for start in range(0, total_chars, chunk_size):
            end = min(start + chunk_size, total_chars)
            chunk = seq[start:end]

            # Clean chunk (keep only 0/1)
            cleaned = re.sub(r'[^01]', '', chunk)
            result.append(cleaned)

        return ''.join(result)

    def _count_clean_bits(seq: str) -> int:
        """
        Count only 0/1 characters without creating a new string.

        MEMORY OPTIMIZED: Much faster than len(_clean_sequence(seq)) for large strings.
        """
        return sum(1 for c in seq if c in '01')

    def _collapse_inside_parentheses(state: str, simplify_fn: Callable[[str], str] = None) -> str:
        """Collapse deterministically from innermost parentheses outward using given simplify_fn."""
        sf = simplify_fn or _simplify_base
        def _collapse_match(match: re.Match) -> str:
            inner = match.group(0)
            inner_clean = _clean_sequence(inner)
            return sf(inner_clean)
        collapsed = re.sub(r'\([01]+\)', _collapse_match, state)
        return sf(collapsed)

    def _collapse_inside_parentheses_one(state: str, simplify_fn: Callable[[str], str] = None) -> str:
        """Perform exactly one innermost-parentheses collapse (left-to-right)."""
        sf = simplify_fn or _simplify_base
        def _collapse_match(match: re.Match) -> str:
            inner_clean = _clean_sequence(match.group(0))
            return sf(inner_clean)
        # count=1 ensures a single replacement (leftmost match)
        collapsed = re.sub(r'\([01]+\)', _collapse_match, state, count=1)
        return sf(collapsed)

    def _collapse_global_ignore_parentheses(state: str, simplify_fn: Callable[[str], str] = None) -> str:
        """Global single-step collapse ignoring parentheses (control variant A)."""
        sf = simplify_fn or _simplify_base
        return sf(_clean_sequence(state))


    def _collapse_inside_parentheses_local(state: str, simplify_fn: Callable[[str], str] = None) -> str:
        """Like _collapse_inside_parentheses but do not simplify the whole state.
        Only simplify the contents of innermost parentheses and replace in-place.
        This preserves outer structure for stratified passes (B/C).
        """
        sf = simplify_fn or _simplify_base
        def _collapse_match(match: re.Match) -> str:
            inner_clean = _clean_sequence(match.group(0))
            return sf(inner_clean)
        return re.sub(r'\([01]+\)', _collapse_match, state)

    def _collapse_inside_parentheses_one_local(state: str, simplify_fn: Callable[[str], str] = None) -> str:
        """Like _collapse_inside_parentheses_one but without simplifying the whole state.
        Performs exactly one innermost replacement, preserving outer structure.
        """
        sf = simplify_fn or _simplify_base
        def _collapse_match(match: re.Match) -> str:
            inner_clean = _clean_sequence(match.group(0))
            return sf(inner_clean)
        return re.sub(r'\([01]+\)', _collapse_match, state, count=1)

    def _simplify_variant_d(seq: str) -> str:
        """Minimal asymmetry: 10→∅, 01→0; then compress runs."""
        result = re.sub(r'10', '', seq)
        result = re.sub(r'01', '0', result)
        result = re.sub(r'0+', '0', result)
        result = re.sub(r'1+', '1', result)
        return result

    def _simplify_variant_e(seq: str) -> str:
        """Ordered passes: remove all 01, then all 10; then compress runs."""
        result = re.sub(r'01', '', seq)
        result = re.sub(r'10', '', result)
        result = re.sub(r'0+', '0', result)
        result = re.sub(r'1+', '1', result)
        return result



    # Procés iteratiu principal (basal‑pure)
    # Per-iteration timing toggle
    ENABLE_TIMING = os.environ.get("HSI_ENABLE_TIMING", "1") == "1"

    # Load config for RAM/disk limits ONCE before loop
    max_ram_gb = config.get("level0_generation", {}).get("max_ram_gb", DEFAULT_MAX_RAM_GB)
    max_disk_tb = config.get("level0_generation", {}).get("max_disk_tb", DEFAULT_MAX_DISK_TB)
    max_ram_bytes = int(max_ram_gb * 1_000_000_000)
    max_disk_bytes = int(max_disk_tb * 1_000_000_000_000)

    for iteration in range(start_iteration, max_iterations):
        iter_t0 = time.perf_counter() if ENABLE_TIMING else None
        # 1) Actualitzar acumulació amb l'estat actual
        # MEMORY OPTIMIZED: Use AccumulationManager instead of string concatenation
        if accumulation_manager:
            accumulation_manager.append(current_state)
        else:
            # Fallback for tests without output_dir
            if 'accumulation' not in locals():
                accumulation = ""
            accumulation += current_state

        # 2) Construir marc de decaïment Dα = (Accα)ABS, on ABS ∈ {"1","10","01"}

        decay_frame_size = 0  # Track size for hybrid decision
        used_hybrid = False  # Flag to track if hybrid collapse was used

        if accumulation_manager:
            # Safety check: stop if accumulation exceeds disk limit
            current_accum_size = accumulation_manager.get_file_size()
            if current_accum_size > max_disk_bytes:
                print(f"\n⚠️  SAFETY STOP: Accumulation ({current_accum_size/1e12:.2f} TB) exceeds {max_disk_tb} TB limit!")
                print(f"    Stopping at iteration {iteration + 1} to protect disk integrity.")
                metadata["safety_stop"] = True
                metadata["safety_stop_reason"] = f"Accumulation exceeded {max_disk_tb} TB at iteration {iteration + 1}"
                break

            # Build decay frame directly on disk (no RAM needed)
            decay_frame_path = accumulation_manager.build_decay_frame(absolute_token)
            decay_frame_size = decay_frame_path.stat().st_size

            if decay_frame_size > max_ram_bytes:
                # File > max RAM: use HYBRID multi-pass collapse with RAM blocks
                used_hybrid = True
                print(f"   [hybrid] Decay frame: {decay_frame_size/1e9:.2f} GB > {max_ram_gb} GB RAM limit")
                print(f"   [hybrid] Using block-based collapse (regex in {max_ram_gb} GB blocks)")

                # Setup for multi-pass hybrid collapse
                temp_dir = accumulation_manager.output_dir
                current_file = decay_frame_path
                pass_num = 0

                # Select simplify function based on variant
                variant_simplify_fns = {
                    "B": _simplify_base,
                    "D": _simplify_variant_d,
                    "E": _simplify_variant_e,
                    "F": _simplify_base,
                    "G": _simplify_base,
                    "H": _simplify_base,
                }
                hybrid_simplify_fn = variant_simplify_fns.get(variant, _simplify_base)
                hybrid_engine = HybridCollapseEngine(max_ram_bytes=max_ram_bytes, simplify_fn=hybrid_simplify_fn)

                while True:
                    pass_num += 1

                    # Check disk safety before each pass
                    current_accum_size = accumulation_manager.get_file_size()
                    if current_accum_size > max_disk_bytes:
                        print(f"\n⚠️  SAFETY STOP: Accumulation ({current_accum_size/1e12:.2f} TB) exceeds limit!")
                        metadata["safety_stop"] = True
                        metadata["safety_stop_reason"] = f"Accumulation exceeded {max_disk_tb} TB during pass {pass_num}"
                        if current_file.exists() and current_file != decay_frame_path:
                            current_file.unlink()
                        if decay_frame_path.exists():
                            decay_frame_path.unlink()
                        break

                    # Append current state to accumulation (HSI semantics!)
                    accumulation_manager.append_from_file(current_file)

                    # Collapse one level using hybrid (regex in RAM blocks)
                    next_file = temp_dir / f"hybrid_{variant}_{pass_num}.tmp"
                    output_size, had_changes = hybrid_engine.collapse_one_pass(
                        current_file, next_file, log_progress=(pass_num == 1)
                    )

                    # Progress update
                    if pass_num % 3 == 0 or output_size < 1000:
                        print(f"   [pass {pass_num}] {current_file.stat().st_size:,} → {output_size:,} bytes")

                    # Clean up previous temp file
                    if current_file.exists() and current_file != decay_frame_path:
                        current_file.unlink()

                    # Check termination conditions
                    if not had_changes or output_size <= 1:
                        with open(next_file, 'r', encoding='utf-8') as f:
                            state = f.read()
                        if next_file.exists():
                            next_file.unlink()
                        print(f"   [hybrid] Completed in {pass_num} passes, final: '{state[:50]}'")
                        break

                    current_file = next_file

                # Clean up original decay frame
                if decay_frame_path.exists():
                    decay_frame_path.unlink()

            else:
                # File fits in RAM: use original fast approach
                if decay_frame_size > 100_000_000:  # >100MB
                    print(f"   [decay] Reading {decay_frame_size/1e9:.2f} GB to RAM...", flush=True)
                    t_read = time.perf_counter()
                with open(decay_frame_path, 'r', encoding='utf-8') as f:
                    decay_frame = f.read()
                if decay_frame_size > 100_000_000:
                    print(f"   [decay] Read complete in {time.perf_counter()-t_read:.1f}s", flush=True)
                if decay_frame_path.exists():
                    decay_frame_path.unlink()
                state = decay_frame
        else:
            decay_frame = f"({accumulation}){absolute_token}"
            state = decay_frame

        # Check if safety stop triggered
        if metadata.get("safety_stop"):
            break

        # 3) Aplicar col·lapse estratificat determinista inside→out
        #    Repetir fins estabilitzar o quedar en un sol símbol
        previous = None
        collapse_pass = 0
        state_len = len(state) if isinstance(state, str) else 0
        if state_len > 100_000_000:  # >100MB
            print(f"   [collapse] Starting collapse on {state_len/1e9:.2f} GB...", flush=True)
            collapse_t0 = time.perf_counter()

        if variant == "B":
            # For streaming path, multi-pass collapse already done with intermediates
            if used_hybrid:
                # After multi-pass streaming, state is already fully collapsed
                # Apply final global simplify if needed
                if len(state) > 1:
                    state = _collapse_global_ignore_parentheses(state)
            else:
                # Original logic for small data
                while state != previous:
                    collapse_pass += 1
                    if state_len > 100_000_000 and collapse_pass % 5 == 0:
                        print(f"   [collapse] Pass {collapse_pass}: {len(state):,} chars...", flush=True)
                    previous = state
                    if accumulation_manager:
                        accumulation_manager.append(state)
                    else:
                        accumulation += state
                    next_state = _collapse_inside_parentheses_local(state)
                    if len(next_state) == 1:
                        state = next_state
                        break
                    state = next_state
                # Final global simplify for B to ensure L0 collapse per iteration
                if len(state) > 1:
                    state = _collapse_global_ignore_parentheses(state)

        if state_len > 100_000_000:
            print(f"   [collapse] Done in {time.perf_counter()-collapse_t0:.1f}s ({collapse_pass} passes)", flush=True)

        elif variant == "G":
            # G — ABS parametritzable, sense auto-col·lapse intra-nivell
            # Semàntica: com B però NO afegim el 'final global simplify'. Manté estrictament el resultat estratificat.
            if used_hybrid:
                # After streaming, state is already collapsed - no further processing needed for G
                pass
            else:
                while state != previous:
                    previous = state
                    if accumulation_manager:
                        accumulation_manager.append(state)
                    else:
                        accumulation += state
                    next_state = _collapse_inside_parentheses_local(state)
                    if len(next_state) == 1 or next_state == state:
                        state = next_state
                        break
                    state = next_state

        elif variant == "H":
            # H — ABS parametritzable, AMB auto-col·lapse intra-nivell
            # Semàntica: després de cada simplificació d'un nivell interior, apliquem una passada global suau.
            if used_hybrid:
                # After streaming, apply final global simplify if needed
                if len(state) > 1:
                    state = _collapse_global_ignore_parentheses(state)
            else:
                stabilization_guard = 0
                while True:
                    if accumulation_manager:
                        accumulation_manager.append(state)
                    else:
                        accumulation += state
                    next_state = _collapse_inside_parentheses_one_local(state)
                    if len(next_state) > 1:
                        next_state = _collapse_global_ignore_parentheses(next_state)
                    stabilization_guard += 1
                    if next_state == state or len(next_state) == 1 or stabilization_guard > 1_000_000:
                        state = next_state
                        break
                    state = next_state

        elif variant == "D":
            # Minimal asymmetry using stratified passes (like B, but with D's simplify rule)
            if used_hybrid:
                # After streaming, apply final global simplify with D's rule
                if len(state) > 1:
                    state = _collapse_global_ignore_parentheses(state, simplify_fn=_simplify_variant_d)
            else:
                while state != previous:
                    previous = state
                    if accumulation_manager:
                        accumulation_manager.append(state)
                    else:
                        accumulation += state
                    # Use _local to preserve outer structure (stratified like B)
                    next_state = _collapse_inside_parentheses_local(state, simplify_fn=_simplify_variant_d)
                    if len(next_state) == 1:
                        state = next_state
                        break
                    state = next_state
                # Final global simplify for D (like B does)
                if len(state) > 1:
                    state = _collapse_global_ignore_parentheses(state, simplify_fn=_simplify_variant_d)

        elif variant == "E":
            # Ordered passes using stratified passes (like B, but with E's simplify rule)
            if used_hybrid:
                # After streaming, apply final global simplify with E's rule
                if len(state) > 1:
                    state = _collapse_global_ignore_parentheses(state, simplify_fn=_simplify_variant_e)
            else:
                while state != previous:
                    previous = state
                    if accumulation_manager:
                        accumulation_manager.append(state)
                    else:
                        accumulation += state
                    # Use _local to preserve outer structure (stratified like B)
                    next_state = _collapse_inside_parentheses_local(state, simplify_fn=_simplify_variant_e)
                    if len(next_state) == 1:
                        state = next_state
                        break
                    state = next_state
                # Final global simplify for E (like B does)
                if len(state) > 1:
                    state = _collapse_global_ignore_parentheses(state, simplify_fn=_simplify_variant_e)

        elif variant == "F":
            # Hybrid: fully stabilize inside→out, then one global pass
            if used_hybrid:
                # After streaming, apply final global simplify if needed
                if len(state) > 1:
                    state = _collapse_global_ignore_parentheses(state)
            else:
                while state != previous:
                    previous = state
                    if accumulation_manager:
                        accumulation_manager.append(state)
                    else:
                        accumulation += state
                    state = _collapse_inside_parentheses(state)
                    if len(state) == 1:
                        break
                if len(state) > 1:
                    state = _collapse_global_ignore_parentheses(state)

        else:
            # Fallback to baseline B
            if used_hybrid:
                if len(state) > 1:
                    state = _collapse_global_ignore_parentheses(state)
            else:
                while state != previous:
                    previous = state
                    if accumulation_manager:
                        accumulation_manager.append(state)
                    else:
                        accumulation += state
                    next_state = _collapse_inside_parentheses(state)
                    if len(next_state) == 1:
                        state = next_state
                        break
                    state = next_state

        # 4) L'estat després del decaïment serà el símbol final si és un sol caràcter
        #    o l'string simplificat si encara hi ha estructura
        current_state = state

        # MEMORY OPTIMIZED: Get length from AccumulationManager
        if accumulation_manager:
            current_length = accumulation_manager.get_clean_bits_count()
        else:
            current_length = _count_clean_bits(accumulation)  # OPTIMIZED: count without creating string

        # 5) Guardar snapshots si cal
        # v33: save structural (with parentheses), v32: save clean (only 0/1)
        # STREAMING OPTIMIZED: Use file-based save for large accumulations
        phi_to_save = None
        use_streaming_save = False

        if storage_format == "v33":
            if accumulation_manager:
                # Check if accumulation is too large for RAM
                accum_size = accumulation_manager.get_file_size()
                if accum_size > max_ram_bytes:
                    # Use streaming save - don't load to RAM
                    use_streaming_save = True
                    phi_to_save = None  # Will use file path instead
                    print(f"   [snapshot] Using streaming save ({accum_size/1e9:.2f} GB > {max_ram_gb} GB limit)", flush=True)
                else:
                    # Show progress for large reads
                    if accum_size > 100_000_000:  # >100MB
                        print(f"   [snapshot] Reading {accum_size/1e9:.2f} GB to RAM...", flush=True)
                    phi_to_save = accumulation_manager.read_all()
                    if accum_size > 100_000_000:
                        print(f"   [snapshot] Read complete.", flush=True)
            else:
                phi_to_save = accumulation  # ✅ KEEPS PARENTHESES
            phi_observable = None
            observable_length = current_length
        else:
            # v32: need to clean sequence (not streaming-compatible)
            if accumulation_manager:
                phi_to_save = _clean_sequence(accumulation_manager.read_all())
            else:
                phi_to_save = _clean_sequence(accumulation)
            phi_observable = phi_to_save
            observable_length = len(phi_to_save)

        if snapshot_manager and snapshot_manager.should_use_compression(observable_length):
            if not metadata["compression_activated"]:
                print(f"[compress] Auto-compression activated at iteration {iteration + 1} ({observable_length:,} bits)", flush=True)
                print(f"[compress] Storage format: {storage_format}", flush=True)
                print("[compress] Tuning: HSI_SNAPSHOT_STATS_EVERY, HSI_COMPRESSION_LEVEL, HSI_COMPRESS_LOG", flush=True)
                metadata["compression_activated"] = True

            # Save using appropriate format
            print(f"   [snapshot] Compressing iteration {iteration + 1}...", flush=True)
            t_snap = time.perf_counter()
            if storage_format == "v33":
                if use_streaming_save:
                    # STREAMING: Save directly from file without loading to RAM
                    save_info = snapshot_manager.save_phi_state_structural_from_file(
                        str(accumulation_manager.file_path),
                        iteration + 1,
                        {}
                    )
                else:
                    save_info = snapshot_manager.save_phi_state_structural(
                        phi_to_save,
                        iteration + 1,
                        {}
                    )
            else:
                save_info = snapshot_manager.save_phi_state(
                    phi_to_save,
                    iteration + 1,
                    {}
                )
            print(f"   [snapshot] Saved in {time.perf_counter()-t_snap:.1f}s", flush=True)
            metadata["snapshots_saved"].append(save_info)
        elif save_snapshots and not use_compression:
            # Only for v32 or small v33 (unlikely)
            if phi_observable is None:
                phi_observable = _clean_sequence(accumulation)
            snapshots.append(phi_observable)

        # Optional per-iteration logging (with flush for subprocess visibility)
        if LOG_EVERY and ((iteration + 1) % LOG_EVERY == 0 or iteration == 0):
            if ENABLE_TIMING and iter_t0 is not None:
                print(f"[iter {iteration+1:>3}] Φ length = {observable_length:,} bits | {time.perf_counter()-iter_t0:.2f}s | variant={variant}", flush=True)
            else:
                print(f"[iter {iteration+1:>3}] Φ length = {observable_length:,} bits | variant={variant}", flush=True)

        metadata["iterations_completed"] = iteration + 1
        entry = {"iteration": iteration + 1, "phi_length": observable_length}
        if ENABLE_TIMING and iter_t0 is not None:
            # round seconds to 2 decimals for readability in reports
            entry["time_sec"] = round(time.perf_counter() - iter_t0, 2)
        metadata.setdefault("per_iteration", []).append(entry)

    # Explicitly mark end of iteration loop
    print(f"\n[loop] All {max_iterations} iterations complete. Starting post-processing...", flush=True)

    # Starting post-processing (reports, streaming metrics, cleanup)
    print("[post] Processing...", flush=True)
    t_post0 = time.perf_counter()

    # Step 1
    print("[post] Step 1/4: computing final length and compression summary...")
    t1 = time.perf_counter()

    # Final length without re-cleaning huge strings if possible
    if metadata.get("per_iteration"):
        metadata["final_length"] = int(metadata["per_iteration"][-1].get("phi_length", 0))
    else:
        # MEMORY OPTIMIZED: Get length from AccumulationManager
        if accumulation_manager:
            metadata["final_length"] = accumulation_manager.get_clean_bits_count()
        else:
            metadata["final_length"] = _count_clean_bits(accumulation)  # OPTIMIZED

    # Afegir informació de compressió si s'ha usat
    if snapshot_manager:
        storage_summary = snapshot_manager.get_storage_summary()
        metadata["compression_summary"] = storage_summary
    print(f"[post] Step 1/4 done in {time.perf_counter()-t1:.2f}s")

    # Step 2
    print("[post] Step 2/4: saving results and computing streaming metrics...")
    t2 = time.perf_counter()
    # Guardar resultats si s'especifica directori
    if output_dir:
        if metadata.get("compression_activated"):
            phi_arg = None  # Data already saved in snapshots
        else:
            # Only create clean sequence if compression not activated (small data)
            # MEMORY OPTIMIZED: Read from AccumulationManager if available
            if accumulation_manager:
                phi_arg = _clean_sequence(accumulation_manager.read_all())
            else:
                phi_arg = _clean_sequence(accumulation)
        _save_results(phi_arg, snapshots, metadata, output_dir)
    print(f"[post] Step 2/4 done in {time.perf_counter()-t2:.2f}s")

    # Step 3
    print("[post] Step 3/4: optional cleanup of old snapshots...")
    t3 = time.perf_counter()
    keep_last = os.environ.get("HSI_CLEANUP_KEEP_LAST")
    if snapshot_manager and keep_last:
        try:
            kept = int(keep_last)
            removed = snapshot_manager.cleanup_old_snapshots(keep_last_n=kept)
            print(f"[cleanup] Removed {removed} old snapshots; kept last {kept}.")
        except Exception as e:
            print(f"[cleanup] Cleanup failed: {e}")
    print(f"[post] Step 3/4 done in {time.perf_counter()-t3:.2f}s")

    # Step 4
    total_post = time.perf_counter()-t_post0
    print("[post] Step 4/4: returning observable Φ and metadata.")
    print(f"[post] Total post-processing time: {total_post:.2f}s")

    # Return observable Φ (clean accumulation), not the final collapsed symbol
    # OPTIMIZED: If compression activated, return placeholder (data in snapshots)
    if metadata.get("compression_activated"):
        # Return placeholder - actual data is in snapshot files
        final_phi = f"<compressed: {metadata['final_length']:,} bits in {len(metadata['snapshots_saved'])} snapshots>"
    else:
        # Small data - return full clean sequence
        # MEMORY OPTIMIZED: Read from AccumulationManager if available
        if accumulation_manager:
            final_phi = _clean_sequence(accumulation_manager.read_all())
        else:
            final_phi = _clean_sequence(accumulation)

    # Cleanup AccumulationManager temporary file
    if accumulation_manager:
        accumulation_manager.cleanup()

    return final_phi, snapshots, metadata


def _get_builtin_simplify_function(rule: str) -> Callable[[str], str]:
    """Retorna la funció de simplificació corresponent a la regla especificada."""

    if rule == "AND":
        return _simplify_and
    elif rule == "ASYM":
        return _simplify_asym
    else:
        raise ValueError(f"Regla de col·lapse desconeguda: {rule}")


def _simplify_and(sequence: str) -> str:
    """
    Regla AND: Substitueix 01 i 10 per 0, reagrupa múltiples 0s i 1s.

    Aquesta regla reflecteix la tensió fonamental entre No-Res (0) i Absolut (1),
    on qualsevol combinació amb 0 condueix a l'aniquilació.
    """
    # Substituir patrons 01 i 10 per 0
    result = re.sub(r'01|10', '0', sequence)

    # Reagrupar múltiples 0s i 1s consecutius
    result = re.sub(r'0+', '0', result)
    result = re.sub(r'1+', '1', result)

    return result


def _simplify_asym(sequence: str) -> str:
    """
    Regla ASYM: 10 → 1, 01 → 0 (asimetria que pot afavorir l'expansió).
    """
    # Aplicar transformacions asimètriques
    result = re.sub(r'10', '1', sequence)
    result = re.sub(r'01', '0', result)

    # Reagrupar múltiples 0s i 1s consecutius
    result = re.sub(r'0+', '0', result)
    result = re.sub(r'1+', '1', result)

    return result


def _iterative_collapse(
    state: str,
    simplify_fn: Callable[[str], str],
    max_depth: int
) -> str:
    """
    Aplica el col·lapse iteratiu fins que l'estat es redueix a una forma estable.

    Args:
        state: Estat actual com a string binari
        simplify_fn: Funció de simplificació a aplicar
        max_depth: Nombre màxim d'iteracions de col·lapse

    Returns:
        Estat col·lapsat i estabilitzat
    """
    current = state

    for _ in range(max_depth):
        previous = current
        current = simplify_fn(current)

        # Si no hi ha canvis, hem arribat a un estat estable
        if current == previous:
            break

    return current


def _estimate_fractal_dimension(sequence: str, max_box_size: int = 50) -> float:
    """
    Estima la dimensió fractal de la seqüència binària utilitzant box-counting.

    Args:
        sequence: Seqüència binària com a string
        max_box_size: Mida màxima de caixa per al càlcul

    Returns:
        Estimació de la dimensió fractal
    """
    if len(sequence) < max_box_size:
        return 1.0

    # Convertir a array numpy per facilitar el processament
    binary_array = np.array([int(bit) for bit in sequence])

    box_sizes = []
    box_counts = []

    # Provar diferents mides de caixa
    for box_size in range(2, min(max_box_size, len(sequence) // 4)):
        # Comptar caixes que contenen almenys un 1
        num_boxes = len(sequence) // box_size
        count = 0

        for i in range(num_boxes):
            start_idx = i * box_size
            end_idx = start_idx + box_size
            box_content = binary_array[start_idx:end_idx]

            if np.any(box_content == 1):
                count += 1

        if count > 0:
            box_sizes.append(box_size)
            box_counts.append(count)

    if len(box_sizes) < 2:
        return 1.0

    # Linear fit on log-log (local, non-streaming)
    log_sizes = np.log(np.array(box_sizes, dtype=np.float64))
    log_counts = np.log(np.array(box_counts, dtype=np.float64))
    coeffs = np.polyfit(log_sizes, log_counts, 1)
    D = -coeffs[0]
    return float(max(0.0, min(2.0, D)))


def _estimate_fractal_dimension_streaming_gz(gz_path: str, max_box_size: int = 50) -> float:
    """Estimate fractal dimension by streaming a gz-compressed Φ string from disk.
    Optimized to avoid per-character Python loops (vectorized per-size processing).
    Supports both v32 (text) and v33 (structural bitarray) formats.
    """
    import gzip
    import math
    import json

    CHUNK = 1_000_000  # characters per read (tune if needed)

    # Detect format from metadata
    gz_dir = Path(gz_path).parent
    iteration = None
    try:
        # Extract iteration from filename (phi_iter{N}.bin.gz or phi_iter{N}.struct.gz)
        fname = Path(gz_path).stem  # removes .gz
        if fname.endswith('.bin'):
            fname = fname[:-4]  # remove .bin
        elif fname.endswith('.struct'):
            fname = fname[:-7]  # remove .struct
        iteration = int(fname.replace('phi_iter', ''))
    except Exception:
        pass

    is_v33 = False
    if iteration is not None:
        json_path = gz_dir / f"phi_iter{iteration}.json"
        if json_path.exists():
            try:
                with open(json_path, "r") as f:
                    meta = json.load(f)
                    is_v33 = meta.get("format") == "v33_structural"
            except Exception:
                pass

    # For v33, load and clean structure
    if is_v33:
        from .phi_snapshot_manager import PhiSnapshotManager
        manager = PhiSnapshotManager(data_dir=str(gz_dir))
        phi_structural, _ = manager.load_phi_state_structural(iteration)
        # Clean: remove parentheses for fractal analysis
        phi_clean = phi_structural.replace('(', '').replace(')', '')
        total_len = len(phi_clean)

        if total_len < 4:
            return 1.0

        # Process as single string (already in memory)
        max_s = min(max_box_size, max(2, total_len // 4))
        sizes = list(range(2, max_s))
        if not sizes:
            return 1.0

        # Box counting on clean string
        box_counts = {s: 0 for s in sizes}
        for s in sizes:
            for i in range(0, total_len, s):
                box = phi_clean[i:i+s]
                if '1' in box:
                    box_counts[s] += 1

        # Linear regression
        log_sizes = [math.log(s) for s in sizes]
        log_counts = [math.log(max(1, box_counts[s])) for s in sizes]
        n = len(sizes)
        sum_x = sum(log_sizes)
        sum_y = sum(log_counts)
        sum_xy = sum(x * y for x, y in zip(log_sizes, log_counts))
        sum_x2 = sum(x * x for x in log_sizes)
        denom = n * sum_x2 - sum_x * sum_x
        if abs(denom) < 1e-12:
            return 1.0
        slope = (n * sum_xy - sum_x * sum_y) / denom
        D = -slope
        return float(max(0.0, min(2.0, D)))

    # v32 format: stream from gzip text file
    # First pass: determine total length
    total_len = 0
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        for chunk in iter(lambda: f.read(CHUNK), ""):
            total_len += len(chunk)
    if total_len < 4:
        return 1.0

    # Choose box sizes
    max_s = min(max_box_size, max(2, total_len // 4))
    sizes = list(range(2, max_s))
    if not sizes:
        return 1.0

    # State per size: position within current box and whether we've seen a '1'
    state_pos = {s: 0 for s in sizes}
    state_has = {s: False for s in sizes}
    box_counts = {s: 0 for s in sizes}

    # Second pass: process chunks as numpy arrays
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        for chunk in iter(lambda: f.read(CHUNK), ""):
            if not chunk:
                break
            # Convert '0'/'1' chars to boolean array efficiently
            arr = np.frombuffer(chunk.encode('ascii'), dtype=np.uint8)
            bits = (arr == 49)  # ord('1') == 49
            n = bits.size
            if n == 0:
                continue
            for s in sizes:
                pos = state_pos[s]
                has1 = state_has[s]
                # Finish current partial box if needed
                if pos:
                    take = min(n, s - pos)
                    if take > 0:
                        if bits[:take].any():
                            has1 = True
                        pos += take
                        if pos == s:
                            box_counts[s] += 1 if has1 else 0
                            pos = 0
                            has1 = False
                    start = take
                else:
                    start = 0
                # Process full boxes via reshape
                rem = n - start
                if rem >= s:
                    full = (rem // s) * s
                    if full:
                        view = bits[start:start+full].reshape((-1, s))
                        anys = view.any(axis=1)
                        box_counts[s] += int(anys.sum())
                    start += full
                # Tail remainder becomes new partial box
                if start < n:
                    tail = bits[start:]
                    has1 = bool(tail.any())
                    pos = tail.size
                state_pos[s] = pos
                state_has[s] = has1

    # Close trailing partial boxes
    for s in sizes:
        if state_pos[s] and state_has[s]:
            box_counts[s] += 1

    # Build log-log arrays
    xs, ys = [], []
    for s in sizes:
        c = box_counts[s]
        if c > 0:
            xs.append(math.log(s))
            ys.append(math.log(c))
    if len(xs) < 2:
        return 1.0

    # Linear fit: log(count) = -D * log(size) + const
    coeffs = np.polyfit(xs, ys, 1)
    D = -coeffs[0]
    return float(max(0.0, min(2.0, D)))



def _attach_fractal_plot_data_to_report(report_obj: Dict[str, Any], per_size_counts: Dict[int, int]):
    # Store raw arrays for reproducibility
    sizes = sorted(per_size_counts.keys())
    counts = [per_size_counts[s] for s in sizes]
    report_obj.setdefault("fractal_plot_data", {})["sizes"] = sizes
    report_obj["fractal_plot_data"]["counts"] = counts


def _save_results(
    phi: str,
    snapshots: List[str],
    metadata: Dict[str, Any],
    output_dir: str
) -> None:
    """Guarda els resultats de la simulació en fitxers."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Guardar seqüència final com a numpy array (només si la tenim en memòria)
    if phi is not None:
        try:
            binary_array = np.array([int(bit) for bit in phi], dtype=np.uint8)
            np.save(output_path / "phi_final.npy", binary_array)
        except Exception as _e:
            print(f"[post] Skipping phi_final.npy save (in-memory Φ not available or too large): {_e}")

    # Guardar snapshots
    if snapshots:
        t_snap0 = time.perf_counter()
        snapshots_data = {
            f"iteration_{i}": snapshot
            for i, snapshot in enumerate(snapshots)
        }
        with open(output_path / "phi_snapshots.json", 'w') as f:
            json.dump(snapshots_data, f, indent=2)
        print(f"[post] 2.1/4 saved phi_snapshots.json (entries={len(snapshots_data)}) in {time.perf_counter()-t_snap0:.2f}s")
    else:
        print("[post] 2.1/4 no in-memory snapshots to save (streaming mode)")

    # Save per-variant JSON result under project reports/
    reports_dir = REPORTS_DIR
    import json as _json
    stamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
    iters_done = metadata.get("iterations_completed", 0)
    variant_code = metadata.get("variant", "?")
    # include small per-iteration metrics for growth curve
    per_iter = metadata.get("per_iteration", [])

    # Prefer streaming metrics for huge runs; only compute direct D if phi is small
    if phi is not None and len(phi) > 100 and len(phi) <= 5_000_000:
        fractal_dim_final = _estimate_fractal_dimension(phi)
    else:
        fractal_dim_final = None
    phi_align_final = abs(fractal_dim_final - 1.618) if fractal_dim_final is not None else None

    total_bits_val = int(metadata.get("final_length", len(phi) if phi is not None else 0))

    result_obj = {
        "variant": variant_code,
        "iterations": iters_done,
        "total_bits": total_bits_val,
        "fractal_dimension": fractal_dim_final,
        "phi_alignment": phi_align_final,
        "per_iteration": per_iter,
        "timestamp": __import__('datetime').datetime.now().isoformat()
    }

    # If we computed streaming fractal metrics later, mirror them into the main fields
    # so enriched and raw reports carry non-null values.

    # Attach fractal plot data if available (automatic for big runs)
    try:
        snaps = metadata.get("snapshots_saved", [])
        latest = max(snaps, key=lambda s: s.get("iteration", 0)) if snaps else None
        if latest and latest.get("compressed"):
            from pathlib import Path as _P
            data_dir = latest.get("data_directory") if latest.get("data_directory") else str(SNAPSHOT_DATA_DIR)

            # Detect format and use appropriate file
            is_v33 = latest.get("format") == "v33_structural"
            if is_v33:
                gz_path = _P(data_dir) / f"phi_iter{latest['iteration']}.struct.gz"
            else:
                gz_path = _P(data_dir) / f"phi_iter{latest['iteration']}.bin.gz"

            if gz_path.exists():
                try:
                    rel_gz = gz_path.resolve().relative_to(BASE_PATH)
                except Exception:
                    rel_gz = gz_path
                print(f"[post] Computing streaming fractal from: {rel_gz}")
                _t_stream = time.perf_counter()
                D_stream = _estimate_fractal_dimension_streaming_gz(str(gz_path), max_box_size=64)
                _t_stream_dur = time.perf_counter() - _t_stream
                result_obj["fractal_dimension_streaming"] = D_stream
                result_obj["phi_alignment_streaming"] = abs(D_stream - 1.618)
                # Mirror into primary fields if they are None
                if result_obj.get("fractal_dimension") is None:
                    result_obj["fractal_dimension"] = float(D_stream)
                    result_obj["phi_alignment"] = float(abs(D_stream - 1.618))
                print(f"[post] Streaming fractal computed: D_stream={D_stream:.4f} (time={_t_stream_dur:.2f}s)")
                try:
                    print("[post] Persisting fractal_plot_data arrays (sizes, counts)...")
                    _t_persist = time.perf_counter()
                    import math, gzip
                    sizes = list(range(2, 65))
                    per_counts = {}
                    # Progress setup
                    total_bits_est = int((latest.get("sequence_length") or 0))
                    next_mark = 0.10 if total_bits_est > 0 else None
                    t0p = time.perf_counter()

                    # Load phi_clean (handle both v32 and v33)
                    if is_v33:
                        from .phi_snapshot_manager import PhiSnapshotManager
                        manager = PhiSnapshotManager(data_dir=data_dir)
                        phi_structural, _ = manager.load_phi_state_structural(latest['iteration'])
                        phi_clean = phi_structural.replace('(', '').replace(')', '')

                        # Process as single string
                        idx = 0
                        ones = {s: 0 for s in sizes}
                        for ch in phi_clean:
                            if ch == '1':
                                for s in sizes: ones[s] += 1
                            idx += 1
                            for s in sizes:
                                if idx % s == 0:
                                    per_counts[s] = per_counts.get(s, 0) + (1 if ones[s] > 0 else 0)
                                    ones[s] = 0

                            # Progress print every 10% (if we know total)
                            if next_mark is not None and total_bits_est > 0 and idx % 1_000_000 == 0:
                                ratio = idx / total_bits_est
                                if ratio >= next_mark:
                                    elapsed = time.perf_counter() - t0p
                                    eta = (elapsed/ratio - elapsed) if ratio > 0 else 0.0
                                    print(f"[post] fractal_plot_data {ratio*100:5.1f}%  ETA ~{eta:.1f}s", end="\r")
                                    next_mark += 0.10
                        for s in sizes:
                            if idx % s != 0 and ones[s] > 0:
                                per_counts[s] = per_counts.get(s, 0) + 1
                    else:
                        # v32: stream from gzip
                        with gzip.open(str(gz_path), "rt", encoding="utf-8") as _f:
                            idx = 0
                            ones = {s: 0 for s in sizes}
                            for chunk in iter(lambda: _f.read(1_000_000), ""):
                                for ch in chunk:
                                    if ch == '1':
                                        for s in sizes: ones[s] += 1
                                    idx += 1
                                    for s in sizes:
                                        if idx % s == 0:
                                            per_counts[s] = per_counts.get(s, 0) + (1 if ones[s] > 0 else 0)
                                            ones[s] = 0

                                # Progress print every 10% (if we know total)
                                if next_mark is not None and total_bits_est > 0:
                                    ratio = idx / total_bits_est
                                    if ratio >= next_mark:
                                        elapsed = time.perf_counter() - t0p
                                        eta = (elapsed/ratio - elapsed) if ratio > 0 else 0.0
                                        print(f"[post] fractal_plot_data {ratio*100:5.1f}%  ETA ~{eta:.1f}s", end="\r")
                                        next_mark += 0.10
                            for s in sizes:
                                if idx % s != 0 and ones[s] > 0:
                                    per_counts[s] = per_counts.get(s, 0) + 1
                    if next_mark is not None:
                        print("[post] fractal_plot_data 100.0%            ")
                    _attach_fractal_plot_data_to_report(result_obj, per_counts)
                    print(f"[post] fractal_plot_data persisted (time={time.perf_counter()-_t_persist:.2f}s; sizes={len(sizes)})")
                except Exception as e:
                    result_obj["fractal_plot_persist_error"] = str(e)
    except Exception as e:
        result_obj["fractal_streaming_error"] = str(e)
    # 2.4/4 — Attach short Φ prefix for diagnostics (works even in streaming mode)
    try:
        pp_len = int(os.environ.get("HSI_REPORT_PREFIX", "256"))
    except Exception:
        pp_len = 256
    phi_prefix = None
    if phi is not None and len(phi) > 0:
        phi_prefix = phi[:pp_len]
    else:
        try:
            import gzip as _gz
            snaps2 = metadata.get("snapshots_saved", [])
            latest2 = max(snaps2, key=lambda s: s.get("iteration", 0)) if snaps2 else None
            if latest2:
                data_dir2 = latest2.get("data_directory") if latest2.get("data_directory") else str(SNAPSHOT_DATA_DIR)
                is_v33_2 = latest2.get("format") == "v33_structural"

                if latest2.get("compressed"):
                    if is_v33_2:
                        # v33: load structural and clean
                        from .phi_snapshot_manager import PhiSnapshotManager
                        manager = PhiSnapshotManager(data_dir=data_dir2)
                        phi_structural, _ = manager.load_phi_state_structural(latest2['iteration'])
                        phi_clean = phi_structural.replace('(', '').replace(')', '')
                        phi_prefix = phi_clean[:pp_len]
                    else:
                        # v32: read text
                        p = Path(data_dir2) / f"phi_iter{latest2['iteration']}.bin.gz"
                        if p.exists():
                            with _gz.open(str(p), "rt", encoding="utf-8") as _f:
                                phi_prefix = _f.read(pp_len)
                else:
                    p = Path(data_dir2) / f"phi_iter{latest2['iteration']}.bin"
                    if p.exists():
                        with open(str(p), "rt", encoding="utf-8") as _f:
                            phi_prefix = _f.read(pp_len)
        except Exception:
            pass
    if phi_prefix:
        result_obj["phi_prefix"] = phi_prefix
        result_obj["phi_prefix_len"] = len(phi_prefix)
    print("[post] 2.4/4 enrich report with phi_prefix (if available)")


    # Write per-variant report including streaming metrics
    out_path = reports_dir / f"variant_{variant_code}_{iters_done}_{stamp}.json"
    metadata["report_path"] = str(out_path)
    rel = out_path.relative_to(BASE_PATH)
    print(f"[post] 2.2/4 writing variant report: {rel}")
    t0w = time.perf_counter()
    with open(out_path, "w") as f:
        _json.dump(result_obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())  # Force write to disk for Windows compatibility
    print(f"[INFO] 2.2/4 report saved to {rel}  (write={time.perf_counter()-t0w:.2f}s)")

    # Save metadata and (optionally) final text sequence
    t_meta0 = time.perf_counter()
    with open(output_path / "phi_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
        f.flush()
        os.fsync(f.fileno())  # Force write to disk for Windows compatibility
    print(f"[post] 2.3/4 saved phi_metadata.json in {time.perf_counter()-t_meta0:.2f}s")
    if phi is not None:
        try:
            with open(output_path / "phi_final.txt", 'w') as f:
                f.write(phi)
        except Exception as _e:
            print(f"[post] Skipping phi_final.txt save: {_e}")
        # Also save per-variant final text under results/var_{VAR}/ for clear isolation
        try:
            var_code = (metadata.get("variant") or "?").upper()
            var_dir = output_path / f"var_{var_code}"
            var_dir.mkdir(parents=True, exist_ok=True)

            with open(var_dir / "phi_final.txt", 'w') as f:
                f.write(phi)
        except Exception as _e:
            print(f"[post] Skipping var-specific phi_final.txt save: {_e}")
    else:
        print("[post] Skipping phi_final.txt save: Φ not available in memory (using streaming)")


# Funcions auxiliars per a compatibilitat amb altres mòduls

def get_phi_array(phi_string: str) -> np.ndarray:
    """Converteix una seqüència Φ string a array numpy. VECTORIZED."""
    return (np.frombuffer(phi_string.encode('ascii'), dtype='S1') == b'1').astype(np.uint8)


def _hilbert_indices_to_xy_vectorized(n_points: int, grid_size: int) -> np.ndarray:
    """
    VECTORIZED Hilbert curve index to (x, y) conversion.
    Computes all coordinates in parallel using NumPy operations.
    ~100x faster than calling _hilbert_index_to_xy in a loop.
    """
    indices = np.arange(n_points, dtype=np.int64)
    x = np.zeros(n_points, dtype=np.int32)
    y = np.zeros(n_points, dtype=np.int32)
    t = indices.copy()
    s = 1

    while s < grid_size:
        rx = (t >> 1) & 1  # 1 & (t // 2)
        ry = (t ^ rx) & 1  # 1 & (t ^ rx)

        # Where ry == 0, apply rotation/flip
        mask_ry0 = (ry == 0)
        mask_rx1 = (rx == 1)
        mask_flip = mask_ry0 & mask_rx1

        # Apply flip: x = s - 1 - x, y = s - 1 - y
        x[mask_flip] = s - 1 - x[mask_flip]
        y[mask_flip] = s - 1 - y[mask_flip]

        # Apply swap for all ry == 0: x, y = y, x
        x_temp = x[mask_ry0].copy()
        x[mask_ry0] = y[mask_ry0]
        y[mask_ry0] = x_temp

        # Translate
        x += s * rx
        y += s * ry

        t >>= 2  # t //= 4
        s *= 2

    return np.column_stack([x, y])


def phi_to_hilbert_coords(phi_array: np.ndarray, grid_size: int = None) -> np.ndarray:
    """
    Converteix una seqüència Φ a coordenades en una corba de Hilbert.
    VECTORIZED version — ~100x faster than loop-based approach.

    Args:
        phi_array: Array binari de la seqüència Φ
        grid_size: Mida de la graella (si None, s'estima automàticament)

    Returns:
        Array de coordenades (x, y) per a cada bit
    """
    n_points = len(phi_array)
    if grid_size is None:
        # Estimar mida de graella basada en la longitud
        grid_size = int(np.ceil(np.sqrt(n_points)))
        # Assegurar que sigui una potència de 2 per a la corba de Hilbert
        grid_size = 2 ** int(np.ceil(np.log2(grid_size)))

    # Limit to grid capacity
    n_points = min(n_points, grid_size * grid_size)

    # VECTORIZED: compute all Hilbert coordinates at once
    return _hilbert_indices_to_xy_vectorized(n_points, grid_size)


def _hilbert_index_to_xy(index: int, n: int) -> Tuple[int, int]:
    """Converteix un índex lineal a coordenades (x, y) en una corba de Hilbert."""
    # Implementació simplificada de la corba de Hilbert
    # Per a una implementació completa, es podria usar una llibreria especialitzada

    x = y = 0
    t = index
    s = 1

    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)

        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x


        x += s * rx
        y += s * ry
        t //= 4
        s *= 2

    return x, y



if __name__ == "__main__":
    import argparse, os, json
    print("HSI Level 0 Generator — Basal‑Pure Dynamics")
    print("=" * 60)

    # Load config.json
    config_path = BASE_PATH / "config.json"
    config = {}
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

    parser = argparse.ArgumentParser(description="HSI Level 0 Generator (Basal‑Pure)")
    parser.add_argument("--iterations", "-i", type=int, default=None,
                        help="Number of iterations (default: env HSI_ITERATIONS or 18)")
    parser.add_argument("--variant", "-v", type=str, default=None,
                        help="Variant code A–H (default: env HSI_VARIANT_CODE or B)")
    parser.add_argument("--log-every", type=int, default=None,
                        help="Print a progress line every N iterations (overrides env HSI_LOG_EVERY)")
    parser.add_argument("--compression-level", type=int, default=None,
                        help="Override compression level (1-9) for gzip; defaults to env HSI_COMPRESSION_LEVEL or 9")
    parser.add_argument("--abs-mode", type=str, default=None,
                        help="Absolute token mode: 1 | 10 | 01 (overrides config/env)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Disable checkpoint recovery (start from iteration 0)")

    # Plot controls (delegate to runner)
    parser.add_argument("--generate-plots", dest="generate_plots", action="store_true", default=True,
                        help="Generate plots at the end (via runner in plot-only mode)")
    parser.add_argument("--no-generate-plots", dest="generate_plots", action="store_false",
                        help="Disable plot generation at the end")
    parser.add_argument("--hilbert-prefix", type=int, default=int(os.environ.get("HSI_HILBERT_PREFIX", "1000000")),
                        help="Max bits for Hilbert map (only if plots enabled)")
    parser.add_argument("--fft-prefix", type=int, default=int(os.environ.get("HSI_FFT_PREFIX", "2000000")),
                        help="Max bits for FFT (only if plots enabled)")
    parser.add_argument("--no-hilbert", action="store_true", help="Skip Hilbert heatmap generation")
    parser.add_argument("--no-fft", action="store_true", help="Skip FFT/DFT chart generation")

    args = parser.parse_args()

    iterations = args.iterations or int(os.environ.get("HSI_ITERATIONS", "18"))
    variant = (args.variant or os.environ.get("HSI_VARIANT_CODE", "B")).upper()

    if args.log_every is not None:
        os.environ["HSI_LOG_EVERY"] = str(args.log_every)
    if args.compression_level is not None:
        os.environ["HSI_COMPRESSION_LEVEL"] = str(args.compression_level)

    # Override ABS mode if provided
    if args.abs_mode is not None:
        if args.abs_mode in ("1","10","01"):
            os.environ["HSI_ABSOLUTE_TOKEN"] = args.abs_mode
        else:
            print(f"[WARN] --abs-mode invalid: {args.abs_mode}; using default/env")

    print(f"Config: variant={variant}, iterations={iterations}, ABS={os.environ.get('HSI_ABSOLUTE_TOKEN','1')}")

    # Checkpoint recovery (enabled by default, disable with --no-resume)
    resume_checkpoint = not args.no_resume

    phi, snapshots, metadata = simulate_phi(
        max_iterations=iterations,
        save_snapshots=True,
        output_dir=str(RESULTS_DIR),
        use_compression=True,
        memory_threshold=10**7,
        variant=variant,
        resume_from_checkpoint=resume_checkpoint,
        config=config
    )

    print("\n" + "=" * 60, flush=True)
    print("✅ GENERATION COMPLETE", flush=True)
    print("=" * 60, flush=True)
    print(f"   Iterations completed: {metadata['iterations_completed']}")
    print(f"   Final Φ length: {metadata['final_length']:,} bits")
    print(f"   Compression activated: {metadata['compression_activated']}")
    print(f"   First 100 bits of Φ: {phi[:100]}...")

    if len(phi) > 100:
        # Avoid expensive direct computation on huge sequences; use streaming if available
        if len(phi) <= 5_000_000:
            print("[post] Computing fractal dimension...", flush=True)
            fractal_dim = _estimate_fractal_dimension(phi)
            if fractal_dim is not None:
                try:
                    print(f"   Estimated fractal dimension: {float(fractal_dim):.4f}")
                    print(f"   Distance to φ (1.618): {abs(float(fractal_dim) - 1.618):.4f}")
                except Exception:
                    print(f"   Estimated fractal dimension: {fractal_dim}")
                    try:
                        print(f"   Distance to φ (1.618): {abs(float(fractal_dim) - 1.618)}")
                    except Exception:
                        pass
            else:
                print("   Estimated fractal dimension: unavailable")
        else:
            fd_stream = metadata.get('fractal_dimension_streaming')
            if fd_stream is not None:
                print(f"   Estimated fractal dimension (streaming): {float(fd_stream):.4f}")
                try:
                    print(f"   Distance to φ (1.618): {abs(float(fd_stream) - 1.618):.4f}")
                except Exception:
                    pass
            else:
                print("   Estimated fractal dimension: skipped (Φ too large); see streaming D in report")

    # Optionally delegate plot generation to the runner in plot-only mode
    if args.generate_plots:
        try:
            import subprocess, sys

            # Small delay to ensure all files are flushed to disk (Windows FS cache)
            import time as _time
            _time.sleep(0.5)

            env = dict(**os.environ)
            env["HSI_PLOT_ONLY"] = "1"
            env["HSI_ITERATIONS"] = str(iterations)
            env["HSI_VARIANT_CODE"] = variant
            env["HSI_REPORT_ITER"] = str(iterations)  # pass target iteration
            # Pass exact report path if available
            try:
                rp = metadata.get("report_path")
                if rp:
                    env["HSI_REPORT_PATH"] = str(Path(rp).resolve())
            except Exception:
                pass
            if args.no_fft:
                env["HSI_NO_FFT"] = "1"
            if args.no_hilbert:
                env["HSI_NO_HILBERT"] = "1"
            env["HSI_HILBERT_PREFIX"] = str(args.hilbert_prefix)
            env["HSI_FFT_PREFIX"] = str(args.fft_prefix)
            # Use absolute path for robustness (avoids import issues with -m)
            runner_path = (BASE_PATH / "run_all_variants.py").resolve()
            cmd = [sys.executable, str(runner_path)]
            print("[post] Launching runner for plot generation (plot-only mode)...", flush=True)
            print(f"[post] Runner command: {' '.join(cmd)}", flush=True)
            subprocess.run(cmd, env=env, check=False, cwd=str(BASE_PATH))
        except Exception as e:
            print(f"[post] Runner plot generation failed: {e}")

    comp_summary = metadata.get('compression_summary')
    if comp_summary:
        print("\n[Compression Summary]")
        print(f"   Available iterations: {comp_summary.get('total_iterations', 0)}")
        print(f"   Iterations range: {comp_summary.get('iterations_range', 'None')}")
        print(f"   Total size: {comp_summary.get('total_size_mb', 0):.2f} MB")
        print(f"   Compressed: {comp_summary.get('compressed_snapshots', 0)}")
        print(f"   Uncompressed: {comp_summary.get('uncompressed_snapshots', 0)}")
