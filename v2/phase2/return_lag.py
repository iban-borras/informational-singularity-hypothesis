from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from utils.progress import ProgressIndicator
from v2.common.io import load_observable_prefix_bits, split_into_segments
from v2.common.null_models import (
    LZ_CALIBRATION_WINDOW_BITS,
    LZ_MAX_WINDOWS,
    MARKOV1_PROGRESS_CHUNK_BITS,
    MATCHED_LZ_BLOCK_BITS,
    PHASE_MATCHED_LZ_PERIOD_BLOCKS,
    generate_control,
)
from v2.phase1.tower import rolling_codes_uint64_max64

from .defects import _require_edge, classify_variant_role, load_policy_fibers

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


SOURCE_CACHE_VERSION = "hsi-v2-source-cache-v1"
SOURCE_CACHE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "hsi_v2"
    / "source_cache"
    / "frozen_sources"
)
SOURCE_CACHE_LOCK_STALE_SECONDS = 6 * 60 * 60


def load_pattern_space(run_dir: Path, pattern_scale: int) -> dict:
    path = run_dir / "pattern_spaces" / f"pattern_space_m{pattern_scale}.json"
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def prepare_pattern_selection(
    run: dict,
    *,
    pattern_scale: int,
    top_patterns: int,
    pattern_selection: str,
) -> dict:
    config = run["dataset"]["config"]
    run_dir = Path(run["_run_dir"])
    pattern_space = load_pattern_space(run_dir, pattern_scale)
    prefix_fibers = None
    suffix_fibers = None
    if pattern_selection == "bridge-linked":
        prefix_fibers = load_policy_fibers(run_dir, "prefix")
        suffix_fibers = load_policy_fibers(run_dir, "suffix")
    selected_patterns = _select_patterns(
        pattern_space,
        top_patterns=top_patterns,
        selection_mode=pattern_selection,
        phase1_scales=list(config.get("scales", [])),
        pattern_scale=pattern_scale,
        prefix_fibers=prefix_fibers,
        suffix_fibers=suffix_fibers,
    )
    return {
        "selected_patterns": [dict(item) for item in selected_patterns],
        "pattern_selection": pattern_selection,
        "source_variant": str(config["variant"]),
        "source_run_dir": str(run_dir),
        "anchor_kind": "run-local",
    }


def build_return_lag_rows(
    runs: list[dict],
    *,
    pattern_scale: int,
    top_patterns: int,
    pattern_selection: str,
    long_lag_threshold: int,
    selection_overrides: dict[str, dict] | None = None,
    preloaded_sources: dict[str, dict] | None = None,
    show_progress: bool = False,
) -> list[dict]:
    rows = []
    normalized_overrides = {
        str(key).upper(): value
        for key, value in (selection_overrides or {}).items()
    }
    for run in runs:
        config = run["dataset"]["config"]
        run_dir = Path(run["_run_dir"])
        selection_info = _resolve_selection_override(config, normalized_overrides)
        if selection_info is None:
            selection_info = prepare_pattern_selection(
                run,
                pattern_scale=pattern_scale,
                top_patterns=top_patterns,
                pattern_selection=pattern_selection,
            )
        selected_patterns = [dict(item) for item in selection_info["selected_patterns"]]
        preloaded_source = None
        if preloaded_sources is not None:
            preloaded_source = preloaded_sources.get(str(run_dir))
        segments, source_info = load_run_segments(
            run["dataset"],
            preloaded_source=preloaded_source,
            show_progress=show_progress,
            progress_label=config["variant"],
        )
        lag_result = compute_return_lag_result(
            segments,
            selected_patterns=selected_patterns,
            pattern_scale=pattern_scale,
            long_lag_threshold=long_lag_threshold,
            show_progress=show_progress,
            progress_label=config["variant"],
        )
        row = {
            "variant": config["variant"],
            "variant_role": classify_variant_role(config["variant"]),
            "iteration": config.get("iteration"),
            "segment_bits": config.get("segment_bits"),
            "num_segments": config.get("num_segments"),
            "segment_offset_bits": config.get("segment_offset_bits", 0),
            "scales": list(config.get("scales", [])),
            "phase1_policies": list(config.get("policies", [])),
            "pattern_scale": pattern_scale,
            "top_patterns": top_patterns,
            "pattern_selection": str(selection_info.get("pattern_selection", pattern_selection)),
            "selection_source_variant": str(selection_info.get("source_variant", config["variant"])),
            "selection_source_run_dir": str(selection_info.get("source_run_dir", run_dir)),
            "selection_anchor_kind": str(selection_info.get("anchor_kind", "run-local")),
            "long_lag_threshold": long_lag_threshold,
            "run_dir": str(run_dir),
            "dataset_path": run["_dataset_path"],
            "summary_path": run["_summary_path"],
            "generated_at": run["generated_at"],
            "source_kind": source_info["source_kind"],
            "source_struct_path": source_info["source_struct_path"],
            "source_freeze_mode": str(source_info.get("source_freeze_mode", "run-local")),
            "frozen_total_bits": source_info.get("frozen_total_bits"),
            "null_model": source_info.get("null_model"),
            "selected_patterns": selected_patterns,
            "lag_result": lag_result,
            "selected_pattern_count": lag_result["selected_pattern_count"],
            "total_occurrence_count": lag_result["total_occurrence_count"],
            "total_return_count": lag_result["total_return_count"],
            "mean_return_lag": lag_result["mean_return_lag"],
            "median_return_lag": lag_result["median_return_lag"],
            "lag_entropy": lag_result["lag_entropy"],
            "dominant_bin_mass_fraction": lag_result["dominant_bin_mass_fraction"],
            "long_lag_fraction": lag_result["long_lag_fraction"],
            "segment_profile_jaccard": lag_result["segment_profile_jaccard"],
            "segment_profile_weighted_jaccard": lag_result["segment_profile_weighted_jaccard"],
        }
        rows.append(row)
    return rows


def _resolve_selection_override(
    config: dict,
    normalized_overrides: dict[str, dict],
) -> dict | None:
    variant_key = str(config.get("variant", "")).upper()
    if variant_key and variant_key in normalized_overrides:
        return normalized_overrides[variant_key]

    source_variant_key = str(config.get("source_variant", "")).upper()
    if source_variant_key and source_variant_key in normalized_overrides:
        return normalized_overrides[source_variant_key]

    if "-" in variant_key:
        base_variant_key = variant_key.split("-", 1)[0]
        if base_variant_key in normalized_overrides:
            return normalized_overrides[base_variant_key]

    return None


def load_run_segments(
    dataset: dict,
    *,
    preloaded_source: dict | None = None,
    show_progress: bool = False,
    progress_label: str | None = None,
) -> tuple[list[np.ndarray], dict]:
    config = dataset["config"]
    segment_bits = int(config["segment_bits"])
    num_segments = int(config["num_segments"])
    segment_offset_bits = int(config.get("segment_offset_bits", 0))
    total_required = int(
        config.get("loaded_observable_bits")
        or (segment_offset_bits + (segment_bits * num_segments))
    )
    if preloaded_source is not None:
        source_bits = preloaded_source["bits"]
        if int(source_bits.size) < total_required:
            raise ValueError(
                f"Frozen source has {source_bits.size} bits but {total_required} are required for the requested window."
            )
        info = dict(preloaded_source.get("source_info", {}))
    else:
        source_bits, info = load_run_source_bits(
            dataset,
            required_bits=total_required,
            show_progress=show_progress,
            progress_label=progress_label,
        )

    segments = split_into_segments(
        source_bits,
        segment_bits=segment_bits,
        num_segments=num_segments,
        start_offset_bits=segment_offset_bits,
    )
    return segments, info


def load_run_source_bits(
    dataset: dict,
    *,
    required_bits: int | None = None,
    show_progress: bool = False,
    progress_label: str | None = None,
) -> tuple[np.ndarray, dict]:
    config = dataset["config"]
    struct_path = Path(config["input_struct_path"]).resolve()
    segment_bits = int(config["segment_bits"])
    num_segments = int(config["num_segments"])
    segment_offset_bits = int(config.get("segment_offset_bits", 0))
    effective_required_bits = int(
        required_bits
        if required_bits is not None
        else (
            config.get("loaded_observable_bits")
            or (segment_offset_bits + (segment_bits * num_segments))
        )
    )

    if config.get("sequence_kind") == "null_surrogate":
        null_name = str(config["null_model"])
        null_seed = int(config["null_seed"])
        cache_identity = _source_cache_identity(
            config=config,
            struct_path=struct_path,
            required_bits=effective_required_bits,
            null_name=null_name,
            null_seed=null_seed,
        )
        cached = _load_cached_null_source(
            cache_identity,
            required_bits=effective_required_bits,
            show_progress=show_progress,
            progress_label=progress_label,
        )
        if cached is not None:
            return cached

        lock_path = _source_cache_root() / f"{cache_identity['identity_hash']}.lock"
        with _source_cache_lock(lock_path):
            cached = _load_cached_null_source(
                cache_identity,
                required_bits=effective_required_bits,
                show_progress=show_progress,
                progress_label=progress_label,
            )
            if cached is not None:
                return cached

            source_bits = _load_observed_source_bits(
                struct_path,
                effective_required_bits=effective_required_bits,
                show_progress=show_progress,
                progress_label=progress_label,
            )
            source_bits, null_metadata = generate_control(
                source_bits,
                null_name,
                null_seed,
                show_progress=show_progress,
                progress_label=progress_label,
            )
            cache_info = _store_cached_null_source(
                cache_identity,
                source_bits,
                null_metadata=null_metadata,
                show_progress=show_progress,
                progress_label=progress_label,
            )
            info = {
                "source_kind": "null_surrogate",
                "source_struct_path": str(struct_path),
                "loaded_observable_bits": effective_required_bits,
                "source_freeze_mode": "run-local",
                "null_model": null_name,
                "null_seed": null_seed,
                "null_metadata": null_metadata,
                **cache_info,
            }
            return source_bits, info

    source_bits = _load_observed_source_bits(
        struct_path,
        effective_required_bits=effective_required_bits,
        show_progress=show_progress,
        progress_label=progress_label,
    )

    info = {
        "source_kind": "observed",
        "source_struct_path": str(struct_path),
        "loaded_observable_bits": effective_required_bits,
        "source_freeze_mode": "run-local",
    }
    return source_bits, info


def _load_observed_source_bits(
    struct_path: Path,
    *,
    effective_required_bits: int,
    show_progress: bool,
    progress_label: str | None,
) -> np.ndarray:
    progress_name = f"Load observable bits ({progress_label})" if progress_label else "Load observable bits"
    if show_progress:
        with ProgressIndicator(progress_name, total=effective_required_bits) as progress:
            return load_observable_prefix_bits(
                struct_path,
                effective_required_bits,
                progress=progress,
            )
    return load_observable_prefix_bits(struct_path, effective_required_bits)


def _source_cache_root() -> Path:
    override = os.environ.get("HSI_V2_SOURCE_CACHE_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return SOURCE_CACHE_ROOT


def _source_cache_identity(
    *,
    config: dict,
    struct_path: Path,
    required_bits: int,
    null_name: str,
    null_seed: int,
) -> dict:
    stat = struct_path.stat()
    identity_payload = {
        "cache_version": SOURCE_CACHE_VERSION,
        "input_struct_path": str(struct_path),
        "input_struct_size": int(stat.st_size),
        "input_struct_mtime_ns": int(stat.st_mtime_ns),
        "source_variant": str(config.get("source_variant", config.get("variant", ""))),
        "iteration": config.get("iteration"),
        "null_model": null_name,
        "null_seed": int(null_seed),
        "generator_constants": {
            "lz_calibration_window_bits": int(LZ_CALIBRATION_WINDOW_BITS),
            "lz_max_windows": int(LZ_MAX_WINDOWS),
            "matched_lz_block_bits": [int(value) for value in MATCHED_LZ_BLOCK_BITS],
            "phase_matched_lz_period_blocks": [
                int(value) for value in PHASE_MATCHED_LZ_PERIOD_BLOCKS
            ],
            "markov1_progress_chunk_bits": int(MARKOV1_PROGRESS_CHUNK_BITS),
        },
    }
    identity_json = json.dumps(identity_payload, sort_keys=True, separators=(",", ":"))
    identity_hash = hashlib.sha256(identity_json.encode("utf-8")).hexdigest()[:24]
    return {
        "identity": identity_payload,
        "identity_hash": identity_hash,
        "requested_bits": int(required_bits),
    }


def _load_cached_null_source(
    cache_identity: dict,
    *,
    required_bits: int,
    show_progress: bool,
    progress_label: str | None,
) -> tuple[np.ndarray, dict] | None:
    cache_root = _source_cache_root()
    identity_hash = str(cache_identity["identity_hash"])
    candidates = []
    if cache_root.exists():
        for meta_path in cache_root.glob(f"{identity_hash}__bits-*.json"):
            try:
                with open(meta_path, "r", encoding="utf-8") as handle:
                    metadata = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue
            cached_bits = int(metadata.get("required_bits", 0))
            bits_path = Path(metadata.get("bits_path", ""))
            if cached_bits >= required_bits and bits_path.is_file():
                candidates.append((cached_bits, bits_path, meta_path, metadata))
    if not candidates:
        return None

    cached_bits, bits_path, meta_path, metadata = min(candidates, key=lambda item: item[0])
    task_name = (
        f"Load cached source ({progress_label})"
        if progress_label
        else "Load cached source"
    )
    if show_progress:
        with ProgressIndicator(task_name, total=1) as progress:
            source_bits = np.load(bits_path, mmap_mode="r")
            progress.update(1, message=f"{required_bits:,}/{cached_bits:,} bits | cache hit")
    else:
        source_bits = np.load(bits_path, mmap_mode="r")
    if int(source_bits.size) < required_bits:
        return None

    identity = dict(metadata.get("identity", {}))
    info = {
        "source_kind": "null_surrogate",
        "source_struct_path": str(identity.get("input_struct_path", "")),
        "loaded_observable_bits": int(required_bits),
        "source_freeze_mode": "run-local",
        "null_model": identity.get("null_model"),
        "null_seed": identity.get("null_seed"),
        "null_metadata": metadata.get("null_metadata"),
        "source_cache_status": "hit",
        "source_cache_key": identity_hash,
        "source_cache_path": str(bits_path),
        "source_cache_metadata_path": str(meta_path),
        "source_cache_total_bits": int(cached_bits),
        "source_cache_requested_bits": int(required_bits),
    }
    return source_bits[:required_bits], info


def _store_cached_null_source(
    cache_identity: dict,
    source_bits: np.ndarray,
    *,
    null_metadata: dict | None,
    show_progress: bool,
    progress_label: str | None,
) -> dict:
    cache_root = _source_cache_root()
    cache_root.mkdir(parents=True, exist_ok=True)

    identity_hash = str(cache_identity["identity_hash"])
    total_bits = int(source_bits.size)
    stem = f"{identity_hash}__bits-{total_bits}"
    bits_path = cache_root / f"{stem}.npy"
    meta_path = cache_root / f"{stem}.json"

    if not bits_path.is_file():
        tmp_bits_path = cache_root / f"{stem}.{os.getpid()}.tmp.npy"
        task_name = (
            f"Save source cache ({progress_label})"
            if progress_label
            else "Save source cache"
        )
        if show_progress:
            with ProgressIndicator(task_name, total=1) as progress:
                np.save(tmp_bits_path, np.ascontiguousarray(source_bits, dtype=np.uint8))
                progress.update(1, message=f"{total_bits:,} bits")
        else:
            np.save(tmp_bits_path, np.ascontiguousarray(source_bits, dtype=np.uint8))
        os.replace(tmp_bits_path, bits_path)

    metadata = {
        "cache_version": SOURCE_CACHE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "identity_hash": identity_hash,
        "identity": cache_identity["identity"],
        "required_bits": total_bits,
        "bits_path": str(bits_path),
        "null_metadata": null_metadata,
    }
    tmp_meta_path = cache_root / f"{stem}.{os.getpid()}.tmp.json"
    with open(tmp_meta_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    os.replace(tmp_meta_path, meta_path)

    return {
        "source_cache_status": "miss-stored",
        "source_cache_key": identity_hash,
        "source_cache_path": str(bits_path),
        "source_cache_metadata_path": str(meta_path),
        "source_cache_total_bits": total_bits,
        "source_cache_requested_bits": total_bits,
    }


@contextmanager
def _source_cache_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = None
    while True:
        try:
            lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(lock_fd, f"{os.getpid()}\n".encode("ascii"))
            break
        except FileExistsError:
            try:
                age = time.time() - lock_path.stat().st_mtime
                if age > SOURCE_CACHE_LOCK_STALE_SECONDS:
                    lock_path.unlink()
                    continue
            except OSError:
                pass
            time.sleep(1.0)
    try:
        yield
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def prepare_frozen_source_cache(
    runs: list[dict],
    *,
    required_bits: int,
    show_progress: bool = False,
) -> dict[str, dict]:
    frozen_sources: dict[str, dict] = {}
    for run in runs:
        run_dir = str(Path(run["_run_dir"]))
        if run_dir in frozen_sources:
            continue
        config = run["dataset"]["config"]
        source_bits, source_info = load_run_source_bits(
            run["dataset"],
            required_bits=required_bits,
            show_progress=show_progress,
            progress_label=str(config["variant"]),
        )
        frozen_sources[run_dir] = {
            "bits": source_bits,
            "source_info": {
                **source_info,
                "source_freeze_mode": "sweep-global-frozen",
                "frozen_total_bits": required_bits,
                "frozen_run_dir": run_dir,
            },
        }
    return frozen_sources


def compute_return_lag_result(
    segments: list[np.ndarray],
    *,
    selected_patterns: list[dict],
    pattern_scale: int,
    long_lag_threshold: int,
    show_progress: bool = False,
    progress_label: str | None = None,
) -> dict:
    if pattern_scale <= 0 or pattern_scale > 64:
        raise ValueError("pattern_scale must be in the range 1..64.")
    if long_lag_threshold <= 0:
        raise ValueError("long_lag_threshold must be positive.")

    pattern_stats = {
        item["pattern"]: {
            "pattern": item["pattern"],
            "raw_count": item["raw_count"],
            "normalized_frequency": item["normalized_frequency"],
            "occurrence_count": 0,
            "return_count": 0,
            "long_lag_count": 0,
            "lag_sum": 0.0,
            "lag_chunks": [],
            "bin_counts": {},
            "segment_occurrences": [],
            "last_global_position": None,
        }
        for item in selected_patterns
    }

    total_occurrence_count = 0
    total_return_count = 0
    total_long_lag_count = 0
    lag_chunks: list[np.ndarray] = []
    global_bin_counts: dict[str, int] = {}
    segment_results = []
    segment_profile_sets = []
    segment_profile_maps = []
    segment_start_bit = 0
    previous_tail = np.array([], dtype=np.uint8)

    segment_iter = _wrap_progress(
        list(enumerate(segments)),
        total=len(segments),
        show_progress=show_progress,
        desc=_progress_desc("Return-lag scan", progress_label),
        unit="segment",
    )
    for segment_index, segment in segment_iter:
        if pattern_scale > 1 and previous_tail.size:
            analysis_bits = np.concatenate((previous_tail, segment))
            analysis_start_bit = int(segment_start_bit - previous_tail.size)
        else:
            analysis_bits = segment
            analysis_start_bit = int(segment_start_bit)
        if analysis_bits.size < pattern_scale:
            codes = np.array([], dtype=np.uint64)
        else:
            codes = rolling_codes_uint64_max64(analysis_bits, pattern_scale)
        segment_bin_counts: dict[str, int] = {}
        segment_occurrence_count = 0
        segment_return_count = 0
        segment_long_lag_count = 0
        active_patterns = set()

        for item in selected_patterns:
            local_positions = np.flatnonzero(codes == item["code"]).astype(np.int64)
            occurrence_count = int(local_positions.size)
            stats = pattern_stats[item["pattern"]]
            stats["occurrence_count"] += occurrence_count
            stats["segment_occurrences"].append(occurrence_count)
            total_occurrence_count += occurrence_count
            segment_occurrence_count += occurrence_count
            if occurrence_count:
                active_patterns.add(item["pattern"])
                global_positions = local_positions + analysis_start_bit
                lags = _build_global_lags(global_positions, stats["last_global_position"])
                stats["last_global_position"] = int(global_positions[-1])
            else:
                lags = np.array([], dtype=np.int64)

            if lags.size == 0:
                continue

            return_count = int(lags.size)
            long_lag_count = int(np.count_nonzero(lags >= long_lag_threshold))
            bin_counts = _bin_lags(lags)

            stats["return_count"] += return_count
            stats["long_lag_count"] += long_lag_count
            stats["lag_sum"] += float(lags.sum())
            stats["lag_chunks"].append(lags)
            _merge_counts(stats["bin_counts"], bin_counts)

            total_return_count += return_count
            total_long_lag_count += long_lag_count
            segment_return_count += return_count
            segment_long_lag_count += long_lag_count
            lag_chunks.append(lags)
            _merge_counts(segment_bin_counts, bin_counts)
            _merge_counts(global_bin_counts, bin_counts)

        segment_profile_sets.append(set(segment_bin_counts))
        segment_profile_maps.append(segment_bin_counts)
        segment_results.append(
            {
                "segment_index": segment_index,
                "occurrence_count": segment_occurrence_count,
                "return_count": segment_return_count,
                "long_lag_fraction": _safe_ratio(segment_long_lag_count, segment_return_count),
                "active_pattern_count": len(active_patterns),
                "lag_entropy": _weight_entropy(segment_bin_counts.values()),
                "top_return_bins": _top_bins(segment_bin_counts, limit=5),
            }
        )
        if pattern_scale > 1:
            carry_size = min(pattern_scale - 1, int(segment.size))
            previous_tail = (
                np.ascontiguousarray(segment[-carry_size:].copy())
                if carry_size > 0
                else np.array([], dtype=np.uint8)
            )
        else:
            previous_tail = np.array([], dtype=np.uint8)
        segment_start_bit += int(segment.size)
    _close_progress(segment_iter)

    if lag_chunks:
        all_lags = np.concatenate(lag_chunks).astype(np.int64, copy=False)
        mean_return_lag = float(all_lags.mean())
        median_return_lag = float(np.median(all_lags))
    else:
        mean_return_lag = 0.0
        median_return_lag = 0.0

    pattern_results = []
    for item in selected_patterns:
        stats = pattern_stats[item["pattern"]]
        pattern_lags = (
            np.concatenate(stats["lag_chunks"]).astype(np.int64, copy=False)
            if stats["lag_chunks"]
            else np.array([], dtype=np.int64)
        )
        pattern_results.append(
            {
                "pattern": item["pattern"],
                "raw_count": item["raw_count"],
                "normalized_frequency": item["normalized_frequency"],
                "occurrence_count": stats["occurrence_count"],
                "return_count": stats["return_count"],
                "mean_return_lag": float(pattern_lags.mean()) if pattern_lags.size else 0.0,
                "median_return_lag": float(np.median(pattern_lags)) if pattern_lags.size else 0.0,
                "long_lag_fraction": _safe_ratio(stats["long_lag_count"], stats["return_count"]),
                "lag_entropy": _weight_entropy(stats["bin_counts"].values()),
                "lag_bin_counts": {
                    key: int(value)
                    for key, value in sorted(stats["bin_counts"].items(), key=lambda item: item[0])
                },
                "segment_occurrences": stats["segment_occurrences"],
                "top_return_bins": _top_bins(stats["bin_counts"], limit=5),
            }
        )

    return {
        "pattern_scale": pattern_scale,
        "selected_pattern_count": len(selected_patterns),
        "long_lag_threshold": long_lag_threshold,
        "total_occurrence_count": total_occurrence_count,
        "total_return_count": total_return_count,
        "mean_return_lag": mean_return_lag,
        "median_return_lag": median_return_lag,
        "lag_entropy": _weight_entropy(global_bin_counts.values()),
        "dominant_bin_mass_fraction": _dominant_mass_fraction(global_bin_counts),
        "long_lag_fraction": _safe_ratio(total_long_lag_count, total_return_count),
        "segment_profile_jaccard": _pairwise_jaccard_mean(segment_profile_sets),
        "segment_profile_weighted_jaccard": _pairwise_weighted_jaccard_mean(segment_profile_maps),
        "top_return_bins": _top_bins(global_bin_counts, limit=10),
        "segment_results": segment_results,
        "pattern_results": pattern_results,
        "notes": [
            "selected patterns are frozen from the configured Phase 1 selection source before lag evaluation begins",
            "return lag = distance in window-start positions between consecutive occurrences of the same selected pattern across the full analyzed strip",
            "cross-segment pattern starts are preserved by carrying an m-1 overlap into each subsequent segment",
            "cross-segment returns are preserved by carrying the last occurrence of each selected pattern across segment boundaries",
            "lag bins are log2 buckets over exact return distances",
            "long_lag_fraction measures the share of returns at or above the configured long-lag threshold",
        ],
    }


def render_console_summary(rows: list[dict]) -> str:
    lines = [
        "Phase 2 return-lag spectrum pilot",
        "-" * 122,
        f"{'variant':<14}{'iter':>6}{'occ':>10}{'ret':>10}{'mean_lag':>12}{'lag_H':>10}{'dom_bin':>10}{'long':>10}{'seg_J':>10}{'seg_wJ':>10}",
    ]
    for row in rows:
        lines.append(
            f"{row['variant']:<14}"
            f"{_fmt_int(row['iteration']):>6}"
            f"{row['total_occurrence_count']:>10}"
            f"{row['total_return_count']:>10}"
            f"{_fmt(row['mean_return_lag']):>12}"
            f"{_fmt(row['lag_entropy']):>10}"
            f"{_fmt(row['dominant_bin_mass_fraction']):>10}"
            f"{_fmt(row['long_lag_fraction']):>10}"
            f"{_fmt(row['segment_profile_jaccard']):>10}"
            f"{_fmt(row['segment_profile_weighted_jaccard']):>10}"
        )
    return "\n".join(lines)


def render_markdown_report(rows: list[dict], selection: dict) -> str:
    lines = [
        "# Phase 2 Return-Lag Spectrum Pilot",
        "",
        "## Selection",
        "",
        f"- Variants: {', '.join(selection['variants'])}",
        f"- Segment protocol: {selection['num_segments']} x {selection['segment_bits']}",
        f"- Segment offset bits: {selection['segment_offset_bits']}",
        f"- Phase 1 scales: {', '.join(str(value) for value in selection['phase1_scales'])}",
        f"- Pattern scale: {selection['pattern_scale']}",
        f"- Top patterns: {selection['top_patterns']}",
        f"- Pattern selection: {selection['pattern_selection']}",
        f"- Long-lag threshold: {selection['long_lag_threshold']}",
        "",
        "## Compact Readout",
        "",
        "| Variant | Iter | Occurrences | Returns | Mean lag | Lag entropy | Dominant bin mass | Long-lag frac | Segment Jaccard | Segment weighted Jaccard |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['iteration']} | {row['total_occurrence_count']} | {row['total_return_count']} | "
            f"{_fmt(row['mean_return_lag'])} | {_fmt(row['lag_entropy'])} | {_fmt(row['dominant_bin_mass_fraction'])} | "
            f"{_fmt(row['long_lag_fraction'])} | {_fmt(row['segment_profile_jaccard'])} | {_fmt(row['segment_profile_weighted_jaccard'])} |"
        )

    for row in rows:
        result = row["lag_result"]
        lines.extend(
            [
                "",
                f"## {row['variant']}",
                "",
                f"- Source kind: {row['source_kind']}",
                f"- Pattern scale: {row['pattern_scale']}",
                f"- Selected patterns: {row['selected_pattern_count']}",
                f"- Pattern selection: {row['pattern_selection']}",
                f"- Total occurrences: {row['total_occurrence_count']}",
                f"- Total returns: {row['total_return_count']}",
                f"- Mean return lag: {_fmt(row['mean_return_lag'])}",
                f"- Median return lag: {_fmt(row['median_return_lag'])}",
                f"- Lag entropy: {_fmt(row['lag_entropy'])}",
                f"- Dominant bin mass fraction: {_fmt(row['dominant_bin_mass_fraction'])}",
                f"- Long-lag fraction: {_fmt(row['long_lag_fraction'])}",
                f"- Segment profile Jaccard: {_fmt(row['segment_profile_jaccard'])}",
                f"- Segment profile weighted Jaccard: {_fmt(row['segment_profile_weighted_jaccard'])}",
                "",
                "Top return bins:",
            ]
        )
        if result["top_return_bins"]:
            for item in result["top_return_bins"]:
                lines.append(
                    f"- `{item['bin']}` | count={item['count']} | mass={_fmt(item['mass_fraction'])}"
                )
        else:
            lines.append("- none")

        lines.append("")
        lines.append("Top patterns:")
        for item in result["pattern_results"][:5]:
            lines.append(
                f"- `{item['pattern']}` | occ={item['occurrence_count']} | returns={item['return_count']} | "
                f"mean_lag={_fmt(item['mean_return_lag'])} | long={_fmt(item['long_lag_fraction'])}"
            )
    return "\n".join(lines)


def _select_patterns(
    pattern_space: dict,
    *,
    top_patterns: int,
    selection_mode: str,
    phase1_scales: list[int] | None = None,
    pattern_scale: int | None = None,
    prefix_fibers: list[dict] | None = None,
    suffix_fibers: list[dict] | None = None,
) -> list[dict]:
    if top_patterns <= 0:
        raise ValueError("top_patterns must be positive.")
    patterns = pattern_space.get("patterns", [])
    if not patterns:
        raise ValueError("Pattern space contains no patterns.")
    selection_mode = str(selection_mode).strip().lower()
    if selection_mode == "top":
        ranked_items = [
            {
                "pattern": item["pattern"],
                "raw_count": int(item["raw_count"]),
                "normalized_frequency": float(item["normalized_frequency"]),
                "segment_support": int(item.get("segment_support", 0)),
                "cv_supporting": float(item.get("cv_supporting", 0.0)),
            }
            for item in patterns
        ]
    elif selection_mode == "rare-stable":
        segment_count = int(pattern_space.get("segment_count", 0))
        ranked_items = sorted(
            [
                {
                    "pattern": item["pattern"],
                    "raw_count": int(item["raw_count"]),
                    "normalized_frequency": float(item["normalized_frequency"]),
                    "segment_support": int(item.get("segment_support", 0)),
                    "cv_supporting": float(item.get("cv_supporting", 0.0)),
                }
                for item in patterns
            ],
            key=lambda item: (
                -(1 if item["segment_support"] == segment_count else 0),
                -item["segment_support"],
                item["normalized_frequency"],
                item["cv_supporting"],
                item["pattern"],
            ),
        )
    elif selection_mode == "bridge-linked":
        ranked_items = _select_bridge_linked_candidates(
            pattern_space,
            phase1_scales=phase1_scales or [],
            pattern_scale=pattern_scale,
            prefix_fibers=prefix_fibers,
            suffix_fibers=suffix_fibers,
        )
    else:
        raise ValueError(f"Unsupported pattern selection mode: {selection_mode}")

    selected = []
    for item in ranked_items[:top_patterns]:
        pattern = item["pattern"]
        selected.append(
            {
                "pattern": pattern,
                "code": int(pattern, 2),
                "raw_count": item["raw_count"],
                "normalized_frequency": item["normalized_frequency"],
                "segment_support": item["segment_support"],
                "cv_supporting": item["cv_supporting"],
                **{
                    key: value
                    for key, value in item.items()
                    if key
                    not in {
                        "pattern",
                        "raw_count",
                        "normalized_frequency",
                        "segment_support",
                        "cv_supporting",
                    }
                },
            }
        )
    return selected


def _select_bridge_linked_candidates(
    pattern_space: dict,
    *,
    phase1_scales: list[int],
    pattern_scale: int | None,
    prefix_fibers: list[dict] | None,
    suffix_fibers: list[dict] | None,
) -> list[dict]:
    if pattern_scale is None:
        raise ValueError("bridge-linked selection requires a concrete pattern_scale.")
    if not prefix_fibers or not suffix_fibers:
        raise ValueError("bridge-linked selection requires both prefix and suffix fiber artifacts.")

    lower_scale, higher_scale = _resolve_adjacent_scales(phase1_scales, pattern_scale)
    if lower_scale is None:
        raise ValueError(
            f"bridge-linked selection requires a lower adjacent scale below m={pattern_scale}."
        )

    prefix_lower_edge = _require_edge(prefix_fibers, target_m=lower_scale, source_m=pattern_scale)
    suffix_lower_edge = _require_edge(suffix_fibers, target_m=lower_scale, source_m=pattern_scale)
    prefix_child_index = _index_child_linkage(prefix_lower_edge)
    suffix_child_index = _index_child_linkage(suffix_lower_edge)

    prefix_future_index: dict[str, dict] = {}
    suffix_future_index: dict[str, dict] = {}
    if higher_scale is not None:
        prefix_future_index = _index_parent_future(
            _require_edge(prefix_fibers, target_m=pattern_scale, source_m=higher_scale)
        )
        suffix_future_index = _index_parent_future(
            _require_edge(suffix_fibers, target_m=pattern_scale, source_m=higher_scale)
        )

    candidates = []
    for item in pattern_space.get("patterns", []):
        pattern = str(item["pattern"])
        prefix_meta = prefix_child_index.get(pattern)
        suffix_meta = suffix_child_index.get(pattern)
        if prefix_meta is None or suffix_meta is None:
            continue

        prefix_future = prefix_future_index.get(pattern, {})
        suffix_future = suffix_future_index.get(pattern, {})
        future_bearing_prefix = bool(prefix_future.get("future_bearing", False))
        future_bearing_suffix = bool(suffix_future.get("future_bearing", False))
        candidates.append(
            {
                "pattern": pattern,
                "raw_count": int(item["raw_count"]),
                "normalized_frequency": float(item["normalized_frequency"]),
                "segment_support": int(item.get("segment_support", 0)),
                "cv_supporting": float(item.get("cv_supporting", 0.0)),
                "bridge_lower_scale": lower_scale,
                "bridge_higher_scale": higher_scale,
                "prefix_parent": prefix_meta["primary_parent"],
                "suffix_parent": suffix_meta["primary_parent"],
                "prefix_parent_count": prefix_meta["parent_count"],
                "suffix_parent_count": suffix_meta["parent_count"],
                "distinct_corner_flag": bool(
                    prefix_meta["primary_parent"] != suffix_meta["primary_parent"]
                ),
                "future_bearing_prefix": future_bearing_prefix,
                "future_bearing_suffix": future_bearing_suffix,
                "future_bearing_both": bool(future_bearing_prefix and future_bearing_suffix),
                "prefix_future_fiber_size": int(prefix_future.get("fiber_size", 0)),
                "suffix_future_fiber_size": int(suffix_future.get("fiber_size", 0)),
            }
        )

    if not candidates:
        raise ValueError(
            f"bridge-linked selection found no candidate patterns at m={pattern_scale}."
        )

    return sorted(
        candidates,
        key=lambda item: (
            -(1 if item["future_bearing_both"] else 0),
            -(1 if item["future_bearing_prefix"] else 0),
            -(1 if item["future_bearing_suffix"] else 0),
            -(1 if item["distinct_corner_flag"] else 0),
            -item["segment_support"],
            item["normalized_frequency"],
            item["cv_supporting"],
            item["pattern"],
        ),
    )


def _resolve_adjacent_scales(
    phase1_scales: list[int],
    pattern_scale: int,
) -> tuple[int | None, int | None]:
    ordered = sorted({int(value) for value in phase1_scales})
    lower_scale = max((value for value in ordered if value < pattern_scale), default=None)
    higher_scale = min((value for value in ordered if value > pattern_scale), default=None)
    return lower_scale, higher_scale


def _index_child_linkage(edge: dict) -> dict[str, dict]:
    child_index: dict[str, dict] = {}
    for parent in edge.get("parents", []):
        parent_pattern = str(parent["pattern"])
        parent_weight = float(parent.get("normalized_frequency", 0.0))
        for child in parent.get("children", []):
            child_pattern = str(child["pattern"])
            entry = child_index.setdefault(
                child_pattern,
                {
                    "parent_patterns": set(),
                    "primary_parent": parent_pattern,
                    "primary_parent_weight": parent_weight,
                },
            )
            entry["parent_patterns"].add(parent_pattern)
            if parent_weight > entry["primary_parent_weight"]:
                entry["primary_parent"] = parent_pattern
                entry["primary_parent_weight"] = parent_weight

    for child_pattern, entry in child_index.items():
        entry["parent_patterns"] = sorted(entry["parent_patterns"])
        entry["parent_count"] = len(entry["parent_patterns"])
        entry["pattern"] = child_pattern
    return child_index


def _index_parent_future(edge: dict) -> dict[str, dict]:
    return {
        str(parent["pattern"]): {
            "future_bearing": bool(parent.get("survival_flag", False)),
            "fiber_size": int(parent.get("fiber_size", 0)),
            "child_entropy": float(parent.get("child_entropy", 0.0)),
            "child_total_mass": float(parent.get("child_total_mass", 0.0)),
        }
        for parent in edge.get("parents", [])
    }


def _bin_lags(lags: np.ndarray) -> dict[str, int]:
    if lags.size == 0:
        return {}
    bin_ids = np.floor(np.log2(lags.astype(np.float64))).astype(np.int64)
    unique_bins, counts = np.unique(bin_ids, return_counts=True)
    return {
        _lag_bin_label(int(bin_id)): int(count)
        for bin_id, count in zip(unique_bins.tolist(), counts.tolist())
    }


def _build_global_lags(
    global_positions: np.ndarray,
    previous_last_position: int | None,
) -> np.ndarray:
    if global_positions.size == 0:
        return np.array([], dtype=np.int64)
    if previous_last_position is None:
        if global_positions.size < 2:
            return np.array([], dtype=np.int64)
        return np.diff(global_positions)

    lags = np.empty(global_positions.size, dtype=np.int64)
    lags[0] = int(global_positions[0] - previous_last_position)
    if global_positions.size > 1:
        lags[1:] = np.diff(global_positions)
    return lags


def _lag_bin_label(bin_id: int) -> str:
    low = 1 << bin_id
    high = (1 << (bin_id + 1)) - 1
    return f"{low}-{high}"


def _merge_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + int(value)


def _dominant_mass_fraction(bin_counts: dict[str, int]) -> float:
    if not bin_counts:
        return 0.0
    total = sum(bin_counts.values())
    if total <= 0:
        return 0.0
    return float(max(bin_counts.values()) / total)


def _top_bins(bin_counts: dict[str, int], *, limit: int) -> list[dict]:
    total = sum(bin_counts.values())
    if total <= 0:
        return []
    return [
        {
            "bin": key,
            "count": int(value),
            "mass_fraction": float(value / total),
        }
        for key, value in sorted(bin_counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]


def _pairwise_jaccard_mean(sets: list[set[str]]) -> float:
    if len(sets) < 2:
        return 1.0 if sets else 0.0
    values = []
    for index in range(len(sets)):
        for other in range(index + 1, len(sets)):
            union = sets[index] | sets[other]
            if not union:
                values.append(1.0)
            else:
                values.append(len(sets[index] & sets[other]) / len(union))
    return float(sum(values) / len(values)) if values else 0.0


def _pairwise_weighted_jaccard_mean(dicts: list[dict[str, int]]) -> float:
    if len(dicts) < 2:
        return 1.0 if dicts else 0.0
    values = []
    for index in range(len(dicts)):
        for other in range(index + 1, len(dicts)):
            keys = set(dicts[index]) | set(dicts[other])
            if not keys:
                values.append(1.0)
                continue
            shared = sum(min(dicts[index].get(key, 0), dicts[other].get(key, 0)) for key in keys)
            union = sum(max(dicts[index].get(key, 0), dicts[other].get(key, 0)) for key in keys)
            values.append(_safe_ratio(shared, union))
    return float(sum(values) / len(values)) if values else 0.0


def _weight_entropy(weights) -> float:
    values = [float(value) for value in weights if float(value) > 0]
    if not values:
        return 0.0
    total = sum(values)
    entropy = 0.0
    for value in values:
        prob = value / total
        entropy -= prob * math.log(prob, 2)
    return float(entropy)


def _safe_ratio(numerator: float | int, denominator: float | int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def _fmt_int(value: int | None) -> str:
    return "-" if value is None else str(value)


def _wrap_progress(iterable, *, total: int, show_progress: bool, desc: str, unit: str):
    if not show_progress:
        return iterable
    if tqdm is not None:
        return tqdm(iterable, total=total, desc=desc, unit=unit, file=sys.stdout)
    return _ProgressIterator(iterable, desc=desc, total=total)


def _close_progress(progress_iter) -> None:
    close = getattr(progress_iter, "close", None)
    if callable(close):
        close()


def _progress_desc(base: str, label: str | None) -> str:
    return base if not label else f"{base} | {label}"


class _ProgressIterator:
    def __init__(self, iterable, *, desc: str, total: int):
        self._iterable = iter(iterable)
        self._indicator = ProgressIndicator(desc, total=total)
        self._entered = False
        self.current = 0

    def __iter__(self):
        if not self._entered:
            self._indicator.__enter__()
            self._entered = True
        return self

    def __next__(self):
        value = next(self._iterable)
        self.current += 1
        self._indicator.update(self.current)
        return value

    def close(self):
        if self._entered:
            self._indicator.__exit__(None, None, None)
            self._entered = False
