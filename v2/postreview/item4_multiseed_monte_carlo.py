from __future__ import annotations

import json
import math
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from v2.common.io import load_observable_cache, split_into_segments
from v2.common.null_models import generate_control
from v2.phase1.coherence import build_coherence_rows
from v2.phase1.report import build_report_rows
from v2.phase1.tower import (
    build_pattern_spaces,
    build_projection_bundle,
    serialize_pattern_spaces,
    summarize_phase1,
)
from v2.phase1.transport import _build_transport_profile, build_transport_rows
from v2.postreview.item2b_growth_shells import (
    ObservableCursor,
    RunLogger,
    V33ObservableReader,
    current_rss_bytes,
    freeze_cache,
    load_phase1_run,
    sha256_file,
    source_sequence_length,
    write_json_atomic,
)


PREREG_SHA256 = "F8A4188170B8980599199B8E1696286A564B982E2703197E75A11E306B6FA31A"
NULL_IMPLEMENTATION_SHA256 = (
    "C367761E6BB7203F2E53D29B7F65170062BB58927E79D780120CD352C8088A73"
)
WINDOW_BITS = 3_000_000
SEGMENT_BITS = 1_000_000
SEGMENT_COUNT = 3
SCALES = (8, 12, 16, 20, 24, 28, 32)
POLICIES = ("prefix", "suffix")
NULL_FAMILIES = ("markov1", "matched-lz", "phase-matched-lz")
SEEDS = tuple(range(1, 512))
TAIL_EDGES = 2


@dataclass(frozen=True)
class SourceSpec:
    source: str
    iteration: int
    sha256: str
    observable_bits: int


SOURCE_SPECS = {
    "B": SourceSpec(
        "B",
        20,
        "B1FE25ACB1729C4862BE843559696206896402E3698E5F2F671F3E37795CA3D8",
        11_198_388_932,
    ),
    "I": SourceSpec(
        "I",
        23,
        "913452A3185F2A603A4447688F679903FAA15438B88FBDB029BA1F26835EB024",
        265_785_965_991,
    ),
    "D": SourceSpec(
        "D",
        20,
        "67E870C136A88C4E2F87DE34F3D84A46FF95FC59C930A7208598945DEF41CA1F",
        10_242_152_744,
    ),
    "G": SourceSpec(
        "G",
        20,
        "3A8977E8AB2DEA07D93D8EA99579B6CFCBE016950C06F58CBE6113630307F467",
        11_198_390_913,
    ),
}

BASELINE_DISPLAY = {
    "B": {"R": "0.8622", "S": "0.9978", "A": "0.3539"},
    "I": {"R": "0.9930", "S": "1.0000", "A": "0.5064"},
    "D": {"R": "0.9785", "S": "1.0000", "A": "0.5055"},
    "G": {"R": "0.9403", "S": "0.9978", "A": "0.4987"},
}

CLEAN_BASELINE_PROVENANCE = {
    "path": (
        "D:/Iban/Projectes/HSI/results/hsi_v2/"
        "d0154_priority_a_phase1_table_reconciliation/"
        "d0154-phase1-table-reconciliation__20260703T121926/"
        "clean_variant_phase1_values.csv"
    ),
    "sha256": "5019617C2FFD171031F410BA4D2413B003F7D7110256526903B46A737AA6538F",
    "b_reference": "clean E@24; audited observable/Phase-1 identity with canonical B",
}

CLEAN_BASELINE_FULL = {
    "B": {
        "R": 0.8622412562455389,
        "S": 0.9978021978021978,
        "A": 0.3538634631882274,
    },
    "I": {
        "R": 0.9929701230228472,
        "S": 1.0,
        "A": 0.5063988225345406,
    },
    "D": {
        "R": 0.9785082174462706,
        "S": 1.0,
        "A": 0.5055203950624123,
    },
    "G": {
        "R": 0.9403225806451613,
        "S": 0.9978260869565218,
        "A": 0.498679343192611,
    },
}


def frozen_protocol() -> dict[str, Any]:
    return {
        "window_bits": WINDOW_BITS,
        "segment_bits": SEGMENT_BITS,
        "num_segments": SEGMENT_COUNT,
        "scales": list(SCALES),
        "policies": list(POLICIES),
        "min_count_floor": 16,
        "min_count_rate": 1e-6,
        "min_segment_support": 2,
        "cv_max": 1.5,
        "main_policy": "prefix",
        "control_policy": "suffix",
        "tail_edges": TAIL_EDGES,
        "null_families": list(NULL_FAMILIES),
        "seeds": [1, 511],
        "seed_count": len(SEEDS),
    }


def source_paths(results_base: Path) -> dict[str, Path]:
    return {
        source: results_base
        / "level0"
        / "phi_snapshots"
        / f"var_{source}"
        / f"phi_iter{spec.iteration}.struct.gz"
        for source, spec in SOURCE_SPECS.items()
    }


def observed_cache_identity(
    spec: SourceSpec, source_path: Path, source_sha256: str
) -> dict[str, Any]:
    return {
        "cache_kind": "observed",
        "source": spec.source,
        "iteration": spec.iteration,
        "source_path": str(source_path.resolve()),
        "source_sha256": source_sha256,
        "source_observable_length": spec.observable_bits,
        "window_start": 0,
        "window_end": WINDOW_BITS,
        "preregistration_sha256": PREREG_SHA256,
    }


def validate_identity_cache(
    cache_path: Path, *, identity: dict[str, Any], expected_bits: int = WINDOW_BITS
) -> dict[str, Any] | None:
    try:
        _, sidecar, _ = load_observable_cache(cache_path, expected_bits=expected_bits)
        if any(sidecar.get(key) != value for key, value in identity.items()):
            return None
        return sidecar
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
        return None


def quarantine_paths(paths: list[Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return
    root = (
        existing[0].parent / "quarantine" / datetime.now().strftime("%Y%m%dT%H%M%S%f")
    )
    root.mkdir(parents=True, exist_ok=True)
    for path in existing:
        shutil.move(str(path), str(root / path.name))


def freeze_identity_cache(
    cache_path: Path, bits: np.ndarray, *, identity: dict[str, Any]
) -> dict[str, Any]:
    existing = validate_identity_cache(cache_path, identity=identity)
    if existing is not None:
        return {**existing, "reused": True}
    quarantine_paths([cache_path, cache_path.with_suffix(".json")])
    return {**freeze_cache(cache_path, bits, metadata=identity), "reused": False}


def extract_observed_prefix(
    source_path: Path,
    *,
    spec: SourceSpec,
    source_sha256: str,
    cache_path: Path,
    raw_chunk_bytes: int,
    logger: RunLogger,
) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = observed_cache_identity(spec, source_path, source_sha256)
    reusable = validate_identity_cache(cache_path, identity=identity)
    if reusable is not None:
        return {**reusable, "reused": True}, {
            "reused": True,
            "sequence_length": source_sequence_length(source_path),
            "observable_bits": spec.observable_bits,
        }

    started = time.monotonic()

    def progress(reader: V33ObservableReader) -> None:
        elapsed = max(time.monotonic() - started, 1e-9)
        rate = reader.observable_bits_decoded / elapsed
        logger.emit(
            "progress",
            force=False,
            phase="stage0-extract-observed",
            source=spec.source,
            packed_bytes=f"{reader.packed_bytes_read}/{reader.packed_bytes_expected}",
            observable_bits=reader.observable_bits_decoded,
            percent=round(
                100.0 * reader.observable_bits_decoded / spec.observable_bits, 6
            ),
            elapsed_seconds=round(elapsed, 3),
            rate_bits_per_second=round(rate, 3),
            eta_seconds=round(
                (spec.observable_bits - reader.observable_bits_decoded)
                / max(rate, 1e-9),
                3,
            ),
            peak_rss_bytes=reader.peak_rss_bytes,
        )

    with V33ObservableReader(
        source_path,
        sequence_length=source_sequence_length(source_path),
        expected_observable_bits=spec.observable_bits,
        raw_chunk_bytes=raw_chunk_bytes,
        progress=progress,
    ) as reader:
        cursor = ObservableCursor(reader)
        bits = cursor.read_exact(WINDOW_BITS)
        cursor.drain()
        if cursor.position != spec.observable_bits:
            raise RuntimeError(
                f"Observable length mismatch for {spec.source}: {cursor.position}."
            )
        telemetry = {
            "reused": False,
            "sequence_length": reader.sequence_length,
            "observable_bits": reader.observable_bits_decoded,
            "packed_bytes_read": reader.packed_bytes_read,
            "padding_symbols_excluded": reader.padding_symbols_excluded,
            "peak_rss_bytes": reader.peak_rss_bytes,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
    return freeze_identity_cache(cache_path, bits, identity=identity), telemetry


def null_cache_identity(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "cache_kind": "null",
        "source": spec["source"],
        "iteration": int(spec["iteration"]),
        "observed_cache_sha256": spec["observed_cache_sha256"],
        "null_family": spec["family"],
        "null_seed": int(spec["seed"]),
        "null_implementation_sha256": NULL_IMPLEMENTATION_SHA256,
        "preregistration_sha256": PREREG_SHA256,
    }


def markov1_fit_metadata(bits: np.ndarray) -> dict[str, Any]:
    density = float(bits.mean()) if bits.size else 0.0
    counts = np.zeros((2, 2), dtype=np.int64)
    if bits.size > 1:
        counts = np.bincount((bits[:-1] << 1) | bits[1:], minlength=4).reshape(2, 2)
    probabilities = np.empty((2, 2), dtype=np.float64)
    for state in (0, 1):
        total = int(counts[state].sum())
        probabilities[state] = (
            counts[state] / total if total else (1.0 - density, density)
        )
    return {
        "method": "maximum_likelihood_first_order_binary_markov",
        "target_density": density,
        "transition_counts": counts.tolist(),
        "transition_probabilities": probabilities.tolist(),
        "initial_state_probability_one": density,
    }


def valid_fit_metadata(sidecar: dict[str, Any], family: str) -> bool:
    fit = sidecar.get("fit_metadata")
    if not isinstance(fit, dict):
        return False
    if family == "markov1":
        return bool(
            fit.get("method") == "maximum_likelihood_first_order_binary_markov"
            and np.asarray(fit.get("transition_counts", [])).shape == (2, 2)
            and np.asarray(fit.get("transition_probabilities", [])).shape == (2, 2)
        )
    common = bool(
        fit.get("candidate_block_bits")
        == [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        and fit.get("window_bits") == 32_768
        and len(fit.get("window_starts", [])) == 3
        and isinstance(fit.get("lz_abs_error"), (int, float))
    )
    if family == "matched-lz":
        return common and fit.get("method") == "block_shuffle_lz_match"
    if family == "phase-matched-lz":
        return bool(
            common
            and fit.get("method") == "phase_class_block_shuffle_lz_match"
            and fit.get("candidate_phase_period_blocks") == [2, 3, 5, 7]
        )
    return False


def generate_null_cache_job(spec: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    cache_path = Path(spec["cache_path"])
    identity = null_cache_identity(spec)
    existing = validate_identity_cache(cache_path, identity=identity)
    if existing is not None and valid_fit_metadata(existing, spec["family"]):
        return {
            "job_id": spec["job_id"],
            "status": "completed",
            "reused": True,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "peak_rss_bytes": current_rss_bytes(),
            "bytes_written": Path(existing["cache_path"]).stat().st_size,
            "sidecar": existing,
        }
    bits, observed_sidecar, _ = load_observable_cache(
        Path(spec["observed_cache_path"]), expected_bits=WINDOW_BITS
    )
    if observed_sidecar["cache_sha256"] != spec["observed_cache_sha256"]:
        raise RuntimeError(f"Observed cache mismatch for {spec['job_id']}.")
    generated, fit_metadata = generate_control(bits, spec["family"], int(spec["seed"]))
    metadata = {
        **identity,
        "fit_metadata": fit_metadata or markov1_fit_metadata(bits),
    }
    quarantine_paths([cache_path, cache_path.with_suffix(".json")])
    sidecar = freeze_cache(cache_path, generated, metadata=metadata)
    return {
        "job_id": spec["job_id"],
        "status": "completed",
        "reused": False,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "peak_rss_bytes": current_rss_bytes(),
        "bytes_written": cache_path.stat().st_size,
        "sidecar": sidecar,
    }


def compact_profile_from_bundle(
    pattern_spaces: dict[int, dict[str, Any]], bundle: dict[str, Any]
) -> dict[str, Any]:
    summary = summarize_phase1(pattern_spaces, bundle)
    transport = {
        policy: _build_transport_profile(
            bundle["fibers"][policy], tail_edges=TAIL_EDGES
        )
        for policy in POLICIES
    }
    profile = {
        "scale_summary": summary["scale_summary"],
        "fiber_summary": summary["fiber_summary"],
        "transport": transport,
    }
    profile["tail_vector"] = tail_vector_from_compact(profile)
    return profile


def tail_vector_from_compact(profile: dict[str, Any]) -> dict[str, float]:
    scales = profile["scale_summary"]
    prefix = profile["fiber_summary"]["prefix"]
    if not scales or not prefix:
        raise ValueError("Compact profile is missing scale or prefix fiber summaries.")
    last = scales[-1]
    denominator = int(last["candidate_pattern_count"])
    if denominator <= 0:
        raise ValueError("Final candidate denominator is zero.")
    vector = {
        "R": float(last["kept_pattern_count"] / denominator),
        "S": float(prefix[-1]["survival_fraction"]),
        "A": float(
            profile["transport"]["prefix"]["summary"]["tail_transport_active_mean"]
        ),
    }
    if not all(
        math.isfinite(value) and 0.0 <= value <= 1.0 for value in vector.values()
    ):
        raise ValueError(f"Invalid compact tail vector: {vector!r}.")
    return vector


def write_full_phase1_artifacts(
    run_dir: Path,
    *,
    spec: dict[str, Any],
    pattern_spaces: dict[int, dict[str, Any]],
    bundle: dict[str, Any],
) -> None:
    generated_at = datetime.now().isoformat(timespec="seconds")
    config = {
        "variant": spec["source"],
        "iteration": int(spec["iteration"]),
        "segment_bits": SEGMENT_BITS,
        "num_segments": SEGMENT_COUNT,
        "segment_offset_bits": 0,
        "scales": list(SCALES),
        "policies": list(POLICIES),
        "min_count_floor": 16,
        "min_count_rate": 1e-6,
        "min_segment_support_effective": 2,
        "cv_max": 1.5,
        "sampling_policy": "contiguous-prefix",
        "pattern_space_mode": "observed",
    }
    dataset = {
        "stage": "phase1_tower",
        "generated_at": generated_at,
        "config": config,
        "inputs": {
            "observable_cache_path": spec["cache_path"],
            "observable_cache_sha256": spec["cache_sha256"],
            "preregistration_sha256": PREREG_SHA256,
        },
    }
    write_json_atomic(run_dir / "dataset.json", dataset)
    for scale, payload in serialize_pattern_spaces(pattern_spaces).items():
        write_json_atomic(
            run_dir / "pattern_spaces" / f"pattern_space_m{scale}.json", payload
        )
    for policy in POLICIES:
        write_json_atomic(
            run_dir / "projection_towers" / f"projection_{policy}.json",
            bundle["projection_towers"][policy],
        )
        write_json_atomic(
            run_dir / "fibers" / f"fibers_{policy}.json", bundle["fibers"][policy]
        )
        write_json_atomic(
            run_dir / "moments" / f"moments_{policy}.json", bundle["moments"][policy]
        )
    write_json_atomic(
        run_dir / "phase1_summary.json", summarize_phase1(pattern_spaces, bundle)
    )
    write_json_atomic(
        run_dir / "manifest.json",
        {
            "generated_at": generated_at,
            "stage": "phase1_tower",
            "source": spec["source"],
            "cache_sha256": spec["cache_sha256"],
            "preregistration_sha256": PREREG_SHA256,
        },
    )


def tail_vector_from_full_run(run_dir: Path) -> dict[str, float]:
    run = load_phase1_run(run_dir)
    report = build_report_rows([run])[0]
    coherence = build_coherence_rows([run], tail_edges=TAIL_EDGES)[0]
    transport = build_transport_rows([run], tail_edges=TAIL_EDGES)[0]
    return {
        "R": float(report["retention_last"]),
        "S": float(coherence["tail_survival_last"]),
        "A": float(transport["tail_transport_active_mean"]),
    }


def assert_vectors_identical(left: dict[str, float], right: dict[str, float]) -> None:
    for coordinate in ("R", "S", "A"):
        if left[coordinate] != right[coordinate]:
            raise RuntimeError(
                f"Compact/full mismatch for {coordinate}: {left[coordinate]!r} != {right[coordinate]!r}."
            )


def profile_job_identity(
    spec: dict[str, Any], *, code_hashes: dict[str, str], stage0_sha256: str
) -> dict[str, Any]:
    return {
        "job_id": spec["job_id"],
        "source": spec["source"],
        "iteration": int(spec["iteration"]),
        "kind": spec["kind"],
        "family": spec.get("family"),
        "seed": spec.get("seed"),
        "cache_path": spec["cache_path"],
        "cache_sha256": spec["cache_sha256"],
        "stage0_manifest_sha256": stage0_sha256,
        "preregistration_sha256": PREREG_SHA256,
        "protocol": frozen_protocol(),
        "code_hashes": code_hashes,
    }


def build_profile_job(spec: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    job_dir = Path(spec["job_dir"])
    existing = validate_profile_job(spec)
    if existing is not None:
        return {**existing, "reused": True}
    if job_dir.exists():
        quarantine_paths([job_dir])
    job_dir.mkdir(parents=True, exist_ok=True)
    bits, sidecar, _ = load_observable_cache(
        Path(spec["cache_path"]), expected_bits=WINDOW_BITS
    )
    if sidecar["cache_sha256"] != spec["cache_sha256"]:
        raise RuntimeError(f"Cache hash mismatch for {spec['job_id']}.")
    segments = split_into_segments(bits, SEGMENT_BITS, SEGMENT_COUNT)
    pattern_spaces = build_pattern_spaces(
        segments,
        SCALES,
        min_count_floor=16,
        min_count_rate=1e-6,
        min_segment_support=2,
        cv_max=1.5,
    )
    bundle = build_projection_bundle(pattern_spaces, list(SCALES), list(POLICIES))
    compact = compact_profile_from_bundle(pattern_spaces, bundle)
    full_run_dir = None
    if spec["kind"] == "observed":
        full_run_dir = job_dir / "full_phase1"
        write_full_phase1_artifacts(
            full_run_dir, spec=spec, pattern_spaces=pattern_spaces, bundle=bundle
        )
        assert_vectors_identical(
            compact["tail_vector"], tail_vector_from_full_run(full_run_dir)
        )

    profile = {
        "identity": spec["identity"],
        **compact,
        "full_phase1_run_dir": None
        if full_run_dir is None
        else str(full_run_dir.resolve()),
    }
    profile_path = job_dir / "compact_profile.json"
    write_json_atomic(profile_path, profile)
    record = {
        "job_id": spec["job_id"],
        "source": spec["source"],
        "iteration": int(spec["iteration"]),
        "kind": spec["kind"],
        "family": spec.get("family"),
        "seed": spec.get("seed"),
        "status": "completed",
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "peak_rss_bytes": current_rss_bytes(),
        "reused": False,
        "identity": spec["identity"],
        "profile_path": str(profile_path.resolve()),
        "profile_sha256": sha256_file(profile_path),
        "tail_vector": compact["tail_vector"],
        "full_phase1_run_dir": None
        if full_run_dir is None
        else str(full_run_dir.resolve()),
    }
    write_json_atomic(job_dir / "job_status.json", record)
    return record


def validate_profile_job(spec: dict[str, Any]) -> dict[str, Any] | None:
    try:
        record = json.loads(
            (Path(spec["job_dir"]) / "job_status.json").read_text(encoding="utf-8")
        )
        if (
            record.get("status") != "completed"
            or record.get("identity") != spec["identity"]
        ):
            return None
        profile_path = Path(record["profile_path"])
        if sha256_file(profile_path) != record["profile_sha256"]:
            return None
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        if profile.get("identity") != spec["identity"]:
            return None
        if tail_vector_from_compact(profile) != record.get("tail_vector"):
            return None
        if spec["kind"] == "observed":
            run_dir = Path(record["full_phase1_run_dir"])
            assert_vectors_identical(
                record["tail_vector"], tail_vector_from_full_run(run_dir)
            )
        return record
    except (
        FileNotFoundError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return None
