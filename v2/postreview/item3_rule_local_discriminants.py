from __future__ import annotations

import gzip
import json
import math
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from v2.common.io import load_observable_cache, split_into_segments
from v2.common.null_models import generate_control
from v2.phase1.tower import build_pattern_spaces, build_projection_bundle
from v2.postreview.item2b_growth_shells import (
    ObservableCursor,
    RunLogger,
    V33ObservableReader,
    current_rss_bytes,
    freeze_cache,
    sha256_file,
    source_sequence_length,
    write_json_atomic,
)


PREREG_SHA256 = "3FBCB935D6FB2835F839EABD0BA8367050D7DDDE4F938AEDDE04E9F74776F3EB"
NULL_IMPLEMENTATION_SHA256 = (
    "C367761E6BB7203F2E53D29B7F65170062BB58927E79D780120CD352C8088A73"
)
SOURCE_SHA256 = {
    "B": "B1FE25ACB1729C4862BE843559696206896402E3698E5F2F671F3E37795CA3D8",
    "BXOR": "AA9EF4FDBCCE59D3892D29C6A66F7B2735C6FC84718258CC6AD084A0C465DD44",
}
SITE_FRACTIONS = (("1/10", 1, 10), ("1/2", 1, 2), ("9/10", 9, 10))
WINDOW_BITS = 3_000_000
SEGMENT_BITS = 1_000_000
SEGMENT_COUNT = 3
SCALES = (8, 12, 16, 20, 24, 28, 32)
POLICIES = ("prefix", "suffix")
NULL_FAMILIES = ("markov1", "phase-matched-lz")
SEEDS = (17, 101, 211, 307, 401, 503)
BOOTSTRAP_REPLICATES = 10_000
EFFECT_FLOOR = 0.02


@dataclass(frozen=True)
class SiteSpec:
    source: str
    q: str
    start: int
    observable_length: int

    @property
    def end(self) -> int:
        return self.start + WINDOW_BITS

    @property
    def slug(self) -> str:
        return f"{self.source.lower()}_q{self.q.replace('/', 'of')}"


def frozen_protocol() -> dict[str, Any]:
    return {
        "segment_bits": SEGMENT_BITS,
        "num_segments": SEGMENT_COUNT,
        "scales": list(SCALES),
        "policies": list(POLICIES),
        "min_count_floor": 16,
        "min_count_rate": 1e-6,
        "min_segment_support": 2,
        "cv_max": 1.5,
        "site_fractions": [item[0] for item in SITE_FRACTIONS],
        "null_families": list(NULL_FAMILIES),
        "seeds": list(SEEDS),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "effect_floor": EFFECT_FLOOR,
        "bootstrap_method": "numpy_quantile_linear",
    }


def derive_site_specs(source: str, observable_length: int) -> list[SiteSpec]:
    if observable_length <= WINDOW_BITS:
        raise ValueError(f"{source} observable length is too short for a frozen site.")
    specs = [
        SiteSpec(
            source=source,
            q=q,
            start=(numerator * (observable_length - WINDOW_BITS)) // denominator,
            observable_length=observable_length,
        )
        for q, numerator, denominator in SITE_FRACTIONS
    ]
    for spec in specs:
        if spec.start < WINDOW_BITS:
            raise ValueError(f"{spec.slug} intersects the previously observed prefix.")
        if spec.end > observable_length:
            raise ValueError(f"{spec.slug} lies outside the observable stream.")
    for left, right in zip(specs, specs[1:]):
        if left.end > right.start:
            raise ValueError(f"Frozen sites overlap for {source}: {left.q}, {right.q}.")
    return specs


def count_observable_bits(
    path: Path,
    *,
    raw_chunk_bytes: int,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, int]:
    """Count 0/1 symbols in a packed v33 stream without materializing it."""
    sequence_length = source_sequence_length(path)
    packed_expected = (sequence_length + 3) // 4
    full_bytes = sequence_length // 4
    remainder = sequence_length % 4
    count_lut = np.fromiter(
        (
            sum(1 for shift in (6, 4, 2, 0) if ((value >> shift) & 0x03) <= 1)
            for value in range(256)
        ),
        dtype=np.uint8,
        count=256,
    )
    packed_read = 0
    observable = 0
    peak_rss = current_rss_bytes()
    with gzip.open(path, "rb") as handle:
        while packed_read < packed_expected:
            raw = handle.read(min(raw_chunk_bytes, packed_expected - packed_read))
            if not raw:
                raise RuntimeError(f"Unexpected EOF while counting {path}.")
            source = np.frombuffer(raw, dtype=np.uint8)
            start = packed_read
            stop = packed_read + source.size
            full_stop = min(stop, full_bytes)
            if full_stop > start:
                observable += int(
                    count_lut[source[: full_stop - start]].sum(dtype=np.uint64)
                )
            if remainder and start <= full_bytes < stop:
                value = int(source[full_bytes - start])
                observable += sum(
                    1
                    for index in range(remainder)
                    if ((value >> (6 - 2 * index)) & 0x03) <= 1
                )
            packed_read = stop
            peak_rss = max(peak_rss, current_rss_bytes())
            if progress is not None:
                progress(packed_read, packed_expected)
        if handle.read(1):
            raise RuntimeError(
                f"Unexpected packed bytes after sequence_length in {path}."
            )
    return {
        "sequence_length": int(sequence_length),
        "observable_length": int(observable),
        "packed_bytes_read": int(packed_read),
        "padding_symbols_excluded": int((4 - remainder) % 4),
        "peak_rss_bytes": int(peak_rss),
    }


def extract_observed_sites(
    path: Path,
    *,
    source: str,
    source_sha256: str,
    count_telemetry: dict[str, int],
    cache_dir: Path,
    raw_chunk_bytes: int,
    logger: RunLogger,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    specs = derive_site_specs(source, int(count_telemetry["observable_length"]))
    last_progress = 0.0

    def progress(reader: V33ObservableReader) -> None:
        nonlocal last_progress
        now = time.monotonic()
        if now - last_progress < 30.0:
            return
        last_progress = now
        logger.emit(
            "progress",
            force=True,
            phase="stage0-extract-observed",
            source=source,
            packed_bytes=f"{reader.packed_bytes_read}/{reader.packed_bytes_expected}",
            observable_bits=reader.observable_bits_decoded,
            rss_bytes=reader.peak_rss_bytes,
        )

    rows: list[dict[str, Any]] = []
    with V33ObservableReader(
        path,
        sequence_length=int(count_telemetry["sequence_length"]),
        expected_observable_bits=int(count_telemetry["observable_length"]),
        raw_chunk_bytes=raw_chunk_bytes,
        progress=progress,
    ) as reader:
        cursor = ObservableCursor(reader)
        for spec in specs:
            cursor.discard(spec.start - cursor.position)
            bits = cursor.read_exact(WINDOW_BITS)
            cache_path = cache_dir / "observed" / f"{spec.slug}.npy"
            identity = {
                "cache_kind": "observed",
                "source": source,
                "site_q": spec.q,
                "site_start": spec.start,
                "site_end": spec.end,
                "source_path": str(path.resolve()),
                "source_sha256": source_sha256,
                "source_observable_length": spec.observable_length,
                "preregistration_sha256": PREREG_SHA256,
            }
            sidecar = freeze_identity_cache(cache_path, bits, identity=identity)
            rows.append(sidecar)
        cursor.drain()
        telemetry = {
            **count_telemetry,
            "decode_packed_bytes_read": reader.packed_bytes_read,
            "decode_observable_bits": reader.observable_bits_decoded,
            "decode_peak_rss_bytes": reader.peak_rss_bytes,
        }
    return rows, telemetry


def freeze_identity_cache(
    cache_path: Path, bits: np.ndarray, *, identity: dict[str, Any]
) -> dict[str, Any]:
    existing = validate_identity_cache(
        cache_path, identity=identity, expected_bits=int(bits.size)
    )
    if existing is not None:
        return {**existing, "reused": True}
    quarantine_cache(cache_path)
    return {**freeze_cache(cache_path, bits, metadata=identity), "reused": False}


def validate_identity_cache(
    cache_path: Path,
    *,
    identity: dict[str, Any],
    expected_bits: int = WINDOW_BITS,
) -> dict[str, Any] | None:
    try:
        _, sidecar, _ = load_observable_cache(cache_path, expected_bits=expected_bits)
        if any(sidecar.get(key) != value for key, value in identity.items()):
            return None
        return sidecar
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
        return None


def quarantine_cache(cache_path: Path) -> None:
    sidecar_path = cache_path.with_suffix(".json")
    existing = [path for path in (cache_path, sidecar_path) if path.exists()]
    if not existing:
        return
    quarantine = (
        cache_path.parent / "quarantine" / datetime.now().strftime("%Y%m%dT%H%M%S%f")
    )
    quarantine.mkdir(parents=True, exist_ok=True)
    for path in existing:
        shutil.move(str(path), str(quarantine / path.name))


def null_cache_identity(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "cache_kind": "null",
        "source": spec["source"],
        "site_q": spec["q"],
        "site_start": int(spec["site_start"]),
        "observed_cache_sha256": spec["observed_cache_sha256"],
        "null_family": spec["family"],
        "null_seed": int(spec["seed"]),
        "null_implementation_sha256": NULL_IMPLEMENTATION_SHA256,
        "preregistration_sha256": PREREG_SHA256,
    }


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
            "sidecar": existing,
        }
    bits, observed_sidecar, _ = load_observable_cache(
        Path(spec["observed_cache_path"]), expected_bits=WINDOW_BITS
    )
    if observed_sidecar["cache_sha256"] != spec["observed_cache_sha256"]:
        raise RuntimeError(f"Observed cache identity mismatch for {spec['job_id']}.")
    generated, fit_metadata = generate_control(bits, spec["family"], int(spec["seed"]))
    metadata = {
        **identity,
        "fit_metadata": fit_metadata or markov1_fit_metadata(bits),
    }
    quarantine_cache(cache_path)
    sidecar = freeze_cache(cache_path, generated, metadata=metadata)
    return {
        "job_id": spec["job_id"],
        "status": "completed",
        "reused": False,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "peak_rss_bytes": current_rss_bytes(),
        "sidecar": sidecar,
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
    if family == "phase-matched-lz":
        return bool(
            fit.get("method") == "phase_class_block_shuffle_lz_match"
            and fit.get("candidate_block_bits")
            == [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
            and fit.get("candidate_phase_period_blocks") == [2, 3, 5, 7]
            and fit.get("window_bits") == 32_768
            and len(fit.get("window_starts", [])) == 3
            and isinstance(fit.get("lz_abs_error"), (int, float))
        )
    return False


def markov1_fit_metadata(bits: np.ndarray) -> dict[str, Any]:
    density = float(bits.mean()) if bits.size else 0.0
    counts = np.zeros((2, 2), dtype=np.int64)
    if bits.size > 1:
        codes = (bits[:-1] << 1) | bits[1:]
        counts = np.bincount(codes, minlength=4).reshape(2, 2)
    probs = np.empty((2, 2), dtype=np.float64)
    for state in (0, 1):
        total = int(counts[state].sum())
        if total:
            probs[state] = counts[state] / total
        else:
            probs[state] = (1.0 - density, density)
    return {
        "method": "maximum_likelihood_first_order_binary_markov",
        "target_density": density,
        "transition_counts": counts.tolist(),
        "transition_probabilities": probs.tolist(),
        "initial_state_probability_one": density,
    }


def profile_job_identity(
    spec: dict[str, Any], *, code_hashes: dict[str, str], stage0_sha256: str
) -> dict[str, Any]:
    return {
        "job_id": spec["job_id"],
        "source": spec["source"],
        "q": spec["q"],
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
    status_path = job_dir / "job_status.json"
    profile_path = job_dir / "job_profile.json"
    existing = validate_profile_job(spec)
    if existing is not None:
        return {**existing, "reused": True}
    if job_dir.exists():
        quarantine = (
            job_dir.parent / "quarantine" / datetime.now().strftime("%Y%m%dT%H%M%S%f")
        )
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(job_dir), str(quarantine))
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
    profile = {
        "identity": spec["identity"],
        "segments": [
            build_segment_profile(bundle, index) for index in range(SEGMENT_COUNT)
        ],
    }
    write_json_atomic(profile_path, profile)
    record = {
        "job_id": spec["job_id"],
        "source": spec["source"],
        "q": spec["q"],
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
    }
    write_json_atomic(status_path, record)
    return record


def validate_profile_job(spec: dict[str, Any]) -> dict[str, Any] | None:
    job_dir = Path(spec["job_dir"])
    status_path = job_dir / "job_status.json"
    try:
        record = json.loads(status_path.read_text(encoding="utf-8"))
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
        return record
    except (FileNotFoundError, OSError, KeyError, ValueError, json.JSONDecodeError):
        return None


def build_segment_profile(bundle: dict[str, Any], segment_index: int) -> dict[str, Any]:
    policies: dict[str, Any] = {}
    for policy in POLICIES:
        edges = bundle["fibers"][policy]
        policies[policy] = {
            "edges": [
                build_edge_profile(edge, segment_index, policy) for edge in edges
            ],
            "transport_steps": build_transport_steps(edges, segment_index),
        }
    return {"segment_index": segment_index, "policies": policies}


def build_edge_profile(
    edge: dict[str, Any], segment_index: int, policy: str
) -> dict[str, Any]:
    degree_mass = np.zeros(17, dtype=np.float64)
    entropy_values: list[float] = []
    entropy_weights: list[float] = []
    extension_mass = np.zeros(16, dtype=np.float64)
    parent_mass_total = 0.0

    for parent in edge.get("parents", []):
        parent_mass = float(parent["segment_counts"][segment_index])
        if parent_mass <= 0:
            continue
        child_counts = np.asarray(
            [
                child["segment_counts"][segment_index]
                for child in parent.get("children", [])
            ],
            dtype=np.float64,
        )
        active = child_counts > 0
        degree = int(active.sum())
        if degree > 16:
            raise RuntimeError(
                f"Fiber degree exceeds fixed four-bit alphabet: {degree}."
            )
        degree_mass[degree] += parent_mass
        parent_mass_total += parent_mass
        entropy = mass_entropy(child_counts[active])
        normalized_entropy = 0.0 if degree <= 1 else entropy / math.log2(degree)
        entropy_values.append(float(normalized_entropy))
        entropy_weights.append(parent_mass)

        for child, child_mass in zip(parent.get("children", []), child_counts):
            if child_mass <= 0:
                continue
            motif = extension_word(parent["pattern"], child["pattern"], policy)
            extension_mass[int(motif, 2)] += float(child_mass)

    degree_distribution = normalize_mass(degree_mass)
    extension_distribution = normalize_mass(extension_mass)
    entropy_weight_array = np.asarray(entropy_weights, dtype=np.float64)
    entropy_weight_array = normalize_mass(entropy_weight_array)
    return {
        "target_m": int(edge["target_m"]),
        "source_m": int(edge["source_m"]),
        "parent_mass_total": parent_mass_total,
        "degree_distribution": degree_distribution.tolist(),
        "sibling_entropy_values": entropy_values,
        "sibling_entropy_weights": entropy_weight_array.tolist(),
        "extension_distribution": extension_distribution.tolist(),
        "extension_mass_total": float(extension_mass.sum()),
    }


def build_transport_steps(
    edges: list[dict[str, Any]], segment_index: int
) -> list[dict[str, Any]]:
    lookup = {int(edge["target_m"]): edge for edge in edges}
    steps = []
    for target_m in sorted(lookup)[:-1]:
        current = lookup[target_m]
        following = lookup.get(int(current["source_m"]))
        if following is not None:
            steps.append(build_transport_step(current, following, segment_index))
    return steps


def build_transport_step(
    current: dict[str, Any], following: dict[str, Any], segment_index: int
) -> dict[str, Any]:
    next_entropy: dict[str, float] = {}
    next_active: dict[str, float] = {}
    for parent in following.get("parents", []):
        counts = np.asarray(
            [
                child["segment_counts"][segment_index]
                for child in parent.get("children", [])
            ],
            dtype=np.float64,
        )
        counts = counts[counts > 0]
        next_entropy[parent["pattern"]] = mass_entropy(counts)
        next_active[parent["pattern"]] = 1.0 if counts.size > 1 else 0.0

    total_parent_mass = 0.0
    transport_entropy = 0.0
    transport_active = 0.0
    for parent in current.get("parents", []):
        parent_mass = float(parent["segment_counts"][segment_index])
        if parent_mass <= 0:
            continue
        child_rows = []
        child_total = 0.0
        for child in parent.get("children", []):
            child_mass = float(child["segment_counts"][segment_index])
            if child_mass > 0:
                child_rows.append((child["pattern"], child_mass))
                child_total += child_mass
        total_parent_mass += parent_mass
        if child_total <= 0:
            continue
        inherited_entropy = sum(
            (mass / child_total) * next_entropy.get(pattern, 0.0)
            for pattern, mass in child_rows
        )
        inherited_active = sum(
            (mass / child_total) * next_active.get(pattern, 0.0)
            for pattern, mass in child_rows
        )
        transport_entropy += parent_mass * inherited_entropy
        transport_active += parent_mass * inherited_active
    if total_parent_mass > 0:
        transport_entropy /= total_parent_mass
        transport_active /= total_parent_mass
    return {
        "target_m": int(current["target_m"]),
        "source_m": int(current["source_m"]),
        "transport_active_mass": float(transport_active),
        "transport_entropy_over_four": float(transport_entropy / 4.0),
        "parent_mass_total": float(total_parent_mass),
    }


def extension_word(parent: str, child: str, policy: str) -> str:
    if len(child) - len(parent) != 4:
        raise ValueError("Item 3 requires an exact four-bit scale gap.")
    if policy == "prefix":
        if not child.startswith(parent):
            raise ValueError("Prefix child does not contain its parent prefix.")
        return child[-4:]
    if policy == "suffix":
        if not child.endswith(parent):
            raise ValueError("Suffix child does not contain its parent suffix.")
        return child[:4]
    raise ValueError(f"Unsupported policy: {policy}")


def mass_entropy(masses: np.ndarray) -> float:
    masses = np.asarray(masses, dtype=np.float64)
    total = float(masses.sum())
    if total <= 0:
        return 0.0
    probabilities = masses[masses > 0] / total
    return float(-(probabilities * np.log2(probabilities)).sum())


def normalize_mass(masses: np.ndarray) -> np.ndarray:
    masses = np.asarray(masses, dtype=np.float64)
    total = float(masses.sum())
    if total <= 0:
        return np.zeros_like(masses, dtype=np.float64)
    return masses / total


def sqrt_js_distance(
    left: list[float] | np.ndarray, right: list[float] | np.ndarray
) -> float:
    p = normalize_mass(np.asarray(left, dtype=np.float64))
    q = normalize_mass(np.asarray(right, dtype=np.float64))
    if p.shape != q.shape:
        raise ValueError("Jensen-Shannon inputs must share a shape.")
    if p.sum() <= 0 or q.sum() <= 0:
        return float("nan")
    midpoint = 0.5 * (p + q)
    p_mask = p > 0
    q_mask = q > 0
    divergence = 0.5 * float(np.sum(p[p_mask] * np.log2(p[p_mask] / midpoint[p_mask])))
    divergence += 0.5 * float(np.sum(q[q_mask] * np.log2(q[q_mask] / midpoint[q_mask])))
    return float(math.sqrt(max(0.0, divergence)))


def weighted_wasserstein_1(
    left_values: list[float],
    left_weights: list[float],
    right_values: list[float],
    right_weights: list[float],
) -> float:
    x = np.asarray(left_values, dtype=np.float64)
    wx = normalize_mass(np.asarray(left_weights, dtype=np.float64))
    y = np.asarray(right_values, dtype=np.float64)
    wy = normalize_mass(np.asarray(right_weights, dtype=np.float64))
    if x.size == 0 or y.size == 0 or wx.sum() <= 0 or wy.sum() <= 0:
        return float("nan")
    order_x = np.argsort(x, kind="mergesort")
    order_y = np.argsort(y, kind="mergesort")
    x, wx = x[order_x], wx[order_x]
    y, wy = y[order_y], wy[order_y]
    points = np.unique(np.concatenate((x, y)))
    if points.size <= 1:
        return 0.0
    cdf_x = np.searchsorted(x, points[:-1], side="right")
    cdf_y = np.searchsorted(y, points[:-1], side="right")
    cum_x = np.concatenate(([0.0], np.cumsum(wx)))
    cum_y = np.concatenate(([0.0], np.cumsum(wy)))
    return float(np.sum(np.abs(cum_x[cdf_x] - cum_y[cdf_y]) * np.diff(points)))


def compare_segment_profiles(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    left_label: str = "left",
    right_label: str = "right",
) -> dict[str, Any]:
    edge_rows = []
    motif_rows = []
    orientation_diagnostics: dict[str, Any] = {}
    for policy in POLICIES:
        left_edges = left["policies"][policy]["edges"]
        right_edges = right["policies"][policy]["edges"]
        if len(left_edges) != 6 or len(right_edges) != 6:
            raise ValueError("Every Item 3 profile must contain six projective edges.")
        for left_edge, right_edge in zip(left_edges, right_edges):
            require_same_edge(left_edge, right_edge)
            degree = sqrt_js_distance(
                left_edge["degree_distribution"], right_edge["degree_distribution"]
            )
            entropy = weighted_wasserstein_1(
                left_edge["sibling_entropy_values"],
                left_edge["sibling_entropy_weights"],
                right_edge["sibling_entropy_values"],
                right_edge["sibling_entropy_weights"],
            )
            motif = sqrt_js_distance(
                left_edge["extension_distribution"],
                right_edge["extension_distribution"],
            )
            edge_rows.append(
                {
                    "policy": policy,
                    "target_m": left_edge["target_m"],
                    "source_m": left_edge["source_m"],
                    "degree_distance": degree,
                    "sibling_entropy_distance": entropy,
                }
            )
            motif_rows.append(
                {
                    "policy": policy,
                    "target_m": left_edge["target_m"],
                    "source_m": left_edge["source_m"],
                    "distance": motif,
                }
            )

    orientation_scores: dict[str, list[float]] = {left_label: [], right_label: []}
    for label, profile in ((left_label, left), (right_label, right)):
        prefix_edges = profile["policies"]["prefix"]["edges"]
        suffix_edges = profile["policies"]["suffix"]["edges"]
        motif_orientation = []
        for prefix, suffix in zip(prefix_edges, suffix_edges):
            require_same_edge(prefix, suffix)
            degree = sqrt_js_distance(
                prefix["degree_distribution"], suffix["degree_distribution"]
            )
            entropy = weighted_wasserstein_1(
                prefix["sibling_entropy_values"],
                prefix["sibling_entropy_weights"],
                suffix["sibling_entropy_values"],
                suffix["sibling_entropy_weights"],
            )
            orientation_scores[label].append(float((degree + entropy) / 2.0))
            motif_orientation.append(
                sqrt_js_distance(
                    prefix["extension_distribution"],
                    reverse_motif_distribution(suffix["extension_distribution"]),
                )
            )
        orientation_diagnostics[label] = {
            "fiber_scores": orientation_scores[label],
            "motif_reverse_scores": motif_orientation,
        }

    d1_edges = []
    for edge_index in range(6):
        prefix = edge_rows[edge_index]
        suffix = edge_rows[6 + edge_index]
        orientation_gap = abs(
            orientation_scores[left_label][edge_index]
            - orientation_scores[right_label][edge_index]
        )
        components = [
            prefix["degree_distance"],
            suffix["degree_distance"],
            prefix["sibling_entropy_distance"],
            suffix["sibling_entropy_distance"],
            orientation_gap,
        ]
        d1_edges.append(
            {
                "target_m": prefix["target_m"],
                "source_m": prefix["source_m"],
                "components": components,
                "distance": float(np.mean(components)),
            }
        )
    d1 = float(np.mean([row["distance"] for row in d1_edges]))
    d2 = float(np.mean([row["distance"] for row in motif_rows]))
    return {
        "D1": d1,
        "D2": d2,
        "fiber_edges": d1_edges,
        "fiber_components": edge_rows,
        "motif_components": motif_rows,
        "orientation_diagnostics": orientation_diagnostics,
    }


def reverse_motif_distribution(distribution: list[float]) -> list[float]:
    values = list(distribution)
    if len(values) != 16:
        raise ValueError("The extension motif alphabet must contain 16 words.")
    output = [0.0] * 16
    for index, mass in enumerate(values):
        reversed_index = int(f"{index:04b}"[::-1], 2)
        output[reversed_index] = float(mass)
    return output


def require_same_edge(left: dict[str, Any], right: dict[str, Any]) -> None:
    if (left["target_m"], left["source_m"]) != (right["target_m"], right["source_m"]):
        raise ValueError("Profile edge identity mismatch.")


def load_job_profiles(
    records: list[dict[str, Any]],
) -> dict[tuple[str, str, str, str | None, int | None], dict]:
    profiles = {}
    for record in records:
        key = (
            record["source"],
            record["q"],
            record["kind"],
            record.get("family"),
            record.get("seed"),
        )
        if key in profiles:
            raise ValueError(f"Duplicate profile key: {key}.")
        profiles[key] = json.loads(
            Path(record["profile_path"]).read_text(encoding="utf-8")
        )
    if len(profiles) != 78:
        raise ValueError(f"Expected 78 complete profiles, got {len(profiles)}.")
    return profiles


def build_item3_readout(records: list[dict[str, Any]]) -> dict[str, Any]:
    profiles = load_job_profiles(records)
    per_segment: list[dict[str, Any]] = []
    profile_archive: dict[str, Any] = {"observed": {}, "null_pairs": {}}
    routing_archive: dict[str, Any] = {}
    degenerate: list[dict[str, Any]] = []

    for site_index, (q, _, _) in enumerate(SITE_FRACTIONS):
        observed_b = profiles[("B", q, "observed", None, None)]
        observed_x = profiles[("BXOR", q, "observed", None, None)]
        profile_archive["observed"][q] = []
        profile_archive["null_pairs"][q] = {}
        null_comparisons: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for segment_index in range(SEGMENT_COUNT):
            comparison = compare_segment_profiles(
                observed_b["segments"][segment_index],
                observed_x["segments"][segment_index],
                left_label="B",
                right_label="BXOR",
            )
            profile_archive["observed"][q].append(comparison)
            if not finite_primary(comparison):
                degenerate.append(
                    {"site": q, "segment": segment_index, "kind": "observed"}
                )

        for family in NULL_FAMILIES:
            for seed in SEEDS:
                pair_rows = []
                left = profiles[("B", q, "null", family, seed)]
                right = profiles[("BXOR", q, "null", family, seed)]
                for segment_index in range(SEGMENT_COUNT):
                    comparison = compare_segment_profiles(
                        left["segments"][segment_index],
                        right["segments"][segment_index],
                        left_label="B",
                        right_label="BXOR",
                    )
                    pair_rows.append(comparison)
                    if not finite_primary(comparison):
                        degenerate.append(
                            {
                                "site": q,
                                "segment": segment_index,
                                "kind": "null",
                                "family": family,
                                "seed": seed,
                            }
                        )
                null_comparisons[(family, seed)] = pair_rows
                profile_archive["null_pairs"][q][f"{family}__seed-{seed}"] = pair_rows

        d3_rows, d3_archive = build_routing_residuals(profiles, q)
        routing_archive[q] = d3_archive
        for segment_index in range(SEGMENT_COUNT):
            observed = profile_archive["observed"][q][segment_index]
            row: dict[str, Any] = {
                "site_index": site_index,
                "q": q,
                "segment_index": segment_index,
                "D1_obs": observed["D1"],
                "D2_obs": observed["D2"],
                "D3": d3_rows[segment_index],
            }
            for probe in (1, 2):
                null_values = {
                    f"{family}__seed-{seed}": rows[segment_index][f"D{probe}"]
                    for (family, seed), rows in null_comparisons.items()
                }
                envelope = max(null_values.values())
                row[f"E{probe}"] = envelope
                row[f"G{probe}"] = row[f"D{probe}_obs"] - envelope
                row[f"D{probe}_null_pairs"] = null_values
            per_segment.append(row)

    per_site = build_site_decisions(per_segment)
    region_positive = {
        f"probe{probe}": region_is_positive(per_site, probe) for probe in (1, 2)
    }
    outcome = classify_item3(degenerate, per_site, region_positive)
    summary = {
        "status": "completed",
        "outcome": outcome,
        "degenerate_components": degenerate,
        "per_site": per_site,
        "region_positive": region_positive,
        "probe3_qualifier": {
            "decision_power": False,
            "site_medians": {
                q: median([row["D3"] for row in per_segment if row["q"] == q])
                for q, _, _ in SITE_FRACTIONS
            },
        },
        "bootstrap_boundary": (
            "The deterministic n=3 within-site bootstrap is a coarse sign/effect stability guard, "
            "not population, cross-site, or asymptotic inference."
        ),
        "wording_boundary": (
            "The discriminant measures what B_XOR preserves and what it loses; no outcome ranks B above B_XOR."
        ),
    }
    return {
        "summary": summary,
        "per_segment": per_segment,
        "per_site": per_site,
        "fiber_archive": {
            "profiles": archive_profiles(profiles, component="fiber"),
            "pairwise_comparisons": profile_archive,
        },
        "motif_archive": {
            "profiles": archive_profiles(profiles, component="motif"),
            "pairwise_comparisons": extract_nested_component(
                profile_archive, "motif_components"
            ),
        },
        "routing_archive": routing_archive,
    }


def archive_profiles(
    profiles: dict[tuple[str, str, str, str | None, int | None], dict],
    *,
    component: str,
) -> dict[str, Any]:
    archive = {}
    for key in sorted(
        profiles,
        key=lambda item: tuple("" if value is None else str(value) for value in item),
    ):
        source, q, kind, family, seed = key
        label = f"{source}__q-{q.replace('/', 'of')}__{kind}"
        if family is not None:
            label += f"__{family}__seed-{seed}"
        segment_rows = []
        for segment in profiles[key]["segments"]:
            policy_rows = {}
            for policy in POLICIES:
                if component == "fiber":
                    policy_rows[policy] = [
                        {
                            "target_m": edge["target_m"],
                            "source_m": edge["source_m"],
                            "parent_mass_total": edge["parent_mass_total"],
                            "degree_distribution": edge["degree_distribution"],
                            "sibling_entropy_values": edge["sibling_entropy_values"],
                            "sibling_entropy_weights": edge["sibling_entropy_weights"],
                        }
                        for edge in segment["policies"][policy]["edges"]
                    ]
                elif component == "motif":
                    policy_rows[policy] = [
                        {
                            "target_m": edge["target_m"],
                            "source_m": edge["source_m"],
                            "extension_distribution": edge["extension_distribution"],
                            "extension_mass_total": edge["extension_mass_total"],
                        }
                        for edge in segment["policies"][policy]["edges"]
                    ]
                else:
                    raise ValueError(
                        f"Unsupported profile archive component: {component}"
                    )
            segment_rows.append(
                {"segment_index": segment["segment_index"], "policies": policy_rows}
            )
        archive[label] = segment_rows
    return archive


def extract_nested_component(payload: Any, key: str) -> Any:
    if isinstance(payload, dict):
        if key in payload:
            return payload[key]
        return {
            name: extract_nested_component(value, key)
            for name, value in payload.items()
        }
    if isinstance(payload, list):
        return [extract_nested_component(value, key) for value in payload]
    return payload


def finite_primary(comparison: dict[str, Any]) -> bool:
    return math.isfinite(float(comparison["D1"])) and math.isfinite(
        float(comparison["D2"])
    )


def build_site_decisions(per_segment: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for site_index, (q, _, _) in enumerate(SITE_FRACTIONS):
        segments = sorted(
            (row for row in per_segment if row["q"] == q),
            key=lambda row: row["segment_index"],
        )
        row: dict[str, Any] = {"site_index": site_index, "q": q}
        for probe in (1, 2):
            observed = [float(item[f"D{probe}_obs"]) for item in segments]
            gaps = [float(item[f"G{probe}"]) for item in segments]
            seed = 20260828 + 100 * probe + site_index
            observed_ci = bootstrap_median_ci(observed, seed=seed)
            gap_ci = bootstrap_median_ci(gaps, seed=seed)
            observed_median = median(observed)
            gap_median = median(gaps)
            row[f"probe{probe}"] = {
                "D_obs_median": observed_median,
                "D_obs_ci95": observed_ci,
                "G_median": gap_median,
                "G_ci95": gap_ci,
                "raw_positive": bool(
                    observed_median >= EFFECT_FLOOR and observed_ci[0] > 0
                ),
                "hard_null_positive": bool(
                    gap_median >= EFFECT_FLOOR and gap_ci[0] > 0
                ),
            }
        row["D3_median"] = median([float(item["D3"]) for item in segments])
        rows.append(row)
    return rows


def bootstrap_median_ci(values: list[float], *, seed: int) -> list[float]:
    sample = np.asarray(values, dtype=np.float64)
    if sample.shape != (3,) or not np.all(np.isfinite(sample)):
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, 3, size=(BOOTSTRAP_REPLICATES, 3))
    medians = np.median(sample[indices], axis=1)
    bounds = np.quantile(medians, (0.025, 0.975), method="linear")
    return [float(bounds[0]), float(bounds[1])]


def median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def region_is_positive(per_site: list[dict[str, Any]], probe: int) -> bool:
    rows = [row[f"probe{probe}"] for row in per_site]
    return bool(
        sum(bool(row["hard_null_positive"]) for row in rows) >= 2
        and all(float(row["G_median"]) > -EFFECT_FLOOR for row in rows)
    )


def classify_item3(
    degenerate: list[dict[str, Any]],
    per_site: list[dict[str, Any]],
    region_positive: dict[str, bool],
) -> str:
    if degenerate:
        return "projective-readout-degenerate"
    positive_count = sum(bool(value) for value in region_positive.values())
    if positive_count == 2:
        return "rule-local-discriminant-two-probe-consensus"
    if positive_count == 1:
        return "mixed-rule-local-signal"
    raw_counts = {
        probe: sum(bool(row[f"probe{probe}"]["raw_positive"]) for row in per_site)
        for probe in (1, 2)
    }
    if any(count >= 2 for count in raw_counts.values()):
        return "source-statistics-or-null-mimicry"
    return "no-tested-rule-local-separation"


def build_routing_residuals(
    profiles: dict[tuple[str, str, str, str | None, int | None], dict],
    q: str,
) -> tuple[list[float], dict[str, Any]]:
    archive: dict[str, Any] = {}
    segment_distances: list[float] = []
    for segment_index in range(SEGMENT_COUNT):
        family_distances = []
        archive[str(segment_index)] = {}
        for family in NULL_FAMILIES:
            source_residuals = {}
            for source in ("B", "BXOR"):
                observed = transport_matrix(
                    profiles[(source, q, "observed", None, None)]["segments"][
                        segment_index
                    ]
                )
                null_stack = np.stack(
                    [
                        transport_matrix(
                            profiles[(source, q, "null", family, seed)]["segments"][
                                segment_index
                            ]
                        )
                        for seed in SEEDS
                    ],
                    axis=0,
                )
                null_median = np.median(null_stack, axis=0)
                source_residuals[source] = observed - null_median
            distance = float(
                np.mean(np.abs(source_residuals["B"] - source_residuals["BXOR"]))
            )
            family_distances.append(distance)
            archive[str(segment_index)][family] = {
                "D3_family": distance,
                "B_residual": source_residuals["B"].tolist(),
                "BXOR_residual": source_residuals["BXOR"].tolist(),
            }
        segment_distances.append(float(np.mean(family_distances)))
    return segment_distances, archive


def transport_matrix(segment_profile: dict[str, Any]) -> np.ndarray:
    rows = []
    for policy in POLICIES:
        steps = segment_profile["policies"][policy]["transport_steps"]
        if len(steps) != 5:
            raise ValueError(
                "Every Item 3 transport profile must contain five steps per policy."
            )
        rows.extend(
            [
                float(step["transport_active_mass"]),
                float(step["transport_entropy_over_four"]),
            ]
            for step in steps
        )
    return np.asarray(rows, dtype=np.float64)
