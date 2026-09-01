from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from v2.common.io import load_observable_cache
from v2.postreview.item2b_growth_shells import (
    RunLogger,
    WindowSpec,
    classify_growth_shell_rows,
    freeze_or_reuse_cache,
    scan_nested_pair,
    sha256_file,
)


REPO_DIR = Path(__file__).resolve().parents[1]


def pack_v33(structural: str) -> bytes:
    mapping = {"0": 0, "1": 1, "(": 2, ")": 3}
    codes = [mapping[char] for char in structural]
    codes.extend([0] * ((4 - len(codes) % 4) % 4))
    packed = bytearray()
    for offset in range(0, len(codes), 4):
        a, b, c, d = codes[offset : offset + 4]
        packed.append((a << 6) | (b << 4) | (c << 2) | d)
    return bytes(packed)


def write_structural(path: Path, structural: str) -> None:
    with gzip.open(path, "wb", compresslevel=1) as handle:
        handle.write(pack_v33(structural))
    path.with_suffix("").with_suffix(".json").write_text(
        json.dumps({"sequence_length": len(structural), "iteration": 1}),
        encoding="utf-8",
    )


def decorate(bits: str) -> str:
    return "".join(f"({bit})" for bit in bits)


def test_nested_scan_extracts_exact_windows_and_excludes_padding(tmp_path: Path) -> None:
    parent_bits = "01011001"
    child_bits = parent_bits + "110010101101001011001010"
    parent_structural = decorate(parent_bits) + "("  # Non-multiple-of-four structural length.
    child_structural = decorate(child_bits) + "(("
    parent = tmp_path / "parent.struct.gz"
    child = tmp_path / "child.struct.gz"
    write_structural(parent, parent_structural)
    write_structural(child, child_structural)
    windows = (
        WindowSpec(20, "1/10", 10, 4),
        WindowSpec(20, "1/2", 16, 4),
        WindowSpec(20, "9/10", 24, 4),
    )

    extracted, telemetry = scan_nested_pair(
        parent,
        child,
        parent_sequence_length=len(parent_structural),
        child_sequence_length=len(child_structural),
        parent_observable_bits=len(parent_bits),
        child_observable_bits=len(child_bits),
        windows=windows,
        raw_chunk_bytes=1,
        logger=RunLogger(tmp_path / "run.log"),
    )

    for spec in windows:
        expected = np.asarray([int(value) for value in child_bits[spec.start : spec.end]], dtype=np.uint8)
        assert np.array_equal(extracted[spec.slug], expected)
    assert telemetry["prefix_bits_compared"] == len(parent_bits)
    assert telemetry["parent_padding_symbols_excluded"] > 0
    assert telemetry["child_padding_symbols_excluded"] > 0


def test_nested_scan_rejects_first_prefix_divergence(tmp_path: Path) -> None:
    parent_bits = "01011001"
    child_bits = "01111001" + "010101010101"
    parent = tmp_path / "parent.struct.gz"
    child = tmp_path / "child.struct.gz"
    write_structural(parent, decorate(parent_bits))
    write_structural(child, decorate(child_bits))
    with pytest.raises(RuntimeError, match="divergence at bit 2"):
        scan_nested_pair(
            parent,
            child,
            parent_sequence_length=len(decorate(parent_bits)),
            child_sequence_length=len(decorate(child_bits)),
            parent_observable_bits=len(parent_bits),
            child_observable_bits=len(child_bits),
            windows=(WindowSpec(20, "1/2", 10, 3),),
            raw_chunk_bytes=1,
            logger=RunLogger(tmp_path / "run.log"),
        )


def test_cache_is_verified_reused_and_mismatch_is_quarantined(tmp_path: Path) -> None:
    path = tmp_path / "window.npy"
    bits = np.asarray([0, 1, 1, 0, 1], dtype=np.uint8)
    metadata = {"source_sha256": "A" * 64, "observable_start": 10}
    first = freeze_or_reuse_cache(path, bits, metadata=metadata)
    assert not first["reused"]
    second = freeze_or_reuse_cache(path, bits.copy(), metadata=metadata)
    assert second["reused"]
    loaded, sidecar, _ = load_observable_cache(path, expected_bits=bits.size)
    assert np.array_equal(loaded, bits)
    assert sidecar["cache_sha256"] == sha256_file(path)

    changed = bits.copy()
    changed[0] = 1
    third = freeze_or_reuse_cache(path, changed, metadata=metadata)
    assert not third["reused"]
    assert any((tmp_path / "quarantine").rglob("window.npy"))


def rows_for_outcome(
    low_g: tuple[float, float, float],
    high_g: tuple[float, float, float],
    low_s: tuple[bool, bool, bool],
    high_s: tuple[bool, bool, bool],
) -> list[dict]:
    rows = []
    for depth, gaps, supports in ((20, low_g, low_s), (24, high_g, high_s)):
        for q, gap, support in zip(("1/10", "1/2", "9/10"), gaps, supports):
            rows.append({"depth": depth, "q": q, "G": gap, "D": 0.3, "S": support})
    return rows


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        (
            rows_for_outcome((0.4, 0.4, 0.4), (0.4, 0.4, 0.4), (True, False, False), (True, True, True)),
            "markov1-shell-baseline-not-established",
        ),
        (
            rows_for_outcome((0.4, 0.4, 0.4), (0.3, 0.3, 0.4), (True, True, True), (True, True, True)),
            "markov1-shell-separation-contracts-at-iter24",
        ),
        (
            rows_for_outcome((0.4, 0.4, 0.4), (0.5, 0.5, 0.4), (True, True, True), (True, True, True)),
            "markov1-shell-separation-expands-at-iter24",
        ),
        (
            rows_for_outcome((0.4, 0.4, 0.4), (0.41, 0.39, 0.4), (True, True, True), (True, True, True)),
            "markov1-shell-separation-stable-at-iter24",
        ),
        (
            rows_for_outcome((0.4, 0.4, 0.4), (0.5, 0.39, 0.3), (True, True, False), (True, True, False)),
            "markov1-shell-position-heterogeneous",
        ),
    ],
)
def test_outcome_hierarchy_is_first_match_and_exhaustive(rows: list[dict], expected: str) -> None:
    assert classify_growth_shell_rows(rows)["outcome"] == expected


def test_phase1_cache_mode_is_scientifically_identical_to_structural_input(tmp_path: Path) -> None:
    bits = np.asarray(([0, 1, 0, 1, 1, 0] * 50), dtype=np.uint8)
    structural = "".join(str(int(value)) for value in bits)
    source = tmp_path / "fixture.struct.gz"
    write_structural(source, structural)

    cache = tmp_path / "fixture.npy"
    with open(cache, "wb") as handle:
        np.save(handle, bits, allow_pickle=False)
    cache.with_suffix(".json").write_text(
        json.dumps({"observable_length": int(bits.size), "cache_sha256": sha256_file(cache)}),
        encoding="utf-8",
    )

    common = [
        "--variant",
        "B",
        "--iteration",
        "1",
        "--segment-bits",
        "100",
        "--num-segments",
        "3",
        "--scales",
        "8,12",
        "--policies",
        "prefix,suffix",
        "--min-count-floor",
        "1",
        "--min-count-rate",
        "0",
        "--min-segment-support",
        "1",
        "--cv-max",
        "99",
        "--quiet",
    ]
    structural_out = tmp_path / "structural-out"
    cache_out = tmp_path / "cache-out"
    structural_run = subprocess.run(
        [
            sys.executable,
            str(REPO_DIR / "hsi_v2_phase1_run.py"),
            *common,
            "--input",
            str(source),
            "--output-dir",
            str(structural_out),
        ],
        cwd=REPO_DIR,
        capture_output=True,
        text=True,
    )
    assert structural_run.returncode == 0, structural_run.stderr
    cache_run = subprocess.run(
        [
            sys.executable,
            str(REPO_DIR / "hsi_v2_phase1_run.py"),
            *common,
            "--observable-cache",
            str(cache),
            "--output-dir",
            str(cache_out),
        ],
        cwd=REPO_DIR,
        capture_output=True,
        text=True,
    )
    assert cache_run.returncode == 0, cache_run.stderr

    structural_summary = json.loads(next(structural_out.rglob("phase1_summary.json")).read_text(encoding="utf-8"))
    cache_summary = json.loads(next(cache_out.rglob("phase1_summary.json")).read_text(encoding="utf-8"))
    assert cache_summary == structural_summary
    cache_dataset = json.loads(next(cache_out.rglob("dataset.json")).read_text(encoding="utf-8"))
    assert cache_dataset["config"]["input_mode"] == "observable_cache"
    assert cache_dataset["config"]["input_observable_cache_sha256"] == sha256_file(cache)

