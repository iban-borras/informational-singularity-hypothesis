from __future__ import annotations

import gzip
import random
from pathlib import Path

import numpy as np
import pytest

from hsi_v2_d0154_d23_fractal_rederivation import (
    PAPER_FRACTAL_DIMENSION,
    classify_reconciliation,
    pack_structural,
)
from v2.signature.streaming_fractal import (
    StreamingBoxCounter,
    legacy_reference,
    load_checkpoint,
    scan_v33_observable_box_counts,
    write_checkpoint_atomic,
)


def binary_array(text: str) -> np.ndarray:
    return np.frombuffer(text.encode("ascii"), dtype=np.uint8) - ord("0")


def scientific_payload(payload: dict) -> dict:
    return {
        "observable_bits": payload["observable_bits"],
        "observable_ones": payload["observable_ones"],
        "fractal_dimension": payload["fractal_dimension"],
        "box_counts": [
            (row["box_size"], row["occupied_boxes"]) for row in payload["box_counts"]
        ],
    }


def write_fixture(path: Path, structural: str) -> None:
    with gzip.open(path, "wb", compresslevel=1) as handle:
        handle.write(pack_structural(structural))


def test_random_chunk_partitions_match_legacy_reference() -> None:
    rng = random.Random(180826)
    for length in (4, 13, 64, 511, 4097):
        bits = np.asarray([rng.randrange(2) for _ in range(length)], dtype=np.uint8)
        expected = legacy_reference(bits)
        expected_counts = [
            (row["box_size"], row["occupied_boxes"]) for row in expected["box_counts"]
        ]

        for max_chunk in (1, 3, 17, 257):
            counter = StreamingBoxCounter()
            offset = 0
            while offset < bits.size:
                width = min(bits.size - offset, rng.randint(1, max_chunk))
                counter.update_observable(bits[offset : offset + width])
                offset += width
            observed = counter.result()
            observed_counts = [
                (row["box_size"], row["occupied_boxes"])
                for row in observed["box_counts"]
            ]
            assert observed_counts == expected_counts
            assert observed["fractal_dimension"] == pytest.approx(
                expected["fractal_dimension"], abs=1e-15
            )


def test_all_zero_semantics_match_legacy_v33() -> None:
    bits = np.zeros(256, dtype=np.uint8)
    expected = legacy_reference(bits)
    counter = StreamingBoxCounter()
    counter.update_observable(bits[:101])
    counter.update_observable(bits[101:])
    observed = counter.result()
    assert expected["fractal_dimension"] == pytest.approx(0.0)
    assert observed["fractal_dimension"] == pytest.approx(0.0)


def test_structural_filter_and_padding_are_exact(tmp_path: Path) -> None:
    structural = "(01)1(0"  # Eight valid chars; no padding ambiguity.
    source = tmp_path / "fixture.struct.gz"
    write_fixture(source, structural)
    observable = "0110"
    observed = scan_v33_observable_box_counts(
        source,
        sequence_length=len(structural),
        expected_observable_bits=len(observable),
        source_sha256="A" * 64,
        raw_chunk_bytes=1,
        checkpoint_interval_bytes=2,
    )
    expected = legacy_reference(binary_array(observable))
    assert scientific_payload(observed)["box_counts"] == [
        (row["box_size"], row["occupied_boxes"]) for row in expected["box_counts"]
    ]
    assert observed["observable_bits"] == len(observable)

    padded_structural = "(1)"  # The packed byte has one padded 00 code.
    padded_source = tmp_path / "padded.struct.gz"
    write_fixture(padded_source, padded_structural)
    padded = scan_v33_observable_box_counts(
        padded_source,
        sequence_length=len(padded_structural),
        expected_observable_bits=1,
        source_sha256="B" * 64,
        raw_chunk_bytes=1,
        checkpoint_interval_bytes=1,
    )
    assert padded["observable_bits"] == 1
    assert padded["observable_ones"] == 1


def test_random_structural_streams_match_reference(tmp_path: Path) -> None:
    rng = random.Random(180827)
    alphabet = "01()"
    for case_index, length in enumerate((17, 63, 128, 509)):
        structural = "".join(rng.choice(alphabet) for _ in range(length))
        if not any(char in "01" for char in structural):
            structural = "0" + structural[1:]
        observable = "".join(char for char in structural if char in "01")
        expected = legacy_reference(binary_array(observable))
        expected_counts = [
            (row["box_size"], row["occupied_boxes"]) for row in expected["box_counts"]
        ]
        source = tmp_path / f"random-{case_index}.struct.gz"
        write_fixture(source, structural)

        for chunk_bytes in (1, 2, 7, 31):
            observed = scan_v33_observable_box_counts(
                source,
                sequence_length=len(structural),
                expected_observable_bits=len(observable),
                source_sha256=f"{case_index:X}" * 64,
                raw_chunk_bytes=chunk_bytes,
                checkpoint_interval_bytes=13,
            )
            assert observed["observable_bits"] == len(observable)
            assert observed["fractal_dimension"] == pytest.approx(
                expected["fractal_dimension"], abs=1e-15
            )
            assert [
                (row["box_size"], row["occupied_boxes"])
                for row in observed["box_counts"]
            ] == expected_counts


def test_checkpoint_resume_is_scientifically_identical(tmp_path: Path) -> None:
    structural = ("0(1)001(()1)0(0)11100" * 31) + "(01"
    source = tmp_path / "resume.struct.gz"
    write_fixture(source, structural)
    observable = "".join(char for char in structural if char in "01")
    source_sha = "C" * 64

    direct = scan_v33_observable_box_counts(
        source,
        sequence_length=len(structural),
        expected_observable_bits=len(observable),
        source_sha256=source_sha,
        raw_chunk_bytes=7,
        checkpoint_interval_bytes=13,
    )

    captured = []

    class StopAfterCheckpoint(RuntimeError):
        pass

    def interrupt(payload: dict) -> None:
        captured.append(payload)
        raise StopAfterCheckpoint

    with pytest.raises(StopAfterCheckpoint):
        scan_v33_observable_box_counts(
            source,
            sequence_length=len(structural),
            expected_observable_bits=len(observable),
            source_sha256=source_sha,
            raw_chunk_bytes=5,
            checkpoint_interval_bytes=11,
            checkpoint_callback=interrupt,
        )

    resumed = scan_v33_observable_box_counts(
        source,
        sequence_length=len(structural),
        expected_observable_bits=len(observable),
        source_sha256=source_sha,
        raw_chunk_bytes=3,
        checkpoint_interval_bytes=17,
        resume_state=captured[-1],
    )
    assert scientific_payload(resumed) == scientific_payload(direct)


def test_observable_count_mismatch_stops(tmp_path: Path) -> None:
    structural = "(0101)"
    source = tmp_path / "mismatch.struct.gz"
    write_fixture(source, structural)
    with pytest.raises(RuntimeError, match="observable bit count mismatch"):
        scan_v33_observable_box_counts(
            source,
            sequence_length=len(structural),
            expected_observable_bits=5,
            source_sha256="D" * 64,
            raw_chunk_bytes=1,
            checkpoint_interval_bytes=1,
        )


def test_checkpoint_identity_is_strict() -> None:
    counter = StreamingBoxCounter()
    counter.update_observable(binary_array("0101101"))
    payload = counter.to_checkpoint(
        source_sha256="E" * 64,
        sequence_length=20,
        expected_observable_bits=7,
    )
    with pytest.raises(ValueError, match="source_sha256"):
        StreamingBoxCounter.from_checkpoint(
            payload,
            source_sha256="F" * 64,
            sequence_length=20,
            expected_observable_bits=7,
            max_box_size=64,
        )

    malformed = dict(payload)
    malformed["packed_bytes_read"] = 1
    with pytest.raises(ValueError, match="packed and structural"):
        StreamingBoxCounter.from_checkpoint(
            malformed,
            source_sha256="E" * 64,
            sequence_length=20,
            expected_observable_bits=7,
            max_box_size=64,
        )


def test_checkpoint_write_is_atomic_and_roundtrips(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    payload = {"version": 1, "counts": [1, 2, 3]}
    write_checkpoint_atomic(path, payload)
    assert load_checkpoint(path) == payload
    assert not path.with_name("checkpoint.json.tmp").exists()


def test_reconciliation_uses_manuscript_display_precision() -> None:
    match = classify_reconciliation(0.9923116264667518)
    assert match["paper_value"] == PAPER_FRACTAL_DIMENSION
    assert match["clean_display"] == "0.992"
    assert match["reconciliation_status"] == "display-match"
    assert not match["review_required"]

    mismatch = classify_reconciliation(0.9931)
    assert mismatch["clean_display"] == "0.993"
    assert mismatch["reconciliation_status"] == "display-mismatch"
    assert mismatch["review_required"]
