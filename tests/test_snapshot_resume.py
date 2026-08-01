"""Equivalence and interruption tests for restartable v33 snapshots."""

import gzip
import random
import sys
import tempfile
from pathlib import Path

from bitarray import bitarray

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.bitarray_encoder import (
    ENCODING_MAP,
    _pack_structural_ascii,
    save_phi_structural_gz_from_file,
)


def _reference_bytes(text: str) -> bytes:
    return bitarray("".join(ENCODING_MAP[c] for c in text)).tobytes()


def test_numpy_packer_matches_canonical_bitarray():
    rng = random.Random(20260801)
    for length in range(1, 257):
        text = "".join(rng.choice("01()") for _ in range(length))
        carry = b""
        packed_parts = []
        position = 0
        while position < len(text):
            step = rng.randint(1, 19)
            raw = text[position:position + step].encode("ascii")
            position += len(raw)
            packed, carry = _pack_structural_ascii(
                raw,
                carry,
                final=(position == len(text)),
            )
            packed_parts.append(packed)
        assert b"".join(packed_parts) == _reference_bytes(text)
        assert carry == b""


def test_restartable_snapshot_resumes_byte_exactly():
    rng = random.Random(42)
    text = "".join(rng.choice("01()") for _ in range(10_003))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        source = tmpdir / "source.txt.gz"
        output = tmpdir / "snapshot.struct.gz"
        with gzip.open(source, "wb", compresslevel=1) as handle:
            handle.write(text.encode("ascii"))

        try:
            save_phi_structural_gz_from_file(
                str(source),
                str(output),
                compresslevel=1,
                chunk_size=37,
                checkpoint_chars=407,
                input_length=len(text),
                checkpoint_key="snapshot-resume-test",
                _test_interrupt_after_parts=3,
            )
            raise AssertionError("The simulated snapshot interruption did not fire")
        except RuntimeError as exc:
            assert str(exc) == "simulated snapshot interruption"

        save_phi_structural_gz_from_file(
            str(source),
            str(output),
            compresslevel=1,
            chunk_size=37,
            checkpoint_chars=407,
            input_length=len(text),
            checkpoint_key="snapshot-resume-test",
        )
        with gzip.open(output, "rb") as handle:
            actual = handle.read()

        assert actual == _reference_bytes(text)
        assert not output.with_name(output.name + ".parts").exists()


if __name__ == "__main__":
    test_numpy_packer_matches_canonical_bitarray()
    print("Packer equivalence: passed")
    test_restartable_snapshot_resumes_byte_exactly()
    print("Interrupted snapshot resume: passed")
