from __future__ import annotations

import gzip
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import hsi_v2_d0154_signature_priority_a_batch as batch
from hsi_v2_d0154_signature_priority_a_batch import (
    iter_bits_numpy_with_progress,
    validate_resume_plan,
)
from level2_transfer_entropy import StreamingTEAccumulator
from v2.signature.resumable_te import (
    accumulator_from_payload,
    accumulator_to_payload,
    build_checkpoint,
    load_checkpoint,
    restore_checkpoint,
    write_checkpoint_atomic,
)


def process_chunks(
    observed: StreamingTEAccumulator,
    shuffled: StreamingTEAccumulator,
    rng: np.random.Generator,
    chunks: list[np.ndarray],
) -> None:
    for bits in chunks:
        observed.process_chunk(bits, method="density")
        randomized = bits.copy()
        rng.shuffle(randomized)
        shuffled.process_chunk(randomized, method="density")


class ResumableTETests(unittest.TestCase):
    def test_accumulator_json_roundtrip_is_exact(self) -> None:
        rng = np.random.default_rng(180819)
        accumulator = StreamingTEAccumulator([2, 4, 8], k=3, n_bins=2)
        for _ in range(4):
            accumulator.process_chunk(
                rng.integers(0, 2, size=4096, dtype=np.int8), method="density"
            )
        payload = accumulator_to_payload(accumulator)
        restored = accumulator_from_payload(payload, StreamingTEAccumulator)
        self.assertEqual(accumulator_to_payload(restored), payload)

    def test_interrupted_run_matches_continuous_run(self) -> None:
        source_rng = np.random.default_rng(12345)
        chunks = [
            source_rng.integers(0, 2, size=8192 + index * 128, dtype=np.int8)
            for index in range(7)
        ]
        identity = {
            "source_sha256": "A" * 64,
            "scales": [2, 4, 8],
            "history": 3,
            "chunk_bits": 10_000_000,
        }

        direct_observed = StreamingTEAccumulator([2, 4, 8], k=3, n_bins=2)
        direct_shuffled = StreamingTEAccumulator([2, 4, 8], k=3, n_bins=2)
        direct_rng = np.random.default_rng(42)
        process_chunks(direct_observed, direct_shuffled, direct_rng, chunks)

        partial_observed = StreamingTEAccumulator([2, 4, 8], k=3, n_bins=2)
        partial_shuffled = StreamingTEAccumulator([2, 4, 8], k=3, n_bins=2)
        partial_rng = np.random.default_rng(42)
        process_chunks(partial_observed, partial_shuffled, partial_rng, chunks[:3])

        with tempfile.TemporaryDirectory() as raw_temp:
            checkpoint_path = Path(raw_temp) / "te_checkpoint.json"
            write_checkpoint_atomic(
                checkpoint_path,
                build_checkpoint(
                    identity=identity,
                    observed=partial_observed,
                    shuffled=partial_shuffled,
                    rng=partial_rng,
                    progress={"chunks_processed": 3, "encoded_bytes_read": 123},
                    status="running",
                ),
            )
            restored = restore_checkpoint(
                load_checkpoint(checkpoint_path), identity, StreamingTEAccumulator
            )

        restored_observed, restored_shuffled, restored_rng, progress = restored
        self.assertEqual(progress, {"chunks_processed": 3, "encoded_bytes_read": 123})
        process_chunks(restored_observed, restored_shuffled, restored_rng, chunks[3:])
        self.assertEqual(
            accumulator_to_payload(restored_observed),
            accumulator_to_payload(direct_observed),
        )
        self.assertEqual(
            accumulator_to_payload(restored_shuffled),
            accumulator_to_payload(direct_shuffled),
        )
        self.assertEqual(
            restored_rng.bit_generator.state, direct_rng.bit_generator.state
        )

    def test_gzip_resume_starts_at_exact_raw_chunk_boundary(self) -> None:
        raw = bytes(range(256)) * 17
        with tempfile.TemporaryDirectory() as raw_temp:
            source = Path(raw_temp) / "struct.gz"
            with gzip.open(source, "wb", compresslevel=1) as handle:
                handle.write(raw)

            complete = list(iter_bits_numpy_with_progress(source, chunk_bytes=137))
            resume_offset = complete[4][1]
            resumed = list(
                iter_bits_numpy_with_progress(
                    source, chunk_bytes=137, start_encoded_bytes=resume_offset
                )
            )

        expected = complete[5:]
        self.assertEqual(len(resumed), len(expected))
        for observed, reference in zip(resumed, expected, strict=True):
            np.testing.assert_array_equal(observed[0], reference[0])
            self.assertEqual(observed[1], reference[1])

    def test_checkpoint_identity_mismatch_is_rejected(self) -> None:
        accumulator = StreamingTEAccumulator([2, 4], k=3, n_bins=2)
        payload = build_checkpoint(
            identity={"source_sha256": "A" * 64},
            observed=accumulator,
            shuffled=StreamingTEAccumulator([2, 4], k=3, n_bins=2),
            rng=np.random.default_rng(42),
            progress={},
            status="running",
        )
        with self.assertRaisesRegex(ValueError, "identity"):
            restore_checkpoint(
                payload,
                {"source_sha256": "B" * 64},
                StreamingTEAccumulator,
            )

    def test_checkpoint_write_is_atomic(self) -> None:
        accumulator = StreamingTEAccumulator([2, 4], k=3, n_bins=2)
        payload = build_checkpoint(
            identity={"source_sha256": "C" * 64},
            observed=accumulator,
            shuffled=StreamingTEAccumulator([2, 4], k=3, n_bins=2),
            rng=np.random.default_rng(42),
            progress={"chunks_processed": 0},
            status="running",
        )
        with tempfile.TemporaryDirectory() as raw_temp:
            path = Path(raw_temp) / "checkpoint.json"
            write_checkpoint_atomic(path, payload)
            self.assertEqual(load_checkpoint(path), payload)
            self.assertFalse(path.with_name("checkpoint.json.tmp").exists())

    def test_cached_te_requires_matching_completed_checkpoint(self) -> None:
        identity = {"source_sha256": "C" * 64}
        accumulator = StreamingTEAccumulator([2, 4], k=3, n_bins=2)
        with tempfile.TemporaryDirectory() as raw_temp:
            run_dir = Path(raw_temp)
            metric_path = run_dir / "F_te.json"
            checkpoint_path = run_dir / "F_te_checkpoint.json"
            metric_path.write_text('{"value": 1.25}\n', encoding="utf-8")

            with self.assertRaisesRegex(
                RuntimeError, "without its completion checkpoint"
            ):
                batch.load_completed_metric(
                    run_dir,
                    "F",
                    "te",
                    True,
                    run_dir / "batch.log",
                    completion_checkpoint=checkpoint_path,
                    completion_identity=identity,
                )

            checkpoint = build_checkpoint(
                identity=identity,
                observed=accumulator,
                shuffled=StreamingTEAccumulator([2, 4], k=3, n_bins=2),
                rng=np.random.default_rng(42),
                progress={"chunks_processed": 0},
                status="completed",
            )
            write_checkpoint_atomic(checkpoint_path, checkpoint)
            cached = batch.load_completed_metric(
                run_dir,
                "F",
                "te",
                True,
                run_dir / "batch.log",
                completion_checkpoint=checkpoint_path,
                completion_identity=identity,
            )
            self.assertEqual(cached, {"value": 1.25})

            with self.assertRaisesRegex(RuntimeError, "identity"):
                batch.load_completed_metric(
                    run_dir,
                    "F",
                    "te",
                    True,
                    run_dir / "batch.log",
                    completion_checkpoint=checkpoint_path,
                    completion_identity={"source_sha256": "D" * 64},
                )

    def test_resume_plan_rejects_scientific_or_code_drift(self) -> None:
        plan = {
            "status": "running",
            "script": "runner.py",
            "results_base": "D:/results",
            "run_dir": "D:/results/run",
            "targets": {"F": 27},
            "metrics": ["te"],
            "code_hashes": {"runner": "A" * 64},
            "protocol": {"te_chunk_bits": 10_000_000},
        }
        validate_resume_plan(plan, dict(plan))
        changed = {**plan, "protocol": {"te_chunk_bits": 2_000_000}}
        with self.assertRaisesRegex(RuntimeError, "protocol"):
            validate_resume_plan(plan, changed)

    def test_compute_te_interruption_and_resume_match_continuous(self) -> None:
        raw = bytes(range(256)) * 64
        source_sha = hashlib.sha256(raw).hexdigest().upper()
        code_hashes = {"runner": "D" * 64, "resumable_te": "E" * 64}

        with tempfile.TemporaryDirectory() as raw_temp:
            root = Path(raw_temp)
            source = root / "fixture.struct.gz"
            with gzip.open(source, "wb", compresslevel=1) as handle:
                handle.write(raw)

            direct = batch.compute_te_metric(
                source,
                [2, 4, 8],
                3,
                512,
                42,
                0,
                3600,
                root / "direct.log",
                True,
                checkpoint_path=root / "direct.checkpoint.json",
                checkpoint_chunks=2,
                source_sha256=source_sha,
                source_size=source.stat().st_size,
                code_hashes=code_hashes,
                resume=False,
            )

            checkpoint_path = root / "resumed.checkpoint.json"
            real_writer = batch.write_te_checkpoint_atomic

            class SyntheticInterruption(RuntimeError):
                pass

            def interrupt_after_checkpoint(path: Path, payload: dict) -> None:
                real_writer(path, payload)
                if payload["status"] == "running":
                    raise SyntheticInterruption

            with mock.patch.object(
                batch,
                "write_te_checkpoint_atomic",
                side_effect=interrupt_after_checkpoint,
            ):
                with self.assertRaises(SyntheticInterruption):
                    batch.compute_te_metric(
                        source,
                        [2, 4, 8],
                        3,
                        512,
                        42,
                        0,
                        3600,
                        root / "resumed.log",
                        True,
                        checkpoint_path=checkpoint_path,
                        checkpoint_chunks=2,
                        source_sha256=source_sha,
                        source_size=source.stat().st_size,
                        code_hashes=code_hashes,
                        resume=False,
                    )

            resumed = batch.compute_te_metric(
                source,
                [2, 4, 8],
                3,
                512,
                42,
                0,
                3600,
                root / "resumed.log",
                True,
                checkpoint_path=checkpoint_path,
                checkpoint_chunks=2,
                source_sha256=source_sha,
                source_size=source.stat().st_size,
                code_hashes=code_hashes,
                resume=True,
            )

        exact_keys = (
            "observable_bits_consumed",
            "observable_bits_yielded",
            "encoded_bytes_read",
            "chunks_processed",
            "te_matrix",
            "shuffle_te_matrix",
            "avg_bottom_up",
            "avg_top_down",
            "avg_observed_te",
            "avg_shuffle_bottom_up",
            "avg_shuffle_top_down",
            "avg_shuffle_te",
            "te_ratio_observed_vs_shuffle",
            "value",
        )
        self.assertTrue(resumed["resumed"])
        for key in exact_keys:
            self.assertEqual(resumed[key], direct[key], key)


if __name__ == "__main__":
    unittest.main()
