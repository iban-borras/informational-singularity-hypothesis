from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import hsi_v2_d0154_det_b_calibration as calibration


class DetBCalibrationTests(unittest.TestCase):
    def test_frozen_sampling_arithmetic(self) -> None:
        self.assertEqual(
            max(
                calibration.MEDIUM_MIN_STEP,
                calibration.STRUCTURAL_LENGTH // calibration.MEDIUM_DIVISOR,
            ),
            calibration.MEDIUM_STEP,
        )
        self.assertEqual(
            -(-calibration.OBSERVABLE_COUNT // calibration.MEDIUM_STEP),
            calibration.MEDIUM_COUNT,
        )
        self.assertEqual(
            calibration.MEDIUM_COUNT // calibration.DET_BITS,
            calibration.REPRESENTATIVE_STEP,
        )
        self.assertEqual(
            calibration.MEDIUM_STEP * calibration.REPRESENTATIVE_STEP,
            calibration.OBSERVABLE_INTERVAL,
        )

    def test_chunk_partition_does_not_change_medium_sample(self) -> None:
        bits = np.asarray([(index * 7 + 1) % 2 for index in range(53)], dtype=np.int8)

        one_chunk = calibration.initial_scan_state({"fixture": 1})
        calibration.capture_observable_chunk(one_chunk, bits, 0, 5)

        partitioned = calibration.initial_scan_state({"fixture": 1})
        cursor = 0
        for width in (3, 11, 1, 17, 21):
            chunk = bits[cursor : cursor + width]
            calibration.capture_observable_chunk(partitioned, chunk, cursor, 5)
            cursor += len(chunk)

        expected = bytearray(bits[::5].tolist())
        self.assertEqual(one_chunk["medium_samples"], expected)
        self.assertEqual(partitioned["medium_samples"], expected)
        self.assertEqual(partitioned["observable_index"], len(bits))
        self.assertEqual(partitioned["next_capture"], 55)

    def test_checkpoint_roundtrip_binds_identity_and_sample_hash(self) -> None:
        identity = {"source": {"sha256": "A" * 64}, "protocol": {"step": 1}}
        state = calibration.initial_scan_state(identity)
        state.update(
            {
                "observable_index": 2 * calibration.MEDIUM_STEP,
                "next_capture": 2 * calibration.MEDIUM_STEP,
                "chunks_completed": 4,
                "medium_samples": bytearray([0, 1]),
            }
        )
        with tempfile.TemporaryDirectory() as raw_temp:
            path = Path(raw_temp) / "checkpoint.json"
            calibration.write_checkpoint(path, state)
            restored = calibration.load_checkpoint(path, identity)
            self.assertEqual(restored, state)
            self.assertEqual(list(path.parent.glob(".*.tmp")), [])

            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["medium_samples_sha256"] = "B" * 64
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(calibration.OperationalAbort, "SHA-256"):
                calibration.load_checkpoint(path, identity)

    def test_checkpoint_rejects_execution_identity_drift(self) -> None:
        identity = {"runner": "A" * 64}
        state = calibration.initial_scan_state(identity)
        with tempfile.TemporaryDirectory() as raw_temp:
            path = Path(raw_temp) / "checkpoint.json"
            calibration.write_checkpoint(path, state)
            with self.assertRaisesRegex(calibration.OperationalAbort, "identity"):
                calibration.load_checkpoint(path, {"runner": "B" * 64})

    def test_representative_sample_and_outcomes_are_exact(self) -> None:
        medium = bytearray((index % 2 for index in range(calibration.MEDIUM_COUNT)))
        representative = calibration.build_representative(medium)
        self.assertEqual(len(representative), calibration.DET_BITS)
        self.assertEqual(
            representative,
            medium[:: calibration.REPRESENTATIVE_STEP][: calibration.DET_BITS],
        )
        self.assertEqual(
            calibration.classify_outcome("0.55"), "b-calibration-reproduces"
        )
        self.assertEqual(
            calibration.classify_outcome("0.54"), "b-calibration-fails"
        )
        self.assertEqual(
            calibration.classify_outcome("0.56"), "b-calibration-fails"
        )

    def test_scan_gates_require_all_frozen_counts(self) -> None:
        valid = {
            "observable_bits_scanned": calibration.OBSERVABLE_COUNT,
            "medium_step": calibration.MEDIUM_STEP,
            "medium_sample_count": calibration.MEDIUM_COUNT,
            "next_capture": calibration.MEDIUM_COUNT * calibration.MEDIUM_STEP,
        }
        calibration.enforce_scan_gates(valid)
        with self.assertRaisesRegex(calibration.OperationalAbort, "Observable count"):
            calibration.enforce_scan_gates(
                {**valid, "observable_bits_scanned": calibration.OBSERVABLE_COUNT - 1}
            )
        with self.assertRaisesRegex(calibration.OperationalAbort, "Medium count"):
            calibration.enforce_scan_gates(
                {**valid, "medium_sample_count": calibration.MEDIUM_COUNT - 1}
            )

    def test_dry_run_never_crosses_target_access_barrier(self) -> None:
        argv = ["hsi_v2_d0154_det_b_calibration.py"]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(
                calibration,
                "validate_frozen_contract_and_audit",
                side_effect=AssertionError("dry-run crossed audit gate"),
            ),
            mock.patch.object(
                calibration,
                "validate_source_identity",
                side_effect=AssertionError("dry-run touched B source"),
            ),
        ):
            self.assertEqual(calibration.main(), 0)

    def test_execute_requires_anchored_code_audit_sha(self) -> None:
        argv = ["hsi_v2_d0154_det_b_calibration.py", "--execute"]
        with mock.patch.object(sys, "argv", argv):
            with self.assertRaisesRegex(SystemExit, "code-audit-sha256"):
                calibration.main()

    def test_audit_gate_binds_exact_runner_hash(self) -> None:
        runner_sha = "C" * 64
        with tempfile.TemporaryDirectory() as raw_temp:
            root = Path(raw_temp)
            prereg = root / "prereg.md"
            ots = root / "prereg.md.ots"
            audit = root / "audit.md"
            decision_log = root / "decision.md"
            prereg.write_text("frozen contract\n", encoding="utf-8")
            ots.write_bytes(b"ots")
            prereg_sha = calibration.sha256_file(prereg)

            audit.write_text("APPROVED but wrong runner\n", encoding="utf-8")
            audit_sha = calibration.sha256_file(audit)
            decision_log.write_text(f"{prereg_sha}\n{audit_sha}\n", encoding="utf-8")
            with (
                mock.patch.object(calibration, "PREREG_PATH", prereg),
                mock.patch.object(calibration, "PREREG_OTS_PATH", ots),
                mock.patch.object(calibration, "PREREG_SHA256", prereg_sha),
                mock.patch.object(calibration, "CODE_AUDIT_PATH", audit),
                mock.patch.object(calibration, "DECISION_LOG_PATH", decision_log),
            ):
                with self.assertRaisesRegex(calibration.OperationalAbort, "runner SHA"):
                    calibration.validate_frozen_contract_and_audit(audit_sha, runner_sha)

                audit.write_text(f"APPROVED\nrunner {runner_sha}\n", encoding="utf-8")
                audit_sha = calibration.sha256_file(audit)
                decision_log.write_text(f"{prereg_sha}\n{audit_sha}\n", encoding="utf-8")
                gates = calibration.validate_frozen_contract_and_audit(
                    audit_sha, runner_sha
                )
                self.assertEqual(gates["code_audit_sha256"], audit_sha)

    def test_sha_normalization_is_strict(self) -> None:
        self.assertEqual(calibration.normalize_sha256("a" * 64), "A" * 64)
        for invalid in ("A" * 63, "A" * 65, "Z" * 64):
            with self.assertRaises(calibration.OperationalAbort):
                calibration.normalize_sha256(invalid)

    @unittest.skipUnless(os.name == "nt", "Windows RSS implementation")
    def test_windows_peak_rss_is_measured(self) -> None:
        self.assertGreater(calibration.current_rss_bytes(), 0)


if __name__ == "__main__":
    unittest.main()
