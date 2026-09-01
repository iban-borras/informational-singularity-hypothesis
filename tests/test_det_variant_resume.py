from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import hsi_v2_d0154_det_provenance_review as det


def sample_payload() -> dict[str, object]:
    return {
        "sample_protocol": "fixture",
        "medium_step": 100,
        "total_chars_hint": 100,
        "medium_sample_count": 4,
        "representative_step": 1,
        "legacy_observable_interval": 100,
        "target_bits": 4,
        "sample_bits": 4,
        "sample_sha256": "A" * 64,
        "observable_bits_scanned": 100,
        "elapsed_seconds": 0.01,
    }


class DetVariantResumeTests(unittest.TestCase):
    def test_atomic_json_write_leaves_no_temporary_file(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            path = Path(raw_temp) / "checkpoint.json"
            det.write_json(path, {"status": "completed", "value": 0.55})
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"status": "completed", "value": 0.55},
            )
            self.assertEqual(list(path.parent.glob(".*.tmp")), [])

    def test_resume_plan_rejects_code_or_protocol_drift(self) -> None:
        plan = {
            "status": "running",
            "script": "runner.py",
            "decision_log_context": "D-0154",
            "results_base": "D:/results",
            "run_dir": "D:/results/run",
            "targets": {"E": 24},
            "protocol": {"det_bits": 10_000},
            "implementation_provenance": {"rqa_metric": "legacy"},
            "code_hashes": {"runner": "A" * 64},
            "resume_protocol": {"schema_version": 1},
        }
        det.validate_resume_plan(plan, dict(plan))

        changed_protocol = {**plan, "protocol": {"det_bits": 9_999}}
        with self.assertRaisesRegex(RuntimeError, "protocol"):
            det.validate_resume_plan(plan, changed_protocol)

        changed_code = {**plan, "code_hashes": {"runner": "B" * 64}}
        with self.assertRaisesRegex(RuntimeError, "code_hashes"):
            det.validate_resume_plan(plan, changed_code)

    def test_resume_source_inventory_must_match_exactly(self) -> None:
        saved = {"E": {"struct_sha256": "A" * 64, "struct_bytes": 10}}
        det.validate_resume_sources(saved, dict(saved))
        with self.assertRaisesRegex(RuntimeError, "source identity"):
            det.validate_resume_sources(
                saved,
                {"E": {"struct_sha256": "B" * 64, "struct_bytes": 10}},
            )

    def test_variant_artifact_requires_completion_and_exact_identity(self) -> None:
        identity = {"schema_version": 1, "source": {"struct_sha256": "A" * 64}}
        with tempfile.TemporaryDirectory() as raw_temp:
            path = Path(raw_temp) / "E_det.json"
            det.write_json(
                path,
                {
                    "status": "completed",
                    "execution_identity": identity,
                    "row": {"variant": "E"},
                    "detail": {"variant": "E"},
                },
            )
            artifact = det.load_completed_variant(path, identity)
            self.assertEqual(artifact["row"]["variant"], "E")

            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                det.load_completed_variant(
                    path,
                    {"schema_version": 1, "source": {"struct_sha256": "B" * 64}},
                )

            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["status"] = "running"
            det.write_json(path, payload)
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                det.load_completed_variant(path, identity)

    def test_interrupted_batch_reuses_only_completed_variant(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            root = Path(raw_temp).resolve()
            output_root = root / "outputs"
            fake_sources = {
                variant: {
                    "variant": variant,
                    "iteration": iteration,
                    "struct_path": str(root / f"{variant}.struct.gz"),
                    "metadata_path": str(root / f"{variant}.json"),
                    "report_path": None,
                    "struct_bytes": 10,
                    "sequence_length": 100,
                    "sampling_length_source": "metadata.sequence_length",
                    "metadata_sha256": variant * 64,
                    "report_sha256": None,
                    "struct_sha256": variant * 64,
                }
                for variant, iteration in {"E": 24, "I": 23}.items()
            }

            first_calls: list[str] = []

            def interrupted_sample(struct_path: Path, *_args: object) -> tuple[str, dict[str, object]]:
                variant = struct_path.name[0]
                first_calls.append(variant)
                if variant == "I":
                    raise RuntimeError("synthetic interruption")
                return "0101", sample_payload()

            first_argv = [
                "hsi_v2_d0154_det_provenance_review.py",
                "--targets",
                "E:24,I:23",
                "--results-base",
                str(root),
                "--output-dir",
                str(output_root),
                "--det-bits",
                "4",
                "--execute",
            ]
            with (
                mock.patch.object(sys, "argv", first_argv),
                mock.patch.object(det, "discover_sources", return_value=fake_sources),
                mock.patch.object(det, "build_legacy_det_bits", side_effect=interrupted_sample),
                mock.patch.object(det, "build_prefix_det_bits", return_value="0011"),
                mock.patch.object(det, "compute_det", return_value={"determinism": 0.55}),
            ):
                with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                    det.main()

            run_dirs = list(output_root.iterdir())
            self.assertEqual(len(run_dirs), 1)
            run_dir = run_dirs[0]
            self.assertEqual(first_calls, ["E", "I"])
            self.assertTrue((run_dir / "E_det.json").is_file())
            self.assertFalse((run_dir / "I_det.json").exists())
            self.assertEqual(det.read_json(run_dir / "manifest.json")["status"], "interrupted")

            resumed_calls: list[str] = []

            def resumed_sample(struct_path: Path, *_args: object) -> tuple[str, dict[str, object]]:
                variant = struct_path.name[0]
                resumed_calls.append(variant)
                if variant == "E":
                    raise AssertionError("completed E must not be recomputed")
                return "0101", sample_payload()

            resume_argv = [
                "hsi_v2_d0154_det_provenance_review.py",
                "--targets",
                "E:24,I:23",
                "--results-base",
                str(root),
                "--det-bits",
                "4",
                "--resume-run",
                str(run_dir),
                "--execute",
            ]
            with (
                mock.patch.object(sys, "argv", resume_argv),
                mock.patch.object(det, "discover_sources", return_value=fake_sources),
                mock.patch.object(det, "build_legacy_det_bits", side_effect=resumed_sample),
                mock.patch.object(det, "build_prefix_det_bits", return_value="0011"),
                mock.patch.object(det, "compute_det", return_value={"determinism": 0.55}),
            ):
                self.assertEqual(det.main(), 0)

            self.assertEqual(resumed_calls, ["I"])
            summary = det.read_json(run_dir / "summary.json")
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["reused_variants"], ["E"])
            self.assertEqual([row["variant"] for row in summary["rows"]], ["E", "I"])
            manifest = det.read_json(run_dir / "manifest.json")
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["resume_count"], 1)


if __name__ == "__main__":
    unittest.main()
