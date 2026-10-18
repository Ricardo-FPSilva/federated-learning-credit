"""Testes do exportador; não executam Flower, Ray ou treinamento."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from contracts import load_artifact, validate_bundle
from research.historical.export_run import ExportError, export_run, parse_stdout


ROOT = Path(__file__).resolve().parents[1]
SOURCE_RUN = ROOT / "results/runs/historical-LqJ1xZ"
INPUT_NAMES = (
    "commit.txt", "exit-code.txt", "source-hashes.txt", "stderr.log",
    "stdout.log", "time.txt",
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ParseHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stdout = (SOURCE_RUN / "stdout.log").read_text(encoding="utf-8")

    def test_extracts_ten_complete_ordered_rounds(self):
        records = parse_stdout(self.stdout)
        self.assertEqual([record["step"] for record in records], list(range(1, 11)))
        self.assertEqual([m["name"] for m in records[-1]["metrics"]],
                         ["accuracy", "f1", "bce_loss"])
        self.assertEqual(records[-1]["metrics"][0]["value"], 0.87751677852349)
        self.assertEqual(records[-1]["metrics"][1]["value"], 0.6054054054054054)
        self.assertEqual(records[-1]["metrics"][2]["value"], 0.3189521301537752)

    def test_rejects_incomplete_log(self):
        incomplete = self.stdout.replace("\tround 10: 0.3189521301537752\n", "")
        with self.assertRaisesRegex(ExportError, "rounds 1–10"):
            parse_stdout(incomplete)

    def test_rejects_duplicate_round(self):
        duplicate = self.stdout.replace(
            "\tround 2: 0.3627481474296042",
            "\tround 1: 0.4000\n\tround 2: 0.3627481474296042",
        )
        with self.assertRaisesRegex(ExportError, "duplicados"):
            parse_stdout(duplicate)

    def test_rejects_missing_metric(self):
        missing = self.stdout.replace(" 'f1': [(1, 0.4705882352941176),", " 'other': [(1, 0.4705882352941176),")
        with self.assertRaisesRegex(ExportError, "accuracy/F1"):
            parse_stdout(missing)

    def test_rejects_missing_metric_round(self):
        missing = self.stdout.replace("              (10, 0.87751677852349)]", "              ]")
        with self.assertRaisesRegex(ExportError, "rounds 1–10"):
            parse_stdout(missing)

    def test_rejects_invalid_metric_values(self):
        for invalid in ("nan", "inf", "-0.1", "1.1"):
            bad = self.stdout.replace("0.8489932885906042", invalid, 1)
            with self.subTest(value=invalid), self.assertRaises(ExportError):
                parse_stdout(bad)

    def test_rejects_invalid_loss_values(self):
        for invalid in ("nan", "inf", "-0.1"):
            bad = self.stdout.replace("0.37982155931623357", invalid, 1)
            with self.subTest(value=invalid), self.assertRaises(ExportError):
                parse_stdout(bad)


class ExportRunTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temporary.name) / "historical-test"
        self.run_dir.mkdir()
        for name in INPUT_NAMES:
            (self.run_dir / name).write_bytes((SOURCE_RUN / name).read_bytes())

    def tearDown(self):
        self.temporary.cleanup()

    def test_exports_valid_bundle_and_exact_final_metrics(self):
        original_hashes = {name: sha256(self.run_dir / name) for name in INPUT_NAMES}
        targets = export_run(self.run_dir)
        self.assertEqual([path.name for path in targets], ["manifest.json", "history.json", "metrics.json"])
        manifest, history, metrics = (load_artifact(path) for path in targets)
        validate_bundle(manifest, history, metrics)

        self.assertEqual(manifest.evidence_kind, "historical_rerun")
        self.assertEqual(history.evidence_kind, "historical_rerun")
        self.assertEqual(metrics.evidence_kind, "historical_rerun")
        self.assertEqual(manifest.seeds["training"], None)
        self.assertEqual(manifest.preprocessing, None)
        self.assertEqual(manifest.dataset["sizes"], {"train": 4768, "test": 1192})
        self.assertEqual(history.sample_count, 1192)
        self.assertEqual([record["step"] for record in history.records], list(range(1, 11)))
        self.assertEqual(metrics.values, history.records[-1]["metrics"])
        self.assertIsNone(metrics.confusion_matrix)
        self.assertIsNone(metrics.resources["training_seconds"])
        self.assertEqual(metrics.resources["total_seconds"], 30.944)
        self.assertIn("not training-only", metrics.resources["measurement_scope"])

        self.assertNotIn("f1", manifest.artifacts)  # Artifact names describe files, not metrics.
        self.assertEqual(manifest.artifacts["history"]["sha256"], sha256(self.run_dir / "history.json"))
        self.assertEqual(manifest.artifacts["metrics"]["sha256"], sha256(self.run_dir / "metrics.json"))
        self.assertEqual(
            {name: sha256(self.run_dir / name) for name in INPUT_NAMES}, original_hashes
        )

    def test_refuses_silent_overwrite_and_preserves_outputs(self):
        targets = export_run(self.run_dir)
        before = {path.name: path.read_bytes() for path in targets}
        with self.assertRaisesRegex(ExportError, "já existem"):
            export_run(self.run_dir)
        self.assertEqual({path.name: path.read_bytes() for path in targets}, before)

    def test_rejects_nonzero_exit_code_without_outputs(self):
        (self.run_dir / "exit-code.txt").write_text("1\n", encoding="utf-8")
        with self.assertRaisesRegex(ExportError, "exit code 0"):
            export_run(self.run_dir)
        self.assertTrue(all(not (self.run_dir / name).exists()
                            for name in ("manifest.json", "history.json", "metrics.json")))

    def test_rejects_changed_source_hash_without_outputs(self):
        content = (self.run_dir / "source-hashes.txt").read_text(encoding="utf-8")
        (self.run_dir / "source-hashes.txt").write_text(content.replace("0de21b", "fde21b", 1), encoding="utf-8")
        with self.assertRaisesRegex(ExportError, "fonte alterada"):
            export_run(self.run_dir)
        self.assertFalse((self.run_dir / "manifest.json").exists())

    def test_json_files_are_utf8_and_do_not_contain_nonfinite_values(self):
        export_run(self.run_dir)
        for name in ("manifest.json", "history.json", "metrics.json"):
            content = (self.run_dir / name).read_text(encoding="utf-8")
            value = json.loads(content, parse_constant=lambda token: self.fail(token))
            self.assertEqual(value["evidence_kind"], "historical_rerun")


if __name__ == "__main__":
    unittest.main()
