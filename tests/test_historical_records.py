"""Confere documentos contra blobs Git, sem executar os scripts históricos."""

from decimal import Decimal
import hashlib
from pathlib import Path
import re
import shutil
import subprocess
import unittest

from contracts import ContractError, load_artifact, parse_artifact, validate_bundle

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = ROOT / "results/historical"
ORIGINAL_BLOBS = {
    "README.md": "b21af9fc1c310d2f9879ee2260c30c86371c60b4",
    "data_analysis.py": "36505e80533fb58cc7d92cc103341c1d68172440",
    "machine_learning.py": "3fe7d3fe44ddd2fa281fc5037a07432af87b9701",
    "hmeq.csv": "669687b31b616ff4030ef015c4d275d6b446db97",
}


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout


class HistoricalTests(unittest.TestCase):
    def setUp(self):
        self.readme = load_artifact(HISTORICAL / "readme-results.json")
        self.dashboard = load_artifact(HISTORICAL / "dashboard-results.json")

    def test_documental_categories_and_unknowns(self):
        for artifact in (self.readme, self.dashboard):
            self.assertEqual(artifact.evidence_kind, "historical_documented")
            self.assertIsNone(artifact.run_id)
            self.assertIsNone(artifact.sample_count)
            self.assertEqual(artifact.evaluation_scope, "unknown")
            self.assertNotIn("environment", artifact.to_dict())
            self.assertNotIn("seeds", artifact.to_dict())
        validate_bundle(self.readme, self.dashboard)

    def test_no_missing_rounds_or_loss_invented_in_readme(self):
        self.assertEqual([r["step"] for r in self.readme.records], [1, 5, 10])
        self.assertEqual([r["step"] for r in self.dashboard.records], list(range(1, 11)))
        for record in self.readme.records:
            self.assertEqual({m["name"] for m in record["metrics"]}, {"accuracy", "f1"})

    def test_disagreements_preserved(self):
        self.assertEqual(self.readme.records[0]["metrics"][0]["value"], 0.8397)
        self.assertEqual(self.dashboard.records[0]["metrics"][0]["value"], 0.8398)
        self.assertEqual(self.readme.records[1]["metrics"][1]["reported_literal"], "0.589")
        self.assertEqual(self.dashboard.records[4]["metrics"][1]["reported_literal"], "0.5895")

    def test_requires_sources_and_preserved_literals(self):
        for alteration in ("source", "literal", "run"):
            value = self.readme.to_dict()
            if alteration == "source":
                value["source_refs"] = []
            elif alteration == "literal":
                value["records"][0]["metrics"][0]["reported_literal"] = None
            else:
                value["run_id"] = "invented-run"
            with self.subTest(alteration=alteration), self.assertRaises(ContractError):
                parse_artifact(value)

    def test_historical_files_unchanged_byte_for_byte(self):
        for name, expected in ORIGINAL_BLOBS.items():
            data = (ROOT / name).read_bytes()
            blob = hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()
            with self.subTest(file=name):
                self.assertEqual(blob, expected)

    @unittest.skipUnless(shutil.which("git") and (ROOT / ".git").exists(), "requer histórico Git local")
    def test_sources_and_literals_match_git_objects(self):
        for artifact in (self.readme, self.dashboard):
            ref = artifact.source_refs[0]
            object_name = f"{ref['commit']}:{ref['path']}"
            self.assertEqual(git("rev-parse", object_name).strip(), ref["blob"])
            lines = git("show", object_name).splitlines()
            self.assertLessEqual(ref["line_end"], len(lines))
            excerpt = "\n".join(lines[ref["line_start"] - 1:ref["line_end"]])
            if artifact is self.readme:
                for record in artifact.records:
                    matches = re.findall(rf"^\|\s*{record['step']}\s*\|([^\n]+)$", excerpt, re.MULTILINE)
                    self.assertEqual(len(matches), 1)
                    cells = [cell.strip().replace("**", "") for cell in matches[0].split("|")[:2]]
                    for cell, metric in zip(cells, record["metrics"]):
                        self.assertEqual(metric["reported_literal"], cell)
                        expected = Decimal(cell.removesuffix("%")) / (100 if cell.endswith("%") else 1)
                        self.assertEqual(Decimal(str(metric["value"])), expected)
            else:
                arrays = {}
                for name in ("ROUNDS", "ACC", "F1", "LOSS"):
                    match = re.search(rf"const\s+{name}\s*=\s*\[([^\]]+)\]", excerpt)
                    self.assertIsNotNone(match)
                    arrays[name] = [item.strip() for item in match.group(1).split(",")]
                self.assertEqual([r["step"] for r in artifact.records], [int(n) for n in arrays["ROUNDS"]])
                for index, record in enumerate(artifact.records):
                    for metric, name in zip(record["metrics"], ("ACC", "F1", "LOSS")):
                        self.assertEqual(metric["reported_literal"], arrays[name][index])
                        self.assertEqual(Decimal(str(metric["value"])), Decimal(arrays[name][index]))


if __name__ == "__main__":
    unittest.main()
