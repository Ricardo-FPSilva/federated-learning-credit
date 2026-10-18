"""Testes do pipeline de dados corrigido; nenhum modelo é treinado."""

import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import tempfile
import unittest

from contracts.validation import identifier
from research.data.partitioning import PartitionError, iid_partition
from research.data.pipeline import (
    DEFAULT_CONFIG,
    PipelineError,
    build_record,
    load_dataset,
    load_record,
    partition_id_for,
    split_id_for,
    validate_record,
    write_record,
)
from research.data.preprocessing import FeaturePreprocessor, PreprocessingError
from research.data.splitting import SPLIT_NAMES, SplitError, stratified_split
from contracts.schemas import preprocessing as validate_preprocessing


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "hmeq.csv"
PUBLISHED = ROOT / "results/data/benchmark-v1-split.json"
TARGET = "BAD"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_digest(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[TARGET, "num", "cat"])
        writer.writeheader()
        writer.writerows(rows)


class RealDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = build_record(DATASET)
        with DATASET.open(newline="", encoding="utf-8") as stream:
            cls.labels = [int(row[TARGET]) for row in csv.DictReader(stream)]

    def test_split_sizes_and_stratification(self):
        splits = self.record["splits"]
        self.assertEqual([splits[name]["size"] for name in SPLIT_NAMES], [3576, 1192, 1192])
        fractions = self.record["config"]["fractions"]
        for label in (0, 1):
            class_total = self.labels.count(label)
            for name in SPLIT_NAMES:
                count = sum(self.labels[index] == label for index in splits[name]["indices"])
                expected = fractions[name] * class_total
                self.assertLessEqual(abs(count - expected), 1.0, (label, name))

    def test_no_overlap_and_full_coverage(self):
        splits = self.record["splits"]
        selected = {name: set(splits[name]["indices"]) for name in SPLIT_NAMES}
        self.assertEqual(set().union(*selected.values()), set(range(len(self.labels))))
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            self.assertFalse(selected[left] & selected[right])

    def test_partition_is_exact_repartition_of_train(self):
        train = set(self.record["splits"]["train"]["indices"])
        covered = set()
        for client, block in self.record["partitions"].items():
            self.assertTrue(re.fullmatch(r"client-\d+", client))
            indices = set(block["indices"])
            self.assertTrue(indices)
            self.assertFalse(indices & covered)
            covered |= indices
        self.assertEqual(covered, train)
        self.assertEqual(sum(block["size"] for block in self.record["partitions"].values()), 3576)

    def test_target_excluded_and_feature_order_frozen(self):
        with DATASET.open(newline="", encoding="utf-8") as stream:
            header = list(csv.DictReader(stream).fieldnames or ())
        self.assertEqual(self.record["feature_order"], [c for c in header if c != TARGET])
        self.assertNotIn(TARGET, self.record["feature_order"])
        self.assertEqual(self.record["dataset"]["features"], 12)
        self.assertGreater(len(self.record["output_features"]), 12)
        kinds = self.record["preprocessing_statistics"]
        self.assertEqual(set(kinds["categorical"]), {"REASON", "JOB"})
        self.assertEqual(set(kinds["numeric"]),
                         set(self.record["feature_order"]) - {"REASON", "JOB"})

    def test_feature_dimensions_and_finiteness(self):
        rows, fields, _ = load_dataset(DATASET)
        preprocessor = FeaturePreprocessor([c for c in fields if c != TARGET])
        preprocessor.fit([rows[i] for i in self.record["splits"]["train"]["indices"]])
        width = len(preprocessor.output_features)
        self.assertEqual(width, len(self.record["output_features"]))
        for name in SPLIT_NAMES:
            vectors = preprocessor.transform(
                [rows[i] for i in self.record["splits"][name]["indices"]]
            )
            self.assertTrue(all(len(vector) == width for vector in vectors))
            self.assertTrue(all(math.isfinite(value) for vector in vectors for value in vector))

    def test_statistics_equal_train_only_reference(self):
        rows, _, _ = load_dataset(DATASET)
        stats = self.record["preprocessing_statistics"]["numeric"]
        train = [rows[i] for i in self.record["splits"]["train"]["indices"]]
        for column, expected in stats.items():
            present = [float(row[column]) for row in train if row[column] is not None]
            self.assertEqual(expected["median"], float(statistics.median(present)), column)
            imputed = [expected["median"] if row[column] is None else float(row[column])
                       for row in train]
            self.assertEqual(expected["mean"], float(statistics.fmean(imputed)), column)
            self.assertEqual(expected["missing_in_train"],
                             sum(row[column] is None for row in train), column)

    def test_determinism_same_config(self):
        self.assertEqual(build_record(DATASET), self.record)
        self.assertEqual(build_record(DATASET)["split_id"], self.record["split_id"])
        self.assertEqual(build_record(DATASET)["partition_id"], self.record["partition_id"])

    def test_no_duplicates_in_hmeq(self):
        audit = self.record["duplicate_audit"]
        self.assertEqual(audit["exact"], {"groups": 0, "rows": 0, "extra_rows": 0})
        self.assertEqual(audit["features"], {"groups": 0, "rows": 0, "extra_rows": 0})
        self.assertEqual(audit["cross_split_feature_groups"], 0)
        self.assertIsNone(audit["group_column"])
        self.assertTrue(audit["group_audit_limitation"])

    def test_identifiers_and_hashes_are_integrity_anchored(self):
        record = self.record
        self.assertEqual(record["dataset"]["sha256"], digest(DATASET))
        for name in ("split_id", "partition_id"):
            identifier(record[name], name)
        for name, block in record["splits"].items():
            self.assertEqual(block["index_sha256"], canonical_digest(block["indices"]), name)
        for client, block in record["partitions"].items():
            self.assertEqual(block["index_sha256"], canonical_digest(block["indices"]), client)
        validate_preprocessing(record["preprocessing"])
        self.assertEqual(record["preprocessing"]["sha256"], canonical_digest(
            record["preprocessing_statistics"]))
        splits = {name: block["indices"] for name, block in record["splits"].items()}
        partitions = {client: block["indices"] for client, block in record["partitions"].items()}
        self.assertEqual(record["split_id"], split_id_for(
            record["dataset"]["sha256"], record["config"]["fractions"],
            record["seeds"]["split"], splits))
        self.assertEqual(record["partition_id"], partition_id_for(
            record["dataset"]["sha256"], record["config"]["strategy"],
            record["config"]["n_clients"], record["seeds"]["partition"], partitions))

    def test_record_is_serializable_and_reusable(self):
        payload = json.dumps(self.record, ensure_ascii=False, allow_nan=False)
        self.assertEqual(json.loads(payload), self.record)
        self.assertEqual(self.record["schema"], "corrected-data-pipeline/1.0")
        self.assertEqual(self.record["protocol_id"], "benchmark-v1")
        self.assertEqual(self.record["seeds"], {"split": 42, "validation": 43, "partition": 44})
        self.assertEqual(self.record["seed_usage"], {
            "split": "consumed",
            "validation": "reserved_not_consumed_in_e2",
            "partition": "consumed",
        })
        self.assertEqual(set(self.record["splits"]), set(SPLIT_NAMES))


class PublishedArtifactTests(unittest.TestCase):
    def test_published_artifact_is_valid_and_reproducible(self):
        published = load_record(PUBLISHED)
        self.assertEqual(published, build_record(DATASET))
        self.assertEqual(published["dataset"]["sha256"], digest(DATASET))

    def test_published_artifact_rejects_tampering(self):
        record = build_record(DATASET)
        record["splits"]["train"]["indices"][0] += 1
        with self.assertRaises(PipelineError):
            validate_record(record)


class ConfigRejectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = build_record(DATASET)

    def test_incompatible_protocol_configurations_are_rejected(self):
        invalid = [
            {**DEFAULT_CONFIG, "strategy": "non-iid"},
            {**DEFAULT_CONFIG, "fractions": {"train": 0.5, "validation": 0.25, "test": 0.25}},
            {**DEFAULT_CONFIG, "fractions": {"train": 0.6, "validation": 0.2}},
            {**DEFAULT_CONFIG, "split_seed": 45},
            {**DEFAULT_CONFIG, "validation_seed": 44},
            {**DEFAULT_CONFIG, "partition_seed": 46},
            {**DEFAULT_CONFIG, "n_clients": 4},
            {**DEFAULT_CONFIG, "extra": 1},
            {key: value for key, value in DEFAULT_CONFIG.items() if key != "strategy"},
        ]
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(PipelineError):
                build_record(DATASET, config)

    def test_invalid_split_and_partition_inputs_are_rejected(self):
        with self.assertRaises(SplitError):
            stratified_split([0, 0, 1], [0, 0, 1])
        with self.assertRaises(SplitError):
            stratified_split([0, 1], [0, 1], {"train": 1.0, "validation": 0.1, "test": 0.1})
        with self.assertRaises(PartitionError):
            iid_partition([0, 1, 2], n_clients=1)
        with self.assertRaises(PartitionError):
            iid_partition([0, 1, 2, 2], n_clients=3)

    def test_write_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "record.json"
            write_record(path, self.record)
            before = path.read_bytes()
            with self.assertRaises(PipelineError):
                write_record(path, build_record(DATASET))
            self.assertEqual(path.read_bytes(), before)


class DuplicateAuditTests(unittest.TestCase):
    def test_duplicate_groups_are_detected_and_registered(self):
        rows = [
            {TARGET: "0", "num": "1", "cat": "a"},   # duplicata exata da próxima
            {TARGET: "0", "num": "1", "cat": "a"},
            {TARGET: "0", "num": "7", "cat": "b"},   # mesmas features, rótulo oposto
            {TARGET: "1", "num": "7", "cat": "b"},
            {TARGET: "1", "num": "9", "cat": "c"},
        ]
        rows += [{TARGET: "0", "num": str(20 + i), "cat": "d"} for i in range(15)]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "duplicates.csv"
            write_csv(path, rows)
            record = build_record(path)
        audit = record["duplicate_audit"]
        self.assertEqual(audit["exact"], {"groups": 1, "rows": 2, "extra_rows": 1})
        self.assertEqual(audit["features"], {"groups": 2, "rows": 4, "extra_rows": 2})
        self.assertIsNone(audit["group_column"])
        self.assertIn("agrupamento", audit["group_audit_limitation"])

        split_of = {index: name for name in SPLIT_NAMES
                    for index in record["splits"][name]["indices"]}
        keys = [(row["num"], row["cat"]) for row in rows]
        members = {key: [i for i, k in enumerate(keys) if k == key]
                   for key in set(keys) if keys.count(key) > 1}
        expected_cross = sum(
            1 for group in members.values() if len({split_of[i] for i in group}) > 1
        )
        self.assertEqual(audit["cross_split_feature_groups"], expected_cross)


class SyntheticLeakageTests(unittest.TestCase):
    train = [
        {"num": 1.0, "cat": "a"}, {"num": 2.0, "cat": "b"},
        {"num": 3.0, "cat": None}, {"num": None, "cat": "a"},
    ]
    evaluation = [
        {"num": 1e12, "cat": "Zz"}, {"num": -1e12, "cat": "Qq"},
        {"num": None, "cat": None}, {"num": 0.0, "cat": "a"},
    ]

    def preprocessor(self):
        return FeaturePreprocessor(["num", "cat"]).fit(self.train)

    def test_statistics_are_learned_from_train_only(self):
        stats = self.preprocessor().stats()
        self.assertEqual(stats["numeric"]["num"]["median"], 2.0)
        self.assertEqual(stats["numeric"]["num"]["mean"], 2.0)
        self.assertEqual(stats["numeric"]["num"]["scale"],
                         float(statistics.pstdev([1.0, 2.0, 3.0, 2.0])))
        self.assertEqual(stats["categorical"]["cat"]["categories"], ["a", "b"])
        self.assertEqual(stats["categorical"]["cat"]["mode"], "a")
        self.assertEqual(stats["numeric"]["num"]["missing_in_train"], 1)
        self.assertEqual(stats["categorical"]["cat"]["missing_in_train"], 1)

    def test_stats_are_effectively_immutable(self):
        preprocessor = self.preprocessor()
        stats = preprocessor.stats()
        stats["categorical"]["cat"]["categories"].append("Zz")
        stats["categorical"]["cat"]["mode"] = "Zz"
        stats["numeric"]["num"]["median"] = 1e12
        fresh = preprocessor.stats()
        self.assertEqual(fresh["categorical"]["cat"]["categories"], ["a", "b"])
        self.assertEqual(fresh["categorical"]["cat"]["mode"], "a")
        self.assertEqual(fresh["numeric"]["num"]["median"], 2.0)
        vector = preprocessor.transform([{"num": 1.0, "cat": "b"}])[0]
        self.assertEqual(vector[1:], [0.0, 1.0])

    def test_extreme_evaluation_values_do_not_change_fitted_state(self):
        preprocessor = self.preprocessor()
        before = json.dumps(preprocessor.stats(), sort_keys=True)
        vectors = preprocessor.transform(self.evaluation)
        after = json.dumps(preprocessor.stats(), sort_keys=True)
        self.assertEqual(before, after)
        self.assertTrue(all(math.isfinite(value) for vector in vectors for value in vector))

    def test_imputation_and_unknown_category_block(self):
        preprocessor = self.preprocessor()
        vector = preprocessor.transform([{"num": None, "cat": "Zz"}])[0]
        stats = preprocessor.stats()
        self.assertEqual(vector[0], (stats["numeric"]["num"]["median"]
                                     - stats["numeric"]["num"]["mean"])
                         / stats["numeric"]["num"]["scale"])
        self.assertEqual(vector[1:], [0.0, 0.0])
        imputed = preprocessor.transform([{"num": None, "cat": None}])[0]
        self.assertEqual(imputed[1:], [1.0, 0.0])

    def test_refit_and_unfitted_transform_are_rejected(self):
        preprocessor = FeaturePreprocessor(["num", "cat"])
        with self.assertRaises(PreprocessingError):
            preprocessor.transform(self.train)
        preprocessor.fit(self.train)
        with self.assertRaises(PreprocessingError):
            preprocessor.fit(self.train)


class PipelineLeakageTests(unittest.TestCase):
    def test_evaluation_extremes_cannot_move_pipeline_statistics(self):
        rows = [{TARGET: "0", "num": str(index), "cat": "a" if index % 2 else "b"}
                for index in range(15)]
        rows += [{TARGET: "1", "num": str(100 + index), "cat": "c"} for index in range(5)]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "synthetic.csv"
            write_csv(path, rows)
            baseline = build_record(path)
            evaluation = set(baseline["splits"]["validation"]["indices"]) \
                | set(baseline["splits"]["test"]["indices"])
            perturbed = [
                {TARGET: row[TARGET], "num": "1e12", "cat": "Zz"}
                if position in evaluation else dict(row)
                for position, row in enumerate(rows)
            ]
            write_csv(path, perturbed)
            record = build_record(path)
            self.assertEqual(record["splits"], baseline["splits"])
            self.assertEqual(record["partitions"], baseline["partitions"])
            self.assertEqual(record["preprocessing_statistics"],
                             baseline["preprocessing_statistics"])
            # O conteúdo do CSV mudou, logo o SHA-256 do dataset (e os IDs dele
            # dependentes) muda; índices e estatísticas permanecem idênticos.
            self.assertNotEqual(record["dataset"]["sha256"], baseline["dataset"]["sha256"])
            self.assertNotIn("Zz",
                             record["preprocessing_statistics"]["categorical"]["cat"]["categories"])

    def test_identifiers_differ_for_distinct_datasets_with_equal_indices(self):
        rows_a = [{TARGET: "0", "num": str(index), "cat": "a"} for index in range(15)]
        rows_a += [{TARGET: "1", "num": str(100 + index), "cat": "b"} for index in range(5)]
        rows_b = [{TARGET: "0", "num": str(9000 + index), "cat": "z"} for index in range(15)]
        rows_b += [{TARGET: "1", "num": str(8000 + index), "cat": "y"} for index in range(5)]
        with tempfile.TemporaryDirectory() as temporary:
            path_a = Path(temporary) / "a.csv"
            path_b = Path(temporary) / "b.csv"
            write_csv(path_a, rows_a)
            write_csv(path_b, rows_b)
            record_a = build_record(path_a)
            record_b = build_record(path_b)
        self.assertEqual(record_a["splits"], record_b["splits"])
        self.assertEqual(record_a["partitions"], record_b["partitions"])
        self.assertNotEqual(record_a["split_id"], record_b["split_id"])
        self.assertNotEqual(record_a["partition_id"], record_b["partition_id"])


if __name__ == "__main__":
    unittest.main()
