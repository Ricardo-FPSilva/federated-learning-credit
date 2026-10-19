"""Testes E3.1A; validam infraestrutura sem criar ou treinar modelos."""

import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import tempfile
import unittest

try:
    import numpy as np
except ImportError:
    np = None
try:
    import torch
except ImportError:
    torch = None
try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except ImportError:
    average_precision_score = None
    roc_auc_score = None

from contracts.validation import relative_path
from research.benchmark.data import BenchmarkDataError, load_experimental_data
from research.benchmark.evaluation import (
    EvaluationError, evaluate_global, evaluate_scores, weighted_batch_loss,
)
from research.benchmark.provenance import (
    DEFAULT_SOURCE_FILES, ProvenanceError, artifact_id, build_provenance,
    file_reference,
)
from research.benchmark.seeding import (
    SeedingError, canonical_json_bytes, configure_determinism, derive_seed,
    paired_initialization_seed, shuffle_seed,
)
from research.data.pipeline import load_dataset, load_record


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "hmeq.csv"
DATA_RECORD = ROOT / "results/data/benchmark-v1-split.json"
EXECUTION_SPEC = ROOT / "protocols/benchmark-v1-execution.json"


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


class ExperimentalDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_experimental_data(DATASET, DATA_RECORD)
        cls.record = load_record(DATA_RECORD)

    def test_reconstructs_frozen_splits_and_feature_order(self):
        self.assertEqual(len(self.data.train), 3576)
        self.assertEqual(len(self.data.validation), 1192)
        self.assertEqual(len(self.data.test), 1192)
        self.assertEqual(self.data.feature_order, tuple(self.record["feature_order"]))
        self.assertEqual(self.data.output_features, tuple(self.record["output_features"]))
        self.assertEqual(len(self.data.output_features), 18)
        self.assertEqual(self.data.train.indices,
                         tuple(self.record["splits"]["train"]["indices"]))
        self.assertEqual(self.data.validation.indices,
                         tuple(self.record["splits"]["validation"]["indices"]))
        self.assertEqual(self.data.test.indices,
                         tuple(self.record["splits"]["test"]["indices"]))
        self.assertTrue(all(len(row) == 18 for row in self.data.train.features))

    def test_reuses_recorded_statistics_without_refit(self):
        rows, _, _ = load_dataset(DATASET)
        original_index = self.data.train.indices[0]
        row = rows[original_index]
        stats = self.record["preprocessing_statistics"]
        loan = float(row["LOAN"])
        expected_loan = (loan - stats["numeric"]["LOAN"]["mean"]) \
            / stats["numeric"]["LOAN"]["scale"]
        vector = self.data.train.features[0]
        self.assertEqual(vector[0], expected_loan)
        reason = row["REASON"] or stats["categorical"]["REASON"]["mode"]
        categories = stats["categorical"]["REASON"]["categories"]
        self.assertEqual(vector[3:5], tuple(1.0 if reason == c else 0.0 for c in categories))

    def test_client_training_views_are_disjoint_and_complete(self):
        self.assertEqual(set(self.data.clients), {"client-0", "client-1", "client-2"})
        covered = set()
        for client in self.data.clients:
            view = self.data.client_train(client)
            self.assertEqual(len(view), 1192)
            self.assertFalse(covered.intersection(view.indices))
            covered.update(view.indices)
        self.assertEqual(covered, set(self.data.train.indices))
        self.assertFalse(covered.intersection(self.data.validation.indices))
        self.assertFalse(covered.intersection(self.data.test.indices))
        with self.assertRaises(BenchmarkDataError):
            self.data.client_train("client-3")

    def test_exposes_frozen_identifiers(self):
        self.assertEqual(self.data.dataset_sha256, self.record["dataset"]["sha256"])
        self.assertEqual(self.data.split_id, self.record["split_id"])
        self.assertEqual(self.data.partition_id, self.record["partition_id"])
        self.assertEqual(self.data.preprocessing_id, self.record["preprocessing"]["id"])

    def test_rejects_tampered_e2_record(self):
        value = json.loads(DATA_RECORD.read_text(encoding="utf-8"))
        value["splits"]["train"]["indices"][0] += 1
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tampered.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_experimental_data(DATASET, path)

    def test_rejects_tampered_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "hmeq.csv"
            path.write_bytes(DATASET.read_bytes() + b"\n")
            with self.assertRaisesRegex(BenchmarkDataError, "SHA-256"):
                load_experimental_data(path, DATA_RECORD)


class SeedingTests(unittest.TestCase):
    def test_canonical_json_and_seed_match_independent_sha256(self):
        payload = {
            "execution_spec_id": "benchmark-v1-execution-1",
            "training_seed": 42,
            "stream": "shuffle",
            "model": "central",
            "client_id": None,
            "step": 1,
            "epoch": 2,
        }
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                               separators=(",", ":"), allow_nan=False).encode("utf-8")
        expected = int.from_bytes(hashlib.sha256(canonical).digest()[:8], "big") % (2**63 - 1)
        self.assertEqual(canonical_json_bytes(payload), canonical)
        self.assertEqual(derive_seed(42, "shuffle", "central", None, 1, 2), expected)
        self.assertEqual(derive_seed(42, "shuffle", "central", None, 1, 2), expected)

    def test_namespaces_are_separate_and_initialization_is_paired(self):
        initialization = paired_initialization_seed(44)
        self.assertEqual(initialization, paired_initialization_seed(44))
        central = shuffle_seed(44, "central", client_id=None, step=1, epoch=0)
        federated = shuffle_seed(44, "federated", client_id="client-0", step=1, epoch=0)
        local = shuffle_seed(44, "local", client_id="client-0", step=1, epoch=0)
        self.assertEqual(len({initialization, central, federated, local}), 4)
        self.assertNotEqual(federated,
                            shuffle_seed(44, "federated", client_id="client-1", step=1, epoch=0))

    def test_rejects_unknown_seed_or_namespace(self):
        with self.assertRaises(SeedingError):
            paired_initialization_seed(47)
        with self.assertRaises(SeedingError):
            shuffle_seed(42, "other", client_id=None, step=0, epoch=0)
        with self.assertRaises(SeedingError):
            shuffle_seed(42, "central", client_id="client-0", step=0, epoch=0)

    def test_configures_supplied_numpy_and_torch_without_importing_them(self):
        class FakeRandom:
            value = None

            @classmethod
            def seed(cls, value):
                cls.value = value

        class FakeNumpy:
            random = FakeRandom

        class FakeTorch:
            seed = None
            threads = None
            interop = None
            deterministic = None

            @classmethod
            def manual_seed(cls, value):
                cls.seed = value

            @classmethod
            def set_num_threads(cls, value):
                cls.threads = value

            @classmethod
            def get_num_threads(cls):
                return cls.threads

            @classmethod
            def set_num_interop_threads(cls, value):
                cls.interop = value

            @classmethod
            def get_num_interop_threads(cls):
                return cls.interop

            @classmethod
            def use_deterministic_algorithms(cls, value):
                cls.deterministic = value

        status = configure_determinism(123, numpy_module=FakeNumpy, torch_module=FakeTorch,
                                       environ={})
        self.assertTrue(status["python_random_seeded"])
        self.assertTrue(status["numpy_seeded"])
        torch_status = status["torch"]
        self.assertTrue(torch_status["configured"])
        self.assertTrue(torch_status["manual_seed"])
        self.assertTrue(torch_status["deterministic_algorithms"])
        self.assertEqual(torch_status["threads_requested"], 1)
        self.assertEqual(torch_status["threads_effective"], 1)
        self.assertEqual(torch_status["interop_threads_requested"], 1)
        self.assertEqual(torch_status["interop_threads_effective"], 1)
        self.assertIsNone(torch_status["interop_error"])
        self.assertEqual(FakeRandom.value, 123)
        self.assertEqual(FakeTorch.seed, 123)
        self.assertEqual(FakeTorch.threads, 1)
        self.assertIs(FakeTorch.deterministic, True)
        self.assertEqual(random.Random(123).random(), random.random())

    def test_distinguishes_requested_from_unverified_environment_settings(self):
        status = configure_determinism(7, environ={"OMP_NUM_THREADS": "8",
                                                   "PYTHONHASHSEED": "0"})
        self.assertIsNone(status["torch"]["threads_effective"])
        self.assertIsNone(status["torch"]["threads_requested"])
        self.assertTrue(any("torch não fornecido" in item for item in status["limitations"]))
        self.assertEqual(status["environment_variables"]["OMP_NUM_THREADS"], "8")
        self.assertEqual(status["environment_variables"]["PYTHONHASHSEED"], "0")
        self.assertTrue(any("não constituem verificação" in item
                            for item in status["limitations"]))
        empty = configure_determinism(7, environ={})
        self.assertIsNone(empty["environment_variables"]["PYTHONHASHSEED"])
        self.assertTrue(any("PYTHONHASHSEED ausente" in item for item in empty["limitations"]))

    @unittest.skipUnless(torch is not None, "PyTorch indisponível")
    def test_real_torch_threads_are_set_and_verified(self):
        status = configure_determinism(42, torch_module=torch, environ={})
        torch_status = status["torch"]
        self.assertTrue(torch_status["manual_seed"])
        self.assertTrue(torch_status["deterministic_algorithms"])
        self.assertEqual(torch_status["threads_effective"], 1)
        if torch_status["interop_error"] is None:
            self.assertEqual(torch_status["interop_threads_effective"], 1)
        else:
            self.assertTrue(any("interop" in item for item in status["limitations"]))


class EvaluationTests(unittest.TestCase):
    def test_threshold_matrix_and_metrics(self):
        labels = [0, 0, 1, 1]
        scores = [0.1, 0.6, 0.4, 0.9]
        result = evaluate_scores(labels, scores)
        self.assertEqual(result["predicted_labels"], [0, 1, 0, 1])
        self.assertEqual(result["confusion_matrix"]["counts"], [[1, 1], [1, 1]])
        for name in ("accuracy", "precision", "recall", "f1"):
            self.assertEqual(result["values"][name], 0.5)
        self.assertAlmostEqual(result["values"]["average_precision"], 5 / 6)
        self.assertEqual(result["values"]["roc_auc"], 0.75)
        expected_bce = -sum(
            math.log(score) if label else math.log1p(-score)
            for label, score in zip(labels, scores)
        ) / len(labels)
        self.assertAlmostEqual(result["values"]["bce_loss"], expected_bce)

    def test_threshold_is_strictly_greater_than_half(self):
        result = evaluate_scores([1, 1], [0.5, 0.5000000001])
        self.assertEqual(result["predicted_labels"], [0, 1])
        self.assertEqual(result["confusion_matrix"]["counts"], [[0, 0], [1, 1]])

    def test_single_class_and_undefined_metrics(self):
        negatives = evaluate_scores([0, 0], [0.1, 0.2])
        for name in ("precision", "recall", "f1", "average_precision", "roc_auc"):
            self.assertIsNone(negatives["values"][name], name)
            self.assertTrue(negatives["undefined_reasons"][name])
        positives = evaluate_scores([1, 1], [0.8, 0.9])
        self.assertEqual(positives["values"]["average_precision"], 1.0)
        self.assertIsNone(positives["values"]["roc_auc"])

    def test_infinite_bce_is_explicitly_undefined_not_clipped(self):
        result = evaluate_scores([1], [0.0])
        self.assertIsNone(result["values"]["bce_loss"])
        self.assertIn("infinita", result["undefined_reasons"]["bce_loss"])

    def test_bce_extreme_logits_are_finite_without_probability_conversion(self):
        # scores coerentes com a sigmoid saturada dos logits ±1000
        result = evaluate_scores([1, 0], [0.0, 1.0], logits=[-1000.0, 1000.0])
        self.assertEqual(result["values"]["bce_loss"], 1000.0)
        self.assertIsNone(result["undefined_reasons"]["bce_loss"])
        self.assertEqual(result["predicted_labels"], [0, 1])
        result = evaluate_scores([1, 0], [1.0, 0.0], logits=[1000.0, -1000.0])
        self.assertAlmostEqual(result["values"]["bce_loss"], 0.0, places=9)
        # a BCE finita vem dos logits: score perfeito 1.0 daria loss infinita
        # pelo caminho de probabilidade, mas o logit errado produz 1000.0
        result = evaluate_scores([0], [1.0], logits=[1000.0])
        self.assertEqual(result["values"]["bce_loss"], 1000.0)
        self.assertIsNone(result["undefined_reasons"]["bce_loss"])

    @unittest.skipUnless(torch is not None, "PyTorch indisponível")
    def test_bce_logits_equivalence_with_pytorch_when_available(self):
        logits = [-1000.0, -745.0, -100.0, -1.5, -0.001, 0.0, 0.001, 1.5, 100.0, 745.0, 1000.0]
        labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        scores = [_sigmoid(value) for value in logits]
        tensor_logits = torch.tensor(logits, dtype=torch.float64)
        tensor_labels = torch.tensor(labels, dtype=torch.float64)
        expected_mean = torch.nn.functional.binary_cross_entropy_with_logits(
            tensor_logits, tensor_labels, reduction="mean").item()
        expected_sum = torch.nn.functional.binary_cross_entropy_with_logits(
            tensor_logits, tensor_labels, reduction="sum").item()
        result = evaluate_scores(labels, scores, logits=logits)
        self.assertTrue(math.isclose(result["values"]["bce_loss"], expected_mean,
                                     rel_tol=1e-12, abs_tol=1e-12))
        self.assertTrue(math.isclose(result["values"]["bce_loss"],
                                     expected_sum / len(labels),
                                     rel_tol=1e-12, abs_tol=1e-12))

    @unittest.skipUnless(np is not None, "NumPy indisponível")
    def test_numpy_scalars_are_accepted_and_normalized(self):
        labels = np.array([0, 1, 1, 0], dtype=np.int64)
        logits = np.array([-2.0, -0.4, 0.4, 2.0], dtype=np.float32)
        scores = np.float32([_sigmoid(value) for value in (-2.0, -0.4, 0.4, 2.0)])
        result = evaluate_scores(labels, scores, logits=logits)
        self.assertEqual(result["sample_count"], 4)
        self.assertTrue(all(type(label) is int for label in result["predicted_labels"]))
        self.assertTrue(all(type(value) is float
                            for value in result["values"].values() if value is not None))
        json.dumps(result, ensure_ascii=False, allow_nan=False)
        self.assertEqual(weighted_batch_loss([(np.float64(2.0), np.int64(3))]), 2.0)
        records = [(np.array([5]), np.array([1]), np.array([0.9], dtype=np.float32))]
        global_result = evaluate_global(records, np.array([5], dtype=np.int64))
        self.assertEqual(global_result["sample_count"], 1)

    def test_rejects_incoherent_score_logit_pairs(self):
        contradictions = (
            ([1], [0.0], [1000.0]),   # auditoria E3.1A.1: loss 0 vs classe 0
            ([0], [1.0], [-1000.0]),
            ([0], [0.9], [-2.0]),     # Sigmoid(-2)=0.1192029...
            ([1], [0.1], [2.0]),      # Sigmoid(2)=0.8807970...
            ([0], [0.12], [-2.0]),    # arredondado a 2 casas, além da tolerância
        )
        for labels, scores, logits in contradictions:
            with self.subTest(scores=scores, logits=logits), \
                    self.assertRaisesRegex(EvaluationError, "incoerente"):
                evaluate_scores(labels, scores, logits=logits)

    def test_accepts_rounded_scores_within_tolerance(self):
        logits = [-2.0, -0.5, 0.5, 2.0]
        labels = [0, 0, 1, 1]
        derived = [_sigmoid(value) for value in logits]
        reference = evaluate_scores(labels, derived, logits=logits)
        for digits in (6, 4):
            with self.subTest(digits=digits):
                result = evaluate_scores(
                    labels, [round(value, digits) for value in derived], logits=logits)
                self.assertEqual(result["values"], reference["values"])
                self.assertEqual(result["predicted_labels"], reference["predicted_labels"])
        with self.subTest(digits=2), self.assertRaisesRegex(EvaluationError, "incoerente"):
            evaluate_scores(labels, [round(value, 2) for value in derived], logits=logits)

    def test_near_zero_logits_preserve_decision_and_loss(self):
        exact = evaluate_scores([1], [0.5], logits=[0.0])
        self.assertEqual(exact["predicted_labels"], [0])
        self.assertTrue(math.isclose(exact["values"]["bce_loss"], math.log(2.0),
                                     rel_tol=1e-12, abs_tol=1e-12))
        # logit ligeiramente positivo decide classe 1 mesmo com score arredondado 0.5
        tiny = evaluate_scores([0], [0.5], logits=[1e-8])
        self.assertEqual(tiny["predicted_labels"], [1])
        self.assertTrue(math.isclose(tiny["values"]["bce_loss"], math.log(2.0),
                                     rel_tol=1e-6, abs_tol=1e-6))
        negative = evaluate_scores([0], [0.5], logits=[-1e-8])
        self.assertEqual(negative["predicted_labels"], [0])

    def test_score_decision_and_loss_align_under_logits(self):
        logits = [-3.0, -0.2, 0.3, 4.0]
        labels = [1, 0, 0, 1]
        scores = [_sigmoid(value) for value in logits]
        result = evaluate_scores(labels, scores, logits=logits)
        for logit, prediction in zip(logits, result["predicted_labels"]):
            self.assertEqual(prediction, 1 if logit > 0.0 else 0)
        self.assertEqual(result["confusion_matrix"]["counts"], [[1, 1], [1, 1]])
        expected_bce = sum(
            max(value, 0.0) - value * label + math.log1p(math.exp(-abs(value)))
            for value, label in zip(logits, labels)
        ) / len(labels)
        self.assertTrue(math.isclose(result["values"]["bce_loss"], expected_bce,
                                     rel_tol=1e-12, abs_tol=1e-12))

    def test_rejects_booleans_nan_and_infinite_values(self):
        invalid_calls = (
            lambda: evaluate_scores([True], [0.5]),
            lambda: evaluate_scores([0], [True]),
            lambda: evaluate_scores([0], [0.5], logits=[True]),
            lambda: evaluate_scores([0], [0.5], logits=[math.inf]),
            lambda: evaluate_scores([0], [0.5], logits=[math.nan]),
            lambda: evaluate_scores([0], [0.5], logits=[0.0, 1.0]),
            lambda: weighted_batch_loss([(True, 4)]),
            lambda: weighted_batch_loss([(1.0, True)]),
        )
        for call in invalid_calls:
            with self.subTest(call=call), self.assertRaises(EvaluationError):
                call()

    def test_global_rejects_duplicate_missing_and_out_of_scope_indices(self):
        expected = [10, 11, 12]
        with self.assertRaisesRegex(EvaluationError, "duplicado"):
            evaluate_global([([10, 10], [1, 0], [0.9, 0.1])], expected)
        with self.assertRaisesRegex(EvaluationError, "duplicado"):
            evaluate_global([([10], [1], [0.9]), ([10], [0], [0.1])], expected)
        with self.assertRaisesRegex(EvaluationError, "escopo"):
            evaluate_global([([10, 11, 99], [1, 0, 1], [0.9, 0.1, 0.5])], expected)
        with self.assertRaisesRegex(EvaluationError, "cobertura"):
            evaluate_global([([10, 11], [1, 0], [0.9, 0.1])], expected)
        with self.assertRaises(EvaluationError):
            evaluate_global([], expected)
        with self.assertRaises(EvaluationError):
            evaluate_global([([10], [1], [0.9])], [])

    def test_global_is_independent_of_record_order(self):
        expected = {1, 2, 3, 4}
        first = evaluate_global([([3, 4], [1, 0], [0.9, 0.1]),
                                 ([1, 2], [1, 1], [0.8, 0.7])], expected)
        second = evaluate_global([([1, 2], [1, 1], [0.8, 0.7]),
                                  ([3, 4], [1, 0], [0.9, 0.1])], expected)
        direct = evaluate_scores([1, 1, 1, 0], [0.8, 0.7, 0.9, 0.1])
        self.assertEqual(first, direct)
        self.assertEqual(second, direct)

    def test_global_computes_bce_from_aligned_logits(self):
        result = evaluate_global([([0, 1], [1, 0], [0.0, 0.0])], [0, 1],
                                 logits=[[-1000.0, -1000.0]])
        self.assertEqual(result["values"]["bce_loss"], 500.0)
        self.assertIsNone(result["undefined_reasons"]["bce_loss"])

    def test_global_metrics_are_recomputed_from_concatenated_predictions(self):
        records = [([10], [1], [0.9]), ([11, 12, 13], [1, 1, 0], [0.1, 0.1, 0.9])]
        global_result = evaluate_global(records, {10, 11, 12, 13})
        direct = evaluate_scores([1, 1, 1, 0], [0.9, 0.1, 0.1, 0.9])
        self.assertEqual(global_result, direct)
        local_f1_mean = sum(
            evaluate_scores(record[1], record[2])["values"]["f1"] for record in records) / 2
        self.assertNotEqual(global_result["values"]["f1"], local_f1_mean)

    def test_ties_in_ap_and_roc_auc_use_reference_values(self):
        cases = (
            ([0, 1, 1, 0], [0.5, 0.5, 0.5, 0.5], 0.5, 0.5),
            ([0, 1, 0, 1, 1, 0], [0.3, 0.3, 0.7, 0.7, 0.7, 0.2],
             0.6444444444444444, 0.7222222222222222),
            ([0, 1, 0, 1], [0.4, 0.6, 0.6, 0.9],
             0.8333333333333333, 0.875),
        )
        for labels, scores, expected_ap, expected_auc in cases:
            with self.subTest(scores=scores):
                values = evaluate_scores(labels, scores)["values"]
                self.assertAlmostEqual(values["average_precision"], expected_ap, places=12)
                self.assertAlmostEqual(values["roc_auc"], expected_auc, places=12)

    @unittest.skipUnless(average_precision_score is not None, "scikit-learn indisponível")
    def test_ties_match_sklearn_when_available(self):
        labels = [0, 1, 0, 1, 1, 0, 0, 1]
        scores = [0.7, 0.7, 0.3, 0.3, 0.7, 0.3, 0.9, 0.9]
        values = evaluate_scores(labels, scores)["values"]
        self.assertAlmostEqual(values["average_precision"],
                               average_precision_score(labels, scores), places=12)
        self.assertAlmostEqual(values["roc_auc"],
                               roc_auc_score(labels, scores), places=12)

    def test_results_are_strictly_serializable(self):
        undefined = evaluate_scores([0, 0], [0.1, 0.2])
        text = json.dumps(undefined, ensure_ascii=False, allow_nan=False)
        self.assertIn("null", text)
        extreme = evaluate_scores([1, 0], [0.0, 1.0], logits=[-1000.0, 1000.0])
        decoded = json.loads(canonical_json_bytes(extreme).decode("utf-8"))
        self.assertEqual(decoded["values"]["bce_loss"], 1000.0)
        self.assertTrue(all(value is None or math.isfinite(value)
                            for value in extreme["values"].values()))

    def test_rejects_invalid_labels_and_scores(self):
        for labels, scores in (([0, 2], [0.1, 0.2]), ([0], [math.nan]), ([0], [1.1])):
            with self.subTest(labels=labels, scores=scores), self.assertRaises(EvaluationError):
                evaluate_scores(labels, scores)

    def test_batch_loss_is_weighted_by_real_batch_size(self):
        self.assertEqual(weighted_batch_loss([(1.0, 32), (3.0, 4)]), 44 / 36)
        self.assertNotEqual(weighted_batch_loss([(1.0, 32), (3.0, 4)]), 2.0)
        with self.assertRaises(EvaluationError):
            weighted_batch_loss([])


class ProvenanceTests(unittest.TestCase):
    def _build(self, **overrides):
        kwargs = dict(
            dataset_path=DATASET, data_record_path=DATA_RECORD,
            execution_spec_path=EXECUTION_SPEC, training_seed=42,
            configuration={"stage": "infrastructure-test", "models_trained": False},
        )
        kwargs.update(overrides)
        return build_provenance(ROOT, **kwargs)

    def test_builds_complete_deterministic_provenance(self):
        first = self._build()
        second = self._build()
        self.assertEqual(first, second)
        record = load_record(DATA_RECORD)
        identity = first["scientific_identity"]
        self.assertEqual(identity["protocol_id"], "benchmark-v1")
        self.assertEqual(identity["execution_spec_id"], "benchmark-v1-execution-1")
        self.assertEqual(identity["dataset_sha256"], record["dataset"]["sha256"])
        self.assertEqual(identity["split_id"], record["split_id"])
        self.assertEqual(identity["partition_id"], record["partition_id"])
        self.assertEqual(identity["preprocessing_id"], record["preprocessing"]["id"])
        self.assertEqual(identity["training_seed"], 42)
        self.assertRegex(identity["source_commit"], r"^[0-9a-f]{40}$")
        self.assertEqual(set(identity["source_files"]), set(DEFAULT_SOURCE_FILES))
        for relative, digest in identity["source_files"].items():
            self.assertEqual(digest,
                             hashlib.sha256((ROOT / relative).read_bytes()).hexdigest())
        operational = first["operational"]
        self.assertIs(type(operational["source_dirty"]), bool)
        self.assertNotIn("source_dirty", identity)
        self.assertNotIn("environment", identity)
        environment = operational["environment"]
        self.assertEqual(environment["device"], "cpu")
        self.assertIsNone(environment["determinism"])
        self.assertEqual(set(environment["packages"]),
                         {"torch", "numpy", "scikit-learn"})
        self.assertNotIn("threads", environment)
        self.assertTrue(any("não verificados" in item for item in environment["limitations"]))
        canonical_json_bytes(first)

    def test_determinism_report_is_operational_and_does_not_change_identity(self):
        report = configure_determinism(42, environ={"OMP_NUM_THREADS": "1"})
        with_report = self._build(determinism_report=report)
        without_report = self._build()
        self.assertEqual(with_report["run_id"], without_report["run_id"])
        recorded = with_report["operational"]["environment"]["determinism"]
        self.assertEqual(recorded, report)
        self.assertIsNone(without_report["operational"]["environment"]["determinism"])

    def test_run_and_artifact_identity_changes_with_seed_or_scope(self):
        kwargs = dict(
            repo_root=ROOT, dataset_path=DATASET, data_record_path=DATA_RECORD,
            execution_spec_path=EXECUTION_SPEC, configuration={"frozen": True},
        )
        run_42 = build_provenance(training_seed=42, **kwargs)["run_id"]
        run_43 = build_provenance(training_seed=43, **kwargs)["run_id"]
        self.assertNotEqual(run_42, run_43)
        self.assertEqual(artifact_id(run_42, "metrics", scope="test"),
                         artifact_id(run_42, "metrics", scope="test"))
        self.assertNotEqual(artifact_id(run_42, "metrics", scope="test"),
                            artifact_id(run_42, "metrics", scope="validation"))

    def test_file_reference_is_relative_and_contract_compatible(self):
        reference = file_reference(ROOT, EXECUTION_SPEC)
        self.assertEqual(reference["path"], "protocols/benchmark-v1-execution.json")
        self.assertFalse(reference["path"].startswith("/"))
        relative_path(reference["path"], "reference.path")
        self.assertEqual(reference["sha256"],
                         hashlib.sha256(EXECUTION_SPEC.read_bytes()).hexdigest())

    def test_file_reference_rejects_paths_outside_root_or_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            external = Path(temporary) / "externo.json"
            external.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ProvenanceError, "fora da raiz"):
                file_reference(ROOT, external)
        with self.assertRaisesRegex(ProvenanceError, "não regular"):
            file_reference(ROOT, ROOT / "arquivo-inexistente.json")
        with self.assertRaises(ProvenanceError):
            file_reference(ROOT, ROOT / "protocols")
        with self.assertRaisesRegex(ProvenanceError, "fora da raiz"):
            file_reference(ROOT / "protocols", DATASET)

    def test_invalid_training_seed_is_rejected(self):
        with self.assertRaises(ProvenanceError):
            self._build(training_seed=47)

    def test_dirty_states_differ_and_artifacts_do_not_change_identity(self):
        kwargs = dict(
            dataset_path=DATASET, data_record_path=DATA_RECORD,
            execution_spec_path=EXECUTION_SPEC, training_seed=42,
            configuration={"scenario": "identity-audit"},
            source_files=("mod.py",),
        )
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            environment = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1",
                           "HOME": str(repository)}

            def git(*arguments):
                subprocess.run(["git", *arguments], cwd=repository, check=True,
                               capture_output=True, text=True, env=environment)

            (repository / "mod.py").write_text("v1\n", encoding="utf-8")
            git("init", "-q")
            git("add", "mod.py")
            git("-c", "user.email=bench@example.com", "-c", "user.name=bench",
                "-c", "commit.gpgsign=false", "commit", "-q", "-m", "base")
            clean = build_provenance(repository, **kwargs)
            self.assertFalse(clean["operational"]["source_dirty"])
            (repository / "results").mkdir()
            (repository / "results" / "metrics.json").write_text("{}", encoding="utf-8")
            with_artifact = build_provenance(repository, **kwargs)
            self.assertEqual(with_artifact["run_id"], clean["run_id"])
            self.assertTrue(with_artifact["operational"]["source_dirty"])
            (repository / "mod.py").write_text("v2 - correcao A\n", encoding="utf-8")
            dirty_a = build_provenance(repository, **kwargs)
            (repository / "mod.py").write_text("v3 - correcao B\n", encoding="utf-8")
            dirty_b = build_provenance(repository, **kwargs)
            self.assertNotEqual(dirty_a["run_id"], dirty_b["run_id"])
            self.assertNotEqual(dirty_a["run_id"], clean["run_id"])
            self.assertTrue(dirty_a["operational"]["source_dirty"])
            self.assertNotEqual(
                dirty_a["scientific_identity"]["source_files"]["mod.py"],
                dirty_b["scientific_identity"]["source_files"]["mod.py"],
            )

    def test_default_source_manifest_covers_executed_python_files(self):
        covered = set(DEFAULT_SOURCE_FILES)
        found = set()
        for directory in ("research", "research/benchmark", "research/data", "contracts"):
            found.update(path.relative_to(ROOT).as_posix()
                         for path in (ROOT / directory).glob("*.py"))
        self.assertEqual(found, covered,
                         "manifesto default diverge dos arquivos .py executados; "
                         "atualize DEFAULT_SOURCE_FILES")

    def test_omitted_executed_dependency_changes_run_id(self):
        kwargs = dict(
            dataset_path=DATASET, data_record_path=DATA_RECORD,
            execution_spec_path=EXECUTION_SPEC, training_seed=42,
            configuration={"scenario": "default-manifest-audit"},
        )
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            environment = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1",
                           "HOME": str(repository)}

            def git(*arguments):
                subprocess.run(["git", *arguments], cwd=repository, check=True,
                               capture_output=True, text=True, env=environment)

            for relative in DEFAULT_SOURCE_FILES:
                target = repository / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((ROOT / relative).read_bytes())
            git("init", "-q")
            git("add", ".")
            git("-c", "user.email=bench@example.com", "-c", "user.name=bench",
                "-c", "commit.gpgsign=false", "commit", "-q", "-m", "base")
            before = build_provenance(repository, **kwargs)
            # dependência executada omitida no manifesto anterior ao E3.1A.2
            dependency = repository / "research/data/pipeline.py"
            dependency.write_bytes(dependency.read_bytes() + b"# alteracao\n")
            after = build_provenance(repository, **kwargs)
            self.assertNotEqual(before["run_id"], after["run_id"])
            # artefato de saída não altera a identidade já mudada pela dependência
            (repository / "results").mkdir()
            (repository / "results" / "metrics.json").write_text("{}", encoding="utf-8")
            with_artifact = build_provenance(repository, **kwargs)
            self.assertEqual(with_artifact["run_id"], after["run_id"])


if __name__ == "__main__":
    unittest.main()
