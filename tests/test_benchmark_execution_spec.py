"""Validação documental da especificação E3.0; não executa modelos."""

import hashlib
import json
import math
from pathlib import Path
import unittest

from contracts import validate_protocol
from contracts.validation import strict_json


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "protocols/benchmark-v1-execution.json"
PARENT_PATH = ROOT / "protocols/benchmark-v1.json"
DATA_PATH = ROOT / "results/data/benchmark-v1-split.json"


def git_blob(path):
    data = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


class BenchmarkExecutionSpecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = strict_json(SPEC_PATH.read_text(encoding="utf-8"))
        cls.parent = strict_json(PARENT_PATH.read_text(encoding="utf-8"))
        cls.data = strict_json(DATA_PATH.read_text(encoding="utf-8"))
        cls.rules = {}
        for rule in cls.spec["rules"]:
            key, separator, value = rule.partition(": ")
            if separator:
                cls.rules[key] = value

    def test_valid_frozen_protocol_envelope(self):
        self.assertIs(validate_protocol(self.spec), self.spec)
        self.assertEqual(self.spec["protocol_id"], "benchmark-v1-execution-1")
        self.assertEqual(self.spec["status"], "frozen")
        self.assertEqual(self.spec["kind"], "corrected")
        self.assertEqual(len(self.rules), len(self.spec["rules"]))

    def test_parent_and_e2_sources_are_immutable_refs(self):
        refs = {ref["path"]: ref for ref in self.spec["source_refs"]}
        self.assertEqual(set(refs), {
            "protocols/benchmark-v1.json", "results/data/benchmark-v1-split.json"
        })
        self.assertEqual(refs["protocols/benchmark-v1.json"]["blob"], git_blob(PARENT_PATH))
        self.assertEqual(refs["results/data/benchmark-v1-split.json"]["blob"], git_blob(DATA_PATH))
        self.assertEqual(self.parent["protocol_id"], "benchmark-v1")
        self.assertIn("benchmark-v1", self.rules["parent_protocol"])

    def test_e2_identifiers_and_input_dimension_are_bound(self):
        binding = self.rules["e2_binding"]
        expected = {
            "dataset_sha256": self.data["dataset"]["sha256"],
            "split_id": self.data["split_id"],
            "partition_id": self.data["partition_id"],
            "preprocessing_id": self.data["preprocessing"]["id"],
        }
        for name, value in expected.items():
            self.assertIn(f"{name}={value}", binding)
        self.assertEqual(len(self.data["output_features"]), 18)
        self.assertIn("18 output_features", self.rules["input"])
        self.assertIn("18->16->8->1", self.rules["mlp_architecture"])
        self.assertIn("449 parâmetros", self.rules["mlp_architecture"])
        self.assertEqual(18 * 16 + 16 + 16 * 8 + 8 + 8 + 1, 449)

    def test_approved_models_and_training_are_explicit(self):
        self.assertIn("C=1.0", self.rules["logistic_regression"])
        self.assertIn("solver=lbfgs", self.rules["logistic_regression"])
        self.assertIn("n_estimators=100", self.rules["random_forest"])
        self.assertIn("n_jobs=1", self.rules["random_forest"])
        self.assertIn("BCEWithLogitsLoss", self.rules["mlp_loss"])
        self.assertIn("Sigmoid", self.rules["mlp_loss"])
        self.assertIn("42, 43, 44, 45 e 46", self.rules["training_seeds"])
        self.assertIn("dez rounds", self.rules["mlp_schedule"])
        self.assertIn("Adam lr=0.01", self.rules["mlp_schedule"])
        self.assertIn("coordenador determinístico in-process", self.rules["fedavg"])

    def test_seed_and_serialization_rules_are_unambiguous(self):
        self.assertIn("chaves de objetos em ordem lexicográfica", self.rules["canonical_serialization"])
        self.assertIn("allow_nan=false", self.rules["canonical_serialization"])
        derivation = self.rules["seed_derivation"]
        for token in ("SHA-256", "execution_spec_id", "training_seed", "stream",
                      "model", "client_id", "step", "epoch", "2^63-1"):
            self.assertIn(token, derivation)
        self.assertIn("hash() do Python é proibido", derivation)
        self.assertIn("único state_dict inicial", self.rules["paired_initialization"])
        self.assertIn("independentes", self.rules["optimizer_state"])

    def test_no_test_leakage_and_client_test_scope(self):
        self.assertIn("não seleciona arquitetura", self.rules["validation_usage"])
        self.assertIn("avaliado uma vez", self.rules["test_usage"])
        self.assertIn("jamais orientam decisão adaptativa", self.rules["test_usage"])
        self.assertIn("exclusivamente em avaliações secundárias", self.rules["client_test_usage"])
        self.assertIn("modelos globais", self.rules["client_test_usage"])
        self.assertIn("teste global como referência principal", self.rules["client_test_usage"])
        for prohibition in ("treino", "ajuste de hiperparâmetros", "seleção de checkpoint",
                            "seleção de seed", "decisão adaptativa"):
            self.assertIn(prohibition, self.rules["client_test_prohibitions"])
        self.assertIn("não representa instituições reais", self.rules["client_test_prohibitions"])

    def test_metrics_budget_failures_and_traces_are_frozen(self):
        for metric in ("bce_loss", "accuracy", "precision", "recall", "f1",
                       "average_precision", "roc_auc"):
            self.assertIn(metric, self.rules["metrics"])
        self.assertIn("predições individuais concatenados", self.rules["metric_computation"])
        self.assertIn("reduction=sum", self.rules["loss_aggregation"])
        train = self.data["splits"]["train"]["size"]
        clients = [block["size"] for block in self.data["partitions"].values()]
        self.assertEqual(train * 50, 178800)
        self.assertEqual(math.ceil(train / 32) * 50, 5600)
        self.assertEqual(sum(math.ceil(size / 32) * 50 for size in clients), 5700)
        for value in ("178800", "5600", "5700"):
            self.assertIn(value, self.rules["budget_accounting"])
        self.assertIn("exatamente um retry", self.rules["failure_policy"])
        self.assertIn("não substituir seed", self.rules["failure_policy"])
        self.assertIn("checkpoints finais congelados", self.rules["trace_export"])
        self.assertIn("proibida durante treinamento oficial", self.rules["visualization_isolation"])

    def test_required_rule_keys_are_present(self):
        required = {
            "parent_protocol", "e3_0_scope", "e2_binding", "canonical_serialization",
            "seed_derivation", "paired_initialization", "optimizer_state", "shuffle_streams",
            "validation_usage", "test_usage", "metric_computation", "loss_aggregation",
            "client_test_derivation", "client_test_usage", "client_test_prohibitions",
            "budget_accounting", "cost_interpretation", "communication_accounting",
            "failure_policy", "provenance", "trace_export", "visualization_isolation",
            "compatibility",
        }
        self.assertTrue(required <= set(self.rules), required - set(self.rules))


if __name__ == "__main__":
    unittest.main()
