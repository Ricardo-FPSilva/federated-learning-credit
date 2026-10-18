"""Fixtures SINTÉTICAS exclusivamente para testes de contrato; não são experimentos."""

import json
from pathlib import Path
import unittest

from contracts import (
    ContractError, History, Manifest, MetricValue, Metrics, Model, Trace,
    parse_artifact, validate_bundle, validate_protocol,
)
from contracts.validation import strict_json

ROOT = Path(__file__).resolve().parents[1]


def common(kind, identifier):
    return {"schema_version": "1.0", "artifact_type": kind, "artifact_id": identifier,
            "evidence_kind": "new_experiment", "source_refs": []}


def metric(name="accuracy", value=0.75, reason=None):
    return {"name": name, "value": value,
            "unit": "loss" if name in {"bce_loss", "log_loss"} else "fraction",
            "definition": "synthetic-test-only", "undefined_reason": reason,
            "reported_literal": None}


def prep():
    return {"id": "synthetic-prep", "sha256": "b" * 64, "fit_scope": "global_train"}


def rule():
    return {"positive_label": 1, "threshold": 0.5, "operator": ">"}


def manifest():
    return common("manifest", "synthetic-manifest") | {
        "run_id": "synthetic-run", "protocol_id": "test-only", "comparison_group_id": "test-only",
        "status": "completed", "source_commit": "a" * 40, "source_dirty": False,
        "source_files": {"synthetic.py": "a" * 64},
        "dataset": {"id": "synthetic", "sha256": "c" * 64, "split_id": "test-split",
                    "partition_id": "test-partition", "sizes": {"train": 6, "validation": 2, "test": 4}},
        "model": {"family": "mlp_federated", "scope": "global", "client_id": None},
        "seeds": {"split": 42, "validation": 43, "partition": 44, "training": 42},
        "environment": {"python": "test-only", "packages": {"synthetic": "test-only"},
                        "platform": "test-only", "hardware": "test-only", "threads": 1},
        "preprocessing": prep(), "decision": rule(), "configuration": {"purpose": "test-only"},
        "artifacts": {"metrics": {"path": "metrics.json", "sha256": "d" * 64}},
        "notes": ["SINTÉTICO: fixture, não experimento"],
    }


def metrics():
    return common("metrics", "synthetic-metrics") | {
        "run_id": "synthetic-run", "evaluation_scope": "test", "client_id": None,
        "sample_count": 4,
        "values": [metric(), metric("precision", 1), metric("recall", 0.5), metric("f1", 2 / 3)],
        "confusion_matrix": {"labels": [0, 1], "orientation": "true_rows_predicted_columns",
                             "counts": [[2, 0], [1, 1]]},
        "resources": {"training_seconds": 1.0, "total_seconds": 2.0,
                      "peak_memory_bytes": 1024, "measurement_scope": "synthetic-test-only"},
    }


def history():
    return common("history", "synthetic-history") | {
        "run_id": "synthetic-run", "evaluation_scope": "validation", "client_id": None,
        "sample_count": 2, "step_type": "round",
        "records": [{"step": 1, "metrics": [metric()]}, {"step": 3, "metrics": [metric()]}],
        "notes": ["SINTÉTICO: não interpolar round ausente"],
    }


def model():
    return common("model", "synthetic-model") | {
        "run_id": "synthetic-run", "model_sha256": "e" * 64, "preprocessing": prep(),
        "dtype": "float32", "weight_layout": "out_in", "feature_order": ["x1", "x2"],
        "layers": [
            {"name": "fc1", "input_dim": 2, "output_dim": 2, "activation": "relu",
             "weights": [[1, -2], [-1, 1]], "bias": [0.5, -0.5]},
            {"name": "fc2", "input_dim": 2, "output_dim": 1, "activation": "sigmoid",
             "weights": [[0.2, 0.4]], "bias": [-1]},
        ], "parameter_count": 9, "decision": rule(),
    }


def trace():
    return common("trace", "synthetic-trace") | {
        "run_id": "synthetic-run", "sample_id": "synthetic-sample", "model_id": "synthetic-model",
        "model_sha256": "e" * 64, "preprocessing": prep(), "input_vector": [-1, 2],
        "layers": [{"name": "fc1", "pre_activation": [-4.5, 2.5], "activation": [0, 2.5]},
                   {"name": "fc2", "pre_activation": [0], "activation": [0.5]}],
        "probability": 0.5, "predicted_label": 0, "decision": rule(),
    }


class SerializationTests(unittest.TestCase):
    def test_all_five_schemas_roundtrip(self):
        for factory, cls in ((manifest, Manifest), (metrics, Metrics), (history, History),
                             (model, Model), (trace, Trace)):
            with self.subTest(schema=cls.__name__):
                value = factory()
                parsed = parse_artifact(value)
                self.assertIsInstance(parsed, cls)
                self.assertEqual(parsed.to_dict(), value)
                self.assertEqual(strict_json(json.dumps(parsed.to_dict(), allow_nan=False)), value)

    def test_unknown_and_missing_fields(self):
        for factory in (manifest, metrics, history, model, trace):
            for operation in ("unknown", "missing"):
                with self.subTest(factory=factory.__name__, operation=operation):
                    value = factory()
                    if operation == "unknown":
                        value["guess"] = 1
                    else:
                        del value["schema_version"]
                    with self.assertRaises(ContractError):
                        parse_artifact(value)

    def test_rejects_unknown_version_kind_and_evidence(self):
        for key, replacement in (("schema_version", "2.0"), ("artifact_type", "pickle"),
                                 ("evidence_kind", "educational")):
            value = manifest()
            value[key] = replacement
            with self.subTest(key=key), self.assertRaises(ContractError):
                parse_artifact(value)

    def test_json_rejects_nonfinite_duplicate_and_invalid_syntax(self):
        for source in ('{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}',
                       '{"x":1e999}', '{"x":1,"x":2}', '{"a":{"x":1,"x":2}}', '{'):
            with self.subTest(source=source), self.assertRaises(ContractError):
                strict_json(source)

    def test_nonstring_key_and_nonjson_value(self):
        value = manifest()
        value[2] = None
        with self.assertRaises(ContractError):
            parse_artifact(value)
        value = manifest()
        value["configuration"]["invalid"] = object()
        with self.assertRaises(ContractError):
            parse_artifact(value)

    def test_input_is_defensively_copied(self):
        value = model()
        parsed = parse_artifact(value)
        value["layers"][0]["weights"][0][0] = 100
        self.assertEqual(parsed.layers[0]["weights"][0][0], 1)

    def test_serialization_revalidates_mutated_nested_values(self):
        parsed = parse_artifact(model())
        parsed.layers[0]["bias"].append(1)
        with self.assertRaises(ContractError):
            parsed.to_dict()

    def test_direct_constructor_validates(self):
        value = metrics()
        del value["artifact_type"]
        value["sample_count"] = True
        with self.assertRaises(ContractError):
            Metrics(**value)


class ManifestTests(unittest.TestCase):
    def test_completed_requires_provenance_and_artifacts(self):
        for key, replacement in (("environment", None), ("source_commit", None),
                                 ("preprocessing", None), ("decision", None), ("artifacts", {})):
            value = manifest()
            value[key] = replacement
            with self.subTest(key=key), self.assertRaises(ContractError):
                parse_artifact(value)

    def test_planned_may_have_unknown_environment(self):
        value = manifest()
        value.update(status="planned", environment=None, artifacts={}, preprocessing=None)
        self.assertEqual(parse_artifact(value).status, "planned")

    def test_rerun_can_finish_with_logs_without_saved_preprocessor(self):
        value = manifest()
        value.update(evidence_kind="historical_rerun", preprocessing=None, decision=None)
        value["artifacts"] = {"stdout": {"path": "stdout.log", "sha256": "a" * 64}}
        self.assertEqual(parse_artifact(value).status, "completed")

    def test_new_run_requires_code_hashes_even_if_git_is_dirty(self):
        value = manifest()
        value["source_dirty"] = True
        parse_artifact(value)
        value["source_files"] = {}
        with self.assertRaises(ContractError):
            parse_artifact(value)

    def test_rerun_can_have_unknown_training_seed(self):
        value = manifest()
        value["evidence_kind"] = "historical_rerun"
        value["seeds"]["training"] = None
        self.assertIsNone(parse_artifact(value).seeds["training"])
        value["evidence_kind"] = "new_experiment"
        with self.assertRaises(ContractError):
            parse_artifact(value)

    def test_completed_new_rejects_full_dataset_fit(self):
        value = manifest()
        value["preprocessing"]["fit_scope"] = "full_dataset"
        with self.assertRaises(ContractError):
            parse_artifact(value)

    def test_invalid_hash_seed_and_path(self):
        for bad in ("../model.json", "/model.json", "a//b", "a/./b", "C:\\model", "a/../b", "file\0.json", "file\n.json"):
            value = manifest()
            value["artifacts"]["metrics"]["path"] = bad
            with self.subTest(path=bad), self.assertRaises(ContractError):
                parse_artifact(value)
        for bad in (True, -1, 42.0, "42"):
            value = manifest()
            value["seeds"]["training"] = bad
            with self.subTest(seed=bad), self.assertRaises(ContractError):
                parse_artifact(value)
        value = manifest()
        value["source_commit"] = "main"
        with self.assertRaises(ContractError):
            parse_artifact(value)

    def test_local_requires_client_and_global_forbids_it(self):
        value = manifest()
        value["model"] = {"family": "mlp_local", "scope": "local", "client_id": None}
        with self.assertRaises(ContractError):
            parse_artifact(value)
        value["model"]["client_id"] = "client-0"
        parse_artifact(value)
        value["model"]["scope"] = "global"
        with self.assertRaises(ContractError):
            parse_artifact(value)


class MetricTests(unittest.TestCase):
    def test_invalid_numeric_values(self):
        for bad in (True, "0.5", -0.1, 1.1, float("nan"), float("inf"), 10 ** 1000):
            with self.subTest(value=str(bad)[:30]), self.assertRaises(ContractError):
                MetricValue.from_dict(metric(value=bad))

    def test_loss_nonnegative_and_not_bounded_by_one(self):
        MetricValue.from_dict(metric("bce_loss", 3.2))
        with self.assertRaises(ContractError):
            MetricValue.from_dict(metric("bce_loss", -1))

    def test_missing_requires_reason_and_existing_forbids_it(self):
        MetricValue.from_dict(metric("precision", None, "no_predicted_positive"))
        for value in (metric(value=None), metric(reason="invented")):
            with self.assertRaises(ContractError):
                MetricValue.from_dict(value)

    def test_ap_and_trapezoidal_are_distinct(self):
        value = metrics()
        value["values"].extend([metric("average_precision", 0.6), metric("pr_auc_trapezoid", 0.7)])
        parse_artifact(value)
        value["values"].append(metric())
        with self.assertRaises(ContractError):
            parse_artifact(value)

    def test_wrong_units(self):
        value = metric()
        value["unit"] = "percent"
        with self.assertRaises(ContractError):
            MetricValue.from_dict(value)

    def test_confusion_matrix_consistency(self):
        for change in ("count", "accuracy", "orientation", "bool", "labels"):
            value = metrics()
            if change == "count":
                value["sample_count"] = 99
            elif change == "accuracy":
                value["values"][0]["value"] = 0.9
            elif change == "orientation":
                value["confusion_matrix"]["orientation"] = "predicted_rows"
            elif change == "bool":
                value["confusion_matrix"]["counts"][0][0] = True
            else:
                value["confusion_matrix"]["labels"] = [False, True]
            with self.subTest(change=change), self.assertRaises(ContractError):
                parse_artifact(value)

    def test_no_positive_predictions(self):
        value = metrics()
        value["confusion_matrix"]["counts"] = [[2, 0], [2, 0]]
        value["values"] = [metric("accuracy", 0.5), metric("precision", None, "no_predicted_positive"),
                           metric("recall", 0), metric("f1", 0)]
        parse_artifact(value)

    def test_computational_cost(self):
        for key, invalid in (("training_seconds", -1), ("total_seconds", 0.5),
                             ("peak_memory_bytes", 1.5), ("peak_memory_bytes", True)):
            value = metrics()
            value["resources"][key] = invalid
            with self.subTest(key=key, invalid=invalid), self.assertRaises(ContractError):
                parse_artifact(value)

    def test_local_evaluation_requires_client_id(self):
        value = metrics()
        value["evaluation_scope"] = "client_test"
        with self.assertRaises(ContractError):
            parse_artifact(value)
        value["client_id"] = "client-0"
        parse_artifact(value)


class HistoryTests(unittest.TestCase):
    def test_gaps_preserved(self):
        parsed = parse_artifact(history())
        self.assertEqual([record["step"] for record in parsed.records], [1, 3])

    def test_invalid_steps(self):
        for bad in (True, -1, 1, 0, 1.5):
            value = history()
            value["records"][1]["step"] = bad
            with self.subTest(step=bad), self.assertRaises(ContractError):
                parse_artifact(value)

    def test_executed_history_requires_evaluation_metadata(self):
        for key, bad in (("evaluation_scope", "unknown"), ("sample_count", None), ("run_id", None)):
            value = history()
            value[key] = bad
            with self.subTest(key=key), self.assertRaises(ContractError):
                parse_artifact(value)


class ModelTraceTests(unittest.TestCase):
    def test_small_model_with_negative_input_and_nonzero_bias(self):
        self.assertEqual(len(validate_bundle(manifest(), model(), trace())), 3)

    def test_shape_bias_parameter_count_and_orientation(self):
        for change in ("weights", "bias", "count", "layout", "features", "target"):
            value = model()
            if change == "weights":
                value["layers"][0]["weights"][0].pop()
            elif change == "bias":
                value["layers"][0]["bias"].pop()
            elif change == "count":
                value["parameter_count"] = 313
            elif change == "layout":
                value["weight_layout"] = "in_out"
            elif change == "features":
                value["feature_order"] = ["x1", "x1"]
            else:
                value["feature_order"] = ["BAD", "x2"]
            with self.subTest(change=change), self.assertRaises(ContractError):
                parse_artifact(value)

    def test_historical_documented_cannot_claim_model_or_trace(self):
        reference = {"commit": "a" * 40, "blob": "b" * 40, "path": "README.md", "line_start": 1, "line_end": 2}
        for factory in (model, trace):
            value = factory()
            value.update(evidence_kind="historical_documented", source_refs=[reference])
            with self.subTest(factory=factory.__name__), self.assertRaises(ContractError):
                parse_artifact(value)

    def test_threshold_is_strict(self):
        self.assertEqual(parse_artifact(trace()).predicted_label, 0)
        value = trace()
        value["predicted_label"] = 1
        with self.assertRaises(ContractError):
            parse_artifact(value)

    def test_trace_probability_and_activation_must_match(self):
        value = trace()
        value["probability"] = 0.4
        with self.assertRaises(ContractError):
            parse_artifact(value)

    def test_trace_has_correct_values_not_only_shapes(self):
        value = trace()
        value["layers"][0]["pre_activation"][0] = -3
        with self.assertRaises(ContractError):
            validate_bundle(manifest(), model(), value)
        value = trace()
        value["layers"][0]["activation"][0] = 1
        with self.assertRaises(ContractError):
            validate_bundle(manifest(), model(), value)


class BundleTests(unittest.TestCase):
    def test_multiple_documental_manifests_do_not_claim_same_run(self):
        first = manifest()
        first.update(evidence_kind="historical_documented", run_id=None, status="documented")
        first["source_refs"] = [{"commit": "a" * 40, "blob": "b" * 40, "path": "README.md", "line_start": 1, "line_end": 2}]
        second = dict(first, artifact_id="second-document")
        self.assertEqual(len(validate_bundle(first, second)), 2)

    def test_complete_bundle(self):
        objects = validate_bundle(manifest(), metrics(), history(), model(), trace())
        self.assertEqual(len(objects), 5)

    def test_missing_dependencies_duplicates_and_empty(self):
        for values in ((), (metrics(),), (manifest(), trace()), (manifest(), manifest())):
            with self.subTest(count=len(values)), self.assertRaises(ContractError):
                validate_bundle(*values)

    def test_cross_artifact_identifiers_hashes_and_scopes(self):
        for key, replacement in (("model_id", "other-model"), ("model_sha256", "f" * 64),
                                 ("run_id", "other-run"), ("evidence_kind", "historical_rerun")):
            value = trace()
            value[key] = replacement
            with self.subTest(key=key), self.assertRaises(ContractError):
                validate_bundle(manifest(), model(), value)
        value = trace()
        value["preprocessing"]["sha256"] = "f" * 64
        with self.assertRaises(ContractError):
            validate_bundle(manifest(), model(), value)

    def test_evaluation_sample_count_matches_manifest(self):
        value = manifest()
        value["dataset"]["sizes"]["test"] = 5
        with self.assertRaises(ContractError):
            validate_bundle(value, metrics())

    def test_model_manifest_preprocessing_matches(self):
        value = manifest()
        value["preprocessing"]["id"] = "other-prep"
        with self.assertRaises(ContractError):
            validate_bundle(value, model())


class ProtocolTests(unittest.TestCase):
    def test_documented_protocols(self):
        for path in (ROOT / "protocols").glob("*.json"):
            with self.subTest(path=path.name):
                validate_protocol(strict_json(path.read_text(encoding="utf-8")))

    def test_benchmark_remains_proposed(self):
        value = strict_json((ROOT / "protocols/benchmark-v1.json").read_text(encoding="utf-8"))
        self.assertEqual(value["status"], "proposed")
        value["status"] = "documented"
        with self.assertRaises(ContractError):
            validate_protocol(value)

    def test_protocol_rejects_unknown_fields(self):
        value = strict_json((ROOT / "protocols/benchmark-v1.json").read_text(encoding="utf-8"))
        value["execution_completed"] = True
        with self.assertRaises(ContractError):
            validate_protocol(value)


if __name__ == "__main__":
    unittest.main()
