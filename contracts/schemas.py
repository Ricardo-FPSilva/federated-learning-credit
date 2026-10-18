"""Schemas Python para documentos JSON científicos, versão 1.0.

Os construtores e from_dict validam sem coerção de tipos. Dados de exemplo
pertencem aos testes, não ao catálogo de resultados.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from math import isclose
from typing import ClassVar

from .validation import (
    choice, decision, digest, identifier, integer, json_value, keys, number,
    relative_path, require, sequence, source_refs, text, vector,
)

EVIDENCE_KINDS = {"historical_documented", "historical_rerun", "new_experiment"}
SCOPES = {"train", "validation", "test", "client_test", "unknown"}
METRIC_UNITS = {
    "accuracy": "fraction", "precision": "fraction", "recall": "fraction",
    "f1": "fraction", "average_precision": "fraction", "pr_auc_trapezoid": "fraction",
    "roc_auc": "fraction", "bce_loss": "loss", "log_loss": "loss",
}


class Document:
    @classmethod
    def from_dict(cls, value):
        expected = {field.name for field in fields(cls)}
        if hasattr(cls, "artifact_type"):
            keys(value, expected | {"artifact_type"}, cls.__name__)
            require(value["artifact_type"] == cls.artifact_type, "artifact_type incorreto")
        else:
            keys(value, expected, cls.__name__)
        return cls(**deepcopy({key: value[key] for key in expected}))

    def to_dict(self):
        # Revalida campos aninhados, que não se tornam imutáveis com frozen=True.
        self.__post_init__()
        value = asdict(self)
        if hasattr(self, "artifact_type"):
            value["artifact_type"] = self.artifact_type
        return value


@dataclass(frozen=True, kw_only=True)
class Artifact(Document):
    schema_version: str
    artifact_id: str
    evidence_kind: str
    source_refs: list[dict]

    def __post_init__(self):
        require(self.schema_version == "1.0", "schema_version: versão não suportada")
        identifier(self.artifact_id, "artifact_id")
        choice(self.evidence_kind, EVIDENCE_KINDS, "evidence_kind")
        source_refs(self.source_refs)
        if self.evidence_kind == "historical_documented":
            require(bool(self.source_refs), "histórico documentado exige fontes")
        json_value(asdict(self))
        self.validate()

    def validate(self):
        raise NotImplementedError


def execution(artifact):
    if artifact.evidence_kind == "historical_documented":
        require(artifact.run_id is None, "histórico documentado não identifica execução comprovada")
    else:
        identifier(artifact.run_id, "run_id")


def evaluation(artifact):
    execution(artifact)
    choice(artifact.evaluation_scope, SCOPES, "evaluation_scope")
    if artifact.evaluation_scope == "client_test":
        identifier(artifact.client_id, "client_id")
    else:
        require(artifact.client_id is None, "client_id exige evaluation_scope client_test")
    if artifact.sample_count is not None:
        integer(artifact.sample_count, "sample_count", 1)
    if artifact.evidence_kind != "historical_documented":
        require(artifact.evaluation_scope != "unknown" and artifact.sample_count is not None,
                "avaliação executada exige escopo e sample_count")


def preprocessing(value):
    keys(value, {"id", "sha256", "fit_scope"}, "preprocessing")
    identifier(value["id"], "preprocessing.id")
    digest(value["sha256"], "preprocessing.sha256")
    choice(value["fit_scope"], {"full_dataset", "global_train", "local_train"}, "preprocessing.fit_scope")


@dataclass(frozen=True, kw_only=True)
class MetricValue(Document):
    name: str
    value: float | None
    unit: str
    definition: str
    undefined_reason: str | None
    reported_literal: str | None

    def __post_init__(self):
        choice(self.name, METRIC_UNITS, "metric.name")
        require(self.unit == METRIC_UNITS[self.name], "metric.unit incompatível")
        text(self.definition, "metric.definition")
        if self.value is None:
            text(self.undefined_reason, "metric.undefined_reason")
            require(self.reported_literal is None, "métrica ausente não tem literal numérico")
        else:
            number(self.value, "metric.value", 0, 1 if self.unit == "fraction" else None)
            require(self.undefined_reason is None, "valor presente não pode ter undefined_reason")
        if self.reported_literal is not None:
            text(self.reported_literal, "metric.reported_literal")


def metric_values(values, evidence_kind):
    sequence(values, "metrics")
    parsed = [MetricValue.from_dict(value) for value in values]
    names = [value.name for value in parsed]
    require(len(names) == len(set(names)), "metrics: nomes duplicados")
    for value in parsed:
        if evidence_kind == "historical_documented" and value.value is not None:
            require(value.reported_literal is not None, "histórico exige literal da fonte")
        if evidence_kind != "historical_documented":
            require(value.definition != "as_reported", "execução exige definição da métrica")
            require(value.reported_literal is None, "literal documental não pertence a resultado novo")
    return {value.name: value.value for value in parsed}


@dataclass(frozen=True, kw_only=True)
class Manifest(Artifact):
    artifact_type: ClassVar[str] = "manifest"
    run_id: str | None
    protocol_id: str
    comparison_group_id: str | None
    status: str
    source_commit: str | None
    source_dirty: bool | None
    source_files: dict
    dataset: dict
    model: dict
    seeds: dict
    environment: dict | None
    preprocessing: dict | None
    decision: dict | None
    configuration: dict
    artifacts: dict
    notes: list[str]

    def validate(self):
        execution(self)
        identifier(self.protocol_id, "protocol_id")
        if self.comparison_group_id is not None:
            identifier(self.comparison_group_id, "comparison_group_id")
        choice(self.status, {"documented", "planned", "running", "completed", "failed"}, "status")
        require((self.status == "documented") == (self.evidence_kind == "historical_documented"),
                "status documented exclusivo de histórico documentado")
        if self.source_commit is not None:
            digest(self.source_commit, "source_commit", 40)
        require(self.source_dirty is None or type(self.source_dirty) is bool,
                "source_dirty: booleano ou null esperado")
        require(type(self.source_files) is dict, "source_files: objeto esperado")
        for path, sha256 in self.source_files.items():
            relative_path(path, "source_files.path")
            digest(sha256, "source_files.sha256")
        keys(self.dataset, {"id", "sha256", "split_id", "partition_id", "sizes"}, "dataset")
        identifier(self.dataset["id"], "dataset.id")
        if self.dataset["sha256"] is not None:
            digest(self.dataset["sha256"], "dataset.sha256")
        for name in ("split_id", "partition_id"):
            if self.dataset[name] is not None:
                identifier(self.dataset[name], f"dataset.{name}")
        require(type(self.dataset["sizes"]) is dict, "dataset.sizes: objeto esperado")
        for name, count in self.dataset["sizes"].items():
            choice(name, {"train", "validation", "test"}, "dataset.sizes")
            integer(count, f"dataset.sizes.{name}", 1)
        keys(self.model, {"family", "scope", "client_id"}, "model")
        choice(self.model["family"], {"majority", "logistic_regression", "random_forest",
                                     "mlp_centralized", "mlp_federated", "mlp_local"}, "model.family")
        choice(self.model["scope"], {"global", "local"}, "model.scope")
        require((self.model["family"] == "mlp_local") == (self.model["scope"] == "local"),
                "model: família e escopo incompatíveis")
        if self.model["scope"] == "local":
            identifier(self.model["client_id"], "model.client_id")
        else:
            require(self.model["client_id"] is None, "modelo global não tem client_id")
        keys(self.seeds, {"split", "validation", "partition", "training"}, "seeds")
        for name, value in self.seeds.items():
            if value is not None:
                integer(value, f"seeds.{name}")
        if self.environment is not None:
            keys(self.environment, {"python", "packages", "platform", "hardware", "threads"}, "environment")
            for name in ("python", "platform", "hardware"):
                text(self.environment[name], f"environment.{name}")
            integer(self.environment["threads"], "environment.threads", 1)
            require(type(self.environment["packages"]) is dict, "environment.packages: objeto esperado")
            for name, version in self.environment["packages"].items():
                text(name, "package.name")
                text(version, "package.version")
        if self.preprocessing is not None:
            preprocessing(self.preprocessing)
        if self.decision is not None:
            decision(self.decision)
        require(type(self.configuration) is dict, "configuration: objeto esperado")
        require(type(self.artifacts) is dict, "artifacts: objeto esperado")
        for name, reference in self.artifacts.items():
            identifier(name, "artifact reference name")
            keys(reference, {"path", "sha256"}, "artifact reference")
            relative_path(reference["path"], "artifact.path")
            digest(reference["sha256"], "artifact.sha256")
        sequence(self.notes, "notes", nonempty=False)
        for note in self.notes:
            text(note, "note")
        if self.status == "completed":
            require(self.source_commit is not None and self.dataset["sha256"] is not None
                    and self.environment is not None and bool(self.environment["packages"]),
                    "execução concluída exige código, dataset e ambiente identificados")
            require(bool(self.artifacts), "execução concluída exige artefatos, ao menos logs")
        if self.evidence_kind == "new_experiment" and self.status == "completed":
            require(self.preprocessing is not None and self.decision is not None,
                    "novo experimento concluído exige preprocessing e decisão")
            require(self.source_dirty is not None and bool(self.source_files),
                    "novo experimento exige estado Git e hashes dos arquivos executados")
            require(self.comparison_group_id is not None and self.dataset["split_id"] is not None,
                    "novo experimento concluído exige grupo e split")
            require(all(v is not None for v in self.seeds.values()), "novo experimento exige seeds explícitas")
            require(self.preprocessing["fit_scope"] != "full_dataset",
                    "protocolo corrigido não ajusta preprocessing no dataset completo")
            require(set(self.dataset["sizes"]) == {"train", "validation", "test"},
                    "benchmark exige tamanhos de treino, validação e teste")


@dataclass(frozen=True, kw_only=True)
class Metrics(Artifact):
    artifact_type: ClassVar[str] = "metrics"
    run_id: str | None
    evaluation_scope: str
    client_id: str | None
    sample_count: int | None
    values: list[dict]
    confusion_matrix: dict | None
    resources: dict | None

    def validate(self):
        evaluation(self)
        values = metric_values(self.values, self.evidence_kind)
        if self.resources is not None:
            keys(self.resources, {"training_seconds", "total_seconds", "peak_memory_bytes", "measurement_scope"}, "resources")
            for name in ("training_seconds", "total_seconds"):
                if self.resources[name] is not None:
                    number(self.resources[name], f"resources.{name}", 0)
            if self.resources["peak_memory_bytes"] is not None:
                integer(self.resources["peak_memory_bytes"], "resources.peak_memory_bytes")
            text(self.resources["measurement_scope"], "resources.measurement_scope")
            if self.resources["training_seconds"] is not None and self.resources["total_seconds"] is not None:
                require(self.resources["total_seconds"] >= self.resources["training_seconds"],
                        "resources: tempo total menor que tempo de treino")
        if self.confusion_matrix is None:
            return
        keys(self.confusion_matrix, {"labels", "orientation", "counts"}, "confusion_matrix")
        labels = self.confusion_matrix["labels"]
        require(type(labels) is list and len(labels) == 2
                and all(type(x) is int for x in labels) and labels == [0, 1], "confusion_matrix.labels: esperado [0,1]")
        require(self.confusion_matrix["orientation"] == "true_rows_predicted_columns", "orientação da matriz inválida")
        rows = self.confusion_matrix["counts"]
        sequence(rows, "confusion_matrix.counts")
        require(len(rows) == 2, "matriz deve ser 2x2")
        for row in rows:
            sequence(row, "confusion_matrix.row")
            require(len(row) == 2, "matriz deve ser 2x2")
            for count in row:
                integer(count, "confusion_matrix.count")
        tn, fp = rows[0]
        fn, tp = rows[1]
        total = tn + fp + fn + tp
        require(total > 0 and self.sample_count == total, "matriz e sample_count divergentes")
        if self.evidence_kind == "historical_documented":
            return  # Valores publicados podem ter sido arredondados.
        expected = {
            "accuracy": (tn + tp) / total,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        }
        for name, actual in values.items():
            if name in expected:
                target = expected[name]
                require((actual is None and target is None)
                        or (actual is not None and target is not None
                            and isclose(actual, target, abs_tol=1e-12, rel_tol=1e-9)),
                        f"{name}: incompatível com matriz de confusão")


@dataclass(frozen=True, kw_only=True)
class History(Artifact):
    artifact_type: ClassVar[str] = "history"
    run_id: str | None
    evaluation_scope: str
    client_id: str | None
    sample_count: int | None
    step_type: str
    records: list[dict]
    notes: list[str]

    def validate(self):
        evaluation(self)
        choice(self.step_type, {"round", "epoch"}, "step_type")
        sequence(self.records, "records")
        previous = -1
        for record in self.records:
            keys(record, {"step", "metrics"}, "history.record")
            integer(record["step"], "history.step")
            require(record["step"] > previous, "steps devem ser únicos e crescentes")
            previous = record["step"]
            metric_values(record["metrics"], self.evidence_kind)
        sequence(self.notes, "notes", nonempty=False)
        for note in self.notes:
            text(note, "note")


@dataclass(frozen=True, kw_only=True)
class Model(Artifact):
    artifact_type: ClassVar[str] = "model"
    run_id: str
    model_sha256: str
    preprocessing: dict
    dtype: str
    weight_layout: str
    feature_order: list[str]
    layers: list[dict]
    parameter_count: int
    decision: dict

    def validate(self):
        require(self.evidence_kind != "historical_documented", "pesos exigem execução identificada")
        execution(self)
        digest(self.model_sha256, "model_sha256")
        preprocessing(self.preprocessing)
        choice(self.dtype, {"float32", "float64"}, "dtype")
        require(self.weight_layout == "out_in", "weight_layout deve ser out_in")
        sequence(self.feature_order, "feature_order")
        for feature in self.feature_order:
            text(feature, "feature_order")
            require(feature != "BAD", "alvo BAD não pode ser feature")
        require(len(set(self.feature_order)) == len(self.feature_order), "features duplicadas")
        sequence(self.layers, "layers")
        previous_dim = len(self.feature_order)
        names = set()
        count = 0
        for index, layer in enumerate(self.layers):
            keys(layer, {"name", "input_dim", "output_dim", "activation", "weights", "bias"}, "layer")
            identifier(layer["name"], "layer.name")
            require(layer["name"] not in names, "nomes de camadas duplicados")
            names.add(layer["name"])
            integer(layer["input_dim"], "layer.input_dim", 1)
            integer(layer["output_dim"], "layer.output_dim", 1)
            require(layer["input_dim"] == previous_dim, "dimensões entre camadas incompatíveis")
            expected_activation = "sigmoid" if index == len(self.layers) - 1 else "relu"
            require(layer["activation"] == expected_activation, "ativação incompatível com MLP binária")
            sequence(layer["weights"], "layer.weights")
            require(len(layer["weights"]) == layer["output_dim"], "shape de weights incorreto")
            for row in layer["weights"]:
                vector(row, "layer.weights.row")
                require(len(row) == layer["input_dim"], "shape de weights incorreto")
            vector(layer["bias"], "layer.bias")
            require(len(layer["bias"]) == layer["output_dim"], "shape de bias incorreto")
            count += layer["output_dim"] * (layer["input_dim"] + 1)
            previous_dim = layer["output_dim"]
        require(previous_dim == 1, "saída binária deve ter dimensão 1")
        integer(self.parameter_count, "parameter_count", 1)
        require(self.parameter_count == count, "parameter_count incorreto")
        decision(self.decision)


@dataclass(frozen=True, kw_only=True)
class Trace(Artifact):
    artifact_type: ClassVar[str] = "trace"
    run_id: str
    sample_id: str
    model_id: str
    model_sha256: str
    preprocessing: dict
    input_vector: list[float]
    layers: list[dict]
    probability: float
    predicted_label: int
    decision: dict

    def validate(self):
        require(self.evidence_kind != "historical_documented", "trace exige execução identificada")
        execution(self)
        identifier(self.sample_id, "sample_id")
        identifier(self.model_id, "model_id")
        digest(self.model_sha256, "model_sha256")
        preprocessing(self.preprocessing)
        vector(self.input_vector, "input_vector")
        sequence(self.layers, "layers")
        names = set()
        for layer in self.layers:
            keys(layer, {"name", "pre_activation", "activation"}, "trace.layer")
            identifier(layer["name"], "trace.layer.name")
            require(layer["name"] not in names, "trace: camadas duplicadas")
            names.add(layer["name"])
            vector(layer["pre_activation"], "trace.pre_activation")
            vector(layer["activation"], "trace.activation")
            require(len(layer["pre_activation"]) == len(layer["activation"]), "trace: shapes divergentes")
        number(self.probability, "probability", 0, 1)
        require(len(self.layers[-1]["activation"]) == 1, "trace: saída deve ter dimensão 1")
        require(isclose(self.layers[-1]["activation"][0], self.probability, abs_tol=1e-7, rel_tol=1e-6),
                "trace: probabilidade difere da ativação final")
        integer(self.predicted_label, "predicted_label")
        decision(self.decision)
        require(self.predicted_label == int(self.probability > self.decision["threshold"]),
                "trace: classe incompatível com threshold estrito")
