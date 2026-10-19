"""Consumo read-only dos dados e transformações congelados pela E2."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

from research.data.pipeline import TARGET, load_dataset, load_record


class BenchmarkDataError(ValueError):
    """Dataset ou registro E2 incompatível com o benchmark congelado."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BenchmarkDataError(message)


@dataclass(frozen=True)
class DatasetView:
    """Visão imutável de índices, features transformadas e labels."""

    indices: tuple[int, ...]
    features: tuple[tuple[float, ...], ...]
    labels: tuple[int, ...]

    def __post_init__(self) -> None:
        _require(len(self.indices) == len(self.features) == len(self.labels),
                 "view: índices, features e labels com tamanhos divergentes")

    def __len__(self) -> int:
        return len(self.indices)


@dataclass(frozen=True)
class ExperimentalData:
    """Conjuntos globais e partições locais exatamente como congelados na E2."""

    train: DatasetView
    validation: DatasetView
    test: DatasetView
    clients: Mapping[str, DatasetView]
    feature_order: tuple[str, ...]
    output_features: tuple[str, ...]
    dataset_sha256: str
    split_id: str
    partition_id: str
    preprocessing_id: str

    def client_train(self, client_id: str) -> DatasetView:
        try:
            return self.clients[client_id]
        except KeyError as exc:
            raise BenchmarkDataError(f"cliente desconhecido: {client_id}") from exc


def _float(value: object) -> float | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _category(value: object) -> str | None:
    if value is None:
        return None
    result = str(value).strip()
    return result or None


def _expected_output_features(feature_order: Sequence[str], statistics: Mapping) -> list[str]:
    output = []
    numeric = statistics["numeric"]
    categorical = statistics["categorical"]
    for column in feature_order:
        if column in numeric:
            output.append(column)
        elif column in categorical:
            output.extend(f"{column}={value}" for value in categorical[column]["categories"])
        else:
            raise BenchmarkDataError(f"preprocessing sem estatística para {column}")
    return output


def _transform_row(row: Mapping[str, object], feature_order: Sequence[str],
                   statistics: Mapping) -> tuple[float, ...]:
    vector: list[float] = []
    numeric = statistics["numeric"]
    categorical = statistics["categorical"]
    for column in feature_order:
        if column in numeric:
            stats = numeric[column]
            value = _float(row.get(column))
            if value is None:
                value = stats["median"]
            transformed = (value - stats["mean"]) / stats["scale"]
            _require(math.isfinite(transformed), f"transformação não finita em {column}")
            vector.append(float(transformed))
            continue
        stats = categorical[column]
        value = _category(row.get(column))
        if value is None:
            value = stats["mode"]
        vector.extend(1.0 if value == category else 0.0 for category in stats["categories"])
    return tuple(vector)


def _view(indices: Sequence[int], features: Sequence[tuple[float, ...]],
          labels: Sequence[int]) -> DatasetView:
    return DatasetView(
        indices=tuple(indices),
        features=tuple(features[index] for index in indices),
        labels=tuple(labels[index] for index in indices),
    )


def load_experimental_data(dataset_path: str | Path, record_path: str | Path) -> ExperimentalData:
    """Valida E2/dataset e reconstrói todos os conjuntos sem qualquer `fit`."""
    dataset_path = Path(dataset_path)
    raw_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    record = load_record(record_path)
    _require(raw_sha256 == record["dataset"]["sha256"], "SHA-256 do dataset divergente")
    _require(dataset_path.name == record["dataset"]["name"], "nome do dataset divergente")

    rows, columns, loaded_sha256 = load_dataset(dataset_path)
    _require(loaded_sha256 == raw_sha256, "dataset mudou durante a leitura")
    _require(len(rows) == record["dataset"]["rows"], "número de linhas divergente")
    feature_order = tuple(column for column in columns if column != TARGET)
    _require(list(feature_order) == record["feature_order"], "ordem de features divergente")

    statistics = record["preprocessing_statistics"]
    _require(statistics["fit_scope"] == "global_train", "preprocessing fora do treino global")
    _require(statistics["feature_order"] == list(feature_order),
             "ordem do preprocessing divergente")
    expected_output = _expected_output_features(feature_order, statistics)
    _require(expected_output == record["output_features"], "output_features divergentes")

    features = tuple(_transform_row(row, feature_order, statistics) for row in rows)
    width = len(record["output_features"])
    _require(all(len(vector) == width for vector in features), "dimensão transformada divergente")
    labels = tuple(int(str(row[TARGET])) for row in rows)

    split_indices = {
        name: tuple(record["splits"][name]["indices"])
        for name in ("train", "validation", "test")
    }
    train = _view(split_indices["train"], features, labels)
    validation = _view(split_indices["validation"], features, labels)
    test = _view(split_indices["test"], features, labels)

    client_views = {
        client: _view(block["indices"], features, labels)
        for client, block in record["partitions"].items()
    }
    covered: set[int] = set()
    for client, view in client_views.items():
        current = set(view.indices)
        _require(not (covered & current), f"sobreposição de treino em {client}")
        covered.update(current)
    _require(covered == set(train.indices), "clientes não repartem exatamente o treino")
    _require(not covered.intersection(validation.indices), "cliente acessa validação")
    _require(not covered.intersection(test.indices), "cliente acessa teste")

    return ExperimentalData(
        train=train,
        validation=validation,
        test=test,
        clients=MappingProxyType(client_views),
        feature_order=feature_order,
        output_features=tuple(record["output_features"]),
        dataset_sha256=raw_sha256,
        split_id=record["split_id"],
        partition_id=record["partition_id"],
        preprocessing_id=record["preprocessing"]["id"],
    )
