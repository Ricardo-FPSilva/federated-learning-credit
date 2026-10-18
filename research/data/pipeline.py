"""Pipeline determinístico de splits, preprocessing e partições do benchmark-v1."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Mapping, Sequence

from contracts.schemas import preprocessing as contract_preprocessing
from contracts.validation import (
    digest as contract_digest,
    identifier as contract_identifier,
    strict_json,
)

from research.data.partitioning import (
    DEFAULT_N_CLIENTS,
    DEFAULT_PARTITION_SEED,
    iid_partition,
    partition_digest,
)
from research.data.preprocessing import FeaturePreprocessor
from research.data.splitting import (
    DEFAULT_FRACTIONS,
    DEFAULT_SPLIT_SEED,
    SPLIT_NAMES,
    stratified_split,
)

SCHEMA = "corrected-data-pipeline/1.0"
PROTOCOL_ID = "benchmark-v1"
TARGET = "BAD"
VALIDATION_SEED = 43
STRATEGY = "iid"
SEED_USAGE = {
    "split": "consumed",
    "validation": "reserved_not_consumed_in_e2",
    "partition": "consumed",
}
SEED_NOTES = (
    "Seed 42 é consumida na separação estratificada e 44 na partição IID; "
    "a seed 43 é reservada pelo benchmark-v1 para execuções futuras e não é "
    "consumida pela E2."
)
DUPLICATE_GROUP_AUDIT = (
    "Nenhuma coluna de agrupamento foi declarada no dataset nem no benchmark-v1; "
    "a auditoria de grupos fica limitada a duplicatas exatas e duplicatas por features."
)

#: Configuração única admitida pelas regras congeladas de benchmark-v1.
PROTOCOL_CONFIG: dict = {
    "fractions": dict(DEFAULT_FRACTIONS),
    "split_seed": DEFAULT_SPLIT_SEED,
    "validation_seed": VALIDATION_SEED,
    "partition_seed": DEFAULT_PARTITION_SEED,
    "n_clients": DEFAULT_N_CLIENTS,
    "strategy": STRATEGY,
}
DEFAULT_CONFIG = PROTOCOL_CONFIG

RECORD_KEYS = {
    "schema", "protocol_id", "dataset", "config", "seeds", "seed_usage",
    "split_id", "partition_id", "partition_digest", "splits", "partitions",
    "duplicate_audit", "feature_order", "output_features", "preprocessing",
    "preprocessing_statistics", "notes",
}


class PipelineError(ValueError):
    """Dataset, configuração ou resultado do pipeline inválido."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PipelineError(message)


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_config(config: object) -> dict:
    """Aceita somente a configuração congelada por benchmark-v1."""
    _require(type(config) is dict, "config: objeto esperado")
    _require(set(config) == set(PROTOCOL_CONFIG), "config: exatamente as chaves de benchmark-v1")
    fractions = validate_fractions_strict(config["fractions"])
    for name in ("split_seed", "validation_seed", "partition_seed", "n_clients"):
        _require(type(config[name]) is int, f"config.{name}: inteiro esperado")
    _require(config["strategy"] == STRATEGY, "config.strategy: benchmark-v1 exige 'iid'")
    clean = {
        "fractions": fractions,
        "strategy": STRATEGY,
        "split_seed": config["split_seed"],
        "validation_seed": config["validation_seed"],
        "partition_seed": config["partition_seed"],
        "n_clients": config["n_clients"],
    }
    _require(clean == PROTOCOL_CONFIG,
             "config: valores incompatíveis com as regras de benchmark-v1 "
             "(60/20/20, seeds 42/43/44, 3 clientes, IID)")
    return clean


def validate_fractions_strict(fractions: object) -> dict[str, float]:
    _require(type(fractions) is dict, "fractions: objeto esperado")
    _require(set(fractions) == set(SPLIT_NAMES), "fractions: exatamente train/validation/test")
    expected = PROTOCOL_CONFIG["fractions"]
    for name in SPLIT_NAMES:
        value = fractions[name]
        _require(type(value) in (int, float) and float(value) == expected[name],
                 f"fractions.{name}: benchmark-v1 exige {expected[name]}")
    return dict(expected)


def load_dataset(dataset_path: str | Path) -> tuple[list[dict], list[str], str]:
    """Lê o CSV preservando a ordem original das linhas; devolve linhas, colunas e SHA-256."""
    path = Path(dataset_path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PipelineError("dataset: UTF-8 inválido") from exc
    reader = csv.DictReader(io.StringIO(text))
    fieldnames = list(reader.fieldnames or ())
    _require(bool(fieldnames), "dataset: cabeçalho ausente")
    _require(len(set(fieldnames)) == len(fieldnames), "dataset: colunas duplicadas")
    _require(all(fieldnames), "dataset: nome de coluna vazio")
    _require(TARGET in fieldnames, f"dataset: coluna alvo {TARGET} ausente")

    rows: list[dict] = []
    for line in reader:
        _require(None not in line, "dataset: linha com mais campos que o cabeçalho")
        row = {}
        for column in fieldnames:
            value = line.get(column)
            row[column] = None if value is None or value.strip() == "" else value
        rows.append(row)
    _require(bool(rows), "dataset: sem observações")

    labels = []
    for position, row in enumerate(rows):
        try:
            label = int(str(row[TARGET]))
        except (TypeError, ValueError) as exc:
            raise PipelineError(f"dataset: rótulo não binário na linha {position}") from exc
        _require(label in (0, 1), f"dataset: rótulo fora de {{0,1}} na linha {position}")
        labels.append(label)
    return rows, fieldnames, digest


def audit_duplicates(rows: Sequence[Mapping], fieldnames: Sequence[str],
                     splits: Mapping[str, Sequence[int]]) -> dict:
    """Audita duplicatas exatas, por features e grupos que atravessam splits."""
    exact = Counter(tuple(row.get(column) for column in fieldnames) for row in rows)
    feature_columns = [column for column in fieldnames if column != TARGET]
    feature_keys = [tuple(row.get(column) for column in feature_columns) for row in rows]
    feature_counts = Counter(feature_keys)
    split_of = {index: name for name, indices in splits.items() for index in indices}
    positions: dict[tuple, list[int]] = defaultdict(list)
    for index, key in enumerate(feature_keys):
        positions[key].append(index)
    cross_split = sum(
        1 for key, members in positions.items() if feature_counts[key] > 1
        and len({split_of[m] for m in members}) > 1
    )
    exact_groups = [count for count in exact.values() if count > 1]
    feature_groups = [count for count in feature_counts.values() if count > 1]
    return {
        "exact": {
            "groups": len(exact_groups),
            "rows": sum(exact_groups),
            "extra_rows": sum(count - 1 for count in exact_groups),
        },
        "features": {
            "groups": len(feature_groups),
            "rows": sum(feature_groups),
            "extra_rows": sum(count - 1 for count in feature_groups),
        },
        "cross_split_feature_groups": cross_split,
        "group_column": None,
        "group_audit_limitation": DUPLICATE_GROUP_AUDIT,
    }


def split_id_for(dataset_sha: str, fractions: Mapping[str, float], split_seed: int,
                 splits: Mapping[str, Sequence[int]]) -> str:
    payload = {
        "dataset_sha256": dataset_sha,
        "protocol_id": PROTOCOL_ID,
        "fractions": dict(fractions),
        "split_seed": split_seed,
        "splits": {name: list(indices) for name, indices in splits.items()},
    }
    return f"split-{split_seed}-{_canonical_digest(payload)[:16]}"


def partition_id_for(dataset_sha: str, strategy: str, n_clients: int, partition_seed: int,
                     partitions: Mapping[str, Sequence[int]]) -> str:
    payload = {
        "dataset_sha256": dataset_sha,
        "protocol_id": PROTOCOL_ID,
        "strategy": strategy,
        "n_clients": n_clients,
        "partition_seed": partition_seed,
        "partitions": {client: list(indices) for client, indices in partitions.items()},
    }
    return f"partition-{strategy}-{n_clients}-{partition_seed}-{_canonical_digest(payload)[:16]}"


def build_record(dataset_path: str | Path, config: Mapping | None = None) -> dict:
    """Monta o registro auditável; só o treino participa do ajuste do preprocessing."""
    clean = validate_config(PROTOCOL_CONFIG if config is None else dict(config))
    path = Path(dataset_path)
    rows, fieldnames, dataset_sha = load_dataset(path)
    feature_order = [column for column in fieldnames if column != TARGET]
    labels = [int(str(row[TARGET])) for row in rows]

    splits = stratified_split(
        list(range(len(rows))), labels, clean["fractions"], clean["split_seed"]
    )
    duplicate_audit = audit_duplicates(rows, fieldnames, splits)
    preprocessor = FeaturePreprocessor(feature_order)
    preprocessor.fit([rows[index] for index in splits["train"]])
    vectors = {
        name: preprocessor.transform([rows[index] for index in splits[name]])
        for name in SPLIT_NAMES
    }
    width = len(preprocessor.output_features)
    for name in SPLIT_NAMES:
        _require(all(len(vector) == width for vector in vectors[name]),
                 f"{name}: dimensão de feature inconsistente")

    partition = iid_partition(splits["train"], clean["n_clients"], clean["partition_seed"])
    _require(set().union(*map(set, partition.values())) == set(splits["train"]),
             "partição divergente do treino")

    splits_record = {
        name: {
            "size": len(indices),
            "index_sha256": _canonical_digest(indices),
            "indices": indices,
        }
        for name, indices in splits.items()
    }
    partition_record = {
        client: {
            "size": len(indices),
            "index_sha256": _canonical_digest(indices),
            "indices": indices,
        }
        for client, indices in partition.items()
    }
    record = {
        "schema": SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "dataset": {
            "name": path.name,
            "sha256": dataset_sha,
            "rows": len(rows),
            "features": len(feature_order),
            "target": TARGET,
            "index_base": 0,
        },
        "config": {
            "fractions": clean["fractions"],
            "strategy": clean["strategy"],
            "n_clients": clean["n_clients"],
        },
        "seeds": {
            "split": clean["split_seed"],
            "validation": clean["validation_seed"],
            "partition": clean["partition_seed"],
        },
        "seed_usage": dict(SEED_USAGE),
        "split_id": split_id_for(dataset_sha, clean["fractions"], clean["split_seed"], splits),
        "partition_id": partition_id_for(
            dataset_sha, clean["strategy"], clean["n_clients"], clean["partition_seed"],
            partition,
        ),
        "partition_digest": partition_digest(partition),
        "splits": splits_record,
        "partitions": partition_record,
        "duplicate_audit": duplicate_audit,
        "feature_order": feature_order,
        "output_features": preprocessor.output_features,
        "preprocessing": preprocessor.record(),
        "preprocessing_statistics": preprocessor.stats(),
        "notes": [
            SEED_NOTES,
            "Ajuste exclusivo do treino global; validação e teste globais ficam fora do treinamento.",
            "Partição IID; nenhum cenário non-IID foi produzido.",
        ],
    }
    validate_record(record)
    return record


def validate_record(record: object) -> None:
    """Validação estrita estrutural e de integridade do registro auditável."""
    _require(type(record) is dict, "registro: objeto esperado")
    _require(set(record) == RECORD_KEYS, "registro: conjunto de chaves inesperado")
    _require(record["schema"] == SCHEMA, "registro: schema inesperado")
    _require(record["protocol_id"] == PROTOCOL_ID, "registro: protocol_id inesperado")
    contract_identifier(record["split_id"], "split_id")
    contract_identifier(record["partition_id"], "partition_id")

    dataset = record["dataset"]
    _require(type(dataset) is dict and set(dataset) ==
             {"name", "sha256", "rows", "features", "target", "index_base"},
             "registro: dataset inválido")
    _require(type(dataset["name"]) is str and dataset["name"], "registro.dataset.name inválido")
    contract_digest(dataset["sha256"], "dataset.sha256")
    _require(type(dataset["rows"]) is int and dataset["rows"] > 0, "registro.dataset.rows inválido")
    _require(type(dataset["features"]) is int and dataset["features"] > 0,
             "registro.dataset.features inválido")
    _require(dataset["target"] == TARGET and dataset["index_base"] == 0,
             "registro.dataset: alvo ou base de índice inesperados")

    _require(record["config"] == {
        "fractions": PROTOCOL_CONFIG["fractions"],
        "strategy": STRATEGY,
        "n_clients": DEFAULT_N_CLIENTS,
    }, "registro.config: incompatível com benchmark-v1")
    expected_seeds = {
        "split": PROTOCOL_CONFIG["split_seed"],
        "validation": PROTOCOL_CONFIG["validation_seed"],
        "partition": PROTOCOL_CONFIG["partition_seed"],
    }
    _require(record["seeds"] == expected_seeds, "registro.seeds: inesperado")
    _require(record["seed_usage"] == SEED_USAGE, "registro.seed_usage: inesperado")
    contract_digest(record["partition_digest"], "partition_digest")

    splits = record["splits"]
    _require(type(splits) is dict and set(splits) == set(SPLIT_NAMES), "registro.splits: inválido")
    seen: set[int] = set()
    indices_by_split: dict[str, list[int]] = {}
    for name in SPLIT_NAMES:
        block = splits[name]
        _require(type(block) is dict and set(block) == {"size", "index_sha256", "indices"},
                 f"registro.splits.{name}: inválido")
        indices = block["indices"]
        _require(type(indices) is list and indices
                 and all(type(index) is int and index >= 0 for index in indices),
                 f"registro.splits.{name}.indices: inválido")
        _require(indices == sorted(indices) and len(indices) == len(set(indices)),
                 f"registro.splits.{name}.indices: devem ser únicos e ordenados")
        _require(block["size"] == len(indices), f"registro.splits.{name}.size divergente")
        _require(block["index_sha256"] == _canonical_digest(indices),
                 f"registro.splits.{name}.index_sha256 divergente")
        _require(not (seen & set(indices)), "registro.splits: sobreposição")
        seen.update(indices)
        indices_by_split[name] = indices
    _require(seen == set(range(dataset["rows"])), "registro.splits: cobertura divergente")
    _require(sum(len(v) for v in indices_by_split.values()) == dataset["rows"],
             "registro.splits: total divergente")

    partitions = record["partitions"]
    _require(type(partitions) is dict and len(partitions) == DEFAULT_N_CLIENTS,
             "registro.partitions: três clientes esperados")
    covered: set[int] = set()
    for client, block in partitions.items():
        contract_identifier(client, "partitions.client")
        _require(type(block) is dict and set(block) == {"size", "index_sha256", "indices"},
                 f"registro.partitions.{client}: inválido")
        indices = block["indices"]
        _require(type(indices) is list and indices
                 and all(type(index) is int and index >= 0 for index in indices),
                 f"registro.partitions.{client}.indices: inválido")
        _require(indices == sorted(indices) and len(indices) == len(set(indices)),
                 f"registro.partitions.{client}.indices: devem ser únicos e ordenados")
        _require(block["size"] == len(indices), f"registro.partitions.{client}.size divergente")
        _require(block["index_sha256"] == _canonical_digest(indices),
                 f"registro.partitions.{client}.index_sha256 divergente")
        _require(not (covered & set(indices)), "registro.partitions: sobreposição")
        covered.update(indices)
    _require(covered == set(indices_by_split["train"]),
             "registro.partitions: deve repartir exatamente o treino")
    _require(record["partition_digest"] == partition_digest(
        {client: block["indices"] for client, block in partitions.items()}),
        "registro.partition_digest divergente")

    train_splits = {name: indices_by_split[name] for name in SPLIT_NAMES}
    _require(record["split_id"] == split_id_for(
        dataset["sha256"], record["config"]["fractions"], record["seeds"]["split"], train_splits),
        "registro.split_id divergente")
    _require(record["partition_id"] == partition_id_for(
        dataset["sha256"], record["config"]["strategy"], record["config"]["n_clients"],
        record["seeds"]["partition"],
        {client: block["indices"] for client, block in partitions.items()}),
        "registro.partition_id divergente")

    audit = record["duplicate_audit"]
    _require(type(audit) is dict and set(audit) ==
             {"exact", "features", "cross_split_feature_groups", "group_column",
              "group_audit_limitation"},
             "registro.duplicate_audit: inválido")
    for scope in ("exact", "features"):
        block = audit[scope]
        _require(type(block) is dict and set(block) == {"groups", "rows", "extra_rows"},
                 f"registro.duplicate_audit.{scope}: inválido")
        for key in ("groups", "rows", "extra_rows"):
            _require(type(block[key]) is int and block[key] >= 0,
                     f"registro.duplicate_audit.{scope}.{key}: inválido")
    _require(type(audit["cross_split_feature_groups"]) is int
             and audit["cross_split_feature_groups"] >= 0,
             "registro.duplicate_audit.cross_split_feature_groups inválido")
    _require(audit["group_column"] is None, "registro.duplicate_audit: group_column inesperado")
    _require(type(audit["group_audit_limitation"]) is str
             and audit["group_audit_limitation"], "registro.duplicate_audit: limitação ausente")

    feature_order = record["feature_order"]
    _require(type(feature_order) is list and len(feature_order) == dataset["features"]
             and TARGET not in feature_order
             and all(type(name) is str and name for name in feature_order),
             "registro.feature_order inválido")
    output_features = record["output_features"]
    _require(type(output_features) is list and output_features
             and all(type(name) is str and name for name in output_features),
             "registro.output_features inválido")

    contract_preprocessing(record["preprocessing"])
    statistics_record = record["preprocessing_statistics"]
    _require(type(statistics_record) is dict
             and statistics_record.get("fit_scope") == "global_train"
             and statistics_record.get("feature_order") == feature_order,
             "registro.preprocessing_statistics inesperado")
    _require(record["preprocessing"]["sha256"] == _canonical_digest(statistics_record),
             "registro.preprocessing.sha256 divergente")

    notes = record["notes"]
    _require(type(notes) is list and notes
             and all(type(note) is str and note for note in notes),
             "registro.notes inválido")


def load_record(record_path: str | Path) -> dict:
    """Lê e valida estritamente um registro auditável publicado."""
    path = Path(record_path)
    try:
        record = strict_json(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise PipelineError(f"registro: JSON inválido em {path}") from exc
    validate_record(record)
    return record


def write_record(record_path: str | Path, record: Mapping) -> Path:
    """Valida e grava o registro auditável sem sobrescrita."""
    validate_record(record)
    path = Path(record_path)
    payload = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(payload)
    except FileExistsError as exc:
        raise PipelineError(f"recusando sobrescrever registro existente: {path}") from exc
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", default="hmeq.csv")
    parser.add_argument("--output", default="results/data/benchmark-v1-split.json")
    args = parser.parse_args(argv)
    try:
        record = build_record(args.dataset)
        path = write_record(args.output, record)
    except (PipelineError, OSError, ValueError) as exc:
        parser.exit(2, f"erro: {exc}\n")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
