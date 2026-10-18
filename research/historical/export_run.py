"""Exporta logs preservados de uma reexecução histórica para os contratos E0."""

from __future__ import annotations

import argparse
import ast
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

from contracts import load_artifact, parse_artifact, validate_bundle, validate_protocol
from contracts.validation import ContractError, strict_json


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENVIRONMENT_RECORD = ROOT / "results/environment/historical/validation.json"
OUTPUT_NAMES = ("manifest.json", "history.json", "metrics.json")
REQUIRED_INPUTS = (
    "commit.txt", "exit-code.txt", "source-hashes.txt", "stderr.log",
    "stdout.log", "time.txt",
)
EXPECTED_SOURCE_FILES = {"README.md", "hmeq.csv", "data_analysis.py", "machine_learning.py"}
EXPECTED_ROUNDS = list(range(1, 11))
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
LOSS_ENTRY = re.compile(r"\s*round\s+(\d+):\s*(\S+)\s*")
SOURCE_HASH = re.compile(r"^([0-9a-f]{64})  ([^\r\n]+)$", re.MULTILINE)


class ExportError(ValueError):
    """Registro incompleto, ambíguo ou incompatível com a exportação."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExportError(message)


def _finite_number(token: str, context: str, *, maximum: float | None = None) -> float:
    try:
        value = float(token)
    except ValueError as exc:
        raise ExportError(f"{context}: valor numérico inválido {token!r}") from exc
    _require(math.isfinite(value), f"{context}: valor não finito")
    _require(value >= 0, f"{context}: valor negativo")
    if maximum is not None:
        _require(value <= maximum, f"{context}: valor acima de {maximum}")
    return value


def _ordered_series(entries: list[tuple[int, float]], name: str) -> dict[int, float]:
    rounds = [round_number for round_number, _ in entries]
    _require(len(rounds) == len(set(rounds)), f"{name}: rounds duplicados")
    _require(rounds == EXPECTED_ROUNDS, f"{name}: esperado rounds 1–10 completos e ordenados")
    return dict(entries)


def parse_stdout(text: str) -> list[dict]:
    """Extrai somente o resumo final emitido pelo objeto History do Flower."""
    loss_header = "History (loss, distributed):"
    metrics_header = "History (metrics, distributed, evaluate):"
    _require(text.count(loss_header) == 1, "stdout: seção de loss ausente ou duplicada")
    _require(text.count(metrics_header) == 1, "stdout: seção de métricas ausente ou duplicada")
    loss_start = text.index(loss_header) + len(loss_header)
    metrics_start = text.index(metrics_header)
    _require(loss_start < metrics_start, "stdout: ordem das seções inválida")

    loss_entries = []
    for line in text[loss_start:metrics_start].splitlines():
        if not line.strip():
            continue
        match = LOSS_ENTRY.fullmatch(line)
        _require(match is not None, f"loss: linha inválida {line!r}")
        round_number, token = match.groups()
        loss_entries.append(
            (int(round_number), _finite_number(token, f"loss round {round_number}"))
        )
    loss = _ordered_series(loss_entries, "loss")

    metrics_text = text[metrics_start + len(metrics_header):].lstrip()
    closing_brace = metrics_text.find("}")
    _require(closing_brace >= 0, "stdout: dicionário de métricas incompleto")
    literal = metrics_text[:closing_brace + 1]
    names = re.findall(r"['\"]([^'\"\n]+)['\"]\s*:", literal)
    _require(names == ["accuracy", "f1"],
             "stdout: blocos accuracy/F1 ausentes, duplicados, reordenados ou inesperados")
    try:
        payload = ast.literal_eval(literal)
    except (SyntaxError, ValueError) as exc:
        raise ExportError("stdout: dicionário de métricas inválido") from exc
    _require(type(payload) is dict and list(payload) == ["accuracy", "f1"],
             "stdout: dicionário de métricas inválido")
    series = {}
    for name, raw_entries in payload.items():
        _require(type(raw_entries) is list, f"{name}: série não é uma lista")
        entries = []
        for item in raw_entries:
            _require(type(item) is tuple and len(item) == 2, f"{name}: entrada inválida")
            round_number, raw_value = item
            _require(type(round_number) is int, f"{name}: round inválido")
            _require(type(raw_value) in (int, float), f"{name} round {round_number}: valor inválido")
            entries.append((round_number, _finite_number(
                str(raw_value), f"{name} round {round_number}", maximum=1
            )))
        series[name] = _ordered_series(entries, name)

    return [
        {
            "step": round_number,
            "metrics": [
                _metric("accuracy", series["accuracy"][round_number], ACCURACY_DEFINITION),
                _metric("f1", series["f1"][round_number], F1_DEFINITION),
                _metric("bce_loss", loss[round_number], LOSS_DEFINITION),
            ],
        }
        for round_number in EXPECTED_ROUNDS
    ]


ACCURACY_DEFINITION = (
    "Flower distributed evaluate weighted mean by client test sample count; all three clients "
    "evaluated the same 1192-sample global test set"
)
F1_DEFINITION = (
    "Flower distributed evaluate weighted mean by client test sample count of client F1 values; "
    "not F1 computed from one global confusion matrix; all clients used the same global test set"
)
LOSS_DEFINITION = (
    "Flower distributed evaluate weighted mean by client test sample count; each client value is "
    "the unweighted mean of BCELoss batch means, so the shorter final batch is over-weighted"
)


def _metric(name: str, value: float, definition: str) -> dict:
    return {
        "name": name,
        "value": value,
        "unit": "loss" if name == "bce_loss" else "fraction",
        "definition": definition,
        "undefined_reason": None,
        "reported_literal": None,
    }


def _read_required(run_dir: Path) -> dict[str, bytes]:
    _require(run_dir.is_dir(), f"diretório de execução inexistente: {run_dir}")
    missing = [name for name in REQUIRED_INPUTS if not (run_dir / name).is_file()]
    _require(not missing, f"registros obrigatórios ausentes: {', '.join(missing)}")
    return {name: (run_dir / name).read_bytes() for name in REQUIRED_INPUTS}


def _decode(data: bytes, name: str) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ExportError(f"{name}: UTF-8 inválido") from exc


def _parse_commit(text: str) -> str:
    value = text.strip()
    _require(re.fullmatch(r"[0-9a-f]{40}", value) is not None, "commit.txt: hash inválido")
    return value


def _parse_source_hashes(text: str, root: Path) -> dict[str, str]:
    matches = SOURCE_HASH.findall(text)
    _require(len(matches) == len(text.strip().splitlines()), "source-hashes.txt: linha inválida")
    result = {}
    for digest, path in matches:
        _require(path not in result, f"source-hashes.txt: caminho duplicado {path}")
        _require(path in EXPECTED_SOURCE_FILES, f"source-hashes.txt: caminho inesperado {path}")
        result[path] = digest
    _require(set(result) == EXPECTED_SOURCE_FILES, "source-hashes.txt: conjunto de fontes incompleto")
    for path, expected in result.items():
        source = root / path
        _require(source.is_file(), f"fonte ausente: {path}")
        _require(_sha256(source.read_bytes()) == expected, f"fonte alterada desde a execução: {path}")
    return result


def _parse_duration(text: str) -> float:
    matches = re.findall(r"^real\s+([^\s]+)\s*$", text, re.MULTILINE)
    _require(len(matches) == 1, "time.txt: duração real ausente ou duplicada")
    token = matches[0].replace(",", ".")
    try:
        value = Decimal(token)
    except InvalidOperation as exc:
        raise ExportError("time.txt: duração inválida") from exc
    _require(value.is_finite() and value >= 0, "time.txt: duração não finita ou negativa")
    return float(value)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _source_ref(commit: str, stdout: bytes, run_id: str, line_start: int, line_end: int) -> dict:
    return {
        "commit": commit,
        "blob": _git_blob(stdout),
        "path": f"results/runs/{run_id}/stdout.log",
        "line_start": line_start,
        "line_end": line_end,
    }


def _load_environment(path: Path) -> dict:
    value = strict_json(path.read_text(encoding="utf-8"))
    _require(value.get("record_type") == "reconstructed_environment_validation",
             "registro de ambiente com tipo inesperado")
    _require(value.get("status") == "passed", "registro de ambiente não aprovado")
    packages = value.get("packages")
    _require(type(packages) is dict and bool(packages), "registro de ambiente sem pacotes")
    return value


def build_artifacts(run_dir: Path, *, root: Path = ROOT,
                    environment_record: Path = DEFAULT_ENVIRONMENT_RECORD) -> tuple[dict, dict, dict]:
    """Constrói documentos em memória; não escreve nem altera os logs."""
    run_dir = Path(run_dir)
    run_id = run_dir.name
    _require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id) is not None,
             "nome do diretório não é um run_id válido")
    inputs = _read_required(run_dir)
    text = {name: _decode(data, name) for name, data in inputs.items()}
    _require(text["exit-code.txt"].strip() == "0", "execução não terminou com exit code 0")
    commit = _parse_commit(text["commit.txt"])
    source_files = _parse_source_hashes(text["source-hashes.txt"], root)
    duration = _parse_duration(text["time.txt"])
    records = parse_stdout(text["stdout.log"])
    stderr_plain = ANSI_ESCAPE.sub("", text["stderr.log"])
    cpu_match = re.search(r"Ray initialized with resources: .*?'CPU':\s*([0-9]+(?:\.[0-9]+)?)", stderr_plain)
    _require(cpu_match is not None, "stderr.log: quantidade de CPUs Ray ausente")
    threads = int(float(cpu_match.group(1)))
    _require(threads > 0, "stderr.log: quantidade de CPUs inválida")

    protocol = strict_json((root / "protocols/historical-v1.json").read_text(encoding="utf-8"))
    validate_protocol(protocol)
    _require(protocol["protocol_id"] == "historical-v1", "protocolo histórico inesperado")
    environment = _load_environment(environment_record)

    history = {
        "schema_version": "1.0",
        "artifact_type": "history",
        "artifact_id": f"{run_id}-history",
        "evidence_kind": "historical_rerun",
        "source_refs": [_source_ref(commit, inputs["stdout.log"], run_id, 25, 56)],
        "run_id": run_id,
        "evaluation_scope": "test",
        "client_id": None,
        "sample_count": 1192,
        "step_type": "round",
        "records": records,
        "notes": [
            "Reexecução do protocolo histórico; distinta dos resultados históricos documentados.",
            "Imputação e LabelEncoder foram ajustados antes do split, preservando o leakage histórico.",
            "Os três clientes avaliaram o mesmo conjunto global de teste com 1192 amostras.",
            "Accuracy e F1 são médias ponderadas das métricas reportadas pelos clientes; F1 não deriva de uma matriz de confusão global única.",
            "A loss local é média não ponderada das médias dos batches; a loss distribuída agrega esses valores por número de amostras.",
        ],
    }
    final_metrics = records[-1]["metrics"]
    metrics = {
        "schema_version": "1.0",
        "artifact_type": "metrics",
        "artifact_id": f"{run_id}-metrics",
        "evidence_kind": "historical_rerun",
        "source_refs": [_source_ref(commit, inputs["stdout.log"], run_id, 25, 61)],
        "run_id": run_id,
        "evaluation_scope": "test",
        "client_id": None,
        "sample_count": 1192,
        "values": final_metrics,
        "confusion_matrix": None,
        "resources": {
            "training_seconds": None,
            "total_seconds": duration,
            "peak_memory_bytes": None,
            "measurement_scope": "total process wall-clock duration from time.txt; not training-only",
        },
    }
    # Validação antes de calcular hashes e construir o manifesto.
    parse_artifact(history)
    parse_artifact(metrics)
    history_bytes = _json_bytes(history)
    metrics_bytes = _json_bytes(metrics)

    artifact_refs = {
        name.replace(".", "-"): {"path": name, "sha256": _sha256(data)}
        for name, data in inputs.items()
    }
    artifact_refs.update({
        "history": {"path": "history.json", "sha256": _sha256(history_bytes)},
        "metrics": {"path": "metrics.json", "sha256": _sha256(metrics_bytes)},
    })
    manifest = {
        "schema_version": "1.0",
        "artifact_type": "manifest",
        "artifact_id": f"{run_id}-manifest",
        "evidence_kind": "historical_rerun",
        "source_refs": [
            _source_ref(commit, inputs["stdout.log"], run_id, 1, 61),
            {
                "commit": commit,
                "blob": _git_blob(inputs["stderr.log"]),
                "path": f"results/runs/{run_id}/stderr.log",
                "line_start": 1,
                "line_end": len(text["stderr.log"].splitlines()),
            },
        ],
        "run_id": run_id,
        "protocol_id": "historical-v1",
        "comparison_group_id": None,
        "status": "completed",
        "source_commit": commit,
        "source_dirty": None,
        "source_files": source_files,
        "dataset": {
            "id": "hmeq",
            "sha256": source_files["hmeq.csv"],
            "split_id": None,
            "partition_id": None,
            "sizes": {"train": 4768, "test": 1192},
        },
        "model": {"family": "mlp_federated", "scope": "global", "client_id": None},
        "seeds": {"split": 42, "validation": None, "partition": None, "training": None},
        "environment": {
            "python": environment["python"],
            "packages": environment["packages"],
            "platform": environment["platform"],
            "hardware": f"{environment['architecture']}; Ray exposed {threads} CPU resources",
            "threads": threads,
        },
        "preprocessing": None,
        "decision": {"positive_label": 1, "threshold": 0.5, "operator": ">"},
        "configuration": {
            "clients": 3,
            "rounds": 10,
            "local_epochs": 5,
            "batch_size": 32,
            "learning_rate": 0.01,
            "optimizer": "Adam recreated for each client fit call",
            "aggregation": "FedAvg",
            "client_resources": {"num_cpus": 1, "num_gpus": 0.0},
            "train_samples_per_client": [1590, 1589, 1589],
            "preprocessing": "median/mode imputation and LabelEncoder before split; StandardScaler fit on train",
        },
        "artifacts": artifact_refs,
        "notes": [
            "Ambiente reconstruído; não comprovado como ambiente original da publicação.",
            "Training seed é null porque não há evidência de sua fixação.",
            "Nenhum checkpoint, predição ou matriz de confusão foi produzido pelo script histórico.",
            "A execução terminou com exit code 0 e dez rounds, mas não constitui reprodução numérica exata dos valores publicados.",
            "source_dirty é null porque o estado do working tree no momento da execução não foi preservado.",
        ],
    }
    validate_bundle(manifest, history, metrics)
    return manifest, history, metrics


def _publish_exclusive(temporary: Path, target: Path) -> None:
    try:
        os.link(temporary, target)
    except FileExistsError as exc:
        raise ExportError(f"recusando sobrescrever artefato existente: {target.name}") from exc


def export_run(run_dir: Path, *, root: Path = ROOT,
               environment_record: Path = DEFAULT_ENVIRONMENT_RECORD) -> tuple[Path, Path, Path]:
    """Valida tudo em memória e publica os três JSONs sem sobrescrita."""
    run_dir = Path(run_dir)
    targets = tuple(run_dir / name for name in OUTPUT_NAMES)
    existing = [path.name for path in targets if path.exists()]
    _require(not existing, f"artefatos já existem: {', '.join(existing)}")
    manifest, history, metrics = build_artifacts(
        run_dir, root=Path(root), environment_record=Path(environment_record)
    )
    documents = {"manifest.json": manifest, "history.json": history, "metrics.json": metrics}

    temporary: dict[str, Path] = {}
    published: list[Path] = []
    try:
        for name, document in documents.items():
            with tempfile.NamedTemporaryFile(
                mode="wb", prefix=f".{name}.", suffix=".tmp", dir=run_dir, delete=False
            ) as stream:
                stream.write(_json_bytes(document))
                temporary[name] = Path(stream.name)
            temporary[name].chmod(0o644)
        for target in targets:
            _publish_exclusive(temporary[target.name], target)
            published.append(target)
        loaded = tuple(load_artifact(path) for path in targets)
        validate_bundle(*loaded)
        return targets
    except BaseException:
        for target in published:
            target.unlink(missing_ok=True)
        raise
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="diretório com logs preservados")
    args = parser.parse_args(argv)
    try:
        paths = export_run(args.run_dir)
    except (ExportError, ContractError, OSError, json.JSONDecodeError, SyntaxError) as exc:
        parser.exit(2, f"erro: {exc}\n")
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
