"""Particionamento IID reproduzível em clientes simulados."""

from __future__ import annotations

import hashlib
import json
import random

DEFAULT_PARTITION_SEED = 44
DEFAULT_N_CLIENTS = 3


class PartitionError(ValueError):
    """Configuração de partição inválida ou resultado inconsistente."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PartitionError(message)


def validate_indices(indices: object) -> list[int]:
    _require(type(indices) in (list, tuple), "indices: lista esperada")
    values = list(indices)
    _require(bool(values), "indices vazio")
    _require(all(type(index) is int and index >= 0 for index in values),
             "índices devem ser inteiros não negativos")
    _require(len(set(values)) == len(values), "índices duplicados")
    return values


def iid_partition(
    indices: list[int],
    n_clients: int = DEFAULT_N_CLIENTS,
    seed: int = DEFAULT_PARTITION_SEED,
) -> dict[str, list[int]]:
    """Distribuição round-robin após embaralhamento; cada treino em um único cliente."""
    values = validate_indices(indices)
    _require(type(n_clients) is int and n_clients >= 2, "n_clients: inteiro >= 2 esperado")
    _require(type(seed) is int and seed >= 0, "seed: inteiro >= 0 esperado")
    _require(len(values) >= n_clients, "menos observações que clientes")

    shuffled = sorted(values)
    random.Random(seed).shuffle(shuffled)
    client_ids = [f"client-{index}" for index in range(n_clients)]
    buckets: dict[str, list[int]] = {client: [] for client in client_ids}
    for position, index in enumerate(shuffled):
        buckets[client_ids[position % n_clients]].append(index)

    partition = {client: sorted(bucket) for client, bucket in buckets.items()}
    covered: set[int] = set()
    for client in client_ids:
        _require(bool(partition[client]), f"{client} sem observações")
        _require(not (covered & set(partition[client])), "clientes com sobreposição")
        covered.update(partition[client])
    _require(covered == set(values), "partição não cobre exatamente o treino")
    return partition


def partition_digest(partition: dict[str, list[int]]) -> str:
    payload = json.dumps(
        partition, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
