"""Separação estratificada treino/validação/teste conforme benchmark-v1."""

from __future__ import annotations

from collections import defaultdict
import math
import random

SPLIT_NAMES = ("train", "validation", "test")
DEFAULT_FRACTIONS: dict[str, float] = {"train": 0.6, "validation": 0.2, "test": 0.2}
DEFAULT_SPLIT_SEED = 42
TOLERANCE = 1e-9


class SplitError(ValueError):
    """Configuração de split inválida ou resultado inconsistente."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SplitError(message)


def validate_fractions(fractions: object) -> dict[str, float]:
    _require(type(fractions) is dict, "fractions: objeto esperado")
    _require(set(fractions) == set(SPLIT_NAMES), "fractions: exatamente train/validation/test")
    clean: dict[str, float] = {}
    for name in SPLIT_NAMES:
        value = fractions[name]  # type: ignore[index]
        _require(type(value) in (int, float), f"fractions.{name}: número esperado")
        number = float(value)
        _require(math.isfinite(number) and number > 0, f"fractions.{name}: positivo e finito esperado")
        clean[name] = number
    _require(abs(sum(clean.values()) - 1.0) <= TOLERANCE, "fractions: soma deve ser 1")
    return clean


def _allocate(count: int, fractions: dict[str, float]) -> dict[str, int]:
    """Maior resto: garante soma exata e desvio por classe <= 1 observação."""
    raw = {name: count * fractions[name] for name in SPLIT_NAMES}
    base = {name: int(math.floor(raw[name])) for name in SPLIT_NAMES}
    remainder = count - sum(base.values())
    order = sorted(
        SPLIT_NAMES,
        key=lambda name: (-(raw[name] - base[name]), SPLIT_NAMES.index(name)),
    )
    for name in order[:remainder]:
        base[name] += 1
    return base


def _check_result(rows: list[int], result: dict[str, list[int]]) -> None:
    seen: set[int] = set()
    for name in SPLIT_NAMES:
        chunk = result[name]
        _require(len(chunk) == len(set(chunk)), f"{name}: índices duplicados")
        _require(not (seen & set(chunk)), "splits com sobreposição")
        seen.update(chunk)
    _require(seen == set(rows), "splits não cobrem exatamente as observações")


def stratified_split(
    row_indices: list[int],
    labels: list[object],
    fractions: dict[str, float] | None = None,
    seed: int | None = None,
) -> dict[str, list[int]]:
    """Devolve índices originais por split, estratificados e ordenados de forma canônica."""
    fractions = DEFAULT_FRACTIONS if fractions is None else fractions
    seed = DEFAULT_SPLIT_SEED if seed is None else seed
    fractions = validate_fractions(fractions)
    _require(type(seed) is int and seed >= 0, "seed: inteiro >= 0 esperado")
    rows = list(row_indices)
    targets = list(labels)
    _require(bool(rows), "conjunto vazio")
    _require(len(rows) == len(targets), "row_indices e labels com tamanhos diferentes")
    _require(all(type(index) is int for index in rows), "índices devem ser inteiros")
    _require(len(set(rows)) == len(rows), "índices de linha duplicados")
    _require(all(type(label) is type(targets[0]) for label in targets),
             "tipos de rótulo inconsistentes")

    groups: dict[object, list[int]] = defaultdict(list)
    for index, label in zip(rows, targets):
        groups[label].append(index)

    generator = random.Random(seed)
    selected: dict[str, list[int]] = {name: [] for name in SPLIT_NAMES}
    for label in sorted(groups):
        group = sorted(groups[label])
        generator.shuffle(group)
        counts = _allocate(len(group), fractions)
        start = 0
        for name in SPLIT_NAMES:
            stop = start + counts[name]
            selected[name].extend(group[start:stop])
            start = stop

    result = {name: sorted(selected[name]) for name in SPLIT_NAMES}
    _check_result(rows, result)
    return result
