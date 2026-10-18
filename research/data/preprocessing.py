"""Imputação, one-hot e padronização ajustados exclusivamente no treino."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import statistics
from typing import Mapping, Sequence

FIT_SCOPE = "global_train"
UNKNOWN_POLICY = "bloco one-hot zerado"
IMPUTATION = "mediana numérica e moda categórica aprendidas no treino"
SCALING = (
    "z-score com média e desvio padrão populacional do treino imputado; "
    "escala 1 quando o desvio é zero"
)


class PreprocessingError(ValueError):
    """Pré-processamento inválido ou uso fora do escopo de ajuste."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PreprocessingError(message)


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _as_float(value: object) -> float | None:
    if _is_missing(value):
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_category(value: object) -> str | None:
    if _is_missing(value):
        return None
    normalized = str(value).strip()
    return normalized or None


def _detect_kind(present: Sequence[object]) -> str:
    for value in present:
        number = _as_float(value)
        if number is None:
            return "categorical"
    return "numeric"


class FeaturePreprocessor:
    """Congela estatísticas após um único `fit` sobre o treino global."""

    def __init__(self, feature_order: Sequence[str]):
        order = list(feature_order)
        _require(bool(order), "feature_order vazio")
        _require(len(set(order)) == len(order), "features duplicadas")
        _require(all(type(name) is str and name for name in order), "nomes de feature inválidos")
        self._order = order
        self._fitted = False
        self._kinds: dict[str, str] = {}
        self._numeric: dict[str, dict] = {}
        self._categorical: dict[str, dict] = {}

    @property
    def feature_order(self) -> list[str]:
        return list(self._order)

    @property
    def fitted(self) -> bool:
        return self._fitted

    @property
    def kinds(self) -> dict[str, str]:
        return dict(self._kinds)

    @property
    def output_features(self) -> list[str]:
        _require(self._fitted, "pré-processador não ajustado")
        output = []
        for column in self._order:
            if self._kinds[column] == "numeric":
                output.append(column)
            else:
                output.extend(f"{column}={category}"
                              for category in self._categorical[column]["categories"])
        return output

    def _columns(self, rows: Sequence[Mapping[str, object]]) -> dict[str, list[object]]:
        rows = list(rows)
        _require(bool(rows), "linhas vazias")
        for row in rows:
            _require(hasattr(row, "get"), "linha sem acesso por chave")
        return {column: [row.get(column) for row in rows] for column in self._order}

    def fit(self, rows: Sequence[Mapping[str, object]]) -> "FeaturePreprocessor":
        _require(not self._fitted, "pré-processador já congelado; refit não é permitido")
        columns = self._columns(rows)
        for column, values in columns.items():
            present = [value for value in values if not _is_missing(value)]
            _require(bool(present), f"coluna {column} sem valores no treino")
            kind = _detect_kind(present)
            self._kinds[column] = kind
            if kind == "numeric":
                self._numeric[column] = self._fit_numeric(values, present)
            else:
                self._categorical[column] = self._fit_categorical(values, present)
        self._fitted = True
        return self

    def _fit_numeric(self, values: Sequence[object], present: Sequence[object]) -> dict:
        numbers = [_as_float(value) for value in present]
        _require(all(number is not None for number in numbers),
                 "coluna numérica com valor não finito no treino")
        numbers = [number for number in numbers if number is not None]
        median = float(statistics.median(numbers))
        imputed = [median if _is_missing(value) else _as_float(value) for value in values]
        _require(all(number is not None for number in imputed), "imputação numérica inválida")
        imputed = [number for number in imputed if number is not None]
        mean = float(statistics.fmean(imputed))
        spread = float(statistics.pstdev(imputed)) if len(imputed) >= 2 else 0.0
        return {
            "median": median,
            "mean": mean,
            "scale": spread if spread > 0 else 1.0,
            "missing_in_train": sum(1 for value in values if _is_missing(value)),
        }

    def _fit_categorical(self, values: Sequence[object], present: Sequence[object]) -> dict:
        del present
        normalized = [_as_category(value) for value in values]
        observed = [value for value in normalized if value is not None]
        _require(bool(observed), "coluna categórica sem valores no treino")
        counts: dict[str, int] = {}
        for value in observed:
            counts[value] = counts.get(value, 0) + 1
        mode = min(counts, key=lambda value: (-counts[value], value))
        categories = sorted(counts)
        return {
            "mode": mode,
            "categories": categories,
            "missing_in_train": sum(1 for value in normalized if value is None),
        }

    def transform(self, rows: Sequence[Mapping[str, object]]) -> list[list[float]]:
        _require(self._fitted, "pré-processador não ajustado")
        columns = self._columns(rows)
        vectors: list[list[float]] = []
        for position in range(len(next(iter(columns.values())))):
            vector: list[float] = []
            for column in self._order:
                if self._kinds[column] == "numeric":
                    vector.append(self._numeric_value(column, columns[column][position]))
                else:
                    vector.extend(self._category_vector(column, columns[column][position]))
            vectors.append(vector)
        return vectors

    def _numeric_value(self, column: str, value: object) -> float:
        stats = self._numeric[column]
        number = _as_float(value)
        if number is None:
            number = stats["median"]
        return (number - stats["mean"]) / stats["scale"]

    def _category_vector(self, column: str, value: object) -> list[float]:
        stats = self._categorical[column]
        category = _as_category(value)
        if category is None:
            category = stats["mode"]
        vector = [0.0] * len(stats["categories"])
        if category in stats["categories"]:
            vector[stats["categories"].index(category)] = 1.0
        return vector

    def stats(self) -> dict:
        """Estrutura nova e independente do estado interno congelado."""
        _require(self._fitted, "pré-processador não ajustado")
        return {
            "fit_scope": FIT_SCOPE,
            "feature_order": self.feature_order,
            "numeric": copy.deepcopy(
                {column: self._numeric[column]
                 for column in self._order if self._kinds[column] == "numeric"}
            ),
            "categorical": copy.deepcopy(
                {column: self._categorical[column]
                 for column in self._order if self._kinds[column] == "categorical"}
            ),
            "imputation": IMPUTATION,
            "scaling": SCALING,
            "unknown_category_policy": UNKNOWN_POLICY,
        }

    def record(self) -> dict:
        """Identidade do transformador no formato `{id, sha256, fit_scope}` dos contratos."""
        payload = json.dumps(
            self.stats(), ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return {"id": f"preprocessing-{digest[:20]}", "sha256": digest, "fit_scope": FIT_SCOPE}
