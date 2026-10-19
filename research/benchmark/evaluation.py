"""Métricas globais calculadas diretamente de labels, scores e logits individuais.

Quando há logits, eles são a fonte de verdade: ``bce_loss`` deriva diretamente
deles por formulação numericamente estável equivalente a ``BCEWithLogitsLoss``
(sem converter para probabilidade e sem clipping), os scores efetivos usados em
predições, AP e ROC-AUC derivam de ``Sigmoid`` estável — preservando a regra
congelada ``score > 0.5`` — e o score fornecido é somente validado contra essa
derivação com ``SCORE_LOGIT_TOLERANCE``; pares score/logit contraditórios são
rejeitados. Sem logits, os scores continuam sendo a fonte documentada para as
métricas classificatórias e para a BCE.
"""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Iterable, Sequence

SCORE_LOGIT_TOLERANCE = 1e-4


class EvaluationError(ValueError):
    """Predições ou losses incompatíveis com a avaliação congelada."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluationError(message)


def _as_index(value: int) -> int:
    _require(type(value) is not bool and isinstance(value, Integral),
             "índice original deve ser um inteiro")
    return int(value)


def _as_label(value: int) -> int:
    _require(type(value) is not bool and isinstance(value, Integral),
             "labels devem ser inteiros 0/1")
    label = int(value)
    _require(label in (0, 1), "labels devem ser inteiros 0/1")
    return label


def _as_score(value: float) -> float:
    _require(type(value) is not bool and isinstance(value, Real),
             "scores devem ser numéricos")
    score = float(value)
    _require(math.isfinite(score) and 0.0 <= score <= 1.0,
             "scores devem ser finitos em [0,1]")
    return score


def _as_logit(value: float) -> float:
    _require(type(value) is not bool and isinstance(value, Real),
             "logits devem ser numéricos")
    logit = float(value)
    _require(math.isfinite(logit), "logits devem ser finitos")
    return logit


def _validate(labels: Sequence[int], scores: Sequence[float]) -> tuple[list[int], list[float]]:
    labels = list(labels)
    scores = list(scores)
    _require(bool(labels) and len(labels) == len(scores), "labels/scores vazios ou divergentes")
    return [_as_label(label) for label in labels], [_as_score(score) for score in scores]


def _average_precision(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    if positives == 0:
        return None
    ordered = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    tp = fp = 0
    previous_recall = 0.0
    result = 0.0
    position = 0
    while position < len(ordered):
        score = ordered[position][0]
        end = position
        while end < len(ordered) and ordered[end][0] == score:
            if ordered[end][1] == 1:
                tp += 1
            else:
                fp += 1
            end += 1
        recall = tp / positives
        precision = tp / (tp + fp)
        result += (recall - previous_recall) * precision
        previous_recall = recall
        position = end
    return result


def _roc_auc(labels: list[int], scores: list[float]) -> float | None:
    positive_scores = [score for label, score in zip(labels, scores) if label == 1]
    negative_scores = [score for label, score in zip(labels, scores) if label == 0]
    if not positive_scores or not negative_scores:
        return None
    wins = 0.0
    for positive in positive_scores:
        for negative in negative_scores:
            wins += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
    return wins / (len(positive_scores) * len(negative_scores))


def _bce_from_logits(labels: list[int], logits: list[float]) -> float:
    """Média de BCEWithLogitsLoss estável: relu(x) - x*y + log1p(exp(-|x|))."""
    total = 0.0
    for label, logit in zip(labels, logits):
        total += max(logit, 0.0) - logit * label + math.log1p(math.exp(-abs(logit)))
    return total / len(labels)


def _sigmoid_stable(logit: float) -> float:
    """Sigmoid estável para qualquer logit finito, sem overflow nem clipping."""
    if logit >= 0.0:
        return 1.0 / (1.0 + math.exp(-logit))
    exp_logit = math.exp(logit)
    return exp_logit / (1.0 + exp_logit)


def _bce_from_scores(labels: list[int], scores: list[float]) -> tuple[float | None, str | None]:
    total = 0.0
    for label, score in zip(labels, scores):
        if (label == 1 and score == 0.0) or (label == 0 and score == 1.0):
            return None, "BCE infinita para score de certeza incorreta"
        total -= math.log(score) if label == 1 else math.log1p(-score)
    return total / len(labels), None


def evaluate_scores(labels: Sequence[int], scores: Sequence[float],
                    *, logits: Sequence[float] | None = None) -> dict:
    """Calcula uma avaliação global; threshold é fixo e estritamente `>0.5`.

    Com ``logits``, a fonte de verdade é o logit: os scores efetivos das métricas
    classificatórias derivam de ``Sigmoid`` estável, o score fornecido deve ser
    coerente com essa derivação (``SCORE_LOGIT_TOLERANCE``) e ``bce_loss`` vem
    diretamente dos logits. Sem ``logits``, ``bce_loss`` deriva dos scores.
    """
    labels, scores = _validate(labels, scores)
    checked_logits: list[float] | None = None
    if logits is not None:
        logits = list(logits)
        _require(len(logits) == len(labels), "logits/labels vazios ou divergentes")
        checked_logits = [_as_logit(logit) for logit in logits]
        derived_scores = [_sigmoid_stable(logit) for logit in checked_logits]
        for position, (provided, derived) in enumerate(zip(scores, derived_scores)):
            _require(abs(provided - derived) <= SCORE_LOGIT_TOLERANCE,
                     f"score {provided} incoerente com Sigmoid(logit)={derived} "
                     f"no índice {position}")
        scores = derived_scores
    predictions = [1 if score > 0.5 else 0 for score in scores]
    tn = fp = fn = tp = 0
    for label, prediction in zip(labels, predictions):
        if label == 0 and prediction == 0:
            tn += 1
        elif label == 0:
            fp += 1
        elif prediction == 0:
            fn += 1
        else:
            tp += 1
    count = len(labels)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1_denominator = 2 * tp + fp + fn
    f1 = 2 * tp / f1_denominator if f1_denominator else None
    average_precision = _average_precision(labels, scores)
    roc_auc = _roc_auc(labels, scores)
    if checked_logits is not None:
        bce_loss, bce_reason = _bce_from_logits(labels, checked_logits), None
    else:
        bce_loss, bce_reason = _bce_from_scores(labels, scores)
    values = {
        "bce_loss": bce_loss,
        "accuracy": (tn + tp) / count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "average_precision": average_precision,
        "roc_auc": roc_auc,
    }
    reasons = {
        "precision": None if precision is not None else "nenhuma predição positiva",
        "recall": None if recall is not None else "nenhum positivo real",
        "f1": None if f1 is not None else "denominador F1 igual a zero",
        "average_precision": None if average_precision is not None else "nenhum positivo real",
        "roc_auc": None if roc_auc is not None else "ROC-AUC exige ambas as classes",
        "bce_loss": bce_reason,
        "accuracy": None,
    }
    return {
        "sample_count": count,
        "values": values,
        "undefined_reasons": reasons,
        "confusion_matrix": {
            "labels": [0, 1],
            "orientation": "true_rows_predicted_columns",
            "counts": [[tn, fp], [fn, tp]],
        },
        "predicted_labels": predictions,
    }


def weighted_batch_loss(batch_means: Iterable[tuple[float, int]]) -> float:
    """Agrega médias de batch ponderando pelo número real de exemplos."""
    total_loss = 0.0
    total_examples = 0
    for mean_loss, count in batch_means:
        _require(type(mean_loss) is not bool and isinstance(mean_loss, Real)
                 and math.isfinite(float(mean_loss)) and float(mean_loss) >= 0,
                 "loss de batch inválida")
        _require(type(count) is not bool and isinstance(count, Integral) and count > 0,
                 "tamanho de batch inválido")
        total_loss += float(mean_loss) * int(count)
        total_examples += int(count)
    _require(total_examples > 0, "nenhum batch para agregar")
    return total_loss / total_examples


def evaluate_global(records: Sequence[tuple[Sequence[int], Sequence[int], Sequence[float]]],
                    expected_indices: Iterable[int],
                    *, logits: Sequence[Sequence[float]] | None = None) -> dict:
    """Agrega registros ``(indices, labels, scores)`` identificados por índice original.

    Rejeita índices duplicados, fora do escopo ou ausentes; valida cobertura
    completa do escopo antes de calcular. Métricas são recalculadas a partir das
    predições individuais concatenadas em ordem de índice original — nunca pela
    média de F1/AP/AUC locais. ``logits`` opcional é alinhado a cada registro e
    alimenta somente a ``bce_loss``.
    """
    expected = [_as_index(index) for index in expected_indices]
    _require(bool(expected), "escopo de índices esperado vazio")
    expected_set = set(expected)
    _require(len(expected_set) == len(expected), "escopo de índices esperado duplicado")
    records = list(records)
    _require(bool(records), "nenhuma predição para avaliação global")
    if logits is not None:
        logits = list(logits)
        _require(len(logits) == len(records), "logits/records divergentes")
    collected: dict[int, tuple[int, float, float | None]] = {}
    for position, record in enumerate(records):
        _require(type(record) in (tuple, list) and len(record) == 3,
                 "registro deve ser a sequência (indices, labels, scores)")
        raw_indices, local_labels, local_scores = record
        checked_labels, checked_scores = _validate(local_labels, local_scores)
        indices = [_as_index(index) for index in raw_indices]
        _require(len(indices) == len(checked_labels), "indices/labels vazios ou divergentes")
        local_logits: list[float] | None = None
        if logits is not None:
            local_logits = [_as_logit(logit) for logit in logits[position]]
            _require(len(local_logits) == len(indices), "logits/indices divergentes")
        for offset, index in enumerate(indices):
            _require(index in expected_set, f"índice {index} fora do escopo esperado")
            _require(index not in collected, f"índice {index} duplicado")
            collected[index] = (
                checked_labels[offset], checked_scores[offset],
                None if local_logits is None else local_logits[offset],
            )
    missing = expected_set - set(collected)
    _require(not missing, f"cobertura incompleta: {len(missing)} índices ausentes")
    ordered = [collected[index] for index in sorted(collected)]
    labels = [row[0] for row in ordered]
    scores = [row[1] for row in ordered]
    if logits is None:
        return evaluate_scores(labels, scores)
    return evaluate_scores(labels, scores, logits=[row[2] for row in ordered])
