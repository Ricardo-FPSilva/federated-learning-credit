"""Modelos clássicos comparadores da E3.1B.1, sem treinamento oficial.

Alcance: Majoria, Logistic Regression e Random Forest, com ajuste somente em
``DatasetView`` de treino e avaliação delegada a ``research.benchmark.
evaluation``. Não há execução das cinco seeds oficiais (42–46), métricas
científicas oficiais, checkpoints nem artefatos finais — isso fica para a E3.2,
cujo ``build_provenance`` é que valida ``training_seed`` oficial.

Decisão metodológica registrada — Majoria (aprovada em 18/10/2026 na E3.1B.1;
não altera os protocolos congelados, que na regra ``models`` somente pedem
prever a classe majoritária do treino global e mantêm o threshold estrito
``> 0.5``):

1. score = prevalência empírica de BAD=1 calculada exclusivamente no treino
   global;
2. score constante para todos os exemplos avaliados;
3. classe prevista segue estritamente ``score > 0.5``;
4. prevalência exatamente 0,5 → classe prevista 0 (empate não satisfaz a
   desigualdade estrita);
5. BCE calculada a partir dessa probabilidade constante, sem clipping
   arbitrário — certeza extrema incorreta resulta em ``bce_loss`` nulo com
   motivo explícito;
6. AP e ROC-AUC seguem as regras existentes, com a limitação de scores
   constantes documentada: havendo positivos, AP = prevalência dos labels;
   havendo ambas as classes, ROC-AUC = 0,5 (todos os pares empatados) — baseline
   de prevalência sem capacidade de discriminação individual; sem positivos AP é
   ``null`` e com classe única ROC-AUC é ``null``, pelas regras existentes;
7. campo ``logit`` da Majoria é ``null``: não existe logit natural (mesmo
   tratamento para Random Forest, que só expõe ``predict_proba``).

Contratos técnicos da correção de auditoria (sem decisão metodológica nova):

- hiperparâmetros congelados em ``MappingProxyType`` e ``params`` como cópia
  defensiva nova a cada acesso; ``fit`` usa exclusivamente a configuração
  congelada — nenhuma opção pública de tuning altera o estimador;
- labels validados antes de importar sklearn: LR exige exatamente as classes
  ``{0,1}``; RF aceita ``{0}``, ``{1}`` ou ``{0,1}``; Majoria aceita labels
  inteiros ``{0,1}`` — todos rejeitam 2, -1, strings, bools e multiclasses com
  ``ClassicModelError`` próprio;
- ``ClassicModel`` é um ABC simples: ``fit``, ``scores`` e ``params`` são
  abstratos; ``logits`` tem implementação padrão ``None``;
- ``RandomForestBaseline.training_seed`` é somente leitura e sempre coincide
  com ``params["random_state"]`` do estimador ajustado.

Recuperação pela E3.2 (runner): importar os modelos deste módulo; para a
Majoria usar ``positive_rate_`` (o próprio score constante) e ``supports_logits
is False`` → ``prediction_artifact.logit = null``; para Logistic Regression
usar ``supports_logits is True`` e ``logits()``; hiperparâmetros congelados
ficam em ``params`` e identidade de família em ``model_id``. Duas verificações
obrigatórias no caminho oficial, sem esquema de proveniência alterado:
``require_official_training_seed(model, seed)`` confere a seed declarada com as
oficiais e com a seed efetiva do RF; ``require_train_scope(model, data.train)``
exige que os índices do fit coincidam exatamente com ``data.train.indices``.
Os pontos 1–7 são regressão obrigatória de ``tests/test_classic_models.py``.
Eventual consolidação desta decisão em documento metodológico exige autorização
prévia.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from types import MappingProxyType
from typing import ClassVar, Mapping, Sequence

from research.benchmark.data import DatasetView
from research.benchmark.seeding import TRAINING_SEEDS

MAX_TRAINING_SEED = 2 ** 32 - 1


class ClassicModelError(ValueError):
    """Entrada, estado ou ambiente inválido para os modelos clássicos."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ClassicModelError(message)


def _binary_classes(model_name: str, labels: Sequence[int]) -> set[int]:
    """Valida labels inteiros 0/1 e devolve as classes presentes.

    Rejeita 2, -1, strings, bools e qualquer valor fora de ``{0,1}`` antes do
    sklearn, com erro próprio e mensagem clara.
    """
    _require(all(type(label) is int and label in (0, 1) for label in labels),
             f"{model_name}: labels devem ser inteiros 0/1")
    return set(labels)


def _positive_proba(estimator, view: DatasetView) -> tuple[float, ...]:
    """``P(y=1)`` por exemplo, inclusive quando o fit viu uma única classe."""
    classes = [int(label) for label in estimator.classes_]
    if classes == [1]:
        return (1.0,) * len(view)
    if classes == [0]:
        return (0.0,) * len(view)
    probabilities = estimator.predict_proba(list(view.features))
    column = classes.index(1)
    return tuple(float(value) for value in probabilities[:, column])


class ClassicModel(ABC):
    """Contrato comum: ``fit``, ``scores``, ``params``, ``logits``, ``model_id``.

    ``fit`` e ``scores`` e a property ``params`` são abstratos; cada modelo
    concreto declara ``model_id`` como ``ClassVar``. ``logits`` devolve ``None``
    por padrão (modelo sem logit natural) e é sobrescrito apenas quando
    ``supports_logits`` é verdadeiro. ``fit_indices`` registra a origem exata do
    ajuste para a verificação de escopo da E3.2.
    """

    supports_logits: ClassVar[bool] = False
    model_id: ClassVar[str]

    def __init__(self) -> None:
        self.fit_indices: tuple[int, ...] = ()
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    @abstractmethod
    def params(self) -> dict[str, object]:
        """Cópia defensiva da configuração congelada, nova a cada acesso."""

    @abstractmethod
    def fit(self, train: DatasetView) -> ClassicModel:
        """Ajusta somente no ``DatasetView`` de treino recebido."""

    @abstractmethod
    def scores(self, view: DatasetView) -> tuple[float, ...]:
        """Score contínuo ``P(BAD=1)`` por exemplo do ``view`` recebido."""

    def _validated_train(self, train: DatasetView) -> DatasetView:
        _require(isinstance(train, DatasetView),
                 f"{type(self).__name__}: fit exige DatasetView")
        _require(len(train) > 0, f"{type(self).__name__}: treino vazio")
        return train

    def _record_fit(self, train: DatasetView) -> None:
        """Registra a origem exata do ajuste para prova de ausência de leakage."""
        self.fit_indices = tuple(train.indices)
        self._fitted = True

    def _ensure_fitted(self) -> None:
        _require(self._fitted, f"{type(self).__name__}: fit() ainda não executado")

    def logits(self, view: DatasetView) -> tuple[float, ...] | None:
        """``None`` para modelos sem logit natural — ponto 7 da decisão."""
        self._ensure_fitted()
        return None


class MajorityBaseline(ClassicModel):
    """Baseline de prevalência aprovada: score constante, sem discriminação.

    ``positive_rate_`` é o próprio score aprovado (pontos 1–2); a decisão usa
    a regra estrita ``score > 0.5`` (ponto 3), com empate em 0,5 prevendo 0
    (ponto 4). Features e labels além dos do treino global nunca são lidos.
    ``params`` é um dicionário vazio: não há hiperparâmetros.
    """

    model_id: ClassVar[str] = "majority"

    def __init__(self) -> None:
        super().__init__()
        self.positive_rate_: float | None = None

    @property
    def params(self) -> dict[str, object]:
        return {}

    def fit(self, train: DatasetView) -> MajorityBaseline:
        train = self._validated_train(train)
        _binary_classes(self.model_id, train.labels)
        self.positive_rate_ = sum(train.labels) / len(train.labels)
        self._record_fit(train)
        return self

    def scores(self, view: DatasetView) -> tuple[float, ...]:
        self._ensure_fitted()
        return (self.positive_rate_,) * len(view)


class LogisticRegressionBaseline(ClassicModel):
    """Logistic Regression com os hiperparâmetros congelados (spec, regra 31).

    Determinística (lbfgs) e sem seed própria. Exige exatamente as classes
    ``{0,1}`` no treino, validadas antes do sklearn. ``logits()`` é o
    ``decision_function`` exato: ``Sigmoid(logit) == score``. A configuração
    congelada é um ``MappingProxyType`` de classe: ``params`` devolve sempre uma
    cópia nova e o ``fit`` nunca lê estado externo mutável.
    """

    model_id: ClassVar[str] = "logistic_regression"
    supports_logits: ClassVar[bool] = True

    _FROZEN_PARAMS: ClassVar[Mapping[str, object]] = MappingProxyType({
        "penalty": "l2",
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 1000,
        "tol": 1e-4,
        "class_weight": None,
    })

    def __init__(self) -> None:
        super().__init__()
        self.estimator_ = None

    @property
    def params(self) -> dict[str, object]:
        return dict(self._FROZEN_PARAMS)

    def fit(self, train: DatasetView) -> LogisticRegressionBaseline:
        train = self._validated_train(train)
        classes = _binary_classes(self.model_id, train.labels)
        _require(classes == {0, 1},
                 "LogisticRegressionBaseline: treino precisa de exatamente as "
                 "classes 0 e 1")
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError as exc:
            raise ClassicModelError("scikit-learn não instalado neste ambiente") from exc
        estimator = LogisticRegression(**dict(self._FROZEN_PARAMS))
        estimator.fit(list(train.features), list(train.labels))
        self.estimator_ = estimator
        self._record_fit(train)
        return self

    def scores(self, view: DatasetView) -> tuple[float, ...]:
        self._ensure_fitted()
        return _positive_proba(self.estimator_, view)

    def logits(self, view: DatasetView) -> tuple[float, ...]:
        self._ensure_fitted()
        return tuple(float(value)
                     for value in self.estimator_.decision_function(list(view.features)))


class RandomForestBaseline(ClassicModel):
    """Random Forest com os hiperparâmetros congelados (spec, regra 32).

    ``random_state`` é exatamente ``training_seed``, propriedade somente
    leitura validada como inteiro em ``[0, 2**32-1]``; as seeds oficiais 42–46
    são conferidas por ``require_official_training_seed`` no caminho oficial da
    E3.2 — os testes usam seeds locais para não executar as cinco seeds
    oficiais neste passo. Aceita exatamente as classes ``{0}``, ``{1}`` ou
    ``{0,1}``. Sem logit natural: só ``predict_proba`` (ponto 7). A configuração
    congelada é um ``MappingProxyType`` de classe acrescido de ``random_state``
    em cada cópia de ``params``.
    """

    model_id: ClassVar[str] = "random_forest"

    _FROZEN_PARAMS: ClassVar[Mapping[str, object]] = MappingProxyType({
        "n_estimators": 100,
        "criterion": "gini",
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "bootstrap": True,
        "class_weight": None,
        "n_jobs": 1,
    })

    def __init__(self, training_seed: int) -> None:
        super().__init__()
        _require(type(training_seed) is int
                 and 0 <= training_seed <= MAX_TRAINING_SEED,
                 "training_seed deve ser inteiro em [0, 2**32-1]")
        self._training_seed = training_seed
        self.estimator_ = None

    @property
    def training_seed(self) -> int:
        """Seed efetiva do modelo, somente leitura; coincide com o estimador."""
        return self._training_seed

    @property
    def params(self) -> dict[str, object]:
        return {**self._FROZEN_PARAMS, "random_state": self._training_seed}

    def fit(self, train: DatasetView) -> RandomForestBaseline:
        train = self._validated_train(train)
        _binary_classes(self.model_id, train.labels)
        try:
            from sklearn.ensemble import RandomForestClassifier
        except ImportError as exc:
            raise ClassicModelError("scikit-learn não instalado neste ambiente") from exc
        estimator = RandomForestClassifier(**dict(self.params))
        estimator.fit(list(train.features), list(train.labels))
        self.estimator_ = estimator
        self._record_fit(train)
        return self

    def scores(self, view: DatasetView) -> tuple[float, ...]:
        self._ensure_fitted()
        return _positive_proba(self.estimator_, view)


def require_official_training_seed(model: ClassicModel, training_seed: int) -> None:
    """Valida a seed declarada pelo runner contra as oficiais e a do modelo.

    Uso obrigatório da E3.2 antes do fit/avaliação oficial: confere que
    ``training_seed`` pertence a 42–46 e, quando o modelo expõe seed própria
    (Random Forest), que ela coincide exatamente com a declarada. Não altera o
    esquema de proveniência congelado — só antecipa erro claro.
    """
    _require(isinstance(model, ClassicModel), "model deve ser um modelo clássico")
    _require(type(training_seed) is int and training_seed in TRAINING_SEEDS,
             "training_seed deve ser uma das seeds oficiais 42–46")
    declared = getattr(model, "training_seed", None)
    if declared is not None:
        _require(declared == training_seed,
                 f"seed do modelo ({declared}) diverge da declarada ({training_seed})")


def require_train_scope(model: ClassicModel, train: DatasetView) -> None:
    """Exige que o ajuste tenha usado exatamente os índices de ``train``.

    Verificação reutilizável e obrigatória no caminho oficial da E3.2: após o
    ``fit``, compara ``model.fit_indices`` com ``train.indices`` (ordem e
    conteúdo) — normalmente ``data.train.indices`` — e falha alto se o modelo
    foi ajustado em validação, teste, partição de cliente ou qualquer outro
    escopo. Não há caminho de silêncio: a divergência sempre levanta
    ``ClassicModelError``.
    """
    _require(isinstance(model, ClassicModel), "model deve ser um modelo clássico")
    _require(model.is_fitted, f"{type(model).__name__}: fit() ainda não executado")
    _require(isinstance(train, DatasetView), "train deve ser um DatasetView")
    _require(model.fit_indices == tuple(train.indices),
             "índices do fit divergem dos índices oficiais de treino")
