"""Testes E3.1B.1 — modelos clássicos; ajustes locais, sem métricas oficiais.

Regressão dos sete pontos da decisão Majoria registrada em
``research/benchmark/classic.py``: (1) score = prevalência exclusiva do treino
global; (2) score constante; (3) decisão estrita ``score > 0.5``; (4) empate
0,5 → classe 0; (5) BCE da probabilidade constante sem clipping; (6) AP =
prevalência e ROC-AUC = 0,5 sob score constante; (7) ``logit`` nulo. Todos os
testes usam apenas o treino global ou views sintéticas — nunca o conjunto de
teste — não produzem resultados oficiais e não executam as seeds oficiais
42–46 (as do RandomForest local são 7 e 11). Inclui as regressões da auditoria
E3.1B.1: imutabilidade dos hiperparâmetros, labels binários, contrato comum dos
modelos, correspondência de seed do runner, escopo exato do fit, probabilidades
extremas da LR, integração da RF com ``evaluate_scores`` e AP/AUC indefinidos
para classes únicas.
"""

import importlib.util
import math
from pathlib import Path
import unittest

from research.benchmark.classic import (
    ClassicModel,
    ClassicModelError,
    LogisticRegressionBaseline,
    MajorityBaseline,
    RandomForestBaseline,
    require_official_training_seed,
    require_train_scope,
)
from research.benchmark.data import DatasetView, load_experimental_data
from research.benchmark.evaluation import evaluate_scores

HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "hmeq.csv"
DATA_RECORD = ROOT / "results/data/benchmark-v1-split.json"


def _view(labels, *, start=10_000):
    """View sintética imutável com índices arbitrários fora dos splits E2."""
    features = tuple((float(index % 5), float(index % 3), 1.0)
                     for index in range(len(labels)))
    return DatasetView(
        indices=tuple(start + offset for offset in range(len(labels))),
        features=features,
        labels=tuple(labels),
    )


class MajorityBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_experimental_data(DATASET, DATA_RECORD)

    def test_score_is_prevalence_of_global_train_only(self):
        model = MajorityBaseline().fit(self.data.train)
        train_labels = self.data.train.labels
        self.assertEqual(model.positive_rate_, sum(train_labels) / len(train_labels))
        # leakage: origem do fit é exatamente o treino, sem validação/teste
        self.assertEqual(model.fit_indices, tuple(self.data.train.indices))
        self.assertTrue(set(model.fit_indices).isdisjoint(self.data.validation.indices))
        self.assertTrue(set(model.fit_indices).isdisjoint(self.data.test.indices))
        # e o score nunca foi derivado de validação, teste ou conjunto completo
        self.assertNotEqual(
            model.positive_rate_,
            sum(self.data.validation.labels) / len(self.data.validation.labels))
        self.assertNotEqual(
            model.positive_rate_,
            sum(self.data.test.labels) / len(self.data.test.labels))

    def test_scores_are_constant_and_decision_is_strict_threshold(self):
        model = MajorityBaseline().fit(self.data.train)
        scores = model.scores(self.data.train)
        self.assertEqual(len(scores), len(self.data.train))
        self.assertTrue(all(type(score) is float for score in scores))
        self.assertTrue(all(score == model.positive_rate_ for score in scores))
        self.assertLess(model.positive_rate_, 0.5)
        result = evaluate_scores(self.data.train.labels, scores)
        self.assertTrue(all(label == 0 for label in result["predicted_labels"]))

    def test_prevalence_exactly_half_predicts_zero(self):
        model = MajorityBaseline().fit(_view([0, 1]))
        self.assertEqual(model.positive_rate_, 0.5)
        result = evaluate_scores([0, 1], model.scores(_view([0, 1])))
        self.assertEqual(result["predicted_labels"], [0, 0])

    def test_prevalence_above_half_predicts_positive_class(self):
        model = MajorityBaseline().fit(_view([1, 1, 0]))
        self.assertEqual(model.positive_rate_, 2 / 3)
        result = evaluate_scores([1, 1, 0], model.scores(_view([1, 1, 0])))
        self.assertEqual(result["predicted_labels"], [1, 1, 1])

    def test_bce_uses_constant_probability_without_clipping(self):
        model = MajorityBaseline().fit(_view([0, 0, 0, 1]))
        result = evaluate_scores([0, 0, 0, 1], model.scores(_view([0, 0, 0, 1])))
        rate = 0.25
        expected = -(rate * math.log(rate) + (1.0 - rate) * math.log(1.0 - rate))
        self.assertIsNotNone(result["values"]["bce_loss"])
        self.assertIsNone(result["undefined_reasons"]["bce_loss"])
        self.assertAlmostEqual(result["values"]["bce_loss"], expected, places=12)

    def test_single_class_extremes_make_bce_undefined_without_clipping(self):
        negatives = MajorityBaseline().fit(_view([0] * 8))
        self.assertEqual(negatives.positive_rate_, 0.0)
        mixed = evaluate_scores([0, 1], negatives.scores(_view([0, 1])))
        self.assertIsNone(mixed["values"]["bce_loss"])
        self.assertIsNotNone(mixed["undefined_reasons"]["bce_loss"])
        pure = evaluate_scores([0, 0], negatives.scores(_view([0, 0])))
        self.assertEqual(pure["values"]["bce_loss"], 0.0)
        positives = MajorityBaseline().fit(_view([1] * 8))
        self.assertEqual(positives.positive_rate_, 1.0)
        mixed = evaluate_scores([1, 0], positives.scores(_view([1, 0])))
        self.assertIsNone(mixed["values"]["bce_loss"])
        self.assertIsNotNone(mixed["undefined_reasons"]["bce_loss"])

    def test_constant_scores_give_ap_prevalence_and_auc_half(self):
        model = MajorityBaseline().fit(self.data.train)
        labels = list(self.data.train.labels)
        result = evaluate_scores(labels, model.scores(self.data.train))
        prevalence = sum(labels) / len(labels)
        self.assertEqual(result["values"]["average_precision"], prevalence)
        self.assertEqual(result["values"]["roc_auc"], 0.5)
        self.assertIsNone(result["undefined_reasons"]["average_precision"])
        self.assertIsNone(result["undefined_reasons"]["roc_auc"])

    def test_logit_is_null_and_supports_flags(self):
        model = MajorityBaseline().fit(self.data.train)
        self.assertIsNone(model.logits(self.data.train))
        self.assertFalse(MajorityBaseline.supports_logits)
        self.assertFalse(RandomForestBaseline.supports_logits)
        self.assertTrue(LogisticRegressionBaseline.supports_logits)

    def test_invalid_inputs_are_rejected(self):
        model = MajorityBaseline()
        with self.assertRaises(ClassicModelError):
            model.scores(self.data.train)
        with self.assertRaises(ClassicModelError):
            model.logits(self.data.train)
        with self.assertRaises(ClassicModelError):
            MajorityBaseline().fit(DatasetView(indices=(), features=(), labels=()))
        with self.assertRaises(ClassicModelError):
            MajorityBaseline().fit(_view([0, 2]))
        with self.assertRaises(ClassicModelError):
            MajorityBaseline().fit("treino")
        with self.assertRaises(ClassicModelError):
            MajorityBaseline().fit([0, 1])

    def test_majority_single_class_leaves_ap_and_auc_undefined(self):
        negatives = MajorityBaseline().fit(_view([0] * 6))
        below = evaluate_scores([0, 0], negatives.scores(_view([0, 0])))
        self.assertIsNone(below["values"]["average_precision"])
        self.assertIsNotNone(below["undefined_reasons"]["average_precision"])
        self.assertIsNone(below["values"]["roc_auc"])
        self.assertIsNotNone(below["undefined_reasons"]["roc_auc"])
        positives = MajorityBaseline().fit(_view([1] * 6))
        above = evaluate_scores([1, 1], positives.scores(_view([1, 1])))
        self.assertEqual(above["values"]["average_precision"], 1.0)
        self.assertIsNone(above["values"]["roc_auc"])
        self.assertIsNotNone(above["undefined_reasons"]["roc_auc"])

    def test_require_train_scope_detects_wrong_and_right_fit(self):
        wrong = MajorityBaseline().fit(_view([0, 1]))
        with self.assertRaises(ClassicModelError):
            require_train_scope(wrong, self.data.train)
        right = MajorityBaseline().fit(self.data.train)
        require_train_scope(right, self.data.train)
        with self.assertRaises(ClassicModelError):
            require_train_scope(MajorityBaseline(), self.data.train)


class FrozenContractTests(unittest.TestCase):
    """Parâmetros e identificadores congelados, verificáveis sem sklearn."""

    def test_logistic_regression_parameters_match_frozen_spec(self):
        model = LogisticRegressionBaseline()
        self.assertEqual(model.params, {
            "penalty": "l2",
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 1000,
            "tol": 1e-4,
            "class_weight": None,
        })

    def test_random_forest_parameters_match_frozen_spec(self):
        self.assertEqual(RandomForestBaseline(training_seed=7).params, {
            "n_estimators": 100,
            "criterion": "gini",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "bootstrap": True,
            "class_weight": None,
            "n_jobs": 1,
            "random_state": 7,
        })
        self.assertEqual(
            RandomForestBaseline(training_seed=11).params["random_state"], 11)

    def test_training_seed_is_validated(self):
        for invalid in (-1, 2 ** 32, 4.2, "7", True, None):
            with self.assertRaises(ClassicModelError):
                RandomForestBaseline(training_seed=invalid)
        RandomForestBaseline(training_seed=0)
        RandomForestBaseline(training_seed=2 ** 32 - 1)

    def test_model_ids_match_protocol_families(self):
        self.assertEqual(MajorityBaseline().model_id, "majority")
        self.assertEqual(LogisticRegressionBaseline().model_id, "logistic_regression")
        self.assertEqual(RandomForestBaseline(training_seed=7).model_id, "random_forest")

    def test_logistic_regression_rejects_single_class_training(self):
        # rejeição acontece antes do sklearn: classe única não é ajustável
        with self.assertRaises(ClassicModelError):
            LogisticRegressionBaseline().fit(_view([0, 0, 0]))

    def test_common_interface_requires_fit_before_use(self):
        models = (
            MajorityBaseline(),
            LogisticRegressionBaseline(),
            RandomForestBaseline(training_seed=7),
        )
        for model in models:
            self.assertIsInstance(model, ClassicModel)
            self.assertFalse(model.is_fitted)
            with self.assertRaises(ClassicModelError):
                model.scores(_view([0, 1]))
            with self.assertRaises(ClassicModelError):
                model.logits(_view([0, 1]))

    def test_concrete_models_share_the_documented_surface(self):
        families = {"majority", "logistic_regression", "random_forest"}
        models = (
            MajorityBaseline(),
            LogisticRegressionBaseline(),
            RandomForestBaseline(training_seed=7),
        )
        for model in models:
            self.assertIn(model.model_id, families)
            self.assertIsInstance(model.params, dict)
            self.assertIsInstance(model.fit_indices, tuple)
            self.assertEqual(model.fit_indices, ())
            self.assertIsInstance(model.supports_logits, bool)
            self.assertTrue(callable(model.fit))
            self.assertTrue(callable(model.scores))
            self.assertTrue(callable(model.logits))
        self.assertEqual(MajorityBaseline().params, {})

    def test_classic_model_base_is_abstract(self):
        with self.assertRaises(TypeError):
            ClassicModel()

    def test_params_are_defensive_copies_for_every_model(self):
        models = (
            MajorityBaseline(),
            LogisticRegressionBaseline(),
            RandomForestBaseline(training_seed=7),
        )
        for model in models:
            with self.subTest(model=model.model_id):
                pristine = model.params
                copy = model.params
                self.assertEqual(pristine, copy)
                self.assertIsNot(copy, model.params)
                for key in copy:
                    copy[key] = "corrompido"
                self.assertEqual(model.params, pristine)
        lr = LogisticRegressionBaseline()
        self.assertEqual(lr.params, {
            "penalty": "l2",
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 1000,
            "tol": 1e-4,
            "class_weight": None,
        })

    def test_lr_rejects_invalid_and_multiclass_labels(self):
        invalid = ([0, 2], [-1, 1], [0, 1, 2], [True, False], ["0", "1"])
        for labels in invalid:
            with self.subTest(labels=labels), self.assertRaises(ClassicModelError):
                LogisticRegressionBaseline().fit(_view(labels))

    def test_rf_rejects_invalid_and_multiclass_labels(self):
        invalid = ([0, 2], [-1, 1], [0, 1, 2], [True, False], ["0", "1"])
        for labels in invalid:
            with self.subTest(labels=labels), self.assertRaises(ClassicModelError):
                RandomForestBaseline(training_seed=7).fit(_view(labels))

    def test_random_forest_training_seed_is_read_only(self):
        model = RandomForestBaseline(training_seed=7)
        self.assertEqual(model.training_seed, 7)
        self.assertEqual(model.params["random_state"], 7)
        with self.assertRaises(AttributeError):
            model.training_seed = 11
        self.assertEqual(model.training_seed, 7)
        self.assertEqual(model.params["random_state"], 7)

    def test_require_official_training_seed_consistency(self):
        local_rf = RandomForestBaseline(training_seed=7)
        # seed oficial com modelo em seed divergente, e seed não oficial
        with self.assertRaises(ClassicModelError):
            require_official_training_seed(local_rf, 42)
        with self.assertRaises(ClassicModelError):
            require_official_training_seed(local_rf, 7)
        # instanciação pura com seed oficial, sem qualquer fit
        official_rf = RandomForestBaseline(training_seed=42)
        require_official_training_seed(official_rf, 42)
        with self.assertRaises(ClassicModelError):
            require_official_training_seed(official_rf, 43)
        # modelos sem seed própria só dependem da declaração oficial
        require_official_training_seed(MajorityBaseline(), 42)
        require_official_training_seed(LogisticRegressionBaseline(), 42)
        with self.assertRaises(ClassicModelError):
            require_official_training_seed(MajorityBaseline(), 41)
        with self.assertRaises(ClassicModelError):
            require_official_training_seed("não é modelo", 42)


@unittest.skipUnless(HAS_SKLEARN, "scikit-learn indisponível")
class SklearnModelTests(unittest.TestCase):
    """Ajustes reais somente no treino global, com seeds locais (7 e 11)."""

    @classmethod
    def setUpClass(cls):
        cls.data = load_experimental_data(DATASET, DATA_RECORD)
        cls.lr = LogisticRegressionBaseline().fit(cls.data.train)
        cls.rf = RandomForestBaseline(training_seed=7).fit(cls.data.train)
        cls.lr_scores = cls.lr.scores(cls.data.train)
        cls.rf_scores = cls.rf.scores(cls.data.train)

    def test_estimators_receive_frozen_parameters(self):
        for model in (self.lr, self.rf):
            estimator_params = model.estimator_.get_params()
            for key, value in model.params.items():
                self.assertEqual(estimator_params[key], value, key)

    def test_scores_have_format_and_probability_bounds(self):
        for scores in (self.lr_scores, self.rf_scores):
            self.assertEqual(len(scores), len(self.data.train))
            self.assertTrue(all(type(score) is float for score in scores))
            self.assertTrue(all(math.isfinite(score) and 0.0 <= score <= 1.0
                                for score in scores))

    def test_lr_logits_are_coherent_with_scores(self):
        logits = self.lr.logits(self.data.train)
        self.assertEqual(len(logits), len(self.lr_scores))
        for logit, score in zip(logits, self.lr_scores):
            self.assertTrue(math.isfinite(logit))
            # sigmoid estável por tanh: Sigmoid(decision_function) == score
            self.assertAlmostEqual(score, 0.5 * (1.0 + math.tanh(logit / 2.0)),
                                   places=9)

    def test_fit_scope_excludes_validation_and_test_indices(self):
        expected = set(self.data.train.indices)
        for model in (self.lr, self.rf):
            self.assertEqual(set(model.fit_indices), expected)
            self.assertTrue(
                set(model.fit_indices).isdisjoint(self.data.validation.indices))
            self.assertTrue(set(model.fit_indices).isdisjoint(self.data.test.indices))

    def test_frozen_preprocessing_width_is_preserved(self):
        for model in (self.lr, self.rf):
            self.assertEqual(model.estimator_.n_features_in_,
                             len(self.data.output_features))

    def test_threshold_integration_with_evaluate_scores_is_strict(self):
        labels = list(self.data.train.labels)
        logits = self.lr.logits(self.data.train)
        with_logits = evaluate_scores(labels, self.lr_scores, logits=logits)
        from_scores = evaluate_scores(labels, self.lr_scores)
        expected = [1 if score > 0.5 else 0 for score in self.lr_scores]
        self.assertEqual(from_scores["predicted_labels"], expected)
        self.assertEqual(with_logits["predicted_labels"], expected)
        self.assertEqual(with_logits["predicted_labels"],
                         [1 if logit > 0.0 else 0 for logit in logits])
        self.assertIsNotNone(with_logits["values"]["bce_loss"])
        self.assertIsNone(with_logits["undefined_reasons"]["bce_loss"])

    def test_same_seed_is_deterministic_and_distinct_seed_differs(self):
        first = RandomForestBaseline(training_seed=7).fit(self.data.train)
        second = RandomForestBaseline(training_seed=7).fit(self.data.train)
        self.assertEqual(first.scores(self.data.train), second.scores(self.data.train))
        other = RandomForestBaseline(training_seed=11).fit(self.data.train)
        self.assertNotEqual(first.scores(self.data.train),
                            other.scores(self.data.train))
        lr_again = LogisticRegressionBaseline().fit(self.data.train)
        self.assertEqual(lr_again.scores(self.data.train), self.lr_scores)
        self.assertEqual(lr_again.logits(self.data.train),
                         self.lr.logits(self.data.train))

    def test_random_forest_single_class_view_yields_constant_extremes(self):
        negatives = RandomForestBaseline(training_seed=7).fit(_view([0] * 12))
        self.assertEqual(negatives.scores(_view([0, 1])), (0.0, 0.0))
        result = evaluate_scores([0, 1], negatives.scores(_view([0, 1])))
        self.assertIsNone(result["values"]["bce_loss"])
        self.assertIsNotNone(result["undefined_reasons"]["bce_loss"])
        positives = RandomForestBaseline(training_seed=7).fit(_view([1] * 12))
        self.assertEqual(positives.scores(_view([0, 1])), (1.0, 1.0))

    def test_external_param_mutation_does_not_change_fitted_estimator(self):
        lr = LogisticRegressionBaseline()
        lr.params["C"] = 999.0
        lr.params["solver"] = "liblinear"
        lr.fit(self.data.train)
        fitted = lr.estimator_.get_params()
        self.assertEqual(fitted["C"], 1.0)
        self.assertEqual(fitted["solver"], "lbfgs")
        self.assertEqual(fitted["penalty"], "l2")
        self.assertEqual(fitted["max_iter"], 1000)
        self.assertEqual(fitted["tol"], 1e-4)
        self.assertIsNone(fitted["class_weight"])
        rf = RandomForestBaseline(training_seed=7)
        rf.params["n_estimators"] = 1
        rf.params["random_state"] = 999
        rf.fit(self.data.train)
        fitted = rf.estimator_.get_params()
        self.assertEqual(fitted["n_estimators"], 100)
        self.assertEqual(fitted["random_state"], 7)
        self.assertEqual(fitted["max_features"], "sqrt")
        self.assertEqual(fitted["class_weight"], None)
        self.assertEqual(rf.training_seed, fitted["random_state"])

    def test_lr_extreme_probabilities_have_finite_logits(self):
        separable = DatasetView(
            (1, 2, 3, 4, 5, 6),
            ((-3.0, 0.0), (-2.0, 0.0), (-1.0, 0.0),
             (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)),
            (0, 0, 0, 1, 1, 1),
        )
        model = LogisticRegressionBaseline().fit(separable)
        extreme = DatasetView((10, 11), ((-1e9, 0.0), (1e9, 0.0)), (0, 1))
        scores = model.scores(extreme)
        logits = model.logits(extreme)
        self.assertEqual(scores, (0.0, 1.0))
        self.assertTrue(all(math.isfinite(logit) for logit in logits))
        result = evaluate_scores([0, 1], scores, logits=logits)
        self.assertIsNotNone(result["values"]["bce_loss"])
        self.assertIsNone(result["undefined_reasons"]["bce_loss"])
        self.assertEqual(result["predicted_labels"], [0, 1])

    def test_rf_binary_scores_integrate_with_evaluate_scores(self):
        labels = list(self.data.train.labels)
        result = evaluate_scores(labels, self.rf_scores)
        self.assertEqual(result["sample_count"], len(labels))
        self.assertEqual(result["predicted_labels"],
                         [1 if score > 0.5 else 0 for score in self.rf_scores])
        bce = result["values"]["bce_loss"]
        if bce is None:
            self.assertIsNotNone(result["undefined_reasons"]["bce_loss"])
        else:
            self.assertTrue(math.isfinite(bce) and bce >= 0.0)
        counts = result["confusion_matrix"]["counts"]
        self.assertEqual(sum(sum(row) for row in counts), len(labels))


if __name__ == "__main__":
    unittest.main()
