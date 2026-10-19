"""Infraestrutura determinística do benchmark, sem implementação de modelos."""

from research.benchmark.data import DatasetView, ExperimentalData, load_experimental_data
from research.benchmark.evaluation import evaluate_scores, evaluate_global, weighted_batch_loss
from research.benchmark.provenance import artifact_id, build_provenance, file_reference
from research.benchmark.seeding import (
    canonical_json_bytes,
    configure_determinism,
    derive_seed,
    paired_initialization_seed,
    shuffle_seed,
)

__all__ = [
    "DatasetView",
    "ExperimentalData",
    "artifact_id",
    "build_provenance",
    "canonical_json_bytes",
    "configure_determinism",
    "derive_seed",
    "evaluate_global",
    "evaluate_scores",
    "file_reference",
    "load_experimental_data",
    "paired_initialization_seed",
    "shuffle_seed",
    "weighted_batch_loss",
]
