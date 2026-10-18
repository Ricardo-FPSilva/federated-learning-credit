"""Preparação determinística e auditável dos dados do protocolo benchmark-v1."""

from research.data.partitioning import iid_partition, partition_digest
from research.data.pipeline import (
    DEFAULT_CONFIG,
    audit_duplicates,
    build_record,
    load_dataset,
    load_record,
    validate_record,
    write_record,
)
from research.data.preprocessing import FeaturePreprocessor
from research.data.splitting import stratified_split

__all__ = [
    "DEFAULT_CONFIG",
    "FeaturePreprocessor",
    "audit_duplicates",
    "build_record",
    "iid_partition",
    "load_dataset",
    "load_record",
    "partition_digest",
    "stratified_split",
    "validate_record",
    "write_record",
]
