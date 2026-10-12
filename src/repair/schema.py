"""Canonical repair and training dataset schema identifiers."""

LEGACY_REPAIR_SCHEMA_VERSION = "aria-repair-dataset-v4"
PREVIOUS_REPAIR_SCHEMA_VERSION = "aria-repair-dataset-v5"
REPAIR_SCHEMA_VERSION = "aria-repair-dataset-v6"
SUPPORTED_REPAIR_SCHEMA_VERSIONS = frozenset(
    {
        LEGACY_REPAIR_SCHEMA_VERSION,
        PREVIOUS_REPAIR_SCHEMA_VERSION,
        REPAIR_SCHEMA_VERSION,
    }
)

TRAINING_READY_SCHEMA_VERSION = "aria-training-ready-v3"
