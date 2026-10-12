"""Paper-aligned ARIA post-training stage builders."""

from aria.post_training.dpo import build_repair_preference_row
from aria.post_training.sft import build_sft_rows
from aria.post_training.transfer_diagnosis import (
    DiagnosisChoice,
    build_transfer_diagnosis_rows,
    select_diagnosis_choice,
)

__all__ = [
    "DiagnosisChoice",
    "build_repair_preference_row",
    "build_sft_rows",
    "build_transfer_diagnosis_rows",
    "select_diagnosis_choice",
]
