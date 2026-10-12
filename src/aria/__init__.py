"""ARIA structured repair training and deployment primitives."""

from aria.deployment import (
    ActiveRepairController,
    PassiveRepairService,
    load_active_hook,
    load_active_repair_controller,
    load_passive_repair_service,
)
from aria.hook import (
    ACTIVE_NATIVE_MAS,
    ActiveHookBridge,
    AriaHook,
    HookRegistration,
    MasHookCapabilities,
    MasHookProfile,
    get_mas_hook_profile,
)
from aria.model import AriaModelConfig, GatePrediction, load_aria_model
from aria.records import (
    FailureLocalization,
    IncidentRecord,
    RepairAttribution,
    RepairRecord,
    TrajectoryEvent,
)
from aria.training import (
    ArtifactOnlyTrainingBackend,
    CommandTrainingBackend,
    PostTrainingPlan,
    TrainingStage,
    run_post_training,
    run_post_training_from_spec,
)

__all__ = [
    "ActiveHookBridge",
    "ACTIVE_NATIVE_MAS",
    "ActiveRepairController",
    "AriaHook",
    "HookRegistration",
    "AriaModelConfig",
    "ArtifactOnlyTrainingBackend",
    "CommandTrainingBackend",
    "GatePrediction",
    "FailureLocalization",
    "IncidentRecord",
    "MasHookCapabilities",
    "MasHookProfile",
    "PassiveRepairService",
    "PostTrainingPlan",
    "RepairAttribution",
    "RepairRecord",
    "TrainingStage",
    "TrajectoryEvent",
    "get_mas_hook_profile",
    "load_active_repair_controller",
    "load_active_hook",
    "load_aria_model",
    "load_passive_repair_service",
    "run_post_training",
    "run_post_training_from_spec",
]
