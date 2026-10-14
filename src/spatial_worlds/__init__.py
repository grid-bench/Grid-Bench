"""Public contracts for the Grid-Bench benchmark."""

from .family import (
    DraftFamily,
    Family,
    FamilyMetadata,
    FamilyProtocol,
    StepResult,
    Transition,
)
from .generation import (
    GENERATOR_VERSION,
    DatasetConfig,
    build_dataset,
    generate_rollout,
    load_config,
)
from .grid import Grid
from .prompt import PROMPT_VERSION, render_answer, render_prompt
from .registry import family_ids, get_family, list_families, validate_instance
from .schema import SCHEMA_VERSION, Event, GoldRecord, Instance, SchemaError, WorldSpec
from .scoring import ScoreResult, score_prediction, score_query_prediction
from .simulate import SimulationResult, StepTrace, gold_record, simulate
from .workload import TransitionWorkload
from .world import WorldState

__all__ = [
    "DraftFamily",
    "DatasetConfig",
    "Event",
    "Family",
    "FamilyMetadata",
    "FamilyProtocol",
    "GENERATOR_VERSION",
    "GoldRecord",
    "Grid",
    "Instance",
    "PROMPT_VERSION",
    "SCHEMA_VERSION",
    "SchemaError",
    "ScoreResult",
    "SimulationResult",
    "StepResult",
    "StepTrace",
    "Transition",
    "TransitionWorkload",
    "WorldSpec",
    "WorldState",
    "build_dataset",
    "family_ids",
    "generate_rollout",
    "get_family",
    "gold_record",
    "list_families",
    "load_config",
    "render_answer",
    "render_prompt",
    "score_prediction",
    "score_query_prediction",
    "simulate",
    "validate_instance",
]
