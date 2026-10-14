"""Procedural generation of benchmark datasets."""

from .config import (
    FAMILY_DEFAULTS,
    GENERATOR_VERSION,
    PRESETS,
    QUERY_POLICIES,
    CellSpec,
    ConfigError,
    DatasetConfig,
    FamilyConfig,
    RejectionConfig,
    config_from_dict,
    expand_cells,
    infeasibility_reason,
    load_config,
)
from .dataset import build_dataset, generate_cell, read_jsonl, sample_queries, select_cells
from .rollout import GenerationError, Rollout, RolloutStep, generate_rollout
from .schedule import core_count, effective_schedule, is_set, set_count, set_schedule
from .seeds import derive_seed
from .shapes import GLOBAL_LEVELS, LEVEL_TABLES, TIER_ALIASES, shape_for, shape_stats

__all__ = [
    "FAMILY_DEFAULTS",
    "GENERATOR_VERSION",
    "GLOBAL_LEVELS",
    "LEVEL_TABLES",
    "PRESETS",
    "QUERY_POLICIES",
    "TIER_ALIASES",
    "CellSpec",
    "ConfigError",
    "DatasetConfig",
    "FamilyConfig",
    "GenerationError",
    "RejectionConfig",
    "Rollout",
    "RolloutStep",
    "build_dataset",
    "config_from_dict",
    "core_count",
    "derive_seed",
    "effective_schedule",
    "expand_cells",
    "generate_cell",
    "generate_rollout",
    "infeasibility_reason",
    "is_set",
    "load_config",
    "read_jsonl",
    "sample_queries",
    "select_cells",
    "set_count",
    "set_schedule",
    "shape_for",
    "shape_stats",
]
