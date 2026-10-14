"""Manuscript-facing identities for the completed eight-trajectory analysis.

The original four-trajectory registry and snapshot remain immutable historical
artifacts.  This module binds current paper generators to the disjoint union of
the original cohort and the predeclared expansion cohort.
"""

from spatial_worlds.eval.analysis_registry import COLORS as COLORS
from spatial_worlds.eval.analysis_registry import LENGTHS as LENGTHS
from spatial_worlds.eval.analysis_registry import MARKERS as MARKERS
from spatial_worlds.eval.analysis_registry import MODELS as MODELS
from spatial_worlds.eval.analysis_registry import TASKS as TASKS

ANALYSIS_ID = "completed-six-model-expand08-20260923"
ANALYSIS_DIR = "data/analysis/six-model-expand08-20260923"

MAIN_TRAJECTORIES = 192
ALL_TRAJECTORIES = 248
PER_MODEL_INSTANCES = 1488
WORKLOAD_INSTANCES = 528
TASK_TRAJECTORIES = 24
DIMENSION_TRAJECTORIES = 64
CELL_TRAJECTORIES = 8

# Count checks bind manuscript-facing generators to the reviewed expanded evidence.
MAIN_CORRECT = dict(
    zip(
        MODELS,
        (
            (192, 183, 145, 87, 57, 25),
            (192, 137, 91, 61, 35, 14),
            (191, 69, 18, 5, 2, 1),
            (191, 96, 17, 8, 3, 0),
            (192, 148, 33, 18, 6, 0),
            (188, 150, 31, 11, 5, 1),
        ),
        strict=True,
    )
)
T1_PARTITIONS = dict(
    zip(
        MODELS,
        (
            (1, 4, 4),
            (30, 10, 14),
            (23, 41, 56),
            (24, 32, 39),
            (15, 11, 17),
            (9, 11, 20),
        ),
        strict=True,
    )
)
PREFIX_REVERSALS = dict(zip(MODELS, (20, 51, 13, 5, 15, 9), strict=True))
POSITIVE_PREFIX_REVERSALS = dict(zip(MODELS, (20, 51, 12, 4, 15, 6), strict=True))
