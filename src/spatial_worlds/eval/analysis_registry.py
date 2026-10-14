"""Shared identities for the completed six-model offline analysis.

This module configures derived evidence only; formal campaign manifests remain
the authority for the original provider requests and collected outcomes.
"""

ANALYSIS_ID = "completed-six-model-20260920"
ANALYSIS_DIR = "data/analysis/six-model-20260920"
CAMPAIGN_CONFIGS = ("formal.json", "formal_sonnet.json", "formal_aws.json", "formal_minimax.json")
MODELS = (
    "gpt-5.6-sol",
    "claude-sonnet",
    "gpt-5.4-mini",
    "zai.glm-5",
    "moonshotai.kimi-k2.5",
    "minimax.minimax-m2.5",
)
NAMES = dict(zip(MODELS, ("Sol", "Sonnet", "Mini", "GLM-5", "Kimi", "MiniMax"), strict=True))
COLORS = dict(
    zip(
        MODELS,
        ("#245C99", "#C77C30", "#3B8D88", "#8C63A8", "#B84A62", "#687A32"),
        strict=True,
    )
)
MARKERS = dict(zip(MODELS, ("o", "s", "^", "D", "P", "X"), strict=True))
LENGTHS = (0, 1, 8, 16, 32, 64)
TASKS = {
    "metric_paint": "PAINT",
    "box_reflection": "REFLECT",
    "local_stencil": "STENCIL",
    "occluded_mix": "MIX",
    "component_relabel": "RELABEL",
    "boundary_layer": "LAYER",
    "obstructed_motion": "MOVE",
    "aggregating_compression": "COMPRESS",
}

# Count checks bind manuscript-facing generators to the reviewed frozen evidence.
MAIN_CORRECT = dict(
    zip(
        MODELS,
        (
            (96, 95, 77, 43, 27, 13),
            (96, 73, 46, 33, 16, 8),
            (96, 37, 8, 3, 1, 1),
            (95, 48, 9, 1, 0, 0),
            (96, 75, 14, 7, 2, 0),
            (94, 78, 14, 5, 1, 1),
        ),
        strict=True,
    )
)
T1_PARTITIONS = dict(
    zip(
        MODELS,
        (
            (0, 1, 0),
            (15, 5, 3),
            (10, 25, 23),
            (13, 18, 17),
            (7, 6, 8),
            (6, 3, 8),
        ),
        strict=True,
    )
)
PREFIX_REVERSALS = dict(zip(MODELS, (10, 26, 3, 2, 8, 5), strict=True))
POSITIVE_PREFIX_REVERSALS = dict(zip(MODELS, (10, 26, 3, 1, 8, 3), strict=True))
