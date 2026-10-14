"""The eight benchmark family declarations."""

from .aggregating_compression import FAMILY as AGGREGATING_COMPRESSION
from .boundary_layer import FAMILY as BOUNDARY_LAYER
from .box_reflection import FAMILY as BOX_REFLECTION
from .component_relabel import FAMILY as COMPONENT_RELABEL
from .local_stencil import FAMILY as LOCAL_STENCIL
from .metric_paint import FAMILY as METRIC_PAINT
from .obstructed_motion import FAMILY as OBSTRUCTED_MOTION
from .occluded_mix import FAMILY as OCCLUDED_MIX

ALL_FAMILIES = (
    METRIC_PAINT,
    BOX_REFLECTION,
    LOCAL_STENCIL,
    OCCLUDED_MIX,
    COMPONENT_RELABEL,
    BOUNDARY_LAYER,
    OBSTRUCTED_MOTION,
    AGGREGATING_COMPRESSION,
)

__all__ = [
    "AGGREGATING_COMPRESSION",
    "ALL_FAMILIES",
    "BOUNDARY_LAYER",
    "BOX_REFLECTION",
    "COMPONENT_RELABEL",
    "LOCAL_STENCIL",
    "METRIC_PAINT",
    "OBSTRUCTED_MOTION",
    "OCCLUDED_MIX",
]
