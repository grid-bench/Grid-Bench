"""Dataset configuration: family defaults, tiers, benchmark cells, JSON presets."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import prod
from pathlib import Path
from typing import Any

from ..registry import family_ids
from ..schema import SchemaError, parse_json_document
from .shapes import GLOBAL_LEVELS, available_dimensions, resolve_level, shape_for, shape_stats

GENERATOR_VERSION = "0.5.0"
DEFAULT_T_GRID = (0, 1, 2, 4, 8, 16, 32, 64, 128)

# Per-family generation defaults.  ``w_targets`` are per-event scan targets
# for ``w_kind == "scan"`` families, update-support targets for
# ``w_kind == "support"`` (the ball size of STENCIL, whose top level equals
# the tier's nominal N and means the whole world, and the region size of MIX),
# and nominal N levels for the global family (``w_kind == "cells"``).
FAMILY_DEFAULTS: dict[str, dict[str, Any]] = {
    "metric_paint": {
        "values": [0, 1, 2, 3],
        "density": [0.45, 0.65],
        "set_rate": 0.2,
        "noop_rate": 0.1,
        "initializer": "uniform",
        "set_policy": "thermostat",
        "w_kind": "scan",
        "w_targets": {"S": [7, 17, 33], "M": [25, 65, 129], "L": [65, 129, 400]},
        "candidates": 64,
        "tolerance": 0.15,
    },
    "box_reflection": {
        "values": [0, 1, 2, 3],
        "density": [0.45, 0.65],
        "set_rate": 0.2,
        "noop_rate": 0.1,
        "initializer": "uniform",
        "set_policy": "thermostat",
        "w_kind": "scan",
        "w_targets": {"S": [4, 9, 16], "M": [8, 27, 64], "L": [16, 64, 256]},
        "candidates": 32,
        "tolerance": 0.25,
    },
    "local_stencil": {
        "values": [0, 1, 2, 3],
        "density": [0.55, 0.85],
        "set_rate": 0.2,
        "noop_rate": None,
        "initializer": "uniform",
        "set_policy": "thermostat",
        "w_kind": "support",
        "w_targets": {"S": [7, 17, 33, 81], "M": [25, 65, 129, 256], "L": [65, 129, 400, 1024]},
        "candidates": 64,
        "tolerance": 0.15,
    },
    "occluded_mix": {
        "values": [0, 1, 2, 3],
        "density": [0.55, 0.70],
        "set_rate": 0.2,
        "noop_rate": None,
        "initializer": "uniform",
        "set_policy": "thermostat",
        "w_kind": "support",
        "w_targets": {"S": [6, 12, 24], "M": [10, 25, 60], "L": [16, 50, 150]},
        "candidates": 32,
        "tolerance": 0.15,
    },
    "component_relabel": {
        "values": [0, 1, 2, 3],
        "density": [0.50, 0.65],
        "set_rate": 0.2,
        "noop_rate": None,
        "initializer": "regions",
        "set_policy": "distinct",
        "w_kind": "scan",
        "w_targets": {"S": [6, 12, 24], "M": [10, 25, 60], "L": [16, 50, 150]},
        "candidates": 32,
        "tolerance": 0.25,
    },
    "boundary_layer": {
        "values": [0, 1, 2, 3, 4, 5, 6],
        "density": [0.45, 0.60],
        "set_rate": 0.2,
        "noop_rate": 0.1,
        "initializer": "regions",
        "set_policy": "distinct",
        "w_kind": "scan",
        "w_targets": {"S": [6, 12, 24], "M": [10, 25, 60], "L": [16, 50, 150]},
        "candidates": 32,
        "tolerance": 0.25,
    },
    "obstructed_motion": {
        "values": [0, 1, 2, 3, 4],
        "density": [0.35, 0.50],
        "set_rate": 0.3,
        "noop_rate": 0.1,
        "initializer": "uniform",
        "set_policy": "thermostat",
        "w_kind": "scan",
        "w_targets": {"S": [12, 20, 32], "M": [30, 60, 100], "L": [120, 240, 400]},
        "candidates": 48,
        "tolerance": 0.25,
    },
    "aggregating_compression": {
        "values": "pow2:12",
        "density": [0.15, 0.60],
        "set_rate": 0.5,
        "noop_rate": 0.1,
        "initializer": "tiles",
        "set_policy": "spawn",
        "w_kind": "cells",
        "w_targets": {tier: list(levels) for tier, levels in GLOBAL_LEVELS.items()},
        "candidates": 8,
        "tolerance": 0.0,
    },
}

REJECTION_DEFAULTS: dict[str, Any] = {
    "max_attempts": 8,
    "density_slack": 0.4,
    "stencil_min_effect_fraction": 0.1,
    "scan_ratio_band": [0.5, 2.0],
    "effective_floor": {"default": 0.5},
    "max_kind_miss": 0.5,
}

INITIALIZERS = ("uniform", "mixed_blob", "regions", "tiles")
# scan: w is a per-event scan target; support: w is a target update-support
# size (a w at or above the tier's nominal N means the whole world); cells: w
# is a nominal N level that selects the shape.
W_KINDS = ("scan", "support", "cells")
SET_POLICIES = ("thermostat", "distinct", "spawn")
# uniform: query cells drawn without looking at the final state; half-changed:
# a changed-cell-first diagnostic with an answer-related sampling prior
QUERY_POLICIES = ("uniform", "half-changed")


class ConfigError(ValueError):
    """Raised when a dataset configuration is malformed."""


def parse_values(spec: object) -> tuple[int, ...]:
    if isinstance(spec, str):
        if spec.startswith("pow2:"):
            exponent = int(spec.split(":", 1)[1])
            return (0, *[2**k for k in range(exponent + 1)])
        raise ConfigError(f"unknown values specification {spec!r}")
    if isinstance(spec, (list, tuple)) and all(isinstance(v, int) for v in spec):
        return tuple(spec)
    raise ConfigError("values must be an integer array or a 'pow2:K' string")


@dataclass(frozen=True, slots=True)
class RejectionConfig:
    max_attempts: int = 8
    density_slack: float = 0.4
    stencil_min_effect_fraction: float = 0.1
    scan_ratio_band: tuple[float, float] = (0.5, 2.0)
    effective_floor: Mapping[str, float] = field(default_factory=dict)
    max_kind_miss: float = 0.5

    def floor_for(self, family: str) -> float:
        return float(self.effective_floor.get(family, self.effective_floor.get("default", 0.5)))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> RejectionConfig:
        if data is not None and not isinstance(data, Mapping):
            raise ConfigError("rejection must be an object")
        unknown = set(data or {}) - REJECTION_DEFAULTS.keys()
        if unknown:
            raise ConfigError(f"unknown rejection keys: {', '.join(sorted(unknown))}")
        merged = dict(REJECTION_DEFAULTS)
        if data:
            merged.update(data)
        band = merged["scan_ratio_band"]
        return cls(
            max_attempts=int(merged["max_attempts"]),
            density_slack=float(merged["density_slack"]),
            stencil_min_effect_fraction=float(merged["stencil_min_effect_fraction"]),
            scan_ratio_band=(float(band[0]), float(band[1])),
            effective_floor=dict(merged["effective_floor"]),
            max_kind_miss=float(merged["max_kind_miss"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_attempts": self.max_attempts,
            "density_slack": self.density_slack,
            "stencil_min_effect_fraction": self.stencil_min_effect_fraction,
            "scan_ratio_band": list(self.scan_ratio_band),
            "effective_floor": dict(self.effective_floor),
            "max_kind_miss": self.max_kind_miss,
        }


@dataclass(frozen=True, slots=True)
class FamilyConfig:
    family: str
    tier: str
    dims: tuple[int, ...]
    values: tuple[int, ...]
    density: tuple[float, float]
    set_rate: float
    noop_rate: float | None
    initializer: str
    set_policy: str
    w_kind: str
    w_targets: tuple[int, ...]
    candidates: int
    tolerance: float

    def __post_init__(self) -> None:
        if self.family not in FAMILY_DEFAULTS:
            raise ConfigError(f"unknown family {self.family!r}")
        if self.tier not in ("S", "M", "L"):
            raise ConfigError("tier must be S, M, or L")
        if not self.dims or any(d < 1 for d in self.dims):
            raise ConfigError("dims must list positive dimensions")
        lo, hi = self.density
        if not 0 < lo <= hi < 1:
            raise ConfigError("density band must satisfy 0 < lo <= hi < 1")
        if not 0 <= self.set_rate < 1:
            raise ConfigError("set_rate must lie in [0, 1)")
        if self.noop_rate is not None and not 0 <= self.noop_rate < 1:
            raise ConfigError("noop_rate must lie in [0, 1) or be null")
        if self.initializer not in INITIALIZERS:
            raise ConfigError(f"initializer must be one of {INITIALIZERS}")
        if self.set_policy not in SET_POLICIES:
            raise ConfigError(f"set_policy must be one of {SET_POLICIES}")
        if self.w_kind not in W_KINDS:
            raise ConfigError(f"w_kind must be one of {W_KINDS}")
        if not self.w_targets or any(w <= 0 for w in self.w_targets):
            raise ConfigError("w_targets must list positive integers")
        if self.candidates < 1:
            raise ConfigError("candidates must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "tier": self.tier,
            "dims": list(self.dims),
            "values": list(self.values),
            "density": list(self.density),
            "set_rate": self.set_rate,
            "noop_rate": self.noop_rate,
            "initializer": self.initializer,
            "set_policy": self.set_policy,
            "w_kind": self.w_kind,
            "w_targets": list(self.w_targets),
            "candidates": self.candidates,
            "tolerance": self.tolerance,
        }


@dataclass(frozen=True, slots=True)
class CellSpec:
    """One benchmark cell: a family at a fixed dimension, shape, and nominal workload."""

    family: str
    tier: str
    dimension: int
    shape: tuple[int, ...]
    w: int
    w_kind: str
    values: tuple[int, ...]
    density: tuple[float, float]
    set_rate: float
    noop_rate: float | None
    initializer: str
    set_policy: str
    candidates: int
    tolerance: float
    t_max: int
    t_grid: tuple[int, ...]
    level: str

    @property
    def cells(self) -> int:
        return prod(self.shape)

    @property
    def nominal_cells(self) -> int:
        """The tier's nominal N (the resolved level), e.g. 256 for the M tier."""

        return int(self.level)

    @property
    def whole_world(self) -> bool:
        """True for a STENCIL cell whose support level means the entire world.

        Only the ball family has a whole-world level; a MIX support target is
        always a region size and is infeasible above the cell count.
        """

        return (
            self.family == "local_stencil"
            and self.w_kind == "support"
            and self.w >= self.nominal_cells
        )

    @property
    def key(self) -> str:
        return f"{self.family}/{self.tier}/d{self.dimension}/w{self.w}"

    def rollout_id(self, split: str, index: int) -> str:
        return f"{self.key}/{split}/{index:04d}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "tier": self.tier,
            "dimension": self.dimension,
            "shape": list(self.shape),
            "w": self.w,
            "w_kind": self.w_kind,
            "values": list(self.values),
            "density": list(self.density),
            "set_rate": self.set_rate,
            "noop_rate": self.noop_rate,
            "initializer": self.initializer,
            "set_policy": self.set_policy,
            "candidates": self.candidates,
            "tolerance": self.tolerance,
            "t_max": self.t_max,
            "t_grid": list(self.t_grid),
            "level": self.level,
        }


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    name: str
    dataset_seed: int
    split: str
    t_grid: tuple[int, ...]
    t_max: int
    seeds_per_cell: int
    queries: int
    families: tuple[FamilyConfig, ...]
    rejection: RejectionConfig
    query_policy: str = "uniform"
    source: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ConfigError("name must be a non-empty string")
        if self.query_policy not in QUERY_POLICIES:
            raise ConfigError(f"query_policy must be one of {QUERY_POLICIES}")
        if not self.split or "/" in self.split:
            raise ConfigError("split must be a non-empty token without '/'")
        if self.t_max < 1:
            raise ConfigError("t_max must be positive")
        if any(t < 0 or t > self.t_max for t in self.t_grid):
            raise ConfigError("every t_grid entry must lie in [0, t_max]")
        if len(set(self.t_grid)) != len(self.t_grid) or list(self.t_grid) != sorted(self.t_grid):
            raise ConfigError("t_grid must be strictly increasing")
        if self.seeds_per_cell < 1:
            raise ConfigError("seeds_per_cell must be positive")
        if self.queries < 0:
            raise ConfigError("queries must be nonnegative")
        if not self.families:
            raise ConfigError("at least one family is required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dataset_seed": self.dataset_seed,
            "split": self.split,
            "t_grid": list(self.t_grid),
            "t_max": self.t_max,
            "seeds_per_cell": self.seeds_per_cell,
            "queries": self.queries,
            "query_policy": self.query_policy,
            "families": [family.to_dict() for family in self.families],
            "rejection": self.rejection.to_dict(),
        }


def _family_config(
    family: str, tier: str, dims: Sequence[int], overrides: Mapping[str, Any]
) -> FamilyConfig:
    try:
        defaults = FAMILY_DEFAULTS[family]
    except KeyError as error:
        raise ConfigError(f"unknown family {family!r}") from error
    if not isinstance(overrides, Mapping):
        raise ConfigError(f"overrides for family {family!r} must be an object")
    allowed = set(defaults) | {"tier", "dims"}
    unknown = set(overrides) - allowed
    if unknown:
        raise ConfigError(
            f"unknown override keys for family {family!r}: {', '.join(sorted(unknown))}"
        )
    merged: dict[str, Any] = dict(defaults)
    merged.update({k: v for k, v in overrides.items() if k not in ("w_targets",)})
    tier = str(merged.get("tier", tier))
    dims_value = merged.get("dims", dims)
    w_targets_spec = overrides.get("w_targets", defaults["w_targets"])
    if isinstance(w_targets_spec, Mapping):
        w_targets = w_targets_spec[tier]
    else:
        w_targets = w_targets_spec
    density = merged["density"]
    return FamilyConfig(
        family=family,
        tier=tier,
        dims=tuple(int(d) for d in dims_value),
        values=parse_values(merged["values"]),
        density=(float(density[0]), float(density[1])),
        set_rate=float(merged["set_rate"]),
        noop_rate=None if merged["noop_rate"] is None else float(merged["noop_rate"]),
        initializer=str(merged["initializer"]),
        set_policy=str(merged["set_policy"]),
        w_kind=str(merged["w_kind"]),
        w_targets=tuple(int(w) for w in w_targets),
        candidates=int(merged["candidates"]),
        tolerance=float(merged["tolerance"]),
    )


def config_from_dict(data: Mapping[str, Any]) -> DatasetConfig:
    if not isinstance(data, Mapping):
        raise ConfigError("dataset config must be a JSON object")
    known = {
        "name",
        "dataset_seed",
        "split",
        "tier",
        "dims",
        "t_grid",
        "t_max",
        "seeds_per_cell",
        "queries",
        "query_policy",
        "families",
        "rejection",
        "description",
    }
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"unknown config keys: {', '.join(sorted(unknown))}")
    tier = str(data.get("tier", "S"))
    if tier not in ("S", "M", "L"):
        raise ConfigError("tier must be S, M, or L")
    dims = data.get("dims", available_dimensions(tier))
    families_spec = data.get("families", sorted(family_ids()))
    if isinstance(families_spec, Mapping):
        family_items = [(name, spec or {}) for name, spec in families_spec.items()]
    elif isinstance(families_spec, (list, tuple)):
        family_items = [(str(name), {}) for name in families_spec]
    else:
        raise ConfigError("families must be an array of ids or an object of overrides")
    families = tuple(_family_config(name, tier, dims, spec) for name, spec in family_items)
    t_max = int(data.get("t_max", max(DEFAULT_T_GRID)))
    t_grid = tuple(int(t) for t in data.get("t_grid", [t for t in DEFAULT_T_GRID if t <= t_max]))
    return DatasetConfig(
        name=str(data.get("name", "dataset")),
        dataset_seed=int(data.get("dataset_seed", 0)),
        split=str(data.get("split", "dev")),
        t_grid=t_grid,
        t_max=t_max,
        seeds_per_cell=int(data.get("seeds_per_cell", 1)),
        queries=int(data.get("queries", 0)),
        families=families,
        rejection=RejectionConfig.from_dict(data.get("rejection")),
        query_policy=str(data.get("query_policy", "uniform")),
        source=dict(data),
    )


def load_config(path: str | Path) -> DatasetConfig:
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = parse_json_document(text)
    except (json.JSONDecodeError, SchemaError) as error:
        raise ConfigError(f"cannot parse {path}: {error}") from error
    return config_from_dict(data)


def expand_cells(config: DatasetConfig) -> list[CellSpec]:
    """Every (family, dimension, w) cell implied by the configuration."""

    cells: list[CellSpec] = []
    for family in config.families:
        for dimension in family.dims:
            if family.w_kind == "cells":
                for level in family.w_targets:
                    try:
                        shape = shape_for(level, dimension)
                    except KeyError:
                        continue
                    cells.append(_cell(config, family, dimension, shape, level, str(level)))
            else:
                try:
                    shape = shape_for(family.tier, dimension)
                except KeyError:
                    continue
                for w in family.w_targets:
                    cells.append(
                        _cell(config, family, dimension, shape, w, resolve_level(family.tier))
                    )
    return cells


def _cell(
    config: DatasetConfig,
    family: FamilyConfig,
    dimension: int,
    shape: tuple[int, ...],
    w: int,
    level: str,
) -> CellSpec:
    return CellSpec(
        family=family.family,
        tier=family.tier,
        dimension=dimension,
        shape=shape,
        w=w,
        w_kind=family.w_kind,
        values=family.values,
        density=family.density,
        set_rate=family.set_rate,
        noop_rate=family.noop_rate,
        initializer=family.initializer,
        set_policy=family.set_policy,
        candidates=family.candidates,
        tolerance=family.tolerance,
        t_max=config.t_max,
        t_grid=config.t_grid,
        level=level,
    )


def infeasibility_reason(cell: CellSpec) -> str | None:
    """A structural reason why the cell's workload target cannot be met, if any."""

    stats = shape_stats(cell.shape)
    if cell.w_kind == "cells" or cell.whole_world:
        return None
    if cell.family == "box_reflection" and cell.w > stats["cells"]:
        return "REFLECT box volume exceeds the world"
    if cell.family in ("metric_paint", "local_stencil"):
        from ..grid import Grid
        from ..schema import WorldSpec

        grid = Grid(WorldSpec(cell.shape, cell.values))
        within = 0
        for index in range(grid.size):
            center = grid.coord(index)
            radius = 0
            while radius <= sum(cell.shape):
                size = len(grid.l1_ball(center, radius))
                if abs(size - cell.w) <= max(1.0, cell.tolerance * cell.w):
                    within += 1
                    break
                if size >= grid.size:
                    break
                radius += 1
        if within < 0.1 * grid.size:
            op = "PAINT" if cell.family == "metric_paint" else "STENCIL"
            return f"{op} ball target {cell.w} reachable from {within}/{grid.size} centers"
    if cell.family in ("component_relabel", "boundary_layer"):
        if cell.w > stats["cells"] * 1.5:
            return "component target exceeds the world"
    if cell.family == "occluded_mix" and cell.w > stats["cells"]:
        return "MIX support target exceeds the world"
    if cell.family == "occluded_mix" and cell.dimension == 1 and cell.initializer == "uniform":
        # A one-dimensional region cannot leave its run of nonzero cells, whose
        # expected length under uniform occupancy is 1 / (1 - density).
        run = 1.0 / max(1e-9, 1.0 - cell.density[1])
        if cell.w > 2 * run + 1:
            return (
                f"MIX support in one dimension is bounded by the run length "
                f"(about {run:.0f} cells at density {cell.density[1]:.2f})"
            )
    return None


PRESETS: dict[str, dict[str, Any]] = {
    "smoke": {
        "name": "smoke",
        "dataset_seed": 20260903,
        "split": "dev",
        "tier": "S",
        "dims": [1, 2, 3],
        "t_grid": [0, 1, 2, 4, 8, 16, 32],
        "t_max": 32,
        "seeds_per_cell": 3,
    },
}
