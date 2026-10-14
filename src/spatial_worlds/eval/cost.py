"""Observed request cost and cost projection from a calibration basis.

Prices are read from a JSON table (``configs/prices.json``); a record's cost
is its observed usage times the model's price.  A projection takes a
selection of instances, a scored calibration run (``results.jsonl``) as the
usage basis, and a price, and multiplies the mean observed usage of matching
(family, T) rows by the price.  Everything a projection assumes (the basis
model, an output-token multiplier for another model, fallbacks when a
(family, T) pair has no basis rows) is reported next to the number.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..schema import Instance, parse_json_document
from .run import iter_jsonl, load_dataset, matches_filter, parse_filter


@dataclass(frozen=True, slots=True)
class Price:
    """Price of a model in USD per million input and output tokens."""

    input_per_million: float
    output_per_million: float

    def cost(self, input_tokens: float, output_tokens: float) -> float:
        return (
            input_tokens * self.input_per_million + output_tokens * self.output_per_million
        ) / 1_000_000.0


def load_prices(path: Path) -> dict[str, Price]:
    """Read ``{"usd_per_million": {model: {"input": x, "output": y, ...}}}``."""

    document = parse_json_document(path.read_text(encoding="utf-8"))
    table = document.get("usd_per_million") if isinstance(document, Mapping) else None
    if not isinstance(table, Mapping):
        raise ValueError(f"{path} must contain a usd_per_million object")
    prices: dict[str, Price] = {}
    for model, entry in table.items():
        if (
            not isinstance(entry, Mapping)
            or not isinstance(entry.get("input"), (int, float))
            or not isinstance(entry.get("output"), (int, float))
        ):
            raise ValueError(f"{path}: price of {model!r} needs numeric input and output")
        prices[str(model)] = Price(float(entry["input"]), float(entry["output"]))
    return prices


def record_cost_usd(usage: Mapping[str, Any] | None, price: Price | None) -> float | None:
    """Cost of one request from its observed usage; None without usage or a price.

    Cached input tokens are charged at the full input price, so this is an
    upper bound where the gateway discounts cache hits.  Attempts that
    returned nothing report no usage and therefore no cost.
    """

    if price is None or not usage:
        return None
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if input_tokens is None or output_tokens is None:
        return None
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        for value in (input_tokens, output_tokens)
    ):
        return None
    return price.cost(float(input_tokens), float(output_tokens))


@dataclass(slots=True)
class UsageStat:
    """Mean observed usage of a group of scored rows."""

    n: int
    mean_input_tokens: float
    mean_output_tokens: float


@dataclass(slots=True)
class Basis:
    """Mean observed usage of one model per (family, T), per T, and overall."""

    model: str
    by_family_T: dict[tuple[str, int], UsageStat]
    by_T: dict[int, UsageStat]
    overall: UsageStat | None

    def lookup(self, family: str, horizon: int) -> tuple[UsageStat | None, str]:
        """The most specific usage available and the name of the level used."""

        stat = self.by_family_T.get((family, horizon))
        if stat is not None:
            return stat, "family_T"
        stat = self.by_T.get(horizon)
        if stat is not None:
            return stat, "T"
        if self.overall is not None:
            return self.overall, "overall"
        return None, "none"


def _stat(rows: Sequence[Mapping[str, Any]]) -> UsageStat | None:
    inputs = [float(r["usage"]["input_tokens"] or 0) for r in rows]
    outputs = [float(r["usage"]["output_tokens"] or 0) for r in rows]
    if not rows:
        return None
    return UsageStat(len(rows), sum(inputs) / len(rows), sum(outputs) / len(rows))


def build_basis(rows: Iterable[Mapping[str, Any]], model: str) -> Basis:
    """Usage basis from scored rows (``results.jsonl``) of ``model``.

    Every row with observed usage counts, including failed requests that
    reported their usage (they were billed too); rows without usage (a stall
    that never returned) are excluded, so the basis understates a condition
    with many stalls.
    """

    usable = [
        r
        for r in rows
        if str(r.get("model")) == model
        and r.get("usage")
        and (
            r["usage"].get("input_tokens") is not None
            or r["usage"].get("output_tokens") is not None
        )
    ]
    by_family_T: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    by_T: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for r in usable:
        by_family_T[(str(r["family"]), int(r["T"]))].append(r)
        by_T[int(r["T"])].append(r)
    return Basis(
        model=model,
        by_family_T={k: s for k, v in by_family_T.items() if (s := _stat(v)) is not None},
        by_T={k: s for k, v in by_T.items() if (s := _stat(v)) is not None},
        overall=_stat(usable),
    )


def project_cost(
    instances: Iterable[Instance],
    basis: Basis,
    price: Price,
    *,
    model: str,
    multiplier: float = 1.0,
) -> dict[str, Any]:
    """Projected cost of ``instances`` for ``model`` from the basis usage.

    ``multiplier`` scales the basis output tokens (a model that reasons
    longer than the basis model); input tokens depend on the prompt only and
    are not scaled.  The result carries the number of instances resolved at
    each basis level so that a projection built mostly on fallbacks is
    visible as such.
    """

    total = 0.0
    count = 0
    by_T: dict[int, dict[str, float]] = defaultdict(lambda: {"instances": 0, "cost_usd": 0.0})
    by_family: dict[str, dict[str, float]] = defaultdict(lambda: {"instances": 0, "cost_usd": 0.0})
    levels: dict[str, int] = {"family_T": 0, "T": 0, "overall": 0, "none": 0}
    for instance in instances:
        count += 1
        stat, level = basis.lookup(instance.family, instance.horizon)
        levels[level] += 1
        if stat is None:
            continue
        cost = price.cost(stat.mean_input_tokens, stat.mean_output_tokens * multiplier)
        total += cost
        by_T[instance.horizon]["instances"] += 1
        by_T[instance.horizon]["cost_usd"] += cost
        by_family[instance.family]["instances"] += 1
        by_family[instance.family]["cost_usd"] += cost
    return {
        "model": model,
        "basis_model": basis.model,
        "multiplier": multiplier,
        "price_per_million": {
            "input": price.input_per_million,
            "output": price.output_per_million,
        },
        "instances": count,
        "cost_usd": total,
        "by_T": {t: dict(v) for t, v in sorted(by_T.items())},
        "by_family": {f: dict(v) for f, v in sorted(by_family.items())},
        "basis_levels": levels,
    }


def select_instances(instances: Sequence[Instance], filters: Sequence[str]) -> list[Instance]:
    """Union of the selections of several ``--filter`` expressions (no filter: all)."""

    if not filters:
        return list(instances)
    parsed = [parse_filter(text) for text in filters]
    return [
        instance for instance in instances if any(matches_filter(instance, item) for item in parsed)
    ]


def format_projection(projection: Mapping[str, Any], label: str | None = None) -> str:
    """One line per projection: total, then the breakdown by horizon."""

    levels = projection["basis_levels"]
    fallback = ", ".join(f"{k} {v}" for k, v in levels.items() if v and k != "family_T")
    head = f"{label + ': ' if label else ''}{projection['model']}"
    if projection["multiplier"] != 1.0:
        head += f" (output x{projection['multiplier']:g})"
    per_t = " ".join(
        f"T={t}:${v['cost_usd']:.0f}/{int(v['instances'])}" for t, v in projection["by_T"].items()
    )
    return (
        f"{head}: {projection['instances']} instances, ${projection['cost_usd']:.0f} "
        f"[basis {projection['basis_model']}"
        + (f"; fallbacks: {fallback}" if fallback else "")
        + f"] {per_t}"
    )


def cost_from_args(args: Any) -> int:
    dataset = load_dataset(args.data)
    prices = load_prices(args.prices)
    price = prices.get(args.model)
    if price is None:
        raise ValueError(f"no price for {args.model!r} in {args.prices}")
    basis_model = args.basis_model or args.model
    basis = build_basis(iter_jsonl(args.basis), basis_model)
    if basis.overall is None:
        raise ValueError(f"{args.basis} has no rows with usage for model {basis_model!r}")
    selected = select_instances(dataset.instances, list(args.filter))
    projection = project_cost(
        selected, basis, price, model=args.model, multiplier=float(args.multiplier)
    )
    if args.label:
        projection["label"] = args.label
    if args.json:
        print(json.dumps(projection, indent=2))
    else:
        print(format_projection(projection, args.label))
    if projection["basis_levels"]["none"]:
        print(
            f"warning: {projection['basis_levels']['none']} selected instance(s) have no basis",
            file=sys.stderr,
        )
    return 0
