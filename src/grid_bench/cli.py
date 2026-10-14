"""Command-line interface for the Grid-Bench benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .prompt import REPRESENTATIONS, render_prompt
from .registry import list_families
from .schema import Event, Instance, SchemaError, WorldSpec, parse_json_document
from .simulate import simulate
from .world import WorldState


def _read_json(path: Path) -> Any:
    return parse_json_document(path.read_text(encoding="utf-8"))


def _int_list(text: str) -> list[int]:
    return [int(item) for item in text.split(",") if item.strip()]


def _str_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _index_range(text: str) -> list[int]:
    if ":" in text:
        start, end = text.split(":", 1)
        return list(range(int(start or 0), int(end)))
    return _int_list(text)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grid-bench",
        description="Simulators, generators, and evaluation for the Grid-Bench benchmark.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    families = subparsers.add_parser("families", help="list the registered families")
    families.add_argument("--json", action="store_true", help="print metadata as JSON")

    validate = subparsers.add_parser("validate-instance", help="validate an instance JSON document")
    validate.add_argument("file", type=Path)

    render = subparsers.add_parser("render-prompt", help="render the model prompt for an instance")
    render.add_argument("file", type=Path)
    render.add_argument("--representation", choices=REPRESENTATIONS, default="cells")

    sim = subparsers.add_parser("simulate", help="simulate an instance and print its gold record")
    sim.add_argument("file", type=Path)
    sim.add_argument("--gold", type=Path, help="write the gold record to this file")
    sim.add_argument("--trace", type=Path, help="write the per-step trace to this file")

    generate = subparsers.add_parser("generate", help="generate a dataset from a config or preset")
    generate.add_argument("--config", required=True, help="config JSON path or preset name")
    generate.add_argument("--out", required=True, type=Path)
    generate.add_argument("--families", type=_str_list, help="comma-separated family ids")
    generate.add_argument("--dims", type=_int_list, help="comma-separated dimensions")
    generate.add_argument("--seeds", type=_index_range, help="rollout index range START:END")
    generate.add_argument("--workers", type=int, default=1)
    generate.add_argument(
        "--full-trace", action="store_true", help="store every intermediate state"
    )
    generate.add_argument("--queries", type=int, help="override the number of query coordinates")
    generate.add_argument(
        "--query-policy",
        choices=("uniform", "half-changed"),
        help="how query coordinates are drawn (default uniform)",
    )
    generate.add_argument("--split", help="override the split token")
    generate.add_argument("--name", help="override the dataset name")
    generate.add_argument("--quiet", action="store_true")

    evaluate = subparsers.add_parser("evaluate", help="evaluate responses on a benchmark view")
    evaluate.add_argument("--data", type=Path, required=True,
                          help="Hugging Face checkout, benchmark JSONL, or generated dataset")
    evaluate.add_argument("--predictions", type=Path, required=True,
                          help="JSONL objects with public_id and response (one model per file)")
    evaluate.add_argument("--out", type=Path, required=True, help="evaluation output directory")
    evaluate.add_argument("--view", choices=("main", "workload", "all"), default="main")
    evaluate.add_argument("--verify-references", action="store_true",
                          help="replay references in addition to checking prompts and hashes")
    subparsers.add_parser("demo", help="run a small example locally, without a model API")

    return parser


def _cmd_families(args: argparse.Namespace) -> int:
    if args.json:
        print(json.dumps([family.metadata.to_dict() for family in list_families()], indent=2))
        return 0
    for family in list_families():
        metadata = family.metadata
        print(
            f"{metadata.id}\t{metadata.status}\t{metadata.support_source}\t"
            f"{metadata.update_type}\t{metadata.name}"
        )
    return 0


def _cmd_simulate(args: argparse.Namespace) -> int:
    instance = Instance.from_dict(_read_json(args.file))
    result = simulate(instance, keep_all_states=args.trace is not None)
    gold = result.gold().to_dict()
    if args.gold is not None:
        args.gold.write_text(json.dumps(gold, indent=2) + "\n", encoding="utf-8")
    else:
        print(json.dumps(gold, indent=2))
    if args.trace is not None:
        trace = {
            "instance_id": instance.id,
            "steps": [step.to_dict() for step in result.steps],
            "states": {
                str(t): result.state_at(t).to_dict()["cells"] for t in range(instance.horizon + 1)
            },
        }
        args.trace.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    from .generation import PRESETS, build_dataset, config_from_dict, load_config

    if args.config in PRESETS:
        data = dict(PRESETS[args.config])
    else:
        data = dict(load_config(args.config).source)
    if args.queries is not None:
        data["queries"] = args.queries
    if args.query_policy is not None:
        data["query_policy"] = args.query_policy
    if args.split is not None:
        data["split"] = args.split
    if args.name is not None:
        data["name"] = args.name
    config = config_from_dict(data)
    progress = None if args.quiet else (lambda message: print(message, file=sys.stderr))
    manifest = build_dataset(
        config,
        args.out,
        families=args.families,
        dims=args.dims,
        indices=args.seeds,
        workers=args.workers,
        full_trace=args.full_trace,
        progress=progress,
    )
    totals = manifest["totals"]
    print(
        f"wrote {totals['instances']} instances from {totals['rollouts']} rollouts in "
        f"{totals['cells']} cells to {args.out} ({totals['infeasible_cells']} infeasible cells, "
        f"{totals['rejected_attempts']} rejected attempts)"
    )
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from .evaluation import evaluate_predictions, load_benchmark

    benchmark = load_benchmark(args.data, view=args.view,
                               verify_references=args.verify_references)
    result = evaluate_predictions(benchmark, args.predictions)
    args.out.mkdir(parents=True, exist_ok=True)
    scores = result.pop("scores")
    (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (args.out / "scores.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in scores), encoding="utf-8"
    )
    print(json.dumps(result["summary"], indent=2))
    return 0


def _cmd_demo() -> int:
    from .evaluation import score_answer

    world = WorldSpec(shape=(2, 2), values=(0, 1, 2))
    instance = Instance(
        id="grid-bench-demo", family="metric_paint", world=world,
        initial=WorldState(world, (((0, 0), 1),)),
        events=(Event("SET", {"p": [1, 1], "v": 2}),),
    )
    gold = simulate(instance).gold()
    print(json.dumps({
        "instance": instance.to_dict(), "prompt": render_prompt(instance),
        "reference": gold.final_state.to_dict(),
        "reference_score": score_answer(json.dumps(gold.final_state.to_dict()), instance, gold),
    }, indent=2))
    return 0


def _run(args: argparse.Namespace) -> int:
    if args.command == "demo":
        return _cmd_demo()
    if args.command == "families":
        return _cmd_families(args)
    if args.command == "validate-instance":
        instance = Instance.from_dict(_read_json(args.file))
        # Static schema/event validation is not sufficient for families whose
        # legality depends on the evolving state (for example, a compression
        # may merge beyond a finite value alphabet).  Replaying the instance
        # makes this command validate the complete transition sequence.
        simulate(instance)
        print(f"valid: {instance.id}")
        return 0
    if args.command == "render-prompt":
        instance = Instance.from_dict(_read_json(args.file))
        print(
            render_prompt(
                instance, representation=args.representation
            )
        )
        return 0
    if args.command == "simulate":
        return _cmd_simulate(args)
    if args.command == "generate":
        return _cmd_generate(args)
    if args.command == "evaluate":
        return _cmd_evaluate(args)
    raise AssertionError(f"unhandled command: {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI, returning a conventional process exit status."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except (OSError, json.JSONDecodeError, SchemaError, KeyError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
