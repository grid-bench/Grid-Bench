"""Extraction, baselines, scoring rows, and aggregation statistics."""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from random import Random
from types import SimpleNamespace

import pytest

from spatial_worlds.eval.aggregate import (
    evaluate,
    score_records,
    split_summary,
    summarize_cells,
    summarize_curves,
)
from spatial_worlds.eval.extract import extract_answer, top_level_object_spans
from spatial_worlds.eval.providers import (
    AnthropicProvider,
    Completion,
    CopyInitialProvider,
    OpenAICompatibleProvider,
    OracleProvider,
    RetryableError,
    make_provider,
)
from spatial_worlds.eval.run import (
    Dataset,
    complete_with_retries,
    failure_kind_of,
    load_dataset,
    matches_filter,
    parse_filter,
    parse_params,
    parse_retry_kinds,
    run_predictions,
    should_retry,
)
from spatial_worlds.eval.stats import bootstrap_epsilon, epsilon_mle, linear_fit, wilson_interval
from spatial_worlds.generation import build_dataset, config_from_dict
from spatial_worlds.prompt import render_answer

TINY = {
    "name": "tiny",
    "dataset_seed": 11,
    "split": "dev",
    "tier": "S",
    "dims": [2],
    "t_grid": [0, 1, 2, 4, 8],
    "t_max": 8,
    "seeds_per_cell": 2,
    "families": {"metric_paint": {"w_targets": [7]}, "boundary_layer": {"w_targets": [6]}},
}


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data")
    build_dataset(config_from_dict(TINY), path)
    return path


@pytest.mark.parametrize(
    "raw,expected_text,method",
    [
        ('{"cells":[]}', '{"cells":[]}', "none"),
        ('  \n{"cells":[[[0,0],1]]}\n', '{"cells":[[[0,0],1]]}', "none"),
        ('```json\n{"cells":[]}\n```', '{"cells":[]}', "fence"),
        ('```\n{"cells":[]}```', '{"cells":[]}', "fence"),
        ('Final answer:\n{"cells":[[[0,0],1]]}\nThanks.', '{"cells":[[[0,0],1]]}', "last_object"),
        ('{"cells":[]} then later {"cells":[[[1,1],2]]}', '{"cells":[[[1,1],2]]}', "last_object"),
        ('text {"a":"}"} more {"cells":[]}', '{"cells":[]}', "last_object"),
        ('Here is ```json\n{"cells":[]}\n``` done', '{"cells":[]}', "last_object"),
        ("no json here", None, "failed"),
        ('{"cells":[', None, "failed"),
        ("[1,2,3]", None, "failed"),
    ],
)
def test_extract_answer(raw: str, expected_text: str | None, method: str) -> None:
    extraction = extract_answer(raw)
    assert extraction.text == expected_text
    assert extraction.method == method
    assert extraction.applied == (method != "none")


def test_top_level_spans_ignore_braces_in_strings() -> None:
    text = 'a {"x":"{"} b {"y":{"z":1}} c'
    spans = top_level_object_spans(text)
    assert [text[a:b] for a, b in spans] == ['{"x":"{"}', '{"y":{"z":1}}']


def test_wilson_and_epsilon_statistics() -> None:
    low, high = wilson_interval(5, 10)
    assert 0.2 < low < 0.5 < high < 0.8
    assert wilson_interval(0, 0) == (0.0, 1.0)
    assert epsilon_mle([]) is None
    assert epsilon_mle([(4, True), (8, True)]) == 0.0
    assert epsilon_mle([(4, False)]) == 1.0
    rng = Random(3)
    true_eps = 0.05
    observations = []
    rollouts = []
    for _ in range(300):
        rollout = []
        for n in (1, 2, 4, 8, 16, 32):
            correct = all(rng.random() > true_eps for _ in range(n))
            rollout.append((n, correct))
        rollouts.append(rollout)
        observations.extend(rollout)
    estimate = epsilon_mle(observations)
    assert estimate is not None and abs(estimate - true_eps) < 0.015
    ci = bootstrap_epsilon(rollouts, resamples=50)
    assert ci is not None and ci[0] <= estimate <= ci[1]


def test_baseline_providers_and_retry_loop(dataset_dir: Path) -> None:
    dataset = load_dataset(dataset_dir)
    instance = dataset.instances[0]
    gold = dataset.gold[instance.id]
    copy = CopyInitialProvider().complete("", instance=instance, gold=gold)
    assert copy.text == render_answer(instance.initial)
    oracle = OracleProvider().complete("", instance=instance, gold=gold)
    assert oracle.text == render_answer(gold.final_state)
    assert OracleProvider().complete("", instance=instance, gold=None).error

    class Flaky:
        name = "flaky"
        model = "flaky"

        def __init__(self) -> None:
            self.calls = 0

        def complete(self, prompt: str, *, instance, gold) -> Completion:
            self.calls += 1
            if self.calls < 3:
                raise RetryableError("busy", retry_after=0.0)
            return Completion(text="{}")

        def request_params(self) -> dict[str, object]:
            return {}

    flaky = Flaky()
    completion, attempts = complete_with_retries(
        flaky, "p", instance=instance, gold=None, max_attempts=5, sleep=lambda _: None
    )
    assert completion.text == "{}" and attempts == 3
    exhausted = Flaky()
    completion, attempts = complete_with_retries(
        exhausted, "p", instance=instance, gold=None, max_attempts=2, sleep=lambda _: None
    )
    assert completion.error and attempts == 2
    with pytest.raises(ValueError):
        make_provider("openai", None)
    with pytest.raises(ValueError):
        parse_filter("nope=1")
    assert parse_filter("family=metric_paint,d=1|2") == {
        "family": {"metric_paint"},
        "d": {"1", "2"},
    }
    assert parse_params(["max_tokens=5", "name=x"]) == {"max_tokens": 5, "name": "x"}


def test_index_filter_selects_rollouts_by_their_id_suffix(dataset_dir: Path) -> None:
    dataset = load_dataset(dataset_dir)
    filters = parse_filter("index=0")
    selected = [i for i in dataset.instances if matches_filter(i, filters)]
    assert selected and all(i.metadata["prefix_of"].endswith("/0000") for i in selected)
    assert any(not i.metadata["prefix_of"].endswith("/0000") for i in dataset.instances)
    assert all(
        matches_filter(i, filters) == i.metadata["prefix_of"].endswith("/0000")
        for i in dataset.instances
    )


def test_anthropic_custom_key_environment_is_enforced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_anthropic(**kwargs: object) -> object:
        calls.append(kwargs)
        return object()

    fake_module = SimpleNamespace(Anthropic=fake_anthropic)
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)
    monkeypatch.delenv("SPATIAL_WORLDS_TEST_KEY", raising=False)
    with pytest.raises(RuntimeError, match="SPATIAL_WORLDS_TEST_KEY"):
        AnthropicProvider("test-model", api_key_env="SPATIAL_WORLDS_TEST_KEY")

    monkeypatch.setenv("SPATIAL_WORLDS_TEST_KEY", "test-secret")
    AnthropicProvider("test-model", api_key_env="SPATIAL_WORLDS_TEST_KEY")
    assert calls[0]["api_key"] == "test-secret"


def test_openai_provider_streams_text_and_reads_usage_from_the_final_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeError(Exception):
        pass

    calls: list[dict[str, object]] = []

    def create(**kwargs: object) -> list[object]:
        calls.append(kwargs)
        chunk = lambda content, finish=None: SimpleNamespace(  # noqa: E731
            usage=None,
            choices=[SimpleNamespace(delta=SimpleNamespace(content=content), finish_reason=finish)],
        )
        usage = SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=7,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=4),
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        )
        return [chunk('{"cells":'), chunk("[]}", "stop"), SimpleNamespace(usage=usage, choices=[])]

    def fake_client(**kwargs: object) -> object:
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    fake_module = SimpleNamespace(
        OpenAI=fake_client,
        RateLimitError=FakeError,
        APIConnectionError=FakeError,
        APITimeoutError=FakeError,
        InternalServerError=FakeError,
        APIStatusError=FakeError,
    )
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    monkeypatch.setenv("SPATIAL_WORLDS_TEST_KEY", "k")
    provider = OpenAICompatibleProvider(
        "m", base_url="http://gateway/v1", api_key_env="SPATIAL_WORLDS_TEST_KEY", max_tokens=5
    )
    completion = provider.complete("prompt", instance=None, gold=None)  # type: ignore[arg-type]
    assert completion.text == '{"cells":[]}' and completion.error is None
    assert completion.finish_reason == "stop"
    assert completion.usage == {
        "input_tokens": 10,
        "output_tokens": 7,
        "reasoning_tokens": 4,
        "cached_input_tokens": 0,
    }
    assert calls[0]["stream"] is True and calls[0]["stream_options"] == {"include_usage": True}
    assert calls[0]["max_tokens"] == 5


def test_openai_provider_reads_deepseek_style_cache_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeError(Exception):
        pass

    def create(**kwargs: object) -> list[object]:
        chunk = SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(delta=SimpleNamespace(content="{}"), finish_reason="stop")],
        )
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=7, prompt_cache_hit_tokens=6)
        return [chunk, SimpleNamespace(usage=usage, choices=[])]

    fake_module = SimpleNamespace(
        OpenAI=lambda **kwargs: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ),
        RateLimitError=FakeError,
        APIConnectionError=FakeError,
        APITimeoutError=FakeError,
        InternalServerError=FakeError,
        APIStatusError=FakeError,
    )
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    monkeypatch.setenv("SPATIAL_WORLDS_TEST_KEY", "k")
    provider = OpenAICompatibleProvider(
        "deepseek-v4-pro",
        base_url="https://api.deepseek.com",
        api_key_env="SPATIAL_WORLDS_TEST_KEY",
        effort="max",
    )
    completion = provider.complete("prompt", instance=None, gold=None)  # type: ignore[arg-type]
    assert completion.usage == {
        "input_tokens": 10,
        "output_tokens": 7,
        "reasoning_tokens": None,
        "cached_input_tokens": 6,
    }
    assert provider.request_params()["reasoning_effort"] == "max"


def test_openai_provider_enforces_a_wall_clock_deadline_on_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeError(Exception):
        pass

    closed: list[bool] = []

    class Stream:
        def __iter__(self) -> Iterator[object]:
            chunk = SimpleNamespace(
                usage=None,
                choices=[SimpleNamespace(delta=SimpleNamespace(content="x"), finish_reason=None)],
            )
            while True:
                yield chunk

        def close(self) -> None:
            closed.append(True)

    fake_client = lambda **kwargs: SimpleNamespace(  # noqa: E731
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: Stream()))
    )
    fake_module = SimpleNamespace(
        OpenAI=fake_client,
        RateLimitError=FakeError,
        APIConnectionError=FakeError,
        APITimeoutError=FakeError,
        InternalServerError=FakeError,
        APIStatusError=FakeError,
    )
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    monkeypatch.setenv("SPATIAL_WORLDS_TEST_KEY", "k")
    provider = OpenAICompatibleProvider(
        "m", base_url="http://gateway/v1", api_key_env="SPATIAL_WORLDS_TEST_KEY", timeout=0.05
    )
    with pytest.raises(RetryableError, match="no completion within"):
        provider.complete("prompt", instance=None, gold=None)  # type: ignore[arg-type]
    assert closed == [True]


def test_openai_deadline_holds_while_the_stream_blocks_between_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading

    class FakeError(Exception):
        pass

    release = threading.Event()
    closed: list[bool] = []

    class BlockingStream:
        def __iter__(self) -> Iterator[object]:
            yield SimpleNamespace(
                usage=None,
                choices=[SimpleNamespace(delta=SimpleNamespace(content="x"), finish_reason=None)],
            )
            # a proxy heartbeat keeps the socket open but no chunk ever arrives
            release.wait()

        def close(self) -> None:
            closed.append(True)
            release.set()

    fake_client = lambda **kwargs: SimpleNamespace(  # noqa: E731
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: BlockingStream()))
    )
    fake_module = SimpleNamespace(
        OpenAI=fake_client,
        RateLimitError=FakeError,
        APIConnectionError=FakeError,
        APITimeoutError=FakeError,
        InternalServerError=FakeError,
        APIStatusError=FakeError,
    )
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    monkeypatch.setenv("SPATIAL_WORLDS_TEST_KEY", "k")
    provider = OpenAICompatibleProvider(
        "m", base_url="http://gateway/v1", api_key_env="SPATIAL_WORLDS_TEST_KEY", timeout=0.2
    )
    assert provider.request_params()["timeout_s"] == 0.2
    started = time.monotonic()
    with pytest.raises(RetryableError, match="no completion within") as info:
        provider.complete("prompt", instance=None, gold=None)  # type: ignore[arg-type]
    assert info.value.kind == "deadline"
    assert time.monotonic() - started < 5.0
    assert closed == [True]


def test_run_evaluate_pipeline_with_baselines(dataset_dir: Path, tmp_path: Path) -> None:
    dataset = load_dataset(dataset_dir)
    oracle_out = tmp_path / "oracle.jsonl"
    summary = run_predictions(dataset, OracleProvider(), out=oracle_out, concurrency=2)
    assert summary["completed"] == len(dataset.instances)
    resumed = run_predictions(dataset, OracleProvider(), out=oracle_out, concurrency=2)
    assert resumed["completed"] == 0 and resumed["skipped_existing"] == len(dataset.instances)
    copy_out = tmp_path / "copy.jsonl"
    run_predictions(
        dataset, CopyInitialProvider(), out=copy_out, filters=parse_filter("family=metric_paint")
    )
    result = evaluate(dataset, [oracle_out, copy_out], tmp_path / "results", bootstrap=10)
    assert result["instances"] == len(dataset.instances) + len(
        [i for i in dataset.instances if i.family == "metric_paint"]
    )
    results = json.loads((tmp_path / "results" / "results.json").read_text())
    oracle_cells = [c for c in results["cells"] if c["model"] == "oracle"]
    assert oracle_cells and all(c["em"] == 1.0 and c["strict_em"] == 1.0 for c in oracle_cells)
    assert all(c["canonical_rate"] == 1.0 and c["valid_state_rate"] == 1.0 for c in oracle_cells)
    copy_cells = [c for c in results["cells"] if c["model"] == "copy-initial"]
    assert all(c["em"] == 1.0 for c in copy_cells if c["T"] == 0)
    assert all(c["family"] == "metric_paint" for c in copy_cells)
    for c in copy_cells:
        assert c["em"] == c["copy_baseline_em"]
        if c["T"] >= 1:
            assert c["mean_core_scan"] is not None
    curves = [c for c in results["curves"] if c["model"] == "oracle"]
    assert all(c["t_star_censored"] and c["epsilon"] == 0.0 for c in curves)
    coverage = {item["model"]: item for item in results["coverage"]}
    assert coverage["oracle"]["coverage"] == 1.0
    assert 0 < coverage["copy-initial"]["coverage"] < 1
    assert (tmp_path / "results" / "summary.md").exists()
    assert (tmp_path / "results" / "summary.csv").exists()
    assert (tmp_path / "results" / "curves.csv").exists()
    summary_md = (tmp_path / "results" / "summary.md").read_text()
    assert "Model `oracle`" in summary_md and "metric_paint" in summary_md
    # the leaderboard uses the largest horizon present (8 here) and the next one >= 8 is absent
    leaderboard = summary_md.split("## Leaderboard", 1)[1].split("## Model", 1)[0]
    assert "EM T=8" in leaderboard and "EM (none)" in leaderboard
    oracle_row = next(line for line in leaderboard.splitlines() if line.startswith("| `oracle`"))
    assert "| 1.00 |" in oracle_row


def test_plots_cover_model_panels_workload_and_erasure(dataset_dir: Path, tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from spatial_worlds.eval.report import write_plots

    dataset = load_dataset(dataset_dir)
    oracle_out = tmp_path / "oracle.jsonl"
    run_predictions(dataset, OracleProvider(), out=oracle_out, concurrency=2)
    run_predictions(
        dataset, make_provider("suffix", "suffix-2"), out=tmp_path / "suffix.jsonl", concurrency=2
    )
    # a second workload level and a synthetic non-baseline condition come from
    # relabelling the oracle records: the plots must draw a model panel per
    # family with the suffix baseline dashed, a workload panel, and the erasure bars
    records = [json.loads(line) for line in oracle_out.read_text().splitlines()]
    fake = tmp_path / "model.jsonl"
    with fake.open("w") as stream:
        for record in records:
            record = {**record, "provider": "openai", "model": "fake-model"}
            record.pop("condition_id", None)
            stream.write(json.dumps(record) + "\n")
    results = tmp_path / "plot-results"
    evaluate(dataset, [fake, tmp_path / "suffix.jsonl"], results, bootstrap=5)
    written = write_plots(results)
    names = {path.name for path in written}
    assert any(
        name.startswith("fake-model__") and not name.endswith("__workload.png") for name in names
    )
    assert all(not name.startswith("suffix-2__") for name in names)


def test_retry_append_is_last_write_wins(dataset_dir: Path, tmp_path: Path) -> None:
    dataset = load_dataset(dataset_dir)
    instance = next(i for i in dataset.instances if i.horizon == 4)
    gold = dataset.gold[instance.id]
    selected = Dataset(instances=[instance], gold={instance.id: gold})

    class FixedProvider:
        name = "fixed"
        model = "fixed-model"

        def __init__(self, completion: Completion) -> None:
            self.completion = completion
            self.calls = 0

        def complete(self, prompt: str, *, instance, gold) -> Completion:
            del prompt, instance, gold
            self.calls += 1
            return self.completion

        def request_params(self) -> dict[str, object]:
            return {"temperature": 0}

    predictions = tmp_path / "attempts.jsonl"
    failed = FixedProvider(Completion(text=None, error="temporary failure"))
    run_predictions(selected, failed, out=predictions, concurrency=1)
    succeeded = FixedProvider(Completion(text=render_answer(gold.final_state)))
    rerun = run_predictions(
        selected,
        succeeded,
        out=predictions,
        concurrency=1,
        retry_errors=True,
    )
    assert rerun["completed"] == 1 and succeeded.calls == 1

    physical = [json.loads(line) for line in predictions.read_text().splitlines()]
    assert len(physical) == 2
    assert physical[0]["condition_id"] == physical[1]["condition_id"]
    result = evaluate(selected, [predictions], tmp_path / "retry-results", bootstrap=0)
    assert result["instances"] == 1
    rows = [
        json.loads(line)
        for line in (tmp_path / "retry-results/results.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 1 and rows[0]["em"] and rows[0]["error"] is None
    cells = json.loads((tmp_path / "retry-results/results.json").read_text())["cells"]
    assert len(cells) == 1 and cells[0]["n"] == 1 and cells[0]["em"] == 1.0


def test_conditions_control_resume_and_aggregation(dataset_dir: Path, tmp_path: Path) -> None:
    dataset = load_dataset(dataset_dir)
    instance = next(i for i in dataset.instances if i.dimension == 2 and i.horizon == 1)
    gold = dataset.gold[instance.id]
    selected = Dataset(instances=[instance], gold={instance.id: gold})
    answer = render_answer(gold.final_state)

    class ConfiguredProvider:
        name = "configured"
        model = "same-model"

        def __init__(self, temperature: int) -> None:
            self.temperature = temperature
            self.calls = 0

        def complete(self, prompt: str, *, instance, gold) -> Completion:
            del prompt, instance, gold
            self.calls += 1
            return Completion(text=answer)

        def request_params(self) -> dict[str, object]:
            return {"temperature": self.temperature}

    predictions = tmp_path / "conditions.jsonl"
    first = ConfiguredProvider(0)
    run_predictions(selected, first, out=predictions, concurrency=1)
    exact_resume = ConfiguredProvider(0)
    resumed = run_predictions(selected, exact_resume, out=predictions, concurrency=1)
    assert resumed["skipped_existing"] == 1 and exact_resume.calls == 0

    changed_request = ConfiguredProvider(1)
    run_predictions(selected, changed_request, out=predictions, concurrency=1)
    changed_representation = ConfiguredProvider(1)
    run_predictions(
        selected,
        changed_representation,
        out=predictions,
        concurrency=1,
        representation="grid",
    )
    assert changed_request.calls == changed_representation.calls == 1
    records = [json.loads(line) for line in predictions.read_text().splitlines()]
    assert len(records) == 3
    assert len({record["condition_id"] for record in records}) == 3

    result = evaluate(selected, [predictions], tmp_path / "condition-results", bootstrap=0)
    assert result["instances"] == 3 and result["cells"] == 3 and result["curves"] == 3
    cells = json.loads((tmp_path / "condition-results/results.json").read_text())["cells"]
    assert {cell["representation"] for cell in cells} == {"cells", "grid"}
    assert len({cell["condition_id"] for cell in cells}) == 3


def test_evaluate_rejects_stale_prompt_hash(dataset_dir: Path, tmp_path: Path) -> None:
    dataset = load_dataset(dataset_dir)
    instance = dataset.instances[0]
    selected = Dataset(instances=[instance], gold={instance.id: dataset.gold[instance.id]})
    predictions = tmp_path / "stale.jsonl"
    run_predictions(selected, OracleProvider(), out=predictions, concurrency=1)
    record = json.loads(predictions.read_text())
    record["prompt_sha256"] = "0" * 64
    predictions.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="mismatched prompt_sha256"):
        evaluate(selected, [predictions], tmp_path / "stale-results", bootstrap=0)

    record.pop("condition_id")
    record.pop("prompt_sha256")
    record["instance_id"] = "not-in-this-dataset"
    predictions.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="unknown instance_id"):
        evaluate(selected, [predictions], tmp_path / "unknown-results", bootstrap=0)


def test_manifest_scopes_loading_and_dataset_ids_are_unique(
    dataset_dir: Path, tmp_path: Path
) -> None:
    source = load_dataset(dataset_dir)
    instance = source.instances[0]
    gold = source.gold[instance.id]
    root = tmp_path / "manifest-data"
    current = root / "dev" / instance.family
    current.mkdir(parents=True)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "split": "dev",
                "cells": [{"family": instance.family, "rollouts": 1}],
            }
        )
        + "\n"
    )
    (current / "instances.jsonl").write_text(json.dumps(instance.to_dict()) + "\n")
    (current / "gold.jsonl").write_text(json.dumps(gold.to_dict()) + "\n")
    (current / "trace.jsonl").write_text("")

    stale = root / "old-split" / instance.family
    stale.mkdir(parents=True)
    (stale / "instances.jsonl").write_text(json.dumps(instance.to_dict()) + "\n")
    (stale / "gold.jsonl").write_text(json.dumps(gold.to_dict()) + "\n")
    loaded = load_dataset(root)
    assert [item.id for item in loaded.instances] == [instance.id]
    assert list(loaded.gold) == [instance.id]

    duplicate_instances = tmp_path / "duplicate-instances.jsonl"
    duplicate_instances.write_text(
        json.dumps(instance.to_dict()) + "\n" + json.dumps(instance.to_dict()) + "\n"
    )
    with pytest.raises(ValueError, match="duplicate instance id"):
        load_dataset(instances=[duplicate_instances])

    one_instance = tmp_path / "one-instance.jsonl"
    one_instance.write_text(json.dumps(instance.to_dict()) + "\n")
    duplicate_gold = tmp_path / "duplicate-gold.jsonl"
    duplicate_gold.write_text(json.dumps(gold.to_dict()) + "\n" + json.dumps(gold.to_dict()) + "\n")
    with pytest.raises(ValueError, match="duplicate gold instance_id"):
        load_dataset(instances=[one_instance], gold=[duplicate_gold])

    mismatched = gold.to_dict()
    mismatched["family"] = "not_the_instance_family"
    mismatched_gold = tmp_path / "mismatched-gold.jsonl"
    mismatched_gold.write_text(json.dumps(mismatched) + "\n")
    with pytest.raises(ValueError, match="does not match instance family"):
        load_dataset(instances=[one_instance], gold=[mismatched_gold])


def test_scoring_rows_distinguish_strict_and_extracted(dataset_dir: Path) -> None:
    dataset = load_dataset(dataset_dir)
    instance = next(i for i in dataset.instances if i.horizon == 4)
    gold = dataset.gold[instance.id]
    answer = render_answer(gold.final_state)
    records = [
        {
            "instance_id": instance.id,
            "model": "m-extracted",
            "raw_text": f"```json\n{answer}\n```",
            "extracted_text": answer,
            "extraction_method": "fence",
            "extraction_applied": True,
            "usage": {"output_tokens": 10, "reasoning_tokens": 3},
        },
        {
            "instance_id": instance.id,
            "model": "m-invalid",
            "raw_text": "garbage",
            "extracted_text": None,
            "extraction_method": "failed",
            "extraction_applied": False,
            "usage": {},
        },
    ]
    rows = score_records(dataset, records)
    assert rows[0]["em"] and not rows[0]["strict_em"]
    assert rows[0]["changed_cell_accuracy"] in (None, 1.0)
    assert not rows[1]["em"] and not rows[1]["valid_json"]
    cells = summarize_cells(rows)
    assert len(cells) == 2 and all(cell["n"] == 1 for cell in cells)
    extracted = next(cell for cell in cells if cell["model"] == "m-extracted")
    assert extracted["em"] == 1.0 and extracted["strict_em"] == 0.0
    assert extracted["mean_reasoning_tokens"] == 3.0
    curves = summarize_curves(rows, bootstrap=5)
    assert len(curves) == 2 and all(curve["n_rollouts"] == 1 for curve in curves)


def test_synthetic_curve_statistics() -> None:
    rows = []
    for rollout in range(20):
        for T in (0, 1, 2, 4, 8):
            rows.append(
                {
                    "model": "m",
                    "family": "f",
                    "tier": "S",
                    "dimension": 2,
                    "w": 7,
                    "T": T,
                    "rollout_id": f"r{rollout}",
                    "n_core": T,
                    "em": T <= 2 or (rollout < 4 and T == 4),
                    "strict_em": False,
                    "valid_json": True,
                    "valid_state": True,
                    "canonical_format": True,
                    "cell_accuracy": 1.0,
                    "usage": {},
                }
            )
    curve = summarize_curves(rows, threshold=0.5, bootstrap=10)[0]
    assert curve["em_by_T"][2] == 1.0 and curve["em_by_T"][4] == 0.2 and curve["em_by_T"][8] == 0.0
    assert curve["t_star_last_pass"] == 2
    assert 2 < curve["t_star_first_cross"] < 4
    assert curve["first_failing_prefix"] == {"4": 16, "8": 4}
    assert curve["epsilon_conditional"] == curve["epsilon"]
    assert curve["epsilon_conditional_rollouts"] == 20
    assert curve["non_monotone_rollouts"] == 0
    assert 0 < curve["epsilon"] < 1
    assert curve["epsilon_ci_low"] <= curve["epsilon"] <= curve["epsilon_ci_high"]
    assert Dataset(instances=[]).gold_for is not None


def _fake_openai(monkeypatch: pytest.MonkeyPatch, chunks: list[object]) -> None:
    class FakeError(Exception):
        pass

    fake_client = lambda **kwargs: SimpleNamespace(  # noqa: E731
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: iter(chunks)))
    )
    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(
            OpenAI=fake_client,
            RateLimitError=FakeError,
            APIConnectionError=FakeError,
            APITimeoutError=FakeError,
            InternalServerError=FakeError,
            APIStatusError=FakeError,
        ),
    )
    monkeypatch.setenv("SPATIAL_WORLDS_TEST_KEY", "k")


def test_openai_provider_classifies_an_exhausted_budget_as_a_budget_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usage = SimpleNamespace(
        prompt_tokens=10,
        completion_tokens=128000,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=128000),
        prompt_tokens_details=None,
    )
    _fake_openai(
        monkeypatch,
        [
            SimpleNamespace(
                usage=None,
                choices=[
                    SimpleNamespace(delta=SimpleNamespace(content=None), finish_reason="length")
                ],
            ),
            SimpleNamespace(usage=usage, choices=[]),
        ],
    )
    provider = OpenAICompatibleProvider(
        "m", base_url="http://gateway/v1", api_key_env="SPATIAL_WORLDS_TEST_KEY"
    )
    completion = provider.complete("prompt", instance=None, gold=None)  # type: ignore[arg-type]
    assert completion.text is None and completion.failure_kind == "budget"
    assert completion.finish_reason == "length"
    assert completion.usage["output_tokens"] == 128000

    _fake_openai(
        monkeypatch,
        [
            SimpleNamespace(
                usage=None,
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=None), finish_reason="content_filter"
                    )
                ],
            )
        ],
    )
    provider = OpenAICompatibleProvider(
        "m", base_url="http://gateway/v1", api_key_env="SPATIAL_WORLDS_TEST_KEY"
    )
    completion = provider.complete("prompt", instance=None, gold=None)  # type: ignore[arg-type]
    assert completion.failure_kind == "empty" and completion.error == "empty response"


def test_failure_kinds_are_inferred_for_old_records_and_gate_retries() -> None:
    deadline = {"error": "gave up after 4 attempts: transient: Request timed out.", "usage": {}}
    stalled = {"error": "gave up after 4 attempts: transient: no completion within 1500s"}
    budget = {"error": "empty response", "finish_reason": "length"}
    empty = {"error": "empty response", "finish_reason": "content_filter"}
    status = {"error": "status 400: bad request"}
    transport = {"error": "gave up after 6 attempts: transient: Connection error."}
    explicit = {"error": "anything", "failure_kind": "refusal"}
    assert failure_kind_of({"error": None}) is None
    assert failure_kind_of(deadline) == "deadline" and failure_kind_of(stalled) == "deadline"
    assert failure_kind_of(budget) == "budget" and failure_kind_of(empty) == "empty"
    assert failure_kind_of(status) == "status" and failure_kind_of(transport) == "transport"
    assert failure_kind_of(explicit) == "refusal"

    assert not should_retry(deadline, None) and not should_retry(budget, None)
    assert should_retry(empty, None) and should_retry(transport, None)
    assert should_retry(budget, parse_retry_kinds("budget,empty"))
    assert not should_retry(transport, parse_retry_kinds("budget"))
    assert should_retry(deadline, parse_retry_kinds("all"))
    assert parse_retry_kinds("") is None
    with pytest.raises(ValueError, match="unknown failure kind"):
        parse_retry_kinds("stall")


def test_give_up_records_carry_the_failure_kind_and_are_not_retried_by_default(
    dataset_dir: Path, tmp_path: Path
) -> None:
    class StallingProvider:
        name = "stall"
        model = "stall"
        calls = 0

        def complete(self, prompt: str, *, instance: object, gold: object) -> Completion:
            type(self).calls += 1
            raise RetryableError("no completion within 1s", kind="deadline")

        def request_params(self) -> dict[str, object]:
            return {"max_tokens": 50}

    dataset = load_dataset(dataset_dir)
    out = tmp_path / "stall.jsonl"
    filters = parse_filter("family=metric_paint,T=1")
    run_predictions(dataset, StallingProvider(), out=out, filters=filters, max_attempts=2)
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert records and all(r["failure_kind"] == "deadline" for r in records)
    assert all(r["attempts"] == 2 for r in records)
    before = StallingProvider.calls
    summary = run_predictions(
        dataset, StallingProvider(), out=out, filters=filters, retry_errors=True, max_attempts=2
    )
    assert summary["completed"] == 0 and StallingProvider.calls == before
    summary = run_predictions(
        dataset,
        StallingProvider(),
        out=out,
        filters=filters,
        retry_errors=True,
        retry_kinds=frozenset({"deadline"}),
        max_attempts=2,
    )
    assert summary["completed"] == len(records)

    rows = score_records(dataset, [json.loads(line) for line in out.read_text().splitlines()])
    cell = summarize_cells(rows)[0]
    assert cell["no_answer_rate"] == 1.0 and cell["other_error_rate"] == 0.0
    # nothing was observed; only the separately labelled imputation uses the budget
    assert cell["output_tokens_p90"] is None and cell["usage_known_rate"] == 0.0
    assert cell["output_tokens_p90_imputed"] == 50.0


def test_token_fit_and_dimension_splits() -> None:
    rows: list[dict[str, object]] = []
    for dimension, base in ((2, 1000.0), (4, 3000.0)):
        for rollout in range(4):
            for T in (0, 1, 8, 16, 32):
                n_core = T - T // 4
                rows.append(
                    {
                        "condition_id": "c",
                        "model": "m",
                        "family": "metric_paint",
                        "tier": "M",
                        "dimension": dimension,
                        "w": 65,
                        "T": T,
                        "set_rate": 0.25,
                        "rollout_id": f"r{rollout}",
                        "n_core": n_core,
                        "em": dimension == 2 or T <= 8,
                        "strict_em": False,
                        "valid_json": True,
                        "valid_state": True,
                        "canonical_format": True,
                        "cell_accuracy": 1.0,
                        "usage": {"output_tokens": base + 400.0 * n_core},
                        "failure_kind": None,
                        "request": {"max_tokens": 128000},
                    }
                )
    rows[-1]["usage"] = {}
    rows[-1]["failure_kind"] = "deadline"
    rows[-1]["error"] = "gave up"
    curves = summarize_curves(rows, bootstrap=5, token_budget=9000)
    two = next(c for c in curves if c["dimension"] == 2)
    assert abs(two["tokens_intercept"] - 1000.0) < 1e-6
    assert abs(two["tokens_per_event"] - 400.0) < 1e-6
    assert two["tokens_r2"] == 1.0
    assert abs(two["n_core_at_budget"] - 20.0) < 1e-6
    assert abs(two["t_at_budget"] - 20.0 / 0.75) < 1e-6
    assert two["t_first_no_answer"] is None
    four = next(c for c in curves if c["dimension"] == 4)
    assert four["t_first_no_answer"] == 32
    assert four["no_answer_by_T"][32] == 0.25 and four["no_answer_by_T"][16] == 0.0

    assert two["tokens_fit_n"] == 16 and two["tokens_resid_sd"] == 0.0
    assert two["t_at_budget_p90"] == two["t_at_budget"]

    cells = summarize_cells(rows)
    splits = {(s["family"], s["split"]): s for s in split_summary(cells)}
    # the split means use the headline horizons only: T = 0 and T = 1 are controls
    assert splits[("metric_paint", "d1-3")]["mean_em"] == 1.0
    assert splits[("metric_paint", "d4-5")]["mean_em"] == pytest.approx(1 / 3)
    assert splits[("all", "d4-5")]["cells"] == 3
    assert splits[("all", "d4-5")]["horizons"] == [8, 16, 32]
    assert splits[("all", "d1-3")]["dimensions"] == [1, 2, 3]
    assert splits[("all", "d1-3")]["tier"] == "M"
    narrow = {(s["family"], s["split"]): s for s in split_summary(cells, horizons=[8])}
    assert narrow[("metric_paint", "d4-5")]["mean_em"] == 1.0
    assert narrow[("metric_paint", "d4-5")]["horizons"] == [8]
    # an explicit dimension mask keeps supplementary dimensions out of the means
    assert splits[("all", "d1-3")]["dimensions_used"] == [2]
    masked = {(s["family"], s["split"]): s for s in split_summary(cells, dimensions=[2, 3])}
    assert ("all", "d4-5") not in masked
    assert masked[("metric_paint", "d1-3")]["dimensions_used"] == [2]
    assert masked[("metric_paint", "d1-3")]["mean_em"] == 1.0
    assert linear_fit([(1.0, 1.0)]) is None and linear_fit([(1.0, 1.0), (1.0, 2.0)]) is None


def test_token_fit_ignores_failed_requests_even_when_they_report_usage() -> None:
    def row(rollout: int, T: int, tokens: float, **extra: object) -> dict[str, object]:
        return {
            "condition_id": "c",
            "model": "m",
            "family": "metric_paint",
            "tier": "M",
            "dimension": 2,
            "w": 65,
            "T": T,
            "set_rate": 0.0,
            "rollout_id": f"r{rollout}",
            "n_core": T,
            "em": True,
            "strict_em": False,
            "valid_json": True,
            "valid_state": True,
            "canonical_format": True,
            "cell_accuracy": 1.0,
            "usage": {"output_tokens": tokens},
            "failure_kind": None,
            "request": {"max_tokens": 1000},
            **extra,
        }

    answered = [row(0, 1, 110.0), row(0, 2, 120.0), row(1, 1, 110.0), row(1, 2, 120.0)]
    curve = summarize_curves(answered, bootstrap=0, token_budget=1000)[0]
    assert curve["tokens_per_event"] == pytest.approx(10.0)
    assert curve["t_at_budget"] == pytest.approx(90.0)
    # a budget failure that still reports its usage must not enter the fit
    failed = row(2, 2, 1000.0, em=False, failure_kind="budget", error="budget exhausted")
    curve = summarize_curves([*answered, failed], bootstrap=0, token_budget=1000)[0]
    assert curve["tokens_per_event"] == pytest.approx(10.0) and curve["tokens_fit_n"] == 4
    assert curve["no_answer_by_T"][2] == pytest.approx(1 / 3)
    cell = next(c for c in summarize_cells([*answered, failed]) if c["T"] == 2)
    assert cell["usage_known_rate"] == 1.0
    assert cell["output_tokens_p90"] == cell["output_tokens_p90_imputed"]
    # a copy control that was not observed keeps the conditional estimate honest
    rows_without_copy = answered
    curve = summarize_curves(rows_without_copy, bootstrap=0)[0]
    assert curve["epsilon_conditional"] is None and curve["epsilon_conditional_rollouts"] == 0
