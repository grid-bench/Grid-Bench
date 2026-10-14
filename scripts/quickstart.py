"""Execute and score one small example locally, without credentials or model calls."""

from __future__ import annotations

import json
from pathlib import Path

from spatial_worlds.prompt import COMPLETION_PROMPT_VERSION, render_prompt
from spatial_worlds.schema import Instance
from spatial_worlds.scoring import score_prediction
from spatial_worlds.simulate import simulate


def main() -> None:
    example = Path(__file__).resolve().parents[1] / "examples/metric_paint.json"
    instance = Instance.from_dict(json.loads(example.read_text()))
    gold = simulate(instance).gold()
    prediction = json.dumps(gold.final_state.to_dict())
    copied = json.dumps(instance.initial.to_dict())
    prompt = render_prompt(instance, prompt_version=COMPLETION_PROMPT_VERSION)
    assert COMPLETION_PROMPT_VERSION == "spatial-worlds-prompt/0.4"
    assert score_prediction(prediction, gold).exact_match
    print(json.dumps({
        "instance_id": instance.id,
        "shape": instance.world.shape,
        "events": instance.horizon,
        "prompt_version": COMPLETION_PROMPT_VERSION,
        "prompt_characters": len(prompt),
        "reference_state": gold.final_state.to_dict(),
        "reference_score": score_prediction(prediction, gold).to_dict(),
        "copy_initial_score": score_prediction(copied, gold).to_dict(),
    }, indent=2))


if __name__ == "__main__":
    main()
