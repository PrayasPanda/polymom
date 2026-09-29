import json
from pathlib import Path

import pytest

from app.core.config import Settings
from scripts import eval_summary

FIXTURES = Path(__file__).parents[1] / "fixtures" / "meetings"


def test_fuzzy_match_is_one_to_one() -> None:
    gold = ["Fix the login bug", "Write release notes"]
    predicted = ["write the release notes", "fix login bug today", "unrelated"]
    assert eval_summary.fuzzy_match(gold, predicted) == {0: 1, 1: 0}
    assert eval_summary.fuzzy_match(gold, []) == {}


def test_fixtures_have_gold_lists() -> None:
    fixtures = eval_summary.load_fixtures(FIXTURES)
    assert len(fixtures) >= 4
    langs = {u["lang"] for _, f in fixtures for u in f["utterances"]}
    assert langs == {"en", "hi", "or"}
    assert all(f["gold"]["decisions"] and f["gold"]["action_items"] for _, f in fixtures)


def test_eval_runs_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    out = tmp_path / "result.json"
    result = eval_summary.main([str(FIXTURES), "--judge", "fuzzy", "--json", str(out)])
    assert result["judge"] == "fuzzy"
    assert result["overall"]["grounding_pass_rate"] == 1.0
    assert json.loads(out.read_text(encoding="utf-8"))["provider"] == "mock"


async def test_llm_judge_with_scripted_mock() -> None:
    from app.services.llm.mock_client import MockLLMClient

    reply = json.dumps(
        {
            "matches": [
                {"gold_index": 0, "predicted_index": 1},
                {"gold_index": 1, "predicted_index": 9},
            ]
        }
    )
    llm = MockLLMClient(Settings(_env_file=None), "m", responses=[reply])
    assert await eval_summary.llm_match(llm, ["a", "b"], ["x", "y"]) == {0: 1}
    assert await eval_summary.llm_match(llm, [], ["x"]) == {}
