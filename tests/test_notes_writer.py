from __future__ import annotations

import json
import math
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest

from ttr.notes.facts import player_facts
from ttr.notes.writer import (
    DEFAULT_MODEL,
    FALLBACK_BETA,
    ClaudeNoteWriter,
    TemplateNoteWriter,
    writer_from_env,
)

FACTS = player_facts(
    7,
    0.012,
    True,
    [
        {"feature": "active_frac_30d", "value": 0.4, "description": "x"},
        {"feature": "log_max_day_stakes_365d", "value": math.log1p(215_000), "description": "x"},
    ],
)

GOOD = {
    "summary": "Betting has become frequent, with some very large single days.",
    "observations": [
        {"feature": "active_frac_30d", "statement": "Bet on 40% of days in the last 30 days."},
        {
            "feature": "log_max_day_stakes_365d",
            "statement": "Largest single-day stake in the last year was €215,000.",
        },
    ],
    "conversation_opener": "Hi, I'm checking in to see how your betting is going and to "
    "show you the limit tools on your account.",
}


class FakeClient:
    def __init__(
        self, text: str = "", stop_reason: str = "end_turn", error: Exception | None = None
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self._text, self._stop, self._error = text, stop_reason, error
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        return SimpleNamespace(
            stop_reason=self._stop,
            model=kwargs["model"],
            content=[
                SimpleNamespace(type="thinking", thinking=""),
                SimpleNamespace(type="text", text=self._text),
            ],
            usage=SimpleNamespace(input_tokens=420, output_tokens=95),
        )


def test_grounded_claude_note_is_returned() -> None:
    client = FakeClient(json.dumps(GOOD))
    result = ClaudeNoteWriter(client=client).write(FACTS)
    assert result.source == "claude" and result.violations == []
    assert result.note.observations[1].statement.endswith("€215,000.")
    assert (result.input_tokens, result.output_tokens) == (420, 95)

    call = client.calls[0]
    assert call["model"] == DEFAULT_MODEL
    assert call["fallbacks"] == "default" and call["betas"] == [FALLBACK_BETA]
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["output_config"]["effort"] == "low"
    sent = json.loads(call["messages"][0]["content"])
    assert {f["feature"] for f in sent["facts"]} == {"active_frac_30d", "log_max_day_stakes_365d"}


@pytest.mark.parametrize(
    ("client", "reason"),
    [
        (FakeClient(json.dumps({**GOOD, "summary": "Stakes tripled to €900,000."})), "numbers"),
        (FakeClient(json.dumps({**GOOD, "summary": "Classic problem gambler."})), "diagnostic"),
        (FakeClient("{not json"), "schema"),
        (FakeClient(json.dumps(GOOD), stop_reason="refusal"), "declined"),
        (FakeClient(json.dumps(GOOD), stop_reason="max_tokens"), "truncated"),
        (
            FakeClient(
                error=anthropic.APIConnectionError(
                    request=httpx2.Request("POST", "https://api.anthropic.com")
                )
            ),
            "api error",
        ),
    ],
)
def test_bad_or_failed_notes_fall_back_to_template(client: FakeClient, reason: str) -> None:
    result = ClaudeNoteWriter(client=client).write(FACTS)
    assert result.source == "template_fallback"
    assert any(reason in v for v in result.violations), result.violations
    assert result.note.observations  # the agent still gets a usable note


def test_writer_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TTR_NOTES_MODE", raising=False)
    assert isinstance(writer_from_env(), TemplateNoteWriter)
    assert TemplateNoteWriter().write(FACTS).source == "template"
