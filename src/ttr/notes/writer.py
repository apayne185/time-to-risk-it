"""Note writers: deterministic template, or Claude with the guard and a template fallback."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Literal, Protocol

from pydantic import ValidationError

from ttr.notes.facts import PlayerFacts
from ttr.notes.guard import check_note
from ttr.notes.schema import AgentNote, NoteResult
from ttr.notes.template import template_note

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5-5"
# Server-side fallback: if the model declines, Anthropic's recommended model for that refusal
# category re-runs the request inside the same call.
FALLBACK_BETA = "server-side-fallback-2026-07-01"

SYSTEM_PROMPT = """\
You write short internal notes for a responsible-gambling (RG) customer-service agent at an \
online betting operator. A risk model has placed a player on this month's outreach list; your \
note tells the agent why, in plain language, and suggests how to open a supportive conversation.

You receive the player's facts: the behaviours that raised their score most, each already \
stated with the player's own value. Write:
- summary: one or two sentences on the overall pattern.
- observations: one to three, each tied to exactly one provided feature name, restating that \
fact for the agent.
- conversation_opener: one warm, non-judgemental sentence the agent could say to the player, \
inviting them to talk and mentioning the account's limit-setting tools.

Use only the facts given. Do not add behaviours, numbers or context that are not in them, and \
keep each number exactly as written. Describe behaviour, not the person: no diagnoses or \
labels, no blame, no sanctions, nothing promotional. The opener is said to the player, so it \
must not mention models, scores, risk, flags or monitoring."""


class MessagesClient(Protocol):
    """The slice of ``anthropic.Anthropic`` the writer uses (lets tests inject a fake)."""

    @property
    def beta(self) -> Any: ...


class NoteWriter(Protocol):
    def write(self, facts: PlayerFacts) -> NoteResult: ...


class TemplateNoteWriter:
    def write(self, facts: PlayerFacts) -> NoteResult:
        return NoteResult(note=template_note(facts), source="template")


class ClaudeNoteWriter:
    def __init__(
        self,
        client: MessagesClient | None = None,
        model: str = DEFAULT_MODEL,
        effort: Literal["low", "medium", "high"] = "low",
        max_tokens: int = 16000,
    ) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self._schema = AgentNote.model_json_schema()

    def _fallback(self, facts: PlayerFacts, reason: list[str], **usage: Any) -> NoteResult:
        log.warning("note for player %s fell back to the template: %s", facts.player_id, reason)
        return NoteResult(
            note=template_note(facts),
            source="template_fallback",
            violations=reason,
            model=self.model,
            **usage,
        )

    def write(self, facts: PlayerFacts) -> NoteResult:
        import anthropic

        payload = {
            "on_outreach_list": facts.flagged,
            "facts": [{"feature": f.feature, "fact": f.statement} for f in facts.facts],
        }
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                output_config={
                    "effort": self.effort,
                    "format": {"type": "json_schema", "schema": self._schema},
                },
                betas=[FALLBACK_BETA],
                fallbacks="default",
            )
        except anthropic.APIError as exc:
            return self._fallback(facts, [f"api error: {type(exc).__name__}"])

        usage = {
            "input_tokens": getattr(response.usage, "input_tokens", None),
            "output_tokens": getattr(response.usage, "output_tokens", None),
        }
        if response.stop_reason == "refusal":
            return self._fallback(facts, ["model declined"], **usage)
        if response.stop_reason == "max_tokens":
            return self._fallback(facts, ["note truncated"], **usage)
        # With output_config.format the answer is the text block (thinking blocks may precede it).
        text = next((str(getattr(b, "text", "")) for b in response.content if b.type == "text"), "")
        try:
            note = AgentNote.model_validate_json(text)
        except ValidationError:
            return self._fallback(facts, ["output did not match the note schema"], **usage)
        problems = check_note(note, facts)
        if problems:
            return self._fallback(facts, problems, **usage)
        return NoteResult(
            note=note, source="claude", model=getattr(response, "model", None), **usage
        )


def writer_from_env() -> NoteWriter:
    """``TTR_NOTES_MODE=claude`` uses Claude (needs the llm extra and credentials); anything
    else uses the deterministic template."""
    if os.environ.get("TTR_NOTES_MODE", "template").lower() == "claude":
        return ClaudeNoteWriter(model=os.environ.get("TTR_NOTES_MODEL", DEFAULT_MODEL))
    return TemplateNoteWriter()
