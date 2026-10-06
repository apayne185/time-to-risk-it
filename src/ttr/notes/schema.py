"""The note an RG agent sees, as a strict schema (also the LLM's structured-output format)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature: str = Field(description="Exactly one of the feature names from the facts provided")
    statement: str = Field(description="One plain sentence restating that fact for the agent")


class AgentNote(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(description="One or two sentences: why this player is on the list")
    observations: list[Observation] = Field(description="One to three observations")
    conversation_opener: str = Field(
        description="A warm, non-judgemental first sentence the agent could say to the player"
    )


class NoteResult(BaseModel):
    """A note plus how it was produced, for auditing."""

    note: AgentNote
    source: Literal["template", "claude", "template_fallback"]
    violations: list[str] = Field(default_factory=list)
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
