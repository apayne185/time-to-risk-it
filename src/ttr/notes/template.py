"""Deterministic note: the default, and the fallback whenever an LLM note fails the guard."""

from __future__ import annotations

from ttr.notes.facts import PlayerFacts
from ttr.notes.schema import AgentNote, Observation

OPENER = (
    "Hi, I'm getting in touch to check how things are going with your betting, and to "
    "make sure you know about the tools for setting limits on your account."
)


def template_note(facts: PlayerFacts) -> AgentNote:
    shown = facts.facts[:3]
    lead = "On this month's outreach list" if facts.flagged else "Not on this month's list"
    return AgentNote(
        summary=f"{lead}. The behaviours below raised this player's 30-day risk the most.",
        observations=[Observation(feature=f.feature, statement=f.statement) for f in shown],
        conversation_opener=OPENER,
    )
