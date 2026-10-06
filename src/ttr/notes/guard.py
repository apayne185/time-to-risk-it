"""Checks every note must pass before an agent sees it.

A note fails if it cites behaviour not in the facts, states a number that is not in the facts,
uses diagnostic, stigmatising, punitive or promotional language, exposes model internals in the
words meant for the player, or runs long. Any failure sends the caller to the template note.
"""

from __future__ import annotations

import re

from ttr.notes.facts import PlayerFacts
from ttr.notes.schema import AgentNote

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")

# Labels a customer-service agent must not attach to a player.
DIAGNOSTIC = (
    "addict",
    "addiction",
    "problem gambler",
    "compulsive",
    "pathological",
    "disorder",
    "gambling problem",
    "diagnos",
)
# The note supports a supportive contact: no sanctions, no selling.
PUNITIVE = ("ban ", "banned", "suspend", "close your account", "block your account", "penalt")
PROMOTIONAL = (
    "bonus",
    "free bet",
    "promotion",
    "promo",
    "offer",
    "odds boost",
    "deposit more",
    "vip",
)
# Words for the agent's eyes only; the opener is said to the player.
INTERNAL = ("model", "score", "algorithm", "flag", "risk", "probability", "predict", "system")

MAX_SUMMARY_WORDS, MAX_OPENER_WORDS, MAX_OBSERVATIONS = 60, 45, 3


def _numbers(text: str) -> set[float]:
    return {float(n.replace(",", "")) for n in _NUMBER.findall(text)}


def _contains(text: str, terms: tuple[str, ...]) -> list[str]:
    low = f" {text.lower()} "
    return [t.strip() for t in terms if t in low]


def check_note(note: AgentNote, facts: PlayerFacts) -> list[str]:
    """Violations (empty list = the note may be shown)."""
    problems: list[str] = []
    allowed_features = {f.feature for f in facts.facts}
    allowed_numbers = set().union(*(_numbers(f.statement) for f in facts.facts)) | {
        float(len(facts.facts))
    }

    if not 1 <= len(note.observations) <= MAX_OBSERVATIONS:
        problems.append(f"expected 1-{MAX_OBSERVATIONS} observations, got {len(note.observations)}")
    for obs in note.observations:
        if obs.feature not in allowed_features:
            problems.append(f"cites a behaviour not in the facts: {obs.feature}")

    text = " ".join(
        [note.summary, note.conversation_opener, *(o.statement for o in note.observations)]
    )
    unknown = sorted(_numbers(text) - allowed_numbers)
    if unknown:
        problems.append(f"numbers not in the facts: {unknown}")
    for label, terms in (
        ("diagnostic", DIAGNOSTIC),
        ("punitive", PUNITIVE),
        ("promotional", PROMOTIONAL),
    ):
        found = _contains(text, terms)
        if found:
            problems.append(f"{label} language: {found}")
    found = _contains(note.conversation_opener, INTERNAL)
    if found:
        problems.append(f"opener mentions internals: {found}")

    if len(note.summary.split()) > MAX_SUMMARY_WORDS:
        problems.append("summary too long")
    if len(note.conversation_opener.split()) > MAX_OPENER_WORDS:
        problems.append("opener too long")
    return problems
