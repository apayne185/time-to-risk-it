import math

import pytest

from ttr.notes.facts import humanise, player_facts
from ttr.notes.guard import check_note
from ttr.notes.schema import AgentNote, Observation
from ttr.notes.template import template_note

DRIVERS = [
    {
        "feature": "log_max_day_stakes_365d",
        "value": math.log1p(215_000),
        "description": "largest single-day stake",
    },
    {"feature": "active_frac_30d", "value": 0.4, "description": "share of days with a bet"},
    {"feature": "days_since_last_bet", "value": 2.0, "description": "days since last bet"},
]
FACTS = player_facts(7, 0.012, True, DRIVERS)


def _note(**kw: object) -> AgentNote:
    base: dict[str, object] = {
        "summary": "Betting has become very frequent and stakes are high.",
        "observations": [
            Observation(
                feature="active_frac_30d", statement="Bet on 40% of days in the last 30 days."
            )
        ],
        "conversation_opener": "Hi, I wanted to check in about how your betting is going.",
    }
    return AgentNote.model_validate({**base, **kw})


def test_humanise_restores_units() -> None:
    assert humanise("log_max_day_stakes_365d", math.log1p(215_000), "x") == (
        "Largest single-day stake in the last year: €215,000"
    )
    assert (
        humanise("active_frac_30d", 0.4, "x") == "Share of days with a bet in the last 30 days: 40%"
    )
    assert humanise("weird_feature", None, "Something") == "Something: not available"


def test_template_note_always_passes() -> None:
    note = template_note(FACTS)
    assert check_note(note, FACTS) == []
    assert [o.feature for o in note.observations] == [d["feature"] for d in DRIVERS]


def test_grounded_note_passes() -> None:
    assert check_note(_note(), FACTS) == []


@pytest.mark.parametrize(
    ("override", "problem"),
    [
        ({"summary": "Stakes rose 300% this month."}, "numbers not in the facts"),
        (
            {"observations": [Observation(feature="chase_rate_90d", statement="Chases losses.")]},
            "not in the facts",
        ),
        ({"summary": "Shows signs of gambling addiction."}, "diagnostic"),
        ({"conversation_opener": "Hi! Here is a bonus to say thanks."}, "promotional"),
        ({"conversation_opener": "Our risk model flagged your account."}, "internals"),
        ({"observations": []}, "observations"),
        ({"summary": "word " * 61}, "summary too long"),
    ],
)
def test_guard_rejects(override: dict[str, object], problem: str) -> None:
    problems = check_note(_note(**override), FACTS)
    assert any(problem in p for p in problems), problems
