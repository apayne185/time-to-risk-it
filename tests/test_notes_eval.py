from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ttr.config import load_data_config, load_landmarks_config
from ttr.data.clean import build_players, clean_daily
from ttr.data.synthetic import SyntheticSpec, generate
from ttr.evaluate.calibration import PlattRecalibrator
from ttr.features import FEATURES, build_feature_table
from ttr.labels import get_definition
from ttr.landmarks import build_landmarks
from ttr.models.base import SurvivalData
from ttr.models.xgb import XGBCox
from ttr.notes.evaluate import evaluate_notes, evaluation_markdown
from ttr.notes.facts import PlayerFacts
from ttr.notes.schema import NoteResult
from ttr.notes.template import template_note
from ttr.notes.writer import TemplateNoteWriter
from ttr.serve.scoring import Scorer


class AlwaysFails:
    """A writer whose LLM output never passes the guard (exercises the fallback accounting)."""

    def write(self, facts: PlayerFacts) -> NoteResult:
        return NoteResult(
            note=template_note(facts),
            source="template_fallback",
            violations=["numbers not in the facts: [900.0]"],
            model="claude-opus-5-5",
            input_tokens=1000,
            output_tokens=200,
        )


def _world() -> tuple[pd.DataFrame, Scorer]:
    cfg = load_data_config()
    tables = generate(SyntheticSpec(n_pairs=80, seed=4), cfg)
    daily = clean_daily(tables["daily"], cfg)
    players = build_players(tables["demographics"], tables["rg"], daily)
    lm = build_landmarks(players, daily, cfg, load_landmarks_config(), get_definition("primary"))
    table = build_feature_table(lm, daily)
    model = XGBCox(num_boost_round=30).fit(SurvivalData.from_table(table, list(FEATURES)))
    return table, Scorer(
        {
            "family": "xgb_cox",
            "model": model,
            "features": list(FEATURES),
            "recalibrator": PlattRecalibrator(),
            "horizon_days": 30,
            "policy": {"capacity_share": 0.01, "threshold": 0.5},
        }
    )


def test_template_evaluation(tmp_path: Path) -> None:
    table, scorer = _world()
    ev = evaluate_notes(table, scorer, TemplateNoteWriter(), n=5, out=tmp_path / "n.jsonl")
    assert ev.sources == {"template": 5}
    lines = [json.loads(x) for x in (tmp_path / "n.jsonl").read_text().splitlines()]
    assert len(lines) == 5 and all(line["violations"] == [] for line in lines)
    assert "n/a (template)" in evaluation_markdown(ev, "template")


def test_fallback_and_cost_accounting() -> None:
    table, scorer = _world()
    ev = evaluate_notes(table, scorer, AlwaysFails(), n=4)
    assert ev.violations == {"numbers not in the facts": 4}
    assert ev.cost_usd() == 4 * (1000 * 4 / 1e6 + 200 * 20 / 1e6)
    text = evaluation_markdown(ev, "Claude (claude-opus-5-5)")
    assert "0% (0/4)" in text and "numbers not in the facts | 4" in text
