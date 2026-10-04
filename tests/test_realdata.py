"""Checks against the licensed dataset. Skipped automatically when it is not present."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ttr.config import DataConfig, load_data_config
from ttr.data.ingest import ingest

CFG = load_data_config()
RAW = CFG.resolve(CFG.raw_dir)

pytestmark = [
    pytest.mark.realdata,
    pytest.mark.skipif(
        not (RAW / CFG.files.daily).exists(), reason="licensed bwin data not available"
    ),
    pytest.mark.filterwarnings("ignore::UserWarning"),
]


@pytest.fixture(scope="module")
def interim(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("interim")
    ingest(CFG, out_dir=out)
    return out


def test_counts_match_codebook(interim: Path) -> None:
    demo = pd.read_parquet(interim / "demographics.parquet")
    daily = pd.read_parquet(interim / "daily.parquet")
    rg = pd.read_parquet(interim / "rg.parquet")
    analytic = pd.read_parquet(interim / "analytic.parquet")
    assert len(demo) == 4134
    assert (demo["rg_case"] == 1).sum() == 2068
    assert len(daily) == 981_782
    assert daily["user_id"].nunique() == 4113
    assert len(rg) == 2068
    assert len(analytic) == 4132


def test_rg_dates_fall_in_event_window(interim: Path, data_cfg: DataConfig = CFG) -> None:
    rg = pd.read_parquet(interim / "rg.parquet")
    first = rg["rg_first_date"].dropna()
    assert first.min() >= pd.Timestamp(data_cfg.event_window.start)
    assert first.max() <= pd.Timestamp(data_cfg.event_window.end)
    assert rg["rg_first_date"].isna().sum() == 3


def test_missing_demographics_only_in_controls(interim: Path) -> None:
    """The leakage trap: demographic missingness alone identifies 167 controls."""
    demo = pd.read_parquet(interim / "demographics.parquet")
    missing = demo["country_name"].isna()
    assert missing.sum() == 167
    assert (demo.loc[missing, "rg_case"] == 0).all()


# --- cleaning rules, verified against the paper's analytic dataset ---------------------------


@pytest.fixture(scope="module")
def processed(interim: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    from ttr.data.clean import clean

    out = tmp_path_factory.mktemp("processed")
    clean(CFG, interim, out)
    return out


def _match_rate(processed: Path, interim: Path, product: int, suffix: str) -> pd.Series:
    daily = pd.read_parquet(processed / "daily.parquet")
    ref = pd.read_parquet(interim / "analytic.parquet").set_index("user_id")
    d = daily[daily["product_type"] == product]
    bets = d[d["is_bet_day"]].groupby("user_id")
    ours = pd.DataFrame(
        {
            "stakes": bets["turnover"].sum(),
            "bets": bets["n_bets"].sum(),
            "days": bets["date"].nunique(),
            "loss": d.groupby("user_id")["hold"].sum(),
        }
    )
    theirs = ref[
        [
            f"sum_stakes_{suffix}",
            f"sum_bets_{suffix}",
            f"bettingdays_{suffix}",
            f"net_loss_{suffix}",
        ]
    ].dropna()
    theirs.columns = ours.columns
    ours = ours.reindex(theirs.index)
    return ((ours - theirs).abs() <= 0.01 + 1e-6 * theirs.abs()).mean()


@pytest.mark.parametrize(
    ("product", "suffix", "floor"), [(1, "fixedodds", 0.95), (2, "liveaction", 0.97)]
)
def test_cleaning_reproduces_paper_totals(
    processed: Path, interim: Path, product: int, suffix: str, floor: float
) -> None:
    """Summed split records + bet days = n_bets > 0 reproduce the paper's per-player stakes,
    bet counts and betting days. The residual few percent are unrelated to split records."""
    rates = _match_rate(processed, interim, product, suffix)
    assert (rates[["stakes", "bets", "days"]] >= floor).all(), rates.to_dict()


def test_cohort_flow(processed: Path) -> None:
    import json

    flow = json.loads((processed / "cohort_flow.json").read_text())
    final = flow[-1]
    assert (final["n_cases"], final["n_controls"]) == (2034, 2045)


def test_all_27_paper_indices_reproduce(processed: Path, interim: Path) -> None:
    """Every index in the paper's analytic dataset is reproduced exactly for >= 95% of players."""
    from ttr.analysis.indices import betting_indices, index_columns

    ours = betting_indices(pd.read_parquet(processed / "daily.parquet"))
    ref = pd.read_parquet(interim / "analytic.parquet").set_index("user_id")
    rates = {}
    for col in index_columns():
        theirs = ref[col].dropna()
        mine = ours[col].reindex(theirs.index)
        rates[col] = ((mine - theirs).abs() <= 0.01 + 1e-4 * theirs.abs()).mean()
    assert min(rates.values()) >= 0.95, rates


def test_landmark_table_regression(processed: Path) -> None:
    """Pins the landmark design's output on the real data (ADR 0003)."""
    from ttr.config import load_landmarks_config
    from ttr.labels import get_definition
    from ttr.landmarks import build_landmarks

    lm = build_landmarks(
        pd.read_parquet(processed / "players.parquet"),
        pd.read_parquet(processed / "daily.parquet"),
        CFG,
        load_landmarks_config(),
        get_definition("primary"),
    )
    assert (len(lm), int(lm["event"].sum()), lm["user_id"].nunique()) == (10_378, 1_680, 1_885)
