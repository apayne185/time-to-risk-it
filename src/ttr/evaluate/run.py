"""Decision-layer evaluation: calibrated 30-day risk, net benefit, capacity, subgroups.

For each trained model family:

1. Out-of-fold (player-disjoint folds) 30-day predictions on train + validation, from the same
   configuration as the final model.
2. A weighted Platt recalibration fitted on those predictions with case-control weights (ADR 0005),
   mapping sample-scale probabilities to the active-player population.
3. The final model (refit on train + validation) scores the test split; its probabilities go
   through the recalibration and are evaluated once.

The decision model is the learned model with the highest out-of-fold within-landmark AUC: the
policy contacts the top of the ranking, and recalibration fixes the probability scale anyway.

Event rates drift over time (RG interventions per month rose through 2009), so a static
calibration under-predicts later months. ``rolling_recalibration`` shows the production fix:
before each landmark, re-estimate the intercept from the most recent month whose 30-day outcomes
are already known.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from ttr.config import DecisionConfig
from ttr.evaluate.calibration import (
    CalibrationSummary,
    PlattRecalibrator,
    apply_shift,
    calibration_summary,
    intercept_shift,
    reliability_table,
)
from ttr.evaluate.decision import capacity_curve, net_benefit, net_benefit_contact_all
from ttr.evaluate.subgroups import attach_groups, subgroup_table
from ttr.evaluate.weights import case_control_weights
from ttr.explain import contributions, global_importance
from ttr.models.base import SurvivalData, SurvivalModel
from ttr.train import _git_sha, build_model

log = logging.getLogger(__name__)


def horizon_outcome(table: pd.DataFrame, horizon: int) -> np.ndarray:
    """1 if a qualifying RG event happened within ``horizon`` days of the landmark."""
    if (table.loc[table["event"] == 0, "time"] < horizon).any():
        raise ValueError(f"some non-event rows are censored before {horizon} days")
    return ((table["event"] == 1) & (table["time"] <= horizon)).to_numpy(dtype=int)


def within_landmark_auc(y: np.ndarray, risk: np.ndarray, landmark: np.ndarray) -> float:
    scores, weights = [], []
    for lm in np.unique(landmark):
        m = landmark == lm
        if 0 < y[m].sum() < m.sum():
            scores.append(roc_auc_score(y[m], risk[m]))
            weights.append(y[m].sum())
    return float(np.average(scores, weights=weights))


@dataclass
class ModelEvaluation:
    family: str
    oof_log_loss: float
    oof_auc: float
    test_auc: float
    calibration: CalibrationSummary
    capacity: pd.DataFrame
    net_benefit: np.ndarray
    recalibrator: PlattRecalibrator
    oof_prob: np.ndarray
    test_prob: np.ndarray
    test_risk: np.ndarray
    model: SurvivalModel
    params: dict[str, Any]
    features: list[str]

    def capacity_at(self, share: float) -> dict[str, float]:
        row = self.capacity[np.isclose(self.capacity["share"], share)]
        if row.empty:
            raise KeyError(f"capacity share {share} was not evaluated")
        return {str(k): float(v) for k, v in row.iloc[0].items()}

    def recall_at(self, share: float) -> float:
        return self.capacity_at(share)["recall"]


def _rounds(model: SurvivalModel) -> int | None:
    best = getattr(model, "best_iteration_", None)
    return None if best is None else int(best) + 1


def oof_predictions(
    family: str,
    params: dict[str, Any],
    rounds: int | None,
    table: pd.DataFrame,
    features: list[str],
    horizon: int,
) -> np.ndarray:
    out = np.full(len(table), np.nan)
    for fold in sorted(table["fold"].unique()):
        held = (table["fold"] == fold).to_numpy()
        model = build_model(family, params, rounds).fit(
            SurvivalData.from_table(table[~held], features)
        )
        out[held] = model.predict_event_prob(table.loc[held, features], horizon)
    return out


def evaluate_family(
    bundle: dict[str, Any],
    trval: pd.DataFrame,
    test: pd.DataFrame,
    w_trval: np.ndarray,
    w_test: np.ndarray,
    cfg: DecisionConfig,
    thresholds: np.ndarray,
) -> ModelEvaluation:
    family = bundle["model"].name
    features: list[str] = bundle["features"]
    h = cfg.horizon_days
    y_trval, y_test = horizon_outcome(trval, h), horizon_outcome(test, h)

    oof = oof_predictions(family, bundle["params"], _rounds(bundle["model"]), trval, features, h)
    cal = PlattRecalibrator().fit(oof, y_trval, w_trval)
    oof_cal = cal.transform(oof)

    model: SurvivalModel = bundle["model"]
    risk = model.predict_risk(test[features])
    prob = cal.transform(model.predict_event_prob(test[features], h))
    landmark = test["landmark"].to_numpy()
    return ModelEvaluation(
        family=family,
        oof_log_loss=float(log_loss(y_trval, oof_cal, sample_weight=w_trval)),
        oof_auc=within_landmark_auc(y_trval, oof, trval["landmark"].to_numpy()),
        test_auc=within_landmark_auc(y_test, risk, landmark),
        calibration=calibration_summary(prob, y_test, w_test),
        capacity=capacity_curve(risk, y_test, w_test, landmark, cfg.capacity_shares),
        net_benefit=net_benefit(prob, y_test, w_test, thresholds),
        recalibrator=cal,
        oof_prob=oof,
        test_prob=prob,
        test_risk=risk,
        model=model,
        params=bundle["params"],
        features=features,
    )


def rate_sensitivity(
    ev: ModelEvaluation, trval: pd.DataFrame, test: pd.DataFrame, cfg: DecisionConfig
) -> pd.DataFrame:
    """Headline capacity and calibration across assumed population case rates."""
    h = cfg.horizon_days
    y_trval, y_test = horizon_outcome(trval, h), horizon_outcome(test, h)
    raw_test = ev.model.predict_event_prob(test[ev.features], h)
    rows = []
    for rate in cfg.population_case_rate_grid:
        w_tv, w_te = case_control_weights(trval, rate), case_control_weights(test, rate)
        cal = PlattRecalibrator().fit(ev.oof_prob, y_trval, w_tv)
        prob = cal.transform(raw_test)
        cap = capacity_curve(
            ev.test_risk, y_test, w_te, test["landmark"].to_numpy(), (cfg.headline_capacity_share,)
        ).iloc[0]
        summ = calibration_summary(prob, y_test, w_te)
        rows.append(
            {
                "population_case_rate": rate,
                "monthly_event_rate": summ.observed_rate,
                "recall": cap["recall"],
                "precision": cap["precision"],
                "contacts_per_case": cap["contacts_per_case"],
                "o_e": summ.o_e_ratio,
            }
        )
    return pd.DataFrame(rows)


def rolling_recalibration(
    ev: ModelEvaluation,
    trval: pd.DataFrame,
    test: pd.DataFrame,
    w_trval: np.ndarray,
    w_test: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Re-estimate the calibration intercept before each test landmark.

    At landmark L the latest outcomes already observed are those of landmark L - 1 month
    (its 30-day window closed by L). The shift that makes that month's predictions match its
    observed rate is applied to L's predictions. Ranking is unchanged.
    """
    feats = ev.features
    rows = pd.concat([trval.assign(_w=w_trval), test.assign(_w=w_test)], ignore_index=True)
    rows["_y"] = horizon_outcome(rows, horizon)
    rows["_p"] = ev.recalibrator.transform(ev.model.predict_event_prob(rows[feats], horizon))
    landmarks = sorted(rows["landmark"].unique())
    adjusted = ev.test_prob.copy()
    log_rows = []
    for lm in sorted(test["landmark"].unique()):
        prev = landmarks[landmarks.index(lm) - 1]
        if prev + pd.Timedelta(days=horizon) > lm:
            raise ValueError("previous landmark's outcomes are not yet observable")
        ref = rows[rows["landmark"] == prev]
        shift = intercept_shift(ref["_p"].to_numpy(), ref["_y"].to_numpy(), ref["_w"].to_numpy())
        m = (test["landmark"] == lm).to_numpy()
        adjusted[m] = apply_shift(ev.test_prob[m], shift)
        log_rows.append({"landmark": lm, "reference_landmark": prev, "logit_shift": shift})
    return adjusted, pd.DataFrame(log_rows)


def policy_threshold(ev: ModelEvaluation, w_trval: np.ndarray, share: float) -> float:
    """Calibrated 30-day risk above which a player falls in the top ``share`` of active players
    (weighted out-of-fold predictions on train + validation)."""
    p = ev.recalibrator.transform(ev.oof_prob)
    order = np.argsort(-p)
    cum = np.cumsum(w_trval[order]) / w_trval.sum()
    return float(p[order][np.searchsorted(cum, share)])


@dataclass
class EvaluationResult:
    evaluations: dict[str, ModelEvaluation]
    decision_family: str
    thresholds: np.ndarray
    contact_all: np.ndarray
    reliability: pd.DataFrame
    reliability_rolling: pd.DataFrame
    calibration_rolling: CalibrationSummary
    rolling_shifts: pd.DataFrame
    sensitivity: pd.DataFrame
    subgroups: pd.DataFrame
    importance: pd.Series
    threshold_at_headline: float
    test_rows: int
    test_events: int


def run_evaluation(
    table: pd.DataFrame, players: pd.DataFrame, model_dir: Path, cfg: DecisionConfig
) -> EvaluationResult:
    trval = table[table["split"].isin(["train", "validation"])].reset_index(drop=True)
    test = table[table["split"] == "test"].reset_index(drop=True)
    rate = cfg.population_case_rate
    w_trval, w_test = case_control_weights(trval, rate), case_control_weights(test, rate)
    t = cfg.dca_thresholds
    thresholds = np.geomspace(t.min, t.max, t.n)

    evaluations = {}
    for family in cfg.models:
        path = model_dir / f"{family}.joblib"
        if not path.exists():
            log.warning("no trained model at %s; skipping", path)
            continue
        log.info("evaluating %s", family)
        evaluations[family] = evaluate_family(
            joblib.load(path), trval, test, w_trval, w_test, cfg, thresholds
        )
    learned = {k: v for k, v in evaluations.items() if k in cfg.serving_families}
    if not learned:
        raise RuntimeError("no servable learned models to evaluate")
    decision = max(learned, key=lambda k: learned[k].oof_auc)
    ev = evaluations[decision]

    y_test = horizon_outcome(test, cfg.horizon_days)
    rolled, shifts = rolling_recalibration(ev, trval, test, w_trval, w_test, cfg.horizon_days)
    frame = attach_groups(test, players, cfg).assign(
        y=y_test, p=rolled, risk=ev.test_risk, w=w_test
    )
    contrib = contributions(ev.model, test[ev.features])
    return EvaluationResult(
        evaluations=evaluations,
        decision_family=decision,
        thresholds=thresholds,
        contact_all=net_benefit_contact_all(y_test, w_test, thresholds),
        reliability=reliability_table(ev.test_prob, y_test, w_test, n_bins=8),
        reliability_rolling=reliability_table(rolled, y_test, w_test, n_bins=8),
        calibration_rolling=calibration_summary(rolled, y_test, w_test),
        rolling_shifts=shifts,
        sensitivity=rate_sensitivity(ev, trval, test, cfg),
        subgroups=subgroup_table(frame),
        importance=global_importance(contrib),
        threshold_at_headline=policy_threshold(ev, w_trval, cfg.headline_capacity_share),
        test_rows=len(test),
        test_events=int(y_test.sum()),
    )


def save_decision_bundle(
    result: EvaluationResult, model_dir: Path, cfg: DecisionConfig, label: str
) -> Path:
    """Everything the scoring service needs: model, recalibration, policy threshold."""
    ev = result.evaluations[result.decision_family]
    bundle = {
        "family": result.decision_family,
        "model": ev.model,
        "features": ev.features,
        "recalibrator": ev.recalibrator,
        "horizon_days": cfg.horizon_days,
        "population_case_rate": cfg.population_case_rate,
        "policy": {
            "capacity_share": cfg.headline_capacity_share,
            "threshold": result.threshold_at_headline,
        },
        "label": label,
        "intercept_shift": float(result.rolling_shifts["logit_shift"].iloc[-1]),
        "git_sha": _git_sha(),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    path = model_dir / "decision_model.joblib"
    joblib.dump(bundle, path)
    summary = {
        "decision_model": result.decision_family,
        "policy": bundle["policy"],
        "calibration_rolling": asdict(result.calibration_rolling),
        "models": {
            k: {
                "test_auc_within_landmark": v.test_auc,
                "oof_auc_within_landmark": v.oof_auc,
                "oof_weighted_log_loss": v.oof_log_loss,
                "calibration": asdict(v.calibration),
                "capacity": v.capacity.to_dict(orient="records"),
            }
            for k, v in result.evaluations.items()
        },
    }
    (model_dir / "evaluation.json").write_text(json.dumps(summary, indent=2, default=float))
    return path
