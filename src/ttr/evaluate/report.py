"""Markdown evaluation report for the decision layer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ttr.config import DecisionConfig
from ttr.evaluate.figures import (
    NAMES,
    calibration_figure,
    capacity_figure,
    decision_curve_figure,
    importance_figure,
)
from ttr.evaluate.run import EvaluationResult


def _pct(x: float, digits: int = 0) -> str:
    return f"{x:.{digits}%}"


def _share(x: float) -> str:
    return f"{x:.1%}".replace(".0%", "%")


def _model_table(result: EvaluationResult, cfg: DecisionConfig) -> str:
    shares = cfg.capacity_shares
    head = (
        "| Model | OOF AUC | Test AUC | "
        + " | ".join(f"Reached @ {_share(s)}" for s in shares)
        + " | Test O/E (static) | Calibration slope |"
    )
    sep = "|---" * (5 + len(shares)) + "|"
    lines = [head, sep]
    for family, ev in result.evaluations.items():
        mark = " **(decision model)**" if family == result.decision_family else ""
        lines.append(
            f"| {NAMES[family]}{mark} | {ev.oof_auc:.3f} | {ev.test_auc:.3f} | "
            + " | ".join(_pct(ev.recall_at(s)) for s in shares)
            + f" | {ev.calibration.o_e_ratio:.2f} | {ev.calibration.calibration_slope:.2f} |"
        )
    return "\n".join(lines)


def _sensitivity_table(result: EvaluationResult, cfg: DecisionConfig) -> str:
    lines = [
        "| Assumed case rate (13 months) | Monthly event rate | Reached | Precision "
        "| Contacts per case reached |",
        "|---|---|---|---|---|",
    ]
    for r in result.sensitivity.to_dict(orient="records"):
        rate = float(r["population_case_rate"])
        mark = " (headline)" if rate == cfg.population_case_rate else ""
        lines.append(
            f"| {_pct(rate, 2)}{mark} | {_pct(float(r['monthly_event_rate']), 3)} | "
            f"{_pct(float(r['recall']))} | {_pct(float(r['precision']), 1)} | "
            f"{float(r['contacts_per_case']):.0f} |"
        )
    return "\n".join(lines)


def _subgroup_table(result: EvaluationResult) -> str:
    lines = ["| Attribute | Group | Rows | Events | AUC | O/E |", "|---|---|---|---|---|---|"]
    for r in result.subgroups.to_dict(orient="records"):
        events = int(r["events"])
        small = " ⚠" if events < 20 else ""
        lines.append(
            f"| {r['attribute']} | {r['group']} | {int(r['rows']):,} | {events}{small} | "
            f"{float(r['auc']):.2f} | {float(r['o_e']):.2f} |"
        )
    return "\n".join(lines)


def _broad_label_section(model_root: Path, primary: Path) -> str:
    broad, prim = model_root / "broad" / "summary.json", primary / "summary.json"
    if not broad.exists() or not prim.exists():
        return "_Run `make train` to train the broad-label models for this comparison._"
    b, p = json.loads(broad.read_text()), json.loads(prim.read_text())
    lines = ["| Model | Primary label: test C | Broad label: test C |", "|---|---|---|"]
    for family in p:
        if family in b:
            lines.append(
                f"| {NAMES.get(family, family)} | "
                f"{p[family]['test_c_within_landmark']:.3f} | "
                f"{b[family]['test_c_within_landmark']:.3f} |"
            )
    return "\n".join(lines)


def _dca_sentence(result: EvaluationResult) -> str:
    nb = result.evaluations[result.decision_family].net_benefit
    rule = result.evaluations["rule_baseline"].net_benefit
    useful = nb > np.maximum(result.contact_all, 0)
    if not useful.any():
        return "The model does not beat contacting everyone or no one at any threshold shown."
    lo, hi = result.thresholds[useful].min(), result.thresholds[useful].max()
    beats_rule = float((nb[useful] >= rule[useful]).mean())
    return (
        f"The model beats both contacting everyone and contacting no one for thresholds "
        f"from {lo:.2%} to {hi:.2%}, and is at or above the rule baseline at "
        f"{beats_rule:.0%} of those thresholds."
    )


def _calibration_sentence(oe: float) -> str:
    if oe > 1.2:
        return (
            f"A static calibration fitted on the training months **under-predicts the test "
            f"months**: observed 30-day events were {oe:.1f}× the predicted number. On the "
            f"real data, RG interventions per month rose through 2009, so the base rate "
            f"drifted while the ranking held up."
        )
    if oe < 0.8:
        return (
            f"A static calibration fitted on the training months **over-predicts the test "
            f"months**: observed 30-day events were {oe:.1f}× the predicted number."
        )
    return f"A static calibration holds on the test months (observed/expected {oe:.2f})."


def _subgroup_sentence(result: EvaluationResult) -> str:
    sub = result.subgroups
    off = sub[(sub["o_e"] > 1.5) | (sub["o_e"] < 0.67)]
    if off.empty:
        return "Calibration is within ±50% of observed rates in every subgroup."
    groups = ", ".join(
        f"{r.attribute} {r.group} (O/E {r.o_e:.1f}, {r.events} events)" for r in off.itertuples()
    )
    return (
        f"Ranking quality is broadly similar across groups, but calibration is off for: "
        f"{groups}. A deployment should monitor per-group calibration and, if a gap "
        f"persists, recalibrate per group rather than add demographic features."
    )


def _slope_text(slope: float) -> str:
    if slope < 0.9:
        return "predictions are too spread out (high risks too high, low risks too low)."
    if slope > 1.1:
        return "predictions are too compressed (high risks too low, low risks too high)."
    return "the spread of predictions is about right."


def write_report(
    result: EvaluationResult, cfg: DecisionConfig, out_dir: Path, model_dir: Path, label: str
) -> Path:
    fig_dir = out_dir / "figures"
    capacity_figure(result, fig_dir / "capacity.png")
    decision_curve_figure(result, fig_dir / "decision_curve.png")
    calibration_figure(result, fig_dir / "calibration.png")
    importance_figure(result, fig_dir / "drivers.png")

    ev = result.evaluations[result.decision_family]
    rule = result.evaluations["rule_baseline"]
    h = cfg.headline_capacity_share
    recall, rule_recall = ev.recall_at(h), rule.recall_at(h)
    ten = ev.recall_at(0.10) if 0.10 in cfg.capacity_shares else None
    sens = result.sensitivity.set_index("population_case_rate")["contacts_per_case"]
    cpc_lo, cpc_hi = float(sens.min()), float(sens.max())
    roll = result.calibration_rolling
    shifts = ", ".join(
        f"{r.landmark:%b}: {r.logit_shift:+.2f}" for r in result.rolling_shifts.itertuples()
    )
    name = NAMES[result.decision_family]
    ten_line = f" With a 10% budget it reaches {_pct(ten)}." if ten is not None else ""

    text = f"""# Decision layer: who should the RG team contact this month?

Generated by `ttr evaluate --label {label}`. Test split: {result.test_rows:,} player-months
(Aug–Oct 2009), {result.test_events} of them followed by a harm-onset RG intervention within
{cfg.horizon_days} days. Probabilities are on the active-player population scale, assuming
{_pct(cfg.population_case_rate, 1)} of active players become harm-onset cases over the 13-month
window (ADR 0005).

## Headline

**Contacting the {_share(h)} of active players with the highest {name} risk each month reaches
{_pct(recall)} of the players who trigger a responsible-gambling intervention in the next
30 days**, against {_pct(rule_recall)} for an operator-style rule on the same
budget.{ten_line}

Depending on the assumed population case rate, that is **{cpc_lo:.0f}–{cpc_hi:.0f} contacts for
every player reached** before their intervention. Whether that is worth it depends on what a
contact costs and what it achieves, which is a decision for the RG team; the decision curve below
makes that trade-off explicit.

![Capacity curve](figures/capacity.png)

## Models

Out-of-fold (OOF) metrics are on train + validation with player-disjoint folds; test metrics use
the final models once. AUC is computed within each monthly landmark (ADR 0003). "Reached" is the
share of next-month RG cases who were among the contacted players, at each monthly budget.

{_model_table(result, cfg)}

The decision model is the servable learned model (Cox or XGBoost; the serving image excludes
PyTorch) with the best OOF AUC. The PyTorch hazard net is evaluated for comparison only.

## Net benefit

A contact threshold of *t* means accepting (1 − *t*)/*t* unnecessary contacts to reach one
player who goes on to trigger an intervention (Vickers & Elkin, 2006). Net benefit counts reached
cases minus unnecessary contacts weighted by that exchange rate, per 10,000 active players.

![Decision curve](figures/decision_curve.png)

{_dca_sentence(result)}

## Calibration

{_calibration_sentence(ev.calibration.o_e_ratio)}

The production fix is a monthly intercept update: before each scoring run, shift the calibration
so last month's predictions match last month's observed rate (whose 30-day outcomes are complete
by then). This uses no future information and leaves the ranking unchanged. Monthly logit shifts:
{shifts}. With it, observed/expected is **{roll.o_e_ratio:.2f}**.

![Calibration](figures/calibration.png)

The calibration slope after updating is {roll.calibration_slope:.2f} (ideal 1.0):
{_slope_text(roll.calibration_slope)}
This does not affect who is contacted under a capacity policy, but probability-threshold
policies should be revisited once more recent data is available.

## Sensitivity to the assumed population case rate

The case-control sample cannot reveal how common RG cases are among all active players. Recall
barely moves with the assumption; precision and contacts per case scale with it.

{_sensitivity_table(result, cfg)}

## Subgroups

{name} on the test split, with the monthly intercept update. Demographics are not model inputs
(ADR 0004); 167 controls with missing demographics are excluded here. ⚠ marks groups with fewer
than 20 events, where estimates are noisy.

{_subgroup_table(result)}

{_subgroup_sentence(result)}

## What drives the score

Mean absolute contribution of each feature to the {name} risk score on the test split (exact
TreeSHAP values for tree models, linear contributions for Cox).

![Drivers](figures/drivers.png)

## Label sensitivity

Test within-landmark C under the primary label (harm onset, ADR 0002) and the broad label (every
RG event, as in the original paper):

{_broad_label_section(model_dir.parent, model_dir)}

## Limitations

- Data from 2005–2010, one operator, mainly German-speaking markets. Products, regulation and
  RG processes have changed since.
- The outcome is an operator RG intervention, not confirmed harm; a model trained on it partly
  learns what the operator's RG team noticed.
- Absolute risks depend on the assumed population case rate (shown above).
- Players who were not active in the 90 days before a scoring date are never scored: 205 of
  1,034 harm-onset cases fall in this gap (ADR 0003).
- The score supports a human review and a supportive contact. It must not be used to restrict,
  target or market to players.
"""
    path = out_dir / "evaluation.md"
    path.write_text(text)
    return path
