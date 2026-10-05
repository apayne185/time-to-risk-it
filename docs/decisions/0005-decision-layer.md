# ADR 0005: Decision layer — 30-day horizon, case-control weights, monthly recalibration

- **Status:** accepted
- **Date:** 2026-10-05

## Context

The models rank players well (ADR 0003, Phase 6). An RG team needs more than a ranking: how many
players to contact each month, what that buys, and probabilities it can reason about. Three
problems stand between the survival models and those answers.

## Decisions

### 1. Decision horizon: 30 days

The team scores players monthly, so the actionable question is "who is at risk before the next
scoring run?". Every landmark row has at least 30 days of follow-up (the last training landmark
is censored at exactly 30 days by the split boundary), so the 30-day outcome is fully observed:
calibration, net benefit and capacity metrics need no censoring adjustment. The survival models
still train on the full 90-day follow-up.

### 2. Case-control weights for the population scale

Cases are ~45% of sampled players but a small fraction of real active players. Each control
player gets weight `w = (n_cases / n_controls) * (1 - pi) / pi` so the weighted case share equals
an assumed population rate `pi` (`configs/decision.yaml`, default 0.5% over the 13-month window).
`pi` cannot be estimated from this sample, so every probability-scale result is also shown for
`pi` in {0.25%, 0.5%, 1%, 2%}. Ranking metrics are unaffected by the weights.

### 3. Weighted Platt recalibration on out-of-fold predictions

Survival-model probabilities are on the sample scale. A two-parameter logistic recalibration is
fitted with the case-control weights on out-of-fold predictions (player-disjoint folds) over train
+ validation, then applied to the final model's test predictions.

### 4. Monthly intercept update

On the real data the static calibration under-predicts the test months by a factor of 2.6: RG
interventions per month rose through 2009 (121 first events in Nov 2008, 198 in Oct 2009). The
ranking held up; the base rate moved. Before each scoring run the intercept is re-estimated so
the previous month's predictions match its observed rate. That month's 30-day outcomes are
complete by the next landmark, so no future information is used. Observed/expected on test goes
from 2.59 to 1.12.

### 5. Model selection by out-of-fold AUC

The capacity policy contacts the top of the ranking, and recalibration fixes the probability scale,
so the decision model is the learned model with the best out-of-fold within-landmark AUC. It is
chosen before looking at test results, and the report keeps it even where another model happens
to do better on test.

## Consequences

- The scoring service ships the model, the recalibration, the latest intercept shift and a policy
  threshold (`models/<label>/decision_model.joblib`).
- Monitoring must track the observed monthly event rate: it drives the intercept update.
- The calibration slope (0.70 on test) shows the model's risks are too spread out; capacity
  policies are unaffected, probability-threshold policies should be revisited with newer data.
