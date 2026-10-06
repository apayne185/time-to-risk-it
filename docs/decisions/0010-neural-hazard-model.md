# ADR 0010: A neural discrete-time hazard model, kept as a comparison

- **Status:** accepted
- **Date:** 2026-10-06

## Context

The survival models so far are a stratified Cox model and two gradient-boosted models (ADR 0003,
Phase 6). A neural model is the obvious next family to try: it can share structure across time
intervals, and it is what a team would reach for if richer inputs (sequences of daily activity)
became available later.

## Decision

1. **Discrete-time hazard network** (Gensheimer & Narasimhan, 2019): an MLP outputs one
   conditional hazard per 15-day interval up to 90 days, trained on the censored-data likelihood
   as masked binary cross-entropy, with AdamW, dropout and early stopping on validation loss. The
   30-day decision horizon is an interval edge, so event probabilities need no baseline-hazard
   estimate (unlike Breslow for XGBoost Cox). The early-stopped epoch count is reused for the
   refit on train + validation, the same way boosting rounds are.
2. **Same pipeline, same rules.** Same 47 features (median imputation, missingness flags,
   winsorising, scaling), same grid search on validation, same single test evaluation, same
   decision-layer evaluation and MLflow tracking. Explanations are gradient × input.
3. **Comparison only, decided in advance.** The decision model is chosen among families the
   serving image can load. CPU PyTorch adds ~700 MB, so the image does not include it, and
   `serving_families` was set to Cox and XGBoost before any neural-model result was seen.
4. **CPU wheels.** uv pulls PyTorch from the CPU index, so CI and laptops do not download the
   CUDA libraries the default Linux wheel brings.

## Results (test split, Aug–Oct 2009)

| Model | Within-month C (90-day) | AUC (30-day) | Reached @ 1% | @ 5% | @ 10% |
|---|---|---|---|---|---|
| Rule baseline | 0.693 | 0.715 | 7% | 22% | 36% |
| **PyTorch hazard net** (hidden 32, dropout 0.3) | 0.750 | 0.761 | 4% | 32% | 50% |
| Cox | 0.759 | 0.775 | 13% | 34% | 52% |
| XGBoost Cox | 0.768 | 0.785 | 11% | 43% | 62% |

The network clearly beats the rule baseline but trails Cox and XGBoost, with overlapping
confidence intervals on the C-index. It is weakest at the very top of the ranking (4% of cases
reached at a 1% budget). That is the expected outcome for ~6,000 rows of hand-built tabular
features, where gradient-boosted trees are hard to beat.

## Consequences

- No change to the decision model or the service.
- The network becomes worth revisiting with inputs a tree model cannot use directly, such as the
  raw daily activity sequence (a recurrent or convolutional encoder in place of the MLP), where
  its shared-interval structure can pay off.
