# ADR 0006: Scoring service, batch scoring and monitoring

- **Status:** accepted
- **Date:** 2026-10-05

## Context

The decision layer (ADR 0005) produces a bundle: model, recalibration, monthly intercept shift
and a capacity-policy threshold. The RG team needs scores in two ways: a monthly batch over all
active players, and on demand for a single player (for example when an agent opens an account).

## Decisions

1. **One scoring core.** `ttr.serve.scoring.Scorer` wraps the bundle. The HTTP service and
   `ttr score` (batch) both use it, so online and batch scores cannot diverge. A test checks the
   API's probabilities equal the batch scorer's.
2. **Two input contracts.** `POST /score` takes precomputed features (the feature pipeline's
   output, validated against the model's feature list). `POST /score/activity` takes raw daily
   activity and computes features with the same cleaning and SQL as the offline pipeline; a test
   checks the online features equal the offline ones exactly.
3. **Explanations in every response.** Each score carries its top three risk-increasing drivers
   with plain-language descriptions, so an agent sees *why* a player was flagged.
4. **Model bundle mounted, not baked in.** It is trained on licensed data. The image contains
   code, configs and SQL only; `TTR_MODEL_BUNDLE` points at the mounted bundle.
5. **Lean serving image.** MLflow and matplotlib are a `train` extra. The image installs the core
   dependencies only; CI asserts MLflow is absent from it.
6. **Monitoring as a monthly job.** `ttr monitor` reports PSI per feature and on the risk score
   against the training data, and computes the intercept shift from the most recent month whose
   30-day outcomes are complete. `--write-shift` stores it in the bundle; the service reports the
   active shift at `GET /model`.

## Consequences

- Serving stays stateless: the bundle is the only artifact, so rollback means mounting the
  previous bundle.
- The intercept shift needs outcome data with a one-month lag, so monitoring depends on the RG
  team's intervention log being joined back to scored players.
- On the real data all 47 features are stable between the training and test periods (max PSI
  0.08) and so is the risk score (PSI 0.04): the calibration drift in ADR 0005 is a change in
  the outcome rate, not in player behaviour.
