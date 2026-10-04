# ADR 0003: Landmark design, split-boundary censoring and the case-control artifact

- **Status:** accepted
- **Date:** 2026-10-04

## Context

An operator scores active customers on a schedule and acts on the riskiest. The model should be
trained and evaluated the same way: at a scoring date, using only the past, predicting the near
future. Two properties of this dataset complicate that: every player can be scored many times,
and the sample was drawn by outcome (all RG cases in Nov 2008 – Nov 2009, one matched control
each).

## Decision

**Landmarks.** The first day of each month from 2008-11-01 to 2009-10-01 (`configs/landmarks.yaml`).
At landmark `L` a player is scored if they deposited before `L`, have not yet had a qualifying RG
event (primary label, ADR 0002), and placed a bet in the 90 days before `L`. Features may only use
activity strictly before `L`.

**Outcome.** Days from `L` to the first qualifying RG event, censored at the earliest of `L + 90`
days, the end of the event window (controls are only known to be event-free inside it), and the
next split's first landmark.

**Temporal splits** by landmark: train Nov 2008 – Apr 2009, validation May – Jul 2009, test
Aug – Oct 2009.

**Split-boundary censoring.** The same players appear in every split. A training row at
2009-04-01 with a 90-day horizon would otherwise be labelled by events in May and June, which
are the validation period's outcomes. Instead of dropping late training landmarks, their follow-up
is censored at the next split's first landmark. Survival models use the partial follow-up, and no
label crosses a split boundary.

**Player folds.** A deterministic hash of `user_id` assigns each player to one of five folds for
user-disjoint cross-validation within a split.

## The case-control artifact

Because every case's event falls inside the window by construction, a case still at risk late in
the window is *guaranteed* an event soon. On the real data, the share of at-risk cases with an
event within 90 days rises from 24% at the 2008-11-01 landmark to 81% at 2009-08-01, and the
overall event rate in the test split (25–33%) is about double the training split's (6–16%). In
the real player population the hazard would not depend on the calendar like this.

Consequences:

1. **No calendar features.** Landmark month, days to window end and similar features would let a
   model learn the sampling design rather than player behaviour. (This reverses the original plan
   to include landmark month.)
2. **Absolute risk needs case-control weights.** Weighted by the inverse sampling fraction of
   controls, the at-risk population at each landmark is dominated by controls and the event rate
   becomes a population hazard. Calibration and decision curves use these weights (Phase 7).
3. **Ranking metrics are reported per landmark** as well as pooled, so a model is not rewarded
   for ranking late landmarks above early ones.
4. **Cox models are stratified by landmark**, which absorbs a separate baseline hazard per
   landmark.

## Coverage

1,885 players appear in at least one risk set (primary label: 10,378 rows, 1,680 events from 829
of 1,034 harm-onset cases). The other 205 harm-onset cases had not bet in the 90 days before any
landmark preceding their event: a model scoring active players cannot flag them, and the
evaluation says so rather than hiding them.
