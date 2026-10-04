# ADR 0002: Primary outcome excludes accounts that were already closed

- **Status:** accepted
- **Date:** 2026-10-04

## Context

The dataset labels a player a "case" if they triggered *any* responsible-gambling (RG) event
between Nov 2008 and Nov 2009. The first event's type and the customer-service intervention vary
widely (codebook Appendices 2–3). An early-warning model needs an outcome that marks the
**onset** of harm, not every customer-service contact.

## Evidence

Aligning each case on its first RG event and measuring the share who bet each month
(`ttr eda`, real data):

| First RG event (intervention code) | n | -9 | -7 | -6 | -3 | -1 | 0 |
|---|---|---|---|---|---|---|---|
| Account re-opened after closure (2) | 559 | 21% | 18% | 3% | 3% | 6% | 90% |
| Account remains closed (16) | 282 | 36% | 32% | 7% | 5% | 7% | 2% |
| Harm-onset cases (primary label) | 1,034 | 42% | 47% | 46% | 57% | 88% | 68% |

Both groups almost stop betting exactly six months before their event, consistent with a
six-month closure or self-exclusion ending in a re-opening request. Their harm began **before**
the observation window: they are prevalent cases. Treated as events, they would teach a model
that a *drop* in betting signals risk, the opposite of what incident cases show.

## Decision

The primary label (`configs/labels.yaml`) counts first RG events of types 1, 2, 3, 4, 7, 9, 10
and excludes interventions 2 (re-opened) and 16 (remains closed). Excluded cases leave the risk
set; they are not relabelled as controls. Non-harm event types (6 heavy complainer, 8 minor,
11 request for a higher limit, 12/13 unknown) are also excluded.

The broad label (every RG event, as in the paper) is kept for sensitivity analysis.

## Consequences

- Fewer events: 1,027 harm-onset cases with pre-event sports or casino betting, against 1,897
  controls, in the replication cohort (2,021 cases under the broad label).
- Results are reported under both labels.
- The finding should be confirmed by bwin.party's RG team in a real deployment: the six-month
  pattern is inferred from activity, not documented in the codebook.
