# ADR 0007: Online scoring from an activity stream

- **Status:** accepted
- **Date:** 2026-10-05

## Context

Batch scoring (ADR 0006) needs the feature pipeline to run over the warehouse each month. An
operator's settlement system already emits betting activity as it happens; scoring from that
stream removes the batch dependency and is how the posting's streaming platform would be used.
The risk is a second code path whose features drift from the offline ones the model was
validated on.

## Decisions

1. **Transport-agnostic core.** `StreamScorer` consumes `ActivityEvent`s and emits `ScoreEvent`s;
   Kafka is a thin adapter around it. The core is tested without a broker.
2. **Event time, not processing time.** Activity events are end-of-day aggregates. The first
   event dated on or after a landmark completes every earlier day, so the landmark fires and all
   eligible players are scored from state that holds only earlier activity: the offline leakage
   rule, enforced by construction. Late events are counted, kept for later landmarks, and do not
   rewrite scores already published.
3. **Same features, same code.** At each landmark the processor calls the offline feature SQL
   (`features_from_activity`). A test checks that every row of the offline landmark table gets
   an identical score from the stream.
4. **At-least-once delivery.** Offsets are committed only after the scores an event triggered
   are flushed, in batches of 500 and at every landmark. Duplicate scores after a restart are
   possible and idempotent for a consumer keyed on (player, landmark). Malformed messages go to a
   dead-letter topic.
5. **Ordering.** Events are keyed by player. The demo topic has one partition, so the stream is
   globally ordered; with several partitions the watermark becomes the minimum event time across
   assigned partitions.

## Consequences

- State holds each player's full history so the offline SQL can run unchanged. That is fine for
  this dataset; at production scale it becomes a 365-day window plus lifetime accumulators in a
  feature store, and the parity test is the contract that implementation must meet.
- Eligibility here is activity-based only. Excluding players with an earlier RG intervention
  needs the operator's intervention log joined to the stream.
- CI runs the whole path against a real Redpanda broker: replay, score, and a check that scores
  reach `rg.scores`.
