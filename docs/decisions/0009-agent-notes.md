# ADR 0009: LLM-written notes for RG agents, behind a guard

- **Status:** accepted
- **Date:** 2026-10-06

## Context

A score with feature contributions (ADR 0006) is precise but not something a customer-service
agent can act on quickly: "`log_max_day_stakes_365d` +0.18" means little on a phone call. An LLM
can turn the drivers into a readable note and a first sentence for the conversation. It can also
invent numbers, attach diagnostic labels to a person, or slip into promotional language, all of
which are unacceptable in responsible-gambling outreach.

## Decisions

1. **Facts are computed, not generated.** `notes/facts.py` turns each top driver into a sentence
   with human units (log stakes back to euros, shares to percentages). The model only rephrases
   these facts and is told to keep every number exactly as written.
2. **Structured output.** Claude returns an `AgentNote` (summary, one to three observations each
   tied to a driver, an opener) through `output_config.format` with the model's JSON schema.
3. **A guard on every note.** Observations must cite provided drivers; every number in the text
   must appear in the facts; no diagnostic, punitive or promotional language; the opener (said to
   the player) must not mention models, scores, risk or flags; length limits. Failing notes are
   replaced by the template, and the reason is kept on the result.
4. **Template first.** A deterministic note is the default everywhere (`TTR_NOTES_MODE=template`)
   and the fallback for refusals, truncation, schema errors, guard failures and API errors, so
   outreach never depends on an external API being up.
5. **Model and safety settings.** Claude Opus 5.5 at low effort (short summarisation), with
   server-side `fallbacks: "default"` so a safety decline is retried on Anthropic's recommended
   model rather than returned empty.
6. **Measured, not assumed.** `ttr notes eval` reports the guard pass rate, violation types,
   tokens and cost on the highest-risk players; per-player notes stay out of git.

## Consequences

- An agent never sees an unguarded LLM note, at the cost of sometimes seeing a plainer one.
- The guard is conservative: a correct note that rounds a number differently fails and falls back.
  The pass rate from `ttr notes eval` shows whether that is a real cost.
- Notes add latency and per-call cost when enabled on the API (`?notes=true`); batch generation
  for the monthly list is the cheaper way to use them.
