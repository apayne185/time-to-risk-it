# ADR 0001: Cleaning rules for daily aggregates

- **Status:** accepted
- **Date:** 2026-10-04

## Context

The daily-aggregates export has 12,104 rows that share a (player, day, product) key with
different values, and 72,235 rows with zero bets. The codebook does not explain either. Feature
values depend directly on how they are handled.

The paper's analytic dataset gives per-player totals (stakes, bets, betting days, net loss) for
fixed-odds and live-action betting. It can be used as ground truth for the rules.

## Options tested (fixed odds, share of players reproduced exactly)

| Rule | Stakes | Betting days |
|---|---|---|
| Sum all rows per key; count every row's day | 95.3% | 10.2% |
| Drop exact duplicate rows, then sum | 94.7% | 10.2% |
| Keep first row per key | 89.4% | 10.2% |
| **Sum all rows per key; betting day = `n_bets > 0`** | **95.3%** | **95.3%** |

Live action: 97.7% under the chosen rule.

All 276 players with split fixed-odds records are reproduced exactly only under summing. The
remaining ~5% of mismatches occur in players without split records; they are consistent with the
paper's figures coming from a slightly different extraction and are not addressed.

## Decision

1. Sum rows sharing a (player, day, product) key. Money columns stay missing for vendor products.
2. A betting day is a day with `n_bets > 0`.
3. Zero-bet rows with non-zero hold are settlements of earlier bets. Keep them: they count towards
   net loss but not towards activity.
4. Drop rows with no bets, no stake and no hold.

## Consequences

- Turnover is stakes placed that day; hold is results settled that day. Loss features must be
  computed over windows, not per bet day.
- `tests/test_realdata.py` pins the reproduction rates, so a regression in cleaning fails CI
  locally whenever the licensed data is present.
