# ADR 0004: Demographics are not model features

- **Status:** accepted
- **Date:** 2026-10-04

## Context

The demographics table has age (year of birth), gender, country, site and language. They might
add predictive signal, but there are three problems with using them.

## Reasons

1. **Leakage.** Country, language, year of birth and registration date are missing for 167
   players, all of them controls. Missingness alone identifies them. Imputation does not fix this
   for tree models: an imputed constant (the median age, the most common country) becomes a spike
   that a model can learn to read as "control".
2. **Fairness and purpose.** The model decides who receives a responsible-gambling contact. A
   player's betting behaviour should drive that decision, not their gender, age or nationality.
   Risk scores that move with protected attributes are harder to justify to players, regulators
   and the RG team.
3. **Transferability.** Country and site mix in a 2005–2009 bwin sample says little about a
   current operator's customers in other markets. Behavioural features travel better.

## Decision

No demographic attribute, missingness indicator or derived quantity (age at landmark, country
group) is a model feature. `tests/test_features.py` fails if a feature name suggests one.

Demographics are kept in the player table for **subgroup evaluation**: discrimination and
calibration are reported by gender, age band and country (Phase 7), using players with known
values.

## Consequences

- Some predictive signal may be lost. If a reviewer wants to quantify it, an ablation with
  demographics (excluding the 167 players) can be added as a diagnostic, never as the shipped model.
- Subgroup metrics exclude the 167 players with missing demographics, and say so.
