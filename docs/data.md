# Data

## Source

*Behavioral Characteristics of Internet Gamblers Who Trigger Corporate Responsible Gambling
Interventions* — Gray, H. M., LaPlante, D. A., & Shaffer, H. J. (2012). *Psychology of Addictive
Behaviors*, 26(3), 527–535.

Published by [The Transparency Project](http://www.thetransparencyproject.org/), Division on
Addiction, Cambridge Health Alliance (a Harvard Medical School teaching affiliate), from a research
collaboration with bwin.party digital entertainment.

## Getting it

1. Register with The Transparency Project and accept its terms.
2. Download the codebook and the four **text** files for this dataset (Analytic dataset, Raw
   datasets I–III). The SAS/SPSS versions hold the same data.
3. Place them, unrenamed, in `data/raw/bwin_rg/`.
4. Run `make ingest`. Ingest verifies each file's SHA-256 against `configs/data.yaml` and fails
   fast on a corrupted or different file.

The data is licensed and is **never committed**: `data/` is gitignored and a pre-commit hook
rejects `.dat`, `.sav` and `.sas7bdat` files. Without it, `make demo` runs the same pipeline on
synthetic data.

## Study design

| Group | n | Definition |
|---|---|---|
| RG cases | 2,068 | Triggered a responsible-gambling intervention between 2008-11-02 and 2009-11-30 |
| Controls | 2,066 | First deposit on the same day as a case; no RG intervention in that window |

This is a matched case-control sample from a much larger player base, so the share of cases
(~50%) says nothing about real-world prevalence. Daily activity covers each player's entire
history, including activity **after** the RG event.

| Table | Rows | Grain |
|---|---|---|
| Demographics | 4,134 | player |
| Daily aggregates | 981,782 | player × day × product (2000-05-01 → 2010-11-10) |
| RG details | 2,068 | case (first and last RG event, first event and intervention type) |
| Analytic | 4,132 | player; whole-history summaries from the paper (validation only) |

## Known quirks (and where they are handled)

| Quirk | Handling |
|---|---|
| Missing values are `" "` in numeric columns and `"."` (SAS) in text columns | `ttr.data.raw` |
| Turnover/hold are invalid for third-party vendor products and set to missing | Enforced by schema; money features use products 1, 2, 4, 8, 17 only |
| 53 casino (product 8) rows lose more than they staked | Schema warning; kept (likely settlement of earlier bets) |
| 12,104 rows share a (player, day, product) key with different values | Split records: summed in cleaning. Reproduces the paper's totals exactly for every player with split records |
| 72,235 rows have zero bets | 23k are settlements of earlier bets (non-zero hold): kept, but not counted as betting days. 48.6k are empty: dropped |
| Country names mix country and site (`Greece.BAW`) and are truncated (`Bosnia and Herzego`) | Split into `country` and `site`, names repaired |
| 167 controls have no country, language, birth year or registration date | **Leakage risk**: missingness alone identifies them as controls. Never use missingness indicators |
| 3 cases have no RG date; 31 have no betting day before their RG event; 21 controls never bet | Excluded from modelling, reported in the cohort flow |
| 572 first RG events are account re-openings after an earlier closure | Prevalent, not incident, cases; excluded under the primary label (`configs/labels.yaml`) |

## Cohort flow

| Step | Players | Cases | Controls |
|---|---|---|---|
| All players | 4,134 | 2,068 | 2,066 |
| Excluding players with no betting activity | 4,113 | 2,068 | 2,045 |
| Excluding cases without an RG date | 4,110 | 2,065 | 2,045 |
| Excluding cases with no betting day before the event | **4,079** | **2,034** | **2,045** |

Reproduced by `make clean-data` (written to `data/processed/cohort_flow.json`).
