# Features

Generated from `ttr.features.FEATURES` (`uv run ttr features --docs`). Every feature uses
only activity strictly before the landmark; `tests/test_features.py` enforces this.
Demographics are not features (ADR 0004).

| Feature | Description |
|---|---|
| `bet_days_30d` | Days with at least one bet in the last 30 days |
| `active_frac_30d` | Share of the last 30 days with a bet |
| `live_day_share_30d` | Share of betting days in the last 30 days with live-action bets |
| `casino_days_30d` | Days with casino play in the last 30 days |
| `sports_bets_per_day_30d` | Sportsbook bets per sportsbook betting day, last 30 days |
| `euros_per_sports_bet_30d` | Average sportsbook stake per bet (EUR), last 30 days |
| `log_stakes_30d` | log(1 + total stakes in EUR), last 30 days |
| `live_stake_share_30d` | Live-action share of sportsbook stakes, last 30 days |
| `pct_lost_30d` | Net loss as a share of stakes, last 30 days |
| `log_max_day_stakes_30d` | log(1 + largest single-day stake in EUR), last 30 days |
| `max_families_30d` | Most product families used on one day, last 30 days |
| `bet_days_90d` | Days with at least one bet in the last 90 days |
| `active_frac_90d` | Share of the last 90 days with a bet |
| `live_day_share_90d` | Share of betting days in the last 90 days with live-action bets |
| `casino_days_90d` | Days with casino play in the last 90 days |
| `sports_bets_per_day_90d` | Sportsbook bets per sportsbook betting day, last 90 days |
| `euros_per_sports_bet_90d` | Average sportsbook stake per bet (EUR), last 90 days |
| `log_stakes_90d` | log(1 + total stakes in EUR), last 90 days |
| `live_stake_share_90d` | Live-action share of sportsbook stakes, last 90 days |
| `pct_lost_90d` | Net loss as a share of stakes, last 90 days |
| `log_max_day_stakes_90d` | log(1 + largest single-day stake in EUR), last 90 days |
| `max_families_90d` | Most product families used on one day, last 90 days |
| `bet_days_365d` | Days with at least one bet in the last 365 days |
| `active_frac_365d` | Share of the last 365 days with a bet |
| `live_day_share_365d` | Share of betting days in the last 365 days with live-action bets |
| `casino_days_365d` | Days with casino play in the last 365 days |
| `sports_bets_per_day_365d` | Sportsbook bets per sportsbook betting day, last 365 days |
| `euros_per_sports_bet_365d` | Average sportsbook stake per bet (EUR), last 365 days |
| `log_stakes_365d` | log(1 + total stakes in EUR), last 365 days |
| `live_stake_share_365d` | Live-action share of sportsbook stakes, last 365 days |
| `pct_lost_365d` | Net loss as a share of stakes, last 365 days |
| `log_max_day_stakes_365d` | log(1 + largest single-day stake in EUR), last 365 days |
| `max_families_365d` | Most product families used on one day, last 365 days |
| `stake_cv_90d` | Coefficient of variation of daily stakes on betting days, last 90 days |
| `chase_rate_90d` | Share of betting days after a losing day on which stakes rose, last 90 days |
| `stake_cv_365d` | Coefficient of variation of daily stakes on betting days, last 365 days |
| `chase_rate_365d` | Share of betting days after a losing day on which stakes rose, last 365 days |
| `bet_days_trend_30_365` | Betting-day rate in the last 30 days relative to the last year |
| `bet_days_trend_30_90` | Betting-day rate in the last 30 days relative to the last 90 |
| `stakes_trend_30_365` | log ratio of last-30-day stakes to the last year's monthly average |
| `live_share_change_30_365` | Change in live-action day share, last 30 days vs last year |
| `days_since_last_bet` | Days since the most recent bet |
| `tenure_days` | Days since first deposit |
| `log_bet_days_all` | log(1 + betting days over the whole history) |
| `log_stakes_all` | log(1 + total stakes over the whole history, EUR) |
| `days_since_first_live` | Days since first live-action bet (missing if never) |
| `casino_started_30` | First ever casino play was in the last 30 days (0/1) |
