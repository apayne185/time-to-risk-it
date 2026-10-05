-- Activity aggregates per (player, landmark, look-back window).
--
-- Inputs (registered by ttr.features):
--   daily      cleaned daily aggregates (one row per player, day, product)
--   landmarks  (user_id, landmark) rows to score
--   windows    (window_days) look-back lengths
--
-- Leakage rule: only days strictly before the landmark are used (d.date < l.landmark).

WITH day_totals AS (
    -- One row per player-day across products.
    SELECT
        user_id,
        date,
        SUM(n_bets)                                                              AS n_bets,
        SUM(n_bets) FILTER (WHERE product_family IN ('fixed_odds', 'live_action')) AS sports_bets,
        SUM(n_bets) FILTER (WHERE product_family = 'live_action')                AS live_bets,
        SUM(turnover) FILTER (WHERE money_valid)                                 AS stakes,
        SUM(turnover) FILTER (WHERE product_family = 'live_action')              AS live_stakes,
        SUM(turnover) FILTER (WHERE product_family IN ('fixed_odds', 'live_action'))
                                                                                 AS sports_stakes,
        SUM(turnover) FILTER (WHERE product_family = 'casino')                   AS casino_stakes,
        -- Hold includes settlements of earlier bets (zero-bet rows), so it is a period result.
        SUM(hold) FILTER (WHERE money_valid)                                     AS net_loss,
        BOOL_OR(is_bet_day)                                                      AS bet_day,
        BOOL_OR(is_bet_day AND product_family = 'fixed_odds')                    AS fixed_day,
        BOOL_OR(is_bet_day AND product_family = 'live_action')                   AS live_day,
        BOOL_OR(is_bet_day AND product_family = 'casino')                        AS casino_day,
        COUNT(DISTINCT product_family) FILTER (WHERE is_bet_day)                 AS n_families
    FROM daily
    GROUP BY user_id, date
),

chasing AS (
    -- Loss chasing proxy: did the player stake more on a betting day that followed a losing one?
    SELECT
        user_id,
        date,
        LAG(net_loss) OVER w > 0                                   AS after_loss,
        COALESCE(stakes, 0) > COALESCE(LAG(stakes) OVER w, 0)      AS stake_up
    FROM day_totals
    WHERE bet_day
    WINDOW w AS (PARTITION BY user_id ORDER BY date)
)

SELECT
    l.user_id,
    l.landmark,
    w.window_days,
    COUNT(*) FILTER (WHERE d.bet_day)                             AS bet_days,
    COUNT(*) FILTER (WHERE d.fixed_day)                           AS fixed_days,
    COUNT(*) FILTER (WHERE d.live_day)                            AS live_days,
    COUNT(*) FILTER (WHERE d.casino_day)                          AS casino_days,
    COUNT(*) FILTER (WHERE d.fixed_day OR d.live_day)             AS sports_days,
    COALESCE(SUM(d.n_bets), 0)                                    AS n_bets,
    COALESCE(SUM(d.sports_bets), 0)                               AS sports_bets,
    COALESCE(SUM(d.live_bets), 0)                                 AS live_bets,
    COALESCE(SUM(d.stakes), 0)                                    AS stakes,
    COALESCE(SUM(d.sports_stakes), 0)                             AS sports_stakes,
    COALESCE(SUM(d.live_stakes), 0)                               AS live_stakes,
    COALESCE(SUM(d.casino_stakes), 0)                             AS casino_stakes,
    COALESCE(SUM(d.net_loss), 0)                                  AS net_loss,
    MAX(d.stakes) FILTER (WHERE d.bet_day)                        AS max_day_stakes,
    MAX(d.net_loss)                                               AS max_day_loss,
    STDDEV_SAMP(d.stakes) FILTER (WHERE d.bet_day)                AS sd_day_stakes,
    AVG(d.stakes) FILTER (WHERE d.bet_day)                        AS mean_day_stakes,
    MAX(d.n_families)                                             AS max_families,
    COUNT(*) FILTER (WHERE c.after_loss)                          AS days_after_loss,
    COUNT(*) FILTER (WHERE c.after_loss AND c.stake_up)           AS chase_days
FROM landmarks AS l
CROSS JOIN windows AS w
JOIN day_totals AS d
    ON  d.user_id = l.user_id
    AND d.date <  l.landmark
    AND d.date >= l.landmark - TO_DAYS(w.window_days)
LEFT JOIN chasing AS c
    ON  c.user_id = d.user_id
    AND c.date = d.date
GROUP BY l.user_id, l.landmark, w.window_days
