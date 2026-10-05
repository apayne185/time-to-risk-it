-- Whole-history (before the landmark) facts per (player, landmark).
-- Leakage rule: only days strictly before the landmark are used.

SELECT
    l.user_id,
    l.landmark,
    COUNT(DISTINCT d.date) FILTER (WHERE d.is_bet_day)                         AS bet_days_all,
    COALESCE(SUM(d.turnover) FILTER (WHERE d.money_valid), 0)                  AS stakes_all,
    MIN(d.date) FILTER (WHERE d.is_bet_day)                                    AS first_bet_date,
    MIN(d.date) FILTER (WHERE d.is_bet_day AND d.product_family = 'live_action') AS first_live_date,
    MIN(d.date) FILTER (WHERE d.is_bet_day AND d.product_family = 'casino')    AS first_casino_date
FROM landmarks AS l
JOIN daily AS d
    ON  d.user_id = l.user_id
    AND d.date < l.landmark
GROUP BY l.user_id, l.landmark
