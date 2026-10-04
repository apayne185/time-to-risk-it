import pandas as pd

from ttr.analysis.trajectories import monthly_trajectories

TS = pd.Timestamp


def test_active_share_never_exceeds_one_for_new_depositors() -> None:
    """A player who first deposits mid-month and bets that month is exposed in that month."""
    players = pd.DataFrame(
        {
            "user_id": [1, 2],
            "rg_case": [1, 0],
            "first_deposit_date": [TS("2009-01-20"), TS("2009-01-20")],
        }
    )
    index_dates = pd.Series([TS("2009-02-01"), TS("2009-02-01")], index=[1, 2])
    daily = pd.DataFrame(
        {
            "user_id": [1],
            "date": [TS("2009-01-25")],
            "product_type": [2],
            "n_bets": [3],
            "turnover": [10.0],
            "hold": [5.0],
            "is_bet_day": [True],
            "money_valid": [True],
            "product_family": ["live_action"],
        }
    )
    traj = monthly_trajectories(daily, players, index_dates, months_before=2, months_after=1)
    case_m1 = traj[(traj["group"] == "case") & (traj["month"] == -1)].iloc[0]
    assert case_m1["exposed"] == 1
    assert case_m1["active_share"] == 1.0
    assert traj["active_share"].dropna().le(1).all()
    # nobody had deposited two months before the index date
    assert traj.loc[traj["month"] == -2, "exposed"].eq(0).all()
