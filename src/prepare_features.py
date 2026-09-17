import pandas as pd


def prepare_features(constructors, constructor_standings, races):
    """
    Prepare season-level features for predicting constructor champions.

    This version is tailored for datasets where constructor_standings
    already includes:
      - season (year)
      - round (race number in season)
    """

    # --- Normalize column names ---
    cs = constructor_standings.copy()
    cs.columns = cs.columns.str.strip().str.lower()

    # Expected columns (based on your error output)
    # ['constructor_id', 'points', 'position', 'round', 'season', 'wins']

    # --- Keep only final round of each season ---
    last_round_per_season = cs.groupby("season")["round"].transform("max")
    cs_final = cs[cs["round"] == last_round_per_season].copy()

    # --- Target: champion ---
    cs_final["position"] = pd.to_numeric(cs_final["position"], errors="coerce")
    cs_final["champion"] = (cs_final["position"] == 1).astype(int)

    # --- Sort for lag features ---
    cs_final = cs_final.sort_values(["constructor_id", "season"])

    # --- Lag features (previous season performance) ---
    cs_final["points_prev"] = cs_final.groupby("constructor_id")["points"].shift(1)
    cs_final["wins_prev"] = cs_final.groupby("constructor_id")["wins"].shift(1)
    cs_final["pos_prev"] = cs_final.groupby("constructor_id")["position"].shift(1)

    # Drop rows without previous-season data
    cs_final = cs_final.dropna(subset=["points_prev", "pos_prev"])

    # --- Relative features (scale-invariant across full and partial seasons) ---
    # points_pct_prev: fraction of the points leader's total — works at any round count
    cs_final["points_pct_prev"] = cs_final.groupby("season")["points_prev"].transform(
        lambda x: x / x.max() if x.max() > 0 else 0.0
    )
    # wins_pct_prev: fraction of the most-wins team's total
    cs_final["wins_pct_prev"] = cs_final.groupby("season")["wins_prev"].transform(
        lambda x: x / x.max() if x.max() > 0 else 0.0
    )
    # champ_prev: binary flag — were they champion the previous season?
    cs_final["champ_prev"] = (cs_final["pos_prev"] == 1).astype(int)

    # --- Feature matrix and labels ---
    X = cs_final[["points_prev", "wins_prev", "pos_prev",
                  "points_pct_prev", "wins_pct_prev", "champ_prev"]].copy()
    y = cs_final["champion"].copy()
    years = cs_final["season"].copy()
    constructor_ids = cs_final["constructor_id"].copy()

    return X, y, years, constructor_ids
