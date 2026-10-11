"""Drive-level modeling table for the effectiveness analysis.

One row per eligible drive with: outcome class, style, pre-drive context (the confounders), and
team-quality controls. Everything in `CONTEXT_CONTROLS` is known *before* the first snap, so
adjusting for it is legitimate; anything that happens during the drive is not a control.
"""
import numpy as np
import pandas as pd
import polars as pl

from drivedna.drives.build import DRIVE_KEY, PROCESSED_DIR
from drivedna.models.cluster import eligible_drives

OUTCOME_CLASSES = ["Touchdown", "Field goal", "Punt", "Turnover", "Failed"]
OUTCOME_MAP = {
    "Touchdown": "Touchdown",
    "Field goal": "Field goal",
    "Punt": "Punt",
    "Turnover": "Turnover",
    "Opp touchdown": "Turnover",   # a turnover returned for a defensive score
    "Safety": "Turnover",
    "Turnover on downs": "Failed",
    "Missed field goal": "Failed",
}
STRENGTH_PRIOR_DRIVES = 40  # shrinkage: ~3 games of drives pull a team toward the league mean

CONTEXT_CONTROLS = [
    "start_yardline_100", "score_diff", "half_secs", "wp_start", "off_strength", "def_strength",
]
CATEGORICAL_CONTROLS = ["qtr", "season"]


def half_seconds_left(qtr: pd.Series, game_secs: pd.Series) -> pd.Series:
    """Seconds left in the current half (OT: seconds left in the period)."""
    return np.where(qtr <= 2, game_secs - 1800, game_secs).clip(min=0)


def team_strength(drives: pd.DataFrame, mode: str = "loo", prior_drives: int = STRENGTH_PRIOR_DRIVES) -> pd.DataFrame:
    """Shrunk points-per-drive for each team as offense and defense, relative to the league.

    mode="loo":   uses all of the team's *other* games that season (for adjustment).
    mode="prior": uses only earlier weeks that season (no future information; for prediction).
    Returns one row per (game_id, team) with `off_strength` / `def_strength` (points per drive
    above league average; defense is points allowed, so lower is better).
    """
    league = drives["points"].mean()
    out = {}
    for role, team_col in [("off", "posteam"), ("def", "defteam")]:
        g = (
            drives.groupby(["season", team_col, "week", "game_id"], as_index=False)
            .agg(s=("points", "sum"), n=("points", "size"))
            .sort_values(["season", team_col, "week"])
        )
        if mode == "loo":
            tot = g.groupby(["season", team_col])[["s", "n"]].transform("sum")
            s, n = tot["s"] - g["s"], tot["n"] - g["n"]
        elif mode == "prior":
            cum = g.groupby(["season", team_col])[["s", "n"]].cumsum()
            s, n = cum["s"] - g["s"], cum["n"] - g["n"]
        else:
            raise ValueError(mode)
        g[f"{role}_strength"] = (s + prior_drives * league) / (n + prior_drives) - league
        out[role] = g[["game_id", team_col, f"{role}_strength"]].rename(columns={team_col: "team"})
    return out["off"].merge(out["def"], on=["game_id", "team"], how="outer")


def build_model_table(strength_mode: str = "loo") -> pd.DataFrame:
    feats = pl.read_parquet(PROCESSED_DIR / "drive_features.parquet")
    styles = pl.read_parquet(PROCESSED_DIR / "drive_styles.parquet")
    kept = feats.filter(pl.col("keep")).to_pandas()
    strength = team_strength(kept, mode=strength_mode)

    elig = eligible_drives(feats).join(styles, on=DRIVE_KEY).to_pandas()
    elig["outcome"] = elig["result"].map(OUTCOME_MAP)
    elig["qtr"] = elig["qtr_start"].astype(int).clip(upper=5)
    elig["score_diff"] = elig["score_diff_start"].clip(-21, 21)
    elig["half_secs"] = half_seconds_left(elig["qtr_start"], elig["start_seconds_remaining"])
    # off_strength is for the offense (posteam); def_strength is for the *defense*, i.e. defteam
    elig = elig.merge(
        strength[["game_id", "team", "off_strength"]].rename(columns={"team": "posteam"}),
        on=["game_id", "posteam"], how="left",
    ).merge(
        strength[["game_id", "team", "def_strength"]].rename(columns={"team": "defteam"}),
        on=["game_id", "defteam"], how="left",
    )
    for c in ["off_strength", "def_strength"]:
        elig[c] = elig[c].fillna(0.0)
    elig["off_team_season"] = elig["posteam"] + "_" + elig["season"].astype(str)
    return elig


if __name__ == "__main__":
    t = build_model_table()
    print(t.shape)
    print(t["outcome"].value_counts(normalize=True).round(3))
