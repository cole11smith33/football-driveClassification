"""Defense scheme tendencies and results, per defense-season.

Two deliberately separate groups (the same style-vs-outcome discipline as the offense work):

  SCHEME   how the defense lines up: box count and blitz rate, each as an *excess* over what the
           situation and the offense's formation would call for (a defense facing three receivers
           shows a light box whatever its philosophy). Man-coverage share is added from 2018.
  RESULTS  what happened to it: EPA / success / explosives allowed, sack and pressure rate.
           Descriptive only; never used to define a scheme.

Tendencies are shrunk toward the league and, for use as a moderator of offensive outcomes,
computed leave-one-game-out so the game being predicted never contributes to its own opponent's
tendency.
"""
import numpy as np
import pandas as pd
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

NUM_CONTROLS = ["ydstogo", "yardline_100", "score_diff", "half_secs"]
CAT_CONTROLS = ["down", "offense_formation", "qtr", "season"]
SCHEME = ["box_excess", "blitz_excess", "big_blitz_excess"]
SCHEME_2018 = [*SCHEME, "man_excess"]
SHRINK_PLAYS = 150  # pseudo-plays of league-average behavior for the leave-one-game-out tendencies


def prepare(plays: pl.DataFrame) -> pd.DataFrame:
    """Pandas frame with the controls cleaned up and the play-level scheme indicators added."""
    d = plays.to_pandas()
    d["score_diff"] = d["score_differential"].clip(-21, 21)
    d["offense_formation"] = d["offense_formation"].fillna("UNK")
    d["qtr"] = d["qtr"].astype(int).clip(upper=5)
    d["is_pass"] = d["play_type"] == "pass"
    d["was_pressure"] = pd.to_numeric(d["was_pressure"], errors="coerce")
    d["blitz"] = (d["number_of_pass_rushers"] >= 5).astype(float).where(d["number_of_pass_rushers"].notna())
    d["big_blitz"] = (d["number_of_pass_rushers"] >= 6).astype(float).where(d["number_of_pass_rushers"].notna())
    d["man"] = (d["man_zone"] == "MAN_COVERAGE").astype(float).where(d["man_zone"].notna())
    d["explosive"] = np.where(d["is_pass"], d["yards_gained"] >= 20, d["yards_gained"] >= 10).astype(float)
    return d


def _pre() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("s", SplineTransformer(n_knots=5, degree=3)), ("z", StandardScaler())]), NUM_CONTROLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_CONTROLS),
    ])


def add_scheme_residuals(d: pd.DataFrame) -> pd.DataFrame:
    """Add box_resid / blitz_resid / big_blitz_resid / man_resid: actual minus situation-expected."""
    d = d.copy()
    specs = [
        ("box_resid", "defenders_in_box", d["defenders_in_box"].notna(), "linear"),
        ("blitz_resid", "blitz", d["is_pass"] & d["blitz"].notna(), "logit"),
        ("big_blitz_resid", "big_blitz", d["is_pass"] & d["big_blitz"].notna(), "logit"),
        ("man_resid", "man", d["is_pass"] & d["man"].notna(), "logit"),
    ]
    for out, target, mask, kind in specs:
        sub = d[mask]
        if sub.empty:
            d[out] = np.nan
            continue
        x = _pre().fit_transform(sub)
        if kind == "linear":
            pred = Ridge(alpha=1.0).fit(x, sub[target]).predict(x)
        elif sub[target].nunique() < 2:  # e.g. no 6+ rusher plays in this subset: nothing to model
            pred = np.full(len(sub), sub[target].mean())
        else:
            pred = LogisticRegression(C=5.0, max_iter=1000).fit(x, sub[target].astype(int)).predict_proba(x)[:, 1]
        d[out] = np.nan
        d.loc[sub.index, out] = sub[target].to_numpy() - pred
    return d


def defense_tendencies(d: pd.DataFrame) -> pd.DataFrame:
    """Per defense-season: scheme excesses (with n) and descriptive results."""
    g = d.groupby(["defteam", "season"])
    pass_g = d[d["is_pass"]].groupby(["defteam", "season"])
    run_g = d[~d["is_pass"]].groupby(["defteam", "season"])
    out = pd.DataFrame({
        "n_plays": g.size(),
        "n_pass": pass_g.size(),
        "box_excess": g["box_resid"].mean(),
        "blitz_excess": pass_g["blitz_resid"].mean(),
        "big_blitz_excess": pass_g["big_blitz_resid"].mean(),
        "man_excess": pass_g["man_resid"].mean(),
        # results (descriptive only)
        "epa_pass_allowed": pass_g["epa"].mean(),
        "epa_run_allowed": run_g["epa"].mean(),
        "success_pass_allowed": pass_g["success"].mean(),
        "success_run_allowed": run_g["success"].mean(),
        "explosive_allowed": g["explosive"].mean(),
        "sack_rate": pass_g["sack"].mean(),
        "pressure_rate": pass_g["was_pressure"].mean(),
    })
    return out.reset_index()


def loo_tendencies(d: pd.DataFrame, shrink: int = SHRINK_PLAYS) -> pd.DataFrame:
    """Leave-one-game-out, shrunk scheme tendencies for each (defteam, season, game_id).

    The tendency a drive faced = the defense's average residual over its *other* games that
    season, shrunk toward 0 (the league average, since residuals are centered).
    """
    rows = []
    for resid, mask_col in [("box_resid", None), ("blitz_resid", "is_pass"), ("big_blitz_resid", "is_pass"), ("man_resid", "is_pass")]:
        sub = d[d[resid].notna()]
        game = sub.groupby(["defteam", "season", "game_id"])[resid].agg(s="sum", n="size").reset_index()
        tot = game.groupby(["defteam", "season"])[["s", "n"]].transform("sum")
        s, n = tot["s"] - game["s"], tot["n"] - game["n"]
        game[resid.replace("_resid", "_loo")] = s / (n + shrink)  # shrinks toward 0
        rows.append(game.drop(columns=["s", "n"]).set_index(["defteam", "season", "game_id"]))
    return pd.concat(rows, axis=1).reset_index()


# ------------------------------------------------------------------------- reliability
def split_half_reliability(d: pd.DataFrame, cols=("box_resid", "blitz_resid", "big_blitz_resid", "man_resid"),
                           n_splits: int = 20, seed: int = 0) -> pd.DataFrame:
    """Spearman-Brown split-half reliability of each scheme tendency across defense-seasons."""
    rng = np.random.default_rng(seed)
    games = d[["defteam", "season", "game_id"]].drop_duplicates()
    corr = {c: [] for c in cols}
    for _ in range(n_splits):
        flag = games.assign(half=rng.integers(0, 2, len(games)))
        halves = d.merge(flag, on=["defteam", "season", "game_id"])
        for c in cols:
            sub = halves[halves[c].notna()]
            m = sub.groupby(["defteam", "season", "half"])[c].mean().unstack("half")
            corr[c].append(m[0].corr(m[1]))
    rows = []
    for c, v in corr.items():
        r = float(np.nanmean(v))
        rows.append({"tendency": c.replace("_resid", "_excess"), "half_corr": r, "reliability": 2 * r / (1 + r)})
    return pd.DataFrame(rows)


def year_to_year(tend: pd.DataFrame, cols=SCHEME_2018) -> pd.DataFrame:
    nxt = tend.assign(season=tend["season"] - 1)
    j = tend.merge(nxt, on=["defteam", "season"], suffixes=("", "_next"))
    return pd.DataFrame({"tendency": list(cols), "year_to_year_corr": [j[c].corr(j[c + "_next"]) for c in cols]})


# ------------------------------------------------------------------------- archetypes
def archetype_study(tend: pd.DataFrame, cols: list[str], ks=range(2, 7), n_boot: int = 30, seed: int = 0) -> pd.DataFrame:
    """Do defenses form discrete archetypes? Silhouette plus the two stability tests used for offense:
    refit on bootstrap resamples (compared with the full-data labels) and a model fit on 2016-2020
    vs. one fit on 2021-2025 (compared on the later seasons)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    X = tend[cols].to_numpy()
    rng = np.random.default_rng(seed)
    early, late = (tend["season"] < 2021).to_numpy(), (tend["season"] >= 2021).to_numpy()
    rows = []
    for k in ks:
        sc = StandardScaler().fit(X)
        ref = KMeans(k, n_init=10, random_state=0).fit(sc.transform(X))
        ref_lab = ref.labels_
        aris = []
        for b in range(n_boot):
            idx = rng.choice(len(X), len(X))
            s2 = StandardScaler().fit(X[idx])
            m = KMeans(k, n_init=10, random_state=b + 1).fit(s2.transform(X[idx]))
            aris.append(adjusted_rand_score(ref_lab, m.predict(s2.transform(X))))
        a = KMeans(k, n_init=10, random_state=0).fit(sc.transform(X[early]))
        b2 = KMeans(k, n_init=10, random_state=0).fit(sc.transform(X[late]))
        rows.append({
            "k": k, "silhouette": silhouette_score(sc.transform(X), ref_lab),
            "bootstrap_ari": float(np.mean(aris)), "frac_below_0_8": float((np.array(aris) < 0.8).mean()),
            "temporal_ari": adjusted_rand_score(a.predict(sc.transform(X[late])), b2.predict(sc.transform(X[late]))),
            "min_share": float(np.bincount(ref_lab).min() / len(ref_lab)),
        })
    return pd.DataFrame(rows)


def two_way_archetypes(tend: pd.DataFrame, cols: list[str] = SCHEME) -> pd.Series:
    """Label each defense-season 'Aggressive' (more blitz) or 'Light' (less) with a 2-cluster KMeans."""
    from sklearn.cluster import KMeans

    X = StandardScaler().fit_transform(tend[cols])
    lab = KMeans(2, n_init=10, random_state=0).fit_predict(X)
    aggressive = tend.groupby(lab)["blitz_excess"].mean().idxmax()
    return pd.Series(np.where(lab == aggressive, "Aggressive", "Light"), index=tend.index, name="archetype")
