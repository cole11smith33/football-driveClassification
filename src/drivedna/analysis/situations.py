"""Which call types work in which situations, and how pass depth interacts with the defensive look.

Estimand (same logic as the drive-level work): for plays that actually used call type c, the
predicted EPA (or success) under c minus the predicted value under the share-weighted mix of calls,
in the same down / distance / field position / score / time. Fit separately inside each situation.

Calls are call-based tokens only (run inside / outside, pass short / mid / deep). Sacks and
scrambles are results of a dropback rather than calls, so they are excluded; this means the
pass-depth numbers describe *thrown* passes, and sack rate against the look is reported separately.
Defenses also react to the offense's pre-snap look, so these are associations, not causal effects.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

CALLS = ["R_in", "R_out", "P_short", "P_mid", "P_deep"]
DEPTHS = ["P_short", "P_mid", "P_deep"]
NUM = ["ydstogo", "yardline_100", "score_diff", "half_secs"]
CAT = ["down", "qtr", "season"]

SITUATIONS = {
    "all plays": lambda d: pd.Series(True, index=d.index),
    "red zone (inside the 20)": lambda d: d["yardline_100"] <= 20,
    "two-minute drill (end of half)": lambda d: (d["half_secs"] <= 120) & d["qtr"].isin([2, 4]),
    "4th quarter, leading by 1-16": lambda d: (d["qtr"] == 4) & d["score_diff"].between(1, 16),
    "4th quarter, trailing by 1-16": lambda d: (d["qtr"] == 4) & d["score_diff"].between(-16, -1),
}


def _design(calls: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("s", SplineTransformer(n_knots=4, degree=3)), ("z", StandardScaler())]), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
        ("call", OneHotEncoder(categories=[calls], drop="first"), ["call"]),
    ])


def adjusted_call_effects(d: pd.DataFrame, calls: list[str], target: str = "epa", alpha: float = 5.0) -> pd.DataFrame:
    """Point estimates: effect of each call vs. the typical mix, for plays that used it."""
    d = d[d["call"].isin(calls)]
    pre = _design(calls).fit(d)
    model = Ridge(alpha=alpha).fit(pre.transform(d), d[target])
    shares = d["call"].value_counts(normalize=True).reindex(calls).fillna(0.0)
    under = {c: model.predict(pre.transform(d.assign(call=c))) for c in calls}
    typical = sum(shares[c] * under[c] for c in calls)
    rows = []
    for c in calls:
        own = (d["call"] == c).to_numpy()
        if own.any():
            rows.append({"call": c, "n": int(own.sum()), "share": float(shares[c]),
                         "raw_mean": float(d.loc[own, target].mean()),
                         "effect": float((under[c][own] - typical[own]).mean())})
    return pd.DataFrame(rows)


def bootstrap_call_effects(d: pd.DataFrame, calls: list[str], target: str = "epa", n_boot: int = 60,
                           seed: int = 0) -> pd.DataFrame:
    """Point estimate plus 95% CI, resampling whole games."""
    d = d[d["call"].isin(calls)].reset_index(drop=True)
    point = adjusted_call_effects(d, calls, target)
    by_game = d.groupby("game_id").indices
    game_ids = np.array(list(by_game))
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(n_boot):
        idx = np.concatenate([by_game[g] for g in rng.choice(game_ids, len(game_ids))])
        reps.append(adjusted_call_effects(d.iloc[idx].reset_index(drop=True), calls, target).set_index("call")["effect"])
    boot = pd.concat(reps, axis=1)
    out = point.set_index("call")
    out["lo"], out["hi"] = boot.quantile(0.025, axis=1), boot.quantile(0.975, axis=1)
    out["significant"] = (out["lo"] > 0) | (out["hi"] < 0)
    return out.reset_index()


def situation_table(plays: pd.DataFrame, target: str = "epa", n_boot: int = 60) -> pd.DataFrame:
    """Call effects inside each situation, long format."""
    rows = []
    for name, mask in SITUATIONS.items():
        sub = plays[mask(plays)]
        res = bootstrap_call_effects(sub, CALLS, target, n_boot)
        res.insert(0, "situation", name)
        res["n_plays"] = int(sub["call"].isin(CALLS).sum())
        rows.append(res)
    return pd.concat(rows, ignore_index=True)


def call_mix(plays: pd.DataFrame) -> pd.DataFrame:
    """Share of each call type within each situation."""
    rows = []
    for name, mask in SITUATIONS.items():
        sub = plays[mask(plays) & plays["call"].isin(CALLS)]
        rows.append(sub["call"].value_counts(normalize=True).reindex(CALLS).fillna(0.0).rename(name))
    return pd.concat(rows, axis=1).T


# ------------------------------------------------------------------ pass depth vs. the defensive look
def add_looks(plays: pd.DataFrame) -> pd.DataFrame:
    p = plays.copy()
    p["look_blitz"] = np.where(p["number_of_pass_rushers"].isna(), np.nan, (p["number_of_pass_rushers"] >= 5).astype(float))
    p["look_man"] = np.where(p["man_zone"].isna(), np.nan, (p["man_zone"] == "MAN_COVERAGE").astype(float))
    return p


def depth_by_look(plays: pd.DataFrame, look: str, labels: dict, target: str = "epa", n_boot: int = 60) -> pd.DataFrame:
    """Adjusted pass-depth effects within each value of a defensive look (thrown passes only)."""
    p = plays[plays["call"].isin(DEPTHS) & plays[look].notna()]
    rows = []
    for value, name in labels.items():
        sub = p[p[look] == value]
        res = bootstrap_call_effects(sub, DEPTHS, target, n_boot)
        res.insert(0, "look", name)
        res["n_attempts"] = len(sub)
        rows.append(res)
    return pd.concat(rows, ignore_index=True)


def sack_rate_by_look(plays: pd.DataFrame, look: str, labels: dict) -> pd.DataFrame:
    """Sacks per dropback (thrown passes + sacks) under each look; explains what thrown-pass EPA leaves out."""
    p = plays[plays["call"].isin([*DEPTHS, "SACK"]) & plays[look].notna()]
    g = p.groupby(look)["call"].agg(sack_rate=lambda s: (s == "SACK").mean(), dropbacks="size")
    g.index = [labels[v] for v in g.index]
    return g.reset_index(names="look")
