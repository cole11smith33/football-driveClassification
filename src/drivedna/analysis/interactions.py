"""Does the value of an opening style depend on the defense it faces?

Model: points ~ controls + opponent scheme tendencies + opening style + style x tendency, fit with
ridge regression (the interactions are shrunk toward zero, which is the 'partial pooling' we want
when a signal might not exist). The tendencies are leave-one-game-out and shrunk, so a drive never
contributes to its own opponent's tendency.

Two questions, kept separate:
  1. Is there heterogeneity at all?  Leave-one-season-out prediction: does adding the
     interactions reduce out-of-sample error vs. main effects only? (bootstrap CI by game)
  2. If so, where?  The effect of each style (vs. a typical mix) with the opponent's tendency
     set to -1 SD vs. +1 SD.
Observational, like all of this: teams also pick styles knowing the defense they face.
"""
import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from drivedna.analysis.effectiveness import LINEAR_CONTROLS, SPLINE_CONTROLS

MODERATORS = ["blitz_loo", "box_loo"]
CAT = ["qtr", "season"]


def standardize_moderators(df: pd.DataFrame, cols=MODERATORS) -> tuple[pd.DataFrame, dict]:
    stats = {c: (df[c].mean(), df[c].std()) for c in cols}
    out = df.copy()
    for c, (m, s) in stats.items():
        out[c + "_z"] = (df[c] - m) / s
    return out, stats


@dataclass
class InteractionModel:
    styles: list[str]
    moderators: list[str]  # standardized column names, e.g. blitz_loo_z
    use_style: bool
    use_interactions: bool
    alpha: float = 20.0

    def _controls(self) -> ColumnTransformer:
        return ColumnTransformer([
            ("spline", Pipeline([("s", SplineTransformer(n_knots=5, degree=3)), ("z", StandardScaler())]), SPLINE_CONTROLS),
            ("linear", StandardScaler(), [*LINEAR_CONTROLS, *self.moderators]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
        ])

    def _style_block(self, style: pd.Series, mods: np.ndarray) -> np.ndarray:
        if not self.use_style:
            return np.zeros((len(style), 0))
        dummies = np.column_stack([(style == s).to_numpy().astype(float) for s in self.styles[1:]])
        if not self.use_interactions:
            return dummies
        inter = (dummies[:, :, None] * mods[:, None, :]).reshape(len(style), -1)
        return np.hstack([dummies, inter])

    def _x(self, df: pd.DataFrame, style: pd.Series | None = None, mod_override: dict | None = None) -> np.ndarray:
        d = df if mod_override is None else df.assign(**mod_override)
        mods = d[self.moderators].to_numpy(dtype=float)
        base = self.pre_.transform(d)
        base = base.toarray() if hasattr(base, "toarray") else base
        return np.hstack([base, self._style_block(df["style"] if style is None else style, mods)])

    def fit(self, df: pd.DataFrame, target: str = "points") -> "InteractionModel":
        self.pre_ = self._controls().fit(df)
        self.model_ = Ridge(alpha=self.alpha).fit(self._x(df), df[target])
        return self

    def predict(self, df: pd.DataFrame, style: str | None = None, mod_override: dict | None = None) -> np.ndarray:
        forced = None if style is None else pd.Series(style, index=df.index)
        return self.model_.predict(self._x(df, forced, mod_override))


SPECS = {
    "controls only": {"use_style": False, "use_interactions": False},
    "+ opening style": {"use_style": True, "use_interactions": False},
    "+ style x opponent scheme": {"use_style": True, "use_interactions": True},
}


def season_cv(df: pd.DataFrame, styles: list[str], moderators: list[str], target: str = "points", alpha: float = 20.0):
    """Leave-one-season-out predictions for each specification."""
    preds = {name: np.zeros(len(df)) for name in SPECS}
    for season in sorted(df["season"].unique()):
        train, test = (df["season"] != season).to_numpy(), (df["season"] == season).to_numpy()
        for name, kw in SPECS.items():
            m = InteractionModel(styles, moderators, alpha=alpha, **kw).fit(df[train], target)
            preds[name][test] = m.predict(df[test])
    return preds


def cv_summary(df: pd.DataFrame, preds: dict, target: str = "points", n_boot: int = 300, seed: int = 0) -> pd.DataFrame:
    """MSE per spec and bootstrap (by game) CI of the MSE reduction from each added term."""
    y = df[target].to_numpy()
    by_game = df.reset_index(drop=True).groupby("game_id").indices
    game_ids = np.array(list(by_game))
    rng = np.random.default_rng(seed)
    names = list(preds)
    deltas = {n: [] for n in names[1:]}
    for _ in range(n_boot):
        idx = np.concatenate([by_game[g] for g in rng.choice(game_ids, len(game_ids))])
        mse = {n: np.mean((y[idx] - preds[n][idx]) ** 2) for n in names}
        for prev, cur in itertools.pairwise(names):
            deltas[cur].append(mse[prev] - mse[cur])  # positive = adding the term helps
    rows = [{"spec": names[0], "mse": np.mean((y - preds[names[0]]) ** 2)}]
    for prev, cur in itertools.pairwise(names):
        d = np.array(deltas[cur])
        rows.append({"spec": cur, "mse": np.mean((y - preds[cur]) ** 2),
                     "improvement_vs_previous": np.mean((y - preds[prev]) ** 2) - np.mean((y - preds[cur]) ** 2),
                     "lo": np.quantile(d, 0.025), "hi": np.quantile(d, 0.975)})
    return pd.DataFrame(rows)


def effects_at_levels(df: pd.DataFrame, styles: list[str], moderators: list[str], levels=(-1.0, 1.0),
                      target: str = "points", alpha: float = 20.0) -> pd.DataFrame:
    """Effect of each style vs. a typical mix (points/drive) with one moderator forced to each level."""
    m = InteractionModel(styles, moderators, True, True, alpha).fit(df, target)
    shares = df["style"].value_counts(normalize=True).reindex(styles).fillna(0.0)
    rows = []
    for mod in moderators:
        for level in levels:
            override = {mod: np.full(len(df), level)}
            under = {s: m.predict(df, s, override) for s in styles}
            typical = sum(shares[s] * under[s] for s in styles)
            for s in styles:
                own = (df["style"] == s).to_numpy()
                rows.append({"moderator": mod, "level": level, "style": s, "effect": float((under[s][own] - typical[own]).mean())})
    return pd.DataFrame(rows)


def bootstrap_effect_differences(df: pd.DataFrame, styles: list[str], moderators: list[str], n_boot: int = 100,
                                 seed: int = 0, alpha: float = 20.0) -> pd.DataFrame:
    """Effect at +1 SD minus effect at -1 SD of each moderator, with a bootstrap (by game) 95% CI."""
    point = effects_at_levels(df, styles, moderators, alpha=alpha)
    by_game = df.reset_index(drop=True).groupby("game_id").indices
    game_ids = np.array(list(by_game))
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(n_boot):
        idx = np.concatenate([by_game[g] for g in rng.choice(game_ids, len(game_ids))])
        e = effects_at_levels(df.iloc[idx].reset_index(drop=True), styles, moderators, alpha=alpha)
        reps.append(e.pivot_table(index=["moderator", "style"], columns="level", values="effect").pipe(lambda p: p[1.0] - p[-1.0]))
    boot = pd.concat(reps, axis=1)
    p = point.pivot_table(index=["moderator", "style"], columns="level", values="effect")
    out = pd.DataFrame({"effect_low": p[-1.0], "effect_high": p[1.0], "high_minus_low": p[1.0] - p[-1.0]})
    out["lo"], out["hi"] = boot.quantile(0.025, axis=1), boot.quantile(0.975, axis=1)
    out["significant"] = (out["lo"] > 0) | (out["hi"] < 0)
    return out.reset_index()


def interaction_frame(drives: pd.DataFrame, loo: pd.DataFrame) -> pd.DataFrame:
    """Drive table (opening style as `style`) with standardized leave-one-game-out opponent tendencies.

    Missing tendencies (none expected) are set to the league average, i.e. 0 after standardizing.
    """
    keys = ["defteam", "season", "game_id"]
    cols = list(MODERATORS)
    df = drives.assign(style=drives["open_style"]).merge(loo[[*keys, *cols]], on=keys, how="left")
    df[cols] = df[cols].fillna(df[cols].mean())
    df, _ = standardize_moderators(df, cols)
    return df.drop(columns=cols).rename(columns={c + "_z": c for c in cols})
