"""Do drive styles differ in outcomes once the situation is held fixed?

Method: standardization (g-computation). Fit outcome models of
    outcome ~ style + pre-drive context + team quality
then predict every drive under each style and compare. Two estimands:

  vs_typical (primary)   For drives that actually used style s: predicted outcome under s minus
                         predicted outcome under the share-weighted mix of styles, same situation.
                         ("Would a typical style have done better in the situations where teams
                         pick s?"). Stays inside the data, so it avoids extrapolation.
  adjusted_all           Predicted outcome if *every* drive used s (population-standardized). Needs
                         overlap: e.g. Hurry-up is almost only used late in halves, so this
                         extrapolates for it. Shown with an overlap diagnostic.

Uncertainty: bootstrap that resamples whole games (drives in a game are not independent).
This is observational. It adjusts for what is measured, not for everything (e.g. a coach's
read of the defense), so read the numbers as "associated with, given the situation".
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from drivedna.analysis.modeling_table import CATEGORICAL_CONTROLS, OUTCOME_CLASSES

SPLINE_CONTROLS = ["start_yardline_100", "score_diff", "half_secs", "wp_start"]
LINEAR_CONTROLS = ["off_strength", "def_strength"]


def _preprocessor(styles: list[str], cat_controls: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("spline", Pipeline([("s", SplineTransformer(n_knots=5, degree=3)), ("z", StandardScaler())]),
         SPLINE_CONTROLS),
        ("linear", StandardScaler(), LINEAR_CONTROLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_controls),
        ("style", OneHotEncoder(categories=[styles], drop="first"), ["style"]),
    ])


@dataclass
class OutcomeModels:
    """Fitted models mapping (context, style) -> class probabilities, points and EPA."""

    styles: list[str]
    cat_controls: list[str]
    pre: ColumnTransformer
    clf: LogisticRegression
    points: Ridge
    epa: Ridge

    @classmethod
    def fit(
        cls, df: pd.DataFrame, styles: list[str], cat_controls: list[str] | None = None, c: float = 5.0
    ) -> "OutcomeModels":
        cat_controls = list(CATEGORICAL_CONTROLS) if cat_controls is None else cat_controls
        pre = _preprocessor(styles, cat_controls)
        x = pre.fit_transform(df)
        clf = LogisticRegression(C=c, max_iter=1000).fit(x, pd.Categorical(df["outcome"], OUTCOME_CLASSES).codes)
        return cls(
            styles, cat_controls, pre, clf,
            Ridge(alpha=1.0).fit(x, df["points"]), Ridge(alpha=1.0).fit(x, df["epa"]),
        )

    def predict(self, df: pd.DataFrame, style: str | None = None) -> pd.DataFrame:
        """Predictions for each row, optionally with the style forced to `style`."""
        d = df if style is None else df.assign(style=style)
        x = self.pre.transform(d)
        proba = self.clf.predict_proba(x)
        out = pd.DataFrame(proba, columns=OUTCOME_CLASSES, index=df.index)
        out["points"], out["epa"] = self.points.predict(x), self.epa.predict(x)
        out["scored"] = out["Touchdown"] + out["Field goal"]
        return out


METRICS = OUTCOME_CLASSES + ["scored", "points", "epa"]


def estimate(
    df: pd.DataFrame, styles: list[str], cat_controls: list[str] | None = None
) -> pd.DataFrame:
    """Point estimates for both estimands, long format: style, estimand, metric, value."""
    models = OutcomeModels.fit(df, styles, cat_controls)
    shares = df["style"].value_counts(normalize=True).reindex(styles).fillna(0.0)
    under = {s: models.predict(df, s)[METRICS] for s in styles}
    typical = sum(shares[s] * under[s] for s in styles)  # same-situation, share-weighted mix
    rows = []
    for s in styles:
        own = (df["style"] == s).to_numpy()
        if own.sum() == 0:
            continue
        att = (under[s][own] - typical[own]).mean()
        everyone = under[s].mean()
        for m in METRICS:
            rows.append((s, "vs_typical", m, att[m]))
            rows.append((s, "adjusted_all", m, everyone[m]))
    return pd.DataFrame(rows, columns=["style", "estimand", "metric", "value"])


def bootstrap_estimates(
    df: pd.DataFrame, styles: list[str], n_boot: int = 100, seed: int = 0,
    cat_controls: list[str] | None = None,
) -> pd.DataFrame:
    """Point estimate plus 95% bootstrap CI (resampling whole games)."""
    point = estimate(df, styles, cat_controls)
    games = df["game_id"].unique()
    by_game = {g: idx for g, idx in df.groupby("game_id").indices.items()}
    rng = np.random.default_rng(seed)
    reps = []
    for b in range(n_boot):
        pick = rng.choice(games, len(games), replace=True)
        idx = np.concatenate([by_game[g] for g in pick])
        sample = df.iloc[idx].reset_index(drop=True)
        r = estimate(sample, styles, cat_controls)
        r["rep"] = b
        reps.append(r)
    boot = pd.concat(reps)
    ci = boot.groupby(["style", "estimand", "metric"])["value"].quantile([0.025, 0.975]).unstack()
    ci.columns = ["lo", "hi"]
    out = point.merge(ci.reset_index(), on=["style", "estimand", "metric"])
    out["significant"] = (out["lo"] > 0) | (out["hi"] < 0)
    return out


def naive_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Unadjusted outcome mix by style, for contrast with the adjusted numbers."""
    g = df.groupby("style")
    out = pd.concat([g["outcome"].value_counts(normalize=True).unstack().reindex(columns=OUTCOME_CLASSES),
                     g[["points", "epa"]].mean()], axis=1)
    out["scored"] = out["Touchdown"] + out["Field goal"]
    return out


def overlap_diagnostic(df: pd.DataFrame, styles: list[str]) -> pd.DataFrame:
    """How well can the situation alone predict style? Low propensity for a drive's own style
    means that style is rarely used in comparable situations (poor overlap)."""
    pre = ColumnTransformer([
        ("spline", Pipeline([("s", SplineTransformer(n_knots=5, degree=3)), ("z", StandardScaler())]),
         SPLINE_CONTROLS),
        ("linear", StandardScaler(), LINEAR_CONTROLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_CONTROLS),
    ])
    x = pre.fit_transform(df)
    codes = pd.Categorical(df["style"], styles).codes
    prop = LogisticRegression(C=5.0, max_iter=1000).fit(x, codes).predict_proba(x)
    own = prop[np.arange(len(df)), codes]
    return pd.DataFrame({
        "style": df["style"].to_numpy(), "own_propensity": own,
        "accuracy_vs_chance": (prop.argmax(1) == codes),
    }).groupby("style").agg(
        mean_own_propensity=("own_propensity", "mean"),
        share_below_0_05=("own_propensity", lambda s: (s < 0.05).mean()),
        context_only_accuracy=("accuracy_vs_chance", "mean"),
    )
