"""How well can we predict how a drive ends after its first 3 designed plays, and what matters?

Information available at that moment, in three groups (each added on top of the previous):
  context   known before the first snap: field position, score, time, win prob, team quality.
            Team quality uses *prior weeks only*, so no future games leak in.
  style     the play-calling of the first 3 plays (the same features as the opening style).
  progress  what happened in those 3 plays and where the drive stands: yards, EPA, first downs,
            sacks, and the down / distance / field position facing the 4th designed play.

Question this answers: beyond the situation and the way the drive has actually gone, does the
*style* of the opening add predictive information? Evaluated out of time (train <= 2022, test >= 2023).
Drives that end within 3 plays are excluded (there is no 4th play to condition on).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from drivedna.analysis.modeling_table import build_model_table
from drivedna.analysis.openings import opening_features
from drivedna.drives.build import DRIVE_KEY, PROCESSED_DIR
from drivedna.features.drive_features import STYLE_FEATURES

CONTEXT_NUM = ["start_yardline_100", "score_diff", "half_secs", "wp_start", "off_strength", "def_strength"]
CONTEXT_CAT = ["qtr"]
STYLE_NUM = STYLE_FEATURES
PROGRESS_NUM = [
    "yards_3", "epa_3", "first_downs_3", "sacks_3", "negative_3", "secs_3",
    "down_4", "ydstogo_4", "yardline_100_4",
]
TRAIN_THROUGH = 2022
TARGETS = {"scored": ["Touchdown", "Field goal"], "touchdown": ["Touchdown"]}


def progress_features(drive_plays: pl.DataFrame) -> pl.DataFrame:
    """What happened in the first 3 designed plays and the state facing the 4th."""
    real = drive_plays.filter(pl.col("play_type").is_in(["pass", "run"])).sort(DRIVE_KEY + ["play_num"])
    real = real.with_columns(rank=pl.int_range(pl.len()).over(DRIVE_KEY))
    first3 = real.filter(pl.col("rank") < 3).group_by(DRIVE_KEY).agg(
        yards_3=pl.col("yards_gained").fill_null(0).sum(),
        epa_3=pl.col("epa").fill_null(0).sum(),
        first_downs_3=pl.col("first_down").fill_null(0).sum(),
        sacks_3=pl.col("sack").fill_null(0).sum(),
        negative_3=(pl.col("yards_gained") < 0).sum(),
        t0=pl.col("game_seconds_remaining").first(),
    )
    fourth = real.filter(pl.col("rank") == 3).select(
        DRIVE_KEY
        + [
            pl.col("down").alias("down_4"),
            pl.col("ydstogo").alias("ydstogo_4"),
            pl.col("yardline_100").alias("yardline_100_4"),
            pl.col("game_seconds_remaining").alias("t4"),
        ]
    )
    return first3.join(fourth, on=DRIVE_KEY, how="inner").with_columns(
        secs_3=(pl.col("t0") - pl.col("t4")).clip(0, 180)
    ).drop("t0", "t4")


def build_early_table() -> pd.DataFrame:
    plays = pl.read_parquet(PROCESSED_DIR / "drive_plays.parquet")
    ctx = build_model_table(strength_mode="prior")
    tab = pl.from_pandas(ctx[DRIVE_KEY + ["season", "outcome"] + CONTEXT_NUM + CONTEXT_CAT])
    tab = tab.join(opening_features(plays).select(DRIVE_KEY + STYLE_NUM), on=DRIVE_KEY)
    tab = tab.join(progress_features(plays), on=DRIVE_KEY, how="inner")
    df = tab.to_pandas()
    df["game_id"] = df["game_id"].astype(str)
    return df


def _pre(num: list[str], cat: list[str], spline: bool) -> ColumnTransformer:
    num_step = (
        Pipeline([
            ("i", SimpleImputer(strategy="median", add_indicator=True)),  # keep 'was missing' visible
            ("s", SplineTransformer(n_knots=4, degree=3)),
            ("z", StandardScaler()),
        ])
        if spline else StandardScaler()
    )
    parts = [("num", num_step, num)]
    if cat:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat))
    return ColumnTransformer(parts)


@dataclass
class FeatureSet:
    name: str
    num: list[str]
    cat: list[str]


FEATURE_SETS = [
    FeatureSet("context", CONTEXT_NUM, CONTEXT_CAT),
    FeatureSet("context+style", CONTEXT_NUM + STYLE_NUM, CONTEXT_CAT),
    FeatureSet("context+progress", CONTEXT_NUM + PROGRESS_NUM, CONTEXT_CAT),
    FeatureSet("context+style+progress", CONTEXT_NUM + STYLE_NUM + PROGRESS_NUM, CONTEXT_CAT),
]


def _fit(fs: FeatureSet, kind: str, x: pd.DataFrame, y: np.ndarray):
    if kind == "logit":
        model = LogisticRegression(C=1.0, max_iter=2000)
        pipe = Pipeline([("pre", _pre(fs.num, fs.cat, spline=True)), ("m", model)])
    else:
        pre = ColumnTransformer(
            [("num", "passthrough", fs.num)]
            + ([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), fs.cat)] if fs.cat else [])
        )
        pipe = Pipeline([("pre", pre), ("m", HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.05, max_iter=300, l2_regularization=1.0, random_state=0))])
    return pipe.fit(x, y)


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {"log_loss": log_loss(y, p), "brier": brier_score_loss(y, p), "auc": roc_auc_score(y, p)}


def evaluate(df: pd.DataFrame, target: str = "scored") -> tuple[pd.DataFrame, dict]:
    """Out-of-time comparison of feature sets (logit) and the full set with gradient boosting."""
    y_all = df["outcome"].isin(TARGETS[target]).to_numpy().astype(int)
    train, test = (df["season"] <= TRAIN_THROUGH).to_numpy(), (df["season"] > TRAIN_THROUGH).to_numpy()
    base_rate = y_all[train].mean()
    rows = [{"model": "base rate", "kind": "constant",
             **_metrics(y_all[test], np.full(test.sum(), base_rate))}]
    preds = {}
    for fs in FEATURE_SETS:
        for kind in ["logit", "gbm"]:
            m = _fit(fs, kind, df[train], y_all[train])
            p = m.predict_proba(df[test])[:, 1]
            preds[(fs.name, kind)] = p
            rows.append({"model": fs.name, "kind": kind, **_metrics(y_all[test], p)})
    out = pd.DataFrame(rows)
    return out, {"y_test": y_all[test], "preds": preds, "test_games": df.loc[test, "game_id"].to_numpy()}


def style_increment_ci(detail: dict, base: tuple, full: tuple, n_boot: int = 300, seed: int = 0) -> dict:
    """Bootstrap (by game) CI of the log-loss improvement from adding style on top of `base`."""
    y, games = detail["y_test"], detail["test_games"]
    pb, pf = detail["preds"][base], detail["preds"][full]
    by_game = pd.Series(np.arange(len(y))).groupby(games).apply(np.array)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        idx = np.concatenate(by_game.sample(len(by_game), replace=True, random_state=rng.integers(1e9)).to_list())
        deltas.append(log_loss(y[idx], pb[idx]) - log_loss(y[idx], pf[idx]))  # positive = full is better
    d = np.array(deltas)
    return {"improvement": log_loss(y, pb) - log_loss(y, pf), "lo": np.quantile(d, 0.025), "hi": np.quantile(d, 0.975)}


def group_permutation_importance(df: pd.DataFrame, target: str = "scored", n_repeats: int = 5, seed: int = 0) -> pd.DataFrame:
    """Increase in test log-loss when a whole feature group is shuffled (full GBM model)."""
    fs = FEATURE_SETS[-1]
    y_all = df["outcome"].isin(TARGETS[target]).to_numpy().astype(int)
    train, test = (df["season"] <= TRAIN_THROUGH).to_numpy(), (df["season"] > TRAIN_THROUGH).to_numpy()
    model = _fit(fs, "gbm", df[train], y_all[train])
    x_test, y_test = df[test].reset_index(drop=True), y_all[test]
    base = log_loss(y_test, model.predict_proba(x_test)[:, 1])
    groups = {
        "context": CONTEXT_NUM + CONTEXT_CAT,
        "opening style": STYLE_NUM,
        "progress (yards, EPA, down/distance...)": PROGRESS_NUM,
    }
    groups.update({c: [c] for c in PROGRESS_NUM + CONTEXT_NUM})
    rng = np.random.default_rng(seed)
    rows = []
    for name, cols in groups.items():
        deltas = []
        for _ in range(n_repeats):
            shuffled = x_test.copy()
            perm = rng.permutation(len(shuffled))
            shuffled[cols] = shuffled[cols].to_numpy()[perm]
            deltas.append(log_loss(y_test, model.predict_proba(shuffled)[:, 1]) - base)
        rows.append({"feature_or_group": name, "is_group": name in ("context", "opening style")
                     or name.startswith("progress"), "delta_log_loss": np.mean(deltas)})
    return pd.DataFrame(rows).sort_values("delta_log_loss", ascending=False).reset_index(drop=True)

