"""Team profiles: style DNA, similarity, change over time, and execution vs. style.

Key ideas
---------
* A team's raw style mix mostly reflects game script (trailing teams hurry up, leading teams run).
  So a team's *tendency* is its actual style share minus the share expected given the situations its
  drives started in (a multinomial model of style on field position, score, time, quarter, opponent).
* ~170 drives per team-season is small, so we check reliability first (split-half and year-to-year)
  before reading anything into differences between teams or seasons.
* "Execution" = points per drive above what the situation, the opponent and the opening style predict.
  Its outcome model must NOT control for the team's own offensive strength, or execution would be
  absorbed by construction. It includes QB, talent, scheme quality and luck.
"""
import numpy as np
import pandas as pd
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from drivedna.analysis.effectiveness import SPLINE_CONTROLS, OutcomeModels
from drivedna.analysis.modeling_table import build_model_table
from drivedna.analysis.openings import OPENING_ORDER, build_opening_styles
from drivedna.data.ingest import DEFAULT_SEASONS, pbp_path
from drivedna.drives.build import DRIVE_KEY
from drivedna.models.cluster import STYLE_ORDER

# situation known before the first snap, plus the opponent. No team identity, no offense strength.
SCRIPT_NUM = [*SPLINE_CONTROLS, "def_strength"]
SCRIPT_CAT = ["qtr", "season"]


# ------------------------------------------------------------------------- metadata
def team_season_meta(seasons=DEFAULT_SEASONS) -> pd.DataFrame:
    """Main head coach and primary QB per team-season, plus change flags vs. the prior season.

    Primary QB = most pass attempts. Coordinators are not in the data, so a new play-caller under
    the same head coach is invisible here.
    """
    coach_rows, qb_rows = [], []
    for s in seasons:
        raw = pl.read_parquet(pbp_path(s))
        games = raw.select("season", "game_id", "home_team", "away_team", "home_coach", "away_coach").unique("game_id")
        coach_rows.append(pl.concat([
            games.select("season", "game_id", pl.col("home_team").alias("team"), pl.col("home_coach").alias("coach")),
            games.select("season", "game_id", pl.col("away_team").alias("team"), pl.col("away_coach").alias("coach")),
        ]))
        qb_rows.append(
            raw.filter((pl.col("pass") == 1) & pl.col("passer_player_id").is_not_null() & pl.col("posteam").is_not_null())
            .group_by("season", pl.col("posteam").alias("team"), "passer_player_id", "passer_player_name")
            .agg(n=pl.len())
        )
    coaches = (
        pl.concat(coach_rows).group_by("season", "team", "coach").len()
        .sort("len", descending=True).unique(["season", "team"], keep="first", maintain_order=True)
        .select("season", "team", "coach", games_with_coach=pl.col("len"))
    )
    qbs = (
        pl.concat(qb_rows).group_by("season", "team", "passer_player_id", "passer_player_name").agg(n=pl.col("n").sum())
        .sort("n", descending=True).unique(["season", "team"], keep="first", maintain_order=True)
        .select("season", "team", qb=pl.col("passer_player_name"), qb_id=pl.col("passer_player_id"))
    )
    meta = coaches.join(qbs, on=["season", "team"], how="left").sort("team", "season").to_pandas()
    for col, flag in [("coach", "hc_changed"), ("qb_id", "qb_changed")]:
        prev = meta.groupby("team")[col].shift(1)
        meta[flag] = np.where(prev.isna(), np.nan, (meta[col] != prev).astype(float))
    return meta


# ------------------------------------------------------------------------- drive table
def drive_table() -> pd.DataFrame:
    """Eligible drives with whole-drive style, opening style and context (one row per drive)."""
    opening, _ = build_opening_styles()
    df = build_model_table().merge(opening.to_pandas(), on=DRIVE_KEY)
    df["team"] = df["posteam"]
    return df


# ------------------------------------------------------------------------- script adjustment
def expected_style_probs(df: pd.DataFrame, style_col: str, order: list[str], c: float = 5.0) -> pd.DataFrame:
    """P(style | situation) for each drive from a multinomial logit on pre-drive context only."""
    pre = ColumnTransformer([
        ("spline", Pipeline([("s", SplineTransformer(n_knots=5, degree=3)), ("z", StandardScaler())]), SPLINE_CONTROLS),
        ("linear", StandardScaler(), ["def_strength"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), SCRIPT_CAT),
    ])
    x = pre.fit_transform(df)
    codes = pd.Categorical(df[style_col], order).codes
    p = LogisticRegression(C=c, max_iter=1000).fit(x, codes).predict_proba(x)
    return pd.DataFrame(p, columns=order, index=df.index)


def team_dna(df: pd.DataFrame) -> pd.DataFrame:
    """Per team-season: n drives, actual and script-expected share for each style, and the excess.

    Columns: {kind}_{style}_actual / _expected / _excess / _se with kind in {drive, open}.
    The excess is the 'tendency': how much more (or less) a team uses a style than its situations
    would predict. _se is the binomial standard error of the actual share minus expected.
    """
    out = df.groupby(["team", "season"]).size().rename("n_drives").to_frame()
    for kind, col, order in [("drive", "style", STYLE_ORDER), ("open", "open_style", OPENING_ORDER)]:
        exp = expected_style_probs(df, col, order)
        for s in order:
            ind = (df[col] == s).astype(float)
            g = pd.DataFrame({"team": df["team"], "season": df["season"], "actual": ind, "expected": exp[s],
                              "var": exp[s] * (1 - exp[s])}).groupby(["team", "season"])
            agg = g.agg(actual=("actual", "mean"), expected=("expected", "mean"), var=("var", "sum"), n=("actual", "size"))
            out[f"{kind}_{s}_actual"] = agg["actual"]
            out[f"{kind}_{s}_expected"] = agg["expected"]
            out[f"{kind}_{s}_excess"] = agg["actual"] - agg["expected"]
            out[f"{kind}_{s}_se"] = np.sqrt(agg["var"]) / agg["n"]
    return out.reset_index()


def excess_columns(dna: pd.DataFrame, kind: str | None = None) -> list[str]:
    cols = [c for c in dna.columns if c.endswith("_excess")]
    return [c for c in cols if kind is None or c.startswith(kind + "_")]


# ------------------------------------------------------------------------- reliability
def _excess_for_subset(df: pd.DataFrame, probs: dict[str, pd.DataFrame], mask: np.ndarray, kind_cols: dict) -> pd.DataFrame:
    """Per team-season excess shares using only the drives in `mask` (expected probs are fixed)."""
    parts = {}
    sub = df[mask]
    for col, order in kind_cols.items():
        for s in order:
            ex = (sub[col] == s).astype(float) - probs[col].loc[sub.index, s]
            parts[(col, s)] = ex.groupby([sub["team"], sub["season"]]).mean()
    return pd.DataFrame(parts)


def split_half_reliability(df: pd.DataFrame, n_splits: int = 20, seed: int = 0) -> pd.DataFrame:
    """Spearman-Brown reliability of each style's excess share across team-seasons.

    Each team-season's games are randomly split into halves; the correlation between halves (across
    team-seasons) is stepped up to full length. ~0 means the differences between teams are noise.
    """
    kind_cols = {"style": STYLE_ORDER, "open_style": OPENING_ORDER}
    probs = {c: expected_style_probs(df, c, o) for c, o in kind_cols.items()}
    rng = np.random.default_rng(seed)
    games = df[["team", "season", "game_id"]].drop_duplicates()
    corr = {k: [] for k in [(c, s) for c, o in kind_cols.items() for s in o]}
    for _ in range(n_splits):
        flip = games.assign(half=rng.integers(0, 2, len(games)))
        half = df.merge(flip, on=["team", "season", "game_id"])["half"].to_numpy()
        a = _excess_for_subset(df, probs, half == 0, kind_cols)
        b = _excess_for_subset(df, probs, half == 1, kind_cols)
        for k, values in corr.items():
            values.append(a[k].corr(b[k].reindex(a.index)))
    rows = []
    for (col, s), v in corr.items():
        r = float(np.nanmean(v))
        rows.append({"kind": "drive" if col == "style" else "open", "style": s, "half_corr": r,
                     "reliability": 2 * r / (1 + r) if r > -1 else np.nan})
    return pd.DataFrame(rows)


def year_to_year(dna: pd.DataFrame) -> pd.DataFrame:
    """Correlation of each style's excess share between a team's consecutive seasons."""
    rows = []
    nxt = dna.assign(season=dna["season"] - 1)
    j = dna.merge(nxt, on=["team", "season"], suffixes=("", "_next"))
    for c in excess_columns(dna):
        kind, style = c.split("_", 1)
        rows.append({"kind": kind, "style": style.removesuffix("_excess"), "year_to_year_corr": j[c].corr(j[c + "_next"])})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------- similarity
def robust_z(dna: pd.DataFrame, cols: list[str], clip: float = 5.0) -> pd.DataFrame:
    """Median/MAD scaling, clipped, so a few extreme team-seasons (e.g. a no-huddle outlier) don't dominate."""
    x = dna[cols]
    med = x.median()
    mad = (x - med).abs().median() * 1.4826
    return ((x - med) / mad.replace(0, np.nan)).clip(-clip, clip).fillna(0.0)


def similarity_matrix(dna: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Cosine similarity between team-seasons in robust-z space."""
    z = robust_z(dna, cols).to_numpy()
    z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-9)
    labels = dna["team"] + " " + dna["season"].astype(str)
    return pd.DataFrame(z @ z.T, index=labels, columns=labels)


def nearest_neighbors(sim: pd.DataFrame, label: str, k: int = 5, other_teams_only: bool = True) -> pd.Series:
    s = sim[label].drop(label)
    if other_teams_only:
        s = s[[not i.startswith(label.split()[0] + " ") for i in s.index]]
    return s.sort_values(ascending=False).head(k)


def distance_to_next_season(dna: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Euclidean distance in robust-z space between a team-season and the next one."""
    z = robust_z(dna, cols)
    z = z.assign(team=dna["team"].to_numpy(), season=dna["season"].to_numpy())
    nxt = z.assign(season=z["season"] - 1)
    j = z.merge(nxt, on=["team", "season"], suffixes=("", "_n"))
    d = np.sqrt(sum((j[c] - j[c + "_n"]) ** 2 for c in cols))
    return pd.DataFrame({"team": j["team"], "season": j["season"] + 1, "distance": d})


# ------------------------------------------------------------------------- change events
def change_events(dna: pd.DataFrame, meta: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Each team's season-to-season style shift, labeled by whether the coach / primary QB changed."""
    dist = distance_to_next_season(dna, cols)
    z = robust_z(dna, cols).assign(team=dna["team"].to_numpy(), season=dna["season"].to_numpy())
    prev = z.assign(season=z["season"] + 1)
    both = z.merge(prev, on=["team", "season"], suffixes=("", "_prev"))
    delta = pd.DataFrame({c: both[c] - both[c + "_prev"] for c in cols}).assign(team=both["team"], season=both["season"])
    big = delta[cols].abs().idxmax(axis=1).str.removesuffix("_excess")
    sign = np.sign(delta[cols].to_numpy()[np.arange(len(delta)), delta[cols].abs().to_numpy().argmax(1)])
    shifts = pd.DataFrame({"team": delta["team"], "season": delta["season"], "biggest_shift": big,
                           "shift_direction": np.where(sign > 0, "more", "less")})
    prev_meta = meta[["team", "season", "coach", "qb"]].assign(season=meta["season"] + 1)
    out = (
        dist.merge(shifts, on=["team", "season"])
        .merge(meta[["team", "season", "coach", "qb", "hc_changed", "qb_changed"]], on=["team", "season"])
        .merge(prev_meta.rename(columns={"coach": "prev_coach", "qb": "prev_qb"}), on=["team", "season"], how="left")
    )
    out["event"] = np.select(
        [(out["hc_changed"] == 1) & (out["qb_changed"] == 1), out["hc_changed"] == 1, out["qb_changed"] == 1],
        ["new coach + new QB", "new coach only", "new QB only"], default="no change",
    )
    return out


def event_summary(events: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Mean style shift by event type with bootstrap CIs and the difference vs. 'no change'."""
    rng = np.random.default_rng(seed)
    base = events.loc[events["event"] == "no change", "distance"].to_numpy()
    rows = []
    for name, g in events.groupby("event"):
        d = g["distance"].to_numpy()
        boots = np.array([rng.choice(d, len(d)).mean() for _ in range(n_boot)])
        row = {"event": name, "n": len(d), "mean_shift": d.mean(), "lo": np.quantile(boots, 0.025), "hi": np.quantile(boots, 0.975)}
        if name != "no change":
            diffs = np.array([rng.choice(d, len(d)).mean() - rng.choice(base, len(base)).mean() for _ in range(n_boot)])
            row.update(diff_vs_no_change=d.mean() - base.mean(), diff_lo=np.quantile(diffs, 0.025), diff_hi=np.quantile(diffs, 0.975))
        rows.append(row)
    return pd.DataFrame(rows).set_index("event").reindex(["no change", "new QB only", "new coach only", "new coach + new QB"]).reset_index()


# ------------------------------------------------------------------------- execution vs. style
def decompose_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Split each team-season's points per drive (vs. league) into situation, style and execution.

        situation  how favorable the drives were: field position, score, time, opponent strength
        style      what the opening styles the team chose are worth in those situations (Phase 4
                   association, so observational), relative to a typical mix
        execution  actual minus predicted given situation, opponent and opening style: QB, talent,
                   scheme quality, and luck. The model deliberately does NOT control for the team's
                   own offensive strength, which would absorb execution.
    The three parts sum to the team's points per drive above the league average.
    """
    d = df.assign(style=df["open_style"])
    models = OutcomeModels.fit(d, OPENING_ORDER, linear_controls=["def_strength"])
    shares = d["style"].value_counts(normalize=True).reindex(OPENING_ORDER).fillna(0.0)
    under = {s: models.predict(d, s)["points"] for s in OPENING_ORDER}
    typical = sum(shares[s] * under[s] for s in OPENING_ORDER)
    actual_style = models.predict(d)["points"]
    work = pd.DataFrame({
        "team": d["team"], "season": d["season"], "points": d["points"],
        "typical": typical, "with_style": actual_style,
    })
    work["style_part"] = work["with_style"] - work["typical"]
    work["exec_part"] = work["points"] - work["with_style"]
    g = work.groupby(["team", "season"])
    out = g.agg(n_drives=("points", "size"), points_per_drive=("points", "mean"),
                typical=("typical", "mean"), style_part=("style_part", "mean"), exec_part=("exec_part", "mean"))
    out["exec_se"] = g["exec_part"].std() / np.sqrt(out["n_drives"])
    out["situation_part"] = out["typical"] - work["typical"].mean()
    out["total_vs_league"] = out["points_per_drive"] - work["points"].mean()
    out["residual_check"] = out["total_vs_league"] - (out["situation_part"] + out["style_part"] + out["exec_part"])
    return out.drop(columns="typical").reset_index()


def component_persistence(decomp: pd.DataFrame) -> pd.DataFrame:
    """Season-to-season correlation of each component (is it a stable team trait?)."""
    nxt = decomp.assign(season=decomp["season"] - 1)
    j = decomp.merge(nxt, on=["team", "season"], suffixes=("", "_next"))
    rows = []
    for c in ["total_vs_league", "situation_part", "style_part", "exec_part"]:
        rows.append({"component": c, "year_to_year_corr": j[c].corr(j[c + "_next"]), "std_across_team_seasons": decomp[c].std()})
    return pd.DataFrame(rows)
