import numpy as np
import pandas as pd
import polars as pl
import pytest

from drivedna.analysis.early import progress_features
from drivedna.analysis.effectiveness import bootstrap_estimates, estimate, naive_outcomes
from drivedna.analysis.modeling_table import OUTCOME_CLASSES, half_seconds_left, team_strength
from drivedna.analysis.openings import OPENING_ORDER, name_opening_clusters, opening_plays
from drivedna.drives.build import DRIVE_KEY, PROCESSED_DIR
from drivedna.models.cluster import REPORTS_DIR


# ------------------------------------------------------------------ team strength
def _drives():
    rows = []
    for week, pts in [(1, 7.0), (2, 0.0), (3, 7.0)]:
        for team, opp in [("A", "B"), ("B", "A")]:
            rows += [{"season": 2020, "week": week, "game_id": f"g{week}", "posteam": team,
                      "defteam": opp, "points": pts if team == "A" else 3.0}] * 4
    return pd.DataFrame(rows)


def test_loo_strength_excludes_the_game_itself():
    s = team_strength(_drives(), mode="loo", prior_drives=0)
    a = s[s.team == "A"].set_index("game_id")["off_strength"]
    league = _drives()["points"].mean()
    # in week 2 A scored 0; leaving it out, A's other games average 7 points per drive
    assert a["g2"] == pytest.approx(7.0 - league)
    # in week 1, the other games average (0 + 7) / 2 = 3.5
    assert a["g1"] == pytest.approx(3.5 - league)


def test_prior_strength_uses_only_earlier_weeks():
    s = team_strength(_drives(), mode="prior", prior_drives=0.0001)
    a = s[s.team == "A"].set_index("game_id")["off_strength"]
    assert a["g1"] == pytest.approx(0.0, abs=1e-3)           # nothing before week 1 -> league mean
    league = _drives()["points"].mean()
    assert a["g2"] == pytest.approx(7.0 - league, abs=1e-2)  # only week 1 (7 points)
    assert a["g3"] == pytest.approx(3.5 - league, abs=1e-2)  # weeks 1-2, never week 3 itself


def test_shrinkage_pulls_toward_league_mean():
    d = _drives()
    loose, tight = team_strength(d, "loo", prior_drives=1), team_strength(d, "loo", prior_drives=200)
    assert tight["off_strength"].abs().max() < loose["off_strength"].abs().max()


def test_half_seconds_left():
    out = half_seconds_left(pd.Series([1, 2, 3, 4, 5]), pd.Series([3500, 1900, 1750, 100, 400]))
    assert out.tolist() == [1700, 100, 1750, 100, 400]


# --------------------------------------------------------- estimator on known truth
def _simulate(n=9000, seed=0, true_effect=0.9):
    """Style B raises TD odds by `true_effect` (logit). Teams choose B more when x is high, and high x
    also raises TD odds on its own, so the naive comparison is biased upward."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    style = np.where(rng.random(n) < 1 / (1 + np.exp(-1.5 * x)), "B", "A")
    p_td = 1 / (1 + np.exp(-(-1.0 + 1.0 * x + true_effect * (style == "B"))))
    u = rng.random(n)
    outcome = np.select(
        [u < p_td, u < p_td + 0.15, u < p_td + 0.15 + 0.05, u < p_td + 0.15 + 0.05 + 0.03],
        ["Touchdown", "Field goal", "Turnover", "Failed"], default="Punt",
    )
    points = np.select([outcome == "Touchdown", outcome == "Field goal"], [7, 3], default=0)
    return pd.DataFrame({
        "game_id": rng.integers(0, 600, n).astype(str), "style": style, "outcome": outcome,
        "points": points.astype(float), "epa": points + rng.normal(size=n),
        "score_diff": x * 7, "start_yardline_100": 75 + rng.normal(size=n) * 8,
        "half_secs": rng.uniform(0, 1800, n), "wp_start": 0.5 + 0.1 * x,
        "off_strength": rng.normal(size=n) * 0.3, "def_strength": rng.normal(size=n) * 0.3,
        "qtr": rng.integers(1, 5, n), "season": rng.integers(2016, 2020, n),
    })


def test_adjusted_estimate_removes_confounding_bias():
    df = _simulate()
    naive = naive_outcomes(df)
    naive_gap = naive.loc["B", "Touchdown"] - naive.loc["A", "Touchdown"]
    est = estimate(df, ["A", "B"]).query("estimand == 'adjusted_all' and metric == 'Touchdown'").set_index("style")["value"]
    adj_gap = est["B"] - est["A"]
    # truth: average over x of sigmoid(-1 + x + .9) - sigmoid(-1 + x), scaled by the TD share of outcomes
    xs = np.random.default_rng(1).normal(size=200000)
    sig = lambda z: 1 / (1 + np.exp(-z))
    true_gap = (sig(-1 + xs + 0.9) - sig(-1 + xs)).mean()
    assert naive_gap > true_gap + 0.05                      # confounded comparison overstates it
    assert abs(adj_gap - true_gap) < 0.04                   # adjustment lands close to the truth
    assert abs(adj_gap - true_gap) < abs(naive_gap - true_gap)


def test_no_effect_gives_no_effect():
    df = _simulate(true_effect=0.0, seed=3)
    res = bootstrap_estimates(df, ["A", "B"], n_boot=20)
    td = res.query("estimand == 'vs_typical' and metric == 'Touchdown'")
    assert (td["lo"] <= td["value"]).all() and (td["value"] <= td["hi"]).all()
    assert td["value"].abs().max() < 0.03


def test_estimates_cover_every_outcome_class():
    res = estimate(_simulate(n=3000), ["A", "B"])
    assert set(OUTCOME_CLASSES) <= set(res["metric"])
    p = res.query("estimand == 'adjusted_all'").pivot_table(index="style", columns="metric", values="value")
    assert np.allclose(p[OUTCOME_CLASSES].sum(axis=1), 1.0)  # class probabilities sum to 1


# ------------------------------------------------------------------------ openings
def _plays():
    base = {"game_id": "g", "posteam": "A", "fixed_drive": 1.0}
    types = ["run", "qb_kneel", "pass", "qb_spike", "pass", "run", "pass"]
    return pl.DataFrame([{**base, "play_type": t, "play_num": i + 1} for i, t in enumerate(types)])


def test_opening_plays_are_the_first_n_designed_plays():
    out = opening_plays(_plays(), n=3)
    assert out["play_num"].to_list() == [1, 3, 5]  # kneel (2) and spike (4) skipped


def test_opening_names_follow_extremes():
    cols = ["pass_rate", "deep_pass_share", "inside_run_share"]
    prof = pd.DataFrame(
        {0: [0.62, 0.08, 0.98], 1: [0.64, 0.08, 0.0], 2: [0.56, 0.82, 0.5], 3: [0.99, 0.19, 0.63], 4: [0.25, 0.0, 0.5]},
        index=cols,
    ).T
    names = name_opening_clusters(prof)
    assert names == {2: "Deep-shot", 3: "Pass-only, up-tempo", 4: "Run-first",
                     0: "Balanced (inside runs)", 1: "Balanced (outside runs)"}
    assert set(names.values()) == set(OPENING_ORDER)


# ------------------------------------------------------------------- early progress
def test_progress_features_sum_first_three_and_need_a_fourth_play():
    rows = []
    for drive, n in [(1.0, 5), (2.0, 3)]:  # drive 2 ends after 3 plays -> excluded
        for i in range(n):
            rows.append({"game_id": "g", "posteam": "A", "fixed_drive": drive, "play_num": i + 1,
                         "play_type": "run", "yards_gained": 4.0, "epa": 0.1, "first_down": float(i == 1),
                         "sack": 0.0, "down": float(i + 1), "ydstogo": 10.0, "yardline_100": 80.0 - 4 * i,
                         "game_seconds_remaining": 900.0 - 30 * i})
    out = progress_features(pl.DataFrame(rows))
    assert out.height == 1
    r = out.row(0, named=True)
    assert r["yards_3"] == 12 and r["first_downs_3"] == 1 and r["secs_3"] == 90
    assert r["down_4"] == 4 and r["yardline_100_4"] == 68  # state facing play 4, not after it


# ----------------------------------------------------------------- real artifacts
eff_path = REPORTS_DIR / "effectiveness.csv"


@pytest.mark.skipif(not eff_path.exists(), reason="run drivedna.analysis.run_effectiveness")
def test_effectiveness_report_is_complete_and_well_formed():
    r = pd.read_csv(eff_path)
    assert {"whole_drive_style", "opening_style", "opening_team_season_fe"} <= set(r["spec"])
    assert (r["lo"] <= r["hi"]).all()
    main = r.query("spec == 'opening_style' and estimand == 'adjusted_all'")
    probs = main.pivot_table(index="style", columns="metric", values="value")[OUTCOME_CLASSES]
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_modeling_table_has_one_row_per_eligible_drive():
    if not (PROCESSED_DIR / "drive_styles.parquet").exists():
        pytest.skip("run drivedna.models.cluster")
    from drivedna.analysis.modeling_table import build_model_table

    t = build_model_table()
    assert not t[DRIVE_KEY].duplicated().any()
    assert t[["off_strength", "def_strength", "score_diff", "half_secs"]].notna().all().all()
    assert set(t["outcome"].unique()) == set(OUTCOME_CLASSES)


def test_early_prediction_features_contain_nothing_from_the_outcome():
    from drivedna.analysis.early import (
        CONTEXT_CAT,
        CONTEXT_NUM,
        FEATURE_SETS,
        PROGRESS_NUM,
        STYLE_NUM,
    )

    used = {c for fs in FEATURE_SETS for c in fs.num + fs.cat}
    forbidden = {"outcome", "result", "points", "scored", "epa", "yards", "n_plays", "n_real_plays", "margin"}
    assert not used & forbidden
    # progress is limited to the first 3 plays and the state *facing* the 4th: all suffixed _3 / _4
    assert all(c.endswith(("_3", "_4")) for c in PROGRESS_NUM)
    assert set(CONTEXT_NUM + CONTEXT_CAT + STYLE_NUM + PROGRESS_NUM) == used
