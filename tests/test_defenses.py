import numpy as np
import pandas as pd
import pytest

from drivedna.analysis.defenses import (
    SCHEME,
    add_scheme_residuals,
    archetype_study,
    defense_tendencies,
    loo_tendencies,
    split_half_reliability,
    two_way_archetypes,
)
from drivedna.analysis.interactions import MODERATORS, cv_summary, season_cv
from drivedna.analysis.situations import (
    CALLS,
    DEPTHS,
    SITUATIONS,
    add_looks,
    adjusted_call_effects,
    call_mix,
    sack_rate_by_look,
)
from drivedna.drives.build import PROCESSED_DIR
from drivedna.models.cluster import REPORTS_DIR


def _plays(n_def=20, seasons=(2020, 2021), per=300, seed=0, blitz_taste=True):
    """Pass/run plays for many defenses. Box depends on the offense's formation (3-WR sets draw light
    boxes), and defense 0 genuinely blitzes more. Defense 1 just faces lots of shotgun formations."""
    rng = np.random.default_rng(seed)
    rows = []
    for dfn in range(n_def):
        for season in seasons:
            for i in range(per):
                shotgun_share = 0.9 if dfn == 1 else 0.5
                formation = "SHOTGUN" if rng.random() < shotgun_share else "I_FORM"
                is_pass = rng.random() < (0.7 if formation == "SHOTGUN" else 0.4)
                box = rng.normal(5.4 if formation == "SHOTGUN" else 6.8, 0.8)
                blitz_p = 0.28 + (0.18 if (blitz_taste and dfn == 0) else 0.0)
                rushers = (5 if rng.random() < blitz_p else 4) if is_pass else np.nan
                rows.append({
                    "defteam": f"D{dfn}", "season": season, "game_id": f"{dfn}_{season}_{i // 20}",
                    "play_type": "pass" if is_pass else "run", "ydstogo": float(rng.integers(1, 12)),
                    "yardline_100": float(rng.integers(5, 95)), "score_differential": float(rng.integers(-14, 14)),
                    "half_secs": float(rng.integers(0, 1800)), "down": float(rng.integers(1, 5)), "qtr": float(rng.integers(1, 5)),
                    "offense_formation": formation, "defenders_in_box": box, "number_of_pass_rushers": rushers,
                    "man_zone": ("MAN_COVERAGE" if rng.random() < 0.3 else "ZONE_COVERAGE") if is_pass else None,
                    "epa": rng.normal(), "success": float(rng.random() < 0.45), "yards_gained": float(rng.integers(-3, 20)),
                    "sack": 0.0, "was_pressure": float(rng.random() < 0.28) if is_pass else np.nan, "call": "P_short",
                })
    return pd.DataFrame(rows)


def _prepared(df):
    d = df.copy()
    d["score_diff"] = d["score_differential"].clip(-21, 21)
    d["is_pass"] = d["play_type"] == "pass"
    d["blitz"] = (d["number_of_pass_rushers"] >= 5).astype(float).where(d["number_of_pass_rushers"].notna())
    d["big_blitz"] = (d["number_of_pass_rushers"] >= 6).astype(float).where(d["number_of_pass_rushers"].notna())
    d["man"] = (d["man_zone"] == "MAN_COVERAGE").astype(float).where(d["man_zone"].notna())
    d["explosive"] = (d["yards_gained"] >= 10).astype(float)
    return d


def test_residuals_are_centered_and_formation_confound_is_removed():
    d = add_scheme_residuals(_prepared(_plays()))
    assert abs(d["box_resid"].mean()) < 1e-9      # ridge with an intercept: exactly centered
    assert abs(d["blitz_resid"].mean()) < 1e-3     # regularized logit: centered up to a tiny shrinkage bias
    t = defense_tendencies(d).groupby("defteam")[["box_excess", "blitz_excess"]].mean()
    # D1 faces mostly shotgun (light boxes), but that is the offense's doing, not its scheme
    raw_box = d[d["defteam"] == "D1"]["defenders_in_box"].mean() - d["defenders_in_box"].mean()
    assert raw_box < -0.4
    assert abs(t.loc["D1", "box_excess"]) < abs(raw_box) / 3
    # D0's real blitz taste shows up
    assert t["blitz_excess"].idxmax() == "D0" and t.loc["D0", "blitz_excess"] > 0.1


def test_loo_tendency_excludes_the_game_itself():
    d = pd.DataFrame({
        "defteam": "A", "season": 2020, "game_id": ["g1"] * 4 + ["g2"] * 4,
        "box_resid": [1.0] * 4 + [-1.0] * 4, "blitz_resid": np.nan, "big_blitz_resid": np.nan, "man_resid": np.nan,
    })
    loo = loo_tendencies(d, shrink=0).set_index("game_id")
    assert loo.loc["g1", "box_loo"] == pytest.approx(-1.0)   # only g2 contributes to g1's opponent view
    assert loo.loc["g2", "box_loo"] == pytest.approx(1.0)
    assert loo_tendencies(d, shrink=4).set_index("game_id").loc["g1", "box_loo"] == pytest.approx(-4 / 8)


def test_scheme_tendency_is_reliable_when_real_and_not_when_absent():
    real = split_half_reliability(add_scheme_residuals(_prepared(_plays())), cols=("blitz_resid",), n_splits=8)
    null = split_half_reliability(add_scheme_residuals(_prepared(_plays(blitz_taste=False, seed=1))), cols=("blitz_resid",), n_splits=8)
    assert real["half_corr"].iloc[0] > 0.3 > null["half_corr"].iloc[0] + 0.15


def test_archetype_study_finds_two_blobs_stable_and_not_five():
    rng = np.random.default_rng(0)
    n = 320
    grp = rng.random(n) < 0.4
    tend = pd.DataFrame({
        "season": rng.integers(2016, 2026, n),
        "box_excess": np.where(grp, 0.3, -0.2) + rng.normal(0, 0.06, n),
        "blitz_excess": np.where(grp, 0.15, -0.1) + rng.normal(0, 0.04, n),
        "big_blitz_excess": np.where(grp, 0.08, -0.05) + rng.normal(0, 0.03, n),
    })
    s = archetype_study(tend, SCHEME, ks=[2, 5], n_boot=10).set_index("k")
    assert s.loc[2, "bootstrap_ari"] > 0.95 and s.loc[2, "temporal_ari"] > 0.95
    assert s.loc[5, "bootstrap_ari"] < s.loc[2, "bootstrap_ari"]   # splitting a clean blob is less reproducible
    tend["blitz_excess"] = tend["blitz_excess"].astype(float)
    labels = two_way_archetypes(tend)
    assert (labels[grp] == "Aggressive").mean() > 0.95


# ------------------------------------------------------------------- interactions
def _drives(interaction: float, n=6000, seed=0):
    rng = np.random.default_rng(seed)
    styles = ["A", "B", "C"]
    d = pd.DataFrame({
        "season": rng.integers(2016, 2022, n), "game_id": rng.integers(0, 700, n).astype(str),
        "style": rng.choice(styles, n), "qtr": rng.integers(1, 5, n),
        "start_yardline_100": 75 + rng.normal(size=n) * 8, "score_diff": rng.normal(size=n) * 7,
        "half_secs": rng.uniform(0, 1800, n), "wp_start": 0.5 + rng.normal(size=n) * 0.1,
        "off_strength": rng.normal(size=n) * 0.3, "def_strength": rng.normal(size=n) * 0.3,
        "blitz_loo": rng.normal(size=n), "box_loo": rng.normal(size=n),
    })
    d["points"] = (2 + 0.5 * (d["style"] == "B") + interaction * (d["style"] == "B") * d["blitz_loo"]
                   + 0.3 * d["blitz_loo"] + rng.normal(size=n) * 2.0)
    return d, styles


def test_interaction_test_detects_a_real_interaction_and_not_a_fake_one():
    d, styles = _drives(interaction=0.6)
    real = cv_summary(d, season_cv(d, styles, MODERATORS)).set_index("spec")
    assert real.loc["+ style x opponent scheme", "lo"] > 0          # held-out error genuinely improves
    d0, styles0 = _drives(interaction=0.0, seed=1)
    null = cv_summary(d0, season_cv(d0, styles0, MODERATORS)).set_index("spec")
    assert null.loc["+ style x opponent scheme", "lo"] <= 0.001       # no phantom heterogeneity


# --------------------------------------------------------------------- situations
def _call_plays(n=20000, seed=0):
    """EPA depends on the call (deep +0.5) AND on down/distance; deep is called more on long yardage,
    which would fool a naive comparison."""
    rng = np.random.default_rng(seed)
    ydstogo = rng.integers(1, 15, n).astype(float)
    p_deep = np.clip(0.05 + 0.03 * ydstogo, 0, 0.6)
    call = np.where(rng.random(n) < p_deep, "P_deep", rng.choice(["R_in", "R_out", "P_short", "P_mid"], n))
    epa = -0.05 * ydstogo + 0.5 * (call == "P_deep") + rng.normal(0, 1, n)
    return pd.DataFrame({
        "call": call, "epa": epa, "success": (epa > 0).astype(float), "ydstogo": ydstogo,
        "yardline_100": rng.integers(5, 95, n).astype(float), "score_diff": rng.integers(-14, 14, n).astype(float),
        "half_secs": rng.integers(0, 1800, n).astype(float), "down": rng.integers(1, 5, n).astype(float),
        "qtr": rng.integers(1, 5, n).astype(float), "season": 2020.0, "game_id": rng.integers(0, 500, n).astype(str),
    })


def test_call_effect_recovers_planted_deep_advantage_despite_confounding():
    p = _call_plays()
    naive = p.groupby("call")["epa"].mean()
    eff = adjusted_call_effects(p, CALLS).set_index("call")["effect"]
    assert naive["P_deep"] - naive["P_short"] < 0.5 - 0.1      # naive gap understated: deep is called when it is hard
    assert eff["P_deep"] > naive["P_deep"] - naive.mean() + 0.05  # adjustment raises it toward the truth
    assert abs((eff["P_deep"] - eff["P_short"]) - 0.5) < 0.1


def test_situations_and_looks_are_well_formed():
    p = add_looks(_prepared(_plays()))
    p["call"] = "P_short"
    assert set(add_looks(p)["look_blitz"].dropna().unique()) <= {0.0, 1.0}
    assert set(SITUATIONS) >= {"all plays", "red zone (inside the 20)"}
    mix = call_mix(_call_plays().assign(yardline_100=10.0))
    assert np.allclose(mix.sum(axis=1), 1.0)
    p.loc[p.index[:50], "call"] = "SACK"
    sacks = sack_rate_by_look(p.assign(call=p["call"].where(p["call"] == "SACK", DEPTHS[0])), "look_blitz", {0.0: "no", 1.0: "yes"})
    assert sacks["dropbacks"].sum() <= len(p) and sacks["sack_rate"].between(0, 1).all()


# ------------------------------------------------------------------- real artifacts
needs_reports = pytest.mark.skipif(
    not (REPORTS_DIR / "defense_season_scheme.csv").exists(), reason="run drivedna.analysis.run_defenses"
)


@needs_reports
def test_defense_report_covers_every_defense_season_and_scheme_is_reliable():
    t = pd.read_csv(REPORTS_DIR / "defense_season_scheme.csv")
    assert len(t) == 320 and not t.duplicated(["defteam", "season"]).any()
    assert set(t["archetype"]) == {"Aggressive", "Light"}
    assert t.groupby("archetype")["blitz_excess"].mean().idxmax() == "Aggressive"
    rel = pd.read_csv(REPORTS_DIR / "defense_reliability.csv").set_index("tendency")
    assert (rel["reliability"] > 0.7).all()


@needs_reports
def test_interaction_and_situation_reports_are_well_formed():
    cv = pd.read_csv(REPORTS_DIR / "interaction_cv.csv")
    assert list(cv["spec"]) == ["controls only", "+ opening style", "+ style x opponent scheme"]
    sit = pd.read_csv(REPORTS_DIR / "situation_effects_epa.csv")
    assert (sit["lo"] <= sit["hi"]).all() and set(sit["call"]) == set(CALLS)
    assert (PROCESSED_DIR / "drive_styles.parquet").exists()
