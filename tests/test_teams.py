import numpy as np
import pandas as pd
import pytest

from drivedna.analysis.openings import OPENING_ORDER
from drivedna.analysis.teams import (
    change_events,
    decompose_performance,
    drive_table,
    event_summary,
    excess_columns,
    expected_style_probs,
    nearest_neighbors,
    robust_z,
    similarity_matrix,
    split_half_reliability,
    team_dna,
    team_season_meta,
)
from drivedna.drives.build import PROCESSED_DIR
from drivedna.models.cluster import REPORTS_DIR, STYLE_ORDER


def _simulate(n_teams=24, seasons=(2016, 2017, 2018, 2019), per=160, seed=0, tendency=True):
    """Drives for many teams. Style depends on score (trailing -> Hurry-up). Team 0 trails much more
    (a pure game-script effect); team 2 genuinely favors Vertical (a real tendency)."""
    rng = np.random.default_rng(seed)
    rows = []
    for team in range(n_teams):
        for season in seasons:
            mu = -6.0 if team == 0 else 0.0
            for i in range(per):
                score = float(np.clip(rng.normal(mu, 8), -21, 21))
                logits = np.zeros(len(STYLE_ORDER))
                logits[STYLE_ORDER.index("Hurry-up")] = -0.12 * score - 1.0
                if tendency and team == 2:
                    logits[STYLE_ORDER.index("Vertical")] += 1.2
                if tendency and team >= 3:  # mild persistent team tastes give the league real between-team spread
                    logits[(team * 7) % len(STYLE_ORDER)] += 0.6
                p = np.exp(logits) / np.exp(logits).sum()
                rows.append({
                    "team": f"T{team}", "season": season, "game_id": f"{team}_{season}_{i // 10}",
                    "style": rng.choice(STYLE_ORDER, p=p), "open_style": rng.choice(OPENING_ORDER),
                    "qtr": int(rng.integers(1, 5)), "start_yardline_100": 75 + rng.normal() * 8,
                    "score_diff": score, "half_secs": rng.uniform(0, 1800), "wp_start": 0.5 + score / 80,
                    "def_strength": rng.normal() * 0.2,
                })
    return pd.DataFrame(rows)


def test_expected_probs_sum_to_one_and_excess_sums_to_zero():
    df = _simulate(n_teams=6, per=60)
    p = expected_style_probs(df, "style", STYLE_ORDER)
    assert np.allclose(p.sum(axis=1), 1.0)
    dna = team_dna(df)
    drive_cols = [c for c in excess_columns(dna) if c.startswith("drive_")]
    assert np.allclose(dna[drive_cols].sum(axis=1), 0.0, atol=1e-9)


def test_script_adjustment_does_not_mistake_trailing_for_a_taste_for_hurry_up():
    dna = team_dna(_simulate(tendency=False)).groupby("team").mean(numeric_only=True)
    raw_gap = dna.loc["T0", "drive_Hurry-up_actual"] - dna.drop("T0")["drive_Hurry-up_actual"].mean()
    adj_gap = dna.loc["T0", "drive_Hurry-up_excess"] - dna.drop("T0")["drive_Hurry-up_excess"].mean()
    assert raw_gap > 0.05            # trailing a lot really does raise the raw hurry-up share
    assert abs(adj_gap) < raw_gap / 3  # ... and the adjusted tendency removes most of it


def test_planted_team_tendency_is_recovered():
    dna = team_dna(_simulate()).groupby("team").mean(numeric_only=True)
    ex = dna["drive_Vertical_excess"]
    assert ex.idxmax() == "T2" and ex["T2"] > 0.08


def test_split_half_reliability_separates_signal_from_noise():
    real = split_half_reliability(_simulate(tendency=True), n_splits=10).set_index("style")
    null = split_half_reliability(_simulate(tendency=False, seed=1), n_splits=10).set_index("style")
    assert real.loc["Vertical", "half_corr"] > 0.4
    assert null[null["kind"] == "drive"]["half_corr"].abs().max() < 0.45  # nothing but noise (T0 aside)


def test_robust_z_clips_outliers_and_similarity_is_a_proper_cosine():
    rng = np.random.default_rng(0)
    dna = pd.DataFrame({"team": [f"T{i}" for i in range(30)], "season": 2020, "a_excess": rng.normal(size=30),
                        "b_excess": rng.normal(size=30)})
    dna.loc[0, "a_excess"] = 100.0
    z = robust_z(dna, ["a_excess", "b_excess"])
    assert z["a_excess"].max() == 5.0
    sim = similarity_matrix(dna, ["a_excess", "b_excess"])
    assert np.allclose(np.diag(sim.to_numpy()), 1.0) and np.allclose(sim, sim.T)
    assert sim.to_numpy().min() >= -1 - 1e-9
    assert all(not i.startswith("T0 ") for i in nearest_neighbors(sim, "T0 2020", 5).index)


def test_change_events_label_and_summarize():
    seasons = [2018, 2019, 2020, 2021]
    rows, meta = [], []
    for team, shift_at in [("A", 2020), ("B", None), ("C", None), ("D", None)]:
        for s in seasons:
            x = 0.0 if (shift_at is None or s < shift_at) else 3.0
            rows.append({"team": team, "season": s, "x_excess": x + 0.1 * (s % 2)})
            changed = float(s == shift_at)
            meta.append({"team": team, "season": s, "coach": f"{team}{int(s >= (shift_at or 9999))}", "qb": "q",
                         "hc_changed": np.nan if s == 2018 else changed, "qb_changed": np.nan if s == 2018 else 0.0})
    dna, meta = pd.DataFrame(rows), pd.DataFrame(meta)
    ev = change_events(dna, meta, ["x_excess"])
    assert set(ev["event"]) == {"no change", "new coach only"}
    assert ev.query("event == 'new coach only'")["distance"].iloc[0] > ev.query("event == 'no change'")["distance"].mean()
    summ = event_summary(ev, n_boot=200).set_index("event")
    assert summ.loc["new coach only", "diff_vs_no_change"] > 0


# ------------------------------------------------------------------- real data
needs_data = pytest.mark.skipif(
    not (PROCESSED_DIR / "drive_styles.parquet").exists(), reason="run drivedna.models.cluster"
)


@needs_data
def test_meta_flags_match_known_history():
    meta = team_season_meta().set_index(["team", "season"])
    assert len(meta) == 320
    assert meta.loc[("KC", 2019), "hc_changed"] == 0 and meta.loc[("KC", 2018), "qb_changed"] == 1  # Mahomes replaces Smith
    assert meta.loc[("KC", 2016), "coach"] == "Andy Reid" and np.isnan(meta.loc[("KC", 2016), "hc_changed"])


@needs_data
def test_decomposition_parts_sum_to_the_total():
    dec = decompose_performance(drive_table())
    assert dec["residual_check"].abs().max() < 1e-6
    assert abs(dec["exec_part"].mean()) < 0.05
    assert dec["exec_part"].std() > dec["style_part"].std()  # execution varies far more than style choice


report = REPORTS_DIR / "team_season_dna.csv"


@pytest.mark.skipif(not report.exists(), reason="run drivedna.analysis.run_teams")
def test_team_report_has_every_team_season_with_reasonable_sample_sizes():
    dna = pd.read_csv(report)
    assert len(dna) == 320 and not dna.duplicated(["team", "season"]).any()
    assert dna["n_drives"].between(80, 220).all()
    rel = pd.read_csv(REPORTS_DIR / "dna_reliability.csv")
    assert rel.query("kind == 'drive' and style == 'No-huddle'")["reliability"].iloc[0] > 0.8  # known play-caller-driven trait
