import json

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.metrics import adjusted_rand_score

from drivedna.drives.build import DRIVE_KEY, PROCESSED_DIR
from drivedna.features.drive_features import STYLE_FEATURES
from drivedna.models.cluster import (
    MIN_REAL_PLAYS,
    REPORTS_DIR,
    STYLE_ORDER,
    StyleModel,
    eligible_drives,
    name_clusters,
)

K = len(STYLE_ORDER)


def _synthetic(n_per=300, n_plays=8, seed=0):
    """Well-separated blobs in feature space, built with the real column names."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0.1, 0.9, size=(K, len(STYLE_FEATURES)))
    rows, truth = [], []
    for c in range(K):
        rows.append(centers[c] + rng.normal(0, 0.01, size=(n_per, len(STYLE_FEATURES))))
        truth += [c] * n_per
    df = pl.DataFrame(np.vstack(rows), schema=STYLE_FEATURES).with_columns(
        n_real_plays=pl.lit(n_plays).cast(pl.Int64)
    )
    return df, np.array(truth)


def test_recovers_well_separated_clusters():
    df, truth = _synthetic()
    pred = StyleModel().fit(df).predict(df)
    assert adjusted_rand_score(truth, pred) > 0.99


def test_fit_is_deterministic_for_a_seed():
    df, _ = _synthetic()
    a = StyleModel(seed=3).fit(df).predict(df)
    b = StyleModel(seed=3).fit(df).predict(df)
    assert (a == b).all()


def test_predict_works_on_unseen_drives_and_handles_nulls():
    df, _ = _synthetic()
    model = StyleModel().fit(df)
    new = df.head(20).with_columns(mean_air_yards=pl.lit(None, dtype=pl.Float64))
    pred = model.predict(new)
    assert len(pred) == 20 and set(pred) <= set(range(K))


def test_short_drives_are_shrunk_toward_the_league_mean():
    df, _ = _synthetic()
    model = StyleModel().fit(df)
    x, _ = model._raw(df.head(1))
    short = model._shrunk(x, np.array([3.0]))
    long = model._shrunk(x, np.array([30.0]))
    mean = model.league_mean_
    assert np.abs(short - mean).sum() < np.abs(long - mean).sum()
    # weight n/(n+k): a 3-play drive keeps 3/7 of its distance from the mean
    assert np.allclose(short - mean, (x - mean) * 3 / 7, atol=1e-9)


def test_margin_is_in_unit_interval():
    df, _ = _synthetic()
    m = StyleModel().fit(df).margin(df)
    assert m.min() >= 0 and m.max() <= 1


def test_model_never_sees_outcome_or_rhythm_columns():
    from drivedna.features.drive_features import CONTEXT_COLS, RHYTHM_FEATURES
    from drivedna.features.outcomes import OUTCOME_COLS

    banned = set(OUTCOME_COLS) | set(RHYTHM_FEATURES) | set(CONTEXT_COLS)
    assert not banned & set(StyleModel().features)
    assert "scramble_rate" not in StyleModel().features


def test_naming_follows_clusters_not_labels():
    cols = STYLE_FEATURES
    base = dict.fromkeys(cols, 0.5) | {"sec_per_play": 33.0, "pass_rate": 0.6,
                                      "no_huddle_rate": 0.04, "deep_pass_share": 0.1}
    rows = {
        7: base | {"no_huddle_rate": 0.5},
        3: base | {"deep_pass_share": 0.55},
        9: base | {"sec_per_play": 17.0, "pass_rate": 0.9},
        1: base | {"pass_rate": 0.35},
        4: base,
    }
    names = name_clusters(pd.DataFrame.from_dict(rows, orient="index"))
    assert names == {7: "No-huddle", 3: "Vertical", 9: "Hurry-up", 1: "Ground-based",
                     4: "Quick-game"}
    assert set(names.values()) == set(STYLE_ORDER)


# ---------------------------------------------------------------- real-data artifact checks
styles_path = PROCESSED_DIR / "drive_styles.parquet"
needs_styles = pytest.mark.skipif(not styles_path.exists(), reason="run drivedna.models.cluster")


@pytest.fixture(scope="module")
def styles():
    return pl.read_parquet(styles_path)


@pytest.fixture(scope="module")
def joined(styles):
    feats = eligible_drives(pl.read_parquet(PROCESSED_DIR / "drive_features.parquet"))
    return feats.join(styles, on=DRIVE_KEY)


@needs_styles
def test_every_eligible_drive_has_exactly_one_style(styles):
    feats = eligible_drives(pl.read_parquet(PROCESSED_DIR / "drive_features.parquet"))
    assert styles.height == feats.height
    assert styles.select(DRIVE_KEY).is_duplicated().sum() == 0
    assert set(styles["style"].unique()) == set(STYLE_ORDER)
    assert (feats["n_real_plays"] >= MIN_REAL_PLAYS).all()


@needs_styles
def test_saved_model_loads_and_reproduces_the_style_table(styles):
    import joblib

    model = joblib.load(PROCESSED_DIR / "style_model.joblib")
    assert type(model).__module__ == "drivedna.models.cluster"
    feats = eligible_drives(pl.read_parquet(PROCESSED_DIR / "drive_features.parquet"))
    sample = feats.head(2000)
    expected = sample.join(styles, on=DRIVE_KEY)["style_id"].to_numpy()
    assert (model.predict(sample) == expected).all()


@needs_styles
def test_no_tiny_or_dominant_styles(styles):
    shares = styles["style"].value_counts()["count"] / styles.height
    assert shares.min() > 0.05 and shares.max() < 0.5


@needs_styles
def test_style_names_match_their_defining_traits(joined):
    """Guards against naming drift: each name must still sit at the extreme it claims."""
    means = joined.group_by("style").agg(pl.col(c).mean() for c in STYLE_FEATURES).to_pandas()
    means = means.set_index("style")
    assert means["deep_pass_share"].idxmax() == "Vertical"
    assert means["mean_air_yards"].idxmax() == "Vertical"
    assert means["sec_per_play"].idxmin() == "Hurry-up"
    assert means["no_huddle_rate"].idxmax() == "No-huddle"
    assert means.loc["No-huddle", "no_huddle_rate"] > 0.4
    assert means.loc["Hurry-up", "pass_rate"] > 0.8  # almost all passing
    assert means["pass_rate"].idxmin() == "Ground-based"
    assert means["short_pass_share"].idxmax() in {"Quick-game", "Ground-based"}


@needs_styles
def test_styles_are_not_mostly_a_drive_length_artifact(joined):
    """Share of drive-length variance explained by style (eta squared) stays small (~2% today)."""
    y = joined["n_real_plays"].to_numpy().astype(float)
    group_means = joined.group_by("style").agg(m=pl.col("n_real_plays").mean(), n=pl.len())
    ss_between = sum(r["n"] * (r["m"] - y.mean()) ** 2 for r in group_means.iter_rows(named=True))
    assert ss_between / ((y - y.mean()) ** 2).sum() < 0.05


stability_path = REPORTS_DIR / "stability.json"


@pytest.mark.skipif(not stability_path.exists(), reason="run drivedna.models.cluster")
def test_stability_thresholds():
    s = json.loads(stability_path.read_text())
    assert s["bootstrap"]["ari_mean"] > 0.85
    assert s["bootstrap"]["frac_below_0_8"] < 0.2  # refits landing on an alternative partition
    assert s["temporal"]["ari_early_model_vs_late_model_on_late_drives"] > 0.8
    assert s["temporal"]["agreement_rate"] > 0.9
