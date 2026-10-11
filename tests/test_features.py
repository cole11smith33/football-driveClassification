import polars as pl
import pytest

from drivedna.drives.build import DRIVE_KEY, PROCESSED_DIR
from drivedna.features.drive_features import (
    CONTEXT_COLS,
    RHYTHM_FEATURES,
    STYLE_FEATURES,
    build_style_and_rhythm,
)
from drivedna.features.outcomes import OUTCOME_COLS, POINTS
from drivedna.features.tokens import TOKENS, add_tokens, drive_token_sequences


def _plays(rows):
    """Minimal drive_plays frame: unspecified columns default to None."""
    cols = {
        "game_id": "g", "posteam": "A", "fixed_drive": 1.0, "play_type": "run", "sack": 0,
        "qb_scramble": 0, "run_gap": None, "air_yards": None, "pass": 0,
    }
    out = [{**cols, **r, "play_num": i + 1, "play_id": float(i)} for i, r in enumerate(rows)]
    return pl.DataFrame(out, infer_schema_length=None)


def test_token_rules():
    plays = _plays([
        {"play_type": "run", "run_gap": "guard"},
        {"play_type": "run", "run_gap": "end"},
        {"play_type": "run", "run_gap": None},                 # middle run, no gap -> inside
        {"play_type": "run", "qb_scramble": 1, "run_gap": "end"},
        {"play_type": "pass", "pass": 1, "air_yards": 3},
        {"play_type": "pass", "pass": 1, "air_yards": 5},      # boundary: short
        {"play_type": "pass", "pass": 1, "air_yards": 6},
        {"play_type": "pass", "pass": 1, "air_yards": 14},
        {"play_type": "pass", "pass": 1, "air_yards": 15},     # boundary: deep
        {"play_type": "pass", "pass": 1, "air_yards": None},
        {"play_type": "pass", "pass": 1, "sack": 1},
        {"play_type": "qb_kneel"},
        {"play_type": "qb_spike", "pass": 1},
    ])
    got = add_tokens(plays)["token"].to_list()
    assert got == [
        "R_in", "R_out", "R_in", "SCR", "P_short", "P_short", "P_mid", "P_mid", "P_deep",
        "P_unk", "SACK", "KNEEL", "SPIKE",
    ]
    assert set(got) <= set(TOKENS)


def test_drive_token_sequence_is_ordered():
    plays = _plays([
        {"play_type": "pass", "pass": 1, "air_yards": 20},
        {"play_type": "run", "run_gap": "guard"},
    ]).sort("play_num", descending=True)  # shuffle input order
    seq = drive_token_sequences(plays)
    assert seq["tokens"][0].to_list() == ["P_deep", "R_in"]
    assert seq["token_str"][0] == "P_deep R_in"


def test_style_features_hand_computed():
    plays = _plays([
        {"play_type": "run", "run_gap": "guard", "down": 1, "game_seconds_remaining": 900},
        {"play_type": "pass", "pass": 1, "air_yards": 20, "down": 2, "pass_location": "left",
         "game_seconds_remaining": 860},
        {"play_type": "pass", "pass": 1, "air_yards": 2, "down": 3, "pass_location": "middle",
         "game_seconds_remaining": 835},
        {"play_type": "qb_kneel", "game_seconds_remaining": 800},  # ignored by style features
    ]).with_columns(
        pl.col("down").cast(pl.Float64), pl.col("game_seconds_remaining").cast(pl.Float64),
        xpass=pl.lit(0.5), shotgun=pl.lit(0.0), no_huddle=pl.lit(0.0), ydstogo=pl.lit(10.0),
        success=pl.lit(1.0), yards_gained=pl.lit(5.0), epa=pl.lit(0.0),
    )
    f = build_style_and_rhythm(plays).row(0, named=True)
    assert f["n_real_plays"] == 3
    assert f["pass_rate"] == pytest.approx(2 / 3)
    assert f["early_down_pass_rate"] == pytest.approx(0.5)  # downs 1-2: one run, one pass
    assert f["mean_air_yards"] == pytest.approx(11.0)
    assert f["deep_pass_share"] == pytest.approx(0.5)
    assert f["short_pass_share"] == pytest.approx(0.5)
    assert f["sideline_pass_share"] == pytest.approx(0.5)
    assert f["sec_per_play"] == pytest.approx(32.5)  # median of gaps 40 and 25
    assert f["third_down_share"] == pytest.approx(1 / 3)


def test_scramble_is_a_dropback_but_not_an_inside_run_or_pass_depth():
    plays = _plays([
        {"play_type": "run", "run_gap": "guard", "yards_gained": 4.0},
        {"play_type": "run", "qb_scramble": 1, "pass": 1, "run_gap": "end", "yards_gained": 12.0},
        {"play_type": "pass", "pass": 1, "air_yards": 8, "yards_gained": 12.0},
    ]).with_columns(
        pl.col("yards_gained").cast(pl.Float64),
        down=pl.lit(1.0), game_seconds_remaining=pl.Series([900.0, 870.0, 840.0]),
        pass_location=pl.Series([None, None, "left"], dtype=pl.String), xpass=pl.lit(0.5), shotgun=pl.lit(0.0), no_huddle=pl.lit(0.0), ydstogo=pl.lit(10.0),
        success=pl.lit(1.0), epa=pl.lit(0.0),
    )
    f = build_style_and_rhythm(plays).row(0, named=True)
    assert f["pass_rate"] == pytest.approx(2 / 3)          # scramble + pass are dropbacks
    assert f["scramble_rate"] == pytest.approx(1 / 3)
    assert f["inside_run_share"] == pytest.approx(1.0)      # the scramble is not a designed run
    assert f["mean_air_yards"] == pytest.approx(8.0)       # ... and has no pass depth
    # scramble gaining 12 is explosive (run threshold 10); the 12-yard pass is not (threshold 20)
    assert f["explosive_rate"] == pytest.approx(1 / 3)


def test_style_features_exclude_outcomes_and_results():
    leaky = set(OUTCOME_COLS) | set(RHYTHM_FEATURES) | set(CONTEXT_COLS) | {"yards_gained", "success"}
    assert not leaky & set(STYLE_FEATURES)
    assert not set(STYLE_FEATURES) & set(RHYTHM_FEATURES)


def test_points_cover_every_result():
    assert {"Touchdown", "Field goal", "Punt", "Safety", "Opp touchdown"} <= set(POINTS)


features_path = PROCESSED_DIR / "drive_features.parquet"
needs_features = pytest.mark.skipif(
    not features_path.exists(), reason="run drivedna.features.drive_features"
)


@pytest.fixture(scope="module")
def table():
    return pl.read_parquet(features_path)


@needs_features
def test_one_row_per_drive_matching_drives_table(table):
    drives = pl.read_parquet(PROCESSED_DIR / "drives.parquet")
    assert table.height == drives.height
    assert table.select(DRIVE_KEY).is_duplicated().sum() == 0


@needs_features
def test_rates_are_in_unit_interval(table):
    rates = [c for c in STYLE_FEATURES + RHYTHM_FEATURES if c.endswith(("_rate", "_share"))]
    for c in rates:
        col = table[c].drop_nulls()
        assert col.min() >= 0 and col.max() <= 1, c


@needs_features
def test_tempo_within_valid_range(table):
    assert table["sec_per_play"].drop_nulls().min() > 0
    assert table["sec_per_play"].drop_nulls().max() <= 60


@needs_features
def test_league_wide_values_are_football_shaped(table):
    kept = table.filter(pl.col("keep"))
    assert 0.55 <= kept["pass_rate"].mean() <= 0.68
    assert 1.8 <= kept["points"].mean() <= 2.5
    assert kept["points"].null_count() == 0


@needs_features
def test_depth_features_are_null_exactly_when_no_pass_has_a_recorded_depth(table):
    plays = pl.read_parquet(PROCESSED_DIR / "drive_plays.parquet")
    measured = (
        plays.filter(
            (pl.col("play_type") == "pass") & pl.col("air_yards").is_not_null()
        )
        .select(DRIVE_KEY).unique().with_columns(has_depth=pl.lit(True))
    )
    j = table.join(measured, on=DRIVE_KEY, how="left").with_columns(
        has_depth=pl.col("has_depth").fill_null(False)
    )
    for c in ["mean_air_yards", "deep_pass_share", "short_pass_share"]:
        assert (j[c].is_not_null() == j["has_depth"]).all(), c


@needs_features
def test_core_style_features_defined_for_every_real_drive(table):
    real = table.filter(pl.col("n_real_plays") >= 1)
    for c in ["pass_rate", "proe", "shotgun_rate", "no_huddle_rate", "scramble_rate"]:
        assert real[c].null_count() == 0, c
