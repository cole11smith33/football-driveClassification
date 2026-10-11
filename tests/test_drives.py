"""Invariants and sanity bands for the drive tables. Requires `python -m drivedna.drives.build`."""
from pathlib import Path

import polars as pl
import pytest

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
KEY = ["game_id", "posteam", "fixed_drive"]
VALID_RESULTS = {
    "Touchdown", "Field goal", "Missed field goal", "Punt", "Turnover",
    "Turnover on downs", "End of half", "Opp touchdown", "Safety",
}

pytestmark = pytest.mark.skipif(
    not (PROCESSED / "drives.parquet").exists(), reason="run drivedna.drives.build first"
)


@pytest.fixture(scope="module")
def drives():
    return pl.read_parquet(PROCESSED / "drives.parquet")


@pytest.fixture(scope="module")
def plays():
    return pl.read_parquet(PROCESSED / "drive_plays.parquet")


def test_one_row_per_drive(drives):
    assert drives.select(KEY).is_duplicated().sum() == 0


def test_results_are_valid(drives):
    assert set(drives["result"].unique().to_list()) <= VALID_RESULTS


def test_every_play_belongs_to_exactly_one_drive(drives, plays):
    assert plays.select(["game_id", "play_id"]).is_duplicated().sum() == 0
    orphan = plays.select(KEY).unique().join(drives.select(KEY), on=KEY, how="anti")
    assert orphan.height == 0


def test_plays_are_ordered_within_drive(plays):
    check = plays.group_by(KEY).agg(
        ordered=(pl.col("play_id").diff().drop_nulls() > 0).all(),
        contiguous=pl.col("play_num").max() == pl.len(),
    )
    assert check["ordered"].all() and check["contiguous"].all()


def test_n_plays_matches_play_table(drives, plays):
    counts = plays.group_by(KEY).agg(n=pl.len())
    joined = drives.join(counts, on=KEY)
    assert (joined["n_plays"] == joined["n"]).all()
    # nflverse's own count also includes penalties/special teams, so ours should not exceed it.
    # A handful (<0.1%) of upstream counts are simply wrong, so allow that much slack.
    over = (drives["n_plays"] > drives["nflverse_play_count"].fill_null(999)).mean()
    assert over < 0.001, over


def test_keep_flag_has_no_nulls(drives):
    assert drives["keep"].null_count() == 0


def test_per_season_sanity_bands(drives):
    by_season = drives.group_by("season").agg(
        n=pl.len(),
        td=(pl.col("result") == "Touchdown").mean(),
        punt=(pl.col("result") == "Punt").mean(),
    )
    for row in by_season.iter_rows(named=True):
        assert 5700 <= row["n"] <= 6500, row
        assert 0.18 <= row["td"] <= 0.27, row
        assert 0.32 <= row["punt"] <= 0.42, row
