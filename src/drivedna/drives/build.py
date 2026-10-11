"""Turn raw play-by-play into a drive table and an ordered drive-plays table."""

import polars as pl

from drivedna.data.ingest import DEFAULT_SEASONS, RAW_DIR, pbp_path
from drivedna.drives.filters import add_drive_flags

PROCESSED_DIR = RAW_DIR.parent / "processed"

# Scrimmage plays that belong to a drive's offensive script.
SCRIMMAGE_TYPES = ["pass", "run", "qb_kneel", "qb_spike"]
# Kept columns for drive_plays (style features + context; outcome cols are separate).
PLAY_COLS = [
    "game_id", "season", "week", "season_type", "posteam", "defteam", "fixed_drive", "play_id",
    "qtr", "down", "ydstogo", "yardline_100", "game_seconds_remaining", "score_differential",
    "wp", "play_type", "pass", "rush", "qb_kneel", "qb_spike", "qb_scramble", "sack",
    "penalty", "air_yards", "pass_location", "run_location", "run_gap", "shotgun", "no_huddle",
    "xpass", "success", "epa", "yards_gained", "touchdown", "interception", "fumble_lost",
    "complete_pass", "yards_after_catch", "first_down",
]
DRIVE_KEY = ["game_id", "posteam", "fixed_drive"]


def load_pbp(seasons=DEFAULT_SEASONS) -> pl.DataFrame:
    frames = [pl.read_parquet(pbp_path(s)) for s in seasons]
    return pl.concat(frames, how="diagonal_relaxed")


def build_drive_plays(pbp: pl.DataFrame) -> pl.DataFrame:
    """Ordered scrimmage plays per drive, with a within-drive play number."""
    plays = (
        pbp.filter(
            pl.col("posteam").is_not_null()
            & pl.col("fixed_drive").is_not_null()
            & pl.col("play_type").is_in(SCRIMMAGE_TYPES)
            & (pl.col("two_point_attempt").fill_null(0) == 0)  # post-TD tries aren't part of the drive
        )
        .select(PLAY_COLS)
        .sort(["game_id", "play_id"])
    )
    return plays.with_columns(play_num=pl.int_range(pl.len()).over(DRIVE_KEY) + 1)


def build_drives(pbp: pl.DataFrame, drive_plays: pl.DataFrame) -> pl.DataFrame:
    """One row per (game, offense, fixed_drive). Result comes from nflverse's fixed_drive_result."""
    results = (
        pbp.filter(pl.col("posteam").is_not_null() & pl.col("fixed_drive").is_not_null())
        .group_by(DRIVE_KEY)
        .agg(
            result=pl.col("fixed_drive_result").drop_nulls().first(),
            nflverse_play_count=pl.col("drive_play_count").drop_nulls().first(),
            start_text=pl.col("drive_start_yard_line").drop_nulls().first(),
        )
    )
    agg = (
        drive_plays.group_by(DRIVE_KEY, maintain_order=True)
        .agg(
            season=pl.col("season").first(),
            week=pl.col("week").first(),
            season_type=pl.col("season_type").first(),
            defteam=pl.col("defteam").first(),
            qtr_start=pl.col("qtr").first(),
            start_yardline_100=pl.col("yardline_100").first(),
            start_seconds_remaining=pl.col("game_seconds_remaining").first(),
            end_seconds_remaining=pl.col("game_seconds_remaining").last(),
            score_diff_start=pl.col("score_differential").first(),
            wp_start=pl.col("wp").first(),
            n_plays=pl.len(),
            n_kneel=pl.col("qb_kneel").fill_null(0).sum(),
            yards=pl.col("yards_gained").fill_null(0).sum(),
            epa=pl.col("epa").fill_null(0).sum(),
        )
        .with_columns(duration_sec=pl.col("start_seconds_remaining") - pl.col("end_seconds_remaining"))
    )
    drives = agg.join(results.drop("start_text"), on=DRIVE_KEY, how="left")
    return add_drive_flags(drives)


def build_all(seasons=DEFAULT_SEASONS) -> tuple[pl.DataFrame, pl.DataFrame]:
    pbp = load_pbp(seasons)
    plays = build_drive_plays(pbp)
    drives = build_drives(pbp, plays)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    drives.write_parquet(PROCESSED_DIR / "drives.parquet")
    plays.write_parquet(PROCESSED_DIR / "drive_plays.parquet")
    return drives, plays


if __name__ == "__main__":
    drives, plays = build_all()
    print(drives.shape, plays.shape)
