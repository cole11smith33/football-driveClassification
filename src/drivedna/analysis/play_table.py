"""Play-level table for defense and situation analyses.

Designed run/pass plays from non-garbage-time drives, with the call type (token), play context,
results, and the defensive look from nflverse participation data. Kneels, spikes and two-point
tries are excluded upstream. `man_zone` / `coverage` exist from 2018 only.
"""
import polars as pl

from drivedna.data.ingest import DEFAULT_SEASONS, participation_path
from drivedna.drives.build import DRIVE_KEY, PROCESSED_DIR
from drivedna.features.tokens import token_expr

PLAY_KEEP = [
    "game_id", "play_id", "season", "week", "posteam", "defteam", "fixed_drive", "play_num",
    "qtr", "down", "ydstogo", "yardline_100", "game_seconds_remaining", "score_differential", "wp",
    "play_type", "pass", "sack", "qb_scramble", "air_yards", "epa", "success",
    "yards_gained", "first_down",
]


def load_participation(seasons=DEFAULT_SEASONS) -> pl.DataFrame:
    frames = [pl.read_parquet(participation_path(s)) for s in seasons]
    return pl.concat(frames, how="diagonal_relaxed").rename({"nflverse_game_id": "game_id"})


def build_play_table(seasons=DEFAULT_SEASONS) -> pl.DataFrame:
    drives = pl.read_parquet(PROCESSED_DIR / "drives.parquet").select(*DRIVE_KEY, "keep")
    plays = pl.read_parquet(PROCESSED_DIR / "drive_plays.parquet")
    plays = (
        plays.filter(pl.col("play_type").is_in(["pass", "run"]) & pl.col("season").is_in(list(seasons)))
        .join(drives, on=DRIVE_KEY)
        .filter(pl.col("keep"))
        .with_columns(call=token_expr())
        .select(*PLAY_KEEP, "call")
    )
    part = load_participation(seasons).rename(
        {"defense_man_zone_type": "man_zone", "defense_coverage_type": "coverage"}
    )
    out = plays.join(part, on=["game_id", "play_id"], how="left")
    # half-seconds left: OT keeps the period clock; otherwise seconds left in the current half
    return out.with_columns(
        half_secs=pl.when(pl.col("qtr") <= 2).then(pl.col("game_seconds_remaining") - 1800)
        .otherwise(pl.col("game_seconds_remaining")).clip(lower_bound=0),
        man_zone=pl.when(pl.col("man_zone").is_in(["MAN_COVERAGE", "ZONE_COVERAGE"])).then(pl.col("man_zone")),
    )


if __name__ == "__main__":
    t = build_play_table()
    print(t.shape)
