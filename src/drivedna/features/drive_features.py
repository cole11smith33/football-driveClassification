"""One feature row per drive, in three deliberately separate groups.

STYLE_FEATURES   how the offense chose to play (calls, depth, tempo). The only inputs to clustering.
RHYTHM_FEATURES  how the drive unfolded (success, explosives, 3rd downs). These are partly
                 results, so they describe and explain styles but are never clustered on.
CONTEXT_COLS     game situation to control for later (field position, score, time, win prob).
Outcomes live in features/outcomes.py.

Note: nflverse's `pass` flag is 1 for every QB dropback, including scrambles (which are
`play_type == "run"`). So `pass_rate`, `early_down_pass_rate` and `proe` are *dropback* rates, which
is what nflverse's `xpass` models. `scramble_rate` separates out how many dropbacks became runs.
"""
from pathlib import Path

import polars as pl

from drivedna.drives.build import DRIVE_KEY, PROCESSED_DIR
from drivedna.features.outcomes import OUTCOME_COLS, add_outcomes

STYLE_FEATURES = [
    "pass_rate", "early_down_pass_rate", "proe", "mean_air_yards", "deep_pass_share",
    "short_pass_share", "sideline_pass_share", "scramble_rate", "inside_run_share",
    "shotgun_rate", "no_huddle_rate", "sec_per_play",
]
RHYTHM_FEATURES = [
    "third_down_share", "mean_third_ydstogo", "early_down_success_rate", "explosive_rate",
    "negative_play_rate", "epa_per_play",
]
CONTEXT_COLS = [
    "start_yardline_100", "score_diff_start", "wp_start", "qtr_start", "start_seconds_remaining",
    "defteam",
]
FLAG_COLS = ["keep", "is_kneel_only", "is_end_of_half", "is_garbage_time", "is_too_short"]

MAX_VALID_GAP_SEC = 60  # longer gaps are timeouts/injuries/challenges, not tempo


def _real_plays(drive_plays: pl.DataFrame) -> pl.DataFrame:
    """Designed run/pass plays only: kneels and spikes say nothing about play-calling style."""
    plays = drive_plays.filter(pl.col("play_type").is_in(["pass", "run"])).sort(
        DRIVE_KEY + ["play_num"]
    )
    gap = (
        pl.col("game_seconds_remaining").shift(1) - pl.col("game_seconds_remaining")
    ).over(DRIVE_KEY)
    return plays.with_columns(
        gap_sec=pl.when((gap > 0) & (gap <= MAX_VALID_GAP_SEC)).then(gap),
        is_pass=pl.col("pass") == 1,
        is_run=(pl.col("play_type") == "run") & (pl.col("qb_scramble").fill_null(0) == 0),
        early=pl.col("down") <= 2,
    )


def build_style_and_rhythm(drive_plays: pl.DataFrame) -> pl.DataFrame:
    p = _real_plays(drive_plays)
    passes_with_depth = pl.col("is_pass") & pl.col("air_yards").is_not_null()
    inside_run = (pl.col("run_gap") == "guard") | pl.col("run_gap").is_null()
    # scrambles are runs for yardage purposes, even though `pass` flags them as dropbacks
    explosive = pl.when(pl.col("play_type") == "pass").then(pl.col("yards_gained") >= 20).otherwise(
        pl.col("yards_gained") >= 10
    )
    return p.group_by(DRIVE_KEY, maintain_order=True).agg(
        n_real_plays=pl.len(),
        # --- style
        pass_rate=pl.col("is_pass").mean(),
        early_down_pass_rate=pl.col("is_pass").filter(pl.col("early")).mean(),
        proe=(pl.col("pass") - pl.col("xpass")).mean(),
        mean_air_yards=pl.col("air_yards").filter(passes_with_depth).mean(),
        deep_pass_share=(pl.col("air_yards") >= 15).filter(passes_with_depth).mean(),
        short_pass_share=(pl.col("air_yards") <= 5).filter(passes_with_depth).mean(),
        sideline_pass_share=(pl.col("pass_location") != "middle").filter(
            pl.col("is_pass") & pl.col("pass_location").is_not_null()
        ).mean(),
        scramble_rate=(pl.col("qb_scramble").fill_null(0) == 1).mean(),
        inside_run_share=inside_run.filter(pl.col("is_run")).mean(),
        shotgun_rate=pl.col("shotgun").fill_null(0).mean(),
        no_huddle_rate=pl.col("no_huddle").fill_null(0).mean(),
        sec_per_play=pl.col("gap_sec").median(),
        # --- rhythm (partly results; never clustered on)
        third_down_share=(pl.col("down") == 3).mean(),
        mean_third_ydstogo=pl.col("ydstogo").filter(pl.col("down") == 3).mean(),
        early_down_success_rate=pl.col("success").filter(pl.col("early")).mean(),
        explosive_rate=explosive.mean(),
        negative_play_rate=((pl.col("yards_gained") < 0) | (pl.col("sack") == 1)).mean(),
        epa_per_play=pl.col("epa").mean(),
    )


def build_feature_table(drives: pl.DataFrame, drive_plays: pl.DataFrame) -> pl.DataFrame:
    """Join identifiers, flags, context, outcomes and features: one row per drive."""
    feats = build_style_and_rhythm(drive_plays)
    ids = ["season", "week", "season_type", "n_plays"]
    base = add_outcomes(drives).select(DRIVE_KEY + ids + FLAG_COLS + CONTEXT_COLS + OUTCOME_COLS)
    return base.join(feats, on=DRIVE_KEY, how="left")


def build_and_save(processed_dir: Path = PROCESSED_DIR) -> pl.DataFrame:
    drives = pl.read_parquet(processed_dir / "drives.parquet")
    plays = pl.read_parquet(processed_dir / "drive_plays.parquet")
    table = build_feature_table(drives, plays)
    table.write_parquet(processed_dir / "drive_features.parquet")
    return table


if __name__ == "__main__":
    t = build_and_save()
    print(t.shape)
