"""Build the two compact tables the app reads: eligible drives (with styles) and their plays."""
import pandas as pd
import polars as pl

from drivedna.analysis.openings import build_opening_styles
from drivedna.analysis.sequences import designed_calls
from drivedna.drives.build import DRIVE_KEY, PROCESSED_DIR
from drivedna.features.drive_features import STYLE_FEATURES
from drivedna.features.tokens import add_tokens, drive_token_sequences
from drivedna.models.cluster import eligible_drives

APP_DRIVES = PROCESSED_DIR / "app_drives.parquet"
APP_PLAYS = PROCESSED_DIR / "app_plays.parquet"

DRIVE_COLS = [
    *DRIVE_KEY, "season", "week", "season_type", "defteam", "result", "points", "scored", "yards", "epa",
    "n_plays", "n_real_plays", "start_yardline_100", "score_diff_start", "wp_start", "qtr_start",
    "start_seconds_remaining", *STYLE_FEATURES,
]
PLAY_COLS = [
    *DRIVE_KEY, "play_num", "play_id", "qtr", "down", "ydstogo", "yardline_100", "game_seconds_remaining",
    "score_differential", "wp", "play_type", "call", "yards_gained", "epa", "success", "first_down",
    "touchdown", "penalty",
]


def build_app_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    feats = pl.read_parquet(PROCESSED_DIR / "drive_features.parquet")
    styles = pl.read_parquet(PROCESSED_DIR / "drive_styles.parquet")
    opening, _ = build_opening_styles()
    plays = pl.read_parquet(PROCESSED_DIR / "drive_plays.parquet")

    seq = drive_token_sequences(plays).with_columns(
        calls=pl.col("tokens").list.eval(
            pl.element().filter(pl.element().is_in(["KNEEL", "SPIKE"]).not_())
        ).list.join(" ")
    ).select(*DRIVE_KEY, "calls")
    drives = (
        eligible_drives(feats).select(DRIVE_COLS)
        .join(styles.select(*DRIVE_KEY, "style", "margin"), on=DRIVE_KEY)
        .join(opening.select(*DRIVE_KEY, "open_style", pl.col("open_margin")), on=DRIVE_KEY)
        .join(seq, on=DRIVE_KEY, how="left")
    )
    keep_keys = drives.select(DRIVE_KEY)
    slim = add_tokens(plays).rename({"token": "call"}).join(keep_keys, on=DRIVE_KEY).select(PLAY_COLS)
    drives.write_parquet(APP_DRIVES)
    slim.write_parquet(APP_PLAYS)
    return drives.to_pandas(), slim.to_pandas()


if __name__ == "__main__":
    d, p = build_app_data()
    print(d.shape, p.shape)
    assert designed_calls(["R_in", "KNEEL"]) == ["R_in"]  # kneels/spikes are not play-calling
