"""Opening style: classify each drive from only its first N designed plays.

Whole-drive style is contaminated by how the drive went (a stalled drive looks 'hurry-up', a
scoring drive's last play is a goal-line run). Style from the opening plays precedes the outcome,
so it is the cleaner treatment for 'does the way a drive starts matter?'.
Every eligible drive has >= 3 designed plays, so N = 3 is defined for all of them.
"""
import pandas as pd
import polars as pl

from drivedna.drives.build import DRIVE_KEY, PROCESSED_DIR
from drivedna.features.drive_features import STYLE_FEATURES, build_style_and_rhythm
from drivedna.models.cluster import StyleModel, eligible_drives

OPENING_PLAYS = 3

# Opening clusters (3 plays) are shaped differently from whole-drive styles, so they get their own names.
OPENING_ORDER = [
    "Run-first", "Balanced (inside runs)", "Balanced (outside runs)", "Deep-shot", "Pass-only, up-tempo",
]


def name_opening_clusters(profile: pd.DataFrame) -> dict[int, str]:
    """Name opening clusters by their own extremes, in the order the rules fire."""
    if len(profile) != len(OPENING_ORDER):
        return {int(c): f"Opening {i}" for i, c in enumerate(profile.sort_values("pass_rate").index)}
    remaining, names = list(profile.index), {}
    for name, col, how in [
        ("Deep-shot", "deep_pass_share", "max"),
        ("Pass-only, up-tempo", "pass_rate", "max"),
        ("Run-first", "pass_rate", "min"),
    ]:
        vals = profile.loc[remaining, col]
        pick = vals.idxmax() if how == "max" else vals.idxmin()
        names[int(pick)] = name
        remaining.remove(pick)
    # the last two differ mainly in which runs they call
    inside = profile.loc[remaining, "inside_run_share"]
    names[int(inside.idxmax())] = "Balanced (inside runs)"
    names[int(inside.idxmin())] = "Balanced (outside runs)"
    return names


def opening_plays(drive_plays: pl.DataFrame, n: int = OPENING_PLAYS) -> pl.DataFrame:
    """First n designed (run/pass) plays of every drive."""
    real = drive_plays.filter(pl.col("play_type").is_in(["pass", "run"])).sort(DRIVE_KEY + ["play_num"])
    return real.filter(pl.int_range(pl.len()).over(DRIVE_KEY) < n)


def opening_features(drive_plays: pl.DataFrame, n: int = OPENING_PLAYS) -> pl.DataFrame:
    """STYLE_FEATURES computed on the opening plays, plus identifiers for joining."""
    feats = build_style_and_rhythm(opening_plays(drive_plays, n))
    return feats.select(DRIVE_KEY + STYLE_FEATURES).with_columns(
        n_real_plays=pl.lit(n).cast(pl.UInt32)
    )


def build_opening_styles(k: int = 5, n: int = OPENING_PLAYS) -> tuple[pl.DataFrame, StyleModel]:
    """Fit a StyleModel on opening-play features. Returns the per-drive table and the model."""
    plays = pl.read_parquet(PROCESSED_DIR / "drive_plays.parquet")
    feats = pl.read_parquet(PROCESSED_DIR / "drive_features.parquet")
    elig = eligible_drives(feats).select(DRIVE_KEY + ["season"])
    df = elig.join(opening_features(plays, n), on=DRIVE_KEY)
    model = StyleModel(k=k, namer=name_opening_clusters, canonical_order=OPENING_ORDER).fit(df)
    ids = model.predict(df)
    table = df.select(DRIVE_KEY).with_columns(
        open_style_id=pl.Series(ids, dtype=pl.Int32),
        open_style=pl.Series([model.style_names_[i] for i in ids]),
        open_margin=pl.Series(model.margin(df)),
    )
    return table, model
