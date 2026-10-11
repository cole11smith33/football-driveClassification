"""Turn plays into symbolic tokens so a drive reads as a sequence, e.g. `R_in P_short P_deep`.

Tokens describe the *call* (run type, pass depth), not how the play turned out, so sequence
mining doesn't just rediscover "plays that gained yards lead to scores". Sacks and scrambles
are the exception: they are not distinguishable as calls, so they get their own tokens.
"""
import polars as pl

from drivedna.drives.build import DRIVE_KEY

DEEP_AIR_YARDS = 15   # ~85th percentile of air yards
SHORT_AIR_YARDS = 5   # at or behind this is a short/quick pass

TOKENS = [
    "R_in", "R_out", "SCR", "P_short", "P_mid", "P_deep", "P_unk", "SACK", "KNEEL", "SPIKE",
]


def token_expr() -> pl.Expr:
    """Polars expression mapping a drive_plays row to one token."""
    air = pl.col("air_yards")
    outside_gap = pl.col("run_gap").is_in(["tackle", "end"])
    return (
        pl.when(pl.col("play_type") == "qb_kneel").then(pl.lit("KNEEL"))
        .when(pl.col("play_type") == "qb_spike").then(pl.lit("SPIKE"))
        .when(pl.col("sack") == 1).then(pl.lit("SACK"))
        .when(pl.col("qb_scramble") == 1).then(pl.lit("SCR"))
        .when(pl.col("play_type") == "run").then(
            pl.when(outside_gap).then(pl.lit("R_out")).otherwise(pl.lit("R_in"))
        )
        # pass with no recorded depth (throwaways etc.)
        .when(air.is_null()).then(pl.lit("P_unk"))
        .when(air >= DEEP_AIR_YARDS).then(pl.lit("P_deep"))
        .when(air <= SHORT_AIR_YARDS).then(pl.lit("P_short"))
        .otherwise(pl.lit("P_mid"))
    )


def add_tokens(drive_plays: pl.DataFrame) -> pl.DataFrame:
    return drive_plays.with_columns(token=token_expr())


def drive_token_sequences(drive_plays: pl.DataFrame) -> pl.DataFrame:
    """One row per drive: ordered token list and a space-joined string."""
    return (
        add_tokens(drive_plays)
        .sort(DRIVE_KEY + ["play_num"])
        .group_by(DRIVE_KEY, maintain_order=True)
        .agg(tokens=pl.col("token"))
        .with_columns(token_str=pl.col("tokens").list.join(" "))
    )
