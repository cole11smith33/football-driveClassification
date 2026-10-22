"""Filter flags for drives. Flags are kept as columns, never used to delete rows."""
import polars as pl

GARBAGE_WP = (0.05, 0.95)  # drive-start WP outside this band => garbage time
MIN_SCRIMMAGE_PLAYS = 2


def add_drive_flags(drives: pl.DataFrame) -> pl.DataFrame:
    """Add boolean flags and a combined `keep` column to the drives table."""
    return drives.with_columns(
        is_kneel_only=pl.col("n_kneel") == pl.col("n_plays"),
        is_end_of_half=pl.col("result") == "End of half",
        is_garbage_time=(pl.col("wp_start") < GARBAGE_WP[0]) | (pl.col("wp_start") > GARBAGE_WP[1]),
        is_too_short=pl.col("n_plays") < MIN_SCRIMMAGE_PLAYS,
        has_unknown_result=pl.col("result").is_null(),
    ).with_columns(
        keep=~(
            pl.col("is_kneel_only")
            | pl.col("is_end_of_half")
            | pl.col("is_garbage_time")
            | pl.col("is_too_short")
            | pl.col("has_unknown_result")
        )
    )
