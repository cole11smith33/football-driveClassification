"""Drive outcomes, kept apart from style features so they can never leak into clustering."""
import polars as pl

# Points for the offense. Defensive scores count against it.
POINTS = {
    "Touchdown": 7, "Field goal": 3, "Missed field goal": 0, "Punt": 0, "Turnover": 0,
    "Turnover on downs": 0, "End of half": 0, "Safety": -2, "Opp touchdown": -7,
}
OUTCOME_COLS = ["result", "points", "scored", "yards", "epa"]  # `yards`, `epa` come from drives


def add_outcomes(drives: pl.DataFrame) -> pl.DataFrame:
    return drives.with_columns(
        points=pl.col("result").replace_strict(POINTS, default=None, return_dtype=pl.Int32),
        scored=pl.col("result").is_in(["Touchdown", "Field goal"]),
    )
