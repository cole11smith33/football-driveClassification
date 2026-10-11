"""Run every effectiveness specification and write reports/effectiveness.csv (+ overlap).

Specs: whole-drive style (descriptive; contaminated by how the drive went), opening style (main),
and sensitivity checks on the opening-style result.
"""
import warnings

import pandas as pd

from drivedna.analysis.effectiveness import bootstrap_estimates, naive_outcomes, overlap_diagnostic
from drivedna.analysis.modeling_table import CATEGORICAL_CONTROLS, build_model_table
from drivedna.analysis.openings import OPENING_ORDER, build_opening_styles
from drivedna.drives.build import DRIVE_KEY
from drivedna.models.cluster import REPORTS_DIR, STYLE_ORDER

TEAM_SEASON_FE = [*CATEGORICAL_CONTROLS, "off_team_season"]


def specs(df: pd.DataFrame) -> list[dict]:
    opening = df.assign(style=df["open_style"])
    whole = df
    confident = opening[opening["open_margin"] >= opening["open_margin"].median()]
    return [
        {"name": "whole_drive_style", "data": whole, "styles": STYLE_ORDER, "n_boot": 100},
        {"name": "opening_style", "data": opening, "styles": OPENING_ORDER, "n_boot": 100},
        {"name": "opening_team_season_fe", "data": opening, "styles": OPENING_ORDER, "n_boot": 60,
             "cat_controls": TEAM_SEASON_FE},
        {"name": "opening_confident_half", "data": confident, "styles": OPENING_ORDER, "n_boot": 60},
        {"name": "opening_trailing", "data": opening[opening["score_diff"] < 0], "styles": OPENING_ORDER, "n_boot": 60},
        {"name": "opening_leading_or_tied", "data": opening[opening["score_diff"] >= 0], "styles": OPENING_ORDER,
             "n_boot": 60},
        {"name": "opening_backed_up", "data": opening[opening["start_yardline_100"] >= 80], "styles": OPENING_ORDER,
             "n_boot": 60},
        {"name": "opening_not_backed_up", "data": opening[opening["start_yardline_100"] < 80], "styles": OPENING_ORDER,
             "n_boot": 60},
    ]


def run() -> pd.DataFrame:
    warnings.filterwarnings("ignore")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    opening_table, _ = build_opening_styles()
    df = build_model_table().merge(opening_table.to_pandas(), on=DRIVE_KEY)
    results, naive = [], []
    for sp in specs(df):
        print("running", sp["name"], len(sp["data"]), flush=True)
        r = bootstrap_estimates(
            sp["data"], sp["styles"], n_boot=sp["n_boot"], cat_controls=sp.get("cat_controls")
        )
        r["spec"], r["n_drives"] = sp["name"], len(sp["data"])
        results.append(r)
        n = naive_outcomes(sp["data"]).reset_index()
        n["spec"] = sp["name"]
        naive.append(n)
    out = pd.concat(results, ignore_index=True)
    out.round(5).to_csv(REPORTS_DIR / "effectiveness.csv", index=False)
    pd.concat(naive, ignore_index=True).round(5).to_csv(REPORTS_DIR / "effectiveness_naive.csv", index=False)
    opening = df.assign(style=df["open_style"])
    overlap_diagnostic(opening, OPENING_ORDER).round(4).to_csv(REPORTS_DIR / "overlap_opening.csv")
    overlap_diagnostic(df, STYLE_ORDER).round(4).to_csv(REPORTS_DIR / "overlap_whole.csv")
    return out


if __name__ == "__main__":
    from drivedna.analysis.run_effectiveness import (
        run as _run,  # keep module path stable for pickles
    )

    _run()
    print("done")
