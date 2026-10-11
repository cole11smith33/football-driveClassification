"""Run the Phase 6 analyses (defenses, style x defense interactions, situations); write CSVs to reports/."""
import warnings

import pandas as pd

from drivedna.analysis.defenses import (
    SCHEME,
    SCHEME_2018,
    add_scheme_residuals,
    archetype_study,
    defense_tendencies,
    loo_tendencies,
    prepare,
    split_half_reliability,
    two_way_archetypes,
    year_to_year,
)
from drivedna.analysis.interactions import (
    MODERATORS,
    bootstrap_effect_differences,
    cv_summary,
    interaction_frame,
    season_cv,
)
from drivedna.analysis.openings import OPENING_ORDER
from drivedna.analysis.play_table import build_play_table
from drivedna.analysis.situations import (
    add_looks,
    call_mix,
    depth_by_look,
    sack_rate_by_look,
    situation_table,
)
from drivedna.analysis.teams import drive_table
from drivedna.models.cluster import REPORTS_DIR


def run() -> None:
    warnings.filterwarnings("ignore")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plays = add_scheme_residuals(prepare(build_play_table()))

    # 1. defense scheme tendencies, reliability, archetypes
    tend = defense_tendencies(plays)
    tend["archetype"] = two_way_archetypes(tend).to_numpy()
    tend.round(5).to_csv(REPORTS_DIR / "defense_season_scheme.csv", index=False)
    rel = split_half_reliability(plays).merge(year_to_year(tend), on="tendency", how="left")
    rel.round(4).to_csv(REPORTS_DIR / "defense_reliability.csv", index=False)
    pd.concat([
        archetype_study(tend, SCHEME).assign(features="box + blitz + big blitz (2016-2025)"),
        archetype_study(tend[tend["season"] >= 2018].reset_index(drop=True), SCHEME_2018).assign(
            features="+ man-coverage share (2018-2025)"),
    ]).round(4).to_csv(REPORTS_DIR / "defense_archetype_stability.csv", index=False)

    # 2. does the value of an opening style depend on the defense faced?
    df = interaction_frame(drive_table(), loo_tendencies(plays))
    preds = season_cv(df, OPENING_ORDER, MODERATORS)
    cv_summary(df, preds).round(5).to_csv(REPORTS_DIR / "interaction_cv.csv", index=False)
    bootstrap_effect_differences(df, OPENING_ORDER, MODERATORS).round(4).to_csv(
        REPORTS_DIR / "interaction_effects.csv", index=False)

    # 3. situations and pass depth vs. the defensive look
    looks = add_looks(plays)
    for target in ["epa", "success"]:
        situation_table(looks, target).round(4).to_csv(REPORTS_DIR / f"situation_effects_{target}.csv", index=False)
    call_mix(looks).round(4).to_csv(REPORTS_DIR / "situation_call_mix.csv")
    blitz = {0.0: "4 or fewer rushers", 1.0: "blitz (5+ rushers)"}
    man = {0.0: "zone coverage", 1.0: "man coverage"}
    pd.concat([
        depth_by_look(looks, "look_blitz", blitz).assign(look_type="blitz"),
        depth_by_look(looks, "look_man", man).assign(look_type="man/zone (2018+)"),
    ]).round(4).to_csv(REPORTS_DIR / "depth_by_look.csv", index=False)
    pd.concat([
        sack_rate_by_look(looks, "look_blitz", blitz).assign(look_type="blitz"),
        sack_rate_by_look(looks, "look_man", man).assign(look_type="man/zone (2018+)"),
    ]).round(4).to_csv(REPORTS_DIR / "sack_by_look.csv", index=False)


if __name__ == "__main__":
    from drivedna.analysis.run_defenses import run as _run

    _run()
    print("done")
