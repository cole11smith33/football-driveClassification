"""Run the team-profile analyses and write CSVs to reports/."""
import warnings

import pandas as pd

from drivedna.analysis.teams import (
    change_events,
    component_persistence,
    decompose_performance,
    drive_table,
    event_summary,
    excess_columns,
    nearest_neighbors,
    similarity_matrix,
    split_half_reliability,
    team_dna,
    team_season_meta,
    year_to_year,
)
from drivedna.models.cluster import REPORTS_DIR

LATEST_N = 1  # nearest neighbors reported for each team's most recent season


def run() -> None:
    warnings.filterwarnings("ignore")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = drive_table()
    meta = team_season_meta()
    dna = team_dna(df)
    cols = excess_columns(dna)

    dna.merge(meta[["team", "season", "coach", "qb", "hc_changed", "qb_changed"]], on=["team", "season"], how="left") \
        .round(5).to_csv(REPORTS_DIR / "team_season_dna.csv", index=False)

    rel = split_half_reliability(df).merge(year_to_year(dna), on=["kind", "style"])
    rel.round(4).to_csv(REPORTS_DIR / "dna_reliability.csv", index=False)

    sim = similarity_matrix(dna, cols)
    latest = dna.sort_values("season").groupby("team").tail(LATEST_N)
    rows = []
    for _, r in latest.iterrows():
        label = f"{r['team']} {int(r['season'])}"
        for rank, (other, s) in enumerate(nearest_neighbors(sim, label, 5).items(), 1):
            rows.append({"team_season": label, "rank": rank, "similar_to": other, "cosine": round(s, 4)})
    pd.DataFrame(rows).to_csv(REPORTS_DIR / "team_neighbors_latest.csv", index=False)

    events = change_events(dna, meta, cols)
    events.round(4).to_csv(REPORTS_DIR / "change_events.csv", index=False)
    event_summary(events).round(4).to_csv(REPORTS_DIR / "change_event_summary.csv", index=False)

    dec = decompose_performance(df)
    dec.round(4).to_csv(REPORTS_DIR / "performance_decomposition.csv", index=False)
    component_persistence(dec).round(4).to_csv(REPORTS_DIR / "component_persistence.csv", index=False)


if __name__ == "__main__":
    from drivedna.analysis.run_teams import run as _run

    _run()
    print("done")
