"""Loaders for the app. Plain functions; the Streamlit pages wrap them in st.cache_data."""
from pathlib import Path

import pandas as pd

from drivedna.app.prepare import APP_DRIVES, APP_PLAYS
from drivedna.models.cluster import REPORTS_DIR

REQUIRED_REPORTS = [
    "style_profiles.csv", "effectiveness.csv", "team_season_dna.csv", "performance_decomposition.csv",
    "change_event_summary.csv", "dna_reliability.csv", "defense_season_scheme.csv",
    "situation_effects_epa.csv", "situation_effects_success.csv", "depth_by_look.csv", "sack_by_look.csv",
    "sequences_anywhere.csv", "sequences_opening.csv", "stability.json", "interaction_cv.csv",
]


def missing_files() -> list[Path]:
    needed = [APP_DRIVES, APP_PLAYS, *[REPORTS_DIR / r for r in REQUIRED_REPORTS]]
    return [p for p in needed if not p.exists()]


def drives() -> pd.DataFrame:
    return pd.read_parquet(APP_DRIVES)


def plays() -> pd.DataFrame:
    return pd.read_parquet(APP_PLAYS)


def report(name: str) -> pd.DataFrame:
    return pd.read_csv(REPORTS_DIR / name)


def headlines() -> dict:
    """The few numbers the home page quotes, read from the saved reports (never hard-coded)."""
    import json

    eff = report("effectiveness.csv")
    deep = eff.query("spec == 'opening_style' and estimand == 'vs_typical' and metric == 'points' and style == 'Deep-shot'").iloc[0]
    dec = report("performance_decomposition.csv")
    ev = report("change_event_summary.csv").set_index("event")
    stab = json.loads((REPORTS_DIR / "stability.json").read_text())
    return {
        "n_drives": len(drives()),
        "deep_shot_points": float(deep["value"]), "deep_lo": float(deep["lo"]), "deep_hi": float(deep["hi"]),
        "exec_variance_share": float(dec["exec_part"].var() / dec["total_vs_league"].var()),
        "style_variance_share": float(dec["style_part"].var() / dec["total_vs_league"].var()),
        "hc_change_shift": float(ev.loc["new coach only", "diff_vs_no_change"]),
        "qb_change_shift": float(ev.loc["new QB only", "diff_vs_no_change"]),
        "style_bootstrap_ari": float(stab["bootstrap"]["ari_mean"]),
        "style_temporal_ari": float(stab["temporal"]["ari_early_model_vs_late_model_on_late_drives"]),
    }
