"""Headless tests of the Streamlit app: every page runs, widgets work, and a missing-data state is handled."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from drivedna.app import charts, data
from drivedna.app.prepare import APP_DRIVES, APP_PLAYS
from drivedna.drives.build import DRIVE_KEY
from drivedna.models.cluster import STYLE_ORDER

pytestmark = pytest.mark.skipif(
    bool(data.missing_files()), reason="run the pipeline and `python -m drivedna.app.prepare` first"
)

ROOT = Path(__file__).resolve().parents[1]
VIEWS = ["home", "styles", "teams", "defenses", "calls", "replay"]


def run(view: str) -> AppTest:
    return AppTest.from_file(str(ROOT / "app" / "views" / f"{view}.py"), default_timeout=120).run()


def no_errors(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error


# ---------------------------------------------------------------- data
def test_app_tables_are_consistent():
    d, p = data.drives(), data.plays()
    assert not d.duplicated(DRIVE_KEY).any()
    assert d["style"].notna().all() and d["open_style"].notna().all() and d["calls"].notna().all()
    assert set(d["style"]) == set(STYLE_ORDER)
    joined = d[DRIVE_KEY].merge(p[DRIVE_KEY].drop_duplicates(), on=DRIVE_KEY, how="left", indicator=True)
    assert (joined["_merge"] == "both").all()  # every drive has plays
    assert APP_DRIVES.exists() and APP_PLAYS.exists()


def test_headlines_come_from_the_reports():
    h = data.headlines()
    assert h["n_drives"] == len(data.drives())
    assert 0 < h["deep_lo"] < h["deep_shot_points"] < h["deep_hi"]
    assert 0.8 < h["exec_variance_share"] < 1.1 and h["style_variance_share"] < 0.05


# --------------------------------------------------------------- charts
def test_charts_build_nonempty_figures():
    d = data.drives()
    for fig in [charts.style_heatmap(d), charts.style_mix_by_season(d), charts.style_mix_by_season(d, "open_style"),
                charts.effects_forest(data.report("effectiveness.csv"), "opening_style", "points"),
                charts.team_dna_bars(data.report("team_season_dna.csv"), "KC", 2023, "open"),
                charts.team_trend(data.report("team_season_dna.csv"), "ARI", ["No-huddle"]),
                charts.decomposition_bars(data.report("performance_decomposition.csv"), "KC"),
                charts.defense_scatter(data.report("defense_season_scheme.csv"), "MIN 2023")]:
        assert len(fig.data) >= 1


def test_replay_chart_handles_a_drive_with_a_sack_and_missing_down():
    plays = pd.DataFrame({
        "play_num": [1, 2, 3], "down": [1.0, 2.0, np.nan], "ydstogo": [10.0, 12.0, 0.0], "yardline_100": [75.0, 77.0, 77.0],
        "call": ["R_in", "SACK", "P_deep"], "yards_gained": [2.0, -5.0, 20.0], "epa": [0.0, -1.0, 2.0],
    })
    fig = charts.drive_replay_chart(plays)
    assert len(fig.data) >= 3


# ---------------------------------------------------------------- pages
@pytest.mark.parametrize("view", VIEWS)
def test_every_page_runs_without_errors(view):
    at = run(view)
    no_errors(at)
    assert at.title


def test_home_quotes_numbers_from_the_reports():
    at = run("home")
    h = data.headlines()
    shown = " ".join(m.value for m in at.markdown)
    assert f"{h['n_drives']:,}" in shown and f"{h['deep_shot_points']:+.2f}" in shown


def test_teams_page_responds_to_team_and_season_selection():
    at = run("teams")
    at.selectbox[0].select("WAS").run()
    at.selectbox[1].select(2024).run()
    no_errors(at)
    shown = " ".join(m.value for m in at.markdown)
    assert "Dan Quinn" in shown and "J.Daniels" in shown          # WAS 2024 coach and primary QB
    assert "new this year" in shown.lower() and "coach + QB" in shown


def test_defenses_page_highlight_shows_that_defense():
    at = run("defenses")
    at.selectbox[0].select("MIN 2023").run()
    no_errors(at)
    shown = " ".join(m.value for m in at.markdown)
    assert "Archetype" in shown and "Aggressive" in shown


def test_calls_page_sequence_filters_apply():
    at = run("calls")
    no_errors(at)
    text = next(t for t in at.text_input if t.label == "Contains")
    text.input("P_deep").run()
    no_errors(at)
    frames = [df.value for df in at.dataframe]
    seq = next(f for f in frames if "ngram" in f.columns)
    assert len(seq) > 0 and seq["ngram"].str.contains("P_deep").all()


def test_replay_page_selects_a_drive_and_shows_its_plays():
    at = run("replay")
    no_errors(at)
    shown = " ".join(m.value for m in at.markdown)
    assert "Whole-drive style" in shown and "Opening style" in shown
    table = next(df.value for df in at.dataframe)
    assert "call" in table.columns and len(table) >= 3


def test_styles_page_examples_respect_the_selected_style():
    at = run("styles")
    next(s for s in at.selectbox if s.label == "Style").select("Vertical").run()
    no_errors(at)
    ex = next(df.value for df in at.dataframe if "calls" in df.value.columns)
    assert 0 < len(ex) <= 10


# ------------------------------------------------------------- entry point
def test_main_app_runs_and_navigates():
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=120).run()
    no_errors(at)
    assert at.title and "Drive DNA" in at.title[0].value


def test_main_app_explains_what_to_run_when_outputs_are_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(data, "missing_files", lambda: [tmp_path / "app_drives.parquet"])
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=60).run()
    assert [e.value for e in at.error] and "missing" in at.error[0].value.lower()
    assert any("drivedna.app.prepare" in c.value for c in at.code)
    assert not at.exception
