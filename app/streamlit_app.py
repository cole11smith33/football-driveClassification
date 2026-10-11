"""Drive DNA: explore NFL drive styles, team profiles, defenses and call effectiveness.

Run:  streamlit run app/streamlit_app.py
Needs the pipeline outputs (see README). If anything is missing the app says what to run.
"""
from pathlib import Path

import streamlit as st

from drivedna.app import data

st.set_page_config(page_title="Drive DNA", page_icon="🏈", layout="wide")

missing = data.missing_files()
if missing:
    st.error("The app needs the pipeline outputs, and some are missing.")
    st.write("Missing files:")
    st.code("\n".join(str(p) for p in missing))
    st.write("Run these from the project root (full details in the README):")
    st.code(
        ".venv/bin/python -m drivedna.data.ingest\n"
        ".venv/bin/python -m drivedna.drives.build\n"
        ".venv/bin/python -m drivedna.features.drive_features\n"
        ".venv/bin/python -m drivedna.models.cluster\n"
        ".venv/bin/python -m drivedna.analysis.run_effectiveness\n"
        ".venv/bin/python -m drivedna.analysis.run_secondary\n"
        ".venv/bin/python -m drivedna.analysis.run_teams\n"
        ".venv/bin/python -m drivedna.analysis.run_defenses\n"
        ".venv/bin/python -m drivedna.app.prepare",
        language="bash",
    )
    st.stop()

here = Path(__file__).parent / "views"
navigation = st.navigation([
    st.Page(here / "home.py", title="Home", icon="🏈", default=True),
    st.Page(here / "styles.py", title="Drive styles", icon="🧬"),
    st.Page(here / "teams.py", title="Team DNA", icon="🏟️"),
    st.Page(here / "defenses.py", title="Defenses", icon="🛡️"),
    st.Page(here / "calls.py", title="Calls & sequences", icon="🔁"),
    st.Page(here / "replay.py", title="Drive replay", icon="🎬"),
])
navigation.run()
