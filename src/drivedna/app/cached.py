"""Streamlit-cached wrappers around the plain loaders (used by the pages)."""
import pandas as pd
import streamlit as st

from drivedna.analysis.teams import excess_columns, nearest_neighbors, similarity_matrix
from drivedna.app import data


@st.cache_data(show_spinner=False)
def drives() -> pd.DataFrame:
    return data.drives()


@st.cache_data(show_spinner=False)
def plays() -> pd.DataFrame:
    return data.plays()


@st.cache_data(show_spinner=False)
def report(name: str) -> pd.DataFrame:
    return data.report(name)


@st.cache_data(show_spinner=False)
def headlines() -> dict:
    return data.headlines()


@st.cache_data(show_spinner=False)
def team_similarity() -> pd.DataFrame:
    dna = data.report("team_season_dna.csv")
    return similarity_matrix(dna, excess_columns(dna))


def neighbors(label: str, k: int = 5) -> pd.Series:
    return nearest_neighbors(team_similarity(), label, k)
