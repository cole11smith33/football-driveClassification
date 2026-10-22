import pandas as pd
import streamlit as st

from drivedna.analysis.openings import OPENING_ORDER
from drivedna.app import cached, charts, ui
from drivedna.models.cluster import STYLE_ORDER

dna = cached.report("team_season_dna.csv")
dec = cached.report("performance_decomposition.csv")
st.title("🏟️ Team DNA")
st.write("A team's **tendency** for a style = its actual share of drives minus the share its game situations predict "
         "(so a team that trails a lot isn't mistaken for one that likes to hurry up).")

teams = sorted(dna["team"].unique())
c1, c2 = st.columns(2)
team = c1.selectbox("Team", teams, index=teams.index("KC") if "KC" in teams else 0)
seasons = sorted(dna.loc[dna["team"] == team, "season"].unique())
season = c2.selectbox("Season", seasons, index=len(seasons) - 1)
ui.team_theme(team)
row = dna[(dna["team"] == team) & (dna["season"] == season)].iloc[0]
cols = st.columns(4)
changes = [x for x, f in [("coach", row["hc_changed"]), ("QB", row["qb_changed"])] if f == 1]
ui.stat(cols[0], "Head coach", row["coach"])
ui.stat(cols[1], "Primary QB", row["qb"])
ui.stat(cols[2], "Drives", int(row["n_drives"]))
ui.stat(cols[3], "New this year", " + ".join(changes) if changes else "no change")

tab_dna, tab_time, tab_exec, tab_compare = st.tabs(["Style DNA", "Over time", "Execution vs. style", "Compare & similar teams"])

with tab_dna:
    kind = st.radio("Style defined by", ["Whole drive", "First 3 plays (opening)"], horizontal=True, key="kind1")
    k = "drive" if kind == "Whole drive" else "open"
    st.plotly_chart(charts.team_dna_bars(dna, team, season, k), width="stretch")
    st.caption("Percentage points above/below what the team's situations predict, with ±1 standard error. A season is only ~140 drives, so small bars are noise.")
    rel = cached.report("dna_reliability.csv")
    with st.expander("How real are these tendencies?"):
        st.dataframe(rel.round(2), hide_index=True, width="stretch")
        st.caption("Reliability = how well two random halves of a team-season's games agree. Year-to-year = the same team's tendency in consecutive seasons. "
                   "No-huddle is the most team-specific trait; the deep-shot opening tendency is mostly noise.")

with tab_time:
    kind = st.radio("Style defined by", ["Whole drive", "First 3 plays (opening)"], horizontal=True, key="kind2")
    k = "drive" if kind == "Whole drive" else "open"
    order = STYLE_ORDER if k == "drive" else OPENING_ORDER
    chosen = st.multiselect("Styles", order, default=[order[-1]])
    if chosen:
        st.plotly_chart(charts.team_trend(dna, team, chosen, k), width="stretch")
        st.caption("Dotted lines mark a new head coach. Coordinators are not in the data, so a new play-caller under the same head coach is invisible.")
    st.dataframe(dna[dna["team"] == team][["season", "coach", "qb", "n_drives"]].sort_values("season"), hide_index=True, width="stretch")

with tab_exec:
    st.plotly_chart(charts.decomposition_bars(dec, team), width="stretch")
    r = dec[(dec["team"] == team) & (dec["season"] == season)].iloc[0]
    st.write(f"**{team} {season}:** {r['points_per_drive']:.2f} points per drive ({r['total_vs_league']:+.2f} vs. league) = "
             f"situation {r['situation_part']:+.2f} + opening style {r['style_part']:+.2f} + execution {r['exec_part']:+.2f} (±{r['exec_se']:.2f}).")
    st.caption("Execution is a residual: QB, talent, scheme quality, injuries and luck. Across all team-seasons it explains ~93% of the differences between offenses; "
               "style choices essentially none, because teams barely differ in how often they use a given opener.")

with tab_compare:
    st.write("Most similar other team-seasons (cosine similarity of script-adjusted tendencies):")
    nb = cached.neighbors(f"{team} {season}", 5)
    st.dataframe(pd.DataFrame({"team-season": nb.index, "similarity": nb.round(2).to_numpy()}), hide_index=True, width="stretch")
    st.divider()
    c1, c2 = st.columns(2)
    other_team = c1.selectbox("Compare with", teams, index=teams.index("WAS") if "WAS" in teams else 0)
    other_seasons = sorted(dna.loc[dna["team"] == other_team, "season"].unique())
    other_season = c2.selectbox("Season ", other_seasons, index=len(other_seasons) - 1)
    k = "drive" if st.radio("Style defined by ", ["Whole drive", "First 3 plays (opening)"], horizontal=True, key="kind3") == "Whole drive" else "open"
    left, right = st.columns(2)
    left.subheader(f"{team} {season}")
    left.plotly_chart(charts.team_dna_bars(dna, team, season, k), width="stretch", key="compare_left")
    right.subheader(f"{other_team} {other_season}")
    right.plotly_chart(charts.team_dna_bars(dna, other_team, other_season, k), width="stretch", key="compare_right")
