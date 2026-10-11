import streamlit as st

from drivedna.analysis.openings import OPENING_ORDER
from drivedna.app import cached, charts
from drivedna.models.cluster import STYLE_ORDER

d = cached.drives()
st.title("🧬 Drive styles")
st.write("Drives are clustered on **11 play-calling features** (pass rate, air yards, tempo, shotgun, run mix...), never on how the drive ended.")

tab_profile, tab_mix, tab_examples, tab_work = st.tabs(["Profiles", "Mix over time", "Example drives", "Do styles work?"])

with tab_profile:
    st.plotly_chart(charts.style_heatmap(d), width="stretch")
    st.caption("Each cell: how many standard deviations a style's average is from the league average, per feature. Red = more, blue = less.")
    prof = cached.report("style_profiles.csv")
    st.dataframe(prof[["style", "share", "n_real_plays", "pass_rate", "mean_air_yards", "deep_pass_share", "shotgun_rate", "no_huddle_rate", "sec_per_play"]]
                 .rename(columns={"n_real_plays": "plays/drive"}).round(2), hide_index=True, width="stretch")
    st.caption("*Silhouette is low for every number of clusters: styles are a continuum, not natural clumps.*")

with tab_mix:
    col = st.radio("Style defined by", ["Whole drive", "First 3 plays (opening)"], horizontal=True)
    st.plotly_chart(charts.style_mix_by_season(d, "style" if col == "Whole drive" else "open_style"), width="stretch")
    st.caption("The mix barely moves between seasons. Opening styles are the cleaner measure of what a team chose to call; whole-drive styles also reflect how the drive went.")

with tab_examples:
    c1, c2, c3 = st.columns(3)
    style = c1.selectbox("Style", STYLE_ORDER)
    results = ["Any"] + sorted(d["result"].unique())
    result = c2.selectbox("Result", results)
    seed = c3.number_input("Shuffle", min_value=0, value=0, step=1)
    pool = d[(d["style"] == style) & (d["margin"] > 0.2)]
    if result != "Any":
        pool = pool[pool["result"] == result]
    st.caption(f"{len(pool):,} drives (confident assignments only). `R_in`/`R_out` runs, `P_short`/`P_mid`/`P_deep` passes, `SCR` scramble, `SACK`.")
    sample = pool.sample(min(10, len(pool)), random_state=int(seed)) if len(pool) else pool
    st.dataframe(sample.assign(game=lambda x: x["season"].astype(str) + " " + x["posteam"] + " vs " + x["defteam"])[["game", "result", "n_real_plays", "calls"]]
                 .rename(columns={"n_real_plays": "plays"}), hide_index=True, width="stretch")

with tab_work:
    st.warning("**Observational.** These compare a drive's *opening* style (first 3 plays) with a typical opening in the same situation. "
               "They hold measured context fixed, not what a coach sees, so read them as associations.")
    eff = cached.report("effectiveness.csv")
    opening_specs = {"Main": "opening_style", "Same team, same season (team-season fixed effects)": "opening_team_season_fe",
                     "Trailing": "opening_trailing", "Leading or tied": "opening_leading_or_tied",
                     "Backed up inside own 20": "opening_backed_up", "Not backed up": "opening_not_backed_up",
                     "Confident assignments only": "opening_confident_half"}
    c1, c2 = st.columns(2)
    label = c1.selectbox("Specification", list(opening_specs))
    metric = c2.selectbox("Outcome", ["points", "scored", "Touchdown", "Field goal", "Punt", "Turnover", "epa"], format_func=lambda m: {"points": "points per drive", "scored": "scoring probability (TD or FG)", "epa": "EPA per drive"}.get(m, f"{m} probability"))
    st.plotly_chart(charts.effects_forest(eff, opening_specs[label], metric, order=OPENING_ORDER), width="stretch")
    st.markdown("Bars are 95% intervals from resampling whole games. **Deep-shot** openers stay clearly positive in every specification; "
                "the deep-shot tendency, though, is mostly noise at the team level (see *Team DNA*).")
    with st.expander("Whole-drive style (for contrast)"):
        st.plotly_chart(charts.effects_forest(eff, "whole_drive_style", metric, order=STYLE_ORDER), width="stretch")
        st.caption("Whole-drive style is contaminated by how the drive went (a stalled drive of incompletions *looks* hurry-up), so these rankings should not be read as which style works.")
