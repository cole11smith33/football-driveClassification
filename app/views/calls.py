import streamlit as st

from drivedna.app import cached, charts

CALLS = ["R_in", "R_out", "P_short", "P_mid", "P_deep"]
st.title("🔁 Calls & sequences")
tab_sit, tab_look, tab_seq = st.tabs(["By situation", "Against the defensive look", "Play-call sequences"])

with tab_sit:
    target = st.radio("Measure", ["EPA per play", "Success rate"], horizontal=True)
    st_df = cached.report("situation_effects_epa.csv" if target == "EPA per play" else "situation_effects_success.csv")
    sits = st.multiselect("Situations", list(st_df["situation"].unique()), default=list(st_df["situation"].unique()))
    st.plotly_chart(charts.grouped_effects(st_df[st_df["situation"].isin(sits)], "call", "situation", CALLS,
                                           f"{target} vs. the typical call"), width="stretch")
    st.caption("For plays that used each call: its value minus the typical call mix's value in the same down, distance, field position, score and time. "
               "Intervals resample whole games.")
    mix = cached.report("situation_call_mix.csv").rename(columns={"Unnamed: 0": "situation"})
    with st.expander("How often each call is used"):
        st.dataframe(mix.round(3), hide_index=True, width="stretch")
    st.info("**Mid-depth passes are the reliable call; deep passes are boom-or-bust** (highest EPA, but little or negative success-rate edge, "
            "especially in the red zone and two-minute drill). **EPA does not value the clock**, so negative run numbers with a fourth-quarter lead are incomplete.")

with tab_look:
    depth = cached.report("depth_by_look.csv")
    kind = st.radio("Defensive look", list(depth["look_type"].unique()), horizontal=True)
    sub = depth[depth["look_type"] == kind]
    st.plotly_chart(charts.grouped_effects(sub, "call", "look", ["P_short", "P_mid", "P_deep"], "EPA per attempt vs. the typical depth"), width="stretch")
    st.caption("Thrown passes only (a sack has no depth). The deep-pass edge shrinks against a blitz and against man coverage; short and mid passes barely change.")
    sack = cached.report("sack_by_look.csv")
    st.write("Sacks per dropback, which thrown-pass EPA leaves out:")
    st.dataframe(sack[sack["look_type"] == kind][["look", "sack_rate", "dropbacks"]].round(3), hide_index=True)
    st.warning("Observational: defenses choose their look pre-snap in response to the offense's formation, which is not controlled for. "
               "Man/zone looks exist from 2018 only.")

with tab_seq:
    view = st.radio("View", ["Anywhere in the drive (compared within equal-length drives)", "Opening (first 4 calls, drives lasting 6+ plays)"])
    seq = cached.report("sequences_anywhere.csv" if view.startswith("Anywhere") else "sequences_opening.csv")
    c1, c2, c3, c4 = st.columns(4)
    min_drives = c1.slider("Minimum drives", 150, 2000, 150, step=50)
    sig_only = c2.checkbox("Significant only (5% FDR)", value=True)
    no_sack = c3.checkbox("Hide sacks (a result, not a call)", value=True)
    text = c4.text_input("Contains", placeholder="e.g. P_deep")
    f = seq[seq["drives"] >= min_drives]
    if sig_only:
        f = f[f["significant"]]
    if no_sack:
        f = f[~f["ngram"].str.contains("SACK")]
    if text:
        f = f[f["ngram"].str.contains(text, case=False)]
    show = f[["ngram", "drives", "share_Touchdown", "share_Punt", "or_naive", "or_stratified", "or_lo", "or_hi", "q"]].rename(columns={
        "or_stratified": "odds ratio (length-matched)", "or_naive": "odds ratio (naive)", "or_lo": "ci low", "or_hi": "ci high",
        "share_Touchdown": "% of TD drives", "share_Punt": "% of punt drives"})
    st.write(f"{len(show)} patterns. Odds ratio > 1: more common in touchdown drives than in punt drives of the same length.")
    st.dataframe(show.round(3), hide_index=True, width="stretch", height=420)
    st.caption("Touchdown drives are longer than punt drives, so a naive comparison finds 'touchdown patterns' that are really just length "
               "(the naive column). The length-matched odds ratio fixes that. Patterns are associated with outcomes, not proven causes.")
