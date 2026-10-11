import streamlit as st

from drivedna.app import cached, charts, ui

tend = cached.report("defense_season_scheme.csv")
tend["label"] = tend["defteam"] + " " + tend["season"].astype(str)
st.title("🛡️ Defenses")
st.write("How much a defense **blitzes** and **stacks the box**, *relative to what the situation and the offense's formation call for* "
         "(a defense facing three receivers shows a light box whatever its philosophy). Results (EPA allowed, pressure) are shown separately and never define a scheme.")

selected = st.selectbox("Highlight a defense-season", ["(none)", *sorted(tend["label"])], index=0)
st.plotly_chart(charts.defense_scatter(tend, None if selected == "(none)" else selected), width="stretch")

if selected != "(none)":
    r = tend[tend["label"] == selected].iloc[0]
    c = st.columns(5)
    ui.stat(c[0], "Archetype", r["archetype"])
    ui.stat(c[1], "Blitz vs. expected", f"{r['blitz_excess'] * 100:+.1f} pp", "of dropbacks")
    ui.stat(c[2], "Box vs. expected", f"{r['box_excess']:+.2f}", "defenders in the box")
    ui.stat(c[3], "EPA/dropback allowed", f"{r['epa_pass_allowed']:+.3f}")
    ui.stat(c[4], "Pressure rate", f"{r['pressure_rate']:.1%}")

st.subheader("Extremes")
c1, c2 = st.columns(2)
c1.write("**Most blitz-heavy**")
c1.dataframe(tend.sort_values("blitz_excess", ascending=False).head(8).assign(blitz_pp=lambda x: (x["blitz_excess"] * 100).round(1))[["label", "blitz_pp"]], hide_index=True)
c2.write("**Least blitz-heavy**")
c2.dataframe(tend.sort_values("blitz_excess").head(8).assign(blitz_pp=lambda x: (x["blitz_excess"] * 100).round(1))[["label", "blitz_pp"]], hide_index=True)

with st.expander("Are these tendencies real? Do defenses form archetypes?"):
    st.dataframe(cached.report("defense_reliability.csv").round(2), hide_index=True, width="stretch")
    st.markdown("Scheme tendencies are highly reliable and persist year to year. **Only a two-way split holds up** (Aggressive 37%, Light 63%): "
                "from three archetypes up, the partitions fall apart when the data are resampled or split by era. So there is no richer taxonomy of defenses. "
                "The two groups differ only slightly on results.")
    prof = tend.groupby("archetype")[["blitz_excess", "box_excess", "epa_pass_allowed", "pressure_rate"]].mean().round(3)
    prof["share"] = tend["archetype"].value_counts(normalize=True).round(2)
    st.dataframe(prof, width="stretch")
    cv = cached.report("interaction_cv.csv")
    st.markdown("**Does the value of an opening style depend on the defense?** Tested on held-out seasons, adding opening style improves prediction of points per drive, "
                "but adding style × opponent blitz/box tendency does not. *No detectable dependence is not no dependence: the test has limited power.*")
    st.dataframe(cv.round(4), hide_index=True, width="stretch")
