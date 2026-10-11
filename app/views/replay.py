import pandas as pd
import streamlit as st

from drivedna.app import cached, charts, ui

d = cached.drives()
p = cached.plays()
st.title("🎬 Drive replay")
st.write("Step through any drive, play by play, with its style classification. (Drives with fewer than 3 designed plays, "
         "garbage-time drives and kneel-downs are excluded, as in the analyses.)")

c1, c2, c3 = st.columns(3)
season = c1.selectbox("Season", sorted(d["season"].unique(), reverse=True))
teams = sorted(d.loc[d["season"] == season, "posteam"].unique())
team = c2.selectbox("Offense", teams, index=teams.index("KC") if "KC" in teams else 0)
games = d[(d["season"] == season) & (d["posteam"] == team)][["game_id", "week", "defteam", "season_type"]].drop_duplicates().sort_values(["season_type", "week"], ascending=[False, True])
labels = {r.game_id: f"{'Playoffs' if r.season_type == 'POST' else 'Week ' + str(int(r.week))} vs {r.defteam}" for r in games.itertuples()}
game_id = c3.selectbox("Game", list(labels), format_func=labels.get)
gd = d[(d["game_id"] == game_id) & (d["posteam"] == team)].sort_values("fixed_drive")
drive_labels = {r.fixed_drive: f"Drive {int(r.fixed_drive)}: {r.result} ({int(r.n_real_plays)} plays, {r.points:+.0f} pts)" for r in gd.itertuples()}
fixed = st.selectbox("Drive", list(drive_labels), format_func=drive_labels.get)
drive = gd[gd["fixed_drive"] == fixed].iloc[0]

m = st.columns([1.1, 1.5, 1.5, 1.3, 0.9])
ui.stat(m[0], "Result", drive["result"])
ui.stat(m[1], "Whole-drive style", drive["style"], f"assignment margin {drive['margin']:.2f} (0 = on a style boundary)")
ui.stat(m[2], "Opening style", drive["open_style"], f"assignment margin {drive['open_margin']:.2f}")
ui.stat(m[3], "Started", f"{drive['start_yardline_100']:.0f} yds out",
        f"score diff {drive['score_diff_start']:+.0f}, quarter {int(drive['qtr_start'])}")
ui.stat(m[4], "EPA", f"{drive['epa']:+.2f}")

dp = p[(p["game_id"] == game_id) & (p["posteam"] == team) & (p["fixed_drive"] == fixed)].sort_values("play_num")
if dp.empty:
    st.warning("No plays found for this drive.")
else:
    st.plotly_chart(charts.drive_replay_chart(dp[dp["call"].isin(["R_in", "R_out", "P_short", "P_mid", "P_deep", "SACK", "SCR", "P_unk"])]), width="stretch")
    table = dp.assign(**{"down & distance": dp.apply(lambda r: f"{int(r['down'])} & {int(r['ydstogo'])}" if pd.notna(r["down"]) else "", axis=1)})
    st.dataframe(table[["play_num", "qtr", "down & distance", "yardline_100", "call", "yards_gained", "epa", "first_down", "touchdown"]]
                 .rename(columns={"play_num": "#", "yardline_100": "yds from end zone", "yards_gained": "yards"}).round(2),
                 hide_index=True, width="stretch")
    st.caption("Calls: `R_in`/`R_out` run inside/outside, `P_short`/`P_mid`/`P_deep` pass by air yards (≤5 / 6-14 / ≥15), `SCR` QB scramble, `SACK`, `KNEEL`, `SPIKE`.")
