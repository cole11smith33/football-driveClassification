"""Plotly figure builders for the app. Pure functions: DataFrames in, figures out."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from drivedna.analysis.openings import OPENING_ORDER
from drivedna.features.drive_features import STYLE_FEATURES
from drivedna.models.cluster import STYLE_ORDER

ORDERS = {"drive": STYLE_ORDER, "open": OPENING_ORDER}
CALL_ORDER = ["R_in", "R_out", "P_short", "P_mid", "P_deep"]
CALL_COLORS = {"R_in": "#4c78a8", "R_out": "#9ecae9", "P_short": "#f58518", "P_mid": "#e45756", "P_deep": "#b279a2",
               "SACK": "#555555", "SCR": "#54a24b", "P_unk": "#999999"}


def style_heatmap(drives: pd.DataFrame) -> go.Figure:
    """Style centroids as standard deviations from the league mean (drive level)."""
    prof = drives.groupby("style")[STYLE_FEATURES].mean().reindex(STYLE_ORDER)
    z = (prof - drives[STYLE_FEATURES].mean()) / drives[STYLE_FEATURES].std()
    fig = px.imshow(z.T, color_continuous_scale="RdBu_r", zmin=-2, zmax=2, aspect="auto", text_auto=".1f")
    fig.update_layout(height=480, margin={"l": 10, "r": 10, "t": 30, "b": 10}, coloraxis_colorbar_title="SD")
    return fig


def style_mix_by_season(drives: pd.DataFrame, col: str = "style") -> go.Figure:
    order = STYLE_ORDER if col == "style" else OPENING_ORDER
    mix = drives.groupby(["season", col]).size().rename("n").reset_index()
    mix["share"] = mix["n"] / mix.groupby("season")["n"].transform("sum")
    fig = px.bar(mix, x="season", y="share", color=col, category_orders={col: order})
    fig.update_layout(height=380, margin={"l": 10, "r": 10, "t": 30, "b": 10}, yaxis_tickformat=".0%")
    return fig


def effects_forest(eff: pd.DataFrame, spec: str, metric: str, estimand: str = "vs_typical", order=None) -> go.Figure:
    d = eff.query("spec == @spec and estimand == @estimand and metric == @metric")
    order = order or list(d["style"])
    d = d.set_index("style").reindex(order).reset_index()
    fig = go.Figure(go.Scatter(
        x=d["value"], y=d["style"], mode="markers", marker={"size": 11},
        error_x={"type": "data", "symmetric": False, "array": d["hi"] - d["value"], "arrayminus": d["value"] - d["lo"]},
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_dash="dot")
    fig.update_layout(height=320, margin={"l": 10, "r": 10, "t": 30, "b": 10}, xaxis_title=f"{metric} vs. a typical opening, same situation",
                      yaxis={"autorange": "reversed"})
    return fig


def team_dna_bars(dna: pd.DataFrame, team: str, season: int, kind: str = "drive") -> go.Figure:
    """Percentage points above/below what the team's game situations predict, with +/-1 SE."""
    row = dna[(dna["team"] == team) & (dna["season"] == season)].iloc[0]
    order = ORDERS[kind]
    vals = np.array([row[f"{kind}_{s}_excess"] for s in order]) * 100
    se = np.array([row[f"{kind}_{s}_se"] for s in order]) * 100
    fig = go.Figure(go.Bar(x=order, y=vals, error_y={"type": "data", "array": se},
                           marker_color=["#d62728" if v > 0 else "#1f77b4" for v in vals]))
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.update_layout(height=340, margin={"l": 10, "r": 10, "t": 30, "b": 10}, yaxis_title="pp above expected")
    return fig


def team_trend(dna: pd.DataFrame, team: str, styles: list[str], kind: str = "drive") -> go.Figure:
    d = dna[dna["team"] == team].sort_values("season")
    fig = go.Figure()
    for s in styles:
        fig.add_trace(go.Scatter(x=d["season"], y=d[f"{kind}_{s}_excess"] * 100, mode="lines+markers", name=s))
    for _, r in d[d["hc_changed"] == 1].iterrows():
        fig.add_vline(x=r["season"] - 0.5, line_dash="dot", line_color="gray",
                      annotation_text=str(r["coach"]), annotation_position="top", annotation_font_size=10)
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.update_layout(height=380, margin={"l": 10, "r": 10, "t": 40, "b": 10}, yaxis_title="pp above expected",
                      xaxis={"dtick": 1})
    return fig


def decomposition_bars(dec: pd.DataFrame, team: str) -> go.Figure:
    d = dec[dec["team"] == team].sort_values("season")
    fig = go.Figure()
    for col, name in [("situation_part", "situation"), ("style_part", "opening style"), ("exec_part", "execution")]:
        fig.add_trace(go.Bar(x=d["season"], y=d[col], name=name))
    fig.add_trace(go.Scatter(x=d["season"], y=d["total_vs_league"], mode="markers", name="total vs. league",
                             marker={"color": "black", "size": 9, "symbol": "diamond"}))
    fig.update_layout(barmode="relative", height=380, margin={"l": 10, "r": 10, "t": 30, "b": 10},
                      yaxis_title="points per drive vs. league", xaxis={"dtick": 1})
    return fig


def defense_scatter(tend: pd.DataFrame, highlight: str | None = None) -> go.Figure:
    t = tend.assign(label=tend["defteam"] + " " + tend["season"].astype(str))
    fig = px.scatter(t, x="blitz_excess", y="box_excess", color="archetype", hover_name="label",
                     color_discrete_map={"Aggressive": "#ef553b", "Light": "#4c9be8"},
                     labels={"blitz_excess": "blitz tendency (share of dropbacks above expected)",
                             "box_excess": "box tendency (defenders above expected)"})
    if highlight is not None and (t["label"] == highlight).any():
        r = t[t["label"] == highlight].iloc[0]
        fig.add_trace(go.Scatter(x=[r["blitz_excess"]], y=[r["box_excess"]], mode="markers+text", text=[highlight],
                                 textposition="top center", marker={"size": 16, "color": "rgba(0,0,0,0)", "line": {"width": 2, "color": "black"}},
                                 showlegend=False))
    fig.update_layout(height=520, margin={"l": 10, "r": 10, "t": 30, "b": 10})
    return fig


def grouped_effects(df: pd.DataFrame, x: str, group: str, order: list[str], ytitle: str) -> go.Figure:
    fig = px.bar(df, x=x, y="effect", color=group, barmode="group", error_y=df["hi"] - df["effect"],
                 error_y_minus=df["effect"] - df["lo"], category_orders={x: order})
    fig.update_layout(height=420, margin={"l": 10, "r": 10, "t": 30, "b": 10}, yaxis_title=ytitle)
    return fig


def drive_replay_chart(plays: pd.DataFrame) -> go.Figure:
    """Field position and cumulative EPA through one drive, each play colored by its call."""
    p = plays.sort_values("play_num").reset_index(drop=True)
    p["n"] = np.arange(1, len(p) + 1)
    p["cum_epa"] = p["epa"].fillna(0).cumsum()
    p["label"] = p.apply(lambda r: f"{int(r['down']) if pd.notna(r['down']) else '-'}&{int(r['ydstogo'])}" if pd.notna(r["down"]) else "", axis=1)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.08,
                        subplot_titles=("Yards from the opponent's end zone at the snap", "Cumulative EPA"))
    for call, g in p.groupby("call"):
        fig.add_trace(go.Scatter(
            x=g["n"], y=g["yardline_100"], mode="markers+text", text=g["label"], textposition="top center", name=call,
            marker={"size": 13, "color": CALL_COLORS.get(call, "#999")},
            customdata=g[["call", "yards_gained", "epa"]].to_numpy(),
            hovertemplate="play %{x}: %{customdata[0]}<br>%{customdata[1]:.0f} yds, EPA %{customdata[2]:.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=p["n"], y=p["yardline_100"], mode="lines", line={"color": "lightgray"}, showlegend=False,
                             hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=p["n"], y=p["cum_epa"], mode="lines+markers", line={"color": "black"}, showlegend=False), row=2, col=1)
    fig.update_yaxes(autorange="reversed", row=1, col=1)
    fig.update_xaxes(dtick=1, title_text="play of the drive", row=2, col=1)
    fig.update_layout(height=560, margin={"l": 10, "r": 10, "t": 50, "b": 10})
    return fig
