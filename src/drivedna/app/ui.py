"""Small UI helpers shared by the pages."""
import html

import streamlit as st


# Team colours (primary, secondary, accent), used to theme the page background.
TEAM_COLORS = {
    "ARI": ("#97233F", "#000000", "#FFB612"), "ATL": ("#A71930", "#000000", "#A5ACAF"),
    "BAL": ("#241773", "#000000", "#9E7C0C"), "BUF": ("#00338D", "#C60C30", "#FFFFFF"),
    "CAR": ("#0085CA", "#101820", "#BFC0BF"), "CHI": ("#0B162A", "#C83803", "#FFFFFF"),
    "CIN": ("#FB4F14", "#000000", "#FFFFFF"), "CLE": ("#311D00", "#FF3C00", "#FFFFFF"),
    "DAL": ("#003594", "#869397", "#FFFFFF"), "DEN": ("#FB4F14", "#002244", "#FFFFFF"),
    "DET": ("#0076B6", "#B0B7BC", "#FFFFFF"), "GB": ("#203731", "#FFB612", "#FFFFFF"),
    "HOU": ("#03202F", "#A71930", "#FFFFFF"), "IND": ("#002C5F", "#A2AAAD", "#FFFFFF"),
    "JAX": ("#006778", "#101820", "#D7A22A"), "KC": ("#E31837", "#FFB81C", "#FFFFFF"),
    "LA": ("#003594", "#FFA300", "#FFFFFF"), "LAC": ("#0080C6", "#FFC20E", "#FFFFFF"),
    "LV": ("#000000", "#A5ACAF", "#FFFFFF"), "MIA": ("#008E97", "#FC4C02", "#FFFFFF"),
    "MIN": ("#4F2683", "#FFC62F", "#FFFFFF"), "NE": ("#002244", "#C60C30", "#B0B7BC"),
    "NO": ("#D3BC8D", "#101820", "#FFFFFF"), "NYG": ("#0B2265", "#A71930", "#FFFFFF"),
    "NYJ": ("#125740", "#000000", "#FFFFFF"), "PHI": ("#004C54", "#A5ACAF", "#FFFFFF"),
    "PIT": ("#FFB612", "#101820", "#FFFFFF"), "SEA": ("#002244", "#69BE28", "#A5ACAF"),
    "SF": ("#AA0000", "#B3995D", "#FFFFFF"), "TB": ("#D50A0A", "#FF7900", "#0A0A08"),
    "TEN": ("#0C2340", "#4B92DB", "#C8102E"), "WAS": ("#5A1414", "#FFB612", "#FFFFFF"),
}


# Franchises that appear under older codes in earlier seasons map to their current colours.
TEAM_ALIASES = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA", "WSH": "WAS", "JAC": "JAX", "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU"}


def team_theme(team: str) -> None:
    """Paint the page background in a team's colours (a diagonal stripe gradient).

    The main content sits on a translucent panel in the app's own background colour so text and
    charts stay readable in both light and dark mode.
    """
    colors = TEAM_COLORS.get(TEAM_ALIASES.get(team, team))
    if not colors:
        return
    a, b, c = (f"color-mix(in srgb, {x} 82%, white)" for x in colors)  # lift very dark colours a touch
    st.markdown(
        f"""<style>
        .stApp {{ background: linear-gradient(135deg, {a} 0%, {a} 55%, {b} 55%, {b} 80%, {c} 80%, {c} 100%) fixed; }}
        [data-testid="stHeader"] {{ background: transparent; }}
        [data-testid="stMainBlockContainer"] {{
            background: color-mix(in srgb, var(--background-color, Canvas) 86%, transparent);
            border-radius: 1rem; margin-top: 1rem; padding: 2rem 2.5rem; border-top: 6px solid {b};
            box-shadow: 0 4px 24px rgba(0,0,0,0.35);
        }}
        </style>""",
        unsafe_allow_html=True,
    )


def stat(container, label: str, value, note: str | None = None) -> None:
    """A compact stat card that wraps long values (st.metric truncates them with an ellipsis).

    Everything interpolated into the HTML is escaped, so data values cannot inject markup.
    """
    note_html = f"<div style='font-size:0.75rem;opacity:0.6;line-height:1.3'>{html.escape(note)}</div>" if note else ""
    container.markdown(
        f"<div style='font-size:0.85rem;opacity:0.7'>{html.escape(label)}</div>"
        f"<div style='font-size:1.5rem;font-weight:600;line-height:1.3'>{html.escape(str(value))}</div>{note_html}",
        unsafe_allow_html=True,
    )


__all__ = ["TEAM_ALIASES", "TEAM_COLORS", "st", "stat", "team_theme"]
