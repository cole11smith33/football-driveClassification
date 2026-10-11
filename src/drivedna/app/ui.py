"""Small UI helpers shared by the pages."""
import html

import streamlit as st


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


__all__ = ["st", "stat"]
