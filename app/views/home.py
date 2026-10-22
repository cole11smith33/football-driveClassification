import streamlit as st

from drivedna.app import cached, ui

h = cached.headlines()
st.title("🏈 Drive DNA")
st.caption("NFL analytics that treat the *drive*, not the play, as the unit of analysis. nflverse play-by-play, 2016-2025.")

c1, c2, c3, c4 = st.columns(4)
ui.stat(c1, "Drives analyzed", f"{h['n_drives']:,}", "2016-2025, non-garbage-time")
ui.stat(c2, "Deep-shot opener", f"{h['deep_shot_points']:+.2f} pts/drive",
        f"vs. a typical opening in the same situation (95% interval {h['deep_lo']:+.2f} to {h['deep_hi']:+.2f})")
ui.stat(c3, "Team gaps from execution", f"{h['exec_variance_share']:.0%}",
        "of the variance in team points per drive; opening-style choices explain almost none")
ui.stat(c4, "New head coach: style shift", f"{h['hc_change_shift']:+.2f}",
        f"extra season-to-season change vs. no change; a new QB alone: {h['qb_change_shift']:+.2f} (not distinguishable from zero)")

st.subheader("What you can explore")
st.markdown(
    """
- **🧬 Drive styles**: the five whole-drive styles, how they changed over the years, example drives, and what the way a drive *opens* is worth.
- **🏟️ Team DNA**: any team-season's style tendencies (adjusted for game script), history with coaching changes, similar teams, and execution vs. style.
- **🛡️ Defenses**: blitz and box tendencies for every defense-season, and the two archetypes that hold up.
- **🔁 Calls & sequences**: which calls work in the red zone, the two-minute drill and the fourth quarter, pass depth against blitzes and coverage, and play-call patterns tied to touchdowns vs. punts.
- **🎬 Drive replay**: step through any drive play by play with its style classification.
"""
)

st.subheader("How to read it")
st.info(
    f"""
- **Style is separated from outcome.** Drive styles are built only from *how* a drive was played, never from how it ended.
- **Everything is observational.** Adjusted comparisons hold measured context fixed (field position, score, time, opponent,
  team quality), not what a coach sees. Read effects as strong associations, not proven causes.
- **Stability was checked.** Refitting the style model on resampled data gives {h['style_bootstrap_ari']:.2f} agreement, and models
  fit on early vs. late seasons agree at {h['style_temporal_ari']:.2f}. Styles are a *useful partition of a continuum*, not natural clumps.
- Each page ends with the caveats that matter for what it shows.
"""
)
