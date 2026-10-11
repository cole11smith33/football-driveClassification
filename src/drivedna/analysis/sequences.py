"""Which play-call sequences go with touchdowns vs. punts?

The trap: touchdown drives are longer than punt drives, so almost any n-gram is more common in TD
drives simply because they have more plays. A naive comparison "discovers" length. So each n-gram is
tested with a Mantel-Haenszel odds ratio *stratified by drive length*: TD vs punt drives of the
same length. Multiple testing is controlled with Benjamini-Hochberg FDR.

Two views:
  anywhere  n-gram appears anywhere in the drive (length-stratified).
  opening   n-gram appears in the first `window` calls of drives that lasted at least `min_len`
            plays. Position-matched and free of end-of-drive mechanics (e.g. a drive that ends on a
            3rd-and-long incompletion tends to end in a short pass).
Tokens are calls, not results, but sequences are still only *associated* with outcomes.
"""
from collections import Counter

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import StratifiedTable
from statsmodels.stats.multitest import multipletests

DESIGNED_TOKENS = {"R_in", "R_out", "SCR", "P_short", "P_mid", "P_deep", "P_unk", "SACK"}
MAX_STRATUM = 13  # drives with this many plays or more share one stratum


def designed_calls(tokens: list[str]) -> list[str]:
    """Drop kneels and spikes: they are not play-calling."""
    return [t for t in tokens if t in DESIGNED_TOKENS]


def ngrams_present(calls: list[str], n: int, window: int | None = None) -> set[str]:
    seq = calls if window is None else calls[:window]
    return {" ".join(seq[i : i + n]) for i in range(len(seq) - n + 1)}


def mh_ngram_table(
    drives: pd.DataFrame,
    group_a: str = "Touchdown",
    group_b: str = "Punt",
    sizes=(2, 3, 4),
    window: int | None = None,
    min_len: int = 0,
    min_drives: int = 150,
    fdr: float = 0.05,
) -> pd.DataFrame:
    """Length-stratified odds ratios for every n-gram.

    drives needs: `calls` (list of call tokens, designed plays only) and `outcome`.
    OR > 1 means the n-gram is more common in `group_a` drives than `group_b` drives of equal length.
    """
    d = drives[drives["outcome"].isin([group_a, group_b]) & (drives["calls"].str.len() >= min_len)].copy()
    d["is_a"] = (d["outcome"] == group_a).to_numpy()
    d["stratum"] = d["calls"].str.len().clip(upper=MAX_STRATUM)
    strata = {s: g.index.to_numpy() for s, g in d.groupby("stratum")}
    n_by_stratum = {s: (d.loc[i, "is_a"].sum(), (~d.loc[i, "is_a"]).sum()) for s, i in strata.items()}

    rows = []
    for n in sizes:
        present = d["calls"].map(lambda c, n=n: ngrams_present(c, n, window))
        counts = Counter(g for s in present for g in s)
        for gram, total in counts.items():
            if total < min_drives:
                continue
            has = present.map(lambda s, gram=gram: gram in s)
            tables = []
            for s, idx in strata.items():
                a_tot, b_tot = n_by_stratum[s]
                a_has = int((has.loc[idx] & d.loc[idx, "is_a"]).sum())
                b_has = int((has.loc[idx] & ~d.loc[idx, "is_a"]).sum())
                tables.append([[a_has, a_tot - a_has], [b_has, b_tot - b_has]])
            st = StratifiedTable(np.array(tables, dtype=float).transpose(1, 2, 0))
            a_all, b_all = d["is_a"], ~d["is_a"]
            naive = _naive_or(int((has & a_all).sum()), int(a_all.sum()), int((has & b_all).sum()), int(b_all.sum()))
            lo, hi = st.oddsratio_pooled_confint()
            rows.append({
                "ngram": gram, "n": n, "drives": total,
                f"share_{group_a}": (has & a_all).sum() / a_all.sum(),
                f"share_{group_b}": (has & b_all).sum() / b_all.sum(),
                "or_naive": naive, "or_stratified": st.oddsratio_pooled, "or_lo": lo, "or_hi": hi,
                "p": st.test_null_odds(correction=True).pvalue,
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q"] = multipletests(out["p"], method="fdr_bh")[1]
    out["significant"] = out["q"] < fdr
    return out.sort_values("or_stratified", ascending=False).reset_index(drop=True)


def _naive_or(a_has: int, a_tot: int, b_has: int, b_tot: int) -> float:
    """Unstratified odds ratio with a 0.5 continuity correction (for contrast with the stratified one)."""
    a1, a0 = a_has + 0.5, a_tot - a_has + 0.5
    b1, b0 = b_has + 0.5, b_tot - b_has + 0.5
    return (a1 / a0) / (b1 / b0)
