import numpy as np
import pandas as pd

from drivedna.analysis.sequences import designed_calls, mh_ngram_table, ngrams_present

TOKENS = ["R_in", "R_out", "P_short", "P_mid", "P_deep"]


def _simulate(n=12000, seed=0, planted="P_deep P_deep", effect=1.0):
    """TD drives are longer (the confound). Only `planted` truly raises TD odds."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        length = int(rng.integers(3, 13))
        calls = list(rng.choice(TOKENS, size=length))
        has_planted = planted in " ".join(calls)
        logit = -1.2 + 0.35 * (length - 7) + effect * has_planted  # length drives TD, so does planted
        outcome = "Touchdown" if rng.random() < 1 / (1 + np.exp(-logit)) else "Punt"
        rows.append({"calls": calls, "outcome": outcome})
    return pd.DataFrame(rows)


def test_ngrams_present_and_window():
    assert ngrams_present(["a", "b", "c"], 2) == {"a b", "b c"}
    assert ngrams_present(["a", "b", "c"], 2, window=2) == {"a b"}
    assert ngrams_present(["a"], 2) == set()


def test_designed_calls_drops_kneels_and_spikes():
    assert designed_calls(["R_in", "KNEEL", "SPIKE", "P_deep"]) == ["R_in", "P_deep"]


def test_with_no_real_effect_stratified_test_finds_nothing_but_naive_is_fooled():
    """TD drives are longer, so every common n-gram looks 'predictive' naively; stratifying fixes it."""
    tab = mh_ngram_table(_simulate(effect=0.0), sizes=(2,), min_drives=300)
    assert tab["significant"].mean() < 0.15                      # ~ the 5% we expect by chance
    assert (tab["or_naive"] - 1).abs().mean() > 3 * (tab["or_stratified"] - 1).abs().mean()
    assert (tab["or_naive"] > 1.2).sum() >= 3                    # naive finds spurious "TD patterns"


def test_planted_effect_is_the_top_ranked_pattern_with_the_right_size():
    tab = mh_ngram_table(_simulate(effect=1.0), sizes=(2,), min_drives=300)
    top = tab.iloc[tab["or_stratified"].sub(1).abs().argmax()]
    assert top["ngram"] == "P_deep P_deep" and top["significant"]
    assert 2.0 < top["or_stratified"] < 3.6                      # true odds ratio is exp(1.0) ~ 2.7


def test_min_len_and_window_restrict_the_comparison():
    d = _simulate(4000)
    all_len = mh_ngram_table(d, sizes=(2,), min_drives=50)
    opening = mh_ngram_table(d, sizes=(2,), window=3, min_len=6, min_drives=50)
    assert opening["drives"].max() < all_len["drives"].max()
