"""Discover drive styles: impute -> shrink short drives -> standardize -> PCA -> KMeans.

Design choices (each backed by experiments in notebooks/02_styles.ipynb):
- Only STYLE_FEATURES are used. Outcomes, rhythm and context never enter the model. In particular
  scramble_rate is excluded: with it, one cluster was mostly "this drive had a QB scramble"
  (83% of its drives had exactly one), which is a reaction to pressure rather than a style.
- Drives need >= MIN_REAL_PLAYS designed plays. Rates on 1-2 plays are mostly noise.
- Rates are shrunk toward the league mean in proportion to drive length (empirical-Bayes style),
  so a 3-play drive can't land in an extreme cluster on a fluke. This halves the drive-length
  confound and makes clusters more stable.
- KMeans, not a GMM: features like no_huddle_rate are exactly 0 for most drives, and a GMM chases
  those point masses (BIC never levels off, bootstrap ARI 0.4-0.75 vs 0.9+ for KMeans).
- k=5, chosen by whole-pipeline stability (reports/k_stability.csv), not by silhouette or BIC:
  refit everything on 30 bootstrap resamples and compare to the full-data model, and check that a
  model fit on 2016-2020 agrees with one fit on 2021-2025. k=3-4 are rock solid (ARI ~0.97, never an
  alternative partition). k=5 is the finest that is still good (mean ARI ~0.95, temporal ~0.91), but
  ~1 in 15 refits lands on a different near-equal partition, so k=4 is the conservative choice.
  k >= 6 degrades (temporal ARI 0.53 at k=6). Silhouette is low for every k (<= ~0.18): styles are
  a continuum, not natural clumps, so these are useful partitions, not discovered species.
"""
import itertools
import json
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from drivedna.drives.build import DRIVE_KEY, PROCESSED_DIR
from drivedna.features.drive_features import RHYTHM_FEATURES, STYLE_FEATURES

REPORTS_DIR = PROCESSED_DIR.parents[1] / "reports"

MIN_REAL_PLAYS = 3
SHRINK_K = 4.0          # a drive with n plays gets weight n / (n + SHRINK_K) on its own rates
N_CLUSTERS = 5
PCA_VARIANCE = 0.90

# Canonical order so style_id means the same thing across refits (only defined for k=5).
STYLE_ORDER = ["Ground-based", "Quick-game", "Vertical", "Hurry-up", "No-huddle"]


def eligible_drives(features: pl.DataFrame) -> pl.DataFrame:
    """Drives passing the Phase 1 filters and long enough to say something about style."""
    return features.filter(pl.col("keep") & (pl.col("n_real_plays") >= MIN_REAL_PLAYS))


NAMING_FEATURES = ["no_huddle_rate", "deep_pass_share", "sec_per_play", "pass_rate"]


def name_clusters(profile: pd.DataFrame) -> dict[int, str]:
    """Map raw cluster label -> style name from centroid profile (mean style features).

    Rules are applied in order, each taking the best remaining cluster, so names follow the
    clusters even when KMeans labels come out permuted. Only defined for 5 clusters.
    """
    if len(profile) != len(STYLE_ORDER) or not set(NAMING_FEATURES) <= set(profile.columns):
        return {int(c): f"Style {i}" for i, c in enumerate(profile.sort_values("pass_rate").index)}
    remaining = list(profile.index)
    names: dict[int, str] = {}
    rules = [
        ("No-huddle", "no_huddle_rate", "max"),
        ("Vertical", "deep_pass_share", "max"),
        ("Hurry-up", "sec_per_play", "min"),
        ("Ground-based", "pass_rate", "min"),
    ]
    for name, col, how in rules:
        vals = profile.loc[remaining, col]
        pick = vals.idxmax() if how == "max" else vals.idxmin()
        names[int(pick)] = name
        remaining.remove(pick)
    names[int(remaining[0])] = "Quick-game"
    return names


@dataclass
class StyleModel:
    k: int = N_CLUSTERS
    shrink_k: float = SHRINK_K
    var_explained: float = PCA_VARIANCE
    seed: int = 0
    features: list[str] = field(default_factory=lambda: list(STYLE_FEATURES))

    def _raw(self, df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x = df.select(self.features).to_numpy().astype(float)
        n = df["n_real_plays"].to_numpy().astype(float)
        return x, n

    def _shrunk(self, x: np.ndarray, n: np.ndarray) -> np.ndarray:
        x = self.imputer_.transform(x)
        w = (n / (n + self.shrink_k))[:, None]
        return w * x + (1 - w) * self.league_mean_

    def transform(self, df: pl.DataFrame) -> np.ndarray:
        """PCA coordinates for each drive."""
        x, n = self._raw(df)
        return self.pca_.transform(self.scaler_.transform(self._shrunk(x, n)))

    def fit(self, df: pl.DataFrame) -> "StyleModel":
        x, n = self._raw(df)
        self.imputer_ = SimpleImputer(strategy="median").fit(x)
        x_imp = self.imputer_.transform(x)
        self.league_mean_ = x_imp.mean(axis=0)
        shrunk = self._shrunk(x, n)
        self.scaler_ = StandardScaler().fit(shrunk)
        self.pca_ = PCA(n_components=self.var_explained, random_state=self.seed).fit(
            self.scaler_.transform(shrunk)
        )
        z = self.pca_.transform(self.scaler_.transform(shrunk))
        km = KMeans(self.k, n_init=10, random_state=self.seed).fit(z)

        # name clusters from their (unshrunk) mean features, then relabel into canonical order
        raw_profile = pd.DataFrame(x_imp, columns=self.features).groupby(km.labels_).mean()
        names = name_clusters(raw_profile)
        order = STYLE_ORDER if set(names.values()) == set(STYLE_ORDER) else [names[c] for c in sorted(names)]
        self.style_names_ = order
        self.label_map_ = {raw: order.index(name) for raw, name in names.items()}
        self.kmeans_ = km
        return self

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        """Canonical style id (index into style_names_) for each drive."""
        raw = self.kmeans_.predict(self.transform(df))
        return np.array([self.label_map_[int(r)] for r in raw])

    def margin(self, df: pl.DataFrame) -> np.ndarray:
        """Assignment confidence in [0, 1]: 0 = equidistant from the two nearest styles."""
        d = np.sort(self.kmeans_.transform(self.transform(df)), axis=1)
        return (d[:, 1] - d[:, 0]) / (d[:, 1] + d[:, 0])


def style_table(model: StyleModel, df: pl.DataFrame) -> pl.DataFrame:
    ids = model.predict(df)
    return df.select(DRIVE_KEY).with_columns(
        style_id=pl.Series(ids, dtype=pl.Int32),
        style=pl.Series([model.style_names_[i] for i in ids]),
        margin=pl.Series(model.margin(df)),
    )


def style_profiles(df: pl.DataFrame, styles: pl.DataFrame) -> pd.DataFrame:
    """Per-style means of style features, rhythm, context and (uncontrolled!) outcomes."""
    cols = (
        STYLE_FEATURES + RHYTHM_FEATURES
        + ["n_real_plays", "start_yardline_100", "score_diff_start", "points", "scored", "epa"]
    )
    joined = df.join(styles, on=DRIVE_KEY).to_pandas()
    prof = joined.groupby("style_id")[cols].mean()
    prof.insert(0, "share", joined.groupby("style_id").size() / len(joined))
    names = joined.drop_duplicates("style_id").set_index("style_id")["style"]
    prof.insert(0, "style", [names[i] for i in prof.index])
    return prof


# --------------------------------------------------------------------------- model selection
def k_selection(
    df: pl.DataFrame, ks=range(2, 11), n_boot: int = 6, seed: int = 0, sil_sample: int = 8000
) -> pd.DataFrame:
    """Silhouette and bootstrap ARI per k for KMeans and a tied-covariance GMM.

    Uses one fixed preprocessing (fit on all data) so methods and k are compared like for like.
    """
    base = StyleModel(k=2, seed=seed).fit(df)
    z = base.transform(df)
    rng = np.random.default_rng(seed)
    sil_idx = rng.choice(len(z), min(sil_sample, len(z)), replace=False)
    rows = []
    for method, k in itertools.product(["kmeans", "gmm_tied"], ks):
        preds = []
        for b in range(n_boot):
            sample = z[rng.choice(len(z), len(z), replace=True)]
            if method == "kmeans":
                m = KMeans(k, n_init=10, random_state=b).fit(sample)
            else:
                m = GaussianMixture(k, covariance_type="tied", random_state=b).fit(sample)
            preds.append(m.predict(z))
        ari = np.mean([adjusted_rand_score(a, b) for a, b in itertools.combinations(preds, 2)])
        rows.append({
            "method": method, "k": k, "bootstrap_ari": ari,
            "silhouette": silhouette_score(z[sil_idx], preds[0][sil_idx]),
            "min_cluster_share": np.bincount(preds[0], minlength=k).min() / len(z),
        })
    return pd.DataFrame(rows)


def bootstrap_stability(
    df: pl.DataFrame, k: int = N_CLUSTERS, n_boot: int = 30, seed: int = 0
) -> dict:
    """Refit the *whole* pipeline on bootstrap resamples; compare each to the full-data model.

    KMeans can have near-equal alternative partitions, so some resamples land on a different one.
    Beyond the mean ARI, `frac_below_0_8` says how often that happens, which a small number of
    draws can easily miss.
    """
    rng = np.random.default_rng(seed)
    reference = StyleModel(k=k, seed=0).fit(df).predict(df)
    aris = []
    for b in range(n_boot):
        idx = rng.choice(df.height, df.height, replace=True)
        pred = StyleModel(k=k, seed=b + 1).fit(df[idx]).predict(df)
        aris.append(adjusted_rand_score(reference, pred))
    aris = np.array(aris)
    return {
        "n_boot": n_boot,
        "ari_mean": float(aris.mean()),
        "ari_min": float(aris.min()),
        "frac_below_0_8": float((aris < 0.8).mean()),
    }


def temporal_stability(df: pl.DataFrame, k: int = N_CLUSTERS, split_season: int = 2021) -> dict:
    """Fit on early seasons and on late seasons separately; compare on the late drives."""
    early, late = df.filter(pl.col("season") < split_season), df.filter(pl.col("season") >= split_season)
    a, b = StyleModel(k=k).fit(early), StyleModel(k=k).fit(late)
    pa, pb = a.predict(late), b.predict(late)
    share = lambda p: (np.bincount(p, minlength=k) / len(p)).round(3).tolist()
    # agreement_rate compares canonical ids, so it is only meaningful when k == len(STYLE_ORDER)
    return {
        "split_season": split_season,
        "ari_early_model_vs_late_model_on_late_drives": float(adjusted_rand_score(pa, pb)),
        "agreement_rate": float((pa == pb).mean()),
        "style_order": STYLE_ORDER,
        "share_early_model_on_early": share(a.predict(early)),
        "share_early_model_on_late": share(pa),
        "share_late_model_on_late": share(pb),
    }


def k_stability(df: pl.DataFrame, ks=range(3, 9), n_boot: int = 30) -> pd.DataFrame:
    """Whole-pipeline stability per k: the evidence behind N_CLUSTERS."""
    rows = []
    for k in ks:
        boot, temporal = bootstrap_stability(df, k=k, n_boot=n_boot), temporal_stability(df, k=k)
        rows.append({
            "k": k,
            "bootstrap_ari_mean": boot["ari_mean"],
            "bootstrap_ari_min": boot["ari_min"],
            "bootstrap_frac_below_0_8": boot["frac_below_0_8"],
            "temporal_ari": temporal["ari_early_model_vs_late_model_on_late_drives"],
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------- pipeline
def build_styles(run_selection: bool = True) -> pl.DataFrame:
    """Fit the final model and write all Phase 3 artifacts."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = eligible_drives(pl.read_parquet(PROCESSED_DIR / "drive_features.parquet"))
    model = StyleModel().fit(df)
    styles = style_table(model, df)
    styles.write_parquet(PROCESSED_DIR / "drive_styles.parquet")
    joblib.dump(model, PROCESSED_DIR / "style_model.joblib")
    style_profiles(df, styles).round(4).to_csv(REPORTS_DIR / "style_profiles.csv")
    stability = {
        "bootstrap": bootstrap_stability(df),
        "temporal": temporal_stability(df),
        "n_drives": df.height,
    }
    (REPORTS_DIR / "stability.json").write_text(json.dumps(stability, indent=2))
    if run_selection:
        k_selection(df, n_boot=8).round(4).to_csv(REPORTS_DIR / "k_selection.csv", index=False)
        k_stability(df).round(4).to_csv(REPORTS_DIR / "k_stability.csv", index=False)
    return styles


if __name__ == "__main__":
    # Import under the real module name: running with `-m` makes StyleModel `__main__.StyleModel`,
    # and a model pickled that way can't be loaded from anywhere else.
    from drivedna.models.cluster import REPORTS_DIR, build_styles

    out = build_styles()
    print(out.group_by("style").len().sort("len", descending=True))
    print((REPORTS_DIR / "stability.json").read_text())
