"""Run the sequence-pattern and early-drive-prediction analyses; write CSVs to reports/."""
import warnings

import pandas as pd
import polars as pl

from drivedna.analysis.early import (
    build_early_table,
    evaluate,
    group_permutation_importance,
    style_increment_ci,
)
from drivedna.analysis.modeling_table import build_model_table
from drivedna.analysis.sequences import designed_calls, mh_ngram_table
from drivedna.drives.build import DRIVE_KEY, PROCESSED_DIR
from drivedna.features.tokens import drive_token_sequences
from drivedna.models.cluster import REPORTS_DIR


def run() -> None:
    warnings.filterwarnings("ignore")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    seq = drive_token_sequences(pl.read_parquet(PROCESSED_DIR / "drive_plays.parquet")).to_pandas()
    df = build_model_table().merge(seq[[*DRIVE_KEY, "tokens"]], on=DRIVE_KEY)
    df["calls"] = df["tokens"].map(lambda t: designed_calls(list(t)))
    mh_ngram_table(df, sizes=(2, 3, 4), min_drives=150).round(5).to_csv(
        REPORTS_DIR / "sequences_anywhere.csv", index=False)
    mh_ngram_table(df, sizes=(2, 3, 4), window=4, min_len=6, min_drives=150).round(5).to_csv(
        REPORTS_DIR / "sequences_opening.csv", index=False)

    early = build_early_table()
    rows, importance = [], []
    for target in ["scored", "touchdown"]:
        res, detail = evaluate(early, target)
        res["target"] = target
        rows.append(res)
        for name, base, full in [
            ("style on top of context", ("context", "logit"), ("context+style", "logit")),
            ("progress on top of context", ("context", "logit"), ("context+progress", "logit")),
            ("style on top of context+progress", ("context+progress", "logit"), ("context+style+progress", "logit")),
        ]:
            inc = style_increment_ci(detail, base, full)
            rows.append(pd.DataFrame([{"model": name, "kind": "increment", "target": target, **inc}]))
        imp = group_permutation_importance(early, target)
        imp["target"] = target
        importance.append(imp)
    pd.concat(rows, ignore_index=True).round(5).to_csv(REPORTS_DIR / "early_prediction.csv", index=False)
    pd.concat(importance, ignore_index=True).round(5).to_csv(REPORTS_DIR / "early_importance.csv", index=False)
    early.groupby("season").size().to_csv(REPORTS_DIR / "early_n_by_season.csv")


if __name__ == "__main__":
    from drivedna.analysis.run_secondary import run as _run

    _run()
    print("done")
