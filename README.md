# Drive DNA

NFL analytics that treat the **drive** as the unit of analysis: find distinct styles of offensive
drives, see how they develop, and test what actually separates successful drives from unsuccessful
ones. Built on free [nflverse](https://github.com/nflverse) play-by-play, 2016 to present.

Guiding rule: **separate "style" from "outcome."** Styles are built only from *how* a drive was run,
never from how it ended. Effectiveness is then measured with the game situation held fixed.

## Pipeline

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"

.venv/bin/python -m drivedna.data.ingest                  # 1. download pbp -> data/raw (Parquet)
.venv/bin/python -m drivedna.drives.build                 # 1. drives + ordered plays + filter flags
.venv/bin/python -m drivedna.features.drive_features      # 2. style / rhythm / context / outcome features
.venv/bin/python -m drivedna.models.cluster               # 3. drive styles + stability reports
.venv/bin/python -m drivedna.analysis.run_effectiveness   # 4. adjusted style effects (~20 min)
.venv/bin/python -m drivedna.analysis.run_secondary       # 4. call sequences + early-drive prediction
.venv/bin/pytest                                          # tests (need the data/reports above)
```

Notebooks in `notebooks/` are thin: they read the saved reports and plot. Logic lives in `src/drivedna/`.

| Phase | Code | Output |
|---|---|---|
| 1 Drive table | `data/ingest.py`, `drives/build.py` | `data/processed/drives.parquet`, `drive_plays.parquet` |
| 2 Features | `features/` | `drive_features.parquet`, play tokens |
| 3 Styles | `models/cluster.py` | `drive_styles.parquet`, `reports/stability.json`, `k_stability.csv` |
| 4 What works | `analysis/` | `reports/effectiveness.csv`, `sequences_*.csv`, `early_*.csv` |

## Findings so far (details and caveats in the notebooks)

- **Styles are a continuum.** Five whole-drive styles (Quick-game, Ground-based, Vertical, Hurry-up,
  No-huddle) are a stable, useful partition, not natural clumps.
- **The way a drive *opens* is associated with how it ends**, holding situation and team quality fixed.
  Deep-shot openers do best. See `notebooks/03_effectiveness.ipynb` for sizes, intervals and checks.
- **The effect runs through progress.** After the first three plays, where the drive stands predicts
  the outcome far better than style does, and style adds nothing on top of it.

All estimates are observational: they adjust for what is measured, not for everything a coach sees.
