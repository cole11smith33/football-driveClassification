#!/usr/bin/env bash
# Re-run every notebook in place and save its outputs, with charts as static PNGs so they show up
# on GitHub (GitHub does not render Plotly's interactive output). Needs the pipeline outputs in
# data/processed and reports/, plus kaleido (in the dev extra) and a local Chrome for the export.
# Opening a notebook yourself still gives interactive charts: the PNG renderer is set only here.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
export PLOTLY_RENDERER=png
[ $# -gt 0 ] || set -- notebooks/*.ipynb
for nb in "$@"; do
  echo "executing $nb"
  "$PY" -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 "$nb"
done

# Drop library warnings (stderr) from the saved outputs: noise for readers, and they embed local paths.
"$PY" - "$@" <<'PYEOF'
import json, sys
for path in sys.argv[1:]:
    nb = json.load(open(path))
    for cell in nb["cells"]:
        if "outputs" in cell:
            cell["outputs"] = [o for o in cell["outputs"] if o.get("name") != "stderr"]
    with open(path, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
PYEOF
