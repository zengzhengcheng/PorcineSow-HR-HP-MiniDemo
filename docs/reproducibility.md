# Reproducibility guide

## Requirements

- Python 3.10 or newer (tested on 3.11).
- About 200 MB of disk space (mostly the dataset + matplotlib + scikit-learn
  wheels).
- A CPU is sufficient. The full 5-fold cross-validation takes under one
  minute on a modern laptop.

## Environment

Either of the two recipes below works.

### Conda

```bash
conda env create -f environment.yml
conda activate sow-mini
```

### Pip + virtualenv

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Verify the dataset

```bash
sha256sum -c data/sha256sums.txt    # Linux / Mac
```

Windows PowerShell:

```powershell
Get-FileHash data\sow_features_minimal.csv -Algorithm SHA256
Get-Content data\sha256sums.txt
```

The two values must match.

## Run

```bash
python code/train_baseline.py --config code/config_baseline.yaml
```

Optional overrides:

```bash
python code/train_baseline.py \
    --config code/config_baseline.yaml \
    --data data/sow_features_minimal.csv \
    --out results
```

## Expected output

`results/cv_summary.csv` and `results/cv_summary.md` will be (re)written. With
the shipped data and `random_state=42` on a recent scikit-learn (1.3+) the
expected mean R^2 is **0.79 +/- 0.01** and the expected mean MAE is **2.31
kcal / 5 min**. Small deviations (third decimal of R^2) across minor
scikit-learn versions are normal.

The shipped `results/cv_summary.csv` in the repo is the reference: a re-run
should match it to four decimal places on the same scikit-learn version.

`results/kfold_pred_vs_true.png` and `results/feature_importance.png` are also
re-generated.

## Troubleshooting

- *`ValueError: missing required columns`* - you are pointing the script at
  a CSV that does not contain the expected schema. Use the shipped
  `data/sow_features_minimal.csv`.
- *Slightly different R^2* - on scikit-learn versions earlier than 1.3 the
  histogram-based gradient booster has slightly different binning. Upgrade
  scikit-learn or accept up to 0.005 absolute R^2 difference.
- *Matplotlib backend error* - the script sets `matplotlib.use("Agg")` so
  no display is required.
