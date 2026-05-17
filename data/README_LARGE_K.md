# Medium- and Large-K datasets

These six datasets are referenced in `preorder4mlc/config.py` but their ARFF
files are gitignored (large + downloadable). Sources below.

## Status (verified loadable + PA/PR fit completes)

### Medium-K (K = 19-53) — fit on commodity RAM (Mac M1 32 GB)

| Key | File | n × features × K | Label freq (min/max) | Load + fit (PR, 50/50 split) |
|---|---|---|---|---|
| `birds` | `birds.arff` | 645 × 260 × 19 | 6 / 103 | load 0.05s, fit 12s, predict_orders 16s |
| `medical` | `medical.arff` | 978 × 1449 × 45 | 1 / 266 | load 0.42s, fit 26s |
| `enron` | `enron.arff` | 1702 × 1001 × 53 | 1 / 913 | load 0.52s, fit 64s |

### Large-K (K = 101-174) — currently requires high-RAM node

| Key | File | n × features × K | Label freq (min/max) | Load time | Status |
|---|---|---|---|---|---|
| `mediamill` | `mediamill.arff` | 43907 × 120 × 101 | 31 / 33869 | 2.5s | Fit needs ~200 GB RAM (G matrix dense) |
| `bibtex` | `bibtex.arff` | 7395 × 1836 × 159 | 51 / 1042 | 3.7s (sparse, via liac-arff) | Blocked: G dense ~1.5 TB |
| `cal500` | `CAL500.arff` | 502 × 68 × 174 | 5 / 444 | 0.04s | Blocked: G dense ~2.5 TB |

## Download

All six ARFFs are hosted at the Cometa archive under a uniform URL:

```bash
cd data/
for ds in birds medical enron mediamill bibtex; do
    curl -sLO "https://cometa.ujaen.es/public/full/${ds}.arff"
done
# CAL500 is uppercase on the server:
curl -sLo CAL500.arff "https://cometa.ujaen.es/public/full/CAL500.arff"
```

COMETA places labels at the end of the attribute list — already configured
via `TARGET_IN_END_FILE_DATASETS` in `preorder4mlc/datasets4experiments.py`.
The sparse-ARFF format used by `bibtex`, `enron`, `medical` is handled
transparently via the `liac-arff` fallback in the same module.

## Smoke test after download

```bash
python -c "from preorder4mlc.datasets4experiments import Datasets4Experiments; \
  d = Datasets4Experiments('./data/', [{'dataset_name':'enron.arff','n_labels_set':53}]); \
  d.load_datasets(); X,Y,name = d.datasets[0]; print(name, X.shape, Y.shape, Y.sum(0).min(), Y.sum(0).max())"
```

Expected: shapes match the tables above; min/max label frequencies > 0.

## Scalability flags

For K > 30 the following baselines are auto-skipped (no point running):
- `lp` — LabelPowerset blows up combinatorially
- `clr` — pairwise expansion is K(K-1)/2 (already costly at K=14, infeasible at K=100+)

Set `PREORDER_MAX_K_LP=999` to override.
