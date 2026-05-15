# Large-K datasets (algorithm improvement study)

These three datasets are referenced in `preorder4mlc/config.py`. ARFFs are
already in this directory (downloaded from COMETA). Total disk: ~57 MB.

## Status (verified)

| Key | File | n × features × K | Label freq (min/max) | Load time |
|---|---|---|---|---|
| `cal500` | `CAL500.arff` | 502 × 68 × 174 | 5 / 444 | 0.04s |
| `bibtex` | `bibtex.arff` | 7395 × 1836 × 159 | 51 / 1042 | 3.7s (sparse, via liac-arff) |
| `mediamill` | `mediamill.arff` | 43907 × 120 × 101 | 31 / 33869 | 2.5s |

These three datasets are referenced in `preorder4mlc/config.py`. If you need to
re-download them yourself:

| Key | Name | K (labels) | n (instances) | Source |
|---|---|---|---|---|
| `cal500` | CAL500 | 174 | 502 | https://www.uco.es/kdis/mllresources/ |
| `mediamill` | mediamill | 101 | 43907 | https://www.uco.es/kdis/mllresources/ |
| `bibtex` | bibtex | 159 | 7395 | https://www.uco.es/kdis/mllresources/ |

## Download steps

1. Go to COMETA Multi-Label Learning Resources (link above).
2. Download the `.arff` for each dataset.
3. Verify label convention: COMETA places labels at the end of the attribute list
   — already configured in `TARGET_IN_END_FILE_DATASETS`.
4. Rename to match the filename in `DATASET_CONFIGS` (case-sensitive):
   - `CAL500.arff`
   - `mediamill.arff`
   - `bibtex.arff`
5. Place all three in this `data/` directory.

## Smoke test after download

```bash
python -c "from preorder4mlc.datasets4experiments import Datasets4Experiments; \
  d = Datasets4Experiments('./data/', [{'dataset_name':'CAL500.arff','n_labels_set':174}]); \
  d.load_datasets(); X,Y,name = d.datasets[0]; print(name, X.shape, Y.shape, Y.sum(0).min(), Y.sum(0).max())"
```

Expected: shapes match the table above; min/max label frequencies > 0.

## Scalability flags

For K > 30 the following baselines are auto-skipped (no point running):
- `lp` — LabelPowerset blows up combinatorially
- `clr` — pairwise expansion is K(K-1)/2 (already costly at K=14, infeasible at K=100+)

Set `PREORDER_MAX_K_LP=999` to override.
