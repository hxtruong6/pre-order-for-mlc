# Reproducing the paper results

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,viz]"
```

## Full sweep

```bash
bash run.sh          # trains + evaluates all datasets into results/run-<date>/
```
This runs the preorder pipeline and the CLR / BR / CC / ECC baselines for each dataset
in `run.sh`'s `DATASETS` list, writing per-fold CSVs and per-dataset logs.

## Aggregate and analyze

```bash
python -m preorder4mlc.utils.summarize_metrics    # build summary tables
python -m preorder4mlc.utils.statistical_tests    # significance tests
python -m preorder4mlc.utils.plot_figures         # render figures
```

## Fast capsule (single dataset)

```bash
bash scripts/reproduce_capsule.sh results/         # CHD-49 only, CPU, quick
```
See also the project-level `REPRODUCE.md`.
