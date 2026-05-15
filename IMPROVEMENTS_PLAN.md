# Algorithm Improvements Plan (dev branch)

Branch: `dev` (force-reset from `main`, history rewritten).
Goal: improve PA/PR algorithm beyond baselines (ECC, LP, ML-kNN) — focus on technique, not paper writing.

## What we are NOT doing
- Not editing paper.tex
- Not rewriting tikz tables
- Not chasing the AFRD paper-side discrepancy here

## Three workstreams

### #2 Cost-sensitive Hamming (existing theory → code)
**Why**: Paper appendix §G already derives cost-sensitive Hamming; baselines like ECC dominate on plain Hamming by predicting majority-0. With per-label cost ratio FP/FN, PA/PR can close the Hamming gap without giving up F1/AFRD/MFRD.

**Files**:
- `preorder4mlc/evaluation_metric.py` — add `cost_sensitive_hamming_accuracy(pred, true, cost_fp, cost_fn)`
- `preorder4mlc/inference_models.py` / `searching_algorithms.py` — cost-aware order→binary extraction

### #3 LightGBM base learner + calibration
**Why**: Current pairwise probability estimator Pij uses sklearn RF with default settings. LightGBM with isotonic calibration gives better-calibrated Pij → tighter orders → better predictions on all metrics.

**Files**:
- `preorder4mlc/estimator.py` — add `calibrate=True/False` option + isotonic wrapper
- `preorder4mlc/config.py` — `BASE_LEARNERS = [BaseLearnerName.LightGBM]`
- `scripts/train_extra_baselines.py` — swap RandomForestClassifier → LGBMClassifier so baselines also use the new base learner (fair comparison)

### #4 Large-K datasets
**Why**: All current datasets have K ≤ 14. Order structure matters more as K grows (more combinatorial label sets). Showing PA/PR scale to K=50–170 is the strongest argument that LP/BR/CC cannot match.

**Targets** (from COMETA, MULAN, MEKA repositories):
- **mediamill** (K=101, n=43k) — video labels, balanced-ish
- **CAL500** (K=174, n=502) — music tags, very imbalanced
- **bibtex** (K=159, n=7395) — text tags

**Files**:
- `preorder4mlc/datasets4experiments.py` — add to data_files list
- `preorder4mlc/config.py` — `DATASET_CONFIGS` entries
- Need `.arff` files in `./data/` (manual download or fetch script)

**Scalability concerns**:
- PA/PR: K(K-1)/2 pairwise classifiers → 8.5k–15k for K=130–174. Memory ≈ K² × per-classifier-size. Probably OK with LightGBM (fast) but slow.
- LP baseline: label powerset blows up at K>20. Skip.
- ILP search in `searching_algorithms.py`: complexity grows fast. May need timeout/heuristic fallback.

## Validation strategy

1. **Smoke test** (this session): one small dataset (CHD_49 K=6), one fold, LightGBM + calibration, verify training + prediction completes without error, sanity-check metrics are in plausible range.
2. **Single-dataset CV** (user runs offline): one dataset full 5×5 CV with LightGBM+calibration, compare against archived RF results. If LightGBM gives ≥ RF on F1/AFRD across noise levels → adopt as default.
3. **Large-K dry run** (user runs offline): mediamill 1 fold, time + memory check, decide whether to include in final benchmarks.
4. **Cost-sensitive Hamming**: parameter sweep on cost ratio ∈ {1, 2, 5, 10} on emotions/scene/water-quality. Plot F1 vs Hamming Pareto curve; show PA/PR dominates BR/CC across the curve.

## What this session delivers
- Code changes for all 3 workstreams
- Smoke test that confirms wiring works
- Concrete TODOs for the offline experiment runs (compute-heavy, can't fit in a turn)

## What requires offline compute (you run)
- Full 5×5 CV rerun on 8 existing datasets with LightGBM + calibration
- Large-K dataset downloads + first full runs
- Cost-sensitive Hamming sweep
