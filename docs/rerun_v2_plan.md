# Rerun v2 Plan: Add Probability Scores and Ranking Metrics

**Author:** planning agent
**Date:** 2026-05-14
**Status:** proposed (no code changes yet)
**Target run id:** `results/20260514_v2/`
**Preserves:** `results/20250624/` and `results/final_0624_summary/` (untouched)

## 0. Background

The 2025-06-24 run produced pickles that store binary 0/1 predictions only. The pipeline currently emits:

- `Y_predicted` (list of binary vectors, shape `[n_test, n_labels]`)
- `Y_BOPOs` (BOPOs preference-order vector encoding)
- `indices_vector`, `partial_abstention`

But it does NOT store per-label probability scores. Standard MLC ranking metrics that reviewers expect (`ranking_loss`, `one_error`, `coverage`, `average_precision`, `auc_macro`, `auc_micro`) all require a real-valued score per label, not a thresholded 0/1 prediction. They cannot be reconstructed retroactively from the existing pickles.

This plan adds a `Y_proba` field to every saved record and registers six new ranking metrics in the evaluation pipeline, writing all output to a new directory so the existing run survives intact.

## 1. Pickle schema extension

### 1.1 Universal new field

Add exactly one new field per record:

```python
"Y_proba": np.ndarray of shape (n_test, n_labels), dtype float64, in [0, 1]
```

Semantics: marginal per-label posterior probability `P(y_k = 1 | x)`. This is the canonical input to every sklearn ranking metric (`label_ranking_loss`, `label_ranking_average_precision_score`, `coverage_error`, `roc_auc_score`). All algorithms can supply this — they differ only in how they compute it.

Add at exactly one site:
- `training_orchestrator.py::TrainingOrchestrator._update_results` (line 352-366): add `"Y_proba": Y_proba.tolist() if Y_proba is not None else None` to the `data` dict. Pass `Y_proba` through a new keyword argument with default `None` so the call sites that have no proba (e.g. legacy CLR) degrade gracefully.

Add to:
- `extra_baselines.py::run` inside the per-fold record (line 184-198): same field.
- `utils/results_manager.py::ResultProcessor.process_predictions` (line 27-36): add `"Y_proba"` to the list of columns to deserialize. Use the same `convert_string_to_array` logic; guard with `if col in df.columns` so the new evaluator can also load OLD pickles without crashing.

Storage cost: for the largest dataset (yeast, 14 labels, ~2400 instances, 5 folds x 2 repeats x 4 noises x 8 algo-configs) the proba field adds roughly `14 * 2400 * 8 bytes * 80 records ~= 21 MB` per BOPOs pickle. Existing BOPOs pickles are ~5 MB, so expect ~4-5x growth. Acceptable.

### 1.2 BOPOs: deriving a marginal score from pairwise probabilities

The BOPOs pipeline trains `K*(K-1)/2` pairwise classifiers (`PRE_ORDER`: 4 classes per pair; `PARTIAL_ORDER`: 3 classes per pair). It does NOT natively produce per-label marginals. We need to synthesize them.

**Recommended derivation** (a soft pairwise-voting scheme analogous to CLR's `voting_scores`, but using probabilities instead of argmax):

For each instance `n` and each label pair `(i, j)`, the pairwise classifier outputs probabilities over four pre-order outcomes:
- class 0: `y_i = 1, y_j = 0`
- class 1: `y_i = 0, y_j = 1`
- class 2: `y_i = 0, y_j = 0`
- class 3: `y_i = 1, y_j = 1`

Marginal `P(y_i = 1 | x)` over pair `(i, j)` is `p_ij[0] + p_ij[3]`. Marginal `P(y_j = 1 | x)` is `p_ij[1] + p_ij[3]`. Average across all `K-1` pairs that mention label `i`:

```python
Y_proba[n, i] = (1 / (K - 1)) * sum_{j != i} P(y_i = 1 | pair_ij, x_n)
```

For `PARTIAL_ORDER` (3 classes: `i > j`, `j > i`, `i ~ j`), the marginals are not directly identifiable from pairwise classes alone (the equivalence class lumps `(1,1)` with `(0,0)`). Use the same averaging but over the available pairwise binary information from the BR-equivalent probabilities already computed in `predict_proba_BR` style. Concretely, for partial order the cleanest approach is to **train an auxiliary BR head** during BOPOs `fit()` and use its `predict_proba` as `Y_proba`. This is a one-line addition: re-use `self.base_classifier.binary_relevance_classifer(X, Y)` (already implemented in `base_classifiers.py:261`), call `clf.predict_proba(X_test)[:, 1]` per label.

**Decision**: use the **auxiliary BR-head approach for BOTH pre- and partial-order** for consistency and reviewer-friendliness. Rationale:
1. Same definition across BOPOs variants: avoids reviewer questions about "why two different scoring rules".
2. Decoupled from the search heuristic: ranking metric measures the *underlying* pairwise classifier's ability to rank labels, independent of the ILP.
3. Cheap: each BR head fit time is dominated by per-pair fit (already in pairwise), but here we fit K classifiers, one per label, against the full Y column. For RF with 100 trees, this is single-digit seconds per dataset.
4. Compatible with sklearn ranking metrics directly.

**Implementation location** (BOPOs):
- `inference_models.py::PredictBOPOs.fit` (line 520): after pairwise fit, also fit a BR head:
  ```python
  self.br_head_for_proba = MultiOutputClassifier(self.base_classifier.get_classifier())
  self.br_head_for_proba.fit(X, Y)
  ```
- New method `inference_models.py::PredictBOPOs.predict_marginal_proba(X)`: return `np.column_stack([clf.estimators_[k].predict_proba(X)[:, 1] for k in range(K)])`. Guard for single-class labels (return constant array if `len(classes_) == 1`).
- Wire it in `training_orchestrator.py::_process_bopos` (line 131): right after the existing `probabilistic_predictions = predict_BOPOs.predict_proba(...)` call (line 157), add:
  ```python
  Y_proba = predict_BOPOs.predict_marginal_proba(X_test)
  ```
  Pass `Y_proba` into `_update_results` for every height/metric variant in the inner loop (line 187-203). The same `Y_proba` is shared across all 4 `(target_metric, height)` combinations because it is a property of the pairwise classifier output, not the ILP.

### 1.3 Baselines: native predict_proba

- **BR (`MultiOutputClassifier`)**: `self.br_classifier.predict_proba(X)` returns `list[(n, 2)]`. Collect column 1: `Y_proba = np.column_stack([p[:, 1] for p in proba_list])`. Add to `PredictBOPOs.predict_BR` (line 569) as second return; update `_process_br` line 269 to capture and pass it.

- **CC (`ClassifierChain`)**: `self.cc_classifier.predict_proba(X)` returns `(n, K)` already — direct use. Update `PredictBOPOs.predict_CC` (line 608) and `_process_cc` line 310 analogously.

- **CLR**: re-use the BR head trick — fit an auxiliary BR head in `fit_CLR` and call `predict_proba` on it. Alternatively, derive from existing pairwise `voting_scores`: `Y_proba = voting_scores.T / (n_labels - 1)` already produces a [0,1] score in `predict_CLR` (line 476-502). Normalize and treat as proba. Decision: **normalize voting_scores** (no extra training): in `predict_CLR` (`inference_models.py` line 504+), before the loop that builds `predicted_Y`, compute `Y_proba = (voting_scores.T) / max(1, n_labels - 1)`. Return it as a third tuple element. This is a true marginal-like score.

- **MLkNN**: `clf.predict_proba(X_test)` returns sparse matrix of shape `(n, K)`. Densify: `Y_proba = np.asarray(clf.predict_proba(X_test).todense())`. Add in `extra_baselines.py::train_one` for the `mlknn` branch (line 129-132): change the function signature to return `(Y_pred, Y_proba)` and propagate.

- **ECC**: ten chains, each can produce `predict_proba` (sparse). Aggregate by mean across chains:
  ```python
  proba_sum = np.zeros((n_test, n_labels))
  for k in range(n_ensembles):
      ...
      proba = chain.predict_proba(X_test)         # may be sparse (n, n_labels)
      if sparse.issparse(proba): proba = proba.toarray()
      inv_perm = np.argsort(perm)
      proba_sum += proba[:, inv_perm]
  Y_proba = proba_sum / n_ensembles
  ```
  Edit `extra_baselines.py::_ecc_predict` (line 85-120): change to return `(Y_pred, Y_proba)`.

- **LP (`LabelPowerset`)**: `LabelPowerset.predict_proba(X)` returns the per-meta-class probability matrix `(n, n_meta_classes)`. To get per-label marginals, sum over those meta-classes whose binary expansion has bit `k` set. skmultilearn exposes `clf.unique_combinations_` (a dict from binary tuple to meta-class id). Implementation:
  ```python
  proba_meta = clf.predict_proba(X_test)
  if sparse.issparse(proba_meta): proba_meta = proba_meta.toarray()
  # Build (n_meta, n_labels) bit matrix
  combos = sorted(clf.unique_combinations_.items(), key=lambda kv: kv[1])
  bit_mat = np.array([list(map(int, combo.split(','))) for combo, _ in combos])
  Y_proba = proba_meta @ bit_mat   # (n, n_labels)
  ```
  Note: skmultilearn versions differ on whether `unique_combinations_` keys are tuples, strings, or `BitArray`. Inspect the installed version (`skmultilearn==0.2.x` typically uses string keys "0,1,0,1,..."). Add a small helper `_lp_marginal_proba(clf, X)` in `extra_baselines.py` that handles both formats. If exposing internals proves brittle, fall back to `Y_proba = Y_pred.astype(float)` and document the limitation — but try the principled approach first.

## 2. New ranking metric implementations

### 2.1 New enum members

Add to `evaluation_metric.py::EvaluationMetricName` (after line 28, in the example-based block):

```python
RANKING_LOSS = "ranking_loss"
ONE_ERROR = "one_error"
COVERAGE = "coverage"
LR_AP = "lr_ap"             # label-ranking average precision
AUC_MACRO = "auc_macro"
AUC_MICRO = "auc_micro"
```

Direction: `RANKING_LOSS`, `ONE_ERROR`, `COVERAGE` are **lower-is-better**; `LR_AP`, `AUC_MACRO`, `AUC_MICRO` are **higher-is-better**.

### 2.2 New methods on EvaluationMetric

Add to `evaluation_metric.py::EvaluationMetric` (after the `example_recall` method, around line 188):

```python
from sklearn.metrics import (
    label_ranking_loss,
    label_ranking_average_precision_score,
    coverage_error,
    roc_auc_score,
)

def ranking_loss(self, Y_proba, true_Y) -> float:
    return float(label_ranking_loss(true_Y, Y_proba))

def one_error(self, Y_proba, true_Y) -> float:
    # Fraction of instances whose top-scored label is NOT a positive label.
    Y_proba = np.asarray(Y_proba)
    true_Y = np.asarray(true_Y)
    top = np.argmax(Y_proba, axis=1)
    n = true_Y.shape[0]
    miss = sum(1 for i in range(n) if true_Y[i, top[i]] == 0)
    return float(miss / n)

def coverage(self, Y_proba, true_Y) -> float:
    # sklearn returns 1-based avg rank; subtract 1 to match the classic MLC formulation.
    return float(coverage_error(true_Y, Y_proba) - 1.0)

def lr_ap(self, Y_proba, true_Y) -> float:
    return float(label_ranking_average_precision_score(true_Y, Y_proba))

def auc_macro(self, Y_proba, true_Y) -> float:
    # roc_auc_score requires at least one positive AND one negative per label for macro.
    try:
        return float(roc_auc_score(true_Y, Y_proba, average="macro"))
    except ValueError:
        # Filter out degenerate labels (all-0 or all-1 in this fold).
        true_Y = np.asarray(true_Y); Y_proba = np.asarray(Y_proba)
        keep = [k for k in range(true_Y.shape[1])
                if 0 < true_Y[:, k].sum() < true_Y.shape[0]]
        if not keep: return float("nan")
        return float(roc_auc_score(true_Y[:, keep], Y_proba[:, keep], average="macro"))

def auc_micro(self, Y_proba, true_Y) -> float:
    try:
        return float(roc_auc_score(true_Y, Y_proba, average="micro"))
    except ValueError:
        return float("nan")
```

Edge cases worth noting:
- `coverage_error` requires at least one positive label per instance (else raises). Drop empty-label rows or assign `nan` for those instances.
- `ranking_loss` is undefined when an instance has no positives or no negatives; sklearn returns 0 for those — acceptable.
- `one_error` is defined for any instance with at least one positive. For all-zero rows, define as 0 (no positive can be missed); document this choice.
- `auc_macro` and `auc_micro` can break on labels with constant truth in a small fold. Filtering as above is safer than emitting NaN that propagates.

### 2.3 Dispatcher: new prediction type or new branch

Two design options:

**Option A** (preferred — minimal code change): treat the new metrics as a new prediction type `PredictionType.SCORE_VECTOR`.

In `evaluation_test.py`:
- Add to `PredictionType` enum (line 31):
  ```python
  SCORE_VECTOR = "ScoreVector"
  ```
- Add to `EvaluationConfig.EVALUATION_METRICS` (line 61):
  ```python
  PredictionType.SCORE_VECTOR: [
      EvaluationMetricName.RANKING_LOSS,
      EvaluationMetricName.ONE_ERROR,
      EvaluationMetricName.COVERAGE,
      EvaluationMetricName.LR_AP,
      EvaluationMetricName.AUC_MACRO,
      EvaluationMetricName.AUC_MICRO,
  ],
  ```
- New dispatcher method `_evaluate_score_vector(self, metric_name, Y_proba, true_Y)` mirroring `_evaluate_binary_vector` (after line 213).
- Branch in `evaluate_metric` (line 144) — add an `elif prediction_type == PredictionType.SCORE_VECTOR` branch.
- New evaluation loop `_evaluate_score_vector_results` mirroring `_evaluate_binary_vector_results` (line 579) — same shape, pulls `df2["Y_proba"].values[0]` instead of `df2["Y_predicted"].values[0]`. Skip the row if `Y_proba` column is missing or value is `None` (backward compat).
- In `evaluate_dataset` (line 409), after the binary-vector / preference-order calls, add a call to `_evaluate_score_vector_results` for both BOPOs and baselines.

**Option B**: extend `_evaluate_binary_vector` to also accept score input. Rejected — conflates two different prediction semantics in one dispatcher and makes the metric → input-type mapping implicit.

### 2.4 `evaluate_extra_baselines.py` changes

In `evaluate_extra_baselines.py`:
- Add `RANKING_METRICS` list (mirror of section 2.2 metrics, lines 25-40).
- Extend `_eval_one` (line 43): if metric in `{RANKING_LOSS, ONE_ERROR, ...}`, route to `em.ranking_loss(...)`, etc.
- In `evaluate_pickle` (line 76): loop over both `BINARY_METRICS` and `RANKING_METRICS`. When iterating the ranking branch, pull `row["Y_proba"]` (skip if absent), pass as `y_pred`.
- Add a row with `"Prediction_Type": "ScoreVector"` for each ranking metric.

## 3. New results directory layout

- Pickles: `results/20260514_v2/dataset_<name>_noisy_<rate>[_clr|_br|_cc|_mlknn|_ecc|_lp].pkl`
- Per-fold CSVs: `results/20260514_v2/evaluation_<Name>_noisy_<rate>_<algotype>.csv` (and `.xlsx`)
- Summary: `results/final_20260514_v2_summary/<DatasetName>_<PredictionType>_summary.csv` (and `.xlsx`)
  - `PredictionType ∈ {BinaryVector, PartialAbstention, ScoreVector}` (new third type)
- Stats: `results/final_20260514_v2_summary/stats/`

These mirror the 2025-06-24 layout exactly. The summarization scripts only need their input/output dir constants changed (see section 4).

## 4. Touch list (file:line + intent)

### 4.1 `inference_models.py`
- **L520-541** (`fit`): after pairwise fit, train auxiliary BR head for proba.
- **Around L621** (new method): add `predict_marginal_proba(self, X) -> np.ndarray`.
- **L569-581** (`predict_BR`): return `(Y_pred, None, Y_proba)` (third element). Internally `Y_proba = np.column_stack([p[:, 1] for p in self.br_classifier.predict_proba(X)])`.
- **L608-620** (`predict_CC`): `Y_proba = self.cc_classifier.predict_proba(X)`, return as third element.
- **L504-518** (`predict_CLR`): compute `Y_proba = voting_scores.T / max(1, n_labels - 1)`, return as third element.

### 4.2 `training_orchestrator.py`
- **L157**: after `probabilistic_predictions = predict_BOPOs.predict_proba(...)`, add `Y_proba_bopos = predict_BOPOs.predict_marginal_proba(X_test)`.
- **L190-203**: pass `Y_proba_bopos` into `_update_results`.
- **L228-244** (CLR), **L268-285** (BR), **L309-326** (CC): unpack the third return element from `predict_*` and pass to `_update_results`.
- **L331-368** (`_update_results`): add `Y_proba=None` parameter; store `"Y_proba": Y_proba.tolist() if Y_proba is not None else None` in the record dict.

### 4.3 `extra_baselines.py`
- **L77** (constants block): no change.
- **L85-120** (`_ecc_predict`): also aggregate `Y_proba` across chains via mean; return tuple `(Y_pred, Y_proba)`.
- **L123-147** (`train_one`): rewrite to return `(Y_pred, Y_proba)`. Densify sparse output from MLkNN and LP. For LP use the meta-class to bit-matrix sum described in 1.3.
- **L150-207** (`run`): unpack `Y_pred, Y_proba = train_one(...)`. Record dict (L184-198) gains `"Y_proba": Y_proba.tolist()`.

### 4.4 `evaluation_metric.py`
- **L1-9**: extend sklearn import line to include `label_ranking_loss`, `label_ranking_average_precision_score`, `coverage_error`, `roc_auc_score`.
- **L11-48**: add the 6 new enum values.
- **After L188** (`example_recall`): add the 6 new methods (`ranking_loss`, `one_error`, `coverage`, `lr_ap`, `auc_macro`, `auc_micro`) as in 2.2.

### 4.5 `evaluation_test.py`
- **L31**: add `SCORE_VECTOR = "ScoreVector"` to `PredictionType`.
- **L61-97**: register `PredictionType.SCORE_VECTOR` block in `EvaluationConfig.EVALUATION_METRICS`.
- **After L213**: add `_evaluate_score_vector(self, metric, Y_proba, true_Y)` returning the corresponding `EvaluationMetric` method's value.
- **L144-167**: add `elif prediction_type == PredictionType.SCORE_VECTOR: return self._evaluate_score_vector(...)` branch.
- **L336-407** (`_evaluate_fold`): in the data extraction block, also pull `df2["Y_proba"].values[0]` (guard with `if "Y_proba" in df2.columns and df2["Y_proba"].values[0] is not None`). Pass as a new optional parameter to `evaluate_metric` for score-vector metrics.
- **After L605** (mirror of `_evaluate_binary_vector_results`): add `_evaluate_score_vector_results`.
- **L409-474** (`evaluate_dataset`): for both BOPOs and baseline branches, add a call to `_evaluate_score_vector_results`.

### 4.6 `evaluate_extra_baselines.py`
- **L25-40**: add a second list `RANKING_METRICS = [RANKING_LOSS, ONE_ERROR, COVERAGE, LR_AP, AUC_MACRO, AUC_MICRO]`.
- **L43-73** (`_eval_one`): dispatch the 6 new metrics. Accept either `y_pred` (binary) or `y_proba` based on metric category.
- **L76-109** (`evaluate_pickle`): add a second loop over `RANKING_METRICS` that consumes `row["Y_proba"]`. Emit rows with `"Prediction_Type": "ScoreVector"`. Skip silently if `Y_proba` missing (backward compat).

### 4.7 `utils/results_manager.py`
- **L29 (`process_predictions`)**: extend the `for col in ["Y_test", "Y_predicted", "Y_BOPOs"]` list to include `"Y_proba"` when present. Use the existing `convert_string_to_array` helper.

### 4.8 `utils/summarize_metrics.py`
- **L7-9** (`RESULTS_DIR`, `OUTPUT_DIR`): parametrize. Two acceptable approaches:
  - Quick fix: change defaults to `"results/20260514_v2"` and `"results/final_20260514_v2_summary"`.
  - Better fix: add `argparse` with `--results_dir` and `--output_dir` defaulting to the above. Recommended.
- **L22-23** (`BOPOS_PREDICTION_TYPES`): extend to `["BinaryVector", "PartialAbstention", "ScoreVector"]` so the new prediction type flows into summary tables.
- **Note**: the "Water-quality" regex case (the file `Water-quality_BinaryVector_summary.csv` collides with dataset name parsing because of the hyphen) is already fixed in the regex on L26 (`[A-Za-z0-9_\-]+`) — verified, no change needed.

### 4.9 `utils/statistical_tests.py`
- **L39-67** (`HIGHER_IS_BETTER`, `LOWER_IS_BETTER`): add the six new metric names to the right sets:
  - HIGHER: `lr_ap`, `auc_macro`, `auc_micro`
  - LOWER: `ranking_loss`, `one_error`, `coverage`
- **L109** (`SUMMARY_RE`): extend the alternation: `(BinaryVector|PartialAbstention|ScoreVector)` so score-vector summaries are picked up.
- **L411-412** (CLI defaults): change defaults to the new `final_20260514_v2_summary` paths.

### 4.10 `config.py`
- **L88-93** (`get_training_config`): decision needed on `total_repeat_times`. See section 6 for cost/benefit analysis. Recommended bump from 2 to 5 only if the slurm queue can host parallel jobs. Otherwise keep 2 (matches old run for direct delta).

### 4.11 New slurm scripts (`job_scripts/`)
Create three new files (model after `submit_yeast.sh` and `submit_extra_baseline.sh`):

- **`job_scripts/submit_bopos_v2.sh`**: parameterized by `$DATASET`. Sets `RESULTS_DIR=results/20260514_v2`, runs `python main.py --dataset $DATASET --results_dir $RESULTS_DIR` then `python evaluation_test.py --dataset $DATASET --results_dir $RESULTS_DIR`.
- **`job_scripts/submit_extra_baseline_v2.sh`**: same as `submit_extra_baseline.sh` but `RESULTS_DIR=results/20260514_v2`.
- **`job_scripts/submit_rerun_all.sh`** (orchestrator, not a slurm script — a plain bash that submits 9 + 27 sbatch jobs):
  ```bash
  #!/bin/bash
  DATASETS=(chd_49 emotions scene viruspseaac yeast water_quality humanpseaac gpositivepseaac plantpseaac)
  for d in "${DATASETS[@]}"; do
      sbatch -p DEF -J "${d}_v2" job_scripts/submit_bopos_v2.sh "$d"
      for algo in mlknn ecc lp; do
          sbatch -p DEF -J "${d}_${algo}_v2" job_scripts/submit_extra_baseline_v2.sh "$d" "$algo"
      done
  done
  ```

### 4.12 Memo: do NOT touch
- `main.py` legacy `process_dataset` / `training` functions — they are dead code per `CLAUDE.md`.
- `searching_algorithms.py` — ILP search is unchanged; we are only adding marginals, not changing the BOPOs prediction pipeline.
- `base_classifiers.py` — `binary_relevance_classifer` (line 261) is already implemented and unused; we can either reuse it inside `PredictBOPOs.fit` or directly use sklearn's `MultiOutputClassifier` (already imported in `inference_models.py` L5). Use the latter for consistency with `fit_BR`.
- `datasets4experiments.py` — splits, noise, RANDOM_STATE all unchanged.

## 5. Backward compatibility / migration

### 5.1 Old pickles
The old run's pickles (under `results/20250624/`) have **no** `Y_proba` key. The new evaluator must not crash on them.

- `utils/results_manager.py::process_predictions`: guard with `if col in df.columns` (already does for most columns).
- `evaluation_test.py::_evaluate_fold`: when extracting `Y_proba`, default to `None` if missing. If `None`, skip score-vector metrics for that record (emit no row, or emit a row with `NaN` mean — prefer skip).
- `evaluate_extra_baselines.py::evaluate_pickle`: same guard; the ranking-metric loop is a no-op when no `Y_proba` is present.

This means the new code can also re-evaluate the old pickles for the legacy (binary) metrics — useful for regression testing (section 7).

### 5.2 Old summarizer
`utils/summarize_metrics.py` will produce three prediction-type-suffixed summaries per dataset (`BinaryVector`, `PartialAbstention`, `ScoreVector`). The old run only has the first two, but the script handles missing files via `defaultdict` and only emits summaries for prediction types that exist in the data. No incompatibility.

### 5.3 Old statistical_tests
`utils/statistical_tests.py` with the extended regex picks up `ScoreVector` summary CSVs when present, silently skips them when absent. No regression on old behaviour.

## 6. Compute budget estimate

### 6.1 Baseline timings (2 repeats x 5 folds, from `logs/20250624/*.log`)

| Dataset | Time (s) | Time (h) |
|---|---|---|
| chd_49 | 256 | 0.07 |
| emotions | 361 | 0.10 |
| scene | 2302 | 0.64 |
| water_quality | 3526 | 0.98 |
| humanpseaac | 8693 | 2.41 |
| viruspseaac | 236 | 0.07 |
| plantpseaac | 1919 | 0.53 |
| gpositivepseaac | 270 | 0.07 |
| yeast | ~3600 | 1.00 |
| **Total (serial, 2 reps)** | ~21163 | ~5.88 |

### 6.2 Cost of new BR-head and proba calls
- BR head fit: `K * fit_time_per_RF`. For `K=14`, RF with 100 trees, on ~2000 instances and 50-300 features: roughly 0.5-5 s per label, 7-70 s total. **Negligible** vs the pairwise fit cost (already `K(K-1)/2 = 91` classifiers for `K=14`).
- `predict_proba` calls: cheap; dominated by RF tree traversal already amortized in the existing predict step.

Expect <10% wall-time overhead in the new run. Practically: humanpseaac stays around 2.5-2.7 h at 2 repeats.

### 6.3 With 5 repeats

- Linear scaling. Total serial: ~14.7 h.
- Parallelized (one sbatch per dataset, queue capacity permitting): max single-dataset cost = humanpseaac ~6 h.

### 6.4 With 10 repeats
- Total serial: ~29.4 h. Single-dataset humanpseaac: ~12 h.

### 6.5 Recommendation

**Recommend 5 repeats** if cluster availability allows the long humanpseaac job (~6 h on DEF partition). Rationale:
- 5x10 = 50 evaluation points per (dataset, noise, algo, metric) cell — gives meaningful Friedman test power and tighter error bars for paper figures.
- The bottleneck is humanpseaac, not the total — and 6h is comfortable in any HPC queue.
- 2 repeats was already enough for the v1 run to show clear ranking-based separation; 5 buys reviewer-defensible confidence intervals.

**Fallback to 2 repeats** if the queue is constrained, or if we want a pure A/B comparison against the v1 run with all other factors identical. This is a reasonable conservative choice.

**10 repeats** is overkill given the existing dataset-level variance dominates fold-level variance; skip.

### 6.6 Baselines compute
The 9 datasets x 3 algos (`mlknn`, `ecc`, `lp`) = 27 baseline jobs. v1 baseline jobs all finished in <30 min wall (per the slurm log file dates). Even at 5 repeats this remains well under 1.5 h per job, parallel.

## 7. Validation plan

### 7.1 Sanity check on a single pickle

After phase B's dry run on `viruspseaac`:

```python
import pickle, numpy as np
with open("results/20260514_v2/dataset_viruspseaac_noisy_0.0.pkl", "rb") as f:
    records = pickle.load(f)
r = records[0]
assert "Y_proba" in r, "Y_proba missing"
yp = np.asarray(r["Y_proba"])
yt = np.asarray(r["Y_test"])
assert yp.shape == yt.shape, f"shape mismatch {yp.shape} vs {yt.shape}"
assert (yp >= 0).all() and (yp <= 1).all(), f"out of [0,1]: [{yp.min()}, {yp.max()}]"
print("OK", yp.shape, yp.mean(), yp.std())
```

Repeat for one BR, CC, CLR, MLkNN, ECC, LP pickle.

### 7.2 Ranking-metric sanity

For each (dataset, noise=0.0) record:
- `ranking_loss in [0, 1]`
- `auc_macro > 0.5` (trained model better than random)
- `lr_ap > random baseline` (random AP for binary multilabel ~= average density)
- `one_error <= 1`
- `coverage <= n_labels - 1`

Add a small assertion script (`utils/validate_rerun.py`, optional) that loops over all new pickles and prints failures.

### 7.3 Regression on common metrics

For binary metrics (`hamming_accuracy`, `f1`, `subset0_1`, `jaccard`, etc.), the new run should agree with the v1 run **within fold-level variance** (since splits are identical via `RANDOM_STATE = 6`, and we only ADDED outputs). Compute per-dataset, per-metric percentage delta:

```python
old = pd.read_csv("results/final_0624_summary/VirusPseAAC_BinaryVector_summary.csv")
new = pd.read_csv("results/final_20260514_v2_summary/VirusPseAAC_BinaryVector_summary.csv")
# Parse mean from "0.7234±0.0123" and compare
```

Expect mean deltas < 0.005 for all algorithms × metrics × noise levels at fixed `total_repeat_times=2`. Deltas at 5 repeats may legitimately differ but should not flip rankings.

If a delta exceeds 0.01 unexpectedly, investigate — most likely cause is accidental classifier-state pollution by the new BR head.

### 7.4 Stats-pipeline smoke test

```bash
python utils/statistical_tests.py \
    --results_dir results/final_20260514_v2_summary \
    --output_dir results/final_20260514_v2_summary/stats
```

Expect `significance_summary.csv` to contain rows for `ptype ∈ {BinaryVector, PartialAbstention, ScoreVector}`, with 6 new metrics under `ScoreVector`. Spot-check CD diagrams for the 4 noise levels x 6 ranking metrics = 24 new PDFs in `stats/`.

## 8. Execution checklist

### Phase A — Code changes (no execution)
- A1. Edit `inference_models.py` (4.1).
- A2. Edit `training_orchestrator.py` (4.2).
- A3. Edit `extra_baselines.py` (4.3).
- A4. Edit `evaluation_metric.py` (4.4).
- A5. Edit `evaluation_test.py` (4.5).
- A6. Edit `evaluate_extra_baselines.py` (4.6).
- A7. Edit `utils/results_manager.py` (4.7).
- A8. Edit `utils/summarize_metrics.py` and `utils/statistical_tests.py` (4.8, 4.9).
- A9. Optionally edit `config.py` to set `total_repeat_times=5` (4.10).
- A10. Create three new slurm scripts under `job_scripts/` (4.11).
- A11. Quick local import sanity: `python -c "import inference_models, evaluation_metric, evaluation_test, extra_baselines, evaluate_extra_baselines"`.

### Phase B — Dry run on viruspseaac (~4 min at 2 reps)
- B1. `python main.py --dataset viruspseaac --results_dir results/20260514_v2_dryrun`
- B2. `python evaluation_test.py --dataset viruspseaac --results_dir results/20260514_v2_dryrun`
- B3. For each of `mlknn ecc lp`: `python extra_baselines.py --dataset viruspseaac --algorithm $a --results_dir results/20260514_v2_dryrun`
- B4. For each of `mlknn ecc lp`: `python evaluate_extra_baselines.py --dataset viruspseaac --algorithm $a --results_dir results/20260514_v2_dryrun`
- B5. Run the 7.1 sanity script on the 7 produced pickles.
- B6. Inspect `evaluation_VirusPseAAC_noisy_0.0_bopos.csv` — confirm `Prediction_Type == "ScoreVector"` rows present with the 6 new metrics.
- B7. Delete `results/20260514_v2_dryrun/` once validated (or keep for traceability).

### Phase C — Submit full slurm batch
- C1. `bash job_scripts/submit_rerun_all.sh` (orchestrator from 4.11) — submits 9 BOPOs + 27 baseline jobs.
- C2. Monitor: `squeue -u $USER`, `tail -f logs/slurm-*_v2-*.log`.
- C3. Expected wall: at 2 reps and parallel queue, ~2.5h (gated by humanpseaac); at 5 reps, ~6h.

### Phase D — Summary + stats
- D1. `python utils/summarize_metrics.py --results_dir results/20260514_v2 --output_dir results/final_20260514_v2_summary`
- D2. `python utils/statistical_tests.py --results_dir results/final_20260514_v2_summary --output_dir results/final_20260514_v2_summary/stats`

### Phase E — Validation + delta vs v1
- E1. Run the regression script (7.3) across all 9 datasets.
- E2. Eyeball CD diagrams for the 6 new ranking metrics at noise=0.0 (`stats/cd_ScoreVector_*_0.0.pdf`) — confirm BOPOs variants rank meaningfully relative to baselines.
- E3. Tabulate `lr_ap` and `auc_macro` per dataset, per algorithm, per noise into the paper's main results table.
- E4. Archive the v2 run by writing a one-pager note in `docs/` describing the schema, the BR-head proba derivation, and any caveats (e.g. LP's meta-class summation approach).

## 9. Open design questions

1. **BOPOs proba derivation**: this plan chooses the auxiliary BR-head approach. An alternative is to derive marginals from the existing pre-order pairwise classifier output (`p_ij[0] + p_ij[3]` averaged over pairs). The BR-head is cleaner and reviewer-defensible, but it slightly decouples the score from the actual BOPOs inference. Decide before phase A.
2. **CLR proba**: `voting_scores / (n_labels - 1)` is monotone-aligned with the existing CLR prediction rule but is not a calibrated probability. For paper writing, consider stating it as "vote-fraction score" rather than "probability".
3. **LP proba**: depends on the installed `skmultilearn` version exposing `unique_combinations_` or an equivalent. If unavailable, fall back to `Y_pred.astype(float)` and document — but the ranking metrics will then degenerate for LP. Verify the version (`pip show skmultilearn`) during phase A.
4. **One-error for empty true sets**: defined here as 0 (cannot miss a positive that does not exist). Some papers exclude such instances entirely. State the choice in the paper.
5. **Repeats=5 vs 2**: pure budget vs power trade-off. Recommend 5; keep 2 as the safe fallback.
6. **AUC degenerate-label filtering**: this plan silently drops labels with all-0 or all-1 in a given fold, then computes macro AUC over the remaining. This affects only edge cases (e.g. yeast's rarest label at high noise). Document if the filter triggers on any (dataset, noise) cell.

## 10. Critical files for implementation

### Critical Files for Implementation
- /home/s2320437/WORK/preorder4MLC/inference_models.py
- /home/s2320437/WORK/preorder4MLC/training_orchestrator.py
- /home/s2320437/WORK/preorder4MLC/extra_baselines.py
- /home/s2320437/WORK/preorder4MLC/evaluation_metric.py
- /home/s2320437/WORK/preorder4MLC/evaluation_test.py
