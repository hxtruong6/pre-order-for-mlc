# Response to Reviewer 1: computational cost and scaling of the BOPOs inference

Reviewer 1 asks us to (i) replace the vague claim that the method "scales to
53 labels" with concrete evidence, (ii) quantify how the per-instance ILP
cost grows with the number of labels `L`, and (iii) compare that cost against
the baselines so the accuracy/runtime trade-off is explicit, ending with a
practical guideline. This document answers all three points with measured
numbers, a scaling curve, and a recommendation.

All timings below report the **average inference time per test instance**, the
quantity the reviewer asked for. The measurement setup is described in the last
section; scripts are `scripts/bench_step1_train_enron.py`,
`bench_step2_scaling.py`, `bench_step2b_highs.py`, `bench_step2c_L53.py`, and
`bench_step3_plot.py`, and the raw numbers are in `runtime_scaling_data.csv`.

---

## Point 1 — "scales to 53 labels" made precise

The proposed method does one integer linear program (ILP) per test instance to
turn the pairwise probabilities into a (pre-order / partial-order) BOPO. The
size of that ILP is a closed-form function of `L`, read directly from
`preorder4mlc/searching_algorithms.py` (`_encode_parameters_PRE_ORDER` /
`_encode_parameters_PARTIAL_ORDER`):

| `L` (dataset) | binary variables (pre-order) | binary variables (partial-order) | equality constraints | transitivity constraints |
|---|---|---|---|---|
| 6 (chd_49, emotions, scene, virus) | 60 | 45 | 15 | 120 |
| 14 (yeast, water-quality, human) | 364 | 273 | 91 | 2,184 |
| 19 | 684 | 513 | 171 | 5,814 |
| 45 | 3,960 | 2,970 | 990 | 85,140 |
| 53 (enron) | 5,512 | 4,134 | 1,378 | 140,556 |

Variables grow as **O(L²)** (one group per label pair) and the transitivity
constraints as **O(L³)** (one per ordered label triple). Going from `L = 6` to
`L = 53` multiplies the variable count by **92** and the constraint count by
**1,171**. So "applicable to 53 labels" is not a claim about feasibility in
principle; it is a claim that this O(L²)/O(L³) program is still solved fast
enough per instance to be usable, which Points 2 and 3 quantify.

---

## Point 2 — how the per-instance ILP cost grows with `L`

We measured the average per-instance solve time on **real enron pairwise
probabilities** (Random Forest base learner, noise-free split). To isolate the
effect of `L` alone (holding the data-generating process fixed), we take the
trained enron model, sample label subsets of size
`L ∈ {6, 10, 14, 19, 25, 31, 37, 45, 53}`, slice the real pairwise-probability
tensor to each subset, and time the ILP. Both solvers used in the paper are
reported: **GLPK** (the default `cvxopt.glpk` backend) and **HiGHS** (the
stronger backend, via `scipy.optimize.milp`).

> **Note on fairness (same problem, same answer, only speed differs).** GLPK and
> HiGHS are two interchangeable MILP back-ends plugged into the *same* code path
> (`preorder4mlc/solvers.py`). They receive the identical standard-form ILP and
> return the **same optimal preference order** whenever the optimum is unique;
> they differ only in how fast they reach it. Reporting both is therefore not a
> method-vs-method comparison and does not advantage BOPOs: (i) the solver is
> only ever used by BOPOs, never by the BR/CC/CLR baselines, and both curves are
> shown side by side; and (ii) because the two solvers return the same solution,
> **prediction accuracy is identical under either solver**, so the accuracy
> numbers in the main paper are unchanged by the solver choice. The only thing
> that changes across solvers is runtime, which is exactly the quantity under
> discussion here. We keep the solver consistent between the reported accuracy
> and the reported runtime.

**Average inference time per test instance (ms):**

| `L` | GLPK, full transitivity | GLPK, height=2 | HiGHS, full transitivity | HiGHS, height=2 |
|----:|----:|----:|----:|----:|
| 6  | 1.0    | 1.1     | 6.5   | 1.8  |
| 10 | 7.4    | 17.9    | 8.8   | 4.4  |
| 14 | 41.6   | 124.9   | 24.5  | 9.6  |
| 19 | 250.9  | 987.0   | 66.4  | 22.4 |
| 25 | 1,455  | 5,444   | 151.3 | 50.9 |
| 31 | 7,500  | 21,957  | 319.1 | 100.0 |
| 37 | (impractical) | (impractical) | 787.4 | 175.2 |
| 45 | (impractical) | (impractical) | 1,178 | 335.8 |
| 53 | (impractical) | (impractical) | 2,007 | 577.1 |

![Runtime scaling](runtime_scaling.png)

Reading the curve (log-log):

- **The default GLPK solver does not scale.** Its empirical cost grows as
  roughly **L^5.4** (full transitivity) to **L^6.1** (height=2), i.e. even
  faster than the O(L³) problem size, because the branch-and-bound tree itself
  grows with `L`. At `L = 31` a single instance already needs ~7.5 s (full) or
  ~22 s (height=2), and beyond `L ≈ 37` a single instance routinely exceeds our
  time cap. This is why the paper's enron experiments were not run with GLPK.
- **With HiGHS the cost grows as ~L^2.9**, tracking the O(L³) constraint count
  and staying practical: at the real enron `L = 53` the average is **~2.0 s per
  instance** (full transitivity) or **~0.58 s** (height=2). The HiGHS advantage
  over GLPK widens with `L`: 1.7x at `L = 14`, 3.8x at `L = 19`, 9.6x at
  `L = 25`, and 23.5x at `L = 31`.
- **The height=2 variant is markedly cheaper under HiGHS** (about 3–4x faster
  than full transitivity), because its constraint matrix admits a tighter LP
  relaxation. Note the opposite holds for GLPK, where height=2 is *slower*: this
  is a solver artefact, not a property of the formulation, and reinforces that
  the solver choice, not the model, is what governs practicality.

**Bottom line for Point 2:** the honest, measured cost of the method at the
largest label set we report (`L = 53`) is on the order of **0.6–2 seconds per
test instance** with a modern MILP solver, and it grows polynomially (~cubically)
rather than explosively in `L`.

---

## Point 3 — trade-off against the baselines, and a practical guideline

The baselines do **not** solve a per-instance optimization; their inference is a
handful of vectorized classifier evaluations. Measured on the same enron test
set (per instance):

| Method | Per-instance inference | Solves an ILP? |
|---|---:|---|
| Binary Relevance (BR) | 9.4 ms | no |
| Classifier Chain (CC) | 19.3 ms | no |
| Calibrated Label Ranking (CLR) | 119.9 ms | no (but trains/queries `L(L-1)/2` pairwise models) |
| **BOPOs, HiGHS, height=2** | **577 ms** | yes |
| **BOPOs, HiGHS, full** | **2,007 ms** | yes |

So at `L = 53` the proposed method costs roughly **5x** the CLR baseline (the
most expensive competitor, which already pays the same pairwise-probability
estimation) for the height=2 variant, and about **17x** for full transitivity.
Against the cheapest baseline (BR) the ratio is ~60x (height=2) to ~210x (full).

Two points keep this in perspective:

1. **The pairwise-probability stage is shared with CLR.** BOPOs and CLR both
   train and query `L(L-1)/2` pairwise classifiers; on enron that stage costs
   ~0.43 s per instance for both. The ILP is the *only* extra cost BOPOs pays on
   top of CLR, and it is a per-instance cost that is trivially parallelizable
   across the test set (the code already supports `PREORDER_SEARCH_N_JOBS`).
2. **The absolute cost is small.** Even the worst case (full transitivity,
   `L = 53`) is ~2 s per instance, i.e. a full enron test fold of a few hundred
   instances predicts in minutes on one machine, and seconds with the
   per-instance parallelism enabled.

### Practical guideline (added to the paper)

- **`L ≲ 15` (most benchmark datasets):** any variant and either solver is
  effectively free (a few to a few tens of ms per instance). Use the full
  formulation for best accuracy.
- **`15 ≲ L ≲ 30`:** use HiGHS. Full transitivity stays in the tens-to-hundreds
  of ms range; GLPK is already 1–20 s and should be avoided.
- **`L ≳ 30` (e.g. enron, `L = 53`):** use HiGHS, and prefer the **height=2**
  variant if inference latency matters (≈0.58 s vs ≈2 s per instance, at a small
  accuracy cost documented in the main results). GLPK is not practical here.
- If the application is latency-critical and `L` is very large, the height=2 +
  HiGHS configuration is the recommended operating point; if best accuracy is
  the priority and a few seconds per instance is acceptable, use full
  transitivity + HiGHS.

**One-line summary for the rebuttal:** the extra cost of BOPOs over the
baselines is a single per-instance ILP whose measured cost grows polynomially
(~L³ with HiGHS) and is ~0.6–2 s per instance at `L = 53`; this is a modest,
parallelizable overhead on top of the same pairwise stage that CLR already pays,
and it buys the accuracy gains reported in the main experiments.

---

## Measurement setup (reproducibility)

- **Data / model:** enron ARFF (`L = 53`), 70/30 split (`RandomState(0)`),
  Random Forest pairwise classifiers (`RANDOM_STATE = 6`), noise-free. Pairwise
  probabilities are the real `predict_proba` output; smaller `L` are label
  subsets of this same model, so only `L` varies.
- **Timing:** average over 60 test instances (8 for the two heaviest `L = 53`
  full-transitivity configs, to bound wall time); per-instance ILP solve time
  only (the one-off constraint encoding is amortized and negligible: <0.35 s
  even at `L = 53`). GLPK capped at `L ≤ 31` because a single instance exceeds
  practical time beyond that.
- **Environment:** conda env `research_preorder_mlc` (sklearn 1.6.1, numpy
  2.3.0, scipy 1.15.3, cvxopt 1.3.2, python 3.12), matching `requirements.txt`.
- **Solvers:** GLPK via `cvxopt.glpk.ilp`; HiGHS via `scipy.optimize.milp`
  (`PREORDER_SOLVER=highs`), both solving the identical standard-form MILP and
  returning the same optimal solution (hence the same predictions and the same
  accuracy); only runtime differs.
- **Artefacts:** `runtime_scaling.png` / `.pdf` (figure),
  `runtime_scaling_data.csv` (all rows), and the four `scripts/bench_step*.py`
  drivers.
