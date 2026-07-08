# Runtime-scaling TikZ figures (reviewer 1)

Four copy-paste `pgfplots` figures for the per-instance ILP runtime scaling,
one per (preference order, target metric):

| file | preference order | target metric |
|---|---|---|
| `runtime_preorder_hamming.tex` | pre-order | Hamming accuracy |
| `runtime_preorder_subset.tex`  | pre-order | Subset (0/1 exact-match) |
| `runtime_partial_hamming.tex`  | partial order | Hamming accuracy |
| `runtime_partial_subset.tex`   | partial order | Subset (0/1 exact-match) |

Each figure shows, on log-log axes (x = number of labels `K`, y = seconds):

- **4 main curves**: GLPK / HiGHS x full-transitivity / height=2, measured on
  the real enron model (mean over 300 test instances, `K` varied by label
  subsets of the same trained model). GLPK's directly measured `K=53` point is
  included where it exists (both height-2 curves; partial-order full); the
  **pre-order full** GLPK curve stops at `K=31` because a single `K=53`
  instance did not finish within a 48-hour wall-clock limit.
- **1 dashed trend line per curve**: a least-squares power-law fit `K^x`; the
  fitted exponent is shown in the legend. HiGHS trends are fitted over and
  drawn across the full `K` range. GLPK trends are fitted over and drawn only
  across the tractable regime `K<=31`; they are **not** extrapolated to `K=53`,
  because the measured `K=53` points show the power law breaks (the solid GLPK
  line jumps above where its dashed trend pointed).
- **4 baseline lines** (dotted, horizontal): BR, CC, CLR, ECC per-instance
  inference times, for context.
- No interior grid; axis labelled in `K` (paper notation). All times in seconds.

## Companion table

`runtime_at_53.tex` is a `booktabs` table of the per-instance inference
time at `K=53` (in **seconds**) for every method across all four figures
(pre-order / partial order x Hamming / Subset). All entries are measured
directly at `K=53` (GLPK partial via `scripts/bench_step9_glpk_k53.py`, GLPK
pre-order via `scripts/bench_step10_one.py`; HiGHS over test instances;
baselines over the full test set). For the pre-order search, GLPK at **full
transitivity** is impractically slow (a single instance did not finish
within a 48-hour wall-clock limit, vs. under a second for HiGHS), so those
two cells are `n/a` with a `$\dagger$` note; the **height-2** pre-order case
is tractable but ~10^3x slower than HiGHS (about 10 min/instance, `622.6` s).
No extrapolation is used. Needs `\usepackage{booktabs}`.

## Usage

Add to the preamble:

```latex
\usepackage{pgfplots}
\pgfplotsset{compat=1.16}
\usepackage{booktabs}   % for runtime_at_53.tex
```

then `\input{runtime_preorder_hamming.tex}` (or paste the `tikzpicture` body)
inside a `figure` environment. Colours, marks, legend position, and the
`width`/`height` are all inline and easy to tweak.

`*.pdf` are compiled previews (regenerate with
`python scripts/bench_step8_tikz.py`).
