# Runtime-scaling TikZ figures (reviewer 1)

Four copy-paste `pgfplots` figures for the per-instance ILP runtime scaling,
one per (preference order, target metric):

| file | preference order | target metric |
|---|---|---|
| `runtime_preorder_hamming.tex` | pre-order | Hamming accuracy |
| `runtime_preorder_subset.tex`  | pre-order | Subset (0/1 exact-match) |
| `runtime_partial_hamming.tex`  | partial order | Hamming accuracy |
| `runtime_partial_subset.tex`   | partial order | Subset (0/1 exact-match) |

Each figure shows, on log-log axes (x = number of labels `K`):

- **4 main curves**: GLPK / HiGHS x full-transitivity / height=2, measured on
  the real enron model (mean over 300 test instances, `K` varied by label
  subsets of the same trained model).
- **1 dashed trend line per curve**: a least-squares power-law fit `K^x`; the
  fitted exponent is shown in the legend. Trend lines span the full `K` range,
  so GLPK's trend is visible even past `K=31` where it is too slow to measure.
- **4 baseline lines** (dotted, horizontal): BR, CC, CLR, ECC per-instance
  inference times, for context.
- No interior grid; axis labelled in `K` (paper notation).

## Companion table

`runtime_at_53.tex` is a `booktabs` table of the per-instance inference
time at `K=53` for every method across all four figures (pre-order /
partial order x Hamming / Subset). HiGHS values are measured; GLPK is
capped at `K<=31`, so its `K=53` entries are extrapolated from the
least-squares power-law fit and flagged with `*`. Needs
`\usepackage{booktabs}`.

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
