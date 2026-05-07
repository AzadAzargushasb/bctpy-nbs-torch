# `nbs_torch` documentation

Two self-contained LaTeX documents describing the torch-accelerated NBS module
in [`bct/nbs_torch.py`](../../bct/nbs_torch.py) and the verification suite in
[`test/nbs_torch_test.py`](../../test/nbs_torch_test.py).

| File | Contents |
| --- | --- |
| [`01_walkthrough.tex`](01_walkthrough.tex) | Line-by-line walkthrough of the reference [`bct/nbs.py`](../../bct/nbs.py): input-data spec, observed t-statistic, threshold + connected components, the permutation loop, and a per-phase computational-cost analysis identifying the two bottlenecks (the K·m scalar Python t-tests and the K+1 calls to `get_components`). |
| [`02_torch_replacement.tex`](02_torch_replacement.tex) | Design of the torch replacement. Derives the four-GEMM batched permutation t-statistic, motivates `scipy.sparse.csgraph.connected_components` as the components replacement, presents the algorithm pseudocode, gives expected runtime/memory budgets, and specifies the V1–V12 verification plan. |

## Building

Each document is a standalone `\documentclass{article}` and uses only
common packages (`amsmath`, `amssymb`, `booktabs`, `listings`, `algorithm2e`,
`geometry`, `xcolor`, `hyperref`). No `--shell-escape` required.

```bash
pdflatex 01_walkthrough.tex
pdflatex 02_torch_replacement.tex
```

Or upload either `.tex` to Overleaf as a fresh project.
