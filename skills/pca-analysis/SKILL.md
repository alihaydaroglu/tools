---
name: pca-analysis
description: Use when the user wants to run PCA on a dataset, interpret principal components, or evaluate how many components are meaningful. Covers which module to use (`tools.dimred.pca`, re-exported as `tools.pca`) and — importantly — which cross-validation approach fits the data structure (cvPCA, SVCA, train/test split, Stringer-style shuffles). Invoke whenever PCA, principal components, variance explained, or dimensionality reduction comes up on tabular or neural data.
---

# PCA analysis

This skill is a starting point — it will be fleshed out as more analysis-specific skills are added. For now it covers: which module to use, and the cross-validation choice.

## Which module

Use `tools.dimred.pca.PCA`, re-exported as `tools.pca.PCA`:

```python
from tools.pca import PCA

pca = PCA(data, nc=20)       # data: (n_samples, n_features)
pca.fit(method="svd")        # or "direct" for eigendecomposition of cov
pca.loadings                 # (nc, n_features) — unit-norm PC directions
pca.coords                   # (n_samples, nc) — projections of training data
pca.expvars                  # (nc,) — variance per PC (ddof=1)
pca.frac_expvars             # (nc,) — fraction of total variance per PC
pca.mu                       # (n_features,) — feature means used for centering
pca.total_var                # scalar
```

Do not use `sklearn.decomposition.PCA` directly for new analyses in this codebase. The in-house class has the attribute names and conventions the rest of the tools expect.

For lower-level operations (explicit centering, reprojection onto existing components, explained-variance utilities), `tools.pca_tools` has `fit_pca`, `transform`, `untransform`, `reconstruct`, `explained_vars`.

## Cross-validation — the important part

Raw PCA on the full dataset gives you `expvars`, but those are inflated by noise — every PC captures *some* variance because overfitting. For any claim about "real" dimensionality, suggest a cross-validated variant. Which one depends on the data structure.

### When the user has stimulus-repeat structure (stimuli × repeats × neurons)

**cvPCA** (Stringer 2019): fit PCs on the mean response over half the repeats, measure how much variance those PCs capture in the other half. Noise doesn't correlate across repeat-splits, so it drops out.

```python
from tools.pca_tools import cv_PCA, shuff_cvpca

# single split
sig_vars = cv_PCA(r0, r1)           # r0, r1: (n_stim, n_neurons)

# with shuffles for stability
sig_vars_mean = shuff_cvpca(respmat, n_shuff=5).mean(axis=0)
# respmat: (n_stim, n_repeats, n_neurons)
```

Plot the log-log decay and fit a power law with `tools.plot.plot_powlaw`.

### When there are two non-overlapping populations (e.g. two hemispheres, two mice) recorded simultaneously

**SVCA** (shared variance component analysis): split cells into two sets, split time into two sets, find dimensions of shared variance across populations that generalize to held-out time. Captures what's *common* across populations, not just total variance.

```python
from tools.pca_tools import split_and_svca
shared_var, tot_cov_space_var, proj_vars, full_vars, svd = split_and_svca(
    spks, n_comp=None, seed=0
)
# fraction of shared variance = shared_var.sum() / tot_cov_space_var.sum()
```

### When there's a clear train/test split (time-series, batches)

Just split explicitly: fit on train, project test, compute held-out variance with `explained_vars(x_test, components, mean)`. This is a baseline — less powerful than cvPCA/SVCA but always available.

```python
from tools.pca_tools import fit_pca, explained_vars
comps, mean = fit_pca(x_train, n_components=50)
held_out_var = explained_vars(x_test, comps, mean, return_frac=True)
```

### When the user has none of the above

If there's no natural repeat / split structure, cross-validation options are thinner. Offer:
- **Leave-one-out or k-fold on samples**: fit on N-k, project the k, see how much variance is captured. Computationally expensive.
- **Parallel analysis / permutation**: shuffle each feature independently, refit PCA, compare real `expvars` to shuffled distribution. Components exceeding the shuffled 95th percentile are "real."
- **Just report `expvars` without CV**, noting the inflation caveat. Fine for exploratory use; not fine for a claim about dimensionality.

## The conversation to have

When the user asks for PCA, don't just run it. Ask:

1. **What is the claim?** Is this exploratory visualization (coords in PC space, 2D scatter), or is it a quantitative claim about dimensionality / variance structure?
2. **What structure does the data have?** Repeats, populations, time-splits, none?
3. **Which CV fits?** Based on 2, suggest cvPCA / SVCA / train-test / shuffle / none.

For pure visualization (plotting data in top-2 or top-3 PC space), uncross-validated PCA is fine — just label it honestly. For any variance-explained claim that will go into a paper or interpretation, cross-validate.

## What to save

Following the `notebook-to-script` / `plotting-conventions` rules: save the quantities that feed into plots.

- `pca.loadings` — needed to reproject new data or re-plot loadings.
- `pca.coords` — needed for PC-space scatter plots.
- `pca.frac_expvars` (or `sig_vars` from cvPCA, `shared_var`/`tot_cov_space_var` from SVCA) — needed for variance-explained plots.
- The shuffle distribution if parallel analysis was used.

Intermediate arrays (centered data, covariance matrices) do not need to be saved.

## What NOT to do

- **Don't use `sklearn.PCA` directly** in new analyses — use `tools.pca.PCA`.
- **Don't report variance-explained without cross-validation** for any claim about real dimensionality. Noise inflates raw `expvars`.
- **Don't pick the CV method silently.** Ask about data structure and justify the choice.
- **Don't forget to center.** Both `tools.pca.PCA` and `tools.pca_tools.fit_pca` center automatically, but if doing projections manually, `transform` will center by the training `mean` — pass it explicitly.
- **Don't conflate `expvars` units.** `expvars` is variance (ddof=1); `frac_expvars` is the fraction of total. Label axes accordingly.

## Future extensions

This skill will expand as specific analysis methods get added — planned additions: NMF, CNMF, sCHPF (already in `tools/dimred/`), UMAP, and CCA-family methods. Each will get the same treatment: which module, which CV, what to save.
