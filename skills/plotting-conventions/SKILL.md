---
name: plotting-conventions
description: Use when the user is making, revising, or reviewing plots — choosing axes, bins, colormaps, layouts, subplot sharing, or save formats. Covers the house formatting rules (shared bins for histograms, shared axes + identity line for same-quantity scatters, when to share axis/colormap limits across subplots), the requirement to produce summary plots aggregating across sweep axes (per-subject, per-cell-type, per-gene-class), the directory convention for summary vs per-condition plots, and the three-format save discipline (svg + png + mpl pickle). Invoke whenever matplotlib, subplots, histograms, scatters, colormaps, summary figures, or `tools.plot` come up.
---

# Plotting conventions

This skill encodes the house rules for plots. The goal: plots are comparable, consistent, and re-editable later without re-running analysis.

Plotting utilities live in `tools/plot.py`. Prefer them over reinventing. When a helper doesn't exist for what you need, write it there (generally useful) or in the project's own plotting module (domain-specific), not inline in the script.

## Separation from analysis

The plotting phase of a script should only read from saved results — never recompute. This is what makes it independently re-runnable: if colors or labels need to change, you re-run plotting against the saved data without paying the analysis cost again.

See the `notebook-to-script` skill for the analysis/plotting split at the script level.

## Hard rules

### Histograms

- **Share bins across all histograms being compared.** Never let matplotlib auto-bin two histograms separately — they become incomparable. Compute the bin edges once (typically from the combined range of all datasets) and pass them to every `ax.hist` call.
- Use `density=True` when comparing distributions of different sample sizes. Use `density=False` (counts) when the absolute number matters.
- If plotting multiple histograms on one axis, use `alpha=0.5` or `histtype="step"` so overlap is visible.

```python
import numpy as np
edges = np.linspace(min(np.concatenate(groups)),
                    max(np.concatenate(groups)), 41)
for group, label in zip(groups, labels):
    ax.hist(group, bins=edges, alpha=0.5, label=label, density=True)
```

### Scatter plots

- **When x and y are the same quantity measured under different conditions**, axes must be shared and an identity line (`y = x`) drawn through the plot. Without these, the reader can't judge whether points deviate from equality.
- **When x and y are different quantities**, a regression line is often useful. `tools.plot.scatter` draws one by default with the R value in the legend.
- For dense scatter, use `tools.plot.density_scatter` (color-by-density) instead of raw points with `alpha`.

```python
lim = (min(x.min(), y.min()), max(x.max(), y.max()))
ax.scatter(x, y, s=5, alpha=0.5)
ax.plot(lim, lim, "k--", lw=1, label="y = x")  # identity
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_aspect("equal")
```

### Subplot sharing — axis limits and colormap ranges

Whether to share axes/colormap limits across subplots depends on intent. **Ask the user which they want** unless it is obvious from context:

| Intent | Share limits? |
|---|---|
| Compare magnitudes across conditions | **Yes** — share |
| Show structure within each condition regardless of scale | **No** — independent |
| Heatmap grid comparing the same quantity | **Yes** — share colormap range (`vmin`/`vmax`) |
| Heatmap grid where each panel has its own scale story | **No** — per-panel colorbars |

When sharing is the answer, use `plt.subplots(..., sharex=True, sharey=True)` and compute the shared `vmin`/`vmax` from all data before plotting.

If unclear: ask one short question like "should the y-axis / colormap be shared across these four panels, or independent?" Don't guess.

### Colormaps

- **Sequential data** (positive-only, e.g. firing rates, variance): `viridis`, `magma`, `cividis`. Avoid `jet`.
- **Diverging data** (signed, e.g. correlations, differences): `tools.plot.diverging_cmap()` or `"RdBu_r"`. Center on zero: `vmin = -abs_max, vmax = abs_max`.
- **Categorical data**: `tools.plot.categorical_cmap()` with an explicit color list — do not sample a continuous colormap at equal intervals for categories.
- Always show a colorbar when a colormap is used for quantitative data.

### General formatting

- Remove top and right spines by default: `tools.plot.turn_off_spines(ax)`.
- Use `bbox_inches="tight"` when saving (chantier's `save_figure` already does this).
- Label axes with units where applicable.
- Legend only when there's more than one series; place it to minimize overlap with data (`loc="best"` is fine as a start, override if it covers points).
- Match figure size to aspect ratio of the data — don't force square when the data is 3:1.

## Save formats — three dirs, three purposes

Saved figures land in up to three formats:

| Dir | Format | When |
|---|---|---|
| `figures/png/` | PNG (150 dpi) | **Always.** Viewing, embedding in markdown/summary. |
| `figures/svg/` | SVG | **Always.** Paper figures — vector, editable in Illustrator/Inkscape. |
| `figures/pkl/` | Matplotlib pickle | **Summary plots only** (names under `summary/`). Reloadable Figure for later restyle. |

**Pkl is summary-only.** Pickling a matplotlib Figure walks its full artist tree and can be the single most expensive save step on sweep-heavy runs (thousands of per-condition figures × ~1 MB pickle each = disk + wall-clock blowup). The use case for pkl — "reload the figure later and tweak labels / colors / limits without re-running analysis" — overwhelmingly applies to *summary* figures that end up in papers / slide decks, not to the hundreds of per-condition diagnostics that ride along. Keep the pkl for the handful of plots you'd actually want to restyle; skip it for the rest.

When used inside a chantier `ResultBuilder`, `rec.save_figure(fig, name)` infers the rule from `name`: paths under `summary/` get the pkl; others get png + svg only. Explicit override with `save_pkl=True/False` if you need it. Outside chantier, replicate the pattern:

```python
import pickle
is_summary = name.startswith("summary/")
fig.savefig(f"figures/png/{name}.png", dpi=150, bbox_inches="tight")
fig.savefig(f"figures/svg/{name}.svg", bbox_inches="tight")
if is_summary:
    with open(f"figures/pkl/{name}.pkl", "wb") as f:
        pickle.dump(fig, f)
```

Reload a pickled summary later:
```python
import pickle
import matplotlib.pyplot as plt
fig = pickle.load(open("figures/pkl/summary/name.pkl", "rb"))
ax = fig.axes[0]
ax.set_xlabel("better label")
fig.savefig("figures/svg/summary/name.svg")
plt.show()
```

The pickle is the key for the handful of plots that get restyled later. For the many per-condition diagnostics produced by a sweep, the cheap fix is to re-run the plotting phase against the saved data parquet — the point of separating analysis and plotting is exactly that.

## Summary plots for sweep analyses

Whenever an analysis is repeated across a sweep axis (per subject, per cell type, per gene class, per parameter value), **you must produce summary plot(s) that aggregate across the sweep**. A stack of N per-condition figures with no aggregate is not a complete result — the summary is where the comparison lives.

Good summary plots for sweeps:
- All curves on one axis, one line per condition (shared bins, shared limits, legend).
- Bar / violin / strip plot of a scalar metric per condition with CIs.
- Heatmap with conditions on one axis and the per-condition metric across.
- Scatter of condition-vs-condition if pairs are meaningful.

Bad summary plots:
- A 3×3 grid of the same per-condition plot at different zoom levels.
- Something that requires the reader to flip between N separate files to do the comparison themselves — that's what the summary is for.

### Directory convention

Summary plots live in a dedicated flat directory, separate from per-condition plots:

```
figures/png/
├── summary/
│   ├── variance_by_celltype.png
│   ├── powerlaw_slopes.png
│   └── crossvalidation_by_genegroup.png
├── per_celltype/
│   ├── loadings_pvalb.png
│   ├── loadings_sst.png
│   └── loadings_vip.png
└── per_genegroup/
    ├── variance_gpcr.png
    └── variance_ion_channel.png
```

Rules:
- `summary/` is **flat**. Do not nest summaries under per-condition subdirectories (e.g., no `per_celltype/pvalb/summary.png`). All summaries for a result live together so you can browse them in one glance.
- Per-condition plots go in descriptive group directories (`per_<axis>/`). If there is only one sweep axis and ≤ ~5 conditions, a flat root with suffixed names is also fine (`loadings_pvalb.png`, `loadings_sst.png`).
- The same structure is mirrored in `figures/svg/` and `figures/pkl/` — `rec.save_figure(fig, "summary/variance_by_celltype")` writes all three.

With chantier:

```python
# summary plots
rec.save_figure(fig_summary, "summary/variance_by_celltype")
rec.save_figure(fig_slopes,  "summary/powerlaw_slopes")

# per-condition plots
for ct in CELL_TYPES:
    rec.save_figure(fig_loadings[ct], f"per_celltype/loadings_{ct}")
```

`rec.save_figure` creates the intermediate directories automatically.

## What to reach for in `tools.plot`

- `scatter` — scatter with optional regression line and binned mean/std overlay.
- `density_scatter` — color points by local density.
- `hist2d` — 2D histogram.
- `pairplot` — pairwise subplot grid.
- `show_covmat` — covariance/correlation matrix heatmap.
- `show_img` — image with consistent treatment.
- `diverging_cmap` / `categorical_cmap` / `linear_cmap` — colormap constructors.
- `plot_powlaw` — power-law fitting overlay on log-log.
- `turn_off_spines` — default cleanup.
- `plot_percentiles` — percentile bands instead of mean±std.

Read the signature in `tools/plot.py` before using — several accept a `**kwargs`-like dict of style params.

## What NOT to do

- **Don't auto-bin histograms separately.** Always compute shared edges first.
- **Don't compare same-quantity scatters without an identity line.** It's not a judgment call; it's a rule.
- **Don't use `jet`.** Or `rainbow`. Perceptually nonlinear.
- **Don't sample a continuous colormap for categorical groups.** Adjacent categories become confusably similar.
- **Don't save only PNG.** You will regret it the first time you need to tweak a label for a paper figure.
- **Don't plot inside the analysis phase** of a script. Keep them separate so plotting is re-runnable without recomputation.
- **Don't assume sharing/not-sharing axes.** Ask if ambiguous.
- **Don't ship a sweep analysis with only per-condition plots.** Every sweep axis needs at least one summary plot that aggregates across it.
- **Don't nest summary plots inside per-condition directories.** `summary/` is flat. Summaries live together so they can be browsed at a glance.
