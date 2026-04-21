"""Shared base class and helpers for matrix factorisation methods."""

import warnings
import numpy as np
import matplotlib.pyplot as plt

from .. import plot as _plot


# ------------------------------------------------------------------ #
#  Module-level helpers
# ------------------------------------------------------------------ #

def _sanitize(X, label='data'):
    """Return a copy of *X* with NaN and Inf replaced by 0, warning if found."""
    nan_mask = ~np.isfinite(X)
    if nan_mask.any():
        n_bad    = nan_mask.sum()
        nan_frac = np.isnan(X).sum() / X.size
        inf_frac = np.isinf(X).sum() / X.size
        parts = []
        if nan_frac > 0:
            parts.append(f'{nan_frac:.2%} NaN')
        if inf_frac > 0:
            parts.append(f'{inf_frac:.2%} Inf')
        warnings.warn(
            f'{label} contains {", ".join(parts)} '
            f'({n_bad} of {X.size} values). Replacing with 0.',
            UserWarning, stacklevel=3,
        )
        X = X.copy()
        X[nan_mask] = 0.0
    return X


def _resolve_series(arg, labels, self_attr, slice_cols=None):
    """Resolve an expvars/coords argument into (list_of_arrays, list_of_labels)."""
    def prep(x):
        a = np.asarray(x)
        if slice_cols is not None:
            a = a[:, :slice_cols]
        return a

    if arg is None:
        series = [prep(self_attr)]
        labs   = [labels[0] if labels else None]
    elif isinstance(arg, np.ndarray):
        series = [prep(arg)]
        labs   = [labels[0] if labels else None]
    elif isinstance(arg, list):
        series, labs = [], []
        for k, item in enumerate(arg):
            series.append(prep(self_attr) if item == 'self' else prep(item))
            labs.append(labels[k] if labels and k < len(labels) else None)
    else:
        series = [prep(np.asarray(arg))]
        labs   = [labels[0] if labels else None]

    return series, labs


def _cycle_colors(color, n):
    default = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if color is None:
        return [default[i % len(default)] for i in range(n)]
    if isinstance(color, list):
        return [color[i % len(color)] for i in range(n)]
    return [color] * n


# ------------------------------------------------------------------ #
#  Base class
# ------------------------------------------------------------------ #

class _FactorizationBase:
    """Abstract base for matrix factorisation methods.

    Subclasses must implement :meth:`fit` and :meth:`transform`.
    After fitting the following attributes are populated:

    * ``loadings``     – (nc, n_features)
    * ``coords``       – (n_samples, nc)
    * ``expvars``      – (nc,)
    * ``frac_expvars`` – (nc,)
    * ``total_var``    – float
    """

    _component_prefix = 'Factor'

    def __init__(self, data, nc=20, name=None):
        self.data = np.asarray(data, dtype=float)
        self.nc = nc
        self.name = name
        self.loadings = None
        self.coords = None
        self.expvars = None
        self.frac_expvars = None
        self.mu = None
        self.total_var = None

    # ------------------------------------------------------------------ #
    #  Fitting / transform (subclass responsibility)
    # ------------------------------------------------------------------ #

    def fit(self, nc=None, **kwargs):
        raise NotImplementedError

    def transform(self, new_data, expvars_frac=False):
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    #  Repr
    # ------------------------------------------------------------------ #

    def _extra_repr_lines(self):
        """Override in subclasses to insert lines between header and table."""
        return []

    def __repr__(self):
        n, p = self.data.shape
        cls = type(self).__name__
        header = self.name if self.name else f'{cls} ({n} x {p})'
        lines = [header]

        if self.loadings is None:
            lines.append(f'  not fitted  (nc={self.nc})')
            return '\n'.join(lines)

        lines.append(f'  data: {n} x {p}  |  {self.nc} components')
        lines.extend(self._extra_repr_lines())

        # cumulative variance table at a few milestones
        cum = np.cumsum(self.frac_expvars)
        milestones = [i for i in (1, 5, 10, 20, self.nc) if i <= self.nc]
        # deduplicate while preserving order
        seen = set()
        milestones = [m for m in milestones if not (m in seen or seen.add(m))]
        col_w = max(len(str(milestones[-1])), 2)
        lines.append(f'  {"#":<{col_w}}  cum.var')
        for m in milestones:
            lines.append(f'  {m:<{col_w}}  {cum[m - 1]:.2%}')

        return '\n'.join(lines)

    # ------------------------------------------------------------------ #
    #  Explained-variance helper
    # ------------------------------------------------------------------ #

    def _compute_expvars(self):
        """Compute explained variance from ``coords`` and ``data``."""
        n = self.coords.shape[0]
        self.expvars = np.var(self.coords, axis=0, ddof=1)
        X_ref = (self.data - self.mu) if self.mu is not None else self.data
        self.total_var = np.sum(X_ref ** 2) / (n - 1)
        self.frac_expvars = self.expvars / self.total_var

    # ------------------------------------------------------------------ #
    #  Plots
    # ------------------------------------------------------------------ #

    def plot_expvars(
        self,
        axs=None,
        frac=True,
        expvars=None,
        labels=None,
        color=None,
        powlaw_kwargs=None,
        fsize=2,
    ):
        """Three-panel explained-variance summary.

        Left:   per-component explained variance (linear axes).
        Middle: cumulative explained variance (log-x).
        Right:  power-law fit via ``plot.plot_powlaw``.

        Parameters
        ----------
        axs : array of 3 Axes, optional
            If None a new (1, 3) figure is created.
        frac : bool
            If True, plot frac_expvars; else plot raw expvars.
        expvars : None | ndarray | list
            None    -> plot self's expvars only.
            ndarray -> plot that array (self not plotted).
            list    -> plot each element; the string 'self' inserts self's
                       expvars in the sequence.
        labels : list of str, optional
            One label per series.
        color : color or list of colors, optional
        powlaw_kwargs : dict, optional
            Extra keyword arguments forwarded to ``plot.plot_powlaw``.
        """
        if axs is None:
            fig, axs = plt.subplots(1, 3, figsize=(fsize * 3, fsize))
        ax_left, ax_mid, ax_right = axs

        self_attr = self.frac_expvars if frac else self.expvars
        series, labs = _resolve_series(expvars, labels, self_attr)
        colors = _cycle_colors(color or 'k', len(series))
        ylabel = 'Frac. exp. var.' if frac else 'Explained var.'
        prefix = self._component_prefix

        # -- left: per-component ----------------------------------------
        for ev, lab, col in zip(series, labs, colors):
            y = np.asarray(ev)
            x = np.arange(1, len(y) + 1)
            kw = dict(color=col)
            if lab is not None:
                kw['label'] = lab
            ax_left.plot(x, y, **kw)
        ax_left.set_xlabel(prefix)
        ax_left.set_ylabel(ylabel)
        if any(l is not None for l in labs):
            ax_left.legend()

        # -- middle: cumulative, log-x ----------------------------------
        for ev, lab, col in zip(series, labs, colors):
            y = np.cumsum(ev)
            x = np.arange(1, len(y) + 1)
            kw = dict(color=col)
            if lab is not None:
                kw['label'] = lab
            ax_mid.plot(x, y, **kw)
        ax_mid.set_xscale('log')
        ax_mid.set_xlabel(prefix)
        ax_mid.set_ylabel('Cum.' + ylabel.lower())
        if any(l is not None for l in labs):
            ax_mid.legend()

        # -- right: power-law fit ---------------------------------------
        pk = dict(powlaw_kwargs) if powlaw_kwargs else {}
        for ev, lab, col in zip(series, labs, colors):
            _plot.plot_powlaw(
                np.asarray(ev), ax=ax_right, color=col,
                label=lab or '', **pk,
            )
        ax_right.legend()

        plt.tight_layout()
        return axs

    def plot_coords(
        self,
        axs=None,
        nc=5,
        coords=None,
        labels=None,
        **pairplot_kwargs,
    ):
        """Pairwise scatter matrix of component coordinates.

        Parameters
        ----------
        nc : int
            Number of components to display (grid is nc x nc).
        coords : None | ndarray | list
            None    -> use self.coords.
            ndarray -> use that array.
            list    -> overlay each element; the string 'self' inserts
                       self.coords in the sequence.
        labels : list of str, optional
            One label per dataset.
        **pairplot_kwargs
            Forwarded to plot.pairplot.
        """
        self._check_fitted()
        nc = min(nc, self.nc)

        self_coords = self.coords
        series, labs = _resolve_series(coords, labels, self_coords, slice_cols=nc)

        dim_labels = [f'{self._component_prefix}{i + 1}' for i in range(nc)]

        return _plot.pairplot(
            series if len(series) > 1 else series[0],
            nc=nc,
            labels=labs if any(l is not None for l in labs) else None,
            dim_labels=dim_labels,
            axs=axs,
            **pairplot_kwargs,
        )

    def plot_loadings(
        self,
        axs=None,
        nc=5,
        loadings=None,
        labels=None,
        **pairplot_kwargs,
    ):
        """Pairwise scatter matrix of component loadings.

        Each point represents a feature projected into the space of loading
        vectors.  Mirrors :meth:`plot_coords` but operates on
        ``self.loadings.T`` (shape ``n_features × nc``) instead of coords.

        Parameters
        ----------
        nc : int
            Number of components to display (grid is nc x nc).
        loadings : None | ndarray | list
            None    -> use ``self.loadings.T``.
            ndarray -> use that array directly (shape ``n_features × nc``).
            list    -> overlay each element; the string ``'self'`` inserts
                       ``self.loadings.T`` in the sequence.
        labels : list of str, optional
            One label per dataset.
        **pairplot_kwargs
            Forwarded to plot.pairplot.
        """
        self._check_fitted()
        nc = min(nc, self.nc)

        self_loadings = self.loadings.T  # (n_features, nc)
        series, labs = _resolve_series(loadings, labels, self_loadings, slice_cols=nc)

        dim_labels = [f'{self._component_prefix}{i + 1}' for i in range(nc)]

        return _plot.pairplot(
            series if len(series) > 1 else series[0],
            nc=nc,
            labels=labs if any(l is not None for l in labs) else None,
            dim_labels=dim_labels,
            axs=axs,
            **pairplot_kwargs,
        )

    # ------------------------------------------------------------------ #
    #  Private
    # ------------------------------------------------------------------ #

    def _check_fitted(self):
        if self.loadings is None:
            raise RuntimeError(
                f"{type(self).__name__} has not been fitted — call .fit() first."
            )
