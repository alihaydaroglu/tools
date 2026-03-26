import warnings
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

from . import plot as _plot


def _sanitize(X, label='data'):
    """Return a copy of X with NaN and Inf replaced by 0, warning if any found."""
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
            f'PCA {label} contains {", ".join(parts)} '
            f'({n_bad} of {X.size} values). Replacing with 0.',
            UserWarning, stacklevel=3,
        )
        X = X.copy()
        X[nan_mask] = 0.0
    return X


class PCA:
    """PCA decomposition via SVD or explicit covariance eigendecomposition.

    Parameters
    ----------
    data : ndarray, shape (n_samples, n_features)
    nc   : int
        Default number of components (default 20).  Clipped to
        min(n_samples, n_features) at fit time.

    Attributes (populated after .fit())
    ------------------------------------
    loadings     : (nc, n_features)  – unit-norm PC directions (rows)
    coords       : (n_samples, nc)   – projections of training data
    expvars      : (nc,)             – variance captured per PC  [ddof=1]
    frac_expvars : (nc,)             – expvars / total_var of training data
    mu           : (n_features,)     – feature means used for centering
    total_var    : float             – total variance of training data
    nc           : int               – number of fitted components
    """

    def __init__(self, data, nc=20):
        self.data = np.asarray(data, dtype=float)
        self.nc = nc
        self.loadings = None
        self.coords = None
        self.expvars = None
        self.frac_expvars = None
        self.mu = None
        self.total_var = None

    # ------------------------------------------------------------------ #
    #  Fitting
    # ------------------------------------------------------------------ #

    def fit(self, nc=None, method='svd'):
        """Fit PCA to self.data.

        Parameters
        ----------
        nc : int, optional
            Number of components.  Overwrites self.nc; clipped to
            min(n_samples, n_features).
        method : {'svd', 'direct'}
            'svd'    – thin SVD of the centred data matrix X_c (default).
            'direct' – explicitly form the sample covariance matrix
                       C = X_c.T @ X_c / (n-1) and eigendecompose with
                       scipy.linalg.eigh.
        """
        if nc is not None:
            self.nc = nc

        X = _sanitize(self.data, label='data')
        n, p = X.shape
        self.nc = min(self.nc, n, p)
        nc = self.nc

        # centre
        self.mu = X.mean(axis=0)       # (p,)
        X_c = X - self.mu              # (n, p), zero column means

        # total variance = trace of sample covariance = ||X_c||_F^2 / (n-1)
        # This equals sum of ALL squared singular values of X_c divided by (n-1),
        # and is used as the denominator for frac_expvars.
        self.total_var = np.sum(X_c ** 2) / (n - 1)

        if method == 'svd':
            # Thin SVD: X_c = U diag(s) Vt
            # U  : (n, k),  s : (k,) descending,  Vt : (k, p),  k = min(n,p)
            U, s, Vt = np.linalg.svd(X_c, full_matrices=False)

            self.loadings = Vt[:nc].copy()                  # (nc, p)
            self.coords   = X_c @ Vt[:nc].T                 # (n, nc)
            # var(coords[:, i], ddof=1)
            #   = Vt[i]^T X_c^T X_c Vt[i] / (n-1) = s[i]^2 / (n-1)
            self.expvars  = s[:nc] ** 2 / (n - 1)           # (nc,)

        elif method == 'direct':
            # Sample covariance matrix
            C = (X_c.T @ X_c) / (n - 1)                    # (p, p)
            # eigh returns eigenvalues in ascending order
            eigenvalues, eigenvectors = linalg.eigh(C)
            # reverse to descending
            eigenvalues  = eigenvalues[::-1].copy()
            eigenvectors = eigenvectors[:, ::-1].copy()

            self.loadings = eigenvectors[:, :nc].T           # (nc, p)
            self.coords   = X_c @ eigenvectors[:, :nc]       # (n, nc)
            # eigenvalue_i = var(X_c @ v_i, ddof=1) by construction
            self.expvars  = eigenvalues[:nc].copy()          # (nc,)

        else:
            raise ValueError(f"method must be 'svd' or 'direct', got '{method}'")

        self.frac_expvars = self.expvars / self.total_var
        return self

    # ------------------------------------------------------------------ #
    #  Transform
    # ------------------------------------------------------------------ #

    def transform(self, new_data, expvars_frac=False):
        """Project new data onto the fitted PCA basis.

        Parameters
        ----------
        new_data : ndarray, shape (m, n_features)
        expvars_frac : bool
            If True, each component's variance is normalised by new_data's
            own total variance (so fractions sum to <=1 if the basis captures
            everything in new_data).
            If False, return absolute variance captured per component.

        Returns
        -------
        new_coords  : ndarray, shape (m, nc)
        new_expvars : ndarray, shape (nc,)
        """
        self._check_fitted()
        X_new = _sanitize(np.asarray(new_data, dtype=float), label='new_data')
        m = X_new.shape[0]
        X_new_c = X_new - self.mu                            # centre with fitted mean
        new_coords  = X_new_c @ self.loadings.T             # (m, nc)
        new_expvars = np.var(new_coords, axis=0, ddof=1)    # (nc,)
        if expvars_frac:
            new_total_var = np.sum(X_new_c ** 2) / (m - 1)
            new_expvars   = new_expvars / new_total_var
        return new_coords, new_expvars

    # ------------------------------------------------------------------ #
    #  Plots
    # ------------------------------------------------------------------ #

    def plot_expvars(
        self,
        ax=None,
        frac=True,
        expvars=None,
        labels=None,
        color=None,
        logx=False,
        logy=False,
        cum=False,
    ):
        """Plot explained variance (or fraction) per PC.

        Parameters
        ----------
        frac : bool
            If True, plot frac_expvars; else plot raw expvars.
        expvars : None | ndarray | list
            None    -> plot self's expvars only.
            ndarray -> plot that array (self not plotted).
            list    -> plot each element; the string 'self' inserts self's
                       expvars in the sequence.
        labels : list of str, optional
            One label per series, in order.
        color : color or list of colors, optional
        logx, logy : bool
        cum : bool
            Plot cumulative explained variance.
        """
        if ax is None:
            _, ax = plt.subplots()

        self_attr = self.frac_expvars if frac else self.expvars
        series, labs = _resolve_series(expvars, labels, self_attr)
        colors = _cycle_colors(color, len(series))

        for ev, lab, col in zip(series, labs, colors):
            y = np.cumsum(ev) if cum else np.asarray(ev)
            x = np.arange(1, len(y) + 1)
            kw = dict(color=col)
            if lab is not None:
                kw['label'] = lab
            ax.plot(x, y, **kw)

        if logx:
            ax.set_xscale('log')
        if logy:
            ax.set_yscale('log')

        prefix = 'Cumulative ' if cum else ''
        ax.set_xlabel('PC')
        ax.set_ylabel(prefix + ('Frac. explained var.' if frac else 'Explained var.'))
        if any(l is not None for l in labs):
            ax.legend()

        return ax

    def plot_coords(
        self,
        axs=None,
        nc=5,
        coords=None,
        labels=None,
        **pairplot_kwargs,
    ):
        """Pairwise scatter matrix of PC coordinates.

        Delegates to plot.pairplot.  Diagonal panels are histograms; all
        off-diagonal panels are scatter plots (density-coloured when
        n_points > 100).

        Parameters
        ----------
        nc : int
            Number of PCs to display (grid is nc x nc).
        coords : None | ndarray | list
            None    -> use self.coords.
            ndarray -> use that array.
            list    -> overlay each element; the string 'self' inserts
                       self.coords in the sequence.
        labels : list of str, optional
            One label per dataset.
        **pairplot_kwargs
            Forwarded verbatim to plot.pairplot (square_axes, share_limits,
            percentile_limit, density_threshold, and any density_scatter
            kwargs).
        """
        self._check_fitted()
        nc = min(nc, self.nc)

        self_coords = self.coords
        series, labs = _resolve_series(coords, labels, self_coords, slice_cols=nc)

        dim_labels = [f'PC{i + 1}' for i in range(nc)]

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
            raise RuntimeError("PCA has not been fitted — call .fit() first.")


# ------------------------------------------------------------------ #
#  Module-level helpers (used by the class; not part of the public API)
# ------------------------------------------------------------------ #

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
