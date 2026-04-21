"""PCA decomposition via SVD or explicit covariance eigendecomposition."""

import numpy as np
from scipy import linalg

from .base import _FactorizationBase, _sanitize


class PCA(_FactorizationBase):
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

    _component_prefix = 'PC'

    def _extra_repr_lines(self):
        # participation ratio = (sum λ)^2 / sum(λ^2)
        pr = self.expvars.sum() ** 2 / (self.expvars ** 2).sum()
        return [f'  participation ratio: {pr:.2f}']

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
        self.total_var = np.sum(X_c ** 2) / (n - 1)

        if method == 'svd':
            U, s, Vt = np.linalg.svd(X_c, full_matrices=False)

            self.loadings = Vt[:nc].copy()                  # (nc, p)
            self.coords   = X_c @ Vt[:nc].T                 # (n, nc)
            self.eigvals = s[:nc]
            self.expvars  = s[:nc] ** 2 / (n - 1)           # (nc,)

        elif method == 'direct':
            C = (X_c.T @ X_c) / (n - 1)                    # (p, p)
            eigenvalues, eigenvectors = linalg.eigh(C)
            eigenvalues  = eigenvalues[::-1].copy()
            eigenvectors = eigenvectors[:, ::-1].copy()

            self.loadings = eigenvectors[:, :nc].T           # (nc, p)
            self.coords   = X_c @ eigenvectors[:, :nc]       # (n, nc)
            self.expvars  = eigenvalues[:nc].copy()          # (nc,)

        else:
            raise ValueError(f"method must be 'svd' or 'direct', got '{method}'")

        self.frac_expvars = self.expvars / self.total_var
        return self

    # ------------------------------------------------------------------ #
    #  Reconstruct
    # ------------------------------------------------------------------ #

    def reconstruct(self, nc, coords=None):
        """Reconstruct data from the first *nc* principal components.

        Parameters
        ----------
        nc : int
            Number of components to use (rank of the reconstruction).
        coords : ndarray, shape (m, >= nc), optional
            Coordinates to reconstruct from.  Defaults to ``self.coords``
            (i.e. the training data projection).

        Returns
        -------
        recon : ndarray, shape (m, n_features)
        """
        self._check_fitted()
        c = self.coords if coords is None else np.asarray(coords, dtype=float)
        return c[:, :nc] @ self.loadings[:nc, :] + self.mu

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
            own total variance.

        Returns
        -------
        new_coords  : ndarray, shape (m, nc)
        new_expvars : ndarray, shape (nc,)
        """
        self._check_fitted()
        X_new = _sanitize(np.asarray(new_data, dtype=float), label='new_data')
        X_new_c = X_new - self.mu
        new_coords  = X_new_c @ self.loadings.T
        new_expvars = np.var(new_coords, axis=0, ddof=1)
        if expvars_frac:
            m = X_new.shape[0]
            new_total_var = np.sum(X_new_c ** 2) / (m - 1)
            new_expvars   = new_expvars / new_total_var
        return new_coords, new_expvars
