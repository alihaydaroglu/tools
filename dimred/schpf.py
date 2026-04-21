"""single-cell Hierarchical Poisson Factorisation (scHPF).

Requires the ``schpf`` package (``pip install schpf``).
"""

import warnings
import numpy as np

from .base import _FactorizationBase, _sanitize


class scHPF(_FactorizationBase):
    """Hierarchical Poisson Factorisation for count data.

    Parameters
    ----------
    data : ndarray, shape (n_samples, n_features)
        Non-negative count matrix.
    nc : int
        Default number of factors (default 20).

    Attributes (populated after .fit())
    ------------------------------------
    loadings     : (nc, n_features) – gene/feature scores (transposed)
    coords       : (n_samples, nc)  – cell/sample scores
    expvars      : (nc,)
    frac_expvars : (nc,)
    total_var    : float
    """

    _component_prefix = 'scHPF'

    def fit(self, nc=None, max_iter=1000, min_iter=30, **schpf_kwargs):
        """Fit scHPF to self.data.

        Parameters
        ----------
        nc : int, optional
            Number of factors.  Overwrites self.nc.
        max_iter : int
        min_iter : int
        **schpf_kwargs
            Forwarded to ``schpf.scHPF``.
        """
        try:
            from schpf import scHPF as _scHPF
        except ImportError:
            raise ImportError(
                "scHPF requires the 'schpf' package. "
                "Install with:  pip install schpf"
            ) from None

        from scipy.sparse import coo_matrix

        if nc is not None:
            self.nc = nc

        X = _sanitize(self.data, label='data')
        if (X < 0).any():
            warnings.warn(
                'scHPF input contains negative values; clipping to 0.',
                UserWarning, stacklevel=2,
            )
            X = np.clip(X, 0, None)

        X_int = np.rint(X).astype(int)
        if not np.allclose(X, X_int):
            warnings.warn(
                'scHPF expects integer counts; rounding to nearest int.',
                UserWarning, stacklevel=2,
            )

        model = _scHPF(
            nfactors=self.nc, max_iter=max_iter,
            min_iter=min_iter, **schpf_kwargs,
        )
        model.fit(coo_matrix(X_int))
        self._model = model

        self.coords   = model.cell_score()             # (n, nc)
        self.loadings = model.gene_score().T            # (nc, p)
        self.mu = None
        self._compute_expvars()
        return self

    def transform(self, new_data, expvars_frac=False):
        """Project new data via row-wise NNLS against the fitted loadings.

        Parameters
        ----------
        new_data : ndarray, shape (m, n_features)
        expvars_frac : bool

        Returns
        -------
        new_coords  : ndarray, shape (m, nc)
        new_expvars : ndarray, shape (nc,)
        """
        self._check_fitted()
        from scipy.optimize import nnls

        X_new = _sanitize(np.asarray(new_data, dtype=float), label='new_data')
        if (X_new < 0).any():
            X_new = np.clip(X_new, 0, None)

        H_T = self.loadings.T                           # (p, nc)
        m = X_new.shape[0]
        new_coords = np.zeros((m, self.nc))
        for i in range(m):
            new_coords[i], _ = nnls(H_T, X_new[i])

        new_expvars = np.var(new_coords, axis=0, ddof=1)
        if expvars_frac:
            new_total_var = np.sum(X_new ** 2) / (m - 1)
            new_expvars   = new_expvars / new_total_var
        return new_coords, new_expvars
