"""Non-negative Matrix Factorisation (NMF) via scikit-learn."""

import warnings
import numpy as np

from .base import _FactorizationBase, _sanitize


class NMF(_FactorizationBase):
    """Non-negative Matrix Factorisation: X ≈ W H.

    Parameters
    ----------
    data : ndarray, shape (n_samples, n_features)
        Non-negative input matrix.
    nc : int
        Default number of components (default 20).

    Attributes (populated after .fit())
    ------------------------------------
    loadings            : (nc, n_features) – H matrix
    coords              : (n_samples, nc)  – W matrix
    expvars             : (nc,)
    frac_expvars        : (nc,)
    total_var           : float
    reconstruction_error_ : float – Frobenius norm ‖X − WH‖
    """

    _component_prefix = 'NMF'

    def _extra_repr_lines(self):
        if hasattr(self, 'reconstruction_error_'):
            return [f'  reconstruction error: {self.reconstruction_error_:.4g}']
        return []

    def fit(self, nc=None, method='cd', max_iter=500, random_state=0,
            **nmf_kwargs):
        """Fit NMF to self.data.

        Parameters
        ----------
        nc : int, optional
            Number of components.  Overwrites self.nc.
        method : {'cd', 'mu'}
            'cd'  – coordinate descent (default).
            'mu'  – multiplicative update.
        max_iter : int
        random_state : int
        **nmf_kwargs
            Forwarded to ``sklearn.decomposition.NMF``.
        """
        from sklearn.decomposition import NMF as _skNMF

        if nc is not None:
            self.nc = nc

        X = _sanitize(self.data, label='data')
        if (X < 0).any():
            warnings.warn(
                'NMF input contains negative values; clipping to 0.',
                UserWarning, stacklevel=2,
            )
            X = np.clip(X, 0, None)

        model = _skNMF(
            n_components=self.nc, solver=method, max_iter=max_iter,
            random_state=random_state, **nmf_kwargs,
        )
        self.coords   = model.fit_transform(X)      # W: (n, nc)
        self.loadings = model.components_            # H: (nc, p)
        self._sklearn_model = model
        self.mu = None
        self.reconstruction_error_ = model.reconstruction_err_
        self._compute_expvars()
        return self

    def transform(self, new_data, expvars_frac=False):
        """Project new data by solving for W_new given the fitted H.

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
        X_new = _sanitize(np.asarray(new_data, dtype=float), label='new_data')
        if (X_new < 0).any():
            X_new = np.clip(X_new, 0, None)
        new_coords  = self._sklearn_model.transform(X_new)
        new_expvars = np.var(new_coords, axis=0, ddof=1)
        if expvars_frac:
            m = X_new.shape[0]
            new_total_var = np.sum(X_new ** 2) / (m - 1)
            new_expvars   = new_expvars / new_total_var
        return new_coords, new_expvars
