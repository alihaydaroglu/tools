"""Consensus NMF (cNMF) following Kotliar et al. 2019.

Runs NMF many times, clusters the resulting component spectra, filters
outliers by distance from the cluster centre, and takes the medoid of
each cluster as the consensus component.
"""

import warnings
import numpy as np

from .base import _FactorizationBase, _sanitize


class cNMF(_FactorizationBase):
    """Consensus Non-negative Matrix Factorisation.

    Parameters
    ----------
    data : ndarray, shape (n_samples, n_features)
        Non-negative input matrix.
    nc : int
        Default number of components (default 20).

    Attributes (populated after .fit())
    ------------------------------------
    loadings     : (nc, n_features)  – consensus H matrix
    coords       : (n_samples, nc)   – W matrix (solved against consensus H)
    expvars      : (nc,)
    frac_expvars : (nc,)
    total_var    : float
    all_H        : (n_iter * nc, n_features) – all L2-normalised H rows
    """

    _component_prefix = 'cNMF'

    def __init__(self, data, nc=20, name=None):
        super().__init__(data, nc, name=name)
        self.all_H = None

    def fit(
        self,
        nc=None,
        n_iter=100,
        max_nmf_iter=500,
        density_threshold=2.0,
        random_state=0,
        method='cd',
        **nmf_kwargs,
    ):
        """Fit consensus NMF to self.data.

        Parameters
        ----------
        nc : int, optional
            Number of components.  Overwrites self.nc.
        n_iter : int
            Number of independent NMF runs (default 100).
        max_nmf_iter : int
            Max iterations per NMF run.
        density_threshold : float
            Outlier filter: rows farther than ``density_threshold`` times the
            median distance from their cluster centre are excluded before
            computing the medoid (default 2.0).
        random_state : int
            Base random seed; run *i* uses ``random_state + i``.
        method : {'cd', 'mu'}
            Solver for each NMF run.
        **nmf_kwargs
            Forwarded to ``sklearn.decomposition.NMF``.
        """
        from sklearn.decomposition import NMF as _skNMF
        from sklearn.cluster import KMeans
        from sklearn.metrics import pairwise_distances

        if nc is not None:
            self.nc = nc

        X = _sanitize(self.data, label='data')
        if (X < 0).any():
            warnings.warn(
                'cNMF input contains negative values; clipping to 0.',
                UserWarning, stacklevel=2,
            )
            X = np.clip(X, 0, None)

        # -- Step 1: run NMF n_iter times, collect L2-normalised H rows ----
        all_H = []
        for i in range(n_iter):
            model = _skNMF(
                n_components=self.nc, solver=method,
                max_iter=max_nmf_iter, random_state=random_state + i,
                **nmf_kwargs,
            )
            model.fit(X)
            H = model.components_                         # (nc, p)
            norms = np.linalg.norm(H, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            all_H.append(H / norms)

        all_H_cat = np.vstack(all_H)                     # (n_iter*nc, p)
        self.all_H = all_H_cat

        # -- Step 2: k-means clustering into nc clusters -------------------
        km = KMeans(n_clusters=self.nc, n_init=10,
                    random_state=random_state)
        cluster_labels = km.fit_predict(all_H_cat)

        # -- Step 3: density-filter + medoid per cluster -------------------
        consensus_H = np.zeros((self.nc, X.shape[1]))
        for k in range(self.nc):
            mask = cluster_labels == k
            pts = all_H_cat[mask]
            centroid = pts.mean(axis=0)
            dists = np.linalg.norm(pts - centroid, axis=1)
            med = np.median(dists)
            keep = dists <= density_threshold * med if med > 0 else np.ones(len(dists), dtype=bool)
            kept = pts[keep]
            # medoid = point minimising total distance to all other kept points
            inner = pairwise_distances(kept)
            medoid_idx = inner.sum(axis=1).argmin()
            consensus_H[k] = kept[medoid_idx]

        self.loadings = consensus_H                       # (nc, p)

        # -- Step 4: solve for W using consensus H -------------------------
        final = _skNMF(
            n_components=self.nc, solver=method,
            max_iter=max_nmf_iter, random_state=random_state,
            **nmf_kwargs,
        )
        final.fit(X)
        final.components_ = consensus_H
        self.coords = final.transform(X)                  # W: (n, nc)
        self._sklearn_model = final
        self.mu = None
        self._compute_expvars()
        return self

    def transform(self, new_data, expvars_frac=False):
        """Project new data by solving for W_new against the consensus H.

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
