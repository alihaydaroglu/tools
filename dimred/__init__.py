"""Dimensionality-reduction / matrix-factorisation modules."""

from .base import _FactorizationBase
from .pca import PCA
from .nmf import NMF
from .cnmf import cNMF
from .schpf import scHPF

__all__ = ["PCA", "NMF", "cNMF", "scHPF"]
