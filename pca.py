"""Backward-compatible re-export — canonical location is tools.dimred.pca."""

from .dimred.pca import PCA
from .dimred.base import _sanitize, _resolve_series, _cycle_colors

__all__ = ['PCA']
