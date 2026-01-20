"""
GlobalSubsampler - Submuestreo global del patch space para topología

Submuestrea aleatoriamente N puntos del patch space 𝓜 para
reducir el tamaño antes de aplicar análisis topológico.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.random import Generator

logger = logging.getLogger(__name__)


class GlobalSubsampler:
    """
    Submuestreo global aleatorio del patch space para topología.
    
    Parameters
    ----------
    n_samples : int, default=50_000
        Número de muestras a extraer (N).
    rng : np.random.Generator | None, optional
        Generador aleatorio para reproducibilidad.
    
    Examples
    --------
    >>> sampler = GlobalSubsampler(n_samples=50_000, rng=np.random.default_rng(42))
    >>> X = sampler.subsample(patch_space)  # patch_space: (4_000_000, 9)
    >>> X.shape
    (50000, 9)
    """
    
    def __init__(
        self,
        n_samples: int = 50_000,
        rng: Generator | None = None
    ) -> None:
        if n_samples < 1:
            raise ValueError(f"n_samples debe ser >= 1, recibido: {n_samples}")
        
        self._n_samples = n_samples
        self._rng = rng if rng is not None else np.random.default_rng()
        
        logger.debug("GlobalSubsampler inicializado: n_samples=%d", n_samples)
    
    @property
    def n_samples(self) -> int:
        """Número de muestras a extraer."""
        return self._n_samples
    
    def subsample(self, patch_space: np.ndarray) -> np.ndarray:
        """
        Submuestrea el patch space.
        
        Parameters
        ----------
        patch_space : np.ndarray
            Patch space 𝓜 de shape (M, patch_dim).
        
        Returns
        -------
        np.ndarray
            Submuestra X de shape (min(n_samples, M), patch_dim).
        """
        M = patch_space.shape[0]
        n = min(self._n_samples, M)
        
        if n == M:
            logger.warning(
                "Patch space (%d) menor o igual a n_samples (%d), "
                "retornando todo",
                M, self._n_samples
            )
            return patch_space.copy()
        
        # Muestreo sin reemplazo
        indices = self._rng.choice(M, size=n, replace=False)
        X = patch_space[indices]
        
        logger.debug(
            "Submuestreados %d/%d puntos (%.2f%%)",
            n, M, 100 * n / M
        )
        
        return X
