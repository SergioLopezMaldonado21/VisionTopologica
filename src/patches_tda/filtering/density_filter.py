"""
DensityFilter - Filtrado por densidad y denoising

Implementa el filtrado X(p,k) del paper:
1. Estimar densidad local usando k-NN
2. Conservar el top p% de puntos más densos
3. Aplicar denoising iterativo (promedio de k vecinos)
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.spatial import KDTree

logger = logging.getLogger(__name__)


class DensityFilter:
    """
    Filtrado por densidad y denoising para obtener X(p,k).
    
    Parameters
    ----------
    k_neighbors : int, default=15
        Número de vecinos para estimación de densidad y denoising.
    top_density_fraction : float, default=0.30
        Fracción de puntos más densos a conservar (p).
    denoise_iterations : int, default=2
        Número de iteraciones de denoising.
    
    Examples
    --------
    >>> filt = DensityFilter(k_neighbors=15, top_density_fraction=0.30)
    >>> X_pk = filt.transform(X)  # X: (50000, 9)
    >>> X_pk.shape  # Aproximadamente (15000, 9)
    """
    
    def __init__(
        self,
        k_neighbors: int = 15,
        top_density_fraction: float = 0.30,
        denoise_iterations: int = 2
    ) -> None:
        if k_neighbors < 1:
            raise ValueError(f"k_neighbors debe ser >= 1, recibido: {k_neighbors}")
        if not 0.0 < top_density_fraction <= 1.0:
            raise ValueError(
                f"top_density_fraction debe estar en (0, 1], recibido: {top_density_fraction}"
            )
        if denoise_iterations < 0:
            raise ValueError(
                f"denoise_iterations debe ser >= 0, recibido: {denoise_iterations}"
            )
        
        self._k_neighbors = k_neighbors
        self._top_density_fraction = top_density_fraction
        self._denoise_iterations = denoise_iterations
        self._last_densities: np.ndarray | None = None
        
        logger.debug(
            "DensityFilter inicializado: k=%d, p=%.2f, iterations=%d",
            k_neighbors, top_density_fraction, denoise_iterations
        )
    
    @property
    def k_neighbors(self) -> int:
        """Número de vecinos."""
        return self._k_neighbors
    
    @property
    def top_density_fraction(self) -> float:
        """Fracción de puntos a conservar."""
        return self._top_density_fraction
    
    @property
    def denoise_iterations(self) -> int:
        """Número de iteraciones de denoising."""
        return self._denoise_iterations
    
    @property
    def last_densities(self) -> np.ndarray | None:
        """Densidades de la última estimación (para diagnóstico)."""
        return self._last_densities
    
    def estimate_density(self, X: np.ndarray) -> np.ndarray:
        """
        Estima densidad local usando distancia al k-ésimo vecino.
        
        Densidad ∝ 1 / distancia_al_k_vecino
        
        Parameters
        ----------
        X : np.ndarray
            Datos de shape (N, D).
        
        Returns
        -------
        np.ndarray
            Densidades de shape (N,).
        """
        n_points = X.shape[0]
        k = min(self._k_neighbors, n_points - 1)
        
        if k < 1:
            logger.warning("Muy pocos puntos para estimar densidad")
            return np.ones(n_points)
        
        # Construir KDTree para búsqueda eficiente
        tree = KDTree(X)
        
        # Buscar k+1 vecinos (incluye el punto mismo)
        distances, _ = tree.query(X, k=k + 1)
        
        # Distancia al k-ésimo vecino (última columna)
        kth_distances = distances[:, -1]
        
        # Evitar división por cero
        kth_distances = np.maximum(kth_distances, 1e-10)
        
        # Densidad inversamente proporcional a la distancia
        densities = 1.0 / kth_distances
        
        self._last_densities = densities.copy()
        
        return densities
    
    def filter_by_density(
        self, 
        X: np.ndarray, 
        densities: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Conserva el top p% de puntos más densos.
        
        Parameters
        ----------
        X : np.ndarray
            Datos de shape (N, D).
        densities : np.ndarray | None, optional
            Densidades precalculadas. Si None, se calculan.
        
        Returns
        -------
        np.ndarray
            Datos filtrados de shape (k, D) donde k = ceil(N * p).
        """
        if densities is None:
            densities = self.estimate_density(X)
        
        n_points = X.shape[0]
        k = max(1, int(np.ceil(n_points * self._top_density_fraction)))
        
        # Obtener índices del top-k usando argpartition
        partition_idx = np.argpartition(densities, -k)[-k:]
        
        X_filtered = X[partition_idx]
        
        logger.debug(
            "Filtrados %d/%d puntos (top %.1f%% por densidad)",
            k, n_points, self._top_density_fraction * 100
        )
        
        return X_filtered
    
    def denoise(self, X: np.ndarray) -> np.ndarray:
        """
        Aplica denoising: reemplaza cada punto por el promedio de sus k vecinos.
        
        Parameters
        ----------
        X : np.ndarray
            Datos de shape (N, D).
        
        Returns
        -------
        np.ndarray
            Datos denoised de shape (N, D).
        """
        if self._denoise_iterations == 0:
            return X.copy()
        
        n_points = X.shape[0]
        k = min(self._k_neighbors, n_points)
        
        X_denoised = X.copy()
        
        for iteration in range(self._denoise_iterations):
            tree = KDTree(X_denoised)
            _, indices = tree.query(X_denoised, k=k)
            
            # Promedio de los k vecinos
            X_denoised = np.mean(X_denoised[indices], axis=1)
            
            logger.debug("Denoising iteración %d/%d", iteration + 1, self._denoise_iterations)
        
        return X_denoised
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Ejecuta filtrado por densidad + denoising completo → X(p,k).
        
        Parameters
        ----------
        X : np.ndarray
            Datos de shape (N, D).
        
        Returns
        -------
        np.ndarray
            Datos transformados X(p,k).
        """
        logger.info("Iniciando DensityFilter.transform() con %d puntos", X.shape[0])
        
        # 1. Estimar densidad
        densities = self.estimate_density(X)
        
        # 2. Filtrar por densidad
        X_filtered = self.filter_by_density(X, densities)
        
        # 3. Denoising
        X_pk = self.denoise(X_filtered)
        
        logger.info(
            "DensityFilter completado: %d → %d puntos",
            X.shape[0], X_pk.shape[0]
        )
        
        return X_pk
