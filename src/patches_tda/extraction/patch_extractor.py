"""
PatchExtractor - Extracción de parches aleatorios de imágenes

Extrae parches cuadrados m×m de una imagen, aplica log-intensidad
y centrado, y los vectoriza en orden row-major.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.random import Generator

logger = logging.getLogger(__name__)


class PatchExtractor:
    """
    Extrae parches aleatorios de una imagen.
    
    Pipeline de extracción:
    1. Muestrear posiciones aleatorias (evitando bordes)
    2. Extraer parches m×m
    3. Aplicar log(1 + x) para log-intensidad
    4. Centrar (restar promedio de cada parche)
    5. Vectorizar en row-major → ℝ^{m²}
    
    Parameters
    ----------
    patch_size : int, default=3
        Tamaño del parche (m×m).
    n_patches : int, default=5000
        Número de parches a extraer por imagen.
    rng : np.random.Generator | None, optional
        Generador aleatorio para reproducibilidad.
    
    Examples
    --------
    >>> extractor = PatchExtractor(patch_size=3, n_patches=1000, rng=np.random.default_rng(42))
    >>> patches = extractor.extract(image)
    >>> patches.shape
    (1000, 9)
    """
    
    def __init__(
        self,
        patch_size: int = 3,
        n_patches: int = 5000,
        rng: Generator | None = None
    ) -> None:
        if patch_size < 1:
            raise ValueError(f"patch_size debe ser >= 1, recibido: {patch_size}")
        if n_patches < 1:
            raise ValueError(f"n_patches debe ser >= 1, recibido: {n_patches}")
        
        self._patch_size = patch_size
        self._n_patches = n_patches
        self._rng = rng if rng is not None else np.random.default_rng()
        
        logger.debug(
            "PatchExtractor inicializado: patch_size=%d, n_patches=%d",
            patch_size, n_patches
        )
    
    @property
    def patch_size(self) -> int:
        """Tamaño del parche (m)."""
        return self._patch_size
    
    @property
    def n_patches(self) -> int:
        """Número de parches a extraer."""
        return self._n_patches
    
    @property
    def patch_dim(self) -> int:
        """Dimensión del vector de parche (m²)."""
        return self._patch_size ** 2
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extrae parches de una imagen.
        
        Parameters
        ----------
        image : np.ndarray
            Imagen 2D de shape (H, W).
        
        Returns
        -------
        np.ndarray
            Array de parches procesados de shape (n_patches, patch_size²).
        
        Raises
        ------
        ValueError
            Si la imagen es muy pequeña para extraer parches.
        """
        # Validar dimensiones
        if image.ndim != 2:
            raise ValueError(f"Imagen debe ser 2D, recibido ndim={image.ndim}")
        
        height, width = image.shape
        m = self._patch_size
        
        # Rango válido para posiciones de parches (esquina superior izquierda)
        max_row = height - m
        max_col = width - m
        
        if max_row < 0 or max_col < 0:
            raise ValueError(
                f"Imagen muy pequeña ({height}×{width}) para parches de {m}×{m}"
            )
        
        # 1. Muestrear posiciones aleatorias
        rows = self._rng.integers(0, max_row + 1, size=self._n_patches)
        cols = self._rng.integers(0, max_col + 1, size=self._n_patches)
        
        # 2. Extraer parches
        patches = np.empty((self._n_patches, m, m), dtype=np.float64)
        for i, (r, c) in enumerate(zip(rows, cols)):
            patches[i] = image[r:r+m, c:c+m]
        
        # 3. Aplicar log-intensidad: log(1 + x)
        # Importante: esto maneja valores cero correctamente
        patches = np.log1p(patches)
        
        # 4. Vectorizar en row-major
        patches = patches.reshape(self._n_patches, -1)
        
        # 5. Centrar cada parche (restar su promedio)
        means = patches.mean(axis=1, keepdims=True)
        patches = patches - means
       

        logger.debug(
            "Extraídos %d parches de imagen %s",
            self._n_patches, image.shape
        )
        
        return patches
