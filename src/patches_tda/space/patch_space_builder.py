"""
PatchSpaceBuilder - Construcción del patch space 𝓜

Acumula parches de múltiples imágenes hasta alcanzar el tamaño objetivo.
Soporta persistencia (guardar/cargar) para datasets grandes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class PatchSpaceBuilder:
    """
    Construye el patch space 𝓜 acumulando parches de múltiples imágenes.
    
    Parameters
    ----------
    target_size : int, default=4_000_000
        Tamaño objetivo del patch space (Z_target).
    patch_dim : int, default=9
        Dimensión de cada parche (m²).
    
    Attributes
    ----------
    current_size : int
        Número actual de parches acumulados.
    is_full : bool
        True si se alcanzó el target_size.
    
    Examples
    --------
    >>> builder = PatchSpaceBuilder(target_size=1_000_000)
    >>> for patches in batches:
    ...     builder.add_patches(patches)
    ...     if builder.is_full:
    ...         break
    >>> patch_space = builder.build()
    """
    
    def __init__(
        self,
        target_size: int = 4_000_000,
        patch_dim: int = 9
    ) -> None:
        if target_size < 1:
            raise ValueError(f"target_size debe ser >= 1, recibido: {target_size}")
        if patch_dim < 1:
            raise ValueError(f"patch_dim debe ser >= 1, recibido: {patch_dim}")
        
        self._target_size = target_size
        self._patch_dim = patch_dim
        self._patches_list: list[np.ndarray] = []
        self._current_size = 0
        
        logger.debug(
            "PatchSpaceBuilder inicializado: target_size=%d, patch_dim=%d",
            target_size, patch_dim
        )
    
    @property
    def target_size(self) -> int:
        """Tamaño objetivo del patch space."""
        return self._target_size
    
    @property
    def patch_dim(self) -> int:
        """Dimensión de cada parche."""
        return self._patch_dim
    
    @property
    def current_size(self) -> int:
        """Número actual de parches acumulados."""
        return self._current_size
    
    @property
    def is_full(self) -> bool:
        """True si se alcanzó el target_size."""
        return self._current_size >= self._target_size
    
    def add_patches(self, patches: np.ndarray) -> int:
        """
        Agrega parches al espacio.
        
        Si se excede el target_size, solo se agregan los necesarios.
        
        Parameters
        ----------
        patches : np.ndarray
            Array de parches de shape (N, patch_dim).
        
        Returns
        -------
        int
            Número de parches efectivamente agregados.
        
        Raises
        ------
        ValueError
            Si la dimensión del parche no coincide.
        """
        if patches.ndim != 2 or patches.shape[1] != self._patch_dim - 1:
            raise ValueError(
                f"Shape incorrecto: {patches.shape}, esperado (N, {self._patch_dim})"
            )
        
        if self.is_full:
            logger.debug("Patch space lleno, ignorando parches adicionales")
            return 0
        
        # Calcular cuántos parches podemos agregar
        remaining = self._target_size - self._current_size
        n_to_add = min(patches.shape[0], remaining)
        
        if n_to_add < patches.shape[0]:
            patches = patches[:n_to_add]
        
        self._patches_list.append(patches.copy())
        self._current_size += n_to_add
        
        logger.debug(
            "Agregados %d parches, total=%d/%d (%.1f%%)",
            n_to_add, self._current_size, self._target_size,
            100 * self._current_size / self._target_size
        )
        
        return n_to_add
    
    def build(self) -> np.ndarray:
        """
        Construye y retorna el patch space completo.
        
        Returns
        -------
        np.ndarray
            Patch space 𝓜 de shape (current_size, patch_dim).
        """
        if not self._patches_list:
            return np.empty((0, self._patch_dim), dtype=np.float64)
        
        patch_space = np.vstack(self._patches_list)
        
        logger.info(
            "Patch space construido: shape=%s, %.2f MB",
            patch_space.shape,
            patch_space.nbytes / (1024 * 1024)
        )
        
        return patch_space
    
    def reset(self) -> None:
        """Reinicia el builder."""
        self._patches_list.clear()
        self._current_size = 0
        logger.debug("PatchSpaceBuilder reiniciado")
    
    def save(self, path: Path | str) -> None:
        """
        Guarda el patch space a disco.
        
        Parameters
        ----------
        path : Path | str
            Ruta del archivo .npy.
        """
        patch_space = self.build()
        np.save(path, patch_space)
        logger.info("Patch space guardado en %s", path)

    def load(self, path: Path | str) -> None:
        """
        Carga un patch space desde disco.
        
        Parameters
        ----------
        path : Path | str
            Ruta del archivo .npy.
        """
        patch_space = np.load(path)
        
        if patch_space.ndim != 2 or patch_space.shape[1] != self._patch_dim-1:
            raise ValueError(
                f"Shape incorrecto en archivo: {patch_space.shape}, "
                f"esperado (N, {self._patch_dim})"
            )
        
        self.reset()
        self._patches_list.append(patch_space)
        self._current_size = patch_space.shape[0]
        
        logger.info("Patch space cargado desde %s: %d parches", path, self._current_size)