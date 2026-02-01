"""
TopDNormSelector - Selección de parches por contraste (D-norm)

Selecciona el top q% de parches con mayor D-norm (contraste).
Este paso se ejecuta por imagen antes de acumular en el patch space.
"""

from __future__ import annotations

import logging

import numpy as np

from patches_tda.dnorm.dnorm_calculator import DNormCalculator

logger = logging.getLogger(__name__)


class TopDNormSelector:
    """
    Selecciona el top q% de parches por D-norm (contraste).
    
    Este filtrado se aplica **por imagen**, no globalmente.
    
    Parameters
    ----------
    dnorm_calculator : DNormCalculator
        Calculador de D-norm.
    top_fraction : float, default=0.20
        Fracción de parches a conservar (e.g., 0.20 = top 20%).
    
    Examples
    --------
    >>> calc = DNormCalculator(patch_size=3)
    >>> selector = TopDNormSelector(calc, top_fraction=0.20)
    >>> selected = selector.select(patches)  # patches: (5000, 9)
    >>> selected.shape
    (1000, 9)
    """
    
    def __init__(
        self,
        dnorm_calculator: DNormCalculator,
        top_fraction: float = 0.20
    ) -> None:
        if not 0.0 < top_fraction <= 1.0:
            raise ValueError(
                f"top_fraction debe estar en (0, 1], recibido: {top_fraction}"
            )
        
        self._dnorm_calculator = dnorm_calculator
        self._top_fraction = top_fraction
        self._last_dnorms: np.ndarray | None = None
        
        logger.debug(
            "TopDNormSelector inicializado: top_fraction=%.2f",
            top_fraction
        )
    
    @property
    def top_fraction(self) -> float:
        """Fracción de parches a conservar."""
        return self._top_fraction
    
    @property
    def last_dnorms(self) -> np.ndarray | None:
        """D-norms de la última selección (para diagnóstico)."""
        return self._last_dnorms
        
    def select(self, patches: np.ndarray) -> np.ndarray:
        """
        Selecciona el top q% de parches con mayor D-norm
        y los normaliza dividiendo por su D-norm.

        Implementa (3c)–(4)–(5) del paper.
        """
        n_patches = patches.shape[0]
        k = max(1, int(np.ceil(n_patches * self._top_fraction)))

        # 1) Calcular D-norm
        dnorms = self._dnorm_calculator.compute(patches)
        self._last_dnorms = dnorms.copy()

        # 2) Índices del top-k
        idx = np.argpartition(dnorms, -k)[-k:]

        # 3) Ordenar descendente (opcional)
        idx = idx[np.argsort(dnorms[idx])[::-1]]

        selected_patches = patches[idx]
        selected_dnorms = dnorms[idx]

        # 4) Normalizar por D-norma (paso del paper)
        selected_dnorms = np.maximum(selected_dnorms, 1e-12)
        selected_patches = selected_patches / selected_dnorms[:, None]

        # 5) Cambio de base a R^8 SOLO si patch_size=3
        if self._dnorm_calculator.patch_size == 3:
            selected_patches = self.dct_change_of_basis(selected_patches)

        logger.debug(
            "Seleccionados %d/%d parches (top %.1f%%), "
            "D-norm antes de normalizar: min=%.4f, max=%.4f",
            k, n_patches, self._top_fraction * 100,
            selected_dnorms.min(), selected_dnorms.max()
        )

        return selected_patches

    def dct_change_of_basis(self,selected_patches):
        """
        selected_patches: (N, 9)  parches vectorizados 3x3 (orden fila/row-major típico)
        selected_dnorms:  (N,)    normas-D de cada parche (para normalizar como en tu snippet)
        retorna:
        Y: (N, 9) parches normalizados
        V: (N, 8) coordenadas en la base DCT no-constante (v = Λ A^T y)
        """


        # 1) Construir A = [e1 ... e8] (cada e_i ya viene con su factor 1/sqrt(...)
        e1 = (1/np.sqrt(6))  * np.array([1, 0, -1, 1, 0, -1, 1, 0, -1], dtype=np.float64)
        e2 = (1/np.sqrt(6))  * np.array([1, 1,  1, 0, 0,  0,-1,-1, -1], dtype=np.float64)
        e3 = (1/np.sqrt(54)) * np.array([1,-2,  1, 1,-2,  1, 1,-2,  1], dtype=np.float64)
        e4 = (1/np.sqrt(54)) * np.array([1, 1,  1,-2,-2, -2, 1, 1,  1], dtype=np.float64)
        e5 = (1/np.sqrt(8))  * np.array([1, 0, -1, 0, 0,  0,-1, 0,  1], dtype=np.float64)
        e6 = (1/np.sqrt(48)) * np.array([1, 0, -1,-2, 0,  2, 1, 0, -1], dtype=np.float64)
        e7 = (1/np.sqrt(48)) * np.array([1,-2,  1, 0, 0,  0,-1, 2, -1], dtype=np.float64)
        e8 = (1/np.sqrt(216))* np.array([1,-2,  1,-2, 4, -2, 1,-2,  1], dtype=np.float64)

        A = np.column_stack([e1,e2,e3,e4,e5,e6,e7,e8])  # (9, 8)

        # 2) Λ diagonal con 1/||e_i||^2 (norma euclídea en R^9)
        col_norm_sq = np.sum(A*A, axis=0)               # (8,)
        lam = 1.0 / col_norm_sq                         # (8,)

        # 3) v = Λ A^T y, vectorizado para N parches:
        #    V = Y A Λ  (donde Λ actúa por columna)
        V = (selected_patches @ A) * lam[None, :]                      # (N, 8)

        return V
