"""
DNormCalculator - Cálculo de la D-norm (norma de contraste)

La D-norm mide el contraste de un parche según el paper de Carlsson et al.
Se define como ‖x‖_D = √(xᵀ D x) donde D es una matriz que penaliza
diferencias entre píxeles adyacentes.
"""

from __future__ import annotations
import numpy as np
import logging

logger = logging.getLogger(__name__)


class DNormCalculator:
    """
    Calcula la D-norm (norma de contraste) para parches vectorizados.

    MISMA API que la versión original, pero implementada de forma eficiente:
    ||x||_D^2 = sum (p_ij - p_kl)^2 sobre píxeles adyacentes.
    """

    def __init__(self, patch_size: int = 3) -> None:
        if patch_size < 2:
            raise ValueError(f"patch_size debe ser >= 2, recibido: {patch_size}")

        self._patch_size = patch_size
        self._patch_dim = patch_size ** 2

        # Se conserva el atributo por compatibilidad
        self._D_matrix = None

        logger.debug(
            "DNormCalculator inicializado (modo eficiente): patch_size=%d",
            patch_size
        )

    @property
    def patch_size(self) -> int:
        return self._patch_size

    @property
    def D_matrix(self) -> np.ndarray:
        """
        Se mantiene por compatibilidad.
        No se construye explícitamente porque no es necesario.
        """
        raise RuntimeError(
            "D_matrix no se construye explícitamente en la versión eficiente. "
            "La D-norm se calcula por diferencias locales."
        )

    def compute(self, patches: np.ndarray) -> np.ndarray:
        """
        Calcula la D-norm para cada parche.

        Parameters
        ----------
        patches : np.ndarray
            Array de parches de shape (N, patch_dim) o (patch_dim,).

        Returns
        -------
        np.ndarray
            Array de D-norms de shape (N,) o escalar.
        """
        patches = np.asarray(patches)
        patches = np.atleast_2d(patches)

        if patches.shape[1] != self._patch_dim:
            raise ValueError(
                f"Dimensión de parche incorrecta: {patches.shape[1]} "
                f"(esperado: {self._patch_dim})"
            )

        m = self._patch_size

        # Reinterpretar como imágenes m×m (sin copia)
        P = patches.reshape(-1, m, m)

        # Diferencias horizontales y verticales (4-neighbors)
        dh = P[:, :, 1:] - P[:, :, :-1]   # (N, m, m-1)
        dv = P[:, 1:, :] - P[:, :-1, :]   # (N, m-1, m)

        # ||x||_D^2
        dnorm_squared = (
            np.sum(dh * dh, axis=(1, 2)) +
            np.sum(dv * dv, axis=(1, 2))
        )

        # Seguridad numérica
        dnorm_squared = np.maximum(dnorm_squared, 0.0)

        return np.sqrt(dnorm_squared)
