"""
Adapters - Adaptadores para generadores de parches

Define el protocolo PatchGenerator y el adaptador
para PolynomialPatchGenerator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from patches_tda.generators.polynomial_patch_generator import PolynomialPatchGenerator


class PatchGenerator(Protocol):
    """Protocolo para generadores de parches."""
    
    def generate(self, theta: float, phi: float) -> np.ndarray:
        """
        Genera un parche para los ángulos dados.
        
        Parameters
        ----------
        theta : float
            Ángulo θ.
        phi : float
            Ángulo φ.
        
        Returns
        -------
        np.ndarray
            Matriz del parche (m × m).
        """
        ...


class PatchGeneratorAdapter:
    """
    Adapta PolynomialPatchGenerator al protocolo PatchGenerator.
    
    Parameters
    ----------
    poly_gen : PolynomialPatchGenerator
        Generador de parches polinomiales.
    
    Examples
    --------
    >>> from patches_tda import PolynomialPatchGenerator
    >>> gen = PolynomialPatchGenerator(patch_size=3)
    >>> adapter = PatchGeneratorAdapter(gen)
    >>> patch = adapter.generate(0.5, 1.0)
    """
    
    def __init__(self, poly_gen: "PolynomialPatchGenerator") -> None:
        self._gen = poly_gen
    
    def generate(self, theta: float, phi: float) -> np.ndarray:
        """Genera un parche usando el generador subyacente."""
        return self._gen.generate_patch(theta, phi)
