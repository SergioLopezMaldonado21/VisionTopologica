"""
PolynomialPatchGenerator - Genera parches a partir de polinomios parametrizados

Genera parches (m×m) a partir de un polinomio parametrizado por (theta, phi),
los centra (media cero) y los normaliza (por norma Euclídea o por D-norma).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from patches_tda.dnorm.dnorm_calculator import DNormCalculator

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


class PolynomialPatchGenerator:
    """
    Genera un parche (m×m) a partir de un polinomio parametrizado por (theta, phi),
    lo centra (media cero) y lo normaliza (por norma Euclídea o por D-norma).
    
    El polinomio es:
        p(x,y) = c*(a x + b y)² + d*(a x + b y)
    donde:
        (a, b) = (cos θ, sin θ)
        (c, d) = (cos φ, sin φ)
    
    Parameters
    ----------
    patch_size : int, default=3
        Tamaño del parche (m×m).
    usar_norma_D : bool, default=True
        Si True, normaliza por D-norma (contraste).
        Si False, normaliza por norma Euclídea.
    
    Examples
    --------
    >>> gen = PolynomialPatchGenerator(patch_size=3)
    >>> patch = gen.generate_patch(theta=0.5, phi=1.0)
    >>> patch.shape
    (3, 3)
    >>> gen.plot_patch(patch)
    """

    def __init__(self, patch_size: int = 3, usar_norma_D: bool = True) -> None:
        if patch_size < 2:
            raise ValueError("patch_size debe ser >= 2")

        self.patch_size = patch_size
        self.usar_norma_D = usar_norma_D
        self._dnorm = DNormCalculator(patch_size=patch_size)

        # Grilla fija: [-1,0,1] si m=3; en general puntos enteros centrados
        coords = np.arange(-(patch_size // 2), patch_size // 2 + 1)
        if coords.size != patch_size:
            # Por si patch_size es par, definimos una grilla simétrica "lo más parecida"
            coords = np.linspace(-(patch_size - 1) / 2, (patch_size - 1) / 2, patch_size)

        X, Y = np.meshgrid(coords, coords, indexing="ij")
        self._X_flat = X.flatten()
        self._Y_flat = Y.flatten()

    def polynomial_value(self, theta: float, phi: float, x: float, y: float) -> float:
        """
        Evalúa el polinomio en un punto (x, y).
        
        p(x,y) = c*(a x + b y)² + d*(a x + b y)
        donde (a,b)=(cos θ, sin θ), (c,d)=(cos φ, sin φ)
        """
        a, b = np.cos(theta), np.sin(theta)
        c, d = np.cos(phi), np.sin(phi)
        t = a * x + b * y
        return c * (t ** 2) + d * t

    def to_vector(self, theta: float, phi: float) -> np.ndarray:
        """
        Evalúa p(x,y) en la grilla y devuelve vector shape (patch_dim,).
        """
        vals = np.array(
            [self.polynomial_value(theta, phi, x, y) for x, y in zip(self._X_flat, self._Y_flat)],
            dtype=float
        )
        return vals

    @staticmethod
    def center(vec: np.ndarray) -> np.ndarray:
        """Resta la media para hacer el parche de media cero."""
        return vec - vec.mean()

    def normalize(self, vec_centered: np.ndarray) -> np.ndarray:
        """
        Normaliza el vector centrado.

        - Si usar_norma_D: divide por D-norma (contraste local).
        - Si no: divide por norma infinito (max abs), como en tu snippet.
        """
        v = np.asarray(vec_centered, dtype=float).reshape(1, -1)  # (1, patch_dim)

        if self.usar_norma_D:
            denom = float(self._dnorm.compute(v)[0])  # D-norma
            if denom > 0:
                return (v / denom).reshape(-1)
            return v.reshape(-1)

        # Caso NO D-norma: normalización por max abs (norma infinito)
        max_abs = float(np.max(np.abs(v)))
        if max_abs > 0:
            return (v / max_abs).reshape(-1)
        return v.reshape(-1)


    def generate_patch(self, theta: float, phi: float) -> np.ndarray:
        """
        Pipeline completo para generar un parche normalizado.
        
        1) Evalúa p en la grilla -> vector
        2) Centra -> media cero
        3) Normaliza (D o Euclídea)
        4) Reshape a (m, m)
        
        Parameters
        ----------
        theta : float
            Ángulo para la dirección (a, b) = (cos θ, sin θ).
        phi : float
            Ángulo para los coeficientes (c, d) = (cos φ, sin φ).
        
        Returns
        -------
        np.ndarray
            Parche de shape (patch_size, patch_size).
        """
        vec = self.to_vector(theta, phi)
        vec_centered = self.center(vec)
        vec_norm = self.normalize(vec_centered)
        return vec_norm.reshape(self.patch_size, self.patch_size)

    def plot_patch(
        self,
        patch: np.ndarray,
        ax: "plt.Axes | None" = None,
        cmap: str = "gray",
        show_colorbar: bool = True,
        title: str | None = None
    ) -> "plt.Axes":
        """
        Visualiza un parche como imagen.
        
        Parameters
        ----------
        patch : np.ndarray
            Parche de shape (m, m).
        ax : plt.Axes | None, optional
            Axes existente. Si None, crea uno nuevo.
        cmap : str, default="gray"
            Colormap a usar.
        show_colorbar : bool, default=True
            Si mostrar barra de color.
        title : str | None, optional
            Título de la figura.
        
        Returns
        -------
        plt.Axes
            El axes con la visualización.
        
        Examples
        --------
        >>> gen = PolynomialPatchGenerator(patch_size=5)
        >>> patch = gen.generate_patch(theta=np.pi/4, phi=np.pi/3)
        >>> gen.plot_patch(patch, title="Parche θ=π/4, φ=π/3")
        """
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))
        
        im = ax.imshow(patch, cmap=cmap, origin="upper")
        
        if show_colorbar:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        if title:
            ax.set_title(title)
        
        ax.set_xticks(range(self.patch_size))
        ax.set_yticks(range(self.patch_size))
        ax.set_xlabel("j")
        ax.set_ylabel("i")
        
        plt.tight_layout()
        plt.show()
        
        return ax

    def plot_patch_grid(
        self,
        n_theta: int = 4,
        n_phi: int = 4,
        figsize: tuple[int, int] = (12, 12),
        cmap: str = "gray"
    ) -> "plt.Figure":
        """
        Visualiza una grilla de parches variando theta y phi.
        
        Parameters
        ----------
        n_theta : int, default=4
            Número de valores de theta.
        n_phi : int, default=4
            Número de valores de phi.
        figsize : tuple, default=(12, 12)
            Tamaño de la figura.
        cmap : str, default="gray"
            Colormap a usar.
        
        Returns
        -------
        plt.Figure
            La figura con la grilla de parches.
        """
        import matplotlib.pyplot as plt
        
        thetas = np.linspace(0, np.pi, n_theta, endpoint=False)
        phis = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        
        fig, axes = plt.subplots(n_theta, n_phi, figsize=figsize)
        
        for i, theta in enumerate(thetas):
            for j, phi in enumerate(phis):
                patch = self.generate_patch(theta, phi)
                ax = axes[i, j] if n_theta > 1 and n_phi > 1 else axes
                ax.imshow(patch, cmap=cmap, origin="upper")
                ax.set_title(f"θ={theta:.2f}, φ={phi:.2f}", fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
        
        fig.suptitle(f"Parches polinomiales {self.patch_size}×{self.patch_size}", fontsize=12)
        plt.tight_layout()
        plt.show()
        
        return fig
