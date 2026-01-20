"""
WitnessTDA - Análisis Topológico de Datos mediante Witness Complex

Implementa el flujo completo para calcular homología persistente
usando EuclideanWitnessComplex de GUDHI.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from numpy.random import Generator
    import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class WitnessTDA:
    """
    Análisis Topológico de Datos mediante Witness Complex.
    
    API fluent para uso en notebooks:
        tda.load().sample().compute()
    
    Parameters
    ----------
    data_path : Path | str
        Ruta al archivo .npy/.npk con la nube de puntos.
    m_points : int | None, default=None
        Puntos a muestrear. None = usar todos.
    n_landmarks : int, default=300
        Número de landmarks para el Witness Complex.
    n_witnesses : int, default=1000
        Número de witnesses para el Witness Complex.
    max_alpha_square : float, default=0.05
        Parámetro alpha² máximo para el complejo.
    limit_dimension : int, default=2
        Dimensión máxima de los símplices.
    seed : int, default=42
        Semilla para reproducibilidad.
    
    Examples
    --------
    >>> tda = (
    ...     WitnessTDA("datos.npy", n_landmarks=200)
    ...     .load()
    ...     .sample()
    ...     .compute()
    ... )
    >>> tda.plot_projection(method="pca")
    >>> tda.plot_persistence()
    """
    
    def __init__(
        self,
        data_path: Path | str,
        m_points: int | None = None,
        n_landmarks: int = 300,
        n_witnesses: int = 1000,
        max_alpha_square: float = 0.05,
        limit_dimension: int = 2,
        seed: int = 42
    ) -> None:
        self._data_path = Path(data_path)
        self._m_points = m_points
        self._n_landmarks = n_landmarks
        self._n_witnesses = n_witnesses
        self._max_alpha_square = max_alpha_square
        self._limit_dimension = limit_dimension
        self._rng: Generator = np.random.default_rng(seed)
        
        # Estado interno
        self.X_: np.ndarray | None = None
        self.points_: np.ndarray | None = None
        self.landmarks_: np.ndarray | None = None
        self.witnesses_: np.ndarray | None = None
        self.simplex_tree_ = None
        self.persistence_: list | None = None
        self.proj_2d_: np.ndarray | None = None
        
        logger.debug(
            "WitnessTDA inicializado: path=%s, n_landmarks=%d, n_witnesses=%d",
            self._data_path, n_landmarks, n_witnesses
        )
    
    def load(self) -> "WitnessTDA":
        """
        Carga el archivo de datos.
        
        Valida:
        - Extensión .npy o .npk
        - Array 2D
        - Valores finitos
        
        Returns
        -------
        self
            Para encadenamiento fluent.
        """
        # Verificar extensión
        suffix = self._data_path.suffix.lower()
        if suffix not in (".npy", ".npk"):
            raise ValueError(
                f"Extensión no soportada: {suffix}. Use .npy o .npk"
            )
        
        if not self._data_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {self._data_path}")
        
        # Cargar datos
        self.X_ = np.load(self._data_path)
        
        # Validar dimensiones
        if self.X_.ndim != 2:
            raise ValueError(
                f"Se esperaba array 2D, recibido ndim={self.X_.ndim}"
            )
        
        # Validar valores finitos
        if not np.isfinite(self.X_).all():
            raise ValueError("El array contiene NaN o infinitos")
        
        # Inicializar points_ con todos los datos
        self.points_ = self.X_.copy()
        
        logger.info(
            "Datos cargados: shape=%s, rango=[%.3f, %.3f]",
            self.X_.shape, self.X_.min(), self.X_.max()
        )
        
        return self
    
    def sample(self) -> "WitnessTDA":
        """
        Submuestrea m_points puntos sin reemplazo.
        
        Si m_points es None, usa todos los puntos.
        
        Returns
        -------
        self
            Para encadenamiento fluent.
        """
        if self.X_ is None:
            raise RuntimeError("Primero debe llamar a load()")
        
        N = self.X_.shape[0]
        
        if self._m_points is None or self._m_points >= N:
            self.points_ = self.X_.copy()
            logger.info("Usando todos los %d puntos", N)
        else:
            indices = self._rng.choice(N, size=self._m_points, replace=False)
            self.points_ = self.X_[indices]
            logger.info("Submuestreados %d/%d puntos", self._m_points, N)
        
        return self
    
    def compute(self) -> "WitnessTDA":
        """
        Construye el Witness Complex y calcula homología persistente.
        
        Returns
        -------
        self
            Para encadenamiento fluent.
        """
        try:
            import gudhi
        except ImportError as e:
            raise ImportError(
                "GUDHI no está instalado. Ejecute: pip install gudhi"
            ) from e
        
        if self.points_ is None:
            raise RuntimeError("Primero debe llamar a load() y sample()")
        
        N = self.points_.shape[0]
        n_landmarks = min(self._n_landmarks, N)
        n_witnesses = min(self._n_witnesses, N)
        
        logger.info(
            "Construyendo Witness Complex: landmarks=%d, witnesses=%d",
            n_landmarks, n_witnesses
        )
        
        # Seleccionar landmarks y witnesses
        idx_landmarks = self._rng.choice(N, size=n_landmarks, replace=False)
        idx_witnesses = self._rng.choice(N, size=n_witnesses, replace=False)
        
        self.landmarks_ = self.points_[idx_landmarks]
        self.witnesses_ = self.points_[idx_witnesses]
        
        # Construir Witness Complex
        witness_complex = gudhi.EuclideanWitnessComplex(
            witnesses=self.witnesses_.tolist(),
            landmarks=self.landmarks_.tolist()
        )
        
        # Crear SimplexTree
        self.simplex_tree_ = witness_complex.create_simplex_tree(
            max_alpha_square=self._max_alpha_square,
            limit_dimension=self._limit_dimension
        )
        
        # Calcular persistencia
        self.persistence_ = self.simplex_tree_.persistence()
        
        n_simplices = self.simplex_tree_.num_simplices()
        logger.info(
            "Witness Complex construido: %d símplices, %d intervalos de persistencia",
            n_simplices, len(self.persistence_)
        )
        
        return self
    
    def plot_projection(
        self,
        method: str = "pca",
        coords: tuple[int, int] = (0, 1),
        ax: "plt.Axes | None" = None,
        **scatter_kwargs
    ) -> "plt.Axes":
        """
        Proyecta y visualiza los puntos en 2D.
        
        Parameters
        ----------
        method : str, default="pca"
            Método de proyección: "pca" o "coords".
        coords : tuple[int, int], default=(0, 1)
            Coordenadas a usar si method="coords".
        ax : plt.Axes | None, optional
            Axes existente. Si None, crea uno nuevo.
        **scatter_kwargs
            Argumentos adicionales para scatter.
        
        Returns
        -------
        plt.Axes
            El axes con el scatter plot.
        """
        import matplotlib.pyplot as plt
        
        if self.points_ is None:
            raise RuntimeError("Primero debe llamar a load()")
        
        # Proyectar
        if method == "pca":
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            self.proj_2d_ = pca.fit_transform(self.points_)
            title = "Proyección PCA"
            explained = pca.explained_variance_ratio_
            xlabel = f"PC1 ({explained[0]*100:.1f}%)"
            ylabel = f"PC2 ({explained[1]*100:.1f}%)"
        elif method == "coords":
            i, j = coords
            self.proj_2d_ = self.points_[:, [i, j]]
            title = f"Coordenadas ({i}, {j})"
            xlabel = f"Dim {i}"
            ylabel = f"Dim {j}"
        else:
            raise ValueError(f"Método no soportado: {method}")
        
        # Crear figura si no existe
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        # Valores por defecto para scatter
        defaults = {"s": 1, "alpha": 0.5, "c": "steelblue"}
        defaults.update(scatter_kwargs)
        
        ax.scatter(self.proj_2d_[:, 0], self.proj_2d_[:, 1], **defaults)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        
        plt.tight_layout()
        plt.show()
        
        return ax
    
    def plot_persistence(
        self,
        plot_fn: Callable | None = None,
        ax: "plt.Axes | None" = None
    ) -> "plt.Axes":
        """
        Visualiza el diagrama de persistencia.
        
        Parameters
        ----------
        plot_fn : Callable | None, optional
            Función personalizada para graficar. Si None, usa gudhi.
        ax : plt.Axes | None, optional
            Axes existente. Si None, crea uno nuevo.
        
        Returns
        -------
        plt.Axes
            El axes con el diagrama.
        """
        import matplotlib.pyplot as plt
        
        if self.persistence_ is None:
            raise RuntimeError("Primero debe llamar a compute()")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        if plot_fn is not None:
            # Usar función personalizada
            plot_fn(self.persistence_, ax=ax)
        else:
            # Usar función de GUDHI
            try:
                import gudhi
                gudhi.plot_persistence_diagram(self.persistence_, axes=ax)
            except Exception:
                # Fallback: graficar manualmente
                self._plot_persistence_manual(ax)
        
        plt.tight_layout()
        plt.show()
        
        return ax
    
    def _plot_persistence_manual(self, ax: "plt.Axes") -> None:
        """Grafica diagrama de persistencia manualmente."""
        colors = {0: "red", 1: "blue", 2: "green"}
        
        for dim, (birth, death) in self.persistence_:
            if death == float("inf"):
                death = birth + 0.1  # Representar infinito
                marker = "^"
            else:
                marker = "o"
            
            color = colors.get(dim, "gray")
            ax.scatter(birth, death, c=color, marker=marker, s=20, alpha=0.7)
        
        # Línea diagonal
        lims = ax.get_xlim()
        ax.plot(lims, lims, "k--", alpha=0.3)
        
        ax.set_xlabel("Birth")
        ax.set_ylabel("Death")
        ax.set_title("Diagrama de Persistencia")
        
        # Leyenda
        for dim, color in colors.items():
            ax.scatter([], [], c=color, label=f"H{dim}")
        ax.legend()

    def plot_barcodes(
        self,
        max_dims: tuple[int, ...] = (0, 1, 2),
        ax: "plt.Axes | None" = None
    ) -> "plt.Axes":
        """
        Dibuja barras (barcode) de persistencia.

        Parameters
        ----------
        max_dims : tuple[int, ...]
            Dimensiones a graficar (ej. (1,) para solo H1).
        ax : plt.Axes | None
            Axes existente. Si None, crea uno nuevo.

        Returns
        -------
        plt.Axes
            Axes con las barras.
        """
        import matplotlib.pyplot as plt

        if self.persistence_ is None:
            raise RuntimeError("Primero debe llamar a compute()")

        # agrupar por dimensión
        by_dim = {}
        births_all = []
        finite_deaths = []

        for dim, (birth, death) in self.persistence_:
            if dim not in max_dims:
                continue
            by_dim.setdefault(dim, []).append((birth, death))
            births_all.append(birth)
            if death != float("inf") and np.isfinite(death):
                finite_deaths.append(death)

        if not by_dim:
            raise RuntimeError(f"No hay intervalos en dimensiones {max_dims}")

        max_birth = max(births_all) if births_all else 1.0
        max_finite = max(finite_deaths) if finite_deaths else max_birth
        inf_cap = max(max_birth, max_finite) * 1.1 if max(max_birth, max_finite) > 0 else 1.0

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))

        color_map = {0: "red", 1: "green", 2: "blue", 3: "purple"}

        y = 0
        for dim in sorted(by_dim.keys()):
            pairs = sorted(by_dim[dim], key=lambda t: t[0])
            for (birth, death) in pairs:
                dplot = death if (death != float("inf") and np.isfinite(death)) else inf_cap
                ax.plot([birth, dplot], [y, y], lw=2, c=color_map.get(dim, "gray"))
                y += 1

        ax.set_title("Barras de Persistencia (Barcode)")
        ax.set_xlabel("Parámetro")
        ax.set_ylabel("Intervalos")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
        return ax
