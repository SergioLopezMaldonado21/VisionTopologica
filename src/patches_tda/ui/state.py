"""
UIState - Estado de la interfaz de usuario

Dataclass que mantiene el estado completo de la UI,
incluyendo punto actual, modo y datos de trayectoria.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass
class UIState:
    """
    Estado de la interfaz de usuario.
    
    Attributes
    ----------
    theta : float
        Ángulo θ actual [0, 2π].
    phi : float
        Ángulo φ actual [0, 2π].
    mode : Literal["point", "trajectory"]
        Modo de interacción.
    use_dnorm : bool
        Si usar normalización por D-norm.
    traj_p1 : tuple[float, float] | None
        Primer punto de la trayectoria.
    traj_p2 : tuple[float, float] | None
        Segundo punto de la trayectoria.
    traj_n : int
        Número de puntos en la trayectoria.
    traj_index : int
        Índice del punto actual en la trayectoria.
    traj_points : list[tuple[float, float]] | None
        Lista de puntos calculados de la trayectoria.
    """
    theta: float = field(default_factory=lambda: np.pi)
    phi: float = field(default_factory=lambda: np.pi)
    mode: Literal["point", "trajectory"] = "point"
    use_dnorm: bool = True
    
    # Trayectoria
    traj_p1: tuple[float, float] | None = None
    traj_p2: tuple[float, float] | None = None
    traj_n: int = 20
    traj_index: int = 0
    traj_points: list[tuple[float, float]] | None = None
    
    def has_complete_trajectory(self) -> bool:
        """Retorna True si P1 y P2 están definidos."""
        return self.traj_p1 is not None and self.traj_p2 is not None
    
    def clear_trajectory(self) -> None:
        """Limpia los datos de trayectoria."""
        self.traj_p1 = None
        self.traj_p2 = None
        self.traj_points = None
        self.traj_index = 0
