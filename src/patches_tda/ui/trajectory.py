"""
Trajectory - Trayectorias en el espacio paramétrico

Define el protocolo Trajectory y la implementación
LineSegmentTrajectory para interpolación lineal.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Trajectory(Protocol):
    """Protocolo para trayectorias en el espacio (θ, φ)."""
    
    def points(self, n: int) -> list[tuple[float, float]]:
        """
        Retorna n puntos a lo largo de la trayectoria.
        
        Parameters
        ----------
        n : int
            Número de puntos (≥ 2).
        
        Returns
        -------
        list[tuple[float, float]]
            Lista de (theta, phi) en orden.
        """
        ...


class LineSegmentTrajectory:
    """
    Trayectoria recta entre dos puntos.
    
    Interpola linealmente: p_i = (1 - t_i) * p1 + t_i * p2
    donde t_i = i / (n - 1)
    
    Parameters
    ----------
    p1 : tuple[float, float]
        Punto inicial (theta1, phi1).
    p2 : tuple[float, float]
        Punto final (theta2, phi2).
    
    Examples
    --------
    >>> traj = LineSegmentTrajectory((0, 0), (np.pi, np.pi))
    >>> pts = traj.points(5)
    >>> len(pts)
    5
    """
    
    def __init__(self, p1: tuple[float, float], p2: tuple[float, float]) -> None:
        self.p1 = p1
        self.p2 = p2
    
    def points(self, n: int) -> list[tuple[float, float]]:
        """
        Genera n puntos sobre el segmento.
        
        Parameters
        ----------
        n : int
            Número de puntos.
        
        Returns
        -------
        list[tuple[float, float]]
            Puntos interpolados.
        """
        if n <= 1:
            return [self.p1]
        
        t = np.linspace(0.0, 1.0, n)
        p1 = np.array(self.p1, dtype=float)
        p2 = np.array(self.p2, dtype=float)
        
        # Interpolación vectorizada
        pts = (1 - t)[:, None] * p1[None, :] + t[:, None] * p2[None, :]
        
        return [(float(a), float(b)) for a, b in pts]
