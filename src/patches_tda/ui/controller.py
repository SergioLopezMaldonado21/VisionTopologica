"""
UIController - Controlador de la interfaz

Mantiene el estado y maneja todos los eventos de la UI.
Conecta la vista con el generador de parches.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import numpy as np

from patches_tda.ui.state import UIState
from patches_tda.ui.trajectory import LineSegmentTrajectory

if TYPE_CHECKING:
    from patches_tda.ui.adapters import PatchGenerator
    from patches_tda.ui.view import PlotView


class UIController:
    """
    Controlador de la UI.
    
    Mantiene el estado (UIState) y maneja eventos.
    """
    
    def __init__(self, generator: "PatchGenerator", view: "PlotView") -> None:
        self.generator = generator
        self.view = view
        self.state = UIState()
        self._traj_patches_cache: List[np.ndarray] = []
    
    def _recompute_trajectory(self) -> None:
        """Recalcula los puntos de la trayectoria y genera parches."""
        if self.state.has_complete_trajectory():
            traj = LineSegmentTrajectory(self.state.traj_p1, self.state.traj_p2)
            self.state.traj_points = traj.points(self.state.traj_n)
            self.state.traj_index = min(
                self.state.traj_index, 
                len(self.state.traj_points) - 1
            )
            
            # Generar parches para toda la trayectoria
            self._traj_patches_cache = [
                self.generator.generate(th, ph) 
                for th, ph in self.state.traj_points
            ]
            
            # Actualizar theta/phi al punto actual
            th, ph = self.state.traj_points[self.state.traj_index]
            self.state.theta, self.state.phi = th, ph
        else:
            self.state.traj_points = None
            self.state.traj_index = 0
            self._traj_patches_cache = []
    
    def set_mode(self, mode: str) -> None:
        """Cambia el modo de interacción."""
        self.state.mode = "trajectory" if mode == "trajectory" else "point"
        self.update()
    
    def set_theta_phi(self, theta: float, phi: float) -> None:
        """Establece los ángulos actuales."""
        # θ: -π/2 a 3π/2, φ: 0 a 2π
        self.state.theta = float(np.clip(theta, -np.pi/2, 3*np.pi/2))
        self.state.phi = float(np.clip(phi, 0, 2 * np.pi))
        self.update()
    
    def set_p1(self, theta: float, phi: float) -> None:
        """Establece P1 de la trayectoria."""
        self.state.traj_p1 = (
            float(np.clip(theta, -np.pi/2, 3*np.pi/2)),
            float(np.clip(phi, 0, 2 * np.pi))
        )
        self._recompute_trajectory()
        self.update()
    
    def set_p2(self, theta: float, phi: float) -> None:
        """Establece P2 de la trayectoria."""
        self.state.traj_p2 = (
            float(np.clip(theta, -np.pi/2, 3*np.pi/2)),
            float(np.clip(phi, 0, 2 * np.pi))
        )
        self._recompute_trajectory()
        self.update()
    
    def on_click(self, theta: float, phi: float) -> None:
        """Maneja un click en el espacio paramétrico."""
        theta = float(np.clip(theta, -np.pi/2, 3*np.pi/2))
        phi = float(np.clip(phi, 0, 2 * np.pi))
        
        if self.state.mode == "point":
            self.set_theta_phi(theta, phi)
            return
        
        # Modo trajectory: definir P1, luego P2
        if self.state.traj_p1 is None:
            self.set_p1(theta, phi)
        elif self.state.traj_p2 is None:
            self.set_p2(theta, phi)
        else:
            self.state.traj_p1 = (theta, phi)
            self.state.traj_p2 = None
            self._recompute_trajectory()
            self.update()
    
    def set_traj_n(self, n: int) -> None:
        """Cambia el número de puntos de la trayectoria."""
        self.state.traj_n = max(2, int(n))
        self._recompute_trajectory()
        self.update()
    
    def set_traj_index(self, k: int) -> None:
        """Cambia el índice del punto actual en la trayectoria."""
        if not self.state.traj_points:
            return
        
        k = int(np.clip(k, 0, len(self.state.traj_points) - 1))
        self.state.traj_index = k
        th, ph = self.state.traj_points[k]
        self.state.theta, self.state.phi = th, ph
        self.update()
    
    def clear_trajectory(self) -> None:
        """Limpia la trayectoria."""
        self.state.clear_trajectory()
        self._traj_patches_cache = []
        self.update()
    
    def update(self) -> None:
        """Actualiza la vista con el estado actual."""
        patch = self.generator.generate(self.state.theta, self.state.phi)
        
        # Texto informativo en unidades de π
        t_pi = self.state.theta / np.pi
        p_pi = self.state.phi / np.pi
        info = f"θ={t_pi:.2f}π  φ={p_pi:.2f}π"
        if self.state.traj_points:
            info += f"\ni={self.state.traj_index + 1}/{len(self.state.traj_points)}"
        
        self.view.draw_param_space(self.state)
        self.view.draw_patch(patch, info)
        self.view.draw_trajectory_patches(self._traj_patches_cache, self.state.traj_index)
        self.view.redraw()
