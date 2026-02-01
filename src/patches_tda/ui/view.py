"""
PlotView - Vista para la interfaz interactiva

Solo dibuja, no mantiene estado.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import numpy as np
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from patches_tda.ui.state import UIState


class PlotView:
    """Vista de la UI con layout simple y centrado."""
    
    def __init__(self, max_traj_patches: int = 10) -> None:
        self._max_traj_patches = max_traj_patches
        
        plt.ioff()  # Prevenir display automático
        
        # Figura simple: 2 filas
        # Fila 1: espacio paramétrico + parche actual
        # Fila 2: parches de trayectoria
        self.fig = plt.figure(figsize=(10, 8))
        
        # Ejes para fila superior (posiciones manuales [left, bottom, width, height])
        self.ax_param = self.fig.add_axes([0.08, 0.45, 0.38, 0.48])
        self.ax_patch = self.fig.add_axes([0.55, 0.45, 0.38, 0.48])
        
        # Área para trayectoria (fila inferior)
        self.ax_traj = self.fig.add_axes([0.08, 0.08, 0.85, 0.28])
        
        self._cbar = None
        self._theta_o = 0.0 #-np.pi/2
        self._theta_f = np.pi #3*np.pi/2
    
    def draw_param_space(self, state: "UIState") -> None:
        """Dibuja el espacio paramétrico."""
        ax = self.ax_param
        ax.clear()
        
        # Límites modificados
        ax.set_xlim(-np.pi/2, 3*np.pi/2)
        ax.set_ylim(0, 2 * np.pi)
        ax.set_xlabel("θ [0, π]")#("θ [-π/2, 3π/2]")
        ax.set_ylabel("φ [0, 2π]")
        ax.set_title("Espacio Paramétrico")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")
        
        # Ticks para θ (eje x)
        ax.set_xticks([ 0, np.pi/2, np.pi])#([-np.pi/2, 0, np.pi/2, np.pi, 3*np.pi/2])
        ax.set_xticklabels(["0", "π/2", "π"])
        
        # Ticks para φ (eje y)
        ax.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax.set_yticklabels(["0", "π/2", "π", "3π/2", "2π"])
        
        # Líneas de referencia
        ax.axhline(np.pi/2, color="green", alpha=0.5, linestyle="--", label="φ=π/2")
        ax.axhline(3*np.pi/2, color="green", alpha=0.5, linestyle="--", label="φ=3π/2")
        ax.axvline(0, color="red", alpha=0.5, linestyle="--", label="θ=0")
        ax.axvline(np.pi/2, color="blue", alpha=0.5, linestyle="--", label="θ=π/2")
        
        # Punto actual
        ax.plot(state.theta, state.phi, "ro", markersize=10, 
                markeredgecolor="white", markeredgewidth=2, zorder=10)
        
        # Trayectoria
        if state.traj_points:
            pts = np.array(state.traj_points)
            ax.plot(pts[:, 0], pts[:, 1], "-", linewidth=2, color="steelblue", alpha=0.7)
            ax.scatter(pts[:, 0], pts[:, 1], s=25, c="steelblue", zorder=5)
            
            k = state.traj_index
            if 0 <= k < len(pts):
                ax.plot(pts[k, 0], pts[k, 1], "o", markersize=14, 
                        color="yellow", markeredgecolor="black", markeredgewidth=2, zorder=11)
        
        if state.traj_p1:
            ax.plot(*state.traj_p1, "s", color="limegreen", markersize=12, 
                    markeredgecolor="white", markeredgewidth=2, label="P1", zorder=9)
        if state.traj_p2:
            ax.plot(*state.traj_p2, "s", color="dodgerblue", markersize=12, 
                    markeredgecolor="white", markeredgewidth=2, label="P2", zorder=9)
        
        if state.traj_p1 or state.traj_p2:
            ax.legend(loc="upper right", fontsize=9)
    
    def draw_patch(self, patch: np.ndarray, info: str) -> None:
        """Dibuja el parche actual."""
        ax = self.ax_patch
        ax.clear()
        ax.set_title("Parche Actual", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        
        im = ax.imshow(patch, cmap="gray", vmin=-1, vmax=1)
        
        # Colorbar simple
        if self._cbar is None:
            self._cbar = self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            self._cbar.update_normal(im)
        
        ax.set_xlabel(info, fontsize=9)
    
    def draw_trajectory_patches(
        self, 
        patches: List[np.ndarray], 
        current_index: int
    ) -> None:
        """Dibuja los parches de la trayectoria como grilla."""
        ax = self.ax_traj
        ax.clear()
        
        if not patches:
            ax.text(0.5, 0.5, "Define P1 y P2 para ver trayectoria", 
                    ha="center", va="center", fontsize=11, color="gray",
                    transform=ax.transAxes)
            ax.axis("off")
            return
        
        n = len(patches)
        
        # Seleccionar índices a mostrar
        if n > self._max_traj_patches:
            step = max(1, n // self._max_traj_patches)
            indices = list(range(0, n, step))[:self._max_traj_patches]
        else:
            indices = list(range(n))
        
        n_show = len(indices)
        patch_size = patches[0].shape[0]
        
        # Concatenar parches SIN separadores grises
        patch_list = [patches[idx] for idx in indices]
        combined = np.hstack(patch_list)
        
        ax.imshow(combined, cmap="gray", vmin=-1, vmax=1, aspect="equal")
        ax.set_title(f"Trayectoria: {n} parches", fontsize=10)
        ax.set_yticks([])
        
        # Líneas verticales para separar parches
        for i in range(1, n_show):
            ax.axvline(i * patch_size - 0.5, color="white", linewidth=1, alpha=0.8)
        
        # Etiquetas y resaltado
        x_ticks = []
        x_labels = []
        for i, idx in enumerate(indices):
            x_pos = i * patch_size + patch_size / 2
            x_ticks.append(x_pos)
            
            if idx == current_index:
                # Resaltar parche actual
                ax.axvspan(i * patch_size - 0.5, (i + 1) * patch_size - 0.5,
                          color="yellow", alpha=0.3)
                x_labels.append(f"►{idx}")
            else:
                x_labels.append(str(idx))
        
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=8)
    
    def redraw(self) -> None:
        """Refresca el canvas."""
        self.fig.canvas.draw_idle()
