"""
interactive - Ensamblaje de la UI interactiva

Conecta todos los componentes con ipywidgets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import ipywidgets as widgets

from patches_tda.ui.view import PlotView
from patches_tda.ui.controller import UIController

if TYPE_CHECKING:
    from patches_tda.ui.adapters import PatchGenerator


def build_ui(generator: "PatchGenerator") -> widgets.VBox:
    """
    Construye la UI interactiva completa.
    
    Parameters
    ----------
    generator : PatchGenerator
        Generador de parches.
    
    Returns
    -------
    widgets.VBox
        Widget contenedor con toda la UI.
    """
    view = PlotView()
    ctrl = UIController(generator, view)
    
    # =========================================================================
    # Widgets de modo
    # =========================================================================
    
    mode_toggle = widgets.ToggleButtons(
        options=["point", "trajectory"],
        value="point",
        description="Modo:",
        button_style=""
    )
    
    # =========================================================================
    # Widgets de punto (θ, φ)
    # =========================================================================
    
    # Sliders en unidades de π (0 a 2 significa 0 a 2π)
    # θ va de -0.5 a 1.5 (×π) = -π/2 a 3π/2
    #theta_slider = widgets.FloatSlider(
    #    value=0.5, min=-0.5, max=1.5, step=0.05,
    #    description="θ (×π)", continuous_update=True, readout_format=".2f"
    #)
    theta_slider = widgets.FloatSlider(
        value=0.5, min=0.0, max=1.0, step=0.05,
        description="θ (×π)", continuous_update=True, readout_format=".2f"
    )
    
    # φ va de 0 a 2 (×π) = 0 a 2π
    phi_slider = widgets.FloatSlider(
        value=0.5, min=0, max=2, step=0.05,
        description="φ (×π)", continuous_update=True, readout_format=".2f"
    )
    
    # =========================================================================
    # Widgets de trayectoria
    # =========================================================================
    
    # Botones para fijar P1/P2 usando los sliders actuales
    set_p1_btn = widgets.Button(description="Fijar P1", button_style="success", icon="map-marker")
    set_p2_btn = widgets.Button(description="Fijar P2", button_style="info", icon="map-marker")
    
    # Labels para mostrar P1/P2
    p1_label = widgets.HTML(value="<b>P1:</b> --")
    p2_label = widgets.HTML(value="<b>P2:</b> --")
    
    n_slider = widgets.IntSlider(
        value=10, min=2, max=50, step=1,
        description="N puntos", continuous_update=False
    )
    
    index_slider = widgets.IntSlider(
        value=0, min=0, max=9, step=1,
        description="Índice i", continuous_update=True
    )
    
    play_button = widgets.Play(
        value=0, min=0, max=9, step=1, interval=200, description="Play"
    )
    widgets.jslink((play_button, "value"), (index_slider, "value"))
    
    clear_button = widgets.Button(
        description="Clear", button_style="warning", icon="trash"
    )
    
    # =========================================================================
    # Callbacks
    # =========================================================================
    
    def update_p_labels():
        if ctrl.state.traj_p1:
            t1, p1 = ctrl.state.traj_p1[0]/np.pi, ctrl.state.traj_p1[1]/np.pi
            p1_label.value = f"<b>P1:</b> θ={t1:.2f}π, φ={p1:.2f}π"
        else:
            p1_label.value = "<b>P1:</b> --"
        if ctrl.state.traj_p2:
            t2, p2 = ctrl.state.traj_p2[0]/np.pi, ctrl.state.traj_p2[1]/np.pi
            p2_label.value = f"<b>P2:</b> θ={t2:.2f}π, φ={p2:.2f}π"
        else:
            p2_label.value = "<b>P2:</b> --"
    
    def update_traj_slider_max():
        if ctrl.state.traj_points:
            max_idx = len(ctrl.state.traj_points) - 1
            index_slider.max = max_idx
            play_button.max = max_idx
    
    def on_mode_change(change):
        ctrl.set_mode(change["new"])
    
    mode_toggle.observe(on_mode_change, names="value")
    
    def on_theta_change(change):
        # Multiplicar por π para obtener valor real
        theta_real = change["new"] * np.pi
        ctrl.set_theta_phi(theta_real, ctrl.state.phi)
    
    def on_phi_change(change):
        # Multiplicar por π para obtener valor real
        phi_real = change["new"] * np.pi
        ctrl.set_theta_phi(ctrl.state.theta, phi_real)
    
    theta_slider.observe(on_theta_change, names="value")
    phi_slider.observe(on_phi_change, names="value")
    
    def on_set_p1(_):
        theta_real = theta_slider.value * np.pi
        phi_real = phi_slider.value * np.pi
        ctrl.set_p1(theta_real, phi_real)
        update_p_labels()
        update_traj_slider_max()
    
    def on_set_p2(_):
        theta_real = theta_slider.value * np.pi
        phi_real = phi_slider.value * np.pi
        ctrl.set_p2(theta_real, phi_real)
        update_p_labels()
        update_traj_slider_max()
    
    set_p1_btn.on_click(on_set_p1)
    set_p2_btn.on_click(on_set_p2)
    
    def on_n_change(change):
        ctrl.set_traj_n(change["new"])
        update_traj_slider_max()
    
    n_slider.observe(on_n_change, names="value")
    
    def on_index_change(change):
        ctrl.set_traj_index(change["new"])
    
    index_slider.observe(on_index_change, names="value")
    
    def on_clear_click(_):
        ctrl.clear_trajectory()
        update_p_labels()
        index_slider.value = 0
    
    clear_button.on_click(on_clear_click)
    
    # =========================================================================
    # Click en matplotlib
    # =========================================================================
    
    def mpl_onclick(event):
        if event.inaxes == view.ax_param and event.xdata is not None:
            ctrl.on_click(event.xdata, event.ydata)
            # Actualizar sliders (convertir de radianes a unidades de π)
            theta_slider.value = ctrl.state.theta / np.pi
            phi_slider.value = ctrl.state.phi / np.pi
            update_p_labels()
            update_traj_slider_max()
    
    view.fig.canvas.mpl_connect("button_press_event", mpl_onclick)
    
    # =========================================================================
    # Inicializar
    # =========================================================================
    
    ctrl.update()
    
    # =========================================================================
    # Layout
    # =========================================================================
    
    point_controls = widgets.HBox([
        widgets.VBox([theta_slider, phi_slider]),
        widgets.VBox([set_p1_btn, set_p2_btn])
    ])
    
    traj_info = widgets.HBox([p1_label, p2_label])
    
    traj_controls = widgets.HBox([
        n_slider, index_slider, play_button, clear_button
    ])
    
    help_text = widgets.HTML(
        "<small><b>Point mode:</b> Sliders o click | "
        "<b>Trajectory:</b> Usa sliders + 'Fijar P1/P2' o click 2 veces</small>"
    )
    
    controls = widgets.VBox([
        mode_toggle,
        point_controls,
        traj_info,
        traj_controls,
        help_text
    ])
    
    return widgets.VBox([controls, view.fig.canvas])
