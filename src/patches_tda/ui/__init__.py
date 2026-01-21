"""Módulo de interfaz de usuario interactiva."""

from patches_tda.ui.state import UIState
from patches_tda.ui.trajectory import Trajectory, LineSegmentTrajectory
from patches_tda.ui.view import PlotView
from patches_tda.ui.controller import UIController
from patches_tda.ui.adapters import PatchGenerator, PatchGeneratorAdapter
from patches_tda.ui.interactive import build_ui

__all__ = [
    "UIState",
    "Trajectory",
    "LineSegmentTrajectory",
    "PlotView",
    "UIController",
    "PatchGenerator",
    "PatchGeneratorAdapter",
    "build_ui",
]
