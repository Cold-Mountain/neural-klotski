"""
Base Framework for Neural-Klotski Visualization

Provides abstract base classes and core framework components that all
visualization components inherit from. Establishes consistent interfaces
and shared functionality across the visualization system.
"""

from .visualizer_base import VisualizerBase
from .component_base import ComponentBase
from .renderer_base import RendererBase

__all__ = [
    'VisualizerBase',
    'ComponentBase',
    'RendererBase'
]