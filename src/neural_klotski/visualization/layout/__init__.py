"""
Layout System for Neural-Klotski Visualization

2D coordinate system and spatial layout management for the 79-block network.
Provides coordinate transformations, block positioning, and viewport management.
"""

from .coordinate_system import (
    CoordinateSystem,
    ActivationLagSpace,
    ScreenSpace,
    WorldSpace,
    CoordinateTransform
)

from .block_layout import (
    BlockLayoutManager,
    LayoutStrategy,
    GridLayout,
    CircularLayout,
    ForceDirectedLayout,
    ActivationLagLayout
)

from .spatial_positioning import (
    SpatialPositioner,
    PositionCalculator,
    CollisionDetector,
    LayoutOptimizer
)

from .viewport import (
    Viewport,
    Camera,
    ViewportConfig,
    ZoomController,
    PanController
)

__all__ = [
    # Coordinate system
    'CoordinateSystem',
    'ActivationLagSpace',
    'ScreenSpace',
    'WorldSpace',
    'CoordinateTransform',

    # Block layout
    'BlockLayoutManager',
    'LayoutStrategy',
    'GridLayout',
    'CircularLayout',
    'ForceDirectedLayout',
    'ActivationLagLayout',

    # Spatial positioning
    'SpatialPositioner',
    'PositionCalculator',
    'CollisionDetector',
    'LayoutOptimizer',

    # Viewport management
    'Viewport',
    'Camera',
    'ViewportConfig',
    'ZoomController',
    'PanController'
]