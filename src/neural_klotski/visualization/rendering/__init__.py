"""
Rendering System for Neural-Klotski Visualization

Visual rendering components for blocks, wires, and network elements.
Provides efficient 2D rendering with state visualization and interactive features.
"""

from .block_renderer import (
    BlockRenderer,
    BlockVisual,
    BlockRenderConfig,
    BlockState,
    BlockInteractionState
)

from .visual_elements import (
    VisualElement,
    CircularBlock,
    RectangularBlock,
    HexagonalBlock,
    BlockLabel,
    BlockIndicator
)

from .render_context import (
    RenderContext,
    RenderQueue,
    RenderLayer,
    DrawingPrimitives
)

from .color_schemes import (
    ColorScheme,
    BlockColorMapping,
    StateColorMapping,
    ThemeManager
)

from .render_optimization import (
    ViewportCuller,
    LevelOfDetail,
    RenderCache,
    BatchRenderer
)

__all__ = [
    # Block rendering
    'BlockRenderer',
    'BlockVisual',
    'BlockRenderConfig',
    'BlockState',
    'BlockInteractionState',

    # Visual elements
    'VisualElement',
    'CircularBlock',
    'RectangularBlock',
    'HexagonalBlock',
    'BlockLabel',
    'BlockIndicator',

    # Render context
    'RenderContext',
    'RenderQueue',
    'RenderLayer',
    'DrawingPrimitives',

    # Color and styling
    'ColorScheme',
    'BlockColorMapping',
    'StateColorMapping',
    'ThemeManager',

    # Optimization
    'ViewportCuller',
    'LevelOfDetail',
    'RenderCache',
    'BatchRenderer'
]