"""
Styling Framework for Neural-Klotski Visualization

Provides visual themes, color schemes, and styling utilities for consistent
appearance across all visualization components.
"""

from .color_schemes import ColorScheme, get_color_scheme
from .block_themes import BlockTheme, get_block_theme
from .wire_themes import WireTheme, get_wire_theme

__all__ = [
    'ColorScheme',
    'get_color_scheme',
    'BlockTheme',
    'get_block_theme',
    'WireTheme',
    'get_wire_theme'
]