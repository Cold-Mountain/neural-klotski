"""
Block Themes for Neural-Klotski Visualization

Defines visual themes specifically for block rendering including colors,
sizes, effects, and state visualizations.
"""

from dataclasses import dataclass
from typing import Tuple, Dict
from enum import Enum
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from neural_klotski.core.block import BlockColor


@dataclass
class BlockTheme:
    """Visual theme for block rendering"""

    # Theme metadata
    name: str
    description: str

    # Base block properties
    base_radius: float = 8.0
    border_width: float = 1.0

    # Block colors by type
    red_block_color: Tuple[float, float, float] = (1.0, 0.3, 0.3)
    blue_block_color: Tuple[float, float, float] = (0.3, 0.5, 1.0)
    yellow_block_color: Tuple[float, float, float] = (1.0, 0.9, 0.3)

    # Border colors
    red_border_color: Tuple[float, float, float] = (0.8, 0.2, 0.2)
    blue_border_color: Tuple[float, float, float] = (0.2, 0.3, 0.8)
    yellow_border_color: Tuple[float, float, float] = (0.8, 0.7, 0.2)

    # State-specific modifications
    firing_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    firing_glow_radius: float = 2.0
    firing_duration: float = 0.3  # seconds

    refractory_alpha: float = 0.4  # Dimming factor
    refractory_border_alpha: float = 0.6

    # Threshold visualization
    threshold_color: Tuple[float, float, float] = (0.8, 0.8, 0.8)
    threshold_width: float = 1.0
    threshold_style: str = "dashed"  # "solid", "dashed", "dotted"

    # Hover and selection effects
    hover_highlight_color: Tuple[float, float, float] = (1.0, 1.0, 0.4)
    hover_radius_scale: float = 1.2
    selection_color: Tuple[float, float, float] = (0.9, 0.6, 0.2)
    selection_width: float = 2.0

    # Animation properties
    position_smoothing: float = 0.8
    size_animation_speed: float = 5.0
    color_transition_speed: float = 3.0

    def get_block_color(self, block_color: BlockColor) -> Tuple[float, float, float]:
        """Get base color for block type"""
        color_map = {
            BlockColor.RED: self.red_block_color,
            BlockColor.BLUE: self.blue_block_color,
            BlockColor.YELLOW: self.yellow_block_color
        }
        return color_map.get(block_color, self.red_block_color)

    def get_border_color(self, block_color: BlockColor) -> Tuple[float, float, float]:
        """Get border color for block type"""
        color_map = {
            BlockColor.RED: self.red_border_color,
            BlockColor.BLUE: self.blue_border_color,
            BlockColor.YELLOW: self.yellow_border_color
        }
        return color_map.get(block_color, self.red_border_color)


# Default block theme
DEFAULT_BLOCK_THEME = BlockTheme(
    name="Default",
    description="Default block theme with standard colors and effects"
)

# High contrast block theme
HIGH_CONTRAST_BLOCK_THEME = BlockTheme(
    name="High Contrast",
    description="High contrast theme for accessibility",
    base_radius=10.0,
    border_width=2.0,
    red_block_color=(1.0, 0.0, 0.0),
    blue_block_color=(0.0, 0.5, 1.0),
    yellow_block_color=(1.0, 1.0, 0.0),
    red_border_color=(0.8, 0.0, 0.0),
    blue_border_color=(0.0, 0.3, 0.8),
    yellow_border_color=(0.8, 0.8, 0.0),
    threshold_width=2.0,
    selection_width=3.0
)

# Minimal block theme
MINIMAL_BLOCK_THEME = BlockTheme(
    name="Minimal",
    description="Clean minimal theme with subtle colors",
    base_radius=6.0,
    border_width=0.5,
    red_block_color=(0.8, 0.4, 0.4),
    blue_block_color=(0.4, 0.6, 0.8),
    yellow_block_color=(0.8, 0.8, 0.4),
    red_border_color=(0.6, 0.3, 0.3),
    blue_border_color=(0.3, 0.4, 0.6),
    yellow_border_color=(0.6, 0.6, 0.3),
    firing_glow_radius=1.0,
    hover_radius_scale=1.1
)

# Scientific block theme
SCIENTIFIC_BLOCK_THEME = BlockTheme(
    name="Scientific",
    description="Professional theme for scientific presentations",
    base_radius=7.0,
    border_width=1.5,
    red_block_color=(0.89, 0.10, 0.11),
    blue_block_color=(0.21, 0.41, 0.69),
    yellow_block_color=(0.94, 0.89, 0.26),
    red_border_color=(0.7, 0.08, 0.09),
    blue_border_color=(0.16, 0.31, 0.54),
    yellow_border_color=(0.75, 0.71, 0.21),
    threshold_color=(0.3, 0.3, 0.3),
    selection_color=(1.0, 0.65, 0.0)
)

# Block theme registry
BLOCK_THEMES: Dict[str, BlockTheme] = {
    'default': DEFAULT_BLOCK_THEME,
    'high_contrast': HIGH_CONTRAST_BLOCK_THEME,
    'minimal': MINIMAL_BLOCK_THEME,
    'scientific': SCIENTIFIC_BLOCK_THEME
}

def get_block_theme(theme_name: str = 'default') -> BlockTheme:
    """Get block theme by name"""
    return BLOCK_THEMES.get(theme_name, DEFAULT_BLOCK_THEME)

def list_block_themes() -> Dict[str, str]:
    """List available block themes with descriptions"""
    return {name: theme.description for name, theme in BLOCK_THEMES.items()}