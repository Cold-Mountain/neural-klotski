"""
Wire Themes for Neural-Klotski Visualization

Defines visual themes for wire and signal rendering including colors,
styles, animations, and connection visualizations.
"""

from dataclasses import dataclass
from typing import Tuple, Dict
from enum import Enum


class WireStyle(Enum):
    """Wire rendering styles"""
    SOLID = "solid"
    DASHED = "dashed"
    DOTTED = "dotted"
    GRADIENT = "gradient"


@dataclass
class WireTheme:
    """Visual theme for wire and signal rendering"""

    # Theme metadata
    name: str
    description: str

    # Base wire properties
    base_thickness: float = 1.0
    thickness_scale: float = 2.0  # Multiplier for strength visualization
    max_thickness: float = 4.0

    # Wire colors (inherit from source block by default)
    inherit_block_colors: bool = True
    default_wire_color: Tuple[float, float, float] = (0.6, 0.6, 0.6)
    wire_alpha: float = 0.7

    # Connection type styles
    knn_style: WireStyle = WireStyle.SOLID
    longrange_style: WireStyle = WireStyle.DASHED
    knn_alpha: float = 0.7
    longrange_alpha: float = 0.5

    # Signal visualization
    signal_size: float = 4.0
    signal_color: Tuple[float, float, float] = (1.0, 1.0, 0.8)
    signal_glow_color: Tuple[float, float, float] = (0.9, 0.9, 1.0)
    signal_glow_radius: float = 2.0

    # Signal trail properties
    trail_length: int = 8
    trail_alpha_start: float = 0.9
    trail_alpha_end: float = 0.1
    trail_size_scale: float = 0.7

    # Animation properties
    signal_speed_scale: float = 1.0
    pulse_frequency: float = 2.0  # Hz
    pulse_amplitude: float = 0.3

    # Strength visualization
    strength_affects_thickness: bool = True
    strength_affects_alpha: bool = True
    min_strength_alpha: float = 0.2
    max_strength_alpha: float = 1.0

    # Dye enhancement visualization
    dye_glow_enabled: bool = True
    dye_glow_intensity: float = 1.5
    dye_color_blend: float = 0.3

    # Hover and selection effects
    hover_highlight_scale: float = 1.5
    hover_glow_color: Tuple[float, float, float] = (1.0, 1.0, 0.4)
    selection_color: Tuple[float, float, float] = (0.9, 0.6, 0.2)
    selection_thickness_scale: float = 2.0


# Default wire theme
DEFAULT_WIRE_THEME = WireTheme(
    name="Default",
    description="Default wire theme with moderate visibility"
)

# High visibility wire theme
HIGH_VISIBILITY_WIRE_THEME = WireTheme(
    name="High Visibility",
    description="High visibility theme with bright colors and effects",
    base_thickness=1.5,
    thickness_scale=3.0,
    wire_alpha=0.9,
    signal_size=6.0,
    signal_glow_radius=3.0,
    trail_length=12,
    dye_glow_intensity=2.0
)

# Minimal wire theme
MINIMAL_WIRE_THEME = WireTheme(
    name="Minimal",
    description="Clean minimal theme with subtle wire rendering",
    base_thickness=0.5,
    thickness_scale=1.5,
    wire_alpha=0.4,
    signal_size=3.0,
    signal_glow_radius=1.0,
    trail_length=4,
    pulse_amplitude=0.1,
    dye_glow_enabled=False
)

# Scientific wire theme
SCIENTIFIC_WIRE_THEME = WireTheme(
    name="Scientific",
    description="Professional theme for scientific presentations",
    base_thickness=1.0,
    thickness_scale=2.0,
    wire_alpha=0.6,
    signal_color=(0.2, 0.2, 0.2),
    signal_glow_color=(0.4, 0.4, 0.4),
    signal_glow_radius=1.5,
    trail_alpha_start=0.8,
    trail_alpha_end=0.2,
    pulse_frequency=1.0
)

# Debug wire theme
DEBUG_WIRE_THEME = WireTheme(
    name="Debug",
    description="Debug theme with maximum visibility and information",
    base_thickness=2.0,
    thickness_scale=4.0,
    max_thickness=8.0,
    wire_alpha=1.0,
    knn_alpha=1.0,
    longrange_alpha=0.8,
    signal_size=8.0,
    signal_glow_radius=4.0,
    trail_length=15,
    dye_glow_intensity=3.0,
    inherit_block_colors=False,
    default_wire_color=(0.8, 0.8, 0.8)
)

# Wire theme registry
WIRE_THEMES: Dict[str, WireTheme] = {
    'default': DEFAULT_WIRE_THEME,
    'high_visibility': HIGH_VISIBILITY_WIRE_THEME,
    'minimal': MINIMAL_WIRE_THEME,
    'scientific': SCIENTIFIC_WIRE_THEME,
    'debug': DEBUG_WIRE_THEME
}

def get_wire_theme(theme_name: str = 'default') -> WireTheme:
    """Get wire theme by name"""
    return WIRE_THEMES.get(theme_name, DEFAULT_WIRE_THEME)

def list_wire_themes() -> Dict[str, str]:
    """List available wire themes with descriptions"""
    return {name: theme.description for name, theme in WIRE_THEMES.items()}