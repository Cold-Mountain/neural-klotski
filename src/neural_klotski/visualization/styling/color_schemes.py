"""
Color Schemes for Neural-Klotski Visualization

Defines comprehensive color schemes for different visual themes and use cases.
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Optional
from enum import Enum


class ColorSchemeType(Enum):
    """Available color scheme types"""
    DEFAULT = "default"
    DARK = "dark"
    LIGHT = "light"
    HIGH_CONTRAST = "high_contrast"
    SCIENTIFIC = "scientific"
    ACCESSIBILITY = "accessibility"


@dataclass
class ColorScheme:
    """Complete color scheme definition"""

    # Scheme metadata
    name: str
    description: str
    scheme_type: ColorSchemeType

    # Background colors
    background_primary: Tuple[float, float, float]
    background_secondary: Tuple[float, float, float]
    surface: Tuple[float, float, float]
    surface_variant: Tuple[float, float, float]

    # Text colors
    text_primary: Tuple[float, float, float]
    text_secondary: Tuple[float, float, float]
    text_disabled: Tuple[float, float, float]

    # Accent and UI colors
    primary: Tuple[float, float, float]
    primary_variant: Tuple[float, float, float]
    secondary: Tuple[float, float, float]
    secondary_variant: Tuple[float, float, float]

    # Semantic colors
    success: Tuple[float, float, float]
    warning: Tuple[float, float, float]
    error: Tuple[float, float, float]
    info: Tuple[float, float, float]

    # Grid and guides
    grid_major: Tuple[float, float, float]
    grid_minor: Tuple[float, float, float]
    axis_lines: Tuple[float, float, float]
    selection: Tuple[float, float, float]
    highlight: Tuple[float, float, float]

    # Neural network specific colors
    block_red: Tuple[float, float, float]
    block_blue: Tuple[float, float, float]
    block_yellow: Tuple[float, float, float]

    # State colors
    firing: Tuple[float, float, float]
    refractory: Tuple[float, float, float]
    threshold: Tuple[float, float, float]

    # Dye colors
    dye_red: Tuple[float, float, float]
    dye_blue: Tuple[float, float, float]
    dye_yellow: Tuple[float, float, float]

    # Wire colors
    wire_active: Tuple[float, float, float]
    wire_inactive: Tuple[float, float, float]
    signal_glow: Tuple[float, float, float]

    # Training metrics colors
    accuracy_color: Tuple[float, float, float]
    loss_color: Tuple[float, float, float]
    learning_rate_color: Tuple[float, float, float]
    convergence_color: Tuple[float, float, float]


# Default color scheme (dark theme optimized for long viewing)
DEFAULT_SCHEME = ColorScheme(
    name="Neural-Klotski Default",
    description="Default dark theme optimized for neural network visualization",
    scheme_type=ColorSchemeType.DEFAULT,

    # Backgrounds
    background_primary=(0.08, 0.08, 0.10),
    background_secondary=(0.12, 0.12, 0.15),
    surface=(0.16, 0.16, 0.20),
    surface_variant=(0.20, 0.20, 0.25),

    # Text
    text_primary=(0.95, 0.95, 0.95),
    text_secondary=(0.75, 0.75, 0.75),
    text_disabled=(0.45, 0.45, 0.45),

    # Accents
    primary=(0.25, 0.65, 1.00),
    primary_variant=(0.15, 0.45, 0.80),
    secondary=(0.60, 0.25, 0.90),
    secondary_variant=(0.45, 0.15, 0.70),

    # Semantic
    success=(0.20, 0.80, 0.20),
    warning=(1.00, 0.70, 0.20),
    error=(0.90, 0.20, 0.20),
    info=(0.20, 0.70, 1.00),

    # Grid
    grid_major=(0.30, 0.30, 0.35),
    grid_minor=(0.20, 0.20, 0.25),
    axis_lines=(0.50, 0.50, 0.55),
    selection=(0.90, 0.60, 0.20),
    highlight=(1.00, 1.00, 0.40),

    # Blocks
    block_red=(1.00, 0.30, 0.30),
    block_blue=(0.30, 0.50, 1.00),
    block_yellow=(1.00, 0.90, 0.30),

    # States
    firing=(1.00, 1.00, 1.00),
    refractory=(0.50, 0.50, 0.60),
    threshold=(0.80, 0.80, 0.85),

    # Dyes
    dye_red=(1.00, 0.20, 0.20),
    dye_blue=(0.20, 0.40, 1.00),
    dye_yellow=(1.00, 0.80, 0.20),

    # Wires
    wire_active=(0.70, 0.70, 0.75),
    wire_inactive=(0.40, 0.40, 0.45),
    signal_glow=(0.90, 0.90, 1.00),

    # Training
    accuracy_color=(0.20, 0.80, 0.40),
    loss_color=(0.90, 0.30, 0.30),
    learning_rate_color=(0.80, 0.60, 0.20),
    convergence_color=(0.60, 0.30, 0.90)
)

# Light theme for presentations
LIGHT_SCHEME = ColorScheme(
    name="Light Theme",
    description="Light theme suitable for presentations and documentation",
    scheme_type=ColorSchemeType.LIGHT,

    # Backgrounds
    background_primary=(0.98, 0.98, 0.98),
    background_secondary=(0.94, 0.94, 0.94),
    surface=(0.90, 0.90, 0.90),
    surface_variant=(0.85, 0.85, 0.85),

    # Text
    text_primary=(0.10, 0.10, 0.10),
    text_secondary=(0.30, 0.30, 0.30),
    text_disabled=(0.60, 0.60, 0.60),

    # Accents
    primary=(0.20, 0.50, 0.90),
    primary_variant=(0.15, 0.35, 0.70),
    secondary=(0.50, 0.20, 0.80),
    secondary_variant=(0.35, 0.15, 0.60),

    # Semantic
    success=(0.15, 0.70, 0.15),
    warning=(0.85, 0.55, 0.15),
    error=(0.80, 0.15, 0.15),
    info=(0.15, 0.55, 0.85),

    # Grid
    grid_major=(0.70, 0.70, 0.70),
    grid_minor=(0.80, 0.80, 0.80),
    axis_lines=(0.50, 0.50, 0.50),
    selection=(0.90, 0.60, 0.20),
    highlight=(0.95, 0.85, 0.20),

    # Blocks
    block_red=(0.85, 0.20, 0.20),
    block_blue=(0.20, 0.35, 0.85),
    block_yellow=(0.85, 0.75, 0.15),

    # States
    firing=(0.95, 0.95, 0.15),
    refractory=(0.60, 0.60, 0.65),
    threshold=(0.40, 0.40, 0.45),

    # Dyes
    dye_red=(0.90, 0.15, 0.15),
    dye_blue=(0.15, 0.30, 0.90),
    dye_yellow=(0.90, 0.70, 0.15),

    # Wires
    wire_active=(0.30, 0.30, 0.35),
    wire_inactive=(0.65, 0.65, 0.70),
    signal_glow=(0.20, 0.20, 0.30),

    # Training
    accuracy_color=(0.15, 0.70, 0.30),
    loss_color=(0.80, 0.25, 0.25),
    learning_rate_color=(0.70, 0.50, 0.15),
    convergence_color=(0.50, 0.25, 0.80)
)

# High contrast theme for accessibility
HIGH_CONTRAST_SCHEME = ColorScheme(
    name="High Contrast",
    description="High contrast theme for accessibility",
    scheme_type=ColorSchemeType.HIGH_CONTRAST,

    # Backgrounds
    background_primary=(0.00, 0.00, 0.00),
    background_secondary=(0.10, 0.10, 0.10),
    surface=(0.15, 0.15, 0.15),
    surface_variant=(0.20, 0.20, 0.20),

    # Text
    text_primary=(1.00, 1.00, 1.00),
    text_secondary=(0.90, 0.90, 0.90),
    text_disabled=(0.60, 0.60, 0.60),

    # Accents
    primary=(0.00, 0.80, 1.00),
    primary_variant=(0.00, 0.60, 0.80),
    secondary=(1.00, 0.80, 0.00),
    secondary_variant=(0.80, 0.60, 0.00),

    # Semantic
    success=(0.00, 1.00, 0.00),
    warning=(1.00, 1.00, 0.00),
    error=(1.00, 0.00, 0.00),
    info=(0.00, 1.00, 1.00),

    # Grid
    grid_major=(0.50, 0.50, 0.50),
    grid_minor=(0.30, 0.30, 0.30),
    axis_lines=(0.70, 0.70, 0.70),
    selection=(1.00, 1.00, 0.00),
    highlight=(1.00, 0.50, 0.00),

    # Blocks
    block_red=(1.00, 0.00, 0.00),
    block_blue=(0.00, 0.50, 1.00),
    block_yellow=(1.00, 1.00, 0.00),

    # States
    firing=(1.00, 1.00, 1.00),
    refractory=(0.50, 0.50, 0.50),
    threshold=(0.80, 0.80, 0.80),

    # Dyes
    dye_red=(1.00, 0.00, 0.00),
    dye_blue=(0.00, 0.50, 1.00),
    dye_yellow=(1.00, 1.00, 0.00),

    # Wires
    wire_active=(0.90, 0.90, 0.90),
    wire_inactive=(0.40, 0.40, 0.40),
    signal_glow=(1.00, 1.00, 1.00),

    # Training
    accuracy_color=(0.00, 1.00, 0.00),
    loss_color=(1.00, 0.00, 0.00),
    learning_rate_color=(1.00, 0.80, 0.00),
    convergence_color=(0.80, 0.00, 1.00)
)

# Scientific theme with precise colors
SCIENTIFIC_SCHEME = ColorScheme(
    name="Scientific",
    description="Scientific theme with precise, distinguishable colors",
    scheme_type=ColorSchemeType.SCIENTIFIC,

    # Backgrounds
    background_primary=(0.95, 0.95, 0.97),
    background_secondary=(0.92, 0.92, 0.94),
    surface=(0.88, 0.88, 0.90),
    surface_variant=(0.84, 0.84, 0.86),

    # Text
    text_primary=(0.05, 0.05, 0.05),
    text_secondary=(0.25, 0.25, 0.25),
    text_disabled=(0.55, 0.55, 0.55),

    # Accents
    primary=(0.12, 0.47, 0.71),  # Professional blue
    primary_variant=(0.08, 0.35, 0.55),
    secondary=(0.55, 0.12, 0.47),  # Professional purple
    secondary_variant=(0.40, 0.08, 0.35),

    # Semantic
    success=(0.13, 0.59, 0.22),  # Nature green
    warning=(0.80, 0.52, 0.08),  # Amber
    error=(0.77, 0.05, 0.13),   # Deep red
    info=(0.05, 0.60, 0.85),    # Information blue

    # Grid
    grid_major=(0.60, 0.60, 0.62),
    grid_minor=(0.75, 0.75, 0.77),
    axis_lines=(0.35, 0.35, 0.37),
    selection=(1.00, 0.65, 0.00),  # Orange selection
    highlight=(0.95, 0.85, 0.25),

    # Blocks (using distinct, scientifically accurate colors)
    block_red=(0.89, 0.10, 0.11),    # True red
    block_blue=(0.21, 0.41, 0.69),   # True blue
    block_yellow=(0.94, 0.89, 0.26), # True yellow

    # States
    firing=(0.95, 0.95, 0.20),
    refractory=(0.45, 0.45, 0.50),
    threshold=(0.35, 0.35, 0.40),

    # Dyes (slightly desaturated for overlay)
    dye_red=(0.85, 0.15, 0.15),
    dye_blue=(0.25, 0.45, 0.75),
    dye_yellow=(0.90, 0.85, 0.30),

    # Wires
    wire_active=(0.25, 0.25, 0.30),
    wire_inactive=(0.60, 0.60, 0.65),
    signal_glow=(0.15, 0.15, 0.25),

    # Training
    accuracy_color=(0.13, 0.59, 0.22),
    loss_color=(0.77, 0.05, 0.13),
    learning_rate_color=(0.80, 0.52, 0.08),
    convergence_color=(0.55, 0.12, 0.47)
)

# Color scheme registry
COLOR_SCHEMES: Dict[ColorSchemeType, ColorScheme] = {
    ColorSchemeType.DEFAULT: DEFAULT_SCHEME,
    ColorSchemeType.LIGHT: LIGHT_SCHEME,
    ColorSchemeType.HIGH_CONTRAST: HIGH_CONTRAST_SCHEME,
    ColorSchemeType.SCIENTIFIC: SCIENTIFIC_SCHEME,
}

def get_color_scheme(scheme_type: ColorSchemeType = ColorSchemeType.DEFAULT) -> ColorScheme:
    """Get color scheme by type"""
    return COLOR_SCHEMES.get(scheme_type, DEFAULT_SCHEME)

def list_available_schemes() -> Dict[str, str]:
    """List all available color schemes with descriptions"""
    return {
        scheme.scheme_type.value: scheme.description
        for scheme in COLOR_SCHEMES.values()
    }

def get_scheme_by_name(name: str) -> Optional[ColorScheme]:
    """Get color scheme by name string"""
    try:
        scheme_type = ColorSchemeType(name.lower())
        return get_color_scheme(scheme_type)
    except ValueError:
        return None