"""
Color Schemes and Styling for Neural-Klotski Visualization

Comprehensive color management, theme system, and visual styling
for blocks, wires, and interface elements.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable
from enum import Enum
import math
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from neural_klotski.core.block import BlockColor, BlockState
from neural_klotski.visualization.rendering.visual_elements import Color


class ColorMode(Enum):
    """Color coding modes"""
    BLOCK_TYPE = "block_type"
    ACTIVATION_LEVEL = "activation_level"
    STATE_BASED = "state_based"
    CONNECTIVITY = "connectivity"
    LEARNING_RATE = "learning_rate"
    CUSTOM = "custom"


class ThemeType(Enum):
    """Visual theme types"""
    DEFAULT = "default"
    LIGHT = "light"
    DARK = "dark"
    HIGH_CONTRAST = "high_contrast"
    SCIENTIFIC = "scientific"
    NEON = "neon"
    MONOCHROME = "monochrome"


@dataclass
class ColorMapping:
    """Color mapping configuration"""
    primary_colors: Dict[Any, Color] = field(default_factory=dict)
    secondary_colors: Dict[Any, Color] = field(default_factory=dict)
    gradient_colors: List[Color] = field(default_factory=list)

    # Color interpolation
    use_interpolation: bool = True
    interpolation_steps: int = 256

    # Dynamic adjustments
    brightness_factor: float = 1.0
    saturation_factor: float = 1.0
    contrast_factor: float = 1.0


@dataclass
class VisualTheme:
    """Complete visual theme definition"""
    name: str
    theme_type: ThemeType

    # Background colors
    background_color: Color = field(default_factory=lambda: Color(0.95, 0.95, 0.95))
    canvas_color: Color = field(default_factory=lambda: Color(1.0, 1.0, 1.0))

    # Block colors
    block_colors: Dict[BlockColor, Color] = field(default_factory=dict)
    state_colors: Dict[BlockState, Color] = field(default_factory=dict)

    # UI colors
    text_color: Color = field(default_factory=lambda: Color(0.1, 0.1, 0.1))
    border_color: Color = field(default_factory=lambda: Color(0.3, 0.3, 0.3))
    highlight_color: Color = field(default_factory=lambda: Color(1.0, 1.0, 0.0))
    selection_color: Color = field(default_factory=lambda: Color(0.0, 0.8, 1.0))

    # Wire colors
    wire_color: Color = field(default_factory=lambda: Color(0.4, 0.4, 0.4))
    active_wire_color: Color = field(default_factory=lambda: Color(0.0, 0.6, 1.0))

    # Effect colors
    firing_color: Color = field(default_factory=lambda: Color(1.0, 0.3, 0.3))
    refractory_color: Color = field(default_factory=lambda: Color(1.0, 0.6, 0.0))

    # Transparency levels
    inactive_alpha: float = 0.3
    normal_alpha: float = 1.0
    highlight_alpha: float = 1.0


class BlockColorMapping:
    """
    Color mapping system for blocks.

    Manages dynamic color assignment based on various
    network properties and visualization modes.
    """

    def __init__(self, theme: VisualTheme):
        """Initialize block color mapping"""
        self.theme = theme
        self.color_mode = ColorMode.BLOCK_TYPE

        # Color caches for performance
        self.color_cache: Dict[Tuple[Any, ...], Color] = {}
        self.gradient_cache: Dict[str, List[Color]] = {}

        # Color interpolation
        self.activation_gradient = self._create_activation_gradient()
        self.state_gradient = self._create_state_gradient()

        # Custom color mappings
        self.custom_mappings: Dict[str, ColorMapping] = {}

    def get_block_color(self,
                       block_color: BlockColor,
                       activation_level: float = 0.0,
                       block_state: BlockState = BlockState.INACTIVE,
                       connection_count: int = 0,
                       custom_value: Optional[float] = None) -> Color:
        """
        Get color for a block based on current color mode.

        Args:
            block_color: Base block color
            activation_level: Activation level (0.0 to 1.0)
            block_state: Current block state
            connection_count: Number of connections
            custom_value: Custom value for custom color modes

        Returns:
            Color for the block
        """
        cache_key = (self.color_mode, block_color, activation_level, block_state, connection_count, custom_value)

        if cache_key in self.color_cache:
            return self.color_cache[cache_key]

        if self.color_mode == ColorMode.BLOCK_TYPE:
            color = self._get_block_type_color(block_color)

        elif self.color_mode == ColorMode.ACTIVATION_LEVEL:
            color = self._get_activation_color(activation_level, block_color)

        elif self.color_mode == ColorMode.STATE_BASED:
            color = self._get_state_color(block_state, block_color)

        elif self.color_mode == ColorMode.CONNECTIVITY:
            color = self._get_connectivity_color(connection_count, block_color)

        elif self.color_mode == ColorMode.LEARNING_RATE:
            if custom_value is not None:
                color = self._get_learning_rate_color(custom_value, block_color)
            else:
                color = self._get_block_type_color(block_color)

        elif self.color_mode == ColorMode.CUSTOM:
            if custom_value is not None:
                color = self._get_custom_color(custom_value, block_color)
            else:
                color = self._get_block_type_color(block_color)
        else:
            color = self._get_block_type_color(block_color)

        # Cache the result
        self.color_cache[cache_key] = color
        return color

    def set_color_mode(self, mode: ColorMode) -> None:
        """Set color mode and clear cache"""
        if mode != self.color_mode:
            self.color_mode = mode
            self.color_cache.clear()

    def _get_block_type_color(self, block_color: BlockColor) -> Color:
        """Get color based on block type"""
        return self.theme.block_colors.get(block_color, Color(0.5, 0.5, 0.5))

    def _get_activation_color(self, activation_level: float, base_color: BlockColor) -> Color:
        """Get color based on activation level"""
        base = self.theme.block_colors.get(base_color, Color(0.5, 0.5, 0.5))

        # Interpolate between dim and bright versions
        dim_factor = 0.3
        bright_factor = 1.2

        factor = dim_factor + (bright_factor - dim_factor) * activation_level

        return Color(
            min(1.0, base.r * factor),
            min(1.0, base.g * factor),
            min(1.0, base.b * factor),
            base.a
        )

    def _get_state_color(self, block_state: BlockState, base_color: BlockColor) -> Color:
        """Get color based on block state"""
        if block_state in self.theme.state_colors:
            return self.theme.state_colors[block_state]

        # Fallback to modified base color
        base = self.theme.block_colors.get(base_color, Color(0.5, 0.5, 0.5))

        if block_state == BlockState.FIRING:
            return base.blend_with(self.theme.firing_color, 0.7)
        elif block_state == BlockState.REFRACTORY:
            return base.blend_with(self.theme.refractory_color, 0.5)
        elif block_state == BlockState.INACTIVE:
            return base.with_alpha(self.theme.inactive_alpha)
        else:
            return base

    def _get_connectivity_color(self, connection_count: int, base_color: BlockColor) -> Color:
        """Get color based on connectivity"""
        base = self.theme.block_colors.get(base_color, Color(0.5, 0.5, 0.5))

        # Normalize connection count (assuming max ~20 connections)
        normalized = min(1.0, connection_count / 20.0)

        # Blend toward white for high connectivity
        return base.blend_with(Color(1, 1, 1), normalized * 0.5)

    def _get_learning_rate_color(self, learning_rate: float, base_color: BlockColor) -> Color:
        """Get color based on learning rate"""
        base = self.theme.block_colors.get(base_color, Color(0.5, 0.5, 0.5))

        # Use gradient from blue (low) to red (high)
        if learning_rate < 0.5:
            # Blue to green
            t = learning_rate * 2
            return Color(0, t, 1 - t)
        else:
            # Green to red
            t = (learning_rate - 0.5) * 2
            return Color(t, 1 - t, 0)

    def _get_custom_color(self, value: float, base_color: BlockColor) -> Color:
        """Get color based on custom value"""
        # Use rainbow gradient for custom values
        hue = value * 360  # 0 to 360 degrees
        return self._hsv_to_rgb(hue, 0.8, 0.9)

    def _create_activation_gradient(self) -> List[Color]:
        """Create gradient for activation visualization"""
        gradient = []
        steps = 100

        for i in range(steps):
            t = i / (steps - 1)
            # Blue (inactive) to red (active)
            r = t
            g = 0.2 * (1 - abs(t - 0.5) * 2)  # Peak at middle
            b = 1 - t
            gradient.append(Color(r, g, b))

        return gradient

    def _create_state_gradient(self) -> List[Color]:
        """Create gradient for state visualization"""
        return [
            Color(0.3, 0.3, 0.3),  # Inactive
            Color(0.0, 0.8, 0.0),  # Active
            Color(1.0, 0.3, 0.3),  # Firing
            Color(1.0, 0.6, 0.0),  # Refractory
        ]

    def _hsv_to_rgb(self, h: float, s: float, v: float) -> Color:
        """Convert HSV to RGB color"""
        h = h % 360
        c = v * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = v - c

        if 0 <= h < 60:
            r, g, b = c, x, 0
        elif 60 <= h < 120:
            r, g, b = x, c, 0
        elif 120 <= h < 180:
            r, g, b = 0, c, x
        elif 180 <= h < 240:
            r, g, b = 0, x, c
        elif 240 <= h < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x

        return Color(r + m, g + m, b + m)


class StateColorMapping:
    """
    Color mapping for block states and transitions.

    Manages color representation of temporal state changes
    and state-based visualizations.
    """

    def __init__(self, theme: VisualTheme):
        """Initialize state color mapping"""
        self.theme = theme

        # State transition colors
        self.transition_colors = {
            (BlockState.INACTIVE, BlockState.ACTIVE): Color(0, 1, 0),
            (BlockState.ACTIVE, BlockState.FIRING): Color(1, 0.5, 0),
            (BlockState.FIRING, BlockState.REFRACTORY): Color(1, 0, 0),
            (BlockState.REFRACTORY, BlockState.INACTIVE): Color(0.5, 0.5, 0.5),
        }

        # State duration colors (for timing visualization)
        self.duration_gradient = self._create_duration_gradient()

    def get_state_color(self, state: BlockState, intensity: float = 1.0) -> Color:
        """Get color for a specific state with intensity"""
        base_color = self.theme.state_colors.get(state, Color(0.5, 0.5, 0.5))

        # Apply intensity
        return Color(
            base_color.r * intensity,
            base_color.g * intensity,
            base_color.b * intensity,
            base_color.a
        )

    def get_transition_color(self, from_state: BlockState, to_state: BlockState, progress: float = 0.5) -> Color:
        """Get color for state transition"""
        from_color = self.theme.state_colors.get(from_state, Color(0.5, 0.5, 0.5))
        to_color = self.theme.state_colors.get(to_state, Color(0.5, 0.5, 0.5))

        # Interpolate between states
        return from_color.blend_with(to_color, progress)

    def get_duration_color(self, state: BlockState, duration_ratio: float) -> Color:
        """Get color based on how long state has been active"""
        base_color = self.get_state_color(state)

        # Fade color based on duration (older = more transparent)
        fade_factor = max(0.3, 1.0 - duration_ratio)
        return base_color.with_alpha(fade_factor)

    def _create_duration_gradient(self) -> List[Color]:
        """Create gradient for duration visualization"""
        gradient = []
        steps = 50

        for i in range(steps):
            alpha = 1.0 - (i / steps) * 0.7  # Fade from 1.0 to 0.3
            gradient.append(Color(1, 1, 1, alpha))

        return gradient


class ThemeManager:
    """
    Central theme management system.

    Manages multiple visual themes and provides unified
    interface for theme switching and customization.
    """

    def __init__(self):
        """Initialize theme manager"""
        self.themes: Dict[str, VisualTheme] = {}
        self.current_theme_name = "default"

        # Create built-in themes
        self._create_builtin_themes()

        # Color mapping systems
        self.block_color_mapping = BlockColorMapping(self.get_current_theme())
        self.state_color_mapping = StateColorMapping(self.get_current_theme())

        # Theme change callbacks
        self.theme_change_callbacks: List[Callable[[VisualTheme], None]] = []

    def get_current_theme(self) -> VisualTheme:
        """Get currently active theme"""
        return self.themes[self.current_theme_name]

    def set_theme(self, theme_name: str) -> bool:
        """Set active theme"""
        if theme_name in self.themes:
            old_theme = self.current_theme_name
            self.current_theme_name = theme_name

            # Update color mapping systems
            new_theme = self.get_current_theme()
            self.block_color_mapping.theme = new_theme
            self.state_color_mapping.theme = new_theme

            # Clear caches
            self.block_color_mapping.color_cache.clear()

            # Notify callbacks
            for callback in self.theme_change_callbacks:
                callback(new_theme)

            return True
        return False

    def register_theme_change_callback(self, callback: Callable[[VisualTheme], None]) -> None:
        """Register callback for theme changes"""
        self.theme_change_callbacks.append(callback)

    def create_custom_theme(self, name: str, base_theme: str = "default") -> VisualTheme:
        """Create a custom theme based on existing theme"""
        if base_theme in self.themes:
            # Deep copy base theme
            base = self.themes[base_theme]
            custom_theme = VisualTheme(
                name=name,
                theme_type=ThemeType.CUSTOM,
                background_color=Color(base.background_color.r, base.background_color.g, base.background_color.b),
                canvas_color=Color(base.canvas_color.r, base.canvas_color.g, base.canvas_color.b),
                block_colors=base.block_colors.copy(),
                state_colors=base.state_colors.copy(),
                text_color=Color(base.text_color.r, base.text_color.g, base.text_color.b),
                border_color=Color(base.border_color.r, base.border_color.g, base.border_color.b),
                highlight_color=Color(base.highlight_color.r, base.highlight_color.g, base.highlight_color.b),
                selection_color=Color(base.selection_color.r, base.selection_color.g, base.selection_color.b),
                wire_color=Color(base.wire_color.r, base.wire_color.g, base.wire_color.b),
                active_wire_color=Color(base.active_wire_color.r, base.active_wire_color.g, base.active_wire_color.b),
                firing_color=Color(base.firing_color.r, base.firing_color.g, base.firing_color.b),
                refractory_color=Color(base.refractory_color.r, base.refractory_color.g, base.refractory_color.b),
                inactive_alpha=base.inactive_alpha,
                normal_alpha=base.normal_alpha,
                highlight_alpha=base.highlight_alpha
            )

            self.themes[name] = custom_theme
            return custom_theme
        else:
            raise ValueError(f"Base theme '{base_theme}' not found")

    def _create_builtin_themes(self) -> None:
        """Create built-in visual themes"""

        # Default theme
        default_theme = VisualTheme(
            name="default",
            theme_type=ThemeType.DEFAULT,
            background_color=Color(0.95, 0.95, 0.95),
            canvas_color=Color(1.0, 1.0, 1.0),
            block_colors={
                BlockColor.RED: Color(0.9, 0.2, 0.2),
                BlockColor.BLUE: Color(0.2, 0.4, 0.9),
                BlockColor.YELLOW: Color(0.9, 0.9, 0.2),
                BlockColor.GREEN: Color(0.2, 0.8, 0.3),
                BlockColor.PURPLE: Color(0.7, 0.2, 0.8)
            },
            state_colors={
                BlockState.INACTIVE: Color(0.6, 0.6, 0.6),
                BlockState.ACTIVE: Color(0.2, 0.8, 0.2),
                BlockState.FIRING: Color(1.0, 0.3, 0.3),
                BlockState.REFRACTORY: Color(1.0, 0.6, 0.0)
            }
        )

        # Light theme
        light_theme = VisualTheme(
            name="light",
            theme_type=ThemeType.LIGHT,
            background_color=Color(0.98, 0.98, 0.98),
            canvas_color=Color(1.0, 1.0, 1.0),
            block_colors={
                BlockColor.RED: Color(1.0, 0.6, 0.6),
                BlockColor.BLUE: Color(0.6, 0.7, 1.0),
                BlockColor.YELLOW: Color(1.0, 1.0, 0.6),
                BlockColor.GREEN: Color(0.6, 1.0, 0.7),
                BlockColor.PURPLE: Color(0.9, 0.6, 1.0)
            },
            state_colors={
                BlockState.INACTIVE: Color(0.8, 0.8, 0.8),
                BlockState.ACTIVE: Color(0.5, 0.9, 0.5),
                BlockState.FIRING: Color(1.0, 0.5, 0.5),
                BlockState.REFRACTORY: Color(1.0, 0.8, 0.4)
            },
            text_color=Color(0.2, 0.2, 0.2),
            border_color=Color(0.4, 0.4, 0.4)
        )

        # Dark theme
        dark_theme = VisualTheme(
            name="dark",
            theme_type=ThemeType.DARK,
            background_color=Color(0.1, 0.1, 0.1),
            canvas_color=Color(0.15, 0.15, 0.15),
            block_colors={
                BlockColor.RED: Color(0.8, 0.3, 0.3),
                BlockColor.BLUE: Color(0.3, 0.5, 0.8),
                BlockColor.YELLOW: Color(0.8, 0.8, 0.3),
                BlockColor.GREEN: Color(0.3, 0.7, 0.4),
                BlockColor.PURPLE: Color(0.6, 0.3, 0.7)
            },
            state_colors={
                BlockState.INACTIVE: Color(0.4, 0.4, 0.4),
                BlockState.ACTIVE: Color(0.3, 0.7, 0.3),
                BlockState.FIRING: Color(0.9, 0.4, 0.4),
                BlockState.REFRACTORY: Color(0.9, 0.7, 0.3)
            },
            text_color=Color(0.9, 0.9, 0.9),
            border_color=Color(0.6, 0.6, 0.6),
            wire_color=Color(0.5, 0.5, 0.5),
            active_wire_color=Color(0.4, 0.7, 1.0)
        )

        # High contrast theme
        high_contrast_theme = VisualTheme(
            name="high_contrast",
            theme_type=ThemeType.HIGH_CONTRAST,
            background_color=Color(0.0, 0.0, 0.0),
            canvas_color=Color(0.0, 0.0, 0.0),
            block_colors={
                BlockColor.RED: Color(1.0, 0.0, 0.0),
                BlockColor.BLUE: Color(0.0, 0.0, 1.0),
                BlockColor.YELLOW: Color(1.0, 1.0, 0.0),
                BlockColor.GREEN: Color(0.0, 1.0, 0.0),
                BlockColor.PURPLE: Color(1.0, 0.0, 1.0)
            },
            state_colors={
                BlockState.INACTIVE: Color(0.3, 0.3, 0.3),
                BlockState.ACTIVE: Color(0.0, 1.0, 0.0),
                BlockState.FIRING: Color(1.0, 0.0, 0.0),
                BlockState.REFRACTORY: Color(1.0, 1.0, 0.0)
            },
            text_color=Color(1.0, 1.0, 1.0),
            border_color=Color(1.0, 1.0, 1.0),
            highlight_color=Color(1.0, 1.0, 1.0),
            selection_color=Color(0.0, 1.0, 1.0)
        )

        # Scientific theme (monochromatic with focus on data)
        scientific_theme = VisualTheme(
            name="scientific",
            theme_type=ThemeType.SCIENTIFIC,
            background_color=Color(0.97, 0.97, 0.97),
            canvas_color=Color(1.0, 1.0, 1.0),
            block_colors={
                BlockColor.RED: Color(0.2, 0.2, 0.2),
                BlockColor.BLUE: Color(0.4, 0.4, 0.4),
                BlockColor.YELLOW: Color(0.6, 0.6, 0.6),
                BlockColor.GREEN: Color(0.3, 0.3, 0.3),
                BlockColor.PURPLE: Color(0.5, 0.5, 0.5)
            },
            state_colors={
                BlockState.INACTIVE: Color(0.8, 0.8, 0.8),
                BlockState.ACTIVE: Color(0.4, 0.4, 0.4),
                BlockState.FIRING: Color(0.0, 0.0, 0.0),
                BlockState.REFRACTORY: Color(0.6, 0.6, 0.6)
            },
            text_color=Color(0.0, 0.0, 0.0),
            border_color=Color(0.0, 0.0, 0.0),
            wire_color=Color(0.7, 0.7, 0.7),
            active_wire_color=Color(0.3, 0.3, 0.3)
        )

        # Store themes
        self.themes["default"] = default_theme
        self.themes["light"] = light_theme
        self.themes["dark"] = dark_theme
        self.themes["high_contrast"] = high_contrast_theme
        self.themes["scientific"] = scientific_theme

    def get_theme_names(self) -> List[str]:
        """Get list of available theme names"""
        return list(self.themes.keys())

    def get_theme_info(self) -> Dict[str, Any]:
        """Get information about current theme"""
        theme = self.get_current_theme()
        return {
            'name': theme.name,
            'type': theme.theme_type.value,
            'background_color': theme.background_color.to_hex(),
            'canvas_color': theme.canvas_color.to_hex(),
            'text_color': theme.text_color.to_hex(),
            'available_themes': self.get_theme_names(),
            'color_mode': self.block_color_mapping.color_mode.value
        }


# Utility functions for color operations

def interpolate_colors(color1: Color, color2: Color, steps: int) -> List[Color]:
    """Interpolate between two colors with specified steps"""
    colors = []
    for i in range(steps):
        t = i / (steps - 1) if steps > 1 else 0
        interpolated = color1.blend_with(color2, t)
        colors.append(interpolated)
    return colors


def create_heatmap_colors(min_value: float, max_value: float, steps: int = 256) -> Dict[float, Color]:
    """Create heatmap color mapping from blue (cold) to red (hot)"""
    color_map = {}

    for i in range(steps):
        value = min_value + (max_value - min_value) * i / (steps - 1)

        # Blue to cyan to green to yellow to red
        if i < steps // 4:
            # Blue to cyan
            t = i / (steps // 4)
            color = Color(0, t, 1)
        elif i < steps // 2:
            # Cyan to green
            t = (i - steps // 4) / (steps // 4)
            color = Color(0, 1, 1 - t)
        elif i < 3 * steps // 4:
            # Green to yellow
            t = (i - steps // 2) / (steps // 4)
            color = Color(t, 1, 0)
        else:
            # Yellow to red
            t = (i - 3 * steps // 4) / (steps // 4)
            color = Color(1, 1 - t, 0)

        color_map[value] = color

    return color_map


def adjust_color_brightness(color: Color, factor: float) -> Color:
    """Adjust color brightness by factor"""
    return Color(
        min(1.0, color.r * factor),
        min(1.0, color.g * factor),
        min(1.0, color.b * factor),
        color.a
    )


def adjust_color_saturation(color: Color, factor: float) -> Color:
    """Adjust color saturation by factor"""
    # Convert to HSV, adjust saturation, convert back
    # Simplified implementation
    gray = (color.r + color.g + color.b) / 3

    new_r = gray + (color.r - gray) * factor
    new_g = gray + (color.g - gray) * factor
    new_b = gray + (color.b - gray) * factor

    return Color(
        max(0, min(1, new_r)),
        max(0, min(1, new_g)),
        max(0, min(1, new_b)),
        color.a
    )