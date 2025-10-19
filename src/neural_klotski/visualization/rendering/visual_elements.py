"""
Visual Elements for Neural-Klotski Block Rendering

Comprehensive visual representation components for blocks including
various shapes, labels, indicators, and interactive elements.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable
from enum import Enum
import math
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from neural_klotski.core.block import BlockColor, Block
from neural_klotski.visualization.layout.coordinate_system import Point2D, BoundingBox


class ShapeType(Enum):
    """Types of block shapes"""
    CIRCLE = "circle"
    RECTANGLE = "rectangle"
    HEXAGON = "hexagon"
    DIAMOND = "diamond"
    TRIANGLE = "triangle"


class RenderState(Enum):
    """Visual rendering states"""
    NORMAL = "normal"
    HIGHLIGHTED = "highlighted"
    SELECTED = "selected"
    FIRING = "firing"
    REFRACTORY = "refractory"
    INACTIVE = "inactive"


@dataclass
class Color:
    """RGBA color representation"""
    r: float  # 0.0 to 1.0
    g: float  # 0.0 to 1.0
    b: float  # 0.0 to 1.0
    a: float = 1.0  # Alpha (transparency)

    def to_hex(self) -> str:
        """Convert to hex color string"""
        r_hex = f"{int(self.r * 255):02x}"
        g_hex = f"{int(self.g * 255):02x}"
        b_hex = f"{int(self.b * 255):02x}"
        return f"#{r_hex}{g_hex}{b_hex}"

    def to_rgba_tuple(self) -> Tuple[float, float, float, float]:
        """Convert to RGBA tuple"""
        return (self.r, self.g, self.b, self.a)

    def with_alpha(self, alpha: float) -> 'Color':
        """Create new color with different alpha"""
        return Color(self.r, self.g, self.b, alpha)

    def blend_with(self, other: 'Color', factor: float) -> 'Color':
        """Blend with another color"""
        inv_factor = 1.0 - factor
        return Color(
            self.r * inv_factor + other.r * factor,
            self.g * inv_factor + other.g * factor,
            self.b * inv_factor + other.b * factor,
            self.a * inv_factor + other.a * factor
        )

    @staticmethod
    def from_hex(hex_color: str) -> 'Color':
        """Create color from hex string"""
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 6:
            r = int(hex_color[0:2], 16) / 255.0
            g = int(hex_color[2:4], 16) / 255.0
            b = int(hex_color[4:6], 16) / 255.0
            return Color(r, g, b)
        else:
            raise ValueError(f"Invalid hex color: {hex_color}")

    @staticmethod
    def from_block_color(block_color: BlockColor) -> 'Color':
        """Create color from BlockColor enum"""
        color_map = {
            BlockColor.RED: Color(0.9, 0.2, 0.2),
            BlockColor.BLUE: Color(0.2, 0.4, 0.9),
            BlockColor.YELLOW: Color(0.9, 0.9, 0.2),
            BlockColor.GREEN: Color(0.2, 0.8, 0.3),
            BlockColor.PURPLE: Color(0.7, 0.2, 0.8)
        }
        return color_map.get(block_color, Color(0.5, 0.5, 0.5))


@dataclass
class VisualStyle:
    """Visual styling properties for elements"""
    # Colors
    fill_color: Color = field(default_factory=lambda: Color(0.5, 0.5, 0.5))
    border_color: Color = field(default_factory=lambda: Color(0.0, 0.0, 0.0))
    text_color: Color = field(default_factory=lambda: Color(1.0, 1.0, 1.0))

    # Sizing
    border_width: float = 2.0
    font_size: float = 12.0

    # Effects
    shadow_enabled: bool = True
    shadow_color: Color = field(default_factory=lambda: Color(0.0, 0.0, 0.0, 0.3))
    shadow_offset: Point2D = field(default_factory=lambda: Point2D(2, 2))
    shadow_blur: float = 4.0

    # Gradients
    gradient_enabled: bool = False
    gradient_start: Color = field(default_factory=lambda: Color(0.8, 0.8, 0.8))
    gradient_end: Color = field(default_factory=lambda: Color(0.4, 0.4, 0.4))

    def copy(self) -> 'VisualStyle':
        """Create a copy of this style"""
        return VisualStyle(
            fill_color=Color(self.fill_color.r, self.fill_color.g, self.fill_color.b, self.fill_color.a),
            border_color=Color(self.border_color.r, self.border_color.g, self.border_color.b, self.border_color.a),
            text_color=Color(self.text_color.r, self.text_color.g, self.text_color.b, self.text_color.a),
            border_width=self.border_width,
            font_size=self.font_size,
            shadow_enabled=self.shadow_enabled,
            shadow_color=Color(self.shadow_color.r, self.shadow_color.g, self.shadow_color.b, self.shadow_color.a),
            shadow_offset=Point2D(self.shadow_offset.x, self.shadow_offset.y),
            shadow_blur=self.shadow_blur,
            gradient_enabled=self.gradient_enabled,
            gradient_start=Color(self.gradient_start.r, self.gradient_start.g, self.gradient_start.b, self.gradient_start.a),
            gradient_end=Color(self.gradient_end.r, self.gradient_end.g, self.gradient_end.b, self.gradient_end.a)
        )


class VisualElement(ABC):
    """
    Abstract base class for visual elements.

    Provides common interface for all renderable components
    including blocks, labels, and indicators.
    """

    def __init__(self, element_id: str, position: Point2D, size: Point2D):
        """Initialize visual element"""
        self.element_id = element_id
        self.position = position
        self.size = size
        self.style = VisualStyle()

        # State
        self.visible = True
        self.render_state = RenderState.NORMAL
        self.z_order = 0

        # Animation
        self.animation_progress = 0.0
        self.is_animating = False

        # Interaction
        self.interactive = True
        self.hover_style: Optional[VisualStyle] = None
        self.selected_style: Optional[VisualStyle] = None

    @abstractmethod
    def get_bounds(self) -> BoundingBox:
        """Get bounding box of element"""
        pass

    @abstractmethod
    def contains_point(self, point: Point2D) -> bool:
        """Check if point is inside element"""
        pass

    @abstractmethod
    def render_geometry(self) -> List[Any]:
        """Get geometry for rendering"""
        pass

    def get_current_style(self) -> VisualStyle:
        """Get current style based on render state"""
        if self.render_state == RenderState.SELECTED and self.selected_style:
            return self.selected_style
        elif self.render_state == RenderState.HIGHLIGHTED and self.hover_style:
            return self.hover_style
        else:
            return self.style

    def set_position(self, position: Point2D) -> None:
        """Set element position"""
        self.position = position

    def set_size(self, size: Point2D) -> None:
        """Set element size"""
        self.size = size

    def set_render_state(self, state: RenderState) -> None:
        """Set visual render state"""
        self.render_state = state

    def update_animation(self, delta_time: float) -> bool:
        """
        Update animation state.

        Returns:
            True if animation is still running
        """
        if not self.is_animating:
            return False

        # Simple linear animation for base class
        self.animation_progress += delta_time * 2.0  # 0.5 second duration
        if self.animation_progress >= 1.0:
            self.animation_progress = 1.0
            self.is_animating = False

        return self.is_animating


class CircularBlock(VisualElement):
    """
    Circular block visual element.

    Represents blocks as circles with optional inner indicators
    for activation level and other properties.
    """

    def __init__(self, element_id: str, position: Point2D, radius: float):
        """Initialize circular block"""
        super().__init__(element_id, position, Point2D(radius * 2, radius * 2))
        self.radius = radius

        # Circular-specific properties
        self.inner_radius_ratio = 0.7  # For hollow circles
        self.activation_indicator = True
        self.activation_level = 0.0  # 0.0 to 1.0

        # Visual effects
        self.pulse_enabled = False
        self.pulse_speed = 2.0
        self.pulse_amplitude = 0.1

    def get_bounds(self) -> BoundingBox:
        """Get bounding box of circular block"""
        return BoundingBox(
            self.position.x - self.radius,
            self.position.y - self.radius,
            self.position.x + self.radius,
            self.position.y + self.radius
        )

    def contains_point(self, point: Point2D) -> bool:
        """Check if point is inside circle"""
        distance = self.position.distance_to(point)
        return distance <= self.radius

    def render_geometry(self) -> List[Dict[str, Any]]:
        """Get geometry for rendering circular block"""
        geometry = []
        current_style = self.get_current_style()

        # Calculate effective radius (with pulsing if enabled)
        effective_radius = self.radius
        if self.pulse_enabled and self.is_animating:
            pulse_factor = 1.0 + self.pulse_amplitude * math.sin(self.animation_progress * self.pulse_speed * 2 * math.pi)
            effective_radius *= pulse_factor

        # Main circle
        circle_geom = {
            'type': 'circle',
            'center': self.position,
            'radius': effective_radius,
            'fill_color': current_style.fill_color,
            'border_color': current_style.border_color,
            'border_width': current_style.border_width,
            'z_order': self.z_order
        }

        # Add shadow if enabled
        if current_style.shadow_enabled:
            shadow_geom = {
                'type': 'circle',
                'center': Point2D(
                    self.position.x + current_style.shadow_offset.x,
                    self.position.y + current_style.shadow_offset.y
                ),
                'radius': effective_radius,
                'fill_color': current_style.shadow_color,
                'border_color': Color(0, 0, 0, 0),  # Transparent border
                'border_width': 0,
                'z_order': self.z_order - 1,
                'blur': current_style.shadow_blur
            }
            geometry.append(shadow_geom)

        geometry.append(circle_geom)

        # Activation level indicator (inner circle)
        if self.activation_indicator and self.activation_level > 0:
            activation_radius = effective_radius * self.inner_radius_ratio * self.activation_level
            activation_color = current_style.fill_color.blend_with(Color(1, 1, 1), 0.5)

            activation_geom = {
                'type': 'circle',
                'center': self.position,
                'radius': activation_radius,
                'fill_color': activation_color,
                'border_color': Color(0, 0, 0, 0),
                'border_width': 0,
                'z_order': self.z_order + 1
            }
            geometry.append(activation_geom)

        return geometry

    def set_activation_level(self, level: float) -> None:
        """Set activation level (0.0 to 1.0)"""
        self.activation_level = max(0.0, min(1.0, level))

    def start_pulse_animation(self) -> None:
        """Start pulsing animation"""
        self.pulse_enabled = True
        self.is_animating = True
        self.animation_progress = 0.0

    def stop_pulse_animation(self) -> None:
        """Stop pulsing animation"""
        self.pulse_enabled = False
        self.is_animating = False


class RectangularBlock(VisualElement):
    """
    Rectangular block visual element.

    Represents blocks as rectangles/squares with optional
    progress bars and state indicators.
    """

    def __init__(self, element_id: str, position: Point2D, size: Point2D, corner_radius: float = 0.0):
        """Initialize rectangular block"""
        super().__init__(element_id, position, size)
        self.corner_radius = corner_radius

        # Rectangular-specific properties
        self.progress_bar_enabled = True
        self.progress_value = 0.0  # 0.0 to 1.0
        self.progress_height = 4.0

        # State indicators
        self.state_indicator_enabled = True
        self.state_indicator_size = 8.0

    def get_bounds(self) -> BoundingBox:
        """Get bounding box of rectangular block"""
        half_width = self.size.x / 2
        half_height = self.size.y / 2

        return BoundingBox(
            self.position.x - half_width,
            self.position.y - half_height,
            self.position.x + half_width,
            self.position.y + half_height
        )

    def contains_point(self, point: Point2D) -> bool:
        """Check if point is inside rectangle"""
        bounds = self.get_bounds()
        return bounds.contains(point)

    def render_geometry(self) -> List[Dict[str, Any]]:
        """Get geometry for rendering rectangular block"""
        geometry = []
        current_style = self.get_current_style()

        # Main rectangle
        bounds = self.get_bounds()
        rect_geom = {
            'type': 'rectangle',
            'bounds': bounds,
            'corner_radius': self.corner_radius,
            'fill_color': current_style.fill_color,
            'border_color': current_style.border_color,
            'border_width': current_style.border_width,
            'z_order': self.z_order
        }

        # Add shadow if enabled
        if current_style.shadow_enabled:
            shadow_bounds = BoundingBox(
                bounds.min_x + current_style.shadow_offset.x,
                bounds.min_y + current_style.shadow_offset.y,
                bounds.max_x + current_style.shadow_offset.x,
                bounds.max_y + current_style.shadow_offset.y
            )
            shadow_geom = {
                'type': 'rectangle',
                'bounds': shadow_bounds,
                'corner_radius': self.corner_radius,
                'fill_color': current_style.shadow_color,
                'border_color': Color(0, 0, 0, 0),
                'border_width': 0,
                'z_order': self.z_order - 1,
                'blur': current_style.shadow_blur
            }
            geometry.append(shadow_geom)

        geometry.append(rect_geom)

        # Progress bar
        if self.progress_bar_enabled and self.progress_value > 0:
            progress_width = self.size.x * self.progress_value
            progress_bounds = BoundingBox(
                bounds.min_x,
                bounds.max_y - self.progress_height,
                bounds.min_x + progress_width,
                bounds.max_y
            )

            progress_color = current_style.fill_color.blend_with(Color(0, 1, 0), 0.7)
            progress_geom = {
                'type': 'rectangle',
                'bounds': progress_bounds,
                'corner_radius': 0,
                'fill_color': progress_color,
                'border_color': Color(0, 0, 0, 0),
                'border_width': 0,
                'z_order': self.z_order + 1
            }
            geometry.append(progress_geom)

        # State indicator (small circle in corner)
        if self.state_indicator_enabled:
            indicator_color = self._get_state_indicator_color()
            if indicator_color:
                indicator_center = Point2D(
                    bounds.max_x - self.state_indicator_size,
                    bounds.min_y + self.state_indicator_size
                )

                indicator_geom = {
                    'type': 'circle',
                    'center': indicator_center,
                    'radius': self.state_indicator_size / 2,
                    'fill_color': indicator_color,
                    'border_color': Color(1, 1, 1),
                    'border_width': 1,
                    'z_order': self.z_order + 2
                }
                geometry.append(indicator_geom)

        return geometry

    def set_progress_value(self, value: float) -> None:
        """Set progress bar value (0.0 to 1.0)"""
        self.progress_value = max(0.0, min(1.0, value))

    def _get_state_indicator_color(self) -> Optional[Color]:
        """Get color for state indicator based on render state"""
        state_colors = {
            RenderState.FIRING: Color(1, 0, 0),       # Red
            RenderState.REFRACTORY: Color(1, 0.5, 0), # Orange
            RenderState.SELECTED: Color(0, 1, 0),     # Green
            RenderState.HIGHLIGHTED: Color(0, 0, 1),  # Blue
        }
        return state_colors.get(self.render_state)


class HexagonalBlock(VisualElement):
    """
    Hexagonal block visual element.

    Represents blocks as hexagons with optional internal
    structure visualization and directional indicators.
    """

    def __init__(self, element_id: str, position: Point2D, radius: float):
        """Initialize hexagonal block"""
        super().__init__(element_id, position, Point2D(radius * 2, radius * 2))
        self.radius = radius

        # Hexagonal-specific properties
        self.orientation = 0.0  # Rotation in radians
        self.internal_structure = True
        self.directional_indicator = False
        self.direction_angle = 0.0

    def get_bounds(self) -> BoundingBox:
        """Get bounding box of hexagonal block"""
        # Hexagon bounds (approximate)
        hex_width = self.radius * 2
        hex_height = self.radius * math.sqrt(3)

        return BoundingBox(
            self.position.x - hex_width / 2,
            self.position.y - hex_height / 2,
            self.position.x + hex_width / 2,
            self.position.y + hex_height / 2
        )

    def contains_point(self, point: Point2D) -> bool:
        """Check if point is inside hexagon"""
        # Simplified hexagon collision (using circle approximation)
        distance = self.position.distance_to(point)
        return distance <= self.radius * 0.9

    def render_geometry(self) -> List[Dict[str, Any]]:
        """Get geometry for rendering hexagonal block"""
        geometry = []
        current_style = self.get_current_style()

        # Calculate hexagon vertices
        vertices = []
        for i in range(6):
            angle = (i * math.pi / 3) + self.orientation
            x = self.position.x + self.radius * math.cos(angle)
            y = self.position.y + self.radius * math.sin(angle)
            vertices.append(Point2D(x, y))

        # Main hexagon
        hex_geom = {
            'type': 'polygon',
            'vertices': vertices,
            'fill_color': current_style.fill_color,
            'border_color': current_style.border_color,
            'border_width': current_style.border_width,
            'z_order': self.z_order
        }

        # Add shadow if enabled
        if current_style.shadow_enabled:
            shadow_vertices = []
            for vertex in vertices:
                shadow_vertices.append(Point2D(
                    vertex.x + current_style.shadow_offset.x,
                    vertex.y + current_style.shadow_offset.y
                ))

            shadow_geom = {
                'type': 'polygon',
                'vertices': shadow_vertices,
                'fill_color': current_style.shadow_color,
                'border_color': Color(0, 0, 0, 0),
                'border_width': 0,
                'z_order': self.z_order - 1,
                'blur': current_style.shadow_blur
            }
            geometry.append(shadow_geom)

        geometry.append(hex_geom)

        # Internal structure (smaller hexagon)
        if self.internal_structure:
            inner_radius = self.radius * 0.6
            inner_vertices = []
            for i in range(6):
                angle = (i * math.pi / 3) + self.orientation
                x = self.position.x + inner_radius * math.cos(angle)
                y = self.position.y + inner_radius * math.sin(angle)
                inner_vertices.append(Point2D(x, y))

            inner_color = current_style.fill_color.blend_with(Color(1, 1, 1), 0.3)
            inner_geom = {
                'type': 'polygon',
                'vertices': inner_vertices,
                'fill_color': inner_color,
                'border_color': current_style.border_color,
                'border_width': 1,
                'z_order': self.z_order + 1
            }
            geometry.append(inner_geom)

        # Directional indicator
        if self.directional_indicator:
            indicator_length = self.radius * 0.8
            end_x = self.position.x + indicator_length * math.cos(self.direction_angle)
            end_y = self.position.y + indicator_length * math.sin(self.direction_angle)

            direction_geom = {
                'type': 'line',
                'start': self.position,
                'end': Point2D(end_x, end_y),
                'color': current_style.border_color,
                'width': 3,
                'z_order': self.z_order + 2
            }
            geometry.append(direction_geom)

        return geometry

    def set_orientation(self, angle: float) -> None:
        """Set hexagon orientation in radians"""
        self.orientation = angle

    def set_direction(self, angle: float) -> None:
        """Set directional indicator angle"""
        self.direction_angle = angle
        self.directional_indicator = True


class BlockLabel(VisualElement):
    """
    Text label for blocks.

    Displays block ID, activation values, or other
    textual information associated with blocks.
    """

    def __init__(self, element_id: str, position: Point2D, text: str, font_size: float = 12.0):
        """Initialize block label"""
        # Estimate text size (rough approximation)
        text_width = len(text) * font_size * 0.6
        text_height = font_size * 1.2
        super().__init__(element_id, position, Point2D(text_width, text_height))

        self.text = text
        self.font_size = font_size
        self.font_family = "Arial"
        self.alignment = "center"  # "left", "center", "right"

    def get_bounds(self) -> BoundingBox:
        """Get bounding box of text label"""
        half_width = self.size.x / 2
        half_height = self.size.y / 2

        return BoundingBox(
            self.position.x - half_width,
            self.position.y - half_height,
            self.position.x + half_width,
            self.position.y + half_height
        )

    def contains_point(self, point: Point2D) -> bool:
        """Check if point is inside text bounds"""
        bounds = self.get_bounds()
        return bounds.contains(point)

    def render_geometry(self) -> List[Dict[str, Any]]:
        """Get geometry for rendering text label"""
        current_style = self.get_current_style()

        text_geom = {
            'type': 'text',
            'position': self.position,
            'text': self.text,
            'font_size': self.font_size,
            'font_family': self.font_family,
            'color': current_style.text_color,
            'alignment': self.alignment,
            'z_order': self.z_order
        }

        return [text_geom]

    def set_text(self, text: str) -> None:
        """Update label text"""
        self.text = text
        # Update size estimate
        text_width = len(text) * self.font_size * 0.6
        self.size = Point2D(text_width, self.font_size * 1.2)


class BlockIndicator(VisualElement):
    """
    Visual indicator for block states.

    Small visual elements that show specific block
    properties like firing state, connectivity, etc.
    """

    def __init__(self, element_id: str, position: Point2D, indicator_type: str, size: float = 8.0):
        """Initialize block indicator"""
        super().__init__(element_id, position, Point2D(size, size))
        self.indicator_type = indicator_type  # "firing", "connection", "threshold", etc.
        self.indicator_size = size
        self.value = 0.0  # Indicator value (0.0 to 1.0)

    def get_bounds(self) -> BoundingBox:
        """Get bounding box of indicator"""
        half_size = self.indicator_size / 2
        return BoundingBox(
            self.position.x - half_size,
            self.position.y - half_size,
            self.position.x + half_size,
            self.position.y + half_size
        )

    def contains_point(self, point: Point2D) -> bool:
        """Check if point is inside indicator"""
        distance = self.position.distance_to(point)
        return distance <= self.indicator_size / 2

    def render_geometry(self) -> List[Dict[str, Any]]:
        """Get geometry for rendering indicator"""
        geometry = []

        if self.indicator_type == "firing":
            # Pulsing red circle for firing state
            intensity = 0.5 + 0.5 * math.sin(self.animation_progress * 4 * math.pi)
            color = Color(1.0, 0.0, 0.0, intensity)

            firing_geom = {
                'type': 'circle',
                'center': self.position,
                'radius': self.indicator_size / 2,
                'fill_color': color,
                'border_color': Color(1, 1, 1),
                'border_width': 1,
                'z_order': self.z_order
            }
            geometry.append(firing_geom)

        elif self.indicator_type == "connection":
            # Small square showing connection strength
            alpha = self.value
            color = Color(0.0, 0.8, 0.0, alpha)

            connection_bounds = BoundingBox(
                self.position.x - self.indicator_size / 2,
                self.position.y - self.indicator_size / 2,
                self.position.x + self.indicator_size / 2,
                self.position.y + self.indicator_size / 2
            )

            connection_geom = {
                'type': 'rectangle',
                'bounds': connection_bounds,
                'corner_radius': 1,
                'fill_color': color,
                'border_color': Color(0, 0, 0),
                'border_width': 1,
                'z_order': self.z_order
            }
            geometry.append(connection_geom)

        elif self.indicator_type == "threshold":
            # Triangular indicator for threshold state
            size = self.indicator_size / 2
            vertices = [
                Point2D(self.position.x, self.position.y - size),
                Point2D(self.position.x - size, self.position.y + size),
                Point2D(self.position.x + size, self.position.y + size)
            ]

            color_intensity = self.value
            color = Color(1.0, 1.0, 0.0, color_intensity)

            threshold_geom = {
                'type': 'polygon',
                'vertices': vertices,
                'fill_color': color,
                'border_color': Color(0, 0, 0),
                'border_width': 1,
                'z_order': self.z_order
            }
            geometry.append(threshold_geom)

        return geometry

    def set_value(self, value: float) -> None:
        """Set indicator value (0.0 to 1.0)"""
        self.value = max(0.0, min(1.0, value))

    def start_animation(self) -> None:
        """Start indicator animation"""
        self.is_animating = True
        self.animation_progress = 0.0