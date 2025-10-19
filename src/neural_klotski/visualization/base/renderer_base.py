"""
Base Renderer Class

Abstract base class for rendering backends. Defines the core rendering
interface that all visualization components use to draw to the screen.
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Any
import numpy as np


class RendererBase(ABC):
    """
    Abstract base class for rendering backends.

    Provides the core rendering interface that all visualization components
    use for drawing. Concrete implementations handle platform-specific
    rendering (e.g., matplotlib, OpenGL, etc.).
    """

    def __init__(self, width: int, height: int):
        """
        Initialize renderer.

        Args:
            width: Render target width in pixels
            height: Render target height in pixels
        """
        self.width = width
        self.height = height
        self.is_initialized = False

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the rendering backend"""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up rendering resources"""
        pass

    @abstractmethod
    def begin_frame(self) -> None:
        """Begin a new rendering frame"""
        pass

    @abstractmethod
    def end_frame(self) -> None:
        """End current rendering frame"""
        pass

    @abstractmethod
    def clear(self, color: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
        """
        Clear the render target.

        Args:
            color: Clear color as RGB tuple (0.0-1.0)
        """
        pass

    @abstractmethod
    def present(self) -> None:
        """Present the rendered frame to the display"""
        pass

    # Basic drawing primitives
    @abstractmethod
    def draw_point(self, x: float, y: float,
                  color: Tuple[float, float, float],
                  size: float = 1.0) -> None:
        """
        Draw a point.

        Args:
            x, y: Point coordinates
            color: RGB color (0.0-1.0)
            size: Point size in pixels
        """
        pass

    @abstractmethod
    def draw_line(self, x1: float, y1: float, x2: float, y2: float,
                 color: Tuple[float, float, float],
                 thickness: float = 1.0) -> None:
        """
        Draw a line.

        Args:
            x1, y1: Start point coordinates
            x2, y2: End point coordinates
            color: RGB color (0.0-1.0)
            thickness: Line thickness in pixels
        """
        pass

    @abstractmethod
    def draw_circle(self, x: float, y: float, radius: float,
                   color: Tuple[float, float, float],
                   filled: bool = True,
                   thickness: float = 1.0) -> None:
        """
        Draw a circle.

        Args:
            x, y: Center coordinates
            radius: Circle radius
            color: RGB color (0.0-1.0)
            filled: Whether to fill the circle
            thickness: Line thickness for unfilled circles
        """
        pass

    @abstractmethod
    def draw_rectangle(self, x: float, y: float, width: float, height: float,
                      color: Tuple[float, float, float],
                      filled: bool = True,
                      thickness: float = 1.0) -> None:
        """
        Draw a rectangle.

        Args:
            x, y: Top-left corner coordinates
            width, height: Rectangle dimensions
            color: RGB color (0.0-1.0)
            filled: Whether to fill the rectangle
            thickness: Line thickness for unfilled rectangles
        """
        pass

    @abstractmethod
    def draw_polygon(self, points: List[Tuple[float, float]],
                    color: Tuple[float, float, float],
                    filled: bool = True,
                    thickness: float = 1.0) -> None:
        """
        Draw a polygon.

        Args:
            points: List of (x, y) coordinates
            color: RGB color (0.0-1.0)
            filled: Whether to fill the polygon
            thickness: Line thickness for unfilled polygons
        """
        pass

    @abstractmethod
    def draw_text(self, x: float, y: float, text: str,
                 color: Tuple[float, float, float],
                 size: int = 12,
                 font_family: str = "Arial") -> None:
        """
        Draw text.

        Args:
            x, y: Text position
            text: Text string to draw
            color: RGB color (0.0-1.0)
            size: Font size in points
            font_family: Font family name
        """
        pass

    # Advanced drawing methods
    def draw_arrow(self, start_x: float, start_y: float,
                  end_x: float, end_y: float,
                  color: Tuple[float, float, float],
                  thickness: float = 1.0,
                  arrow_size: float = 10.0) -> None:
        """
        Draw an arrow.

        Args:
            start_x, start_y: Arrow start point
            end_x, end_y: Arrow end point (tip)
            color: RGB color (0.0-1.0)
            thickness: Line thickness
            arrow_size: Size of arrow head
        """
        # Draw main line
        self.draw_line(start_x, start_y, end_x, end_y, color, thickness)

        # Calculate arrow head
        import math
        angle = math.atan2(end_y - start_y, end_x - start_x)
        arrow_angle = math.pi / 6  # 30 degrees

        # Arrow head points
        p1_x = end_x - arrow_size * math.cos(angle - arrow_angle)
        p1_y = end_y - arrow_size * math.sin(angle - arrow_angle)
        p2_x = end_x - arrow_size * math.cos(angle + arrow_angle)
        p2_y = end_y - arrow_size * math.sin(angle + arrow_angle)

        # Draw arrow head
        self.draw_line(end_x, end_y, p1_x, p1_y, color, thickness)
        self.draw_line(end_x, end_y, p2_x, p2_y, color, thickness)

    def draw_grid(self, spacing: float,
                 color: Tuple[float, float, float] = (0.3, 0.3, 0.3),
                 thickness: float = 1.0,
                 bounds: Optional[Tuple[float, float, float, float]] = None) -> None:
        """
        Draw a grid.

        Args:
            spacing: Grid spacing
            color: Grid line color
            thickness: Line thickness
            bounds: Grid bounds (min_x, min_y, max_x, max_y), uses viewport if None
        """
        if bounds is None:
            bounds = self.get_viewport_bounds()

        min_x, min_y, max_x, max_y = bounds

        # Vertical lines
        x = min_x - (min_x % spacing)
        while x <= max_x:
            self.draw_line(x, min_y, x, max_y, color, thickness)
            x += spacing

        # Horizontal lines
        y = min_y - (min_y % spacing)
        while y <= max_y:
            self.draw_line(min_x, y, max_x, y, color, thickness)
            y += spacing

    def draw_crosshair(self, x: float, y: float,
                      color: Tuple[float, float, float],
                      size: float = 10.0,
                      thickness: float = 1.0) -> None:
        """
        Draw a crosshair.

        Args:
            x, y: Crosshair center
            color: RGB color
            size: Crosshair size
            thickness: Line thickness
        """
        self.draw_line(x - size/2, y, x + size/2, y, color, thickness)
        self.draw_line(x, y - size/2, x, y + size/2, color, thickness)

    # Coordinate transformation methods
    @abstractmethod
    def world_to_screen(self, world_x: float, world_y: float) -> Tuple[int, int]:
        """
        Convert world coordinates to screen pixels.

        Args:
            world_x, world_y: World coordinates

        Returns:
            Screen coordinates as (pixel_x, pixel_y)
        """
        pass

    @abstractmethod
    def screen_to_world(self, screen_x: int, screen_y: int) -> Tuple[float, float]:
        """
        Convert screen pixels to world coordinates.

        Args:
            screen_x, screen_y: Screen coordinates

        Returns:
            World coordinates as (world_x, world_y)
        """
        pass

    @abstractmethod
    def set_viewport(self, min_x: float, min_y: float, max_x: float, max_y: float) -> None:
        """
        Set the viewport (visible world region).

        Args:
            min_x, min_y: Viewport minimum coordinates
            max_x, max_y: Viewport maximum coordinates
        """
        pass

    @abstractmethod
    def get_viewport_bounds(self) -> Tuple[float, float, float, float]:
        """
        Get current viewport bounds.

        Returns:
            Viewport bounds as (min_x, min_y, max_x, max_y)
        """
        pass

    # Clipping and masking
    def push_clip_rect(self, x: float, y: float, width: float, height: float) -> None:
        """
        Push a clipping rectangle onto the clip stack.

        Args:
            x, y: Rectangle top-left corner
            width, height: Rectangle dimensions
        """
        # Default implementation does nothing - subclasses can override
        pass

    def pop_clip_rect(self) -> None:
        """Pop the last clipping rectangle from the clip stack"""
        # Default implementation does nothing - subclasses can override
        pass

    # State management
    def push_transform(self) -> None:
        """Push current transformation matrix onto stack"""
        # Default implementation does nothing - subclasses can override
        pass

    def pop_transform(self) -> None:
        """Pop transformation matrix from stack"""
        # Default implementation does nothing - subclasses can override
        pass

    def translate(self, dx: float, dy: float) -> None:
        """
        Apply translation to current transformation.

        Args:
            dx, dy: Translation offset
        """
        # Default implementation does nothing - subclasses can override
        pass

    def rotate(self, angle: float) -> None:
        """
        Apply rotation to current transformation.

        Args:
            angle: Rotation angle in radians
        """
        # Default implementation does nothing - subclasses can override
        pass

    def scale(self, sx: float, sy: float) -> None:
        """
        Apply scaling to current transformation.

        Args:
            sx, sy: Scale factors
        """
        # Default implementation does nothing - subclasses can override
        pass

    # Utility methods
    def resize(self, width: int, height: int) -> None:
        """
        Resize the render target.

        Args:
            width: New width in pixels
            height: New height in pixels
        """
        self.width = width
        self.height = height

    def get_size(self) -> Tuple[int, int]:
        """Get render target size"""
        return (self.width, self.height)

    def is_point_in_viewport(self, x: float, y: float) -> bool:
        """Check if world point is visible in current viewport"""
        min_x, min_y, max_x, max_y = self.get_viewport_bounds()
        return min_x <= x <= max_x and min_y <= y <= max_y

    def is_rect_in_viewport(self, x: float, y: float, width: float, height: float) -> bool:
        """Check if rectangle intersects with current viewport"""
        vmin_x, vmin_y, vmax_x, vmax_y = self.get_viewport_bounds()
        return not (x + width < vmin_x or x > vmax_x or
                   y + height < vmin_y or y > vmax_y)

    # Performance and debugging
    def get_render_stats(self) -> dict:
        """Get rendering statistics"""
        return {
            'width': self.width,
            'height': self.height,
            'initialized': self.is_initialized
        }

    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture current frame as numpy array.

        Returns:
            Frame data as RGB array or None if not supported
        """
        # Default implementation returns None - subclasses can override
        return None

    def __str__(self) -> str:
        """String representation"""
        return f"{self.__class__.__name__}(size={self.width}x{self.height}, initialized={self.is_initialized})"