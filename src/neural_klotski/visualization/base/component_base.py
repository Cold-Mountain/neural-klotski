"""
Base Component Class

Abstract base class for all visualization components. Provides the core
interface that all visual components must implement.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, List, Any
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from neural_klotski.visualization.config import VisualizationConfig, VisualizationTheme


class ComponentBase(ABC):
    """
    Abstract base class for all visualization components.

    Provides the standard interface that all visual components must implement,
    including update, render, and lifecycle management methods.
    """

    def __init__(self,
                 name: str,
                 config: VisualizationConfig,
                 theme: VisualizationTheme):
        """
        Initialize base component.

        Args:
            name: Unique name for this component
            config: Visualization configuration
            theme: Visual theme
        """
        self.name = name
        self.config = config
        self.theme = theme

        # Component state
        self.visible = True
        self.enabled = True
        self.initialized = False

        # Spatial properties
        self._bounds = (0.0, 0.0, 0.0, 0.0)  # min_x, min_y, max_x, max_y
        self._z_order = 0  # Rendering order (higher = front)

        # Performance tracking
        self.last_update_time = 0.0
        self.last_render_time = 0.0

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the component.

        This method must be implemented by subclasses to set up any
        required resources or state.
        """
        pass

    @abstractmethod
    def update(self, dt: float) -> None:
        """
        Update component state.

        Args:
            dt: Delta time since last update in seconds
        """
        pass

    @abstractmethod
    def render(self, renderer: 'RendererBase') -> None:
        """
        Render the component.

        Args:
            renderer: Rendering backend to use for drawing
        """
        pass

    def cleanup(self) -> None:
        """
        Clean up component resources.

        Base implementation handles common cleanup. Subclasses should
        call super().cleanup() and add their own cleanup logic.
        """
        self.initialized = False

    def reset(self) -> None:
        """
        Reset component to initial state.

        Base implementation does nothing. Subclasses should override
        to implement reset behavior.
        """
        pass

    # Visibility and state management
    @property
    def visible(self) -> bool:
        """Get component visibility"""
        return self._visible

    @visible.setter
    def visible(self, value: bool) -> None:
        """Set component visibility"""
        self._visible = value

    @property
    def enabled(self) -> bool:
        """Get component enabled state"""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Set component enabled state"""
        self._enabled = value

    @property
    def z_order(self) -> int:
        """Get component Z-order (rendering depth)"""
        return self._z_order

    @z_order.setter
    def z_order(self, value: int) -> None:
        """Set component Z-order"""
        self._z_order = value

    # Bounds management
    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Get component bounding box (min_x, min_y, max_x, max_y)"""
        return self._bounds

    def set_bounds(self, min_x: float, min_y: float, max_x: float, max_y: float) -> None:
        """Set component bounding box"""
        self._bounds = (min_x, min_y, max_x, max_y)

    def contains_point(self, x: float, y: float) -> bool:
        """Check if point is within component bounds"""
        min_x, min_y, max_x, max_y = self._bounds
        return min_x <= x <= max_x and min_y <= y <= max_y

    def intersects_bounds(self, other_bounds: Tuple[float, float, float, float]) -> bool:
        """Check if this component intersects with other bounds"""
        min_x1, min_y1, max_x1, max_y1 = self._bounds
        min_x2, min_y2, max_x2, max_y2 = other_bounds

        return not (max_x1 < min_x2 or max_x2 < min_x1 or
                   max_y1 < min_y2 or max_y2 < min_y1)

    # Configuration and theme updates
    def on_config_changed(self, new_config: VisualizationConfig) -> None:
        """Handle configuration changes"""
        self.config = new_config

    def on_theme_changed(self, new_theme: VisualizationTheme) -> None:
        """Handle theme changes"""
        self.theme = new_theme

    # Event handling
    def on_mouse_click(self, x: float, y: float, button: str) -> bool:
        """
        Handle mouse click events.

        Args:
            x, y: Mouse position in world coordinates
            button: Mouse button ('left', 'right', 'middle')

        Returns:
            True if event was handled, False otherwise
        """
        return False

    def on_mouse_move(self, x: float, y: float) -> bool:
        """
        Handle mouse move events.

        Args:
            x, y: Mouse position in world coordinates

        Returns:
            True if event was handled, False otherwise
        """
        return False

    def on_key_press(self, key: str) -> bool:
        """
        Handle key press events.

        Args:
            key: Key that was pressed

        Returns:
            True if event was handled, False otherwise
        """
        return False

    def on_window_resize(self, width: int, height: int) -> None:
        """
        Handle window resize events.

        Args:
            width: New window width
            height: New window height
        """
        pass

    # Performance monitoring
    def get_performance_info(self) -> dict:
        """Get component performance information"""
        return {
            'name': self.name,
            'visible': self.visible,
            'enabled': self.enabled,
            'last_update_time': self.last_update_time,
            'last_render_time': self.last_render_time,
            'bounds': self._bounds,
            'z_order': self._z_order
        }

    # Validation
    def validate(self) -> List[str]:
        """
        Validate component state.

        Returns:
            List of validation issues (empty if valid)
        """
        issues = []

        if not self.initialized:
            issues.append(f"Component '{self.name}' not initialized")

        if not self.name:
            issues.append("Component has no name")

        if self.config is None:
            issues.append("Component has no configuration")

        if self.theme is None:
            issues.append("Component has no theme")

        return issues

    # Utility methods
    def mark_dirty(self) -> None:
        """Mark component as needing re-render"""
        # Base implementation does nothing, but subclasses can override
        # to implement dirty tracking for performance optimization
        pass

    def is_dirty(self) -> bool:
        """Check if component needs re-render"""
        # Base implementation always returns True, but subclasses can override
        # to implement dirty tracking
        return True

    def get_center(self) -> Tuple[float, float]:
        """Get center point of component bounds"""
        min_x, min_y, max_x, max_y = self._bounds
        return ((min_x + max_x) / 2, (min_y + max_y) / 2)

    def get_size(self) -> Tuple[float, float]:
        """Get size of component bounds"""
        min_x, min_y, max_x, max_y = self._bounds
        return (max_x - min_x, max_y - min_y)

    def expand_bounds(self, margin: float) -> None:
        """Expand component bounds by margin"""
        min_x, min_y, max_x, max_y = self._bounds
        self._bounds = (min_x - margin, min_y - margin,
                       max_x + margin, max_y + margin)

    def translate_bounds(self, dx: float, dy: float) -> None:
        """Translate component bounds by offset"""
        min_x, min_y, max_x, max_y = self._bounds
        self._bounds = (min_x + dx, min_y + dy,
                       max_x + dx, max_y + dy)

    def __str__(self) -> str:
        """String representation of component"""
        return f"{self.__class__.__name__}(name='{self.name}', visible={self.visible}, enabled={self.enabled})"

    def __repr__(self) -> str:
        """Detailed string representation"""
        return (f"{self.__class__.__name__}(name='{self.name}', "
                f"visible={self.visible}, enabled={self.enabled}, "
                f"bounds={self._bounds}, z_order={self._z_order})")


class ContainerComponent(ComponentBase):
    """
    Base class for components that contain other components.

    Provides functionality for managing child components in a hierarchical
    structure.
    """

    def __init__(self, name: str, config: VisualizationConfig, theme: VisualizationTheme):
        super().__init__(name, config, theme)
        self.children: List[ComponentBase] = []

    def add_child(self, child: ComponentBase) -> None:
        """Add child component"""
        if child not in self.children:
            self.children.append(child)
            if not child.initialized and self.initialized:
                child.initialize()

    def remove_child(self, child: ComponentBase) -> None:
        """Remove child component"""
        if child in self.children:
            self.children.remove(child)
            child.cleanup()

    def get_child(self, name: str) -> Optional[ComponentBase]:
        """Get child component by name"""
        for child in self.children:
            if child.name == name:
                return child
        return None

    def clear_children(self) -> None:
        """Remove all child components"""
        for child in self.children:
            child.cleanup()
        self.children.clear()

    def initialize(self) -> None:
        """Initialize container and all children"""
        super().initialize()
        for child in self.children:
            if not child.initialized:
                child.initialize()

    def update(self, dt: float) -> None:
        """Update container and all visible children"""
        if not self.enabled:
            return

        for child in self.children:
            if child.visible and child.enabled:
                child.update(dt)

    def render(self, renderer: 'RendererBase') -> None:
        """Render container and all visible children in Z-order"""
        if not self.visible:
            return

        # Sort children by Z-order for proper depth rendering
        sorted_children = sorted(self.children, key=lambda c: c.z_order)

        for child in sorted_children:
            if child.visible:
                child.render(renderer)

    def cleanup(self) -> None:
        """Clean up container and all children"""
        for child in self.children:
            child.cleanup()
        super().cleanup()

    def on_config_changed(self, new_config: VisualizationConfig) -> None:
        """Propagate config changes to children"""
        super().on_config_changed(new_config)
        for child in self.children:
            child.on_config_changed(new_config)

    def on_theme_changed(self, new_theme: VisualizationTheme) -> None:
        """Propagate theme changes to children"""
        super().on_theme_changed(new_theme)
        for child in self.children:
            child.on_theme_changed(new_theme)

    def validate(self) -> List[str]:
        """Validate container and all children"""
        issues = super().validate()

        for child in self.children:
            child_issues = child.validate()
            issues.extend([f"{child.name}: {issue}" for issue in child_issues])

        return issues