"""
Viewport and Camera System for Neural-Klotski Visualization

Interactive viewport management with zoom, pan, and camera controls
for navigating the 2D visualization space.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Callable, Dict, Any, List
from enum import Enum
import math
import time
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from neural_klotski.visualization.layout.coordinate_system import Point2D, BoundingBox, CoordinateSystem


class ViewportMode(Enum):
    """Viewport interaction modes"""
    NORMAL = "normal"
    PAN = "pan"
    ZOOM = "zoom"
    FIT_TO_CONTENT = "fit_to_content"
    FOLLOW_TARGET = "follow_target"


@dataclass
class ViewportConfig:
    """Configuration for viewport behavior"""
    # Zoom settings
    min_zoom: float = 0.1
    max_zoom: float = 10.0
    zoom_speed: float = 1.2
    smooth_zoom: bool = True

    # Pan settings
    pan_speed: float = 1.0
    smooth_pan: bool = True
    constrain_pan: bool = True

    # Animation settings
    animation_duration: float = 0.3
    easing_function: str = "ease_out"  # "linear", "ease_in", "ease_out", "ease_in_out"

    # Bounds
    world_bounds: Optional[BoundingBox] = None
    maintain_aspect_ratio: bool = True

    # Auto-fit settings
    fit_margin: float = 50.0
    auto_fit_on_resize: bool = True


@dataclass
class ViewportState:
    """Current viewport state"""
    zoom: float = 1.0
    pan_offset: Point2D = field(default_factory=lambda: Point2D(0, 0))
    screen_size: Tuple[int, int] = (800, 600)
    mode: ViewportMode = ViewportMode.NORMAL

    # Animation state
    is_animating: bool = False
    animation_start_time: float = 0.0
    animation_target_zoom: float = 1.0
    animation_target_pan: Point2D = field(default_factory=lambda: Point2D(0, 0))
    animation_start_zoom: float = 1.0
    animation_start_pan: Point2D = field(default_factory=lambda: Point2D(0, 0))


class Camera:
    """
    Virtual camera for viewport navigation.

    Handles smooth camera movements, tracking, and automatic framing.
    """

    def __init__(self, coordinate_system: CoordinateSystem):
        """Initialize camera"""
        self.coordinate_system = coordinate_system
        self.state = ViewportState()

        # Target tracking
        self.target_position: Optional[Point2D] = None
        self.target_zoom: Optional[float] = None

        # Movement smoothing
        self.position_velocity = Point2D(0, 0)
        self.zoom_velocity = 0.0
        self.smoothing_factor = 0.15

    def set_position(self, world_position: Point2D, animate: bool = False) -> None:
        """Set camera position in world coordinates"""
        if animate:
            self._start_animation(
                target_pan=world_position,
                target_zoom=self.state.zoom
            )
        else:
            self.state.pan_offset = world_position
            self.coordinate_system.set_pan(world_position)

    def set_zoom(self, zoom: float, animate: bool = False) -> None:
        """Set camera zoom level"""
        if animate:
            self._start_animation(
                target_pan=self.state.pan_offset,
                target_zoom=zoom
            )
        else:
            self.state.zoom = zoom
            self.coordinate_system.set_zoom(zoom)

    def pan(self, world_delta: Point2D, animate: bool = False) -> None:
        """Pan camera by world space delta"""
        new_position = self.state.pan_offset + world_delta
        self.set_position(new_position, animate)

    def zoom_at_point(self, screen_point: Point2D, zoom_factor: float) -> None:
        """Zoom centered on a screen point"""
        # Convert screen point to world coordinates before zoom
        world_point = self.coordinate_system.screen_to_world(screen_point)

        # Apply zoom
        new_zoom = self.state.zoom * zoom_factor
        self.state.zoom = new_zoom
        self.coordinate_system.set_zoom(new_zoom)

        # Convert same world point back to screen coordinates after zoom
        new_screen_point = self.coordinate_system.world_to_screen(world_point)

        # Calculate pan adjustment to keep world point under cursor
        screen_delta = screen_point - new_screen_point
        world_delta = self.coordinate_system.screen_to_world(screen_delta) - self.coordinate_system.screen_to_world(Point2D(0, 0))

        # Apply pan adjustment
        self.state.pan_offset = self.state.pan_offset + world_delta
        self.coordinate_system.set_pan(self.state.pan_offset)

    def focus_on_point(self, world_point: Point2D, zoom: Optional[float] = None, animate: bool = True) -> None:
        """Focus camera on a specific world point"""
        target_zoom = zoom if zoom is not None else self.state.zoom

        if animate:
            self._start_animation(
                target_pan=world_point,
                target_zoom=target_zoom
            )
        else:
            self.set_position(world_point)
            if zoom is not None:
                self.set_zoom(zoom)

    def focus_on_bounds(self, bounds: BoundingBox, margin: float = 50.0, animate: bool = True) -> None:
        """Focus camera to fit bounding box in view"""
        # Calculate center of bounds
        center = bounds.center

        # Calculate zoom to fit bounds
        screen_width, screen_height = self.state.screen_size
        content_width = bounds.width + margin * 2
        content_height = bounds.height + margin * 2

        zoom_x = screen_width / content_width
        zoom_y = screen_height / content_height
        target_zoom = min(zoom_x, zoom_y)

        self.focus_on_point(center, target_zoom, animate)

    def start_following_target(self, world_position: Point2D) -> None:
        """Start following a moving target"""
        self.target_position = world_position
        self.state.mode = ViewportMode.FOLLOW_TARGET

    def stop_following_target(self) -> None:
        """Stop following target"""
        self.target_position = None
        self.state.mode = ViewportMode.NORMAL

    def update(self, current_time: float) -> bool:
        """
        Update camera state (animations, target following).

        Returns:
            True if camera state changed
        """
        changed = False

        # Update animations
        if self.state.is_animating:
            if self._update_animation(current_time):
                changed = True

        # Update target following
        if self.state.mode == ViewportMode.FOLLOW_TARGET and self.target_position:
            # Smooth camera movement toward target
            target_delta = self.target_position - self.state.pan_offset
            movement = target_delta * self.smoothing_factor

            if movement.magnitude() > 0.1:  # Minimum movement threshold
                new_position = self.state.pan_offset + movement
                self.set_position(new_position)
                changed = True

        return changed

    def _start_animation(self, target_pan: Point2D, target_zoom: float) -> None:
        """Start viewport animation"""
        self.state.is_animating = True
        self.state.animation_start_time = time.time()
        self.state.animation_start_pan = self.state.pan_offset
        self.state.animation_start_zoom = self.state.zoom
        self.state.animation_target_pan = target_pan
        self.state.animation_target_zoom = target_zoom

    def _update_animation(self, current_time: float) -> bool:
        """Update viewport animation"""
        if not self.state.is_animating:
            return False

        elapsed = current_time - self.state.animation_start_time
        progress = min(1.0, elapsed / 0.3)  # 300ms animation duration

        # Apply easing
        eased_progress = self._apply_easing(progress)

        # Interpolate pan
        pan_delta = self.state.animation_target_pan - self.state.animation_start_pan
        new_pan = self.state.animation_start_pan + pan_delta * eased_progress

        # Interpolate zoom
        zoom_delta = self.state.animation_target_zoom - self.state.animation_start_zoom
        new_zoom = self.state.animation_start_zoom + zoom_delta * eased_progress

        # Apply changes
        self.state.pan_offset = new_pan
        self.state.zoom = new_zoom
        self.coordinate_system.set_pan(new_pan)
        self.coordinate_system.set_zoom(new_zoom)

        # Check if animation is complete
        if progress >= 1.0:
            self.state.is_animating = False

        return True

    def _apply_easing(self, t: float) -> float:
        """Apply easing function to animation progress"""
        # Ease-out cubic
        return 1 - (1 - t) ** 3

    def get_view_bounds(self) -> BoundingBox:
        """Get world bounds currently visible in viewport"""
        return self.coordinate_system.get_visible_world_bounds()

    def is_point_visible(self, world_point: Point2D, margin: float = 0.0) -> bool:
        """Check if world point is visible in current view"""
        return self.coordinate_system.is_point_visible(world_point, margin)

    def get_camera_info(self) -> Dict[str, Any]:
        """Get comprehensive camera information"""
        return {
            'position': self.state.pan_offset.__dict__,
            'zoom': self.state.zoom,
            'screen_size': self.state.screen_size,
            'mode': self.state.mode.value,
            'is_animating': self.state.is_animating,
            'target_position': self.target_position.__dict__ if self.target_position else None,
            'view_bounds': self.get_view_bounds().__dict__
        }


class ZoomController:
    """
    Specialized controller for zoom operations.

    Handles zoom gestures, limits, and smooth zooming.
    """

    def __init__(self, camera: Camera, config: ViewportConfig):
        """Initialize zoom controller"""
        self.camera = camera
        self.config = config

        # Zoom state
        self.zoom_center: Optional[Point2D] = None
        self.accumulated_zoom = 1.0

    def zoom_in(self, center: Optional[Point2D] = None) -> None:
        """Zoom in by zoom speed factor"""
        center = center or self._get_screen_center()
        self.zoom_by_factor(self.config.zoom_speed, center)

    def zoom_out(self, center: Optional[Point2D] = None) -> None:
        """Zoom out by zoom speed factor"""
        center = center or self._get_screen_center()
        self.zoom_by_factor(1.0 / self.config.zoom_speed, center)

    def zoom_by_factor(self, factor: float, center: Optional[Point2D] = None) -> None:
        """Zoom by arbitrary factor"""
        center = center or self._get_screen_center()

        # Calculate new zoom level
        current_zoom = self.camera.state.zoom
        new_zoom = current_zoom * factor

        # Apply zoom limits
        new_zoom = max(self.config.min_zoom, min(self.config.max_zoom, new_zoom))

        if new_zoom != current_zoom:
            actual_factor = new_zoom / current_zoom
            self.camera.zoom_at_point(center, actual_factor)

    def set_zoom_level(self, zoom: float, center: Optional[Point2D] = None, animate: bool = False) -> None:
        """Set absolute zoom level"""
        zoom = max(self.config.min_zoom, min(self.config.max_zoom, zoom))

        if center:
            current_zoom = self.camera.state.zoom
            factor = zoom / current_zoom
            self.camera.zoom_at_point(center, factor)
        else:
            self.camera.set_zoom(zoom, animate)

    def zoom_to_fit(self, bounds: BoundingBox, animate: bool = True) -> None:
        """Zoom to fit content bounds"""
        self.camera.focus_on_bounds(bounds, self.config.fit_margin, animate)

    def _get_screen_center(self) -> Point2D:
        """Get center point of screen"""
        width, height = self.camera.state.screen_size
        return Point2D(width / 2, height / 2)

    def get_zoom_info(self) -> Dict[str, Any]:
        """Get zoom controller information"""
        return {
            'current_zoom': self.camera.state.zoom,
            'min_zoom': self.config.min_zoom,
            'max_zoom': self.config.max_zoom,
            'zoom_speed': self.config.zoom_speed,
            'zoom_range': self.config.max_zoom - self.config.min_zoom
        }


class PanController:
    """
    Specialized controller for pan operations.

    Handles pan gestures, constraints, and smooth panning.
    """

    def __init__(self, camera: Camera, config: ViewportConfig):
        """Initialize pan controller"""
        self.camera = camera
        self.config = config

        # Pan state
        self.is_panning = False
        self.pan_start_position: Optional[Point2D] = None
        self.pan_start_world: Optional[Point2D] = None

    def start_pan(self, screen_position: Point2D) -> None:
        """Start pan operation"""
        self.is_panning = True
        self.pan_start_position = screen_position
        self.pan_start_world = self.camera.coordinate_system.screen_to_world(screen_position)
        self.camera.state.mode = ViewportMode.PAN

    def update_pan(self, screen_position: Point2D) -> None:
        """Update pan operation"""
        if not self.is_panning or not self.pan_start_position:
            return

        # Calculate world delta
        current_world = self.camera.coordinate_system.screen_to_world(screen_position)
        world_delta = self.pan_start_world - current_world

        # Apply pan
        new_pan_offset = self.camera.state.pan_offset + world_delta * self.config.pan_speed

        # Apply constraints if enabled
        if self.config.constrain_pan and self.config.world_bounds:
            new_pan_offset = self._constrain_pan(new_pan_offset)

        self.camera.set_position(new_pan_offset)

    def end_pan(self) -> None:
        """End pan operation"""
        self.is_panning = False
        self.pan_start_position = None
        self.pan_start_world = None
        self.camera.state.mode = ViewportMode.NORMAL

    def pan_by_screen_delta(self, screen_delta: Point2D) -> None:
        """Pan by screen space delta"""
        # Convert screen delta to world delta
        world_delta = self.camera.coordinate_system.screen_to_world(screen_delta) - self.camera.coordinate_system.screen_to_world(Point2D(0, 0))
        self.camera.pan(world_delta * self.config.pan_speed)

    def pan_by_world_delta(self, world_delta: Point2D) -> None:
        """Pan by world space delta"""
        self.camera.pan(world_delta * self.config.pan_speed)

    def center_on_point(self, world_point: Point2D, animate: bool = True) -> None:
        """Center viewport on world point"""
        self.camera.focus_on_point(world_point, animate=animate)

    def _constrain_pan(self, pan_offset: Point2D) -> Point2D:
        """Constrain pan offset to world bounds"""
        if not self.config.world_bounds:
            return pan_offset

        bounds = self.config.world_bounds
        view_bounds = self.camera.get_view_bounds()

        # Calculate constraints
        max_x = bounds.max_x - view_bounds.width / 2
        min_x = bounds.min_x + view_bounds.width / 2
        max_y = bounds.max_y - view_bounds.height / 2
        min_y = bounds.min_y + view_bounds.height / 2

        # Apply constraints
        constrained_x = max(min_x, min(max_x, pan_offset.x))
        constrained_y = max(min_y, min(max_y, pan_offset.y))

        return Point2D(constrained_x, constrained_y)

    def get_pan_info(self) -> Dict[str, Any]:
        """Get pan controller information"""
        return {
            'current_offset': self.camera.state.pan_offset.__dict__,
            'is_panning': self.is_panning,
            'pan_speed': self.config.pan_speed,
            'constrain_pan': self.config.constrain_pan,
            'world_bounds': self.config.world_bounds.__dict__ if self.config.world_bounds else None
        }


class Viewport:
    """
    Complete viewport management system.

    Integrates camera, zoom, and pan controllers to provide
    comprehensive viewport navigation for visualization.
    """

    def __init__(self, coordinate_system: CoordinateSystem, config: Optional[ViewportConfig] = None):
        """Initialize viewport"""
        self.coordinate_system = coordinate_system
        self.config = config or ViewportConfig()

        # Create controllers
        self.camera = Camera(coordinate_system)
        self.zoom_controller = ZoomController(self.camera, self.config)
        self.pan_controller = PanController(self.camera, self.config)

        # Event callbacks
        self.viewport_change_callbacks: List[Callable[[Dict[str, Any]], None]] = []

        # State tracking
        self.last_update_time = time.time()

    def resize(self, width: int, height: int) -> None:
        """Resize viewport"""
        old_size = self.camera.state.screen_size
        self.camera.state.screen_size = (width, height)
        self.coordinate_system.resize_screen(width, height)

        # Auto-fit content if enabled
        if self.config.auto_fit_on_resize and old_size != (width, height):
            # Maintain relative zoom
            size_ratio = min(width / old_size[0], height / old_size[1])
            new_zoom = self.camera.state.zoom * size_ratio
            self.zoom_controller.set_zoom_level(new_zoom)

        self._notify_viewport_change()

    def update(self, current_time: Optional[float] = None) -> None:
        """Update viewport state"""
        if current_time is None:
            current_time = time.time()

        # Update camera
        camera_changed = self.camera.update(current_time)

        # Notify if viewport changed
        if camera_changed:
            self._notify_viewport_change()

        self.last_update_time = current_time

    def handle_mouse_wheel(self, screen_position: Point2D, delta: float) -> None:
        """Handle mouse wheel zoom"""
        zoom_factor = self.config.zoom_speed if delta > 0 else 1.0 / self.config.zoom_speed
        self.zoom_controller.zoom_by_factor(zoom_factor, screen_position)
        self._notify_viewport_change()

    def handle_mouse_drag_start(self, screen_position: Point2D) -> None:
        """Handle start of mouse drag for panning"""
        self.pan_controller.start_pan(screen_position)

    def handle_mouse_drag(self, screen_position: Point2D) -> None:
        """Handle mouse drag for panning"""
        self.pan_controller.update_pan(screen_position)
        self._notify_viewport_change()

    def handle_mouse_drag_end(self) -> None:
        """Handle end of mouse drag"""
        self.pan_controller.end_pan()

    def fit_to_content(self, content_bounds: BoundingBox, animate: bool = True) -> None:
        """Fit viewport to show all content"""
        self.zoom_controller.zoom_to_fit(content_bounds, animate)
        self._notify_viewport_change()

    def reset_view(self, animate: bool = True) -> None:
        """Reset viewport to default view"""
        self.camera.focus_on_point(Point2D(0, 0), 1.0, animate)
        self._notify_viewport_change()

    def get_viewport_info(self) -> Dict[str, Any]:
        """Get comprehensive viewport information"""
        return {
            'camera': self.camera.get_camera_info(),
            'zoom': self.zoom_controller.get_zoom_info(),
            'pan': self.pan_controller.get_pan_info(),
            'screen_size': self.camera.state.screen_size,
            'view_bounds': self.camera.get_view_bounds().__dict__,
            'config': {
                'min_zoom': self.config.min_zoom,
                'max_zoom': self.config.max_zoom,
                'zoom_speed': self.config.zoom_speed,
                'pan_speed': self.config.pan_speed,
                'smooth_zoom': self.config.smooth_zoom,
                'smooth_pan': self.config.smooth_pan
            }
        }

    def register_viewport_change_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register callback for viewport changes"""
        self.viewport_change_callbacks.append(callback)

    def _notify_viewport_change(self) -> None:
        """Notify callbacks of viewport change"""
        viewport_info = self.get_viewport_info()

        for callback in self.viewport_change_callbacks:
            try:
                callback(viewport_info)
            except Exception as e:
                print(f"Viewport callback error: {e}")

    def world_to_screen(self, world_point: Point2D) -> Point2D:
        """Convert world coordinates to screen coordinates"""
        return self.coordinate_system.world_to_screen(world_point)

    def screen_to_world(self, screen_point: Point2D) -> Point2D:
        """Convert screen coordinates to world coordinates"""
        return self.coordinate_system.screen_to_world(screen_point)

    def is_point_visible(self, world_point: Point2D, margin: float = 0.0) -> bool:
        """Check if world point is visible"""
        return self.camera.is_point_visible(world_point, margin)