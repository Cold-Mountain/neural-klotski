"""
Coordinate System for Neural-Klotski Visualization

Comprehensive 2D coordinate transformation system supporting multiple
coordinate spaces: Activation-Lag space, World space, and Screen space.
"""

from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Any, Union
import numpy as np
import math
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from neural_klotski.core.block import BlockColor


@dataclass
class Point2D:
    """2D point with coordinate utilities"""
    x: float
    y: float

    def __add__(self, other: 'Point2D') -> 'Point2D':
        return Point2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Point2D') -> 'Point2D':
        return Point2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> 'Point2D':
        return Point2D(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar: float) -> 'Point2D':
        return Point2D(self.x / scalar, self.y / scalar)

    def distance_to(self, other: 'Point2D') -> float:
        """Calculate Euclidean distance to another point"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def magnitude(self) -> float:
        """Calculate magnitude of point as vector"""
        return math.sqrt(self.x**2 + self.y**2)

    def normalize(self) -> 'Point2D':
        """Normalize point to unit vector"""
        mag = self.magnitude()
        if mag == 0:
            return Point2D(0, 0)
        return Point2D(self.x / mag, self.y / mag)

    def to_tuple(self) -> Tuple[float, float]:
        """Convert to tuple"""
        return (self.x, self.y)

    def to_array(self) -> np.ndarray:
        """Convert to numpy array"""
        return np.array([self.x, self.y])


@dataclass
class BoundingBox:
    """2D bounding box"""
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y

    @property
    def center(self) -> Point2D:
        return Point2D((self.min_x + self.max_x) / 2, (self.min_y + self.max_y) / 2)

    def contains(self, point: Point2D) -> bool:
        """Check if point is inside bounding box"""
        return (self.min_x <= point.x <= self.max_x and
                self.min_y <= point.y <= self.max_y)

    def expand(self, margin: float) -> 'BoundingBox':
        """Expand bounding box by margin"""
        return BoundingBox(
            self.min_x - margin,
            self.min_y - margin,
            self.max_x + margin,
            self.max_y + margin
        )

    def intersects(self, other: 'BoundingBox') -> bool:
        """Check if this bounding box intersects with another"""
        return not (self.max_x < other.min_x or self.min_x > other.max_x or
                   self.max_y < other.min_y or self.min_y > other.max_y)


class CoordinateSpace:
    """Base class for coordinate spaces"""

    def __init__(self, name: str, bounds: Optional[BoundingBox] = None):
        self.name = name
        self.bounds = bounds or BoundingBox(-100, -100, 100, 100)

    def clamp_point(self, point: Point2D) -> Point2D:
        """Clamp point to coordinate space bounds"""
        return Point2D(
            max(self.bounds.min_x, min(self.bounds.max_x, point.x)),
            max(self.bounds.min_y, min(self.bounds.max_y, point.y))
        )


class ActivationLagSpace(CoordinateSpace):
    """
    Activation-Lag coordinate space for Neural-Klotski blocks.

    X-axis: Block activation level (0.0 to 100.0)
    Y-axis: Lag position (0.0 to 100.0)
    """

    def __init__(self, activation_range: Tuple[float, float] = (0.0, 100.0),
                 lag_range: Tuple[float, float] = (0.0, 100.0)):
        """
        Initialize activation-lag space.

        Args:
            activation_range: (min_activation, max_activation)
            lag_range: (min_lag, max_lag)
        """
        bounds = BoundingBox(
            activation_range[0], lag_range[0],
            activation_range[1], lag_range[1]
        )
        super().__init__("activation_lag", bounds)

        self.activation_range = activation_range
        self.lag_range = lag_range

    def create_point(self, activation: float, lag: float) -> Point2D:
        """Create point from activation and lag values"""
        return Point2D(activation, lag)

    def get_activation(self, point: Point2D) -> float:
        """Get activation value from point"""
        return point.x

    def get_lag(self, point: Point2D) -> float:
        """Get lag value from point"""
        return point.y

    def normalize_activation(self, activation: float) -> float:
        """Normalize activation to 0-1 range"""
        min_act, max_act = self.activation_range
        return (activation - min_act) / (max_act - min_act)

    def normalize_lag(self, lag: float) -> float:
        """Normalize lag to 0-1 range"""
        min_lag, max_lag = self.lag_range
        return (lag - min_lag) / (max_lag - min_lag)


class WorldSpace(CoordinateSpace):
    """
    World coordinate space for visualization layout.

    General-purpose coordinate system for positioning blocks
    in the visualization world. Supports arbitrary positioning
    and transformations.
    """

    def __init__(self, world_size: float = 1000.0):
        """
        Initialize world space.

        Args:
            world_size: Size of world coordinate system
        """
        bounds = BoundingBox(-world_size/2, -world_size/2, world_size/2, world_size/2)
        super().__init__("world", bounds)

        self.world_size = world_size

    def world_to_normalized(self, point: Point2D) -> Point2D:
        """Convert world coordinates to normalized (0-1) coordinates"""
        return Point2D(
            (point.x + self.world_size/2) / self.world_size,
            (point.y + self.world_size/2) / self.world_size
        )

    def normalized_to_world(self, point: Point2D) -> Point2D:
        """Convert normalized (0-1) coordinates to world coordinates"""
        return Point2D(
            point.x * self.world_size - self.world_size/2,
            point.y * self.world_size - self.world_size/2
        )


class ScreenSpace(CoordinateSpace):
    """
    Screen coordinate space for rendering.

    Pixel-based coordinate system with origin typically
    at top-left (0,0) to bottom-right (width, height).
    """

    def __init__(self, width: int, height: int, origin_top_left: bool = True):
        """
        Initialize screen space.

        Args:
            width: Screen width in pixels
            height: Screen height in pixels
            origin_top_left: Whether origin is at top-left (True) or bottom-left (False)
        """
        if origin_top_left:
            bounds = BoundingBox(0, 0, width, height)
        else:
            bounds = BoundingBox(0, 0, width, height)

        super().__init__("screen", bounds)

        self.width = width
        self.height = height
        self.origin_top_left = origin_top_left

    def resize(self, new_width: int, new_height: int) -> None:
        """Resize screen space"""
        self.width = new_width
        self.height = new_height

        if self.origin_top_left:
            self.bounds = BoundingBox(0, 0, new_width, new_height)
        else:
            self.bounds = BoundingBox(0, 0, new_width, new_height)

    def get_center(self) -> Point2D:
        """Get center point of screen"""
        return Point2D(self.width / 2, self.height / 2)

    def screen_to_normalized(self, point: Point2D) -> Point2D:
        """Convert screen coordinates to normalized (0-1) coordinates"""
        norm_x = point.x / self.width

        if self.origin_top_left:
            norm_y = point.y / self.height
        else:
            norm_y = 1.0 - (point.y / self.height)

        return Point2D(norm_x, norm_y)

    def normalized_to_screen(self, point: Point2D) -> Point2D:
        """Convert normalized (0-1) coordinates to screen coordinates"""
        screen_x = point.x * self.width

        if self.origin_top_left:
            screen_y = point.y * self.height
        else:
            screen_y = (1.0 - point.y) * self.height

        return Point2D(screen_x, screen_y)


class CoordinateTransform:
    """
    Coordinate transformation engine.

    Handles transformations between different coordinate spaces
    including scaling, rotation, translation, and projection.
    """

    def __init__(self):
        """Initialize coordinate transform"""
        # Transformation matrices (3x3 for 2D homogeneous coordinates)
        self._transforms: Dict[Tuple[str, str], np.ndarray] = {}
        self._registered_spaces: Dict[str, CoordinateSpace] = {}

    def register_space(self, space: CoordinateSpace) -> None:
        """Register a coordinate space"""
        self._registered_spaces[space.name] = space

    def set_transform(self, from_space: str, to_space: str, transform_matrix: np.ndarray) -> None:
        """Set transformation matrix between two spaces"""
        if transform_matrix.shape != (3, 3):
            raise ValueError("Transform matrix must be 3x3")

        self._transforms[(from_space, to_space)] = transform_matrix.copy()

        # Store inverse transform
        try:
            inverse_matrix = np.linalg.inv(transform_matrix)
            self._transforms[(to_space, from_space)] = inverse_matrix
        except np.linalg.LinAlgError:
            print(f"Warning: Cannot compute inverse transform from {to_space} to {from_space}")

    def create_translation_matrix(self, tx: float, ty: float) -> np.ndarray:
        """Create translation transformation matrix"""
        return np.array([
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1]
        ])

    def create_scale_matrix(self, sx: float, sy: float) -> np.ndarray:
        """Create scale transformation matrix"""
        return np.array([
            [sx, 0, 0],
            [0, sy, 0],
            [0, 0, 1]
        ])

    def create_rotation_matrix(self, angle_radians: float) -> np.ndarray:
        """Create rotation transformation matrix"""
        cos_a = math.cos(angle_radians)
        sin_a = math.sin(angle_radians)

        return np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])

    def combine_transforms(self, *matrices: np.ndarray) -> np.ndarray:
        """Combine multiple transformation matrices"""
        result = np.eye(3)
        for matrix in matrices:
            result = np.dot(result, matrix)
        return result

    def transform_point(self, point: Point2D, from_space: str, to_space: str) -> Point2D:
        """Transform point between coordinate spaces"""
        if from_space == to_space:
            return point

        # Get transformation matrix
        transform_key = (from_space, to_space)
        if transform_key not in self._transforms:
            raise ValueError(f"No transform defined from {from_space} to {to_space}")

        matrix = self._transforms[transform_key]

        # Convert to homogeneous coordinates
        homogeneous = np.array([point.x, point.y, 1])

        # Apply transformation
        transformed = np.dot(matrix, homogeneous)

        # Convert back to 2D coordinates
        return Point2D(transformed[0], transformed[1])

    def transform_points(self, points: List[Point2D], from_space: str, to_space: str) -> List[Point2D]:
        """Transform multiple points between coordinate spaces"""
        return [self.transform_point(p, from_space, to_space) for p in points]

    def setup_activation_lag_to_world_transform(self,
                                              activation_lag_space: ActivationLagSpace,
                                              world_space: WorldSpace,
                                              scale_factor: float = 1.0,
                                              offset: Point2D = Point2D(0, 0)) -> None:
        """
        Setup transformation from activation-lag space to world space.

        Args:
            activation_lag_space: Source coordinate space
            world_space: Target coordinate space
            scale_factor: Scaling factor for the transformation
            offset: Translation offset
        """
        # Calculate scaling to fit activation-lag space into world space
        al_width = activation_lag_space.bounds.width
        al_height = activation_lag_space.bounds.height
        world_width = world_space.bounds.width
        world_height = world_space.bounds.height

        # Scale to fit while maintaining aspect ratio
        scale_x = (world_width * scale_factor) / al_width
        scale_y = (world_height * scale_factor) / al_height
        scale = min(scale_x, scale_y)

        # Center in world space
        center_x = world_space.bounds.center.x + offset.x
        center_y = world_space.bounds.center.y + offset.y

        # Create transformation matrix
        # 1. Translate activation-lag origin to (0,0)
        translate_origin = self.create_translation_matrix(
            -activation_lag_space.bounds.min_x,
            -activation_lag_space.bounds.min_y
        )

        # 2. Scale to world space
        scale_matrix = self.create_scale_matrix(scale, scale)

        # 3. Center in world space
        center_matrix = self.create_translation_matrix(
            center_x - (al_width * scale) / 2,
            center_y - (al_height * scale) / 2
        )

        # Combine transformations
        transform = self.combine_transforms(translate_origin, scale_matrix, center_matrix)

        self.set_transform("activation_lag", "world", transform)

    def setup_world_to_screen_transform(self,
                                      world_space: WorldSpace,
                                      screen_space: ScreenSpace,
                                      zoom: float = 1.0,
                                      pan_offset: Point2D = Point2D(0, 0)) -> None:
        """
        Setup transformation from world space to screen space.

        Args:
            world_space: Source coordinate space
            screen_space: Target coordinate space
            zoom: Zoom factor
            pan_offset: Pan offset in world coordinates
        """
        # Calculate scaling to fit world space into screen space
        world_width = world_space.bounds.width
        world_height = world_space.bounds.height
        screen_width = screen_space.width
        screen_height = screen_space.height

        # Scale to fit while maintaining aspect ratio
        scale_x = screen_width / world_width
        scale_y = screen_height / world_height
        scale = min(scale_x, scale_y) * zoom

        # Center on screen
        screen_center = screen_space.get_center()

        # Create transformation matrix
        # 1. Translate world center to origin
        translate_world_center = self.create_translation_matrix(
            -world_space.bounds.center.x - pan_offset.x,
            -world_space.bounds.center.y - pan_offset.y
        )

        # 2. Scale to screen space
        scale_matrix = self.create_scale_matrix(scale, scale)

        # 3. Handle screen coordinate system (flip Y if needed)
        flip_matrix = np.eye(3)
        if screen_space.origin_top_left:
            flip_matrix[1, 1] = -1  # Flip Y axis

        # 4. Translate to screen center
        translate_screen_center = self.create_translation_matrix(
            screen_center.x,
            screen_center.y
        )

        # Combine transformations
        transform = self.combine_transforms(
            translate_world_center,
            scale_matrix,
            flip_matrix,
            translate_screen_center
        )

        self.set_transform("world", "screen", transform)


class CoordinateSystem:
    """
    Complete coordinate system manager for Neural-Klotski visualization.

    Manages multiple coordinate spaces and provides high-level
    transformation utilities for visualization components.
    """

    def __init__(self):
        """Initialize coordinate system"""
        # Create default coordinate spaces
        self.activation_lag_space = ActivationLagSpace()
        self.world_space = WorldSpace()
        self.screen_space = ScreenSpace(800, 600)  # Default screen size

        # Create transformation engine
        self.transform = CoordinateTransform()

        # Register coordinate spaces
        self.transform.register_space(self.activation_lag_space)
        self.transform.register_space(self.world_space)
        self.transform.register_space(self.screen_space)

        # Setup default transformations
        self._setup_default_transforms()

        # Viewport parameters
        self.zoom_factor = 1.0
        self.pan_offset = Point2D(0, 0)

    def _setup_default_transforms(self) -> None:
        """Setup default coordinate transformations"""
        # Activation-lag to world transform
        self.transform.setup_activation_lag_to_world_transform(
            self.activation_lag_space,
            self.world_space,
            scale_factor=0.8  # Leave some margin
        )

        # World to screen transform
        self.transform.setup_world_to_screen_transform(
            self.world_space,
            self.screen_space,
            zoom=self.zoom_factor,
            pan_offset=self.pan_offset
        )

    def resize_screen(self, width: int, height: int) -> None:
        """Resize screen space and update transformations"""
        self.screen_space.resize(width, height)
        self.transform.register_space(self.screen_space)
        self._update_world_to_screen_transform()

    def set_zoom(self, zoom_factor: float) -> None:
        """Set zoom factor and update transformations"""
        self.zoom_factor = max(0.1, min(10.0, zoom_factor))  # Clamp zoom
        self._update_world_to_screen_transform()

    def set_pan(self, pan_offset: Point2D) -> None:
        """Set pan offset and update transformations"""
        self.pan_offset = pan_offset
        self._update_world_to_screen_transform()

    def _update_world_to_screen_transform(self) -> None:
        """Update world to screen transformation"""
        self.transform.setup_world_to_screen_transform(
            self.world_space,
            self.screen_space,
            zoom=self.zoom_factor,
            pan_offset=self.pan_offset
        )

    def activation_lag_to_screen(self, activation: float, lag: float) -> Point2D:
        """Transform activation-lag coordinates directly to screen coordinates"""
        al_point = self.activation_lag_space.create_point(activation, lag)
        world_point = self.transform.transform_point(al_point, "activation_lag", "world")
        screen_point = self.transform.transform_point(world_point, "world", "screen")
        return screen_point

    def screen_to_activation_lag(self, screen_point: Point2D) -> Tuple[float, float]:
        """Transform screen coordinates to activation-lag coordinates"""
        world_point = self.transform.transform_point(screen_point, "screen", "world")
        al_point = self.transform.transform_point(world_point, "world", "activation_lag")
        return (al_point.x, al_point.y)

    def world_to_screen(self, world_point: Point2D) -> Point2D:
        """Transform world coordinates to screen coordinates"""
        return self.transform.transform_point(world_point, "world", "screen")

    def screen_to_world(self, screen_point: Point2D) -> Point2D:
        """Transform screen coordinates to world coordinates"""
        return self.transform.transform_point(screen_point, "screen", "world")

    def get_visible_world_bounds(self) -> BoundingBox:
        """Get world coordinate bounds visible on screen"""
        # Transform screen corners to world coordinates
        corners = [
            Point2D(0, 0),
            Point2D(self.screen_space.width, 0),
            Point2D(self.screen_space.width, self.screen_space.height),
            Point2D(0, self.screen_space.height)
        ]

        world_corners = [self.screen_to_world(corner) for corner in corners]

        # Find bounding box
        min_x = min(corner.x for corner in world_corners)
        max_x = max(corner.x for corner in world_corners)
        min_y = min(corner.y for corner in world_corners)
        max_y = max(corner.y for corner in world_corners)

        return BoundingBox(min_x, min_y, max_x, max_y)

    def is_point_visible(self, world_point: Point2D, margin: float = 0.0) -> bool:
        """Check if world point is visible on screen with optional margin"""
        screen_point = self.world_to_screen(world_point)

        return ((-margin <= screen_point.x <= self.screen_space.width + margin) and
                (-margin <= screen_point.y <= self.screen_space.height + margin))

    def get_coordinate_info(self) -> Dict[str, Any]:
        """Get comprehensive coordinate system information"""
        return {
            'activation_lag_space': {
                'bounds': self.activation_lag_space.bounds.__dict__,
                'activation_range': self.activation_lag_space.activation_range,
                'lag_range': self.activation_lag_space.lag_range
            },
            'world_space': {
                'bounds': self.world_space.bounds.__dict__,
                'world_size': self.world_space.world_size
            },
            'screen_space': {
                'bounds': self.screen_space.bounds.__dict__,
                'width': self.screen_space.width,
                'height': self.screen_space.height,
                'origin_top_left': self.screen_space.origin_top_left
            },
            'viewport': {
                'zoom_factor': self.zoom_factor,
                'pan_offset': self.pan_offset.__dict__,
                'visible_world_bounds': self.get_visible_world_bounds().__dict__
            }
        }