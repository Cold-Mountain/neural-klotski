"""
Render Context and Drawing Primitives for Neural-Klotski Visualization

Low-level rendering infrastructure including render queues, drawing primitives,
and rendering context management for efficient visualization rendering.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from enum import Enum
import time
import math
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from neural_klotski.visualization.layout.coordinate_system import Point2D, BoundingBox
from neural_klotski.visualization.rendering.visual_elements import Color


class PrimitiveType(Enum):
    """Types of drawing primitives"""
    POINT = "point"
    LINE = "line"
    RECTANGLE = "rectangle"
    CIRCLE = "circle"
    POLYGON = "polygon"
    TEXT = "text"
    IMAGE = "image"
    PATH = "path"


class BlendMode(Enum):
    """Blending modes for compositing"""
    NORMAL = "normal"
    MULTIPLY = "multiply"
    SCREEN = "screen"
    OVERLAY = "overlay"
    ADD = "add"
    SUBTRACT = "subtract"


@dataclass
class RenderLayer:
    """Rendering layer with z-ordering and blending"""
    name: str
    z_order: int = 0
    visible: bool = True
    opacity: float = 1.0
    blend_mode: BlendMode = BlendMode.NORMAL

    # Layer transformations
    transform_matrix: Optional[List[List[float]]] = None
    clip_bounds: Optional[BoundingBox] = None

    # Primitives in this layer
    primitives: List['DrawingPrimitive'] = field(default_factory=list)

    def add_primitive(self, primitive: 'DrawingPrimitive') -> None:
        """Add primitive to layer"""
        self.primitives.append(primitive)

    def clear(self) -> None:
        """Clear all primitives from layer"""
        self.primitives.clear()

    def sort_primitives(self) -> None:
        """Sort primitives by z-order"""
        self.primitives.sort(key=lambda p: p.z_order)


@dataclass
class DrawingPrimitive:
    """Base drawing primitive"""
    primitive_type: PrimitiveType
    z_order: int = 0
    visible: bool = True

    # Transform properties
    position: Point2D = field(default_factory=lambda: Point2D(0, 0))
    rotation: float = 0.0
    scale: Point2D = field(default_factory=lambda: Point2D(1, 1))

    # Visual properties
    fill_color: Optional[Color] = None
    stroke_color: Optional[Color] = None
    stroke_width: float = 1.0
    opacity: float = 1.0

    # Effects
    shadow_enabled: bool = False
    shadow_offset: Point2D = field(default_factory=lambda: Point2D(2, 2))
    shadow_color: Color = field(default_factory=lambda: Color(0, 0, 0, 0.3))
    shadow_blur: float = 4.0

    # Animation properties
    animation_progress: float = 0.0
    is_animating: bool = False

    # Primitive-specific data
    data: Dict[str, Any] = field(default_factory=dict)


class DrawingPrimitives:
    """
    Factory for creating drawing primitives.

    Provides convenient methods for creating common
    geometric shapes and drawing elements.
    """

    @staticmethod
    def create_point(position: Point2D,
                    color: Color,
                    size: float = 1.0,
                    z_order: int = 0) -> DrawingPrimitive:
        """Create a point primitive"""
        return DrawingPrimitive(
            primitive_type=PrimitiveType.POINT,
            position=position,
            fill_color=color,
            z_order=z_order,
            data={'size': size}
        )

    @staticmethod
    def create_line(start: Point2D,
                   end: Point2D,
                   color: Color,
                   width: float = 1.0,
                   z_order: int = 0) -> DrawingPrimitive:
        """Create a line primitive"""
        return DrawingPrimitive(
            primitive_type=PrimitiveType.LINE,
            position=start,
            stroke_color=color,
            stroke_width=width,
            z_order=z_order,
            data={'end': end}
        )

    @staticmethod
    def create_rectangle(bounds: BoundingBox,
                        fill_color: Optional[Color] = None,
                        stroke_color: Optional[Color] = None,
                        stroke_width: float = 1.0,
                        corner_radius: float = 0.0,
                        z_order: int = 0) -> DrawingPrimitive:
        """Create a rectangle primitive"""
        center = bounds.center
        size = Point2D(bounds.width, bounds.height)

        return DrawingPrimitive(
            primitive_type=PrimitiveType.RECTANGLE,
            position=center,
            fill_color=fill_color,
            stroke_color=stroke_color,
            stroke_width=stroke_width,
            z_order=z_order,
            data={
                'size': size,
                'corner_radius': corner_radius
            }
        )

    @staticmethod
    def create_circle(center: Point2D,
                     radius: float,
                     fill_color: Optional[Color] = None,
                     stroke_color: Optional[Color] = None,
                     stroke_width: float = 1.0,
                     z_order: int = 0) -> DrawingPrimitive:
        """Create a circle primitive"""
        return DrawingPrimitive(
            primitive_type=PrimitiveType.CIRCLE,
            position=center,
            fill_color=fill_color,
            stroke_color=stroke_color,
            stroke_width=stroke_width,
            z_order=z_order,
            data={'radius': radius}
        )

    @staticmethod
    def create_polygon(vertices: List[Point2D],
                      fill_color: Optional[Color] = None,
                      stroke_color: Optional[Color] = None,
                      stroke_width: float = 1.0,
                      z_order: int = 0) -> DrawingPrimitive:
        """Create a polygon primitive"""
        if not vertices:
            raise ValueError("Polygon must have at least one vertex")

        # Calculate centroid
        centroid_x = sum(v.x for v in vertices) / len(vertices)
        centroid_y = sum(v.y for v in vertices) / len(vertices)
        centroid = Point2D(centroid_x, centroid_y)

        return DrawingPrimitive(
            primitive_type=PrimitiveType.POLYGON,
            position=centroid,
            fill_color=fill_color,
            stroke_color=stroke_color,
            stroke_width=stroke_width,
            z_order=z_order,
            data={'vertices': vertices}
        )

    @staticmethod
    def create_text(position: Point2D,
                   text: str,
                   color: Color,
                   font_size: float = 12.0,
                   font_family: str = "Arial",
                   alignment: str = "center",
                   z_order: int = 0) -> DrawingPrimitive:
        """Create a text primitive"""
        return DrawingPrimitive(
            primitive_type=PrimitiveType.TEXT,
            position=position,
            fill_color=color,
            z_order=z_order,
            data={
                'text': text,
                'font_size': font_size,
                'font_family': font_family,
                'alignment': alignment
            }
        )

    @staticmethod
    def create_arc(center: Point2D,
                  radius: float,
                  start_angle: float,
                  end_angle: float,
                  stroke_color: Color,
                  stroke_width: float = 1.0,
                  z_order: int = 0) -> DrawingPrimitive:
        """Create an arc primitive"""
        return DrawingPrimitive(
            primitive_type=PrimitiveType.PATH,
            position=center,
            stroke_color=stroke_color,
            stroke_width=stroke_width,
            z_order=z_order,
            data={
                'arc_radius': radius,
                'start_angle': start_angle,
                'end_angle': end_angle
            }
        )

    @staticmethod
    def create_bezier_curve(start: Point2D,
                           control1: Point2D,
                           control2: Point2D,
                           end: Point2D,
                           stroke_color: Color,
                           stroke_width: float = 1.0,
                           z_order: int = 0) -> DrawingPrimitive:
        """Create a bezier curve primitive"""
        return DrawingPrimitive(
            primitive_type=PrimitiveType.PATH,
            position=start,
            stroke_color=stroke_color,
            stroke_width=stroke_width,
            z_order=z_order,
            data={
                'bezier_control1': control1,
                'bezier_control2': control2,
                'bezier_end': end
            }
        )


class RenderQueue:
    """
    Render queue for batching and optimizing drawing operations.

    Manages primitive submission, sorting, and batching for
    efficient rendering performance.
    """

    def __init__(self):
        """Initialize render queue"""
        self.layers: Dict[str, RenderLayer] = {}
        self.default_layer_name = "default"

        # Performance tracking
        self.queue_stats = {
            'primitives_submitted': 0,
            'primitives_rendered': 0,
            'batches_created': 0,
            'total_layers': 0,
            'queue_time_ms': 0.0
        }

        # Batching configuration
        self.enable_batching = True
        self.max_batch_size = 1000

        # Create default layer
        self.create_layer(self.default_layer_name, z_order=0)

    def create_layer(self, name: str, z_order: int = 0, **kwargs) -> RenderLayer:
        """Create a new render layer"""
        layer = RenderLayer(name=name, z_order=z_order, **kwargs)
        self.layers[name] = layer
        return layer

    def remove_layer(self, name: str) -> bool:
        """Remove a render layer"""
        if name in self.layers and name != self.default_layer_name:
            del self.layers[name]
            return True
        return False

    def get_layer(self, name: str) -> Optional[RenderLayer]:
        """Get render layer by name"""
        return self.layers.get(name)

    def submit_primitive(self, primitive: DrawingPrimitive, layer_name: Optional[str] = None) -> None:
        """Submit primitive to render queue"""
        target_layer = layer_name or self.default_layer_name

        if target_layer not in self.layers:
            self.create_layer(target_layer)

        layer = self.layers[target_layer]
        layer.add_primitive(primitive)

        self.queue_stats['primitives_submitted'] += 1

    def submit_primitives(self, primitives: List[DrawingPrimitive], layer_name: Optional[str] = None) -> None:
        """Submit multiple primitives to render queue"""
        for primitive in primitives:
            self.submit_primitive(primitive, layer_name)

    def clear_layer(self, layer_name: str) -> None:
        """Clear all primitives from a layer"""
        if layer_name in self.layers:
            self.layers[layer_name].clear()

    def clear_all(self) -> None:
        """Clear all primitives from all layers"""
        for layer in self.layers.values():
            layer.clear()

    def sort_layers(self) -> List[RenderLayer]:
        """Get layers sorted by z-order"""
        sorted_layers = sorted(self.layers.values(), key=lambda l: l.z_order)

        # Sort primitives within each layer
        for layer in sorted_layers:
            layer.sort_primitives()

        return sorted_layers

    def get_render_batches(self, viewport_bounds: Optional[BoundingBox] = None) -> List[Dict[str, Any]]:
        """
        Get optimized render batches.

        Args:
            viewport_bounds: Optional viewport bounds for culling

        Returns:
            List of render batches with primitives and metadata
        """
        queue_start = time.perf_counter()

        batches = []
        sorted_layers = self.sort_layers()

        for layer in sorted_layers:
            if not layer.visible:
                continue

            # Viewport culling
            visible_primitives = []
            for primitive in layer.primitives:
                if not primitive.visible:
                    continue

                if viewport_bounds and self._is_primitive_outside_viewport(primitive, viewport_bounds):
                    continue

                visible_primitives.append(primitive)

            # Create batches for this layer
            if self.enable_batching:
                layer_batches = self._create_batches(visible_primitives, layer)
            else:
                layer_batches = [self._create_single_batch(visible_primitives, layer)]

            batches.extend(layer_batches)

        # Update stats
        queue_time = (time.perf_counter() - queue_start) * 1000
        self.queue_stats.update({
            'primitives_rendered': sum(len(batch['primitives']) for batch in batches),
            'batches_created': len(batches),
            'total_layers': len([l for l in self.layers.values() if l.visible]),
            'queue_time_ms': queue_time
        })

        return batches

    def _create_batches(self, primitives: List[DrawingPrimitive], layer: RenderLayer) -> List[Dict[str, Any]]:
        """Create optimized batches from primitives"""
        batches = []

        # Group primitives by type for batching
        primitive_groups = {}
        for primitive in primitives:
            ptype = primitive.primitive_type
            if ptype not in primitive_groups:
                primitive_groups[ptype] = []
            primitive_groups[ptype].append(primitive)

        # Create batches for each primitive type
        for ptype, group_primitives in primitive_groups.items():
            # Split large groups into smaller batches
            for i in range(0, len(group_primitives), self.max_batch_size):
                batch_primitives = group_primitives[i:i + self.max_batch_size]

                batch = {
                    'primitive_type': ptype,
                    'primitives': batch_primitives,
                    'layer': layer,
                    'batch_size': len(batch_primitives),
                    'metadata': {
                        'layer_name': layer.name,
                        'layer_opacity': layer.opacity,
                        'blend_mode': layer.blend_mode,
                        'transform_matrix': layer.transform_matrix,
                        'clip_bounds': layer.clip_bounds
                    }
                }
                batches.append(batch)

        return batches

    def _create_single_batch(self, primitives: List[DrawingPrimitive], layer: RenderLayer) -> Dict[str, Any]:
        """Create single batch with all primitives"""
        return {
            'primitive_type': PrimitiveType.MIXED,
            'primitives': primitives,
            'layer': layer,
            'batch_size': len(primitives),
            'metadata': {
                'layer_name': layer.name,
                'layer_opacity': layer.opacity,
                'blend_mode': layer.blend_mode,
                'transform_matrix': layer.transform_matrix,
                'clip_bounds': layer.clip_bounds
            }
        }

    def _is_primitive_outside_viewport(self, primitive: DrawingPrimitive, viewport_bounds: BoundingBox) -> bool:
        """Check if primitive is outside viewport bounds"""
        # Simple bounding box check
        if primitive.primitive_type == PrimitiveType.CIRCLE:
            radius = primitive.data.get('radius', 0)
            prim_bounds = BoundingBox(
                primitive.position.x - radius,
                primitive.position.y - radius,
                primitive.position.x + radius,
                primitive.position.y + radius
            )
        elif primitive.primitive_type == PrimitiveType.RECTANGLE:
            size = primitive.data.get('size', Point2D(0, 0))
            prim_bounds = BoundingBox(
                primitive.position.x - size.x / 2,
                primitive.position.y - size.y / 2,
                primitive.position.x + size.x / 2,
                primitive.position.y + size.y / 2
            )
        else:
            # Conservative approach for other primitives
            return False

        return not viewport_bounds.intersects(prim_bounds)

    def get_queue_statistics(self) -> Dict[str, Any]:
        """Get render queue statistics"""
        return self.queue_stats.copy()


class RenderContext:
    """
    Central rendering context for Neural-Klotski visualization.

    Manages render state, coordinate transformations, and provides
    high-level interface for rendering operations.
    """

    def __init__(self, screen_width: int = 800, screen_height: int = 600):
        """Initialize render context"""
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Render queue
        self.render_queue = RenderQueue()

        # Render state
        self.current_transform = self._identity_matrix()
        self.transform_stack: List[List[List[float]]] = []

        # Viewport and clipping
        self.viewport_bounds = BoundingBox(0, 0, screen_width, screen_height)
        self.clip_stack: List[BoundingBox] = []

        # Render settings
        self.clear_color = Color(1.0, 1.0, 1.0)  # White background
        self.antialias_enabled = True
        self.vsync_enabled = True

        # Performance tracking
        self.frame_stats = {
            'frame_count': 0,
            'total_primitives': 0,
            'render_time_ms': 0.0,
            'last_frame_time': time.time()
        }

        # Render callbacks
        self.pre_render_callbacks: List[Callable[[], None]] = []
        self.post_render_callbacks: List[Callable[[], None]] = []

    def resize(self, width: int, height: int) -> None:
        """Resize render context"""
        self.screen_width = width
        self.screen_height = height
        self.viewport_bounds = BoundingBox(0, 0, width, height)

    def clear(self, color: Optional[Color] = None) -> None:
        """Clear render surface"""
        clear_color = color or self.clear_color
        self.render_queue.clear_all()

        # Add clear primitive
        clear_primitive = DrawingPrimitives.create_rectangle(
            self.viewport_bounds,
            fill_color=clear_color,
            z_order=-1000
        )
        self.render_queue.submit_primitive(clear_primitive, "background")

    def begin_frame(self) -> None:
        """Begin rendering frame"""
        # Notify pre-render callbacks
        for callback in self.pre_render_callbacks:
            callback()

        # Clear buffers
        self.clear()

    def end_frame(self) -> None:
        """End rendering frame and present"""
        frame_start = time.time()

        # Get render batches
        batches = self.render_queue.get_render_batches(self.viewport_bounds)

        # Simulate rendering (actual rendering would be done by backend)
        total_primitives = sum(batch['batch_size'] for batch in batches)

        # Update frame stats
        render_time = (time.time() - frame_start) * 1000
        self.frame_stats.update({
            'frame_count': self.frame_stats['frame_count'] + 1,
            'total_primitives': total_primitives,
            'render_time_ms': render_time,
            'last_frame_time': time.time()
        })

        # Notify post-render callbacks
        for callback in self.post_render_callbacks:
            callback()

    def push_transform(self, matrix: List[List[float]]) -> None:
        """Push transformation matrix onto stack"""
        self.transform_stack.append(self.current_transform)
        self.current_transform = self._multiply_matrices(self.current_transform, matrix)

    def pop_transform(self) -> None:
        """Pop transformation matrix from stack"""
        if self.transform_stack:
            self.current_transform = self.transform_stack.pop()

    def push_clip(self, bounds: BoundingBox) -> None:
        """Push clipping bounds onto stack"""
        # Intersect with current clip bounds
        if self.clip_stack:
            current_clip = self.clip_stack[-1]
            clipped_bounds = self._intersect_bounds(current_clip, bounds)
        else:
            clipped_bounds = bounds

        self.clip_stack.append(clipped_bounds)

        # Update render queue layer clipping
        if self.render_queue.layers:
            for layer in self.render_queue.layers.values():
                layer.clip_bounds = clipped_bounds

    def pop_clip(self) -> None:
        """Pop clipping bounds from stack"""
        if self.clip_stack:
            self.clip_stack.pop()

            # Restore previous clip bounds
            if self.clip_stack:
                current_clip = self.clip_stack[-1]
            else:
                current_clip = None

            for layer in self.render_queue.layers.values():
                layer.clip_bounds = current_clip

    def submit_primitive(self, primitive: DrawingPrimitive, layer: Optional[str] = None) -> None:
        """Submit primitive for rendering"""
        # Apply current transform to primitive
        self._apply_transform_to_primitive(primitive)

        self.render_queue.submit_primitive(primitive, layer)

    def draw_circle(self, center: Point2D, radius: float, fill_color: Optional[Color] = None,
                   stroke_color: Optional[Color] = None, stroke_width: float = 1.0,
                   layer: Optional[str] = None) -> None:
        """Draw a circle"""
        primitive = DrawingPrimitives.create_circle(
            center, radius, fill_color, stroke_color, stroke_width
        )
        self.submit_primitive(primitive, layer)

    def draw_rectangle(self, bounds: BoundingBox, fill_color: Optional[Color] = None,
                      stroke_color: Optional[Color] = None, stroke_width: float = 1.0,
                      corner_radius: float = 0.0, layer: Optional[str] = None) -> None:
        """Draw a rectangle"""
        primitive = DrawingPrimitives.create_rectangle(
            bounds, fill_color, stroke_color, stroke_width, corner_radius
        )
        self.submit_primitive(primitive, layer)

    def draw_line(self, start: Point2D, end: Point2D, color: Color,
                 width: float = 1.0, layer: Optional[str] = None) -> None:
        """Draw a line"""
        primitive = DrawingPrimitives.create_line(start, end, color, width)
        self.submit_primitive(primitive, layer)

    def draw_text(self, position: Point2D, text: str, color: Color,
                 font_size: float = 12.0, font_family: str = "Arial",
                 alignment: str = "center", layer: Optional[str] = None) -> None:
        """Draw text"""
        primitive = DrawingPrimitives.create_text(
            position, text, color, font_size, font_family, alignment
        )
        self.submit_primitive(primitive, layer)

    def draw_polygon(self, vertices: List[Point2D], fill_color: Optional[Color] = None,
                    stroke_color: Optional[Color] = None, stroke_width: float = 1.0,
                    layer: Optional[str] = None) -> None:
        """Draw a polygon"""
        primitive = DrawingPrimitives.create_polygon(
            vertices, fill_color, stroke_color, stroke_width
        )
        self.submit_primitive(primitive, layer)

    def register_pre_render_callback(self, callback: Callable[[], None]) -> None:
        """Register pre-render callback"""
        self.pre_render_callbacks.append(callback)

    def register_post_render_callback(self, callback: Callable[[], None]) -> None:
        """Register post-render callback"""
        self.post_render_callbacks.append(callback)

    def _identity_matrix(self) -> List[List[float]]:
        """Create 3x3 identity matrix"""
        return [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]

    def _multiply_matrices(self, a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
        """Multiply two 3x3 matrices"""
        result = [[0.0, 0.0, 0.0] for _ in range(3)]
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    result[i][j] += a[i][k] * b[k][j]
        return result

    def _apply_transform_to_primitive(self, primitive: DrawingPrimitive) -> None:
        """Apply current transformation to primitive"""
        # Transform position
        pos = primitive.position
        transformed = self._transform_point(pos, self.current_transform)
        primitive.position = transformed

        # Store original transform for complex primitives
        primitive.data['render_transform'] = self.current_transform

    def _transform_point(self, point: Point2D, matrix: List[List[float]]) -> Point2D:
        """Transform point by matrix"""
        x = point.x * matrix[0][0] + point.y * matrix[0][1] + matrix[0][2]
        y = point.x * matrix[1][0] + point.y * matrix[1][1] + matrix[1][2]
        return Point2D(x, y)

    def _intersect_bounds(self, bounds1: BoundingBox, bounds2: BoundingBox) -> BoundingBox:
        """Intersect two bounding boxes"""
        return BoundingBox(
            max(bounds1.min_x, bounds2.min_x),
            max(bounds1.min_y, bounds2.min_y),
            min(bounds1.max_x, bounds2.max_x),
            min(bounds1.max_y, bounds2.max_y)
        )

    def get_render_statistics(self) -> Dict[str, Any]:
        """Get comprehensive render statistics"""
        queue_stats = self.render_queue.get_queue_statistics()

        return {
            'frame_stats': self.frame_stats,
            'queue_stats': queue_stats,
            'screen_size': (self.screen_width, self.screen_height),
            'viewport': self.viewport_bounds.__dict__,
            'layers': len(self.render_queue.layers),
            'antialias_enabled': self.antialias_enabled,
            'vsync_enabled': self.vsync_enabled
        }