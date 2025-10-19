"""
Spatial Positioning Algorithms for Neural-Klotski Visualization

Advanced spatial positioning, collision detection, and layout optimization
for intelligent block placement and wire routing.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Callable, Any
from enum import Enum
import numpy as np
import math
import time
from collections import defaultdict
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from neural_klotski.core.block import Block, BlockColor
from neural_klotski.core.wire import Wire
from neural_klotski.visualization.layout.coordinate_system import Point2D, BoundingBox


class CollisionType(Enum):
    """Types of spatial collisions"""
    BLOCK_BLOCK = "block_block"
    BLOCK_WIRE = "block_wire"
    WIRE_WIRE = "wire_wire"
    BOUNDARY = "boundary"


@dataclass
class SpatialNode:
    """Spatial node for spatial indexing"""
    position: Point2D
    radius: float
    object_id: int
    object_type: str  # "block" or "wire"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CollisionInfo:
    """Information about a spatial collision"""
    collision_type: CollisionType
    object1_id: int
    object2_id: int
    overlap_distance: float
    collision_point: Point2D
    resolution_vector: Point2D  # Direction to resolve collision


@dataclass
class PositionConstraint:
    """Constraint for positioning algorithm"""
    target_position: Point2D
    weight: float = 1.0
    min_distance: float = 0.0
    max_distance: float = float('inf')
    constraint_type: str = "attraction"  # "attraction", "repulsion", "fixed"


class SpatialHash:
    """
    Spatial hash for efficient collision detection and nearest neighbor queries.

    Divides space into grid cells for O(1) spatial lookups.
    """

    def __init__(self, cell_size: float = 50.0, world_bounds: Optional[BoundingBox] = None):
        """
        Initialize spatial hash.

        Args:
            cell_size: Size of each grid cell
            world_bounds: Bounds of the world space
        """
        self.cell_size = cell_size
        self.world_bounds = world_bounds or BoundingBox(-1000, -1000, 1000, 1000)

        # Hash table: (grid_x, grid_y) -> list of SpatialNode
        self.grid: Dict[Tuple[int, int], List[SpatialNode]] = defaultdict(list)

        # Object tracking
        self.objects: Dict[int, SpatialNode] = {}

    def _get_grid_coords(self, position: Point2D) -> Tuple[int, int]:
        """Get grid coordinates for a position"""
        grid_x = int(position.x // self.cell_size)
        grid_y = int(position.y // self.cell_size)
        return (grid_x, grid_y)

    def _get_cells_in_radius(self, center: Point2D, radius: float) -> List[Tuple[int, int]]:
        """Get all grid cells that intersect with a circle"""
        cells = []

        # Calculate bounding box of circle
        min_x = center.x - radius
        max_x = center.x + radius
        min_y = center.y - radius
        max_y = center.y + radius

        # Get grid range
        min_grid_x = int(min_x // self.cell_size)
        max_grid_x = int(max_x // self.cell_size)
        min_grid_y = int(min_y // self.cell_size)
        max_grid_y = int(max_y // self.cell_size)

        # Add all cells in range
        for grid_x in range(min_grid_x, max_grid_x + 1):
            for grid_y in range(min_grid_y, max_grid_y + 1):
                cells.append((grid_x, grid_y))

        return cells

    def insert(self, node: SpatialNode) -> None:
        """Insert spatial node into hash"""
        # Remove existing node if present
        if node.object_id in self.objects:
            self.remove(node.object_id)

        # Add to grid
        grid_coords = self._get_grid_coords(node.position)
        self.grid[grid_coords].append(node)

        # Track object
        self.objects[node.object_id] = node

    def remove(self, object_id: int) -> bool:
        """Remove object from spatial hash"""
        if object_id not in self.objects:
            return False

        node = self.objects[object_id]
        grid_coords = self._get_grid_coords(node.position)

        # Remove from grid
        if grid_coords in self.grid:
            self.grid[grid_coords] = [n for n in self.grid[grid_coords] if n.object_id != object_id]
            if not self.grid[grid_coords]:
                del self.grid[grid_coords]

        # Remove from tracking
        del self.objects[object_id]
        return True

    def query_radius(self, center: Point2D, radius: float) -> List[SpatialNode]:
        """Query all objects within radius of center"""
        results = []
        cells = self._get_cells_in_radius(center, radius)

        for grid_coords in cells:
            if grid_coords in self.grid:
                for node in self.grid[grid_coords]:
                    distance = center.distance_to(node.position)
                    if distance <= radius + node.radius:
                        results.append(node)

        return results

    def query_nearest(self, center: Point2D, max_count: int = 10) -> List[Tuple[SpatialNode, float]]:
        """Query nearest objects to center"""
        # Start with immediate cell
        candidates = []
        search_radius = self.cell_size

        # Expand search radius until we have enough candidates
        while len(candidates) < max_count * 2 and search_radius < 1000:
            candidates = self.query_radius(center, search_radius)
            search_radius *= 2

        # Calculate distances and sort
        results = []
        for node in candidates:
            distance = center.distance_to(node.position)
            results.append((node, distance))

        results.sort(key=lambda x: x[1])
        return results[:max_count]

    def clear(self) -> None:
        """Clear all objects from spatial hash"""
        self.grid.clear()
        self.objects.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """Get spatial hash statistics"""
        occupied_cells = len(self.grid)
        total_objects = len(self.objects)
        avg_objects_per_cell = total_objects / max(1, occupied_cells)

        return {
            'total_objects': total_objects,
            'occupied_cells': occupied_cells,
            'avg_objects_per_cell': avg_objects_per_cell,
            'cell_size': self.cell_size
        }


class CollisionDetector:
    """
    Collision detection system for blocks and wires.

    Efficiently detects and resolves spatial collisions using
    spatial hashing and geometric algorithms.
    """

    def __init__(self, spatial_hash: Optional[SpatialHash] = None):
        """Initialize collision detector"""
        self.spatial_hash = spatial_hash or SpatialHash()

        # Collision settings
        self.block_radius = 15.0
        self.wire_thickness = 2.0
        self.min_block_distance = 30.0

        # Statistics
        self.collision_checks = 0
        self.collisions_detected = 0

    def update_spatial_index(self,
                           block_positions: Dict[int, Point2D],
                           wire_positions: Optional[Dict[int, List[Point2D]]] = None) -> None:
        """Update spatial index with current positions"""
        self.spatial_hash.clear()

        # Add blocks
        for block_id, position in block_positions.items():
            node = SpatialNode(
                position=position,
                radius=self.block_radius,
                object_id=block_id,
                object_type="block"
            )
            self.spatial_hash.insert(node)

        # Add wire points (if provided)
        if wire_positions:
            for wire_id, points in wire_positions.items():
                for i, point in enumerate(points):
                    node = SpatialNode(
                        position=point,
                        radius=self.wire_thickness,
                        object_id=wire_id * 1000 + i,  # Unique ID for wire segment
                        object_type="wire",
                        metadata={"wire_id": wire_id, "segment_index": i}
                    )
                    self.spatial_hash.insert(node)

    def detect_collisions(self, block_positions: Dict[int, Point2D]) -> List[CollisionInfo]:
        """Detect all collisions in current layout"""
        collisions = []
        self.collision_checks = 0

        for block_id, position in block_positions.items():
            # Query nearby objects
            nearby_objects = self.spatial_hash.query_radius(position, self.block_radius * 2)

            for nearby_object in nearby_objects:
                if nearby_object.object_id == block_id:
                    continue  # Skip self

                self.collision_checks += 1

                if nearby_object.object_type == "block":
                    collision = self._check_block_block_collision(
                        block_id, position, nearby_object.object_id, nearby_object.position
                    )
                    if collision:
                        collisions.append(collision)

        self.collisions_detected = len(collisions)
        return collisions

    def _check_block_block_collision(self,
                                   block1_id: int, pos1: Point2D,
                                   block2_id: int, pos2: Point2D) -> Optional[CollisionInfo]:
        """Check collision between two blocks"""
        distance = pos1.distance_to(pos2)
        min_distance = self.min_block_distance

        if distance < min_distance:
            overlap = min_distance - distance
            direction = (pos1 - pos2).normalize()

            # Handle case where blocks are at exactly same position
            if direction.magnitude() == 0:
                direction = Point2D(1, 0)  # Default direction

            collision_point = pos1 + pos2
            collision_point = collision_point / 2  # Midpoint

            return CollisionInfo(
                collision_type=CollisionType.BLOCK_BLOCK,
                object1_id=block1_id,
                object2_id=block2_id,
                overlap_distance=overlap,
                collision_point=collision_point,
                resolution_vector=direction * overlap
            )

        return None

    def resolve_collisions(self,
                         block_positions: Dict[int, Point2D],
                         collisions: List[CollisionInfo],
                         resolution_strength: float = 0.5) -> Dict[int, Point2D]:
        """Resolve collisions by adjusting positions"""
        resolved_positions = block_positions.copy()
        resolution_vectors = defaultdict(lambda: Point2D(0, 0))

        # Accumulate resolution vectors
        for collision in collisions:
            if collision.collision_type == CollisionType.BLOCK_BLOCK:
                # Split resolution between both blocks
                half_resolution = collision.resolution_vector * (resolution_strength / 2)

                resolution_vectors[collision.object1_id] = (
                    resolution_vectors[collision.object1_id] + half_resolution
                )
                resolution_vectors[collision.object2_id] = (
                    resolution_vectors[collision.object2_id] - half_resolution
                )

        # Apply resolution vectors
        for block_id, resolution in resolution_vectors.items():
            if block_id in resolved_positions:
                resolved_positions[block_id] = resolved_positions[block_id] + resolution

        return resolved_positions

    def get_statistics(self) -> Dict[str, Any]:
        """Get collision detection statistics"""
        return {
            'collision_checks': self.collision_checks,
            'collisions_detected': self.collisions_detected,
            'spatial_hash_stats': self.spatial_hash.get_statistics()
        }


class LayoutOptimizer:
    """
    Layout optimization system using constraint satisfaction.

    Optimizes block positions to satisfy multiple constraints
    including spacing, connectivity, and aesthetic preferences.
    """

    def __init__(self):
        """Initialize layout optimizer"""
        self.constraints: List[PositionConstraint] = []
        self.optimization_iterations = 50
        self.learning_rate = 0.1
        self.convergence_threshold = 1.0

        # Statistics
        self.optimizations_run = 0
        self.total_optimization_time = 0.0

    def add_constraint(self, block_id: int, constraint: PositionConstraint) -> None:
        """Add positioning constraint for a block"""
        constraint.metadata = {"block_id": block_id}
        self.constraints.append(constraint)

    def clear_constraints(self) -> None:
        """Clear all positioning constraints"""
        self.constraints.clear()

    def optimize_layout(self,
                       initial_positions: Dict[int, Point2D],
                       fixed_blocks: Optional[Set[int]] = None) -> Dict[int, Point2D]:
        """
        Optimize layout using constraint satisfaction.

        Args:
            initial_positions: Starting positions for optimization
            fixed_blocks: Set of block IDs that should not be moved

        Returns:
            Optimized positions
        """
        start_time = time.perf_counter()

        positions = initial_positions.copy()
        fixed_blocks = fixed_blocks or set()

        # Gradient descent optimization
        for iteration in range(self.optimization_iterations):
            gradients = self._calculate_gradients(positions)
            max_movement = 0.0

            # Update positions based on gradients
            for block_id, gradient in gradients.items():
                if block_id not in fixed_blocks:
                    old_pos = positions[block_id]
                    new_pos = old_pos - gradient * self.learning_rate

                    positions[block_id] = new_pos
                    movement = old_pos.distance_to(new_pos)
                    max_movement = max(max_movement, movement)

            # Check for convergence
            if max_movement < self.convergence_threshold:
                break

        # Update statistics
        optimization_time = time.perf_counter() - start_time
        self.optimizations_run += 1
        self.total_optimization_time += optimization_time

        return positions

    def _calculate_gradients(self, positions: Dict[int, Point2D]) -> Dict[int, Point2D]:
        """Calculate gradients for constraint forces"""
        gradients = defaultdict(lambda: Point2D(0, 0))

        # Process each constraint
        for constraint in self.constraints:
            block_id = constraint.metadata["block_id"]
            if block_id not in positions:
                continue

            current_pos = positions[block_id]
            target_pos = constraint.target_position

            # Calculate force based on constraint type
            if constraint.constraint_type == "attraction":
                force = self._calculate_attraction_gradient(current_pos, target_pos, constraint)
            elif constraint.constraint_type == "repulsion":
                force = self._calculate_repulsion_gradient(current_pos, target_pos, constraint)
            else:  # fixed
                force = Point2D(0, 0)

            gradients[block_id] = gradients[block_id] + force * constraint.weight

        return gradients

    def _calculate_attraction_gradient(self,
                                     current_pos: Point2D,
                                     target_pos: Point2D,
                                     constraint: PositionConstraint) -> Point2D:
        """Calculate gradient for attraction constraint"""
        distance_vec = target_pos - current_pos
        distance = distance_vec.magnitude()

        if distance == 0:
            return Point2D(0, 0)

        # Spring-like force
        if distance < constraint.min_distance:
            # Repulsion if too close
            return distance_vec.normalize() * (constraint.min_distance - distance)
        elif distance > constraint.max_distance:
            # Strong attraction if too far
            return distance_vec.normalize() * (distance - constraint.max_distance)
        else:
            # Gentle attraction toward target
            return distance_vec * 0.1

    def _calculate_repulsion_gradient(self,
                                    current_pos: Point2D,
                                    target_pos: Point2D,
                                    constraint: PositionConstraint) -> Point2D:
        """Calculate gradient for repulsion constraint"""
        distance_vec = current_pos - target_pos
        distance = distance_vec.magnitude()

        if distance == 0 or distance > constraint.max_distance:
            return Point2D(0, 0)

        # Inverse square law repulsion
        force_magnitude = constraint.min_distance / (distance * distance + 1)
        return distance_vec.normalize() * force_magnitude

    def get_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        avg_time = self.total_optimization_time / max(1, self.optimizations_run)

        return {
            'optimizations_run': self.optimizations_run,
            'total_time_ms': self.total_optimization_time * 1000,
            'avg_time_ms': avg_time * 1000,
            'active_constraints': len(self.constraints)
        }


class SpatialPositioner:
    """
    High-level spatial positioning coordinator.

    Integrates collision detection, layout optimization, and spatial
    indexing to provide intelligent block positioning.
    """

    def __init__(self):
        """Initialize spatial positioner"""
        self.spatial_hash = SpatialHash()
        self.collision_detector = CollisionDetector(self.spatial_hash)
        self.layout_optimizer = LayoutOptimizer()

        # Configuration
        self.enable_collision_resolution = True
        self.enable_layout_optimization = True
        self.max_resolution_iterations = 5

    def position_blocks(self,
                       initial_positions: Dict[int, Point2D],
                       constraints: Optional[List[Tuple[int, PositionConstraint]]] = None,
                       fixed_blocks: Optional[Set[int]] = None) -> Dict[int, Point2D]:
        """
        Intelligently position blocks with collision avoidance and optimization.

        Args:
            initial_positions: Starting positions
            constraints: List of (block_id, constraint) tuples
            fixed_blocks: Blocks that should not be moved

        Returns:
            Final optimized positions
        """
        positions = initial_positions.copy()
        fixed_blocks = fixed_blocks or set()

        # Add constraints to optimizer
        if constraints:
            self.layout_optimizer.clear_constraints()
            for block_id, constraint in constraints:
                self.layout_optimizer.add_constraint(block_id, constraint)

        # Layout optimization
        if self.enable_layout_optimization and constraints:
            positions = self.layout_optimizer.optimize_layout(positions, fixed_blocks)

        # Collision resolution
        if self.enable_collision_resolution:
            positions = self._resolve_all_collisions(positions, fixed_blocks)

        return positions

    def _resolve_all_collisions(self,
                               positions: Dict[int, Point2D],
                               fixed_blocks: Set[int]) -> Dict[int, Point2D]:
        """Iteratively resolve all collisions"""
        current_positions = positions.copy()

        for iteration in range(self.max_resolution_iterations):
            # Update spatial index
            self.collision_detector.update_spatial_index(current_positions)

            # Detect collisions
            collisions = self.collision_detector.detect_collisions(current_positions)

            if not collisions:
                break  # No more collisions

            # Resolve collisions (but don't move fixed blocks)
            resolved_positions = self.collision_detector.resolve_collisions(
                current_positions, collisions
            )

            # Restore fixed block positions
            for block_id in fixed_blocks:
                if block_id in positions:
                    resolved_positions[block_id] = positions[block_id]

            current_positions = resolved_positions

        return current_positions

    def query_nearby_blocks(self,
                           center: Point2D,
                           radius: float,
                           block_positions: Dict[int, Point2D]) -> List[Tuple[int, float]]:
        """Query blocks near a position"""
        # Update spatial index
        self.collision_detector.update_spatial_index(block_positions)

        # Query spatial hash
        nearby_nodes = self.spatial_hash.query_radius(center, radius)

        # Filter for blocks and return with distances
        results = []
        for node in nearby_nodes:
            if node.object_type == "block":
                distance = center.distance_to(node.position)
                results.append((node.object_id, distance))

        results.sort(key=lambda x: x[1])
        return results

    def calculate_layout_quality(self, positions: Dict[int, Point2D]) -> Dict[str, float]:
        """Calculate quality metrics for a layout"""
        if not positions:
            return {}

        # Update spatial index
        self.collision_detector.update_spatial_index(positions)

        # Detect collisions
        collisions = self.collision_detector.detect_collisions(positions)

        # Calculate metrics
        total_blocks = len(positions)
        collision_count = len(collisions)
        collision_ratio = collision_count / max(1, total_blocks)

        # Calculate spacing uniformity
        all_positions = list(positions.values())
        distances = []
        for i, pos1 in enumerate(all_positions):
            for pos2 in all_positions[i+1:]:
                distances.append(pos1.distance_to(pos2))

        spacing_mean = np.mean(distances) if distances else 0
        spacing_std = np.std(distances) if distances else 0
        spacing_uniformity = 1.0 - (spacing_std / max(1, spacing_mean))

        # Calculate bounds utilization
        if all_positions:
            xs = [pos.x for pos in all_positions]
            ys = [pos.y for pos in all_positions]
            used_area = (max(xs) - min(xs)) * (max(ys) - min(ys))
            bounds_utilization = min(1.0, used_area / 1000000)  # Normalize to reasonable area
        else:
            bounds_utilization = 0.0

        return {
            'collision_ratio': collision_ratio,
            'spacing_uniformity': spacing_uniformity,
            'bounds_utilization': bounds_utilization,
            'overall_quality': (
                (1.0 - collision_ratio) * 0.5 +
                spacing_uniformity * 0.3 +
                bounds_utilization * 0.2
            )
        }

    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get statistics from all components"""
        return {
            'spatial_hash': self.spatial_hash.get_statistics(),
            'collision_detector': self.collision_detector.get_statistics(),
            'layout_optimizer': self.layout_optimizer.get_statistics()
        }


class PositionCalculator:
    """
    Utility class for position calculations and geometric operations.

    Provides common geometric algorithms for spatial positioning.
    """

    @staticmethod
    def calculate_circle_positions(center: Point2D,
                                 radius: float,
                                 count: int,
                                 start_angle: float = 0.0) -> List[Point2D]:
        """Calculate positions arranged in a circle"""
        positions = []

        for i in range(count):
            angle = start_angle + (2 * math.pi * i) / count
            x = center.x + radius * math.cos(angle)
            y = center.y + radius * math.sin(angle)
            positions.append(Point2D(x, y))

        return positions

    @staticmethod
    def calculate_grid_positions(top_left: Point2D,
                               rows: int,
                               cols: int,
                               cell_width: float,
                               cell_height: float) -> List[List[Point2D]]:
        """Calculate positions arranged in a grid"""
        grid = []

        for row in range(rows):
            row_positions = []
            for col in range(cols):
                x = top_left.x + col * cell_width
                y = top_left.y + row * cell_height
                row_positions.append(Point2D(x, y))
            grid.append(row_positions)

        return grid

    @staticmethod
    def calculate_spiral_positions(center: Point2D,
                                 count: int,
                                 spacing: float) -> List[Point2D]:
        """Calculate positions arranged in a spiral"""
        positions = []
        angle = 0.0
        radius = 0.0

        for i in range(count):
            x = center.x + radius * math.cos(angle)
            y = center.y + radius * math.sin(angle)
            positions.append(Point2D(x, y))

            # Update for next position
            angle += 0.5  # Angle increment
            radius += spacing * 0.1  # Radius increment

        return positions

    @staticmethod
    def find_centroid(positions: List[Point2D]) -> Point2D:
        """Calculate centroid of positions"""
        if not positions:
            return Point2D(0, 0)

        total_x = sum(pos.x for pos in positions)
        total_y = sum(pos.y for pos in positions)
        count = len(positions)

        return Point2D(total_x / count, total_y / count)

    @staticmethod
    def calculate_bounding_box(positions: List[Point2D]) -> BoundingBox:
        """Calculate bounding box of positions"""
        if not positions:
            return BoundingBox(0, 0, 0, 0)

        xs = [pos.x for pos in positions]
        ys = [pos.y for pos in positions]

        return BoundingBox(min(xs), min(ys), max(xs), max(ys))