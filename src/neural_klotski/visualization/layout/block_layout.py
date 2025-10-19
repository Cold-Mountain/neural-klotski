"""
Block Layout Management for Neural-Klotski Visualization

Intelligent positioning and layout strategies for the 79-block network.
Supports multiple layout algorithms optimized for different visualization needs.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable, Any, Set
from enum import Enum
import numpy as np
import math
import random
import time
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from neural_klotski.core.block import Block, BlockColor, BlockState
from neural_klotski.core.wire import Wire
from neural_klotski.visualization.layout.coordinate_system import Point2D, BoundingBox, CoordinateSystem


class LayoutType(Enum):
    """Types of layout strategies"""
    ACTIVATION_LAG = "activation_lag"
    GRID = "grid"
    CIRCULAR = "circular"
    FORCE_DIRECTED = "force_directed"
    HIERARCHICAL = "hierarchical"
    CUSTOM = "custom"


@dataclass
class LayoutConfig:
    """Configuration for layout algorithms"""
    # General settings
    layout_type: LayoutType = LayoutType.ACTIVATION_LAG
    auto_update: bool = True
    animation_enabled: bool = True
    animation_duration: float = 1.0

    # Spacing and sizing
    block_size: float = 20.0
    min_spacing: float = 30.0
    margin: float = 50.0

    # Layout-specific parameters
    grid_columns: int = 10
    circular_radius: float = 200.0
    force_strength: float = 1.0
    force_iterations: int = 100

    # Visual grouping
    group_by_color: bool = True
    color_separation: float = 50.0
    maintain_connectivity: bool = True

    # Performance optimization
    use_spatial_indexing: bool = True
    update_threshold: float = 1.0  # Minimum movement to trigger update


@dataclass
class BlockLayoutInfo:
    """Layout information for a single block"""
    block_id: int
    current_position: Point2D
    target_position: Point2D
    velocity: Point2D = field(default_factory=lambda: Point2D(0, 0))

    # Layout metadata
    layout_group: Optional[str] = None
    connection_count: int = 0
    priority: float = 1.0

    # Animation state
    is_animating: bool = False
    animation_start_time: float = 0.0
    animation_progress: float = 0.0

    def update_animation(self, current_time: float, duration: float) -> bool:
        """
        Update animation progress.

        Returns:
            True if animation is complete, False otherwise
        """
        if not self.is_animating:
            return True

        elapsed = current_time - self.animation_start_time
        self.animation_progress = min(1.0, elapsed / duration)

        # Smooth interpolation (ease-out)
        t = self.animation_progress
        smooth_t = 1 - (1 - t) ** 3

        # Update current position
        self.current_position = Point2D(
            self.current_position.x + (self.target_position.x - self.current_position.x) * smooth_t,
            self.current_position.y + (self.target_position.y - self.current_position.y) * smooth_t
        )

        if self.animation_progress >= 1.0:
            self.current_position = self.target_position
            self.is_animating = False
            return True

        return False


class LayoutStrategy(ABC):
    """
    Abstract base class for block layout strategies.

    Each strategy implements a different algorithm for positioning
    the 79 blocks in the Neural-Klotski network.
    """

    def __init__(self, name: str, config: LayoutConfig):
        """Initialize layout strategy"""
        self.name = name
        self.config = config

        # Statistics
        self.layout_calculations = 0
        self.total_calculation_time = 0.0
        self.last_update_time = 0.0

    @abstractmethod
    def calculate_positions(self,
                          blocks: Dict[int, Block],
                          wires: List[Wire],
                          coordinate_system: CoordinateSystem) -> Dict[int, Point2D]:
        """
        Calculate target positions for all blocks.

        Args:
            blocks: Dictionary of block_id -> Block
            wires: List of wire connections
            coordinate_system: Coordinate system for transformations

        Returns:
            Dictionary mapping block_id to target position
        """
        pass

    def get_statistics(self) -> Dict[str, Any]:
        """Get layout calculation statistics"""
        avg_time = (self.total_calculation_time / max(1, self.layout_calculations))

        return {
            'name': self.name,
            'calculations': self.layout_calculations,
            'total_time_ms': self.total_calculation_time * 1000,
            'avg_time_ms': avg_time * 1000,
            'last_update': self.last_update_time
        }


class ActivationLagLayout(LayoutStrategy):
    """
    Layout blocks based on their activation and lag positions.

    Maps blocks directly to activation-lag coordinate space,
    providing an intuitive view of network dynamics.
    """

    def __init__(self, config: LayoutConfig):
        super().__init__("activation_lag", config)

    def calculate_positions(self,
                          blocks: Dict[int, Block],
                          wires: List[Wire],
                          coordinate_system: CoordinateSystem) -> Dict[int, Point2D]:
        """Calculate positions based on activation and lag"""
        start_time = time.perf_counter()

        positions = {}

        for block_id, block in blocks.items():
            # Use block's current activation and lag position
            activation = getattr(block, 'position', 0.0)
            lag = getattr(block, 'lag_position', 0.0)

            # Add slight offset for blocks with same activation/lag to prevent overlap
            if self.config.group_by_color:
                color_offset = self._get_color_offset(block.color)
                activation += color_offset.x
                lag += color_offset.y

            # Create position in activation-lag space
            al_position = coordinate_system.activation_lag_space.create_point(activation, lag)

            # Transform to world coordinates
            world_position = coordinate_system.transform.transform_point(
                al_position, "activation_lag", "world"
            )

            positions[block_id] = world_position

        # Update statistics
        calculation_time = time.perf_counter() - start_time
        self.layout_calculations += 1
        self.total_calculation_time += calculation_time
        self.last_update_time = time.time()

        return positions

    def _get_color_offset(self, color: BlockColor) -> Point2D:
        """Get small offset based on block color to prevent overlap"""
        color_offsets = {
            BlockColor.RED: Point2D(-2, -2),
            BlockColor.BLUE: Point2D(2, -2),
            BlockColor.YELLOW: Point2D(0, 2),
            BlockColor.GREEN: Point2D(-2, 2),
            BlockColor.PURPLE: Point2D(2, 2)
        }
        return color_offsets.get(color, Point2D(0, 0))


class GridLayout(LayoutStrategy):
    """
    Arrange blocks in a regular grid pattern.

    Useful for systematic examination of all blocks
    and their properties.
    """

    def __init__(self, config: LayoutConfig):
        super().__init__("grid", config)

    def calculate_positions(self,
                          blocks: Dict[int, Block],
                          wires: List[Wire],
                          coordinate_system: CoordinateSystem) -> Dict[int, Point2D]:
        """Calculate grid positions"""
        start_time = time.perf_counter()

        positions = {}
        block_ids = sorted(blocks.keys())

        # Calculate grid dimensions
        total_blocks = len(block_ids)
        cols = self.config.grid_columns
        rows = math.ceil(total_blocks / cols)

        # Calculate spacing
        cell_width = self.config.block_size + self.config.min_spacing
        cell_height = self.config.block_size + self.config.min_spacing

        # Calculate grid bounds in world space
        grid_width = cols * cell_width
        grid_height = rows * cell_height

        # Center grid in world space
        start_x = -grid_width / 2
        start_y = -grid_height / 2

        # Position blocks in grid
        for i, block_id in enumerate(block_ids):
            row = i // cols
            col = i % cols

            x = start_x + col * cell_width + cell_width / 2
            y = start_y + row * cell_height + cell_height / 2

            positions[block_id] = Point2D(x, y)

        # Update statistics
        calculation_time = time.perf_counter() - start_time
        self.layout_calculations += 1
        self.total_calculation_time += calculation_time
        self.last_update_time = time.time()

        return positions


class CircularLayout(LayoutStrategy):
    """
    Arrange blocks in concentric circles.

    Groups blocks by color or connectivity and arranges
    them in aesthetically pleasing circular patterns.
    """

    def __init__(self, config: LayoutConfig):
        super().__init__("circular", config)

    def calculate_positions(self,
                          blocks: Dict[int, Block],
                          wires: List[Wire],
                          coordinate_system: CoordinateSystem) -> Dict[int, Point2D]:
        """Calculate circular positions"""
        start_time = time.perf_counter()

        positions = {}

        if self.config.group_by_color:
            positions = self._calculate_color_grouped_positions(blocks)
        else:
            positions = self._calculate_single_circle_positions(blocks)

        # Update statistics
        calculation_time = time.perf_counter() - start_time
        self.layout_calculations += 1
        self.total_calculation_time += calculation_time
        self.last_update_time = time.time()

        return positions

    def _calculate_color_grouped_positions(self, blocks: Dict[int, Block]) -> Dict[int, Point2D]:
        """Calculate positions with color grouping"""
        positions = {}

        # Group blocks by color
        color_groups = {}
        for block_id, block in blocks.items():
            color = block.color
            if color not in color_groups:
                color_groups[color] = []
            color_groups[color].append(block_id)

        # Calculate circle positions for each color group
        num_colors = len(color_groups)
        base_radius = self.config.circular_radius

        for i, (color, block_ids) in enumerate(color_groups.items()):
            # Calculate center position for this color group
            angle = (2 * math.pi * i) / num_colors
            center_x = self.config.color_separation * math.cos(angle)
            center_y = self.config.color_separation * math.sin(angle)

            # Calculate radius for this group
            group_radius = base_radius * math.sqrt(len(block_ids)) / 10.0

            # Position blocks in circle around group center
            for j, block_id in enumerate(block_ids):
                if len(block_ids) == 1:
                    # Single block at center
                    x, y = center_x, center_y
                else:
                    block_angle = (2 * math.pi * j) / len(block_ids)
                    x = center_x + group_radius * math.cos(block_angle)
                    y = center_y + group_radius * math.sin(block_angle)

                positions[block_id] = Point2D(x, y)

        return positions

    def _calculate_single_circle_positions(self, blocks: Dict[int, Block]) -> Dict[int, Point2D]:
        """Calculate positions in a single circle"""
        positions = {}
        block_ids = sorted(blocks.keys())

        # Calculate circle parameters
        num_blocks = len(block_ids)
        radius = self.config.circular_radius

        # Position blocks around circle
        for i, block_id in enumerate(block_ids):
            angle = (2 * math.pi * i) / num_blocks
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)

            positions[block_id] = Point2D(x, y)

        return positions


class ForceDirectedLayout(LayoutStrategy):
    """
    Physics-based layout using force-directed algorithms.

    Simulates physical forces between connected blocks to create
    natural clustering and spacing based on network connectivity.
    """

    def __init__(self, config: LayoutConfig):
        super().__init__("force_directed", config)

        # Physics simulation parameters
        self.repulsion_strength = 1000.0
        self.attraction_strength = 0.1
        self.damping = 0.9
        self.min_movement = 1.0

    def calculate_positions(self,
                          blocks: Dict[int, Block],
                          wires: List[Wire],
                          coordinate_system: CoordinateSystem) -> Dict[int, Point2D]:
        """Calculate positions using force-directed algorithm"""
        start_time = time.perf_counter()

        # Initialize random positions if needed
        positions = self._initialize_positions(blocks)
        velocities = {block_id: Point2D(0, 0) for block_id in blocks.keys()}

        # Create connectivity map
        connections = self._build_connection_map(wires)

        # Run simulation
        for iteration in range(self.config.force_iterations):
            forces = self._calculate_forces(positions, connections)

            # Update velocities and positions
            max_movement = 0.0
            for block_id in positions.keys():
                force = forces.get(block_id, Point2D(0, 0))

                # Update velocity with damping
                velocities[block_id] = velocities[block_id] * self.damping + force

                # Update position
                old_pos = positions[block_id]
                positions[block_id] = old_pos + velocities[block_id]

                # Track maximum movement for convergence
                movement = old_pos.distance_to(positions[block_id])
                max_movement = max(max_movement, movement)

            # Check for convergence
            if max_movement < self.min_movement:
                break

        # Update statistics
        calculation_time = time.perf_counter() - start_time
        self.layout_calculations += 1
        self.total_calculation_time += calculation_time
        self.last_update_time = time.time()

        return positions

    def _initialize_positions(self, blocks: Dict[int, Block]) -> Dict[int, Point2D]:
        """Initialize random positions for blocks"""
        positions = {}

        for block_id in blocks.keys():
            # Random position within reasonable bounds
            x = random.uniform(-200, 200)
            y = random.uniform(-200, 200)
            positions[block_id] = Point2D(x, y)

        return positions

    def _build_connection_map(self, wires: List[Wire]) -> Dict[int, List[int]]:
        """Build map of block connections"""
        connections = {}

        for wire in wires:
            source_id = wire.source_block_id
            target_id = wire.target_block_id

            if source_id not in connections:
                connections[source_id] = []
            if target_id not in connections:
                connections[target_id] = []

            connections[source_id].append(target_id)
            connections[target_id].append(source_id)

        return connections

    def _calculate_forces(self,
                         positions: Dict[int, Point2D],
                         connections: Dict[int, List[int]]) -> Dict[int, Point2D]:
        """Calculate forces for each block"""
        forces = {}

        for block_id, position in positions.items():
            total_force = Point2D(0, 0)

            # Repulsion forces (all other blocks)
            for other_id, other_position in positions.items():
                if block_id != other_id:
                    force = self._calculate_repulsion_force(position, other_position)
                    total_force = total_force + force

            # Attraction forces (connected blocks)
            connected_blocks = connections.get(block_id, [])
            for connected_id in connected_blocks:
                if connected_id in positions:
                    connected_position = positions[connected_id]
                    force = self._calculate_attraction_force(position, connected_position)
                    total_force = total_force + force

            forces[block_id] = total_force

        return forces

    def _calculate_repulsion_force(self, pos1: Point2D, pos2: Point2D) -> Point2D:
        """Calculate repulsion force between two positions"""
        distance_vec = pos1 - pos2
        distance = distance_vec.magnitude()

        if distance < 1.0:  # Avoid division by zero
            distance = 1.0
            distance_vec = Point2D(random.uniform(-1, 1), random.uniform(-1, 1))

        # Force decreases with square of distance
        force_magnitude = self.repulsion_strength / (distance * distance)
        force_direction = distance_vec.normalize()

        return force_direction * force_magnitude

    def _calculate_attraction_force(self, pos1: Point2D, pos2: Point2D) -> Point2D:
        """Calculate attraction force between connected positions"""
        distance_vec = pos2 - pos1
        distance = distance_vec.magnitude()

        # Force proportional to distance (spring-like)
        force_magnitude = self.attraction_strength * distance

        if distance > 0:
            force_direction = distance_vec.normalize()
            return force_direction * force_magnitude
        else:
            return Point2D(0, 0)


class BlockLayoutManager:
    """
    Central manager for block layout and positioning.

    Coordinates layout strategies, manages animations, and provides
    high-level interface for visualization components.
    """

    def __init__(self, coordinate_system: CoordinateSystem, config: Optional[LayoutConfig] = None):
        """Initialize block layout manager"""
        self.coordinate_system = coordinate_system
        self.config = config or LayoutConfig()

        # Layout strategies
        self.strategies: Dict[LayoutType, LayoutStrategy] = {
            LayoutType.ACTIVATION_LAG: ActivationLagLayout(self.config),
            LayoutType.GRID: GridLayout(self.config),
            LayoutType.CIRCULAR: CircularLayout(self.config),
            LayoutType.FORCE_DIRECTED: ForceDirectedLayout(self.config)
        }

        self.current_strategy = self.strategies[self.config.layout_type]

        # Block layout information
        self.block_layouts: Dict[int, BlockLayoutInfo] = {}

        # Update tracking
        self.last_update_time = 0.0
        self.pending_update = False

    def set_layout_strategy(self, layout_type: LayoutType) -> None:
        """Change layout strategy"""
        if layout_type in self.strategies:
            self.current_strategy = self.strategies[layout_type]
            self.config.layout_type = layout_type
            self.pending_update = True

    def update_layout(self,
                     blocks: Dict[int, Block],
                     wires: List[Wire],
                     force_update: bool = False) -> None:
        """Update block layout positions"""
        current_time = time.time()

        # Check if update is needed
        if not force_update and not self.pending_update:
            time_since_update = current_time - self.last_update_time
            if time_since_update < self.config.update_threshold:
                return

        # Calculate new target positions
        target_positions = self.current_strategy.calculate_positions(
            blocks, wires, self.coordinate_system
        )

        # Update block layout information
        for block_id, target_position in target_positions.items():
            if block_id not in self.block_layouts:
                # Initialize new block layout
                self.block_layouts[block_id] = BlockLayoutInfo(
                    block_id=block_id,
                    current_position=target_position,
                    target_position=target_position
                )
            else:
                # Update existing block layout
                layout_info = self.block_layouts[block_id]

                # Check if position changed significantly
                movement = layout_info.target_position.distance_to(target_position)
                if movement > self.config.update_threshold:
                    layout_info.target_position = target_position

                    if self.config.animation_enabled:
                        layout_info.is_animating = True
                        layout_info.animation_start_time = current_time
                        layout_info.animation_progress = 0.0

        self.last_update_time = current_time
        self.pending_update = False

    def update_animations(self, current_time: Optional[float] = None) -> bool:
        """
        Update block position animations.

        Returns:
            True if any animations are still running
        """
        if current_time is None:
            current_time = time.time()

        any_animating = False

        for layout_info in self.block_layouts.values():
            if layout_info.is_animating:
                is_complete = layout_info.update_animation(current_time, self.config.animation_duration)
                if not is_complete:
                    any_animating = True

        return any_animating

    def get_block_position(self, block_id: int) -> Optional[Point2D]:
        """Get current position of block"""
        layout_info = self.block_layouts.get(block_id)
        return layout_info.current_position if layout_info else None

    def get_block_screen_position(self, block_id: int) -> Optional[Point2D]:
        """Get block position in screen coordinates"""
        world_position = self.get_block_position(block_id)
        if world_position:
            return self.coordinate_system.world_to_screen(world_position)
        return None

    def get_all_positions(self) -> Dict[int, Point2D]:
        """Get current positions of all blocks"""
        return {
            block_id: info.current_position
            for block_id, info in self.block_layouts.items()
        }

    def get_visible_blocks(self, margin: float = 50.0) -> List[int]:
        """Get list of block IDs that are currently visible"""
        visible_blocks = []

        for block_id, layout_info in self.block_layouts.items():
            if self.coordinate_system.is_point_visible(layout_info.current_position, margin):
                visible_blocks.append(block_id)

        return visible_blocks

    def get_layout_statistics(self) -> Dict[str, Any]:
        """Get comprehensive layout statistics"""
        total_blocks = len(self.block_layouts)
        animating_blocks = sum(1 for info in self.block_layouts.values() if info.is_animating)

        return {
            'current_strategy': self.current_strategy.name,
            'total_blocks': total_blocks,
            'animating_blocks': animating_blocks,
            'last_update': self.last_update_time,
            'strategy_stats': {
                strategy_name: strategy.get_statistics()
                for strategy_name, strategy in self.strategies.items()
            }
        }