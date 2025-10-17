"""
Network architecture system for Neural-Klotski addition networks.

Implements the complete 79-block addition network architecture according to Section 8.2
of the specification, including shelf layout, connectivity patterns, and color distribution.
"""

from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np
import random
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from neural_klotski.core.block import BlockState, BlockColor, create_block
from neural_klotski.core.wire import Wire, create_wire
from neural_klotski.core.network import NeuralKlotskiNetwork
from neural_klotski.config import NetworkConfig, ThresholdConfig
from neural_klotski.math_utils import lag_distance


class ShelfType(Enum):
    """Network shelf types for addition task"""
    INPUT = "input"
    HIDDEN1 = "hidden1"
    HIDDEN2 = "hidden2"
    OUTPUT = "output"


@dataclass
class BlockSpec:
    """Specification for a block in the network"""
    block_id: int
    shelf_type: ShelfType
    position_on_shelf: int  # 0-based position within shelf
    lag_position: float
    activation_position: float
    color: BlockColor
    threshold: float

    def __repr__(self) -> str:
        return f"BlockSpec(id={self.block_id}, shelf={self.shelf_type.value}, pos={self.position_on_shelf})"


@dataclass
class WireSpec:
    """Specification for a wire in the network"""
    wire_id: int
    source_block_id: int
    target_block_id: int
    connection_type: str  # "local", "longrange", "feedforward"
    base_strength: float
    spatial_position: Tuple[float, float]

    def __repr__(self) -> str:
        return f"WireSpec(id={self.wire_id}, {self.source_block_id}->{self.target_block_id}, {self.connection_type})"


class NetworkArchitect:
    """
    Architect for building Neural-Klotski addition networks.

    Implements the complete network architecture according to Section 8.2:
    - 79 total blocks arranged in 4 shelves
    - Proper spatial layout and color distribution
    - K-nearest neighbor and long-range connectivity
    """

    def __init__(self, network_config: NetworkConfig, threshold_config: ThresholdConfig):
        """
        Initialize network architect.

        Args:
            network_config: Network architecture configuration
            threshold_config: Block threshold configuration
        """
        self.network_config = network_config
        self.threshold_config = threshold_config

        # Validate configuration
        self._validate_config()

        # Architecture state
        self.block_specs: List[BlockSpec] = []
        self.wire_specs: List[WireSpec] = []
        self.shelf_blocks: Dict[ShelfType, List[int]] = {shelf: [] for shelf in ShelfType}

        # ID counters
        self.next_block_id = 1
        self.next_wire_id = 1

    def _validate_config(self):
        """Validate network configuration parameters"""
        if not self.network_config.validate():
            raise ValueError("Invalid network configuration")

        # Check total blocks match sum of shelf blocks
        expected_total = (self.network_config.input_blocks +
                         self.network_config.hidden1_blocks +
                         self.network_config.hidden2_blocks +
                         self.network_config.output_blocks)

        if expected_total != self.network_config.total_blocks:
            raise ValueError(f"Block count mismatch: {expected_total} != {self.network_config.total_blocks}")

    def create_shelf_layout(self) -> Dict[ShelfType, Tuple[float, List[float]]]:
        """
        Create spatial layout for all shelves.

        Returns:
            Dictionary mapping shelf types to (lag_center, activation_positions)
        """
        shelf_layout = {}

        # Define shelf centers and widths
        shelf_configs = {
            ShelfType.INPUT: (self.network_config.shelf1_lag_center, self.network_config.input_blocks),
            ShelfType.HIDDEN1: (self.network_config.shelf2_lag_center, self.network_config.hidden1_blocks),
            ShelfType.HIDDEN2: (self.network_config.shelf3_lag_center, self.network_config.hidden2_blocks),
            ShelfType.OUTPUT: (self.network_config.shelf4_lag_center, self.network_config.output_blocks)
        }

        for shelf_type, (lag_center, block_count) in shelf_configs.items():
            # Calculate activation positions evenly distributed
            if block_count == 1:
                activation_positions = [0.0]
            else:
                # Distribute blocks evenly across activation axis
                span = 80.0  # Total activation span for shelf
                start = -span / 2
                step = span / (block_count - 1) if block_count > 1 else 0
                activation_positions = [start + i * step for i in range(block_count)]

            shelf_layout[shelf_type] = (lag_center, activation_positions)

        return shelf_layout

    def generate_block_colors(self) -> Dict[ShelfType, List[BlockColor]]:
        """
        Generate block colors according to specification.

        Returns:
            Dictionary mapping shelf types to lists of block colors
        """
        colors = {}

        # Input shelf: alternating red/blue for binary representation
        input_colors = []
        for i in range(self.network_config.input_blocks):
            input_colors.append(BlockColor.RED if i % 2 == 0 else BlockColor.BLUE)
        colors[ShelfType.INPUT] = input_colors

        # Hidden shelves: specified color distribution
        for shelf_type, block_count in [(ShelfType.HIDDEN1, self.network_config.hidden1_blocks),
                                       (ShelfType.HIDDEN2, self.network_config.hidden2_blocks)]:
            shelf_colors = []

            # Calculate color counts
            red_count = int(block_count * self.network_config.hidden_red_fraction)
            blue_count = int(block_count * self.network_config.hidden_blue_fraction)
            yellow_count = block_count - red_count - blue_count  # Remainder

            # Create color list
            shelf_colors.extend([BlockColor.RED] * red_count)
            shelf_colors.extend([BlockColor.BLUE] * blue_count)
            shelf_colors.extend([BlockColor.YELLOW] * yellow_count)

            # Shuffle for random distribution
            random.shuffle(shelf_colors)
            colors[shelf_type] = shelf_colors

        # Output shelf: alternating red/blue for sum representation
        output_colors = []
        for i in range(self.network_config.output_blocks):
            output_colors.append(BlockColor.RED if i % 2 == 0 else BlockColor.BLUE)
        colors[ShelfType.OUTPUT] = output_colors

        return colors

    def generate_block_thresholds(self, shelf_type: ShelfType) -> float:
        """
        Generate threshold for block based on shelf type.

        Args:
            shelf_type: Type of shelf for threshold selection

        Returns:
            Threshold value within appropriate range
        """
        if shelf_type == ShelfType.INPUT:
            return random.uniform(self.threshold_config.input_threshold_min,
                                self.threshold_config.input_threshold_max)
        elif shelf_type in [ShelfType.HIDDEN1, ShelfType.HIDDEN2]:
            return random.uniform(self.threshold_config.hidden_threshold_min,
                                self.threshold_config.hidden_threshold_max)
        elif shelf_type == ShelfType.OUTPUT:
            return random.uniform(self.threshold_config.output_threshold_min,
                                self.threshold_config.output_threshold_max)
        else:
            raise ValueError(f"Unknown shelf type: {shelf_type}")

    def create_blocks(self):
        """Create all blocks for the network architecture"""
        shelf_layout = self.create_shelf_layout()
        shelf_colors = self.generate_block_colors()

        for shelf_type in ShelfType:
            lag_center, activation_positions = shelf_layout[shelf_type]
            colors = shelf_colors[shelf_type]

            for pos_idx, (activation_pos, color) in enumerate(zip(activation_positions, colors)):
                threshold = self.generate_block_thresholds(shelf_type)

                block_spec = BlockSpec(
                    block_id=self.next_block_id,
                    shelf_type=shelf_type,
                    position_on_shelf=pos_idx,
                    lag_position=lag_center,
                    activation_position=activation_pos,
                    color=color,
                    threshold=threshold
                )

                self.block_specs.append(block_spec)
                self.shelf_blocks[shelf_type].append(self.next_block_id)
                self.next_block_id += 1

    def calculate_k_nearest_neighbors(self, source_block_id: int, k: int,
                                    exclude_shelves: Optional[Set[ShelfType]] = None) -> List[int]:
        """
        Find K nearest neighbors for a block.

        Args:
            source_block_id: Source block ID
            k: Number of neighbors to find
            exclude_shelves: Shelf types to exclude from neighbors

        Returns:
            List of K nearest neighbor block IDs
        """
        # Find source block
        source_block = next(b for b in self.block_specs if b.block_id == source_block_id)

        # Calculate distances to all other blocks
        distances = []
        for target_block in self.block_specs:
            if target_block.block_id == source_block_id:
                continue  # Skip self

            if exclude_shelves and target_block.shelf_type in exclude_shelves:
                continue  # Skip excluded shelves

            # Calculate spatial distance
            lag_dist = abs(target_block.lag_position - source_block.lag_position)
            activation_dist = abs(target_block.activation_position - source_block.activation_position)
            total_dist = np.sqrt(lag_dist**2 + activation_dist**2)

            distances.append((total_dist, target_block.block_id))

        # Sort by distance and take K nearest
        distances.sort(key=lambda x: x[0])
        return [block_id for _, block_id in distances[:k]]

    def calculate_long_range_targets(self, source_block_id: int, l: int,
                                   min_distance: float) -> List[int]:
        """
        Find long-range connection targets.

        Args:
            source_block_id: Source block ID
            l: Number of long-range connections
            min_distance: Minimum distance for long-range connections

        Returns:
            List of long-range target block IDs
        """
        source_block = next(b for b in self.block_specs if b.block_id == source_block_id)

        # Find blocks beyond minimum distance
        distant_blocks = []
        for target_block in self.block_specs:
            if target_block.block_id == source_block_id:
                continue

            lag_dist = abs(target_block.lag_position - source_block.lag_position)
            activation_dist = abs(target_block.activation_position - source_block.activation_position)
            total_dist = np.sqrt(lag_dist**2 + activation_dist**2)

            if total_dist >= min_distance:
                distant_blocks.append(target_block.block_id)

        # Randomly select L long-range targets
        if len(distant_blocks) <= l:
            return distant_blocks
        else:
            return random.sample(distant_blocks, l)

    def create_local_connections(self):
        """Create K-nearest neighbor connections for all blocks"""
        for block_spec in self.block_specs:
            # Skip output blocks (they don't have outgoing connections)
            if block_spec.shelf_type == ShelfType.OUTPUT:
                continue

            # Find K nearest neighbors
            neighbors = self.calculate_k_nearest_neighbors(
                block_spec.block_id,
                self.network_config.local_connections,
                exclude_shelves={block_spec.shelf_type}  # Exclude same shelf
            )

            # Create wires to neighbors
            for target_id in neighbors:
                target_block = next(b for b in self.block_specs if b.block_id == target_id)

                # Calculate base strength (random within bounds)
                base_strength = random.uniform(1.0, 2.5)

                # Calculate spatial position (midpoint)
                spatial_pos = (
                    (block_spec.activation_position + target_block.activation_position) / 2,
                    (block_spec.lag_position + target_block.lag_position) / 2
                )

                wire_spec = WireSpec(
                    wire_id=self.next_wire_id,
                    source_block_id=block_spec.block_id,
                    target_block_id=target_id,
                    connection_type="local",
                    base_strength=base_strength,
                    spatial_position=spatial_pos
                )

                self.wire_specs.append(wire_spec)
                self.next_wire_id += 1

    def create_long_range_connections(self):
        """Create long-range connections for all blocks"""
        for block_spec in self.block_specs:
            # Skip output blocks
            if block_spec.shelf_type == ShelfType.OUTPUT:
                continue

            # Only create long-range if L > 0
            if self.network_config.longrange_connections <= 0:
                continue

            # Find long-range targets
            targets = self.calculate_long_range_targets(
                block_spec.block_id,
                self.network_config.longrange_connections,
                self.network_config.longrange_min_distance
            )

            # Create long-range wires
            for target_id in targets:
                target_block = next(b for b in self.block_specs if b.block_id == target_id)

                # Long-range connections are typically stronger
                base_strength = random.uniform(2.0, 3.5)

                spatial_pos = (
                    (block_spec.activation_position + target_block.activation_position) / 2,
                    (block_spec.lag_position + target_block.lag_position) / 2
                )

                wire_spec = WireSpec(
                    wire_id=self.next_wire_id,
                    source_block_id=block_spec.block_id,
                    target_block_id=target_id,
                    connection_type="longrange",
                    base_strength=base_strength,
                    spatial_position=spatial_pos
                )

                self.wire_specs.append(wire_spec)
                self.next_wire_id += 1

    def create_feedforward_connections(self):
        """Create feedforward connections between adjacent shelves"""
        shelf_pairs = [
            (ShelfType.INPUT, ShelfType.HIDDEN1),
            (ShelfType.HIDDEN1, ShelfType.HIDDEN2),
            (ShelfType.HIDDEN2, ShelfType.OUTPUT)
        ]

        for source_shelf, target_shelf in shelf_pairs:
            source_blocks = self.shelf_blocks[source_shelf]
            target_blocks = self.shelf_blocks[target_shelf]

            # Create connections from each source to multiple targets
            connections_per_source = max(1, len(target_blocks) // len(source_blocks))

            for source_id in source_blocks:
                source_block = next(b for b in self.block_specs if b.block_id == source_id)

                # Find nearest targets in next shelf
                targets = self.calculate_k_nearest_neighbors(
                    source_id, connections_per_source,
                    exclude_shelves={s for s in ShelfType if s != target_shelf}
                )

                for target_id in targets:
                    target_block = next(b for b in self.block_specs if b.block_id == target_id)

                    base_strength = random.uniform(1.5, 3.0)

                    spatial_pos = (
                        (source_block.activation_position + target_block.activation_position) / 2,
                        (source_block.lag_position + target_block.lag_position) / 2
                    )

                    wire_spec = WireSpec(
                        wire_id=self.next_wire_id,
                        source_block_id=source_id,
                        target_block_id=target_id,
                        connection_type="feedforward",
                        base_strength=base_strength,
                        spatial_position=spatial_pos
                    )

                    self.wire_specs.append(wire_spec)
                    self.next_wire_id += 1

    def build_architecture(self) -> Tuple[List[BlockSpec], List[WireSpec]]:
        """
        Build complete network architecture.

        Returns:
            Tuple of (block_specs, wire_specs)
        """
        # Reset state
        self.block_specs.clear()
        self.wire_specs.clear()
        self.shelf_blocks = {shelf: [] for shelf in ShelfType}
        self.next_block_id = 1
        self.next_wire_id = 1

        # Create blocks
        self.create_blocks()

        # Create connections
        self.create_local_connections()
        self.create_long_range_connections()
        self.create_feedforward_connections()

        return self.block_specs.copy(), self.wire_specs.copy()

    def get_architecture_statistics(self) -> Dict[str, any]:
        """Get comprehensive architecture statistics"""
        stats = {
            'total_blocks': len(self.block_specs),
            'total_wires': len(self.wire_specs),
            'blocks_by_shelf': {shelf.value: len(blocks) for shelf, blocks in self.shelf_blocks.items()},
            'wires_by_type': {},
            'color_distribution': {},
            'connectivity_stats': {
                'avg_connections_per_block': 0,
                'max_connections_per_block': 0,
                'min_connections_per_block': 0
            }
        }

        # Wire type distribution
        wire_types = {}
        for wire_spec in self.wire_specs:
            wire_types[wire_spec.connection_type] = wire_types.get(wire_spec.connection_type, 0) + 1
        stats['wires_by_type'] = wire_types

        # Color distribution by shelf
        for shelf_type in ShelfType:
            shelf_colors = {}
            for block_spec in self.block_specs:
                if block_spec.shelf_type == shelf_type:
                    color = block_spec.color.value
                    shelf_colors[color] = shelf_colors.get(color, 0) + 1
            stats['color_distribution'][shelf_type.value] = shelf_colors

        # Connectivity statistics
        if self.block_specs:
            source_counts = {}
            for wire_spec in self.wire_specs:
                source_counts[wire_spec.source_block_id] = source_counts.get(wire_spec.source_block_id, 0) + 1

            if source_counts:
                counts = list(source_counts.values())
                stats['connectivity_stats'] = {
                    'avg_connections_per_block': np.mean(counts),
                    'max_connections_per_block': max(counts),
                    'min_connections_per_block': min(counts)
                }

        return stats


def create_addition_network(network_config: Optional[NetworkConfig] = None,
                          threshold_config: Optional[ThresholdConfig] = None,
                          enable_learning: bool = True) -> NeuralKlotskiNetwork:
    """
    Factory function to create complete addition network.

    Args:
        network_config: Network architecture configuration
        threshold_config: Threshold configuration
        enable_learning: Whether to enable learning systems

    Returns:
        Complete Neural-Klotski network configured for addition task
    """
    from neural_klotski.config import get_default_config

    config = get_default_config()
    if network_config is not None:
        config.network = network_config
    if threshold_config is not None:
        config.thresholds = threshold_config

    # Create network with all systems enabled
    network = NeuralKlotskiNetwork(
        config=config,
        enable_dye_system=enable_learning,
        enable_plasticity=enable_learning,
        enable_learning=enable_learning
    )

    # Initialize dye system with appropriate bounds
    if enable_learning:
        # Calculate bounds from network configuration
        max_activation = 50.0  # From shelf width calculations
        max_lag = config.network.shelf4_lag_center + config.network.shelf_width

        network.initialize_dye_system(
            activation_range=(-max_activation, max_activation),
            lag_range=(0.0, max_lag + 20.0),  # Add buffer
            resolution=2.0
        )

    # Build architecture
    architect = NetworkArchitect(config.network, config.thresholds)
    block_specs, wire_specs = architect.build_architecture()

    # Create blocks
    for block_spec in block_specs:
        block = network.create_block(
            block_id=block_spec.block_id,
            lag_position=block_spec.lag_position,
            color=block_spec.color,
            threshold=block_spec.threshold,
            initial_position=block_spec.activation_position
        )

    # Create wires
    for wire_spec in wire_specs:
        wire = network.create_wire(
            wire_id=wire_spec.wire_id,
            source_block_id=wire_spec.source_block_id,
            target_block_id=wire_spec.target_block_id,
            strength=wire_spec.base_strength,
            spatial_position=wire_spec.spatial_position
        )

    return network


if __name__ == "__main__":
    # Test network architecture creation
    from neural_klotski.config import get_default_config

    print("Testing Neural-Klotski Network Architecture...")

    config = get_default_config()
    architect = NetworkArchitect(config.network, config.thresholds)

    print(f"Building {config.network.total_blocks}-block addition network...")

    # Build architecture
    block_specs, wire_specs = architect.build_architecture()

    print(f"Created {len(block_specs)} blocks and {len(wire_specs)} wires")

    # Show statistics
    stats = architect.get_architecture_statistics()
    print(f"\nArchitecture Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Test network creation
    print(f"\nCreating complete network...")
    network = create_addition_network(enable_learning=True)

    # Validate network
    is_valid, errors = network.validate_network()
    print(f"Network validation: {'PASSED' if is_valid else 'FAILED'}")
    if errors:
        for error in errors[:5]:  # Show first 5 errors
            print(f"  Error: {error}")

    print(f"\nNetwork creation completed successfully!")
    print(f"Final network: {len(network.blocks)} blocks, {len(network.wires)} wires")