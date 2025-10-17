"""
Comprehensive tests for Neural-Klotski network architecture system.

Tests the complete 79-block addition network architecture including
shelf layout, connectivity patterns, and encoding/decoding systems.
"""

import unittest
import numpy as np
from typing import Dict, List
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from neural_klotski.core.architecture import (
    NetworkArchitect, ShelfType, BlockSpec, WireSpec, create_addition_network
)
from neural_klotski.core.encoding import (
    BinaryEncoder, BinaryDecoder, AdditionTaskManager, AdditionProblem,
    EncodingType
)
from neural_klotski.core.block import BlockColor
from neural_klotski.config import get_default_config


class TestNetworkArchitect(unittest.TestCase):
    """Test network architecture creation"""

    def setUp(self):
        self.config = get_default_config()
        self.architect = NetworkArchitect(self.config.network, self.config.thresholds)

    def test_architect_initialization(self):
        """Test network architect initialization"""
        self.assertIsNotNone(self.architect.network_config)
        self.assertIsNotNone(self.architect.threshold_config)
        self.assertEqual(len(self.architect.block_specs), 0)
        self.assertEqual(len(self.architect.wire_specs), 0)

    def test_config_validation(self):
        """Test network configuration validation"""
        # Valid configuration should pass
        self.architect._validate_config()

        # Invalid configuration should fail
        invalid_config = self.config.network
        invalid_config.total_blocks = 100  # Mismatch with shelf totals

        with self.assertRaises(ValueError):
            invalid_architect = NetworkArchitect(invalid_config, self.config.thresholds)

    def test_shelf_layout_creation(self):
        """Test spatial shelf layout"""
        layout = self.architect.create_shelf_layout()

        # Should have all shelf types
        self.assertEqual(len(layout), 4)
        for shelf_type in ShelfType:
            self.assertIn(shelf_type, layout)

        # Check shelf properties
        input_lag, input_positions = layout[ShelfType.INPUT]
        self.assertEqual(input_lag, self.config.network.shelf1_lag_center)
        self.assertEqual(len(input_positions), self.config.network.input_blocks)

        # Positions should be distributed across activation axis
        self.assertGreater(max(input_positions), min(input_positions))

    def test_color_distribution(self):
        """Test block color distribution"""
        colors = self.architect.generate_block_colors()

        # Input shelf: alternating red/blue
        input_colors = colors[ShelfType.INPUT]
        self.assertEqual(len(input_colors), self.config.network.input_blocks)

        # Check alternating pattern
        for i, color in enumerate(input_colors):
            expected_color = BlockColor.RED if i % 2 == 0 else BlockColor.BLUE
            self.assertEqual(color, expected_color)

        # Hidden shelves: proper distribution
        for shelf_type in [ShelfType.HIDDEN1, ShelfType.HIDDEN2]:
            shelf_colors = colors[shelf_type]

            red_count = sum(1 for c in shelf_colors if c == BlockColor.RED)
            blue_count = sum(1 for c in shelf_colors if c == BlockColor.BLUE)
            yellow_count = sum(1 for c in shelf_colors if c == BlockColor.YELLOW)

            total_blocks = len(shelf_colors)

            # Check ratios are approximately correct
            red_ratio = red_count / total_blocks
            blue_ratio = blue_count / total_blocks
            yellow_ratio = yellow_count / total_blocks

            self.assertAlmostEqual(red_ratio, self.config.network.hidden_red_fraction, delta=0.1)
            self.assertAlmostEqual(blue_ratio, self.config.network.hidden_blue_fraction, delta=0.1)
            self.assertAlmostEqual(yellow_ratio, self.config.network.hidden_yellow_fraction, delta=0.1)

    def test_threshold_generation(self):
        """Test block threshold generation"""
        # Test each shelf type
        input_threshold = self.architect.generate_block_thresholds(ShelfType.INPUT)
        self.assertGreaterEqual(input_threshold, self.config.thresholds.input_threshold_min)
        self.assertLessEqual(input_threshold, self.config.thresholds.input_threshold_max)

        hidden_threshold = self.architect.generate_block_thresholds(ShelfType.HIDDEN1)
        self.assertGreaterEqual(hidden_threshold, self.config.thresholds.hidden_threshold_min)
        self.assertLessEqual(hidden_threshold, self.config.thresholds.hidden_threshold_max)

        output_threshold = self.architect.generate_block_thresholds(ShelfType.OUTPUT)
        self.assertGreaterEqual(output_threshold, self.config.thresholds.output_threshold_min)
        self.assertLessEqual(output_threshold, self.config.thresholds.output_threshold_max)

    def test_block_creation(self):
        """Test block specification creation"""
        self.architect.create_blocks()

        # Should have correct number of blocks
        self.assertEqual(len(self.architect.block_specs), self.config.network.total_blocks)

        # Check shelf distribution
        for shelf_type in ShelfType:
            shelf_blocks = self.architect.shelf_blocks[shelf_type]

            if shelf_type == ShelfType.INPUT:
                expected_count = self.config.network.input_blocks
            elif shelf_type == ShelfType.HIDDEN1:
                expected_count = self.config.network.hidden1_blocks
            elif shelf_type == ShelfType.HIDDEN2:
                expected_count = self.config.network.hidden2_blocks
            elif shelf_type == ShelfType.OUTPUT:
                expected_count = self.config.network.output_blocks

            self.assertEqual(len(shelf_blocks), expected_count)

        # Check block properties
        for block_spec in self.architect.block_specs:
            self.assertIsInstance(block_spec.block_id, int)
            self.assertIsInstance(block_spec.shelf_type, ShelfType)
            self.assertIsInstance(block_spec.color, BlockColor)
            self.assertGreater(block_spec.threshold, 0)

    def test_k_nearest_neighbors(self):
        """Test K-nearest neighbor calculation"""
        self.architect.create_blocks()

        # Test with first block
        first_block_id = self.architect.block_specs[0].block_id
        neighbors = self.architect.calculate_k_nearest_neighbors(
            first_block_id, 5, exclude_shelves={self.architect.block_specs[0].shelf_type}
        )

        # Should find requested number of neighbors
        self.assertEqual(len(neighbors), 5)

        # Neighbors should not include source block
        self.assertNotIn(first_block_id, neighbors)

        # Neighbors should be from different shelves
        source_shelf = self.architect.block_specs[0].shelf_type
        for neighbor_id in neighbors:
            neighbor_spec = next(b for b in self.architect.block_specs if b.block_id == neighbor_id)
            self.assertNotEqual(neighbor_spec.shelf_type, source_shelf)

    def test_long_range_connections(self):
        """Test long-range connection calculation"""
        self.architect.create_blocks()

        # Test with minimum distance
        first_block_id = self.architect.block_specs[0].block_id
        long_range = self.architect.calculate_long_range_targets(
            first_block_id, 3, 100.0  # Minimum distance 100
        )

        # Should find some long-range targets
        self.assertLessEqual(len(long_range), 3)

        # All targets should be beyond minimum distance
        source_spec = next(b for b in self.architect.block_specs if b.block_id == first_block_id)

        for target_id in long_range:
            target_spec = next(b for b in self.architect.block_specs if b.block_id == target_id)

            lag_dist = abs(target_spec.lag_position - source_spec.lag_position)
            activation_dist = abs(target_spec.activation_position - source_spec.activation_position)
            total_dist = np.sqrt(lag_dist**2 + activation_dist**2)

            self.assertGreaterEqual(total_dist, 100.0)

    def test_complete_architecture_build(self):
        """Test complete architecture building"""
        block_specs, wire_specs = self.architect.build_architecture()

        # Should have correct counts
        self.assertEqual(len(block_specs), self.config.network.total_blocks)
        self.assertGreater(len(wire_specs), 0)

        # All blocks should have unique IDs
        block_ids = [b.block_id for b in block_specs]
        self.assertEqual(len(block_ids), len(set(block_ids)))

        # All wires should have unique IDs
        wire_ids = [w.wire_id for w in wire_specs]
        self.assertEqual(len(wire_ids), len(set(wire_ids)))

        # Wires should reference valid blocks
        for wire_spec in wire_specs:
            self.assertIn(wire_spec.source_block_id, block_ids)
            self.assertIn(wire_spec.target_block_id, block_ids)

    def test_architecture_statistics(self):
        """Test architecture statistics collection"""
        self.architect.build_architecture()
        stats = self.architect.get_architecture_statistics()

        # Should include required statistics
        required_keys = [
            'total_blocks', 'total_wires', 'blocks_by_shelf',
            'wires_by_type', 'color_distribution', 'connectivity_stats'
        ]

        for key in required_keys:
            self.assertIn(key, stats)

        # Check statistics values
        self.assertEqual(stats['total_blocks'], self.config.network.total_blocks)
        self.assertGreater(stats['total_wires'], 0)

        # Check shelf distribution
        shelf_totals = sum(stats['blocks_by_shelf'].values())
        self.assertEqual(shelf_totals, self.config.network.total_blocks)


class TestBinaryEncoder(unittest.TestCase):
    """Test binary encoding system"""

    def setUp(self):
        self.encoder = BinaryEncoder(input_scaling_factor=100.0)

    def test_number_to_binary_conversion(self):
        """Test number to binary conversion"""
        # Test simple cases
        self.assertEqual(self.encoder.number_to_binary(0, 4), [0, 0, 0, 0])
        self.assertEqual(self.encoder.number_to_binary(1, 4), [1, 0, 0, 0])
        self.assertEqual(self.encoder.number_to_binary(5, 4), [1, 0, 1, 0])  # 5 = 0101
        self.assertEqual(self.encoder.number_to_binary(15, 4), [1, 1, 1, 1])

        # Test error cases
        with self.assertRaises(ValueError):
            self.encoder.number_to_binary(-1, 4)  # Negative number

        with self.assertRaises(ValueError):
            self.encoder.number_to_binary(16, 4)  # Exceeds 4-bit capacity

    def test_addition_problem_encoding(self):
        """Test complete addition problem encoding"""
        problem = AdditionProblem(42, 27, 69)
        input_block_ids = list(range(1, 21))

        result = self.encoder.encode_addition_problem(problem, input_block_ids)

        # Should have force for each input block
        self.assertEqual(len(result.block_forces), 20)

        # Check encoding info
        self.assertEqual(result.encoding_info['operand1'], 42)
        self.assertEqual(result.encoding_info['operand2'], 27)
        self.assertGreater(result.encoding_info['active_blocks'], 0)

        # Check force values
        for block_id, force in result.block_forces.items():
            self.assertIn(block_id, input_block_ids)
            # Force should be positive for active bits, small negative for inactive
            self.assertTrue(force > 0 or force < 0)

    def test_bit_position_info(self):
        """Test bit position information"""
        # Test operand 1 positions
        info_0 = self.encoder.get_bit_position_info(0)
        self.assertEqual(info_0['operand'], 1)
        self.assertEqual(info_0['bit_position'], 0)
        self.assertEqual(info_0['bit_value'], 1)

        info_9 = self.encoder.get_bit_position_info(9)
        self.assertEqual(info_9['operand'], 1)
        self.assertEqual(info_9['bit_position'], 9)
        self.assertEqual(info_9['bit_value'], 512)

        # Test operand 2 positions
        info_10 = self.encoder.get_bit_position_info(10)
        self.assertEqual(info_10['operand'], 2)
        self.assertEqual(info_10['bit_position'], 0)
        self.assertEqual(info_10['bit_value'], 1)

        # Test error case
        with self.assertRaises(ValueError):
            self.encoder.get_bit_position_info(20)


class TestBinaryDecoder(unittest.TestCase):
    """Test binary decoding system"""

    def setUp(self):
        self.decoder = BinaryDecoder(output_threshold=55.0)

    def test_expected_bit_pattern(self):
        """Test expected bit pattern generation"""
        # Test simple cases
        self.assertEqual(self.decoder.get_expected_bit_pattern(0), [0] * 19)
        self.assertEqual(self.decoder.get_expected_bit_pattern(1), [1] + [0] * 18)

        # Test specific number (5 = 101 in binary)
        pattern_5 = self.decoder.get_expected_bit_pattern(5)
        expected_5 = [1, 0, 1] + [0] * 16
        self.assertEqual(pattern_5, expected_5)

        # Test error cases
        with self.assertRaises(ValueError):
            self.decoder.get_expected_bit_pattern(-1)  # Negative

        with self.assertRaises(ValueError):
            self.decoder.get_expected_bit_pattern(1 << 19)  # Exceeds 19-bit capacity

    def test_binary_sum_decoding(self):
        """Test binary sum decoding"""
        # Create bit values for sum = 5 (binary 101)
        bit_values = []
        expected_bits = [1, 0, 1] + [0] * 16

        for bit in expected_bits:
            if bit == 1:
                bit_values.append(75.0)  # Above threshold
            else:
                bit_values.append(35.0)  # Below threshold

        result = self.decoder.decode_binary_sum(bit_values)

        self.assertEqual(result.decoded_sum, 5)
        self.assertGreater(result.confidence, 0.0)
        self.assertEqual(len(result.bit_values), 19)

        # Check decoding info
        self.assertEqual(result.decoding_info['threshold_used'], 55.0)
        self.assertEqual(len(result.decoding_info['binary_bits']), 19)

    def test_decoding_edge_cases(self):
        """Test decoding edge cases"""
        # Test all zeros
        zero_values = [30.0] * 19  # All below threshold
        result = self.decoder.decode_binary_sum(zero_values)
        self.assertEqual(result.decoded_sum, 0)

        # Test maximum value (all ones)
        max_values = [80.0] * 19  # All above threshold
        result = self.decoder.decode_binary_sum(max_values)
        self.assertEqual(result.decoded_sum, (1 << 19) - 1)

        # Test error case
        with self.assertRaises(ValueError):
            self.decoder.decode_binary_sum([50.0] * 18)  # Wrong number of bits


class TestAdditionTaskManager(unittest.TestCase):
    """Test complete addition task management"""

    def setUp(self):
        self.config = get_default_config()
        self.task_manager = AdditionTaskManager(self.config.simulation)

    def test_task_manager_initialization(self):
        """Test task manager initialization"""
        self.assertIsNotNone(self.task_manager.encoder)
        self.assertIsNotNone(self.task_manager.decoder)
        self.assertEqual(self.task_manager.problems_attempted, 0)
        self.assertEqual(self.task_manager.problems_correct, 0)

    def test_random_problem_generation(self):
        """Test random problem generation"""
        problem = self.task_manager.generate_random_problem(max_operand=100)

        self.assertIsInstance(problem, AdditionProblem)
        self.assertLessEqual(problem.operand1, 100)
        self.assertLessEqual(problem.operand2, 100)
        self.assertEqual(problem.expected_sum, problem.operand1 + problem.operand2)

    def test_task_statistics(self):
        """Test task statistics tracking"""
        initial_stats = self.task_manager.get_task_statistics()
        self.assertEqual(initial_stats['problems_attempted'], 0)
        self.assertEqual(initial_stats['accuracy_rate'], 0.0)

        # Simulate some problems
        self.task_manager.problems_attempted = 10
        self.task_manager.problems_correct = 7
        self.task_manager.total_error = 15.0

        stats = self.task_manager.get_task_statistics()
        self.assertEqual(stats['problems_attempted'], 10)
        self.assertEqual(stats['problems_correct'], 7)
        self.assertEqual(stats['accuracy_rate'], 0.7)
        self.assertEqual(stats['average_error'], 1.5)

    def test_statistics_reset(self):
        """Test statistics reset"""
        # Add some data
        self.task_manager.problems_attempted = 5
        self.task_manager.problems_correct = 3
        self.task_manager.total_error = 10.0

        # Reset
        self.task_manager.reset_statistics()

        # Check reset
        self.assertEqual(self.task_manager.problems_attempted, 0)
        self.assertEqual(self.task_manager.problems_correct, 0)
        self.assertEqual(self.task_manager.total_error, 0.0)


class TestNetworkIntegration(unittest.TestCase):
    """Test complete network integration"""

    def test_addition_network_creation(self):
        """Test creation of complete addition network"""
        network = create_addition_network(enable_learning=False)

        # Should have correct number of blocks and wires
        self.assertEqual(len(network.blocks), 79)
        self.assertGreater(len(network.wires), 100)  # Should have many connections

        # Validate network structure
        is_valid, errors = network.validate_network()
        self.assertTrue(is_valid, f"Network validation failed: {errors}")

    def test_network_with_learning(self):
        """Test network creation with learning enabled"""
        network = create_addition_network(enable_learning=True)

        # Learning systems should be enabled
        self.assertTrue(network.enable_learning)
        self.assertTrue(network.enable_dye_system)
        self.assertTrue(network.enable_plasticity)

        # Should have dye system initialized
        self.assertIsNotNone(network.dye_system)
        self.assertIsNotNone(network.learning_system)

    def test_network_shelf_identification(self):
        """Test identification of network shelves"""
        from neural_klotski.core.encoding import AdditionTaskManager

        network = create_addition_network(enable_learning=False)
        task_manager = AdditionTaskManager(get_default_config().simulation)

        # Test shelf block extraction
        input_blocks = task_manager._get_shelf_blocks(network, "input")
        hidden1_blocks = task_manager._get_shelf_blocks(network, "hidden1")
        hidden2_blocks = task_manager._get_shelf_blocks(network, "hidden2")
        output_blocks = task_manager._get_shelf_blocks(network, "output")

        # Should have correct counts
        self.assertEqual(len(input_blocks), 20)
        self.assertEqual(len(hidden1_blocks), 20)
        self.assertEqual(len(hidden2_blocks), 20)
        self.assertEqual(len(output_blocks), 19)

        # Total should be 79
        total_blocks = len(input_blocks) + len(hidden1_blocks) + len(hidden2_blocks) + len(output_blocks)
        self.assertEqual(total_blocks, 79)


class TestArchitectureSpecificationCompliance(unittest.TestCase):
    """Test compliance with architecture specification"""

    def setUp(self):
        self.config = get_default_config()

    def test_network_size_compliance(self):
        """Test network follows specification size requirements"""
        # Total blocks: 79
        self.assertEqual(self.config.network.total_blocks, 79)

        # Shelf distribution
        self.assertEqual(self.config.network.input_blocks, 20)
        self.assertEqual(self.config.network.hidden1_blocks, 20)
        self.assertEqual(self.config.network.hidden2_blocks, 20)
        self.assertEqual(self.config.network.output_blocks, 19)

    def test_shelf_positioning_compliance(self):
        """Test shelf positions follow specification"""
        # Shelf centers (from Section 8.2)
        self.assertEqual(self.config.network.shelf1_lag_center, 50.0)
        self.assertEqual(self.config.network.shelf2_lag_center, 100.0)
        self.assertEqual(self.config.network.shelf3_lag_center, 150.0)
        self.assertEqual(self.config.network.shelf4_lag_center, 200.0)

    def test_connectivity_parameters_compliance(self):
        """Test connectivity parameters follow specification"""
        # K-nearest neighbors
        self.assertEqual(self.config.network.local_connections, 20)

        # Long-range connections
        self.assertEqual(self.config.network.longrange_connections, 5)
        self.assertEqual(self.config.network.longrange_min_distance, 50.0)

    def test_color_distribution_compliance(self):
        """Test color distribution follows specification"""
        # Hidden layer color fractions
        self.assertEqual(self.config.network.hidden_red_fraction, 0.7)   # 70%
        self.assertEqual(self.config.network.hidden_blue_fraction, 0.25) # 25%
        self.assertEqual(self.config.network.hidden_yellow_fraction, 0.05) # 5%

        # Should sum to 1.0
        total_fraction = (self.config.network.hidden_red_fraction +
                         self.config.network.hidden_blue_fraction +
                         self.config.network.hidden_yellow_fraction)
        self.assertAlmostEqual(total_fraction, 1.0, places=6)

    def test_threshold_ranges_compliance(self):
        """Test threshold ranges follow specification"""
        thresholds = self.config.thresholds

        # Input thresholds (40-50)
        self.assertEqual(thresholds.input_threshold_min, 40.0)
        self.assertEqual(thresholds.input_threshold_max, 50.0)

        # Hidden thresholds (45-60)
        self.assertEqual(thresholds.hidden_threshold_min, 45.0)
        self.assertEqual(thresholds.hidden_threshold_max, 60.0)

        # Output thresholds (50-60)
        self.assertEqual(thresholds.output_threshold_min, 50.0)
        self.assertEqual(thresholds.output_threshold_max, 60.0)

        # Absolute bounds (30-80)
        self.assertEqual(thresholds.threshold_min, 30.0)
        self.assertEqual(thresholds.threshold_max, 80.0)


if __name__ == '__main__':
    unittest.main()