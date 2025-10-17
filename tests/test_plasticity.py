"""
Comprehensive tests for Neural-Klotski plasticity system.

Tests all plasticity mechanisms including Hebbian learning, STDP,
threshold adaptation, and integration with the dye system.
"""

import unittest
import numpy as np
from typing import Dict, List
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from neural_klotski.core.plasticity import (
    PlasticityManager, HebbianLearningRule, STDPLearningRule, ThresholdAdaptationRule,
    FiringHistoryTracker, FiringEvent, PlasticityUpdate, PlasticityType
)
from neural_klotski.core.block import BlockState, BlockColor, create_block
from neural_klotski.core.wire import Wire, create_wire
from neural_klotski.core.dye import DyeSystem, DyeColor, create_dye_system_for_network
from neural_klotski.config import get_default_config


class TestFiringHistoryTracker(unittest.TestCase):
    """Test firing history tracking functionality"""

    def setUp(self):
        self.tracker = FiringHistoryTracker(window_size=100.0)

    def test_firing_history_initialization(self):
        """Test firing history tracker initialization"""
        self.assertEqual(self.tracker.window_size, 100.0)
        self.assertEqual(len(self.tracker.firing_events), 0)
        self.assertEqual(len(self.tracker.block_firing_counts), 0)

    def test_add_firing_events(self):
        """Test adding firing events to history"""
        self.tracker.add_firing_event(1, 10.0, 5.0, 50.0)
        self.tracker.add_firing_event(2, 15.0, -2.0, 100.0)

        self.assertEqual(len(self.tracker.firing_events), 2)
        self.assertEqual(self.tracker.block_firing_counts[1], 1)
        self.assertEqual(self.tracker.block_firing_counts[2], 1)

    def test_firing_rate_calculation(self):
        """Test firing rate calculation over time window"""
        # Add multiple firing events for same block
        for i in range(5):
            self.tracker.add_firing_event(1, 10.0 + i * 5.0, 0.0, 50.0)

        # Calculate firing rate over 30 timestep window
        firing_rate = self.tracker.get_firing_rate(1, 40.0, 30.0)
        expected_rate = 5 / 30.0  # 5 events in 30 timestep window
        self.assertAlmostEqual(firing_rate, expected_rate, places=6)

    def test_cleanup_old_events(self):
        """Test cleanup of old firing events"""
        # Add events across time range
        self.tracker.add_firing_event(1, 10.0, 0.0, 50.0)
        self.tracker.add_firing_event(1, 50.0, 0.0, 50.0)
        self.tracker.add_firing_event(1, 150.0, 0.0, 50.0)

        # Cleanup should remove events older than 100 timesteps
        self.tracker.cleanup_old_events(200.0)

        # Only events from time 100+ should remain
        remaining_times = [event.firing_time for event in self.tracker.firing_events]
        self.assertNotIn(10.0, remaining_times)
        self.assertNotIn(50.0, remaining_times)
        self.assertIn(150.0, remaining_times)

    def test_paired_events_for_stdp(self):
        """Test finding paired events for STDP calculations"""
        # Add source and target events with proper timing
        self.tracker.add_firing_event(1, 10.0, 0.0, 50.0)  # Source
        self.tracker.add_firing_event(2, 15.0, 0.0, 100.0)  # Target (5 timesteps later)
        self.tracker.add_firing_event(1, 20.0, 0.0, 50.0)  # Source
        self.tracker.add_firing_event(2, 25.0, 0.0, 100.0)  # Target (5 timesteps later)

        # Current time 30, time window 20 should capture all events
        pairs = self.tracker.get_paired_events(1, 2, 30.0, 20.0)

        # Should find both pairs: (10,15) and (20,25)
        # However, algorithm also finds cross-pairs: (10,25) since 25-10=15 <= 20
        # So we expect more than 2 pairs due to the cross-matching
        self.assertGreaterEqual(len(pairs), 2)

        # Check that at least the direct pairs exist
        times = [(s.firing_time, t.firing_time) for s, t in pairs]
        self.assertIn((10.0, 15.0), times)
        self.assertIn((20.0, 25.0), times)


class TestHebbianLearningRule(unittest.TestCase):
    """Test Hebbian learning rule calculations"""

    def setUp(self):
        self.config = get_default_config().learning
        self.hebbian = HebbianLearningRule(self.config)
        self.wire = create_wire(1, 1, 2, 2.0, BlockColor.RED)

    def test_hebbian_strengthening(self):
        """Test Hebbian strengthening when both blocks fire"""
        dye_conc = 0.5
        dt = 0.5

        delta_w = self.hebbian.calculate_strength_update(
            self.wire, True, True, dye_conc, dt
        )

        expected_delta = (self.config.wire_learning_rate *
                         self.config.dye_amplification * dye_conc * dt)
        self.assertAlmostEqual(delta_w, expected_delta, places=6)
        self.assertGreater(delta_w, 0)  # Should be strengthening

    def test_no_update_without_firing(self):
        """Test no update when blocks don't fire"""
        delta_w = self.hebbian.calculate_strength_update(
            self.wire, False, False, 0.5, 0.5
        )
        self.assertEqual(delta_w, 0.0)

    def test_anti_hebbian_weakening(self):
        """Test anti-Hebbian weakening when only one block fires"""
        dye_conc = 0.3
        dt = 0.5

        # Only source fires
        delta_w = self.hebbian.calculate_anti_hebbian_update(
            self.wire, True, False, dye_conc, dt
        )
        self.assertLess(delta_w, 0)  # Should be weakening

        # Only target fires
        delta_w = self.hebbian.calculate_anti_hebbian_update(
            self.wire, False, True, dye_conc, dt
        )
        self.assertLess(delta_w, 0)  # Should be weakening


class TestSTDPLearningRule(unittest.TestCase):
    """Test STDP learning rule calculations"""

    def setUp(self):
        self.config = get_default_config().learning
        self.stdp = STDPLearningRule(self.config)
        self.wire = create_wire(1, 1, 2, 2.0, BlockColor.RED)

    def test_stdp_strength_update(self):
        """Test STDP strength update calculation"""
        source_event = FiringEvent(1, 10.0, 0.0, 50.0)
        target_event = FiringEvent(2, 15.0, 0.0, 100.0)  # 5 timesteps later
        dye_conc = 0.4

        delta_w = self.stdp.calculate_stdp_strength_update(
            self.wire, source_event, target_event, dye_conc
        )

        # Should be positive for pre-before-post timing
        self.assertGreater(delta_w, 0)

        # Check exponential decay component
        delta_t = 5.0
        expected_kernel = np.exp(-delta_t / self.config.stdp_time_constant)
        expected_delta = (self.config.wire_learning_rate *
                         self.config.dye_amplification * dye_conc * expected_kernel)
        self.assertAlmostEqual(delta_w, expected_delta, places=6)

    def test_stdp_no_update_reversed_timing(self):
        """Test no STDP update for reversed timing"""
        source_event = FiringEvent(1, 15.0, 0.0, 50.0)
        target_event = FiringEvent(2, 10.0, 0.0, 100.0)  # Target before source

        delta_w = self.stdp.calculate_stdp_strength_update(
            self.wire, source_event, target_event, 0.4
        )
        self.assertEqual(delta_w, 0.0)


class TestThresholdAdaptationRule(unittest.TestCase):
    """Test threshold adaptation rule"""

    def setUp(self):
        self.config = get_default_config().learning
        self.threshold_rule = ThresholdAdaptationRule(self.config)
        self.block = create_block(1, 50.0, BlockColor.RED, 45.0)

    def test_threshold_increase_high_firing_rate(self):
        """Test threshold increase when firing rate is too high"""
        high_firing_rate = self.config.target_firing_rate * 2.0
        dt = 0.5

        delta_threshold = self.threshold_rule.calculate_threshold_update(
            self.block, high_firing_rate, dt
        )

        self.assertGreater(delta_threshold, 0)  # Should increase threshold

    def test_threshold_decrease_low_firing_rate(self):
        """Test threshold decrease when firing rate is too low"""
        low_firing_rate = self.config.target_firing_rate * 0.5
        dt = 0.5

        delta_threshold = self.threshold_rule.calculate_threshold_update(
            self.block, low_firing_rate, dt
        )

        self.assertLess(delta_threshold, 0)  # Should decrease threshold

    def test_no_threshold_change_target_rate(self):
        """Test no threshold change at target firing rate"""
        target_rate = self.config.target_firing_rate
        dt = 0.5

        delta_threshold = self.threshold_rule.calculate_threshold_update(
            self.block, target_rate, dt
        )

        self.assertAlmostEqual(delta_threshold, 0.0, places=6)


class TestPlasticityManager(unittest.TestCase):
    """Test complete plasticity manager functionality"""

    def setUp(self):
        self.config = get_default_config()
        self.plasticity = PlasticityManager(self.config.learning)

        # Create test blocks
        self.blocks = {
            1: create_block(1, 50.0, BlockColor.RED, 45.0),
            2: create_block(2, 100.0, BlockColor.BLUE, 50.0),
            3: create_block(3, 150.0, BlockColor.YELLOW, 55.0)
        }

        # Create test wires
        self.wires = [
            create_wire(1, 1, 2, 2.0, BlockColor.RED, (10.0, 75.0)),
            create_wire(2, 2, 3, 1.5, BlockColor.BLUE, (-5.0, 125.0))
        ]

        # Create dye system
        self.dye_system = create_dye_system_for_network((-50, 50), (0, 200), 2.0)
        self.dye_system.inject_dye(DyeColor.RED, 10.0, 75.0, 0.5)
        self.dye_system.inject_dye(DyeColor.BLUE, -5.0, 125.0, 0.3)

    def test_plasticity_manager_initialization(self):
        """Test plasticity manager initialization"""
        self.assertIsNotNone(self.plasticity.hebbian_rule)
        self.assertIsNotNone(self.plasticity.stdp_rule)
        self.assertIsNotNone(self.plasticity.threshold_rule)
        self.assertIsNotNone(self.plasticity.firing_history)

    def test_firing_event_processing(self):
        """Test processing of firing events"""
        fired_blocks = [1, 2]
        current_time = 10.0

        self.plasticity.process_firing_events(fired_blocks, current_time, self.blocks)

        # Check firing history was updated
        self.assertEqual(len(self.plasticity.firing_history.firing_events), 2)
        self.assertIn(1, self.plasticity.firing_history.block_firing_counts)
        self.assertIn(2, self.plasticity.firing_history.block_firing_counts)

    def test_hebbian_updates_calculation(self):
        """Test Hebbian plasticity updates calculation"""
        # Add some firing history
        self.plasticity.process_firing_events([1, 2], 10.0, self.blocks)

        # Calculate Hebbian updates
        updates = self.plasticity.calculate_hebbian_updates(
            10.5, self.blocks, self.wires, self.dye_system, 0.5
        )

        # Should have updates for wires with activity
        self.assertGreater(len(updates), 0)

        # Check update structure
        for update in updates:
            self.assertEqual(update.target_type, "wire")
            self.assertEqual(update.parameter, "strength")
            self.assertEqual(update.learning_rule, PlasticityType.HEBBIAN)

    def test_plasticity_updates_application(self):
        """Test application of plasticity updates"""
        # Create test updates
        updates = [
            PlasticityUpdate("wire", 1, "strength", 0.1, PlasticityType.HEBBIAN),
            PlasticityUpdate("block", 1, "threshold", -0.5, PlasticityType.THRESHOLD_ADAPTATION)
        ]

        # Record initial values
        initial_strength = self.wires[0].base_strength
        initial_threshold = self.blocks[1].threshold

        # Apply updates
        applied = self.plasticity.apply_plasticity_updates(updates, self.blocks, self.wires)

        self.assertEqual(applied, 2)
        self.assertAlmostEqual(self.wires[0].base_strength, initial_strength + 0.1, places=6)
        self.assertAlmostEqual(self.blocks[1].threshold, initial_threshold - 0.5, places=6)

    def test_strength_bounds_enforcement(self):
        """Test that wire strength bounds are enforced"""
        # Create update that would exceed bounds
        updates = [PlasticityUpdate("wire", 1, "strength", 20.0, PlasticityType.HEBBIAN)]

        self.plasticity.apply_plasticity_updates(
            updates, self.blocks, self.wires, strength_bounds=(0.1, 10.0)
        )

        # Strength should be clamped to maximum
        self.assertEqual(self.wires[0].base_strength, 10.0)

    def test_threshold_bounds_enforcement(self):
        """Test that block threshold bounds are enforced"""
        # Create update that would exceed bounds
        updates = [PlasticityUpdate("block", 1, "threshold", -50.0, PlasticityType.THRESHOLD_ADAPTATION)]

        self.plasticity.apply_plasticity_updates(
            updates, self.blocks, self.wires, threshold_bounds=(30.0, 80.0)
        )

        # Threshold should be clamped to minimum
        self.assertEqual(self.blocks[1].threshold, 30.0)

    def test_complete_plasticity_timestep(self):
        """Test complete plasticity timestep execution"""
        fired_blocks = [1, 2]
        current_time = 10.0
        dt = 0.5

        stats = self.plasticity.execute_plasticity_timestep(
            fired_blocks, current_time, self.blocks, self.wires, self.dye_system, dt
        )

        # Check statistics structure
        required_keys = [
            'hebbian_updates', 'stdp_updates', 'threshold_updates',
            'total_updates', 'updates_applied', 'firing_events_added'
        ]
        for key in required_keys:
            self.assertIn(key, stats)

        # Check that events were processed
        self.assertEqual(stats['firing_events_added'], len(fired_blocks))

    def test_plasticity_statistics(self):
        """Test plasticity statistics collection"""
        stats = self.plasticity.get_plasticity_statistics()

        required_keys = [
            'total_updates_applied', 'updates_by_type', 'firing_history_size',
            'tracked_blocks', 'learning_config'
        ]
        for key in required_keys:
            self.assertIn(key, stats)

        # Check learning config is included
        self.assertIn('wire_learning_rate', stats['learning_config'])
        self.assertIn('dye_amplification', stats['learning_config'])

    def test_plasticity_reset(self):
        """Test plasticity system reset"""
        # Add some history and updates
        self.plasticity.process_firing_events([1, 2], 10.0, self.blocks)
        self.plasticity.total_updates_applied = 5

        # Reset
        self.plasticity.reset_plasticity()

        # Check everything is cleared
        self.assertEqual(len(self.plasticity.firing_history.firing_events), 0)
        self.assertEqual(self.plasticity.total_updates_applied, 0)
        self.assertEqual(len(self.plasticity.firing_history.block_firing_counts), 0)


class TestPlasticitySpecificationCompliance(unittest.TestCase):
    """Test compliance with plasticity specification requirements"""

    def setUp(self):
        self.config = get_default_config()
        self.plasticity = PlasticityManager(self.config.learning)

    def test_hebbian_learning_equation(self):
        """Test Hebbian learning follows specification equation: Δw = η_w × β × C_dye"""
        wire = create_wire(1, 1, 2, 2.0, BlockColor.RED)
        dye_conc = 0.6
        dt = 0.5

        delta_w = self.plasticity.hebbian_rule.calculate_strength_update(
            wire, True, True, dye_conc, dt
        )

        expected = (self.config.learning.wire_learning_rate *
                   self.config.learning.dye_amplification * dye_conc * dt)
        self.assertAlmostEqual(delta_w, expected, places=8)

    def test_stdp_time_constant_compliance(self):
        """Test STDP follows specification time constant"""
        wire = create_wire(1, 1, 2, 2.0, BlockColor.RED)
        source_event = FiringEvent(1, 10.0, 0.0, 50.0)
        target_event = FiringEvent(2, 25.0, 0.0, 100.0)  # 15 timesteps later
        dye_conc = 0.5

        delta_w = self.plasticity.stdp_rule.calculate_stdp_strength_update(
            wire, source_event, target_event, dye_conc
        )

        # Check exponential decay matches specification
        delta_t = 15.0
        expected_kernel = np.exp(-delta_t / self.config.learning.stdp_time_constant)
        expected = (self.config.learning.wire_learning_rate *
                   self.config.learning.dye_amplification * dye_conc * expected_kernel)
        self.assertAlmostEqual(delta_w, expected, places=8)

    def test_threshold_adaptation_homeostasis(self):
        """Test threshold adaptation maintains firing rate homeostasis"""
        block = create_block(1, 50.0, BlockColor.RED, 45.0)

        # High firing rate should increase threshold
        high_rate = self.config.learning.target_firing_rate * 1.5
        delta_high = self.plasticity.threshold_rule.calculate_threshold_update(
            block, high_rate, 0.5
        )
        self.assertGreater(delta_high, 0)

        # Low firing rate should decrease threshold
        low_rate = self.config.learning.target_firing_rate * 0.5
        delta_low = self.plasticity.threshold_rule.calculate_threshold_update(
            block, low_rate, 0.5
        )
        self.assertLess(delta_low, 0)

    def test_parameter_bounds_compliance(self):
        """Test all plasticity parameters are within specification bounds"""
        config = self.config.learning

        # Wire learning rate bounds (0.001-0.01)
        self.assertGreaterEqual(config.wire_learning_rate, 0.001)
        self.assertLessEqual(config.wire_learning_rate, 0.01)

        # Dye amplification bounds (2.0-5.0)
        self.assertGreaterEqual(config.dye_amplification, 2.0)
        self.assertLessEqual(config.dye_amplification, 5.0)

        # STDP time constant bounds (10-20)
        self.assertGreaterEqual(config.stdp_time_constant, 10.0)
        self.assertLessEqual(config.stdp_time_constant, 20.0)

        # Threshold learning rate bounds (0.001-0.01)
        self.assertGreaterEqual(config.threshold_learning_rate, 0.001)
        self.assertLessEqual(config.threshold_learning_rate, 0.01)

        # Target firing rate bounds (0.05-0.2)
        self.assertGreaterEqual(config.target_firing_rate, 0.05)
        self.assertLessEqual(config.target_firing_rate, 0.2)


if __name__ == '__main__':
    unittest.main()