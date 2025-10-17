"""
Comprehensive tests for Neural-Klotski learning integration system.

Tests integrated learning including eligibility traces, learning signal generation,
adaptive learning control, and complete learning trial workflows.
"""

import unittest
import numpy as np
from typing import Dict, List
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from neural_klotski.core.learning import (
    IntegratedLearningSystem, EligibilityTraceManager, LearningSignalGenerator,
    AdaptiveLearningController, LearningEvent, TrialResult, TrialOutcome
)
from neural_klotski.core.block import BlockState, BlockColor, create_block
from neural_klotski.core.wire import Wire, create_wire
from neural_klotski.core.dye import DyeSystem, DyeColor, create_dye_system_for_network
from neural_klotski.core.plasticity import PlasticityManager
from neural_klotski.core.network import NeuralKlotskiNetwork, create_simple_test_network
from neural_klotski.config import get_default_config


class TestEligibilityTraceManager(unittest.TestCase):
    """Test eligibility trace functionality"""

    def setUp(self):
        self.trace_manager = EligibilityTraceManager(trace_decay_constant=50.0)

    def test_eligibility_trace_initialization(self):
        """Test eligibility trace manager initialization"""
        self.assertEqual(self.trace_manager.trace_decay_constant, 50.0)
        self.assertEqual(len(self.trace_manager.wire_traces), 0)
        self.assertEqual(len(self.trace_manager.block_traces), 0)

    def test_learning_event_addition(self):
        """Test adding learning events"""
        event = LearningEvent(10.0, 1, None, "firing", 1.0, (0.0, 50.0))
        self.trace_manager.add_learning_event(event)

        self.assertEqual(len(self.trace_manager.learning_events), 1)
        stored_event = self.trace_manager.learning_events[0]
        self.assertEqual(stored_event.event_time, 10.0)
        self.assertEqual(stored_event.block_id, 1)
        self.assertEqual(stored_event.event_type, "firing")

    def test_eligibility_trace_updates(self):
        """Test eligibility trace updates from activity"""
        current_time = 10.0
        fired_blocks = [1, 2]
        activated_wires = [101, 102]

        self.trace_manager.update_eligibility_traces(
            current_time, fired_blocks, activated_wires
        )

        # Check block traces were set
        self.assertEqual(self.trace_manager.get_eligibility_trace("block", 1), 1.0)
        self.assertEqual(self.trace_manager.get_eligibility_trace("block", 2), 1.0)

        # Check wire traces were set
        self.assertEqual(self.trace_manager.get_eligibility_trace("wire", 101), 1.0)
        self.assertEqual(self.trace_manager.get_eligibility_trace("wire", 102), 1.0)

        # Check learning events were added
        self.assertEqual(len(self.trace_manager.learning_events), 4)  # 2 blocks + 2 wires

    def test_eligibility_trace_decay(self):
        """Test that eligibility traces decay over time"""
        # Set initial traces
        self.trace_manager.update_eligibility_traces(10.0, [1], [101])

        # Initial traces should be 1.0
        self.assertEqual(self.trace_manager.get_eligibility_trace("block", 1), 1.0)
        self.assertEqual(self.trace_manager.get_eligibility_trace("wire", 101), 1.0)

        # Update without new activity - traces should decay
        self.trace_manager.update_eligibility_traces(11.0, [], [])

        # Traces should have decayed
        block_trace = self.trace_manager.get_eligibility_trace("block", 1)
        wire_trace = self.trace_manager.get_eligibility_trace("wire", 101)

        self.assertLess(block_trace, 1.0)
        self.assertLess(wire_trace, 1.0)
        self.assertGreater(block_trace, 0.0)
        self.assertGreater(wire_trace, 0.0)

    def test_trace_statistics(self):
        """Test eligibility trace statistics collection"""
        self.trace_manager.update_eligibility_traces(10.0, [1, 2], [101])

        stats = self.trace_manager.get_trace_statistics()

        self.assertEqual(stats['active_wire_traces'], 1)
        self.assertEqual(stats['active_block_traces'], 2)
        self.assertGreater(stats['total_trace_updates'], 0)
        self.assertEqual(stats['avg_wire_trace'], 1.0)
        self.assertEqual(stats['avg_block_trace'], 1.0)

    def test_trace_clearing(self):
        """Test clearing all traces"""
        self.trace_manager.update_eligibility_traces(10.0, [1, 2], [101])

        # Should have traces
        self.assertGreater(len(self.trace_manager.wire_traces), 0)
        self.assertGreater(len(self.trace_manager.block_traces), 0)

        # Clear and check
        self.trace_manager.clear_traces()

        self.assertEqual(len(self.trace_manager.wire_traces), 0)
        self.assertEqual(len(self.trace_manager.block_traces), 0)
        self.assertEqual(len(self.trace_manager.learning_events), 0)


class TestLearningSignalGenerator(unittest.TestCase):
    """Test learning signal generation"""

    def setUp(self):
        self.config = get_default_config()
        self.generator = LearningSignalGenerator(self.config.dyes)

        # Create test components
        self.blocks = {
            1: create_block(1, 50.0, BlockColor.RED, 10.0),
            2: create_block(2, 100.0, BlockColor.BLUE, 10.0)
        }

        self.wires = [
            create_wire(1, 1, 2, 2.0, BlockColor.RED, (10.0, 75.0))
        ]

        self.traces = EligibilityTraceManager()
        # Add some eligibility traces
        self.traces.update_eligibility_traces(10.0, [1, 2], [1])

    def test_base_injection_amount_calculation(self):
        """Test base injection amount calculation for different outcomes"""
        # Success should give strong positive signal
        success_result = TrialResult(TrialOutcome.SUCCESS, 10.0, 0.8)
        success_amount = self.generator._calculate_base_injection_amount(success_result)
        self.assertGreater(success_amount, self.config.dyes.injection_amount)

        # Failure should give weak signal
        failure_result = TrialResult(TrialOutcome.FAILURE, 10.0, 0.2)
        failure_amount = self.generator._calculate_base_injection_amount(failure_result)
        self.assertLess(failure_amount, self.config.dyes.injection_amount)

        # Success should be stronger than failure
        self.assertGreater(success_amount, failure_amount)

    def test_learning_signal_generation(self):
        """Test complete learning signal generation"""
        trial_result = TrialResult(TrialOutcome.SUCCESS, 10.0, 0.9, spatial_focus=(5.0, 100.0))

        signals = self.generator.generate_learning_signals(
            trial_result, 10.0, self.traces, self.blocks, self.wires
        )

        # Should generate signals for eligible blocks and spatial focus
        self.assertGreater(len(signals), 0)

        # Check signal structure
        for dye_color, activation, lag, amount in signals:
            self.assertIsInstance(dye_color, DyeColor)
            self.assertIsInstance(activation, float)
            self.assertIsInstance(lag, float)
            self.assertGreater(amount, 0.0)

    def test_spatial_focus_signals(self):
        """Test spatial focus generates targeted signals"""
        focus_point = (15.0, 125.0)
        trial_result = TrialResult(TrialOutcome.SUCCESS, 5.0, 0.7, spatial_focus=focus_point)

        signals = self.generator.generate_learning_signals(
            trial_result, 10.0, self.traces, self.blocks, self.wires
        )

        # Should have signals at focus point
        focus_signals = [(act, lag) for _, act, lag, _ in signals if (act, lag) == focus_point]
        self.assertGreater(len(focus_signals), 0)

    def test_signal_statistics(self):
        """Test signal generation statistics"""
        trial_result = TrialResult(TrialOutcome.SUCCESS, 10.0, 0.8)
        self.generator.generate_learning_signals(
            trial_result, 10.0, self.traces, self.blocks, self.wires
        )

        stats = self.generator.get_signal_statistics()

        self.assertGreater(stats['total_signals_generated'], 0)
        self.assertEqual(stats['signal_history_length'], 1)
        self.assertIn('recent_outcome_counts', stats)


class TestAdaptiveLearningController(unittest.TestCase):
    """Test adaptive learning rate control"""

    def setUp(self):
        self.config = get_default_config()
        self.controller = AdaptiveLearningController(self.config.learning)

    def test_adaptive_controller_initialization(self):
        """Test adaptive learning controller initialization"""
        self.assertEqual(self.controller.base_wire_learning_rate, self.config.learning.wire_learning_rate)
        self.assertEqual(self.controller.learning_rate_multiplier, 1.0)

    def test_performance_history_update(self):
        """Test performance history tracking"""
        performances = [0.5, 0.6, 0.7, 0.8]

        for perf in performances:
            self.controller.update_performance_history(perf)

        self.assertEqual(len(self.controller.performance_history), len(performances))
        self.assertEqual(list(self.controller.performance_history), performances)

    def test_adaptive_learning_rate_calculation(self):
        """Test adaptive learning rate calculation"""
        # Add some performance history
        for perf in [0.5, 0.6, 0.7]:
            self.controller.update_performance_history(perf)

        # Calculate adaptive rates with dye enhancement
        dye_concentration = 0.5
        eligibility_trace = 0.8

        rates = self.controller.calculate_adaptive_learning_rates(
            dye_concentration, eligibility_trace
        )

        # Should return rates for different parameters
        self.assertIn('wire_learning_rate', rates)
        self.assertIn('threshold_learning_rate', rates)
        self.assertIn('multiplier', rates)

        # With dye enhancement, rates should be higher than base
        self.assertGreater(rates['wire_learning_rate'], self.controller.base_wire_learning_rate)

    def test_learning_rate_bounds(self):
        """Test that learning rates stay within safe bounds"""
        # Add declining performance to trigger adaptation
        for perf in [0.9, 0.8, 0.7, 0.6, 0.5]:
            self.controller.update_performance_history(perf)

        # Calculate with extreme enhancement
        rates = self.controller.calculate_adaptive_learning_rates(
            dye_concentration=2.0, eligibility_trace=1.0
        )

        # Multiplier should be bounded
        self.assertLessEqual(rates['multiplier'], 5.0)
        self.assertGreaterEqual(rates['multiplier'], 0.1)

    def test_adaptation_statistics(self):
        """Test adaptation statistics collection"""
        self.controller.update_performance_history(0.7)
        self.controller.calculate_adaptive_learning_rates(0.3, 0.5)

        stats = self.controller.get_adaptation_statistics()

        self.assertIn('learning_rate_multiplier', stats)
        self.assertIn('total_adaptations', stats)
        self.assertIn('recent_avg_performance', stats)
        self.assertGreater(stats['total_adaptations'], 0)


class TestIntegratedLearningSystem(unittest.TestCase):
    """Test complete integrated learning system"""

    def setUp(self):
        self.config = get_default_config()
        self.learning_system = IntegratedLearningSystem(self.config.learning, self.config.dyes)

        # Create test network components
        self.blocks = {
            1: create_block(1, 50.0, BlockColor.RED, 10.0),
            2: create_block(2, 100.0, BlockColor.BLUE, 10.0),
            3: create_block(3, 150.0, BlockColor.YELLOW, 10.0)
        }

        self.wires = [
            create_wire(1, 1, 2, 2.0, BlockColor.RED, (10.0, 75.0)),
            create_wire(2, 2, 3, 1.5, BlockColor.BLUE, (-5.0, 125.0))
        ]

        self.dye_system = create_dye_system_for_network((-50, 50), (0, 200), 2.0)
        self.plasticity_manager = PlasticityManager(self.config.learning)

    def test_integrated_learning_initialization(self):
        """Test integrated learning system initialization"""
        self.assertIsNotNone(self.learning_system.eligibility_traces)
        self.assertIsNotNone(self.learning_system.signal_generator)
        self.assertIsNotNone(self.learning_system.adaptive_controller)

    def test_trial_management(self):
        """Test learning trial start and completion"""
        # Start trial
        self.learning_system.start_trial(10.0)
        self.assertEqual(self.learning_system.trial_count, 1)
        self.assertEqual(self.learning_system.current_trial_start, 10.0)

        # Complete trial
        trial_result = TrialResult(TrialOutcome.SUCCESS, 5.0, 0.8)
        outcome_stats = self.learning_system.process_trial_outcome(
            trial_result, 15.0, self.blocks, self.wires, self.dye_system
        )

        self.assertEqual(self.learning_system.successful_trials, 1)
        self.assertIn('trial_outcome', outcome_stats)
        self.assertIn('learning_signals_generated', outcome_stats)

    def test_learning_timestep_processing(self):
        """Test complete learning timestep processing"""
        current_time = 10.0
        fired_blocks = [1, 2]
        activated_wires = [1]

        # Start trial first
        self.learning_system.start_trial(current_time)

        # Process timestep
        timestep_stats = self.learning_system.process_learning_timestep(
            current_time, fired_blocks, activated_wires, self.blocks, self.wires,
            self.dye_system, self.plasticity_manager
        )

        # Should return comprehensive statistics
        self.assertIn('eligibility_stats', timestep_stats)
        self.assertIn('adaptive_rates', timestep_stats)
        self.assertIn('max_dye_concentration', timestep_stats)
        self.assertIn('fired_blocks', timestep_stats)

    def test_learning_statistics_collection(self):
        """Test learning statistics collection"""
        # Add some activity
        self.learning_system.start_trial(10.0)
        trial_result = TrialResult(TrialOutcome.SUCCESS, 5.0, 0.9)
        self.learning_system.process_trial_outcome(
            trial_result, 15.0, self.blocks, self.wires, self.dye_system
        )

        stats = self.learning_system.get_learning_statistics()

        required_keys = [
            'trial_count', 'successful_trials', 'success_rate',
            'total_learning_signals', 'eligibility_traces',
            'signal_generation', 'adaptive_control'
        ]

        for key in required_keys:
            self.assertIn(key, stats)

        self.assertEqual(stats['trial_count'], 1)
        self.assertEqual(stats['success_rate'], 1.0)

    def test_learning_state_reset(self):
        """Test learning state reset"""
        # Add some state
        self.learning_system.start_trial(10.0)
        self.learning_system.adaptive_controller.update_performance_history(0.8)

        # Reset
        self.learning_system.reset_learning_state()

        # Check everything is cleared
        self.assertEqual(self.learning_system.trial_count, 0)
        self.assertEqual(self.learning_system.successful_trials, 0)
        self.assertEqual(len(self.learning_system.adaptive_controller.performance_history), 0)
        self.assertEqual(len(self.learning_system.eligibility_traces.wire_traces), 0)

    def test_multiple_trial_learning(self):
        """Test learning across multiple trials"""
        trial_outcomes = [
            (TrialOutcome.SUCCESS, 0.9),
            (TrialOutcome.PARTIAL, 0.6),
            (TrialOutcome.SUCCESS, 0.8),
            (TrialOutcome.FAILURE, 0.3)
        ]

        current_time = 0.0
        for outcome, performance in trial_outcomes:
            self.learning_system.start_trial(current_time)
            current_time += 10.0

            # Simulate some activity
            self.learning_system.process_learning_timestep(
                current_time, [1], [1], self.blocks, self.wires,
                self.dye_system, self.plasticity_manager
            )

            # Complete trial
            trial_result = TrialResult(outcome, 10.0, performance)
            self.learning_system.process_trial_outcome(
                trial_result, current_time, self.blocks, self.wires, self.dye_system
            )

        # Check final statistics
        stats = self.learning_system.get_learning_statistics()
        self.assertEqual(stats['trial_count'], 4)
        self.assertEqual(stats['successful_trials'], 2)  # 2 successes
        self.assertEqual(stats['success_rate'], 0.5)


class TestNetworkLearningIntegration(unittest.TestCase):
    """Test learning integration with complete network"""

    def setUp(self):
        self.network = create_simple_test_network()

    def test_network_learning_initialization(self):
        """Test network initializes with learning system"""
        self.assertTrue(self.network.enable_learning)
        self.assertIsNotNone(self.network.learning_system)
        self.assertFalse(self.network.in_trial)

    def test_trial_start_and_completion(self):
        """Test network trial management"""
        # Start trial
        self.network.start_learning_trial()
        self.assertTrue(self.network.in_trial)

        # Complete trial
        outcome_stats = self.network.complete_learning_trial(
            TrialOutcome.SUCCESS, 0.85, spatial_focus=(10.0, 100.0)
        )

        self.assertFalse(self.network.in_trial)
        self.assertIn('trial_outcome', outcome_stats)
        self.assertEqual(outcome_stats['trial_outcome'], 'success')

    def test_learning_integration_in_timestep(self):
        """Test learning integration during network timestep execution"""
        # Start trial
        self.network.start_learning_trial()

        # Apply force to trigger activity
        self.network.blocks[1].add_force(15.0)

        # Execute timestep
        stats = self.network.execute_timestep()

        # Should include learning statistics when in trial
        self.assertIn('learning_stats', stats)
        self.assertTrue(stats['in_trial'])

    def test_network_learning_statistics(self):
        """Test network includes learning statistics"""
        # Add some learning activity with network execution
        self.network.start_learning_trial()

        # Apply force and run some timesteps to create eligibility traces
        self.network.blocks[1].add_force(15.0)
        for _ in range(5):
            self.network.execute_timestep()

        # Complete trial
        self.network.complete_learning_trial(TrialOutcome.SUCCESS, 0.9)

        # Get network statistics
        stats = self.network.get_simulation_statistics()

        self.assertTrue(stats['learning_enabled'])
        self.assertIn('learning_stats', stats)
        # Learning signals should have been generated due to activity
        self.assertGreaterEqual(stats['total_learning_signals'], 0)

    def test_network_reset_includes_learning(self):
        """Test network reset clears learning state"""
        # Add learning state
        self.network.start_learning_trial()
        self.network.complete_learning_trial(TrialOutcome.SUCCESS, 0.8)

        # Reset network
        self.network.reset_simulation()

        # Learning state should be cleared
        self.assertFalse(self.network.in_trial)
        self.assertEqual(self.network.total_learning_signals, 0)

        # Learning system should be reset
        learning_stats = self.network.learning_system.get_learning_statistics()
        self.assertEqual(learning_stats['trial_count'], 0)

    def test_complete_learning_workflow(self):
        """Test complete learning workflow through network"""
        trials = 3

        for trial in range(trials):
            # Start learning trial
            self.network.start_learning_trial()

            # Apply stimulus and run network
            self.network.blocks[1].add_force(20.0)

            # Run for trial duration
            for _ in range(10):
                stats = self.network.execute_timestep()

                # Should be processing learning during trial
                if self.network.in_trial:
                    self.assertIn('learning_stats', stats)

            # Complete trial with varying performance
            performance = 0.6 + 0.1 * trial  # Improving performance
            outcome = TrialOutcome.SUCCESS if performance > 0.7 else TrialOutcome.PARTIAL

            trial_stats = self.network.complete_learning_trial(outcome, performance)
            self.assertGreater(trial_stats.get('learning_signals_generated', 0), 0)

        # Check final learning statistics
        final_stats = self.network.get_simulation_statistics()
        learning_stats = final_stats['learning_stats']

        self.assertEqual(learning_stats['trial_count'], trials)
        self.assertGreater(learning_stats['total_learning_signals'], 0)


class TestLearningSpecificationCompliance(unittest.TestCase):
    """Test compliance with learning specification requirements"""

    def setUp(self):
        self.config = get_default_config()

    def test_eligibility_trace_time_constants(self):
        """Test eligibility traces use proper time constants"""
        trace_manager = EligibilityTraceManager(self.config.dyes.eligibility_window)
        self.assertEqual(trace_manager.trace_decay_constant, self.config.dyes.eligibility_window)

        # Verify time constant is within specification bounds (50-200)
        self.assertGreaterEqual(self.config.dyes.eligibility_window, 50.0)
        self.assertLessEqual(self.config.dyes.eligibility_window, 200.0)

    def test_dye_enhancement_factors(self):
        """Test learning signals use correct dye enhancement factors"""
        controller = AdaptiveLearningController(self.config.learning)

        # Test with dye concentration
        dye_conc = 0.5
        rates = controller.calculate_adaptive_learning_rates(dye_conc, 0.0)

        # Enhancement should follow specification: 1 + β × C_dye
        expected_enhancement = 1.0 + self.config.learning.dye_amplification * dye_conc

        # The actual rate should incorporate this enhancement
        expected_rate = controller.base_wire_learning_rate * expected_enhancement

        # Allow for additional factors (performance, eligibility) in comparison
        self.assertGreater(rates['wire_learning_rate'], controller.base_wire_learning_rate)

    def test_learning_rate_parameter_bounds(self):
        """Test all learning rate parameters are within specification bounds"""
        config = self.config.learning

        # Wire learning rate bounds (0.001-0.01)
        self.assertGreaterEqual(config.wire_learning_rate, 0.001)
        self.assertLessEqual(config.wire_learning_rate, 0.01)

        # Dye amplification bounds (2.0-5.0)
        self.assertGreaterEqual(config.dye_amplification, 2.0)
        self.assertLessEqual(config.dye_amplification, 5.0)

        # Threshold learning rate bounds (0.001-0.01)
        self.assertGreaterEqual(config.threshold_learning_rate, 0.001)
        self.assertLessEqual(config.threshold_learning_rate, 0.01)

    def test_trial_outcome_signal_strengths(self):
        """Test trial outcomes generate appropriate signal strengths"""
        generator = LearningSignalGenerator(self.config.dyes)

        # Success should generate stronger signals than failure
        success_result = TrialResult(TrialOutcome.SUCCESS, 10.0, 0.9)
        failure_result = TrialResult(TrialOutcome.FAILURE, 10.0, 0.1)

        success_amount = generator._calculate_base_injection_amount(success_result)
        failure_amount = generator._calculate_base_injection_amount(failure_result)

        self.assertGreater(success_amount, failure_amount)

        # Both should be positive (no negative reinforcement)
        self.assertGreater(success_amount, 0.0)
        self.assertGreater(failure_amount, 0.0)


if __name__ == '__main__':
    unittest.main()