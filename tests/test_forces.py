"""
Unit tests for Force Application and Network Integration systems.

Tests all aspects of force calculations and network dynamics including:
- Wire color-specific force calculations
- Signal-to-force conversion and application
- Complete network timestep execution
- Signal queue management and temporal ordering
- Network validation and error handling
"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from neural_klotski.core.forces import (
    ForceCalculator, SignalProcessor, NetworkSignalManager, ForceApplication
)
from neural_klotski.core.network import NeuralKlotskiNetwork, NetworkState, create_simple_test_network
from neural_klotski.core.block import BlockState, BlockColor, create_block
from neural_klotski.core.wire import Signal, SignalQueue, Wire, create_wire
from neural_klotski.config import get_default_config


class TestForceCalculator:
    """Test force calculation for different wire colors"""

    def test_red_wire_force(self):
        """Test excitatory red wire force calculation"""
        signal = Signal(10.0, 2.5, BlockColor.RED, 1, 2)
        force = ForceCalculator.calculate_red_wire_force(signal)
        assert force == 2.5  # Positive (rightward) force

    def test_blue_wire_force(self):
        """Test inhibitory blue wire force calculation"""
        signal = Signal(10.0, 1.8, BlockColor.BLUE, 1, 2)
        force = ForceCalculator.calculate_blue_wire_force(signal)
        assert force == -1.8  # Negative (leftward) force

    def test_yellow_wire_force(self):
        """Test electrical coupling yellow wire force calculation"""
        signal = Signal(10.0, 1.5, BlockColor.YELLOW, 1, 2)

        # Source position > target position (rightward force)
        force1 = ForceCalculator.calculate_yellow_wire_force(signal, 5.0, 3.0)
        expected1 = 1.5 * (5.0 - 3.0)  # 1.5 * 2.0 = 3.0
        assert force1 == expected1

        # Source position < target position (leftward force)
        force2 = ForceCalculator.calculate_yellow_wire_force(signal, 2.0, 7.0)
        expected2 = 1.5 * (2.0 - 7.0)  # 1.5 * (-5.0) = -7.5
        assert force2 == expected2

        # Equal positions (no force)
        force3 = ForceCalculator.calculate_yellow_wire_force(signal, 4.0, 4.0)
        assert force3 == 0.0

    def test_calculate_signal_force_integration(self):
        """Test integrated signal force calculation"""
        calculator = ForceCalculator()

        # Red signal
        red_signal = Signal(10.0, 2.0, BlockColor.RED, 1, 2)
        red_force = calculator.calculate_signal_force(red_signal)
        assert red_force == 2.0

        # Blue signal
        blue_signal = Signal(10.0, 3.0, BlockColor.BLUE, 1, 2)
        blue_force = calculator.calculate_signal_force(blue_signal)
        assert blue_force == -3.0

        # Yellow signal with positions
        yellow_signal = Signal(10.0, 1.5, BlockColor.YELLOW, 1, 2)
        yellow_force = calculator.calculate_signal_force(yellow_signal, 8.0, 3.0)
        expected = 1.5 * (8.0 - 3.0)  # 1.5 * 5.0 = 7.5
        assert yellow_force == expected

    def test_yellow_wire_missing_positions(self):
        """Test that yellow wire requires position information"""
        calculator = ForceCalculator()
        yellow_signal = Signal(10.0, 1.5, BlockColor.YELLOW, 1, 2)

        # Missing both positions
        with pytest.raises(ValueError, match="Yellow wire force calculation requires source and target positions"):
            calculator.calculate_signal_force(yellow_signal)

        # Missing target position
        with pytest.raises(ValueError, match="Yellow wire force calculation requires source and target positions"):
            calculator.calculate_signal_force(yellow_signal, source_position=5.0)

        # Missing source position
        with pytest.raises(ValueError, match="Yellow wire force calculation requires source and target positions"):
            calculator.calculate_signal_force(yellow_signal, target_position=3.0)


class TestSignalProcessor:
    """Test signal processing and force application"""

    def test_process_signals_for_timestep(self):
        """Test processing signals for current timestep"""
        processor = SignalProcessor()

        # Create test blocks
        blocks = {
            1: create_block(1, 50.0, BlockColor.RED, 10.0, initial_position=5.0),
            2: create_block(2, 100.0, BlockColor.BLUE, 10.0, initial_position=-2.0),
            3: create_block(3, 150.0, BlockColor.YELLOW, 10.0, initial_position=3.0)
        }

        # Create signal queue with different arrival times
        queue = SignalQueue()
        queue.add_signal(Signal(10.0, 2.0, BlockColor.RED, 1, 2))      # Current time
        queue.add_signal(Signal(10.0, 1.5, BlockColor.YELLOW, 1, 3))   # Current time
        queue.add_signal(Signal(12.0, 3.0, BlockColor.BLUE, 2, 1))     # Future time

        # Process signals at t=10.0
        force_apps = processor.process_signals_for_timestep(queue, 10.0, blocks)

        # Should process 2 signals (not the future one)
        assert len(force_apps) == 2

        # Check force applications
        red_app = next(app for app in force_apps if app.source_signal.color == BlockColor.RED)
        assert red_app.target_block_id == 2
        assert red_app.force_magnitude == 2.0

        yellow_app = next(app for app in force_apps if app.source_signal.color == BlockColor.YELLOW)
        assert yellow_app.target_block_id == 3
        expected_yellow = 1.5 * (5.0 - 3.0)  # 1.5 * 2.0 = 3.0
        assert yellow_app.force_magnitude == expected_yellow

    def test_group_forces_by_target(self):
        """Test grouping and summing forces by target block"""
        processor = SignalProcessor()

        # Create force applications with multiple forces to same target
        force_apps = [
            ForceApplication(2, 3.0, Signal(10.0, 3.0, BlockColor.RED, 1, 2)),
            ForceApplication(2, -1.5, Signal(10.0, 1.5, BlockColor.BLUE, 3, 2)),
            ForceApplication(3, 2.0, Signal(10.0, 2.0, BlockColor.RED, 4, 3)),
            ForceApplication(2, 0.5, Signal(10.0, 0.5, BlockColor.RED, 5, 2))
        ]

        force_totals = processor.group_forces_by_target(force_apps)

        # Check totals
        assert force_totals[2] == 2.0  # 3.0 - 1.5 + 0.5 = 2.0
        assert force_totals[3] == 2.0

    def test_apply_forces_to_blocks(self):
        """Test applying forces to blocks"""
        processor = SignalProcessor()

        # Create test blocks
        blocks = {
            1: create_block(1, 50.0, BlockColor.RED, 10.0),
            2: create_block(2, 100.0, BlockColor.BLUE, 10.0),
            3: create_block(3, 150.0, BlockColor.YELLOW, 10.0)
        }

        # Initial forces should be zero
        for block in blocks.values():
            assert block.total_force == 0.0

        # Apply forces
        force_totals = {1: 5.0, 2: -3.0, 3: 0.0, 999: 10.0}  # Include non-existent block
        blocks_affected = processor.apply_forces_to_blocks(force_totals, blocks)

        # Check that forces were applied
        assert blocks[1].total_force == 5.0
        assert blocks[2].total_force == -3.0
        assert blocks[3].total_force == 0.0
        assert blocks_affected == 3  # Non-existent block should be skipped

    def test_apply_forces_with_magnitude_limit(self):
        """Test force application with magnitude limits"""
        processor = SignalProcessor()

        blocks = {1: create_block(1, 50.0, BlockColor.RED, 10.0)}
        force_totals = {1: 100.0}  # Very large force

        # Apply with limit
        processor.apply_forces_to_blocks(force_totals, blocks, max_force_magnitude=10.0)
        assert blocks[1].total_force == 10.0  # Should be clamped

        # Reset and test negative limit
        blocks[1].reset_forces()
        force_totals = {1: -100.0}
        processor.apply_forces_to_blocks(force_totals, blocks, max_force_magnitude=10.0)
        assert blocks[1].total_force == -10.0  # Should be clamped to -10.0

    def test_complete_timestep_processing(self):
        """Test complete signal processing pipeline"""
        processor = SignalProcessor()

        # Create test scenario
        blocks = {
            1: create_block(1, 50.0, BlockColor.RED, 10.0, initial_position=4.0),
            2: create_block(2, 100.0, BlockColor.BLUE, 10.0, initial_position=-1.0)
        }

        queue = SignalQueue()
        queue.add_signal(Signal(15.0, 2.5, BlockColor.RED, 1, 2))
        queue.add_signal(Signal(15.0, 1.8, BlockColor.BLUE, 2, 1))

        # Process timestep
        signals_processed, blocks_affected = processor.process_timestep(queue, 15.0, blocks)

        assert signals_processed == 2
        assert blocks_affected == 2
        assert blocks[1].total_force == -1.8  # From blue signal
        assert blocks[2].total_force == 2.5   # From red signal


class TestNetworkSignalManager:
    """Test high-level signal management"""

    def test_add_firing_signals(self):
        """Test creating signals from fired blocks"""
        manager = NetworkSignalManager()

        # Create blocks and wires
        blocks = {
            1: create_block(1, 50.0, BlockColor.RED, 10.0),
            2: create_block(2, 100.0, BlockColor.BLUE, 10.0),
            3: create_block(3, 150.0, BlockColor.YELLOW, 10.0)
        }

        wires = [
            create_wire(1, 1, 2, 2.0, BlockColor.RED),
            create_wire(2, 1, 3, 1.5, BlockColor.RED),
            create_wire(3, 2, 1, 1.8, BlockColor.BLUE)
        ]

        # Fire block 1
        fired_blocks = [1]
        signals_created = manager.add_firing_signals(
            fired_blocks, 10.0, blocks, wires
        )

        # Block 1 should create 2 signals (wires 1 and 2)
        assert signals_created == 2
        assert len(manager.signal_queue) == 2

    def test_process_network_timestep(self):
        """Test complete network timestep processing"""
        manager = NetworkSignalManager()

        # Add some signals
        signal1 = Signal(20.0, 3.0, BlockColor.RED, 1, 2)
        signal2 = Signal(20.0, 2.0, BlockColor.BLUE, 3, 1)
        manager.signal_queue.add_signal(signal1)
        manager.signal_queue.add_signal(signal2)

        # Create blocks
        blocks = {
            1: create_block(1, 50.0, BlockColor.RED, 10.0),
            2: create_block(2, 100.0, BlockColor.BLUE, 10.0),
            3: create_block(3, 150.0, BlockColor.YELLOW, 10.0)
        }

        # Process timestep
        signals_processed, blocks_affected = manager.process_network_timestep(20.0, blocks)

        assert signals_processed == 2
        assert blocks_affected == 2
        assert len(manager.signal_queue) == 0  # Signals should be consumed


class TestNeuralKlotskiNetwork:
    """Test complete network integration"""

    def test_network_creation(self):
        """Test creating network with blocks and wires"""
        network = NeuralKlotskiNetwork()

        # Add blocks
        block1 = network.create_block(1, 50.0, BlockColor.RED, 10.0)
        block2 = network.create_block(2, 100.0, BlockColor.BLUE, 10.0)

        assert len(network.blocks) == 2
        assert network.blocks[1] == block1
        assert network.blocks[2] == block2

        # Add wire
        wire = network.create_wire(1, 1, 2, 2.5)
        assert len(network.wires) == 1
        assert wire.source_block_id == 1
        assert wire.target_block_id == 2
        assert wire.color == BlockColor.RED  # Inherited from source

    def test_network_validation(self):
        """Test network validation"""
        network = NeuralKlotskiNetwork()

        # Valid network
        network.create_block(1, 50.0, BlockColor.RED, 10.0)
        network.create_block(2, 100.0, BlockColor.BLUE, 10.0)
        network.create_wire(1, 1, 2, 2.0)

        is_valid, errors = network.validate_network()
        assert is_valid
        assert len(errors) == 0

        # Invalid network - non-existent source block (bypass create_wire validation)
        invalid_wire = Wire(2, 999, 2, 1.5, BlockColor.RED)
        network.wires.append(invalid_wire)
        is_valid, errors = network.validate_network()
        assert not is_valid
        assert any("non-existent source block" in error for error in errors)

    def test_timestep_execution(self):
        """Test single timestep execution"""
        network = create_simple_test_network()

        # Apply force to trigger activity
        network.blocks[1].add_force(15.0)

        # Execute timestep
        stats = network.execute_timestep()

        assert stats['timestep'] == 1
        assert stats['time'] == network.config.dynamics.dt
        assert 'blocks_fired' in stats
        assert 'signals_created' in stats
        assert 'signals_processed' in stats

    def test_simulation_run(self):
        """Test running multi-timestep simulation"""
        network = create_simple_test_network()

        # Apply initial force
        network.blocks[1].add_force(20.0)

        # Run simulation
        stats_list = network.run_simulation(5, verbose=False)

        assert len(stats_list) == 5
        assert network.total_timesteps == 5
        assert network.current_time == 5 * network.config.dynamics.dt

        # Check that some activity occurred
        total_firings = sum(stats['blocks_fired'] for stats in stats_list)
        assert total_firings > 0

    def test_simulation_reset(self):
        """Test simulation reset functionality"""
        network = create_simple_test_network()

        # Run some simulation
        network.blocks[1].add_force(20.0)
        network.run_simulation(3)

        # Verify state changed
        assert network.total_timesteps == 3
        assert network.current_time > 0

        # Reset
        network.reset_simulation()

        # Verify reset
        assert network.total_timesteps == 0
        assert network.current_time == 0.0
        assert network.total_firings == 0
        assert len(network.signal_manager.signal_queue) == 0

        # Check blocks reset
        for block in network.blocks.values():
            assert block.total_force == 0.0
            assert block.refractory_timer == 0.0

    def test_network_state_snapshot(self):
        """Test network state capture"""
        network = create_simple_test_network()

        state = network.get_network_state()

        assert isinstance(state, NetworkState)
        assert state.current_time == 0.0
        assert len(state.blocks) == 3
        assert len(state.wires) == 3
        assert isinstance(state.dye_concentrations, dict)

        summary = state.get_network_summary()
        assert summary['blocks'] == 3
        assert summary['wires'] == 3
        assert summary['time'] == 0.0

    def test_simple_test_network_creation(self):
        """Test the simple test network factory function"""
        network = create_simple_test_network()

        # Should have 3 blocks
        assert len(network.blocks) == 3

        # Should have 3 wires
        assert len(network.wires) == 3

        # Check block colors
        assert network.blocks[1].color == BlockColor.RED
        assert network.blocks[2].color == BlockColor.BLUE
        assert network.blocks[3].color == BlockColor.YELLOW

        # Check wire colors (should inherit from source)
        wire_colors = [wire.color for wire in network.wires]
        assert BlockColor.RED in wire_colors
        assert BlockColor.BLUE in wire_colors
        assert BlockColor.YELLOW in wire_colors

        # Network should validate
        is_valid, errors = network.validate_network()
        assert is_valid

    def test_network_statistics(self):
        """Test network statistics collection"""
        network = create_simple_test_network()

        # Initial statistics
        stats = network.get_simulation_statistics()
        assert stats['total_timesteps'] == 0
        assert stats['total_firings'] == 0
        assert stats['network_size']['blocks'] == 3
        assert stats['network_size']['wires'] == 3

        # Run some simulation
        network.blocks[1].add_force(25.0)
        network.run_simulation(5)

        # Updated statistics
        stats = network.get_simulation_statistics()
        assert stats['total_timesteps'] == 5
        assert stats['total_firings'] >= 0  # May or may not have fired


class TestForceApplicationIntegration:
    """Test integration of all force application components"""

    def test_end_to_end_force_flow(self):
        """Test complete force flow from signal to block"""
        network = NeuralKlotskiNetwork()

        # Create simple 2-block network
        block1 = network.create_block(1, 50.0, BlockColor.RED, 10.0, initial_position=0.0)
        block2 = network.create_block(2, 100.0, BlockColor.BLUE, 10.0, initial_position=0.0)
        wire = network.create_wire(1, 1, 2, 3.0)

        # Manually trigger block 1 firing by setting it above threshold
        # We need to simulate crossing from below, so set previous position
        block1.position = 15.0  # Above threshold

        # For threshold crossing, we need to call step_dynamics which checks prev vs current
        # But step_dynamics updates position, so we need to trigger it differently
        # Let's manually trigger firing
        fired_blocks = {1}  # Manually say block 1 fired

        # Should create signal
        signals_created = network.create_signals_from_firings(fired_blocks)
        assert signals_created == 1

        # Advance time to signal arrival
        signal_arrival_time = network.signal_manager.signal_queue.peek_next_arrival_time()
        network.current_time = signal_arrival_time

        # Process arriving signal
        signals_processed, blocks_affected = network.process_signals()
        assert signals_processed == 1
        assert blocks_affected == 1

        # Block 2 should have received red wire force (clamped by max_damage=2.5)
        assert block2.total_force == 2.5  # Positive force from red wire, clamped from 3.0

    def test_multi_color_force_interaction(self):
        """Test interaction of different colored wire forces"""
        network = NeuralKlotskiNetwork()

        # Create blocks
        source_red = network.create_block(1, 50.0, BlockColor.RED, 10.0, initial_position=5.0)
        source_blue = network.create_block(2, 75.0, BlockColor.BLUE, 10.0, initial_position=-3.0)
        source_yellow = network.create_block(3, 100.0, BlockColor.YELLOW, 10.0, initial_position=8.0)
        target = network.create_block(4, 125.0, BlockColor.RED, 10.0, initial_position=2.0)

        # Create wires to same target
        wire_red = network.create_wire(1, 1, 4, 2.0)
        wire_blue = network.create_wire(2, 2, 4, 1.5)
        wire_yellow = network.create_wire(3, 3, 4, 1.0)

        # Manually add signals from all sources with wire strengths
        signals = [
            Signal(10.0, 2.0, BlockColor.RED, 1, 4),    # Red wire strength = 2.0
            Signal(10.0, 1.5, BlockColor.BLUE, 2, 4),   # Blue wire strength = 1.5
            Signal(10.0, 1.0, BlockColor.YELLOW, 3, 4)  # Yellow wire strength = 1.0
        ]

        for signal in signals:
            network.signal_manager.signal_queue.add_signal(signal)

        # Process signals
        network.current_time = 10.0
        signals_processed, blocks_affected = network.process_signals()

        # Should process all 3 signals affecting 1 block
        assert signals_processed == 3
        assert blocks_affected == 1

        # Calculate expected total force
        red_force = 2.0                           # +2.0 (rightward)
        blue_force = -1.5                         # -1.5 (leftward)
        yellow_force = 1.0 * (8.0 - 2.0)         # 1.0 * 6.0 = 6.0 (rightward)
        unclamped_total = red_force + blue_force + yellow_force  # 2.0 - 1.5 + 6.0 = 6.5

        # Force is clamped by max_damage parameter (2.5)
        expected_total = 2.5  # Clamped from 6.5 to 2.5

        assert target.total_force == expected_total


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])