"""
Unit tests for Wire and Signal systems in Neural-Klotski.

Tests all aspects of wire behavior including:
- Wire creation, strength, and damage mechanics
- Signal propagation and delay calculations
- Signal queue temporal ordering
- Color inheritance and effective strength calculation
- Serialization and validation
"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from neural_klotski.core.wire import (
    Wire, Signal, SignalQueue, create_wire
)
from neural_klotski.core.block import BlockColor
from neural_klotski.config import get_default_config, WireConfig


class TestSignal:
    """Test Signal class functionality"""

    def test_signal_creation(self):
        """Test basic signal creation"""
        signal = Signal(
            arrival_time=15.5,
            strength=2.3,
            color=BlockColor.RED,
            source_block_id=1,
            target_block_id=2,
            wire_id=5
        )

        assert signal.arrival_time == 15.5
        assert signal.strength == 2.3
        assert signal.color == BlockColor.RED
        assert signal.source_block_id == 1
        assert signal.target_block_id == 2
        assert signal.wire_id == 5

    def test_signal_validation(self):
        """Test signal parameter validation"""
        # Negative arrival time
        with pytest.raises(ValueError, match="Arrival time must be non-negative"):
            Signal(
                arrival_time=-1.0,
                strength=1.0,
                color=BlockColor.RED,
                source_block_id=1,
                target_block_id=2
            )

        # Negative strength
        with pytest.raises(ValueError, match="Signal strength must be non-negative"):
            Signal(
                arrival_time=10.0,
                strength=-0.5,
                color=BlockColor.RED,
                source_block_id=1,
                target_block_id=2
            )

    def test_signal_ordering(self):
        """Test signal ordering for priority queue"""
        signal1 = Signal(10.0, 1.0, BlockColor.RED, 1, 2)
        signal2 = Signal(12.0, 1.0, BlockColor.BLUE, 2, 3)
        signal3 = Signal(8.0, 1.0, BlockColor.YELLOW, 3, 4)

        signals = [signal1, signal2, signal3]
        signals.sort()

        # Should be ordered by arrival time
        assert signals[0].arrival_time == 8.0
        assert signals[1].arrival_time == 10.0
        assert signals[2].arrival_time == 12.0

    def test_signal_serialization(self):
        """Test signal to/from dictionary conversion"""
        original = Signal(
            arrival_time=25.3,
            strength=1.8,
            color=BlockColor.BLUE,
            source_block_id=42,
            target_block_id=84,
            wire_id=123
        )

        # Convert to dict and back
        data = original.to_dict()
        restored = Signal.from_dict(data)

        assert restored.arrival_time == original.arrival_time
        assert restored.strength == original.strength
        assert restored.color == original.color
        assert restored.source_block_id == original.source_block_id
        assert restored.target_block_id == original.target_block_id
        assert restored.wire_id == original.wire_id


class TestWire:
    """Test Wire class functionality"""

    def test_wire_creation(self):
        """Test basic wire creation"""
        wire = Wire(
            wire_id=10,
            source_block_id=1,
            target_block_id=5,
            base_strength=2.5,
            color=BlockColor.RED,
            damage_level=0.1,
            fatigue_level=0.05,
            spatial_position=(25.0, 50.0)
        )

        assert wire.wire_id == 10
        assert wire.source_block_id == 1
        assert wire.target_block_id == 5
        assert wire.base_strength == 2.5
        assert wire.color == BlockColor.RED
        assert wire.damage_level == 0.1
        assert wire.fatigue_level == 0.05
        assert wire.spatial_position == (25.0, 50.0)

    def test_wire_factory_function(self):
        """Test create_wire factory function"""
        wire = create_wire(
            wire_id=1,
            source_block_id=10,
            target_block_id=20,
            strength=1.5,
            source_color=BlockColor.BLUE,
            spatial_position=(30.0, 75.0)
        )

        assert wire.wire_id == 1
        assert wire.source_block_id == 10
        assert wire.target_block_id == 20
        assert wire.base_strength == 1.5
        assert wire.color == BlockColor.BLUE
        assert wire.damage_level == 0.0  # Default
        assert wire.fatigue_level == 0.0  # Default
        assert wire.spatial_position == (30.0, 75.0)

    def test_wire_validation(self):
        """Test wire parameter validation"""
        # Negative strength
        with pytest.raises(ValueError, match="Wire strength must be non-negative"):
            Wire(1, 1, 2, -1.0, BlockColor.RED)

        # Invalid damage level
        with pytest.raises(ValueError, match="Damage level must be between 0 and 1"):
            Wire(1, 1, 2, 1.0, BlockColor.RED, damage_level=1.5)

        # Invalid fatigue level
        with pytest.raises(ValueError, match="Fatigue level must be between 0 and 1"):
            Wire(1, 1, 2, 1.0, BlockColor.RED, fatigue_level=-0.1)

    def test_wire_functionality(self):
        """Test wire functional status"""
        wire = create_wire(1, 1, 2, 2.0, BlockColor.RED)

        # Initially functional
        assert wire.is_functional()

        # Damaged but still functional
        wire.damage_level = 0.8
        assert wire.is_functional()

        # Completely damaged
        wire.damage_level = 1.0
        assert not wire.is_functional()

    def test_effective_strength_calculation(self):
        """Test effective strength with damage and enhancement"""
        wire = create_wire(1, 1, 2, 2.0, BlockColor.RED)

        # Base strength with no damage or enhancement
        assert wire.effective_strength() == 2.0

        # With dye enhancement
        strength_enhanced = wire.effective_strength(dye_concentration=0.5, enhancement_factor=2.0)
        expected = 2.0 * (1.0 + 2.0 * 0.5)  # 2.0 * 2.0 = 4.0
        assert strength_enhanced == expected

        # With damage
        wire.damage_level = 0.3  # 30% damage
        strength_damaged = wire.effective_strength()
        expected_damaged = 2.0 * (1.0 - 0.3)  # 2.0 * 0.7 = 1.4
        assert strength_damaged == expected_damaged

        # With fatigue
        wire.fatigue_level = 0.4  # 40% fatigue
        strength_fatigued = wire.effective_strength()
        fatigue_factor = 1.0 - 0.5 * 0.4  # 1.0 - 0.2 = 0.8
        expected_fatigued = 2.0 * (1.0 - 0.3) * fatigue_factor  # 1.4 * 0.8 = 1.12
        assert abs(strength_fatigued - expected_fatigued) < 1e-10

        # Completely damaged wire
        wire.damage_level = 1.0
        assert wire.effective_strength() == 0.0

    def test_damage_mechanics(self):
        """Test damage application and repair"""
        config = WireConfig(damage_rate=0.1, repair_rate=0.05)
        wire = create_wire(1, 1, 2, 2.0, BlockColor.RED)

        # Apply damage
        wire.apply_damage(0.5, config)
        expected_damage = 0.5 * config.damage_rate  # 0.5 * 0.1 = 0.05
        assert wire.damage_level == expected_damage

        # Apply more damage
        wire.apply_damage(1.0, config)
        expected_damage += 1.0 * config.damage_rate  # 0.05 + 0.1 = 0.15
        assert wire.damage_level == expected_damage

        # Repair damage
        wire.repair(2.0, config)
        repair_amount = 2.0 * config.repair_rate  # 2.0 * 0.05 = 0.1
        expected_damage -= repair_amount  # 0.15 - 0.1 = 0.05
        assert abs(wire.damage_level - expected_damage) < 1e-10

        # Damage should not go below 0
        wire.repair(10.0, config)
        assert wire.damage_level == 0.0

    def test_fatigue_mechanics(self):
        """Test fatigue application and recovery"""
        config = WireConfig(fatigue_rate=0.02, repair_rate=0.05)
        wire = create_wire(1, 1, 2, 2.0, BlockColor.RED)

        # Apply fatigue
        wire.apply_fatigue(2.0, config)
        expected_fatigue = 2.0 * config.fatigue_rate  # 2.0 * 0.02 = 0.04
        assert wire.fatigue_level == expected_fatigue

        # Fatigue recovery through repair
        initial_fatigue = wire.fatigue_level
        wire.repair(1.0, config)
        recovery = 1.0 * config.repair_rate * 0.5  # 1.0 * 0.05 * 0.5 = 0.025
        expected_fatigue = initial_fatigue - recovery  # 0.04 - 0.025 = 0.015
        assert abs(wire.fatigue_level - expected_fatigue) < 1e-10

    def test_signal_creation(self):
        """Test signal creation from wire firing"""
        wire = create_wire(1, 10, 20, 2.5, BlockColor.RED)

        signal = wire.create_signal(
            current_time=15.0,
            source_lag=50.0,
            target_lag=100.0,
            signal_speed=100.0
        )

        assert signal is not None
        assert signal.source_block_id == 10
        assert signal.target_block_id == 20
        assert signal.color == BlockColor.RED
        assert signal.strength == 2.5
        assert signal.wire_id == 1

        # Check propagation delay calculation
        expected_delay = abs(100.0 - 50.0) / 100.0  # distance=50, speed=100, delay=0.5
        expected_arrival = 15.0 + expected_delay
        assert signal.arrival_time == expected_arrival

    def test_signal_creation_destroyed_wire(self):
        """Test that destroyed wires don't create signals"""
        wire = create_wire(1, 10, 20, 2.5, BlockColor.RED)
        wire.damage_level = 1.0  # Completely damaged

        signal = wire.create_signal(
            current_time=10.0,
            source_lag=50.0,
            target_lag=100.0,
            signal_speed=100.0
        )

        assert signal is None

    def test_strength_bounds_enforcement(self):
        """Test wire strength bounds checking"""
        config = WireConfig(strength_min=0.5, strength_max=5.0)
        wire = create_wire(1, 1, 2, 10.0, BlockColor.RED)  # Above max

        wire.update_strength_bounds(config)
        assert wire.base_strength == config.strength_max

        wire.base_strength = 0.1  # Below min
        wire.update_strength_bounds(config)
        assert wire.base_strength == config.strength_min

    def test_wire_serialization(self):
        """Test wire serialization and deserialization"""
        original = Wire(
            wire_id=42,
            source_block_id=10,
            target_block_id=20,
            base_strength=2.3,
            color=BlockColor.YELLOW,
            damage_level=0.25,
            fatigue_level=0.15,
            spatial_position=(45.0, 80.0)
        )

        # Convert to dict and back
        data = original.to_dict()
        restored = Wire.from_dict(data)

        assert restored.wire_id == original.wire_id
        assert restored.source_block_id == original.source_block_id
        assert restored.target_block_id == original.target_block_id
        assert restored.base_strength == original.base_strength
        assert restored.color == original.color
        assert restored.damage_level == original.damage_level
        assert restored.fatigue_level == original.fatigue_level
        assert restored.spatial_position == original.spatial_position


class TestSignalQueue:
    """Test SignalQueue temporal ordering system"""

    def test_queue_creation(self):
        """Test basic queue creation and properties"""
        queue = SignalQueue()
        assert queue.size() == 0
        assert len(queue) == 0
        assert queue.peek_next_arrival_time() is None

    def test_signal_addition_and_ordering(self):
        """Test adding signals and temporal ordering"""
        queue = SignalQueue()

        # Add signals in random order
        signals = [
            Signal(15.0, 1.0, BlockColor.RED, 1, 2),
            Signal(10.0, 1.0, BlockColor.BLUE, 2, 3),
            Signal(20.0, 1.0, BlockColor.YELLOW, 3, 4),
            Signal(12.0, 1.0, BlockColor.RED, 4, 5)
        ]

        for signal in signals:
            queue.add_signal(signal)

        assert queue.size() == 4
        assert queue.peek_next_arrival_time() == 10.0  # Earliest time

    def test_current_signal_extraction(self):
        """Test extracting signals for current timestep"""
        queue = SignalQueue()

        # Add signals with different arrival times
        queue.add_signal(Signal(10.0, 1.0, BlockColor.RED, 1, 2))
        queue.add_signal(Signal(10.0, 1.5, BlockColor.BLUE, 2, 3))  # Same time
        queue.add_signal(Signal(12.0, 2.0, BlockColor.YELLOW, 3, 4))
        queue.add_signal(Signal(15.0, 0.5, BlockColor.RED, 4, 5))

        # Extract at t=10.0
        current_signals = queue.get_current_signals(10.0)
        assert len(current_signals) == 2
        assert queue.size() == 2  # Two signals remaining

        # Check that correct signals were extracted
        arrival_times = [s.arrival_time for s in current_signals]
        assert all(t == 10.0 for t in arrival_times)

        # Extract at t=12.0
        current_signals = queue.get_current_signals(12.0)
        assert len(current_signals) == 1
        assert current_signals[0].arrival_time == 12.0
        assert queue.size() == 1

        # Extract at t=20.0 (should get remaining signal)
        current_signals = queue.get_current_signals(20.0)
        assert len(current_signals) == 1
        assert current_signals[0].arrival_time == 15.0
        assert queue.size() == 0

    def test_time_tolerance(self):
        """Test floating point time tolerance in signal extraction"""
        queue = SignalQueue()

        # Add signal with floating point arrival time
        queue.add_signal(Signal(10.000001, 1.0, BlockColor.RED, 1, 2))

        # Should extract with small tolerance
        current_signals = queue.get_current_signals(10.0)
        assert len(current_signals) == 1

    def test_signals_for_target(self):
        """Test extracting signals for specific target block"""
        queue = SignalQueue()

        # Add signals for different targets at same time
        queue.add_signal(Signal(10.0, 1.0, BlockColor.RED, 1, 5))    # Target 5
        queue.add_signal(Signal(10.0, 1.5, BlockColor.BLUE, 2, 5))   # Target 5
        queue.add_signal(Signal(10.0, 2.0, BlockColor.YELLOW, 3, 6)) # Target 6

        # Get signals for target 5
        target_signals = queue.get_signals_for_target(5, 10.0)
        assert len(target_signals) == 2
        assert all(s.target_block_id == 5 for s in target_signals)

        # Queue should now be empty (all signals at t=10.0 were extracted)
        assert queue.size() == 0

    def test_queue_clear(self):
        """Test clearing queue"""
        queue = SignalQueue()

        # Add multiple signals
        for i in range(5):
            queue.add_signal(Signal(10.0 + i, 1.0, BlockColor.RED, i, i+1))

        assert queue.size() == 5

        queue.clear()
        assert queue.size() == 0
        assert queue.peek_next_arrival_time() is None

    def test_empty_queue_operations(self):
        """Test operations on empty queue"""
        queue = SignalQueue()

        # Should handle empty queue gracefully
        current_signals = queue.get_current_signals(10.0)
        assert len(current_signals) == 0

        target_signals = queue.get_signals_for_target(1, 10.0)
        assert len(target_signals) == 0


class TestWireSignalIntegration:
    """Test integration between wires and signals"""

    def test_end_to_end_signal_flow(self):
        """Test complete signal creation and delivery flow"""
        # Create wire
        wire = create_wire(1, 10, 20, 2.0, BlockColor.RED)

        # Create signal queue
        queue = SignalQueue()

        # Wire creates signal when block fires
        signal = wire.create_signal(
            current_time=5.0,
            source_lag=25.0,
            target_lag=75.0,
            signal_speed=100.0
        )

        # Add signal to queue
        queue.add_signal(signal)

        # Signal should arrive later based on propagation delay
        expected_delay = abs(75.0 - 25.0) / 100.0  # distance=50, speed=100, delay=0.5
        expected_arrival = 5.0 + expected_delay

        assert queue.peek_next_arrival_time() == expected_arrival

        # Extract signal at arrival time
        delivered_signals = queue.get_current_signals(expected_arrival)
        assert len(delivered_signals) == 1

        delivered_signal = delivered_signals[0]
        assert delivered_signal.source_block_id == 10
        assert delivered_signal.target_block_id == 20
        assert delivered_signal.strength == 2.0
        assert delivered_signal.color == BlockColor.RED

    def test_multiple_wire_coordination(self):
        """Test multiple wires creating signals simultaneously"""
        wires = [
            create_wire(1, 1, 10, 1.5, BlockColor.RED),
            create_wire(2, 2, 10, 2.0, BlockColor.BLUE),
            create_wire(3, 3, 10, 1.0, BlockColor.YELLOW)
        ]

        queue = SignalQueue()
        current_time = 10.0

        # All wires fire and create signals to same target
        for i, wire in enumerate(wires):
            signal = wire.create_signal(
                current_time=current_time,
                source_lag=50.0,  # Same source lag
                target_lag=100.0,  # Same target lag
                signal_speed=100.0
            )
            queue.add_signal(signal)

        # All signals should arrive at same time
        expected_arrival = current_time + 0.5  # delay = 50/100 = 0.5
        target_signals = queue.get_signals_for_target(10, expected_arrival)

        assert len(target_signals) == 3
        assert sum(s.strength for s in target_signals) == 4.5  # 1.5 + 2.0 + 1.0

    def test_wire_damage_affects_signals(self):
        """Test that wire damage affects signal strength"""
        wire = create_wire(1, 1, 2, 3.0, BlockColor.RED)

        # Undamaged signal
        signal1 = wire.create_signal(10.0, 50.0, 100.0, 100.0)
        assert signal1.strength == 3.0

        # Apply damage
        wire.damage_level = 0.5  # 50% damage

        # Damaged signal should have reduced strength
        signal2 = wire.create_signal(15.0, 50.0, 100.0, 100.0)
        expected_strength = 3.0 * (1.0 - 0.5)  # 3.0 * 0.5 = 1.5
        assert signal2.strength == expected_strength


class TestSpecificationCompliance:
    """Test compliance with Neural-Klotski specification"""

    def test_propagation_delay_equations(self):
        """Test that delay calculations match Section 5.4.2"""
        wire = create_wire(1, 1, 2, 2.0, BlockColor.RED)

        # Test various lag distances and speeds
        test_cases = [
            (50.0, 100.0, 100.0, 0.5),    # distance=50, speed=100, delay=0.5
            (25.0, 125.0, 200.0, 0.5),    # distance=100, speed=200, delay=0.5
            (0.0, 75.0, 150.0, 0.5),      # distance=75, speed=150, delay=0.5
        ]

        for source_lag, target_lag, speed, expected_delay in test_cases:
            signal = wire.create_signal(10.0, source_lag, target_lag, speed)
            actual_delay = signal.arrival_time - 10.0
            assert abs(actual_delay - expected_delay) < 1e-10

    def test_wire_strength_bounds(self):
        """Test wire strength bounds match specification"""
        config = get_default_config().wires

        # Test bounds enforcement
        wire = create_wire(1, 1, 2, 100.0, BlockColor.RED)  # Way above max
        wire.update_strength_bounds(config)
        assert wire.base_strength <= config.strength_max

        wire.base_strength = -1.0  # Below min
        wire.update_strength_bounds(config)
        assert wire.base_strength >= config.strength_min

    def test_signal_queue_temporal_ordering(self):
        """Test that signal queue maintains strict temporal ordering"""
        queue = SignalQueue()

        # Add many signals in random order
        import random
        arrival_times = [random.uniform(0, 100) for _ in range(50)]

        for i, t in enumerate(arrival_times):
            signal = Signal(t, 1.0, BlockColor.RED, i, i+1)
            queue.add_signal(signal)

        # Extract all signals - should come out in temporal order
        extracted_times = []
        for t in sorted(arrival_times):
            current_signals = queue.get_current_signals(t)
            extracted_times.extend(s.arrival_time for s in current_signals)

        # Should be in sorted order
        assert extracted_times == sorted(extracted_times)


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])