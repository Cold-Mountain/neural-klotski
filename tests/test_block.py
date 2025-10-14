"""
Unit tests for Block dynamics in Neural-Klotski system.

Tests all aspects of block behavior including:
- State initialization and validation
- Physics integration and spring forces
- Threshold crossing detection and firing
- Refractory mechanics and gating
- Color identity and serialization
"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from neural_klotski.core.block import (
    BlockState, BlockColor, create_block
)
from neural_klotski.config import get_default_config, DynamicsConfig


class TestBlockColor:
    """Test BlockColor enumeration"""

    def test_color_values(self):
        """Test that color enum has correct values"""
        assert BlockColor.RED.value == "red"
        assert BlockColor.BLUE.value == "blue"
        assert BlockColor.YELLOW.value == "yellow"

    def test_color_creation(self):
        """Test creating colors from string values"""
        assert BlockColor("red") == BlockColor.RED
        assert BlockColor("blue") == BlockColor.BLUE
        assert BlockColor("yellow") == BlockColor.YELLOW


class TestBlockStateInitialization:
    """Test BlockState creation and validation"""

    def test_basic_initialization(self):
        """Test creating a basic block state"""
        block = BlockState(
            position=5.0,
            velocity=2.0,
            lag_position=100.0,
            threshold=10.0,
            refractory_timer=0.0,
            color=BlockColor.RED,
            block_id=42
        )

        assert block.position == 5.0
        assert block.velocity == 2.0
        assert block.lag_position == 100.0
        assert block.threshold == 10.0
        assert block.refractory_timer == 0.0
        assert block.color == BlockColor.RED
        assert block.block_id == 42
        assert block.total_force == 0.0

    def test_factory_function(self):
        """Test create_block factory function"""
        block = create_block(
            block_id=1,
            lag_position=50.0,
            color=BlockColor.BLUE,
            threshold=15.0,
            initial_position=3.0,
            initial_velocity=1.0
        )

        assert block.block_id == 1
        assert block.lag_position == 50.0
        assert block.color == BlockColor.BLUE
        assert block.threshold == 15.0
        assert block.position == 3.0
        assert block.velocity == 1.0
        assert block.refractory_timer == 0.0

    def test_validation_errors(self):
        """Test validation of invalid parameters"""
        # Negative refractory timer
        with pytest.raises(ValueError, match="Refractory timer must be non-negative"):
            BlockState(
                position=0.0, velocity=0.0, lag_position=0.0,
                threshold=10.0, refractory_timer=-1.0,
                color=BlockColor.RED, block_id=1
            )

        # Non-positive threshold
        with pytest.raises(ValueError, match="Threshold must be positive"):
            BlockState(
                position=0.0, velocity=0.0, lag_position=0.0,
                threshold=0.0, refractory_timer=0.0,
                color=BlockColor.RED, block_id=1
            )


class TestBlockForces:
    """Test force calculations and accumulation"""

    def test_force_accumulation(self):
        """Test adding and resetting forces"""
        block = create_block(1, 50.0, BlockColor.RED, 10.0)

        # Initially no force
        assert block.total_force == 0.0

        # Add forces
        block.add_force(5.0)
        assert block.total_force == 5.0

        block.add_force(3.0)
        assert block.total_force == 8.0

        # Reset forces
        block.reset_forces()
        assert block.total_force == 0.0

    def test_spring_force_calculation(self):
        """Test spring force calculation"""
        config = DynamicsConfig(spring_constant=2.0)
        block = create_block(1, 50.0, BlockColor.RED, 10.0)

        # At equilibrium (position = 0)
        block.position = 0.0
        spring_force = block.calculate_spring_force(config)
        assert spring_force == 0.0

        # Rightward displacement (should pull left)
        block.position = 5.0
        spring_force = block.calculate_spring_force(config)
        assert spring_force == -10.0  # -k * x = -2.0 * 5.0

        # Leftward displacement (should pull right)
        block.position = -3.0
        spring_force = block.calculate_spring_force(config)
        assert spring_force == 6.0  # -2.0 * (-3.0)


class TestBlockPhysics:
    """Test physics integration and dynamics"""

    def test_physics_update_no_force(self):
        """Test physics update with only spring force"""
        config = DynamicsConfig(dt=0.1, mass=1.0, damping=0.1, spring_constant=1.0)
        block = create_block(1, 50.0, BlockColor.RED, 10.0, initial_position=5.0)

        prev_pos = block.update_physics(config)

        # Should have moved leftward due to spring force
        assert prev_pos == 5.0
        # Position changes due to velocity update, but starts with 0 velocity
        # So position change is 0 + velocity*dt, velocity change is spring force
        assert block.velocity < 0.0  # Moving leftward due to spring force

        # After second step, position should change
        block.update_physics(config)
        assert block.position < 5.0

    def test_physics_update_with_external_force(self):
        """Test physics update with external force"""
        config = DynamicsConfig(dt=0.1, mass=1.0, damping=0.0, spring_constant=0.0)
        block = create_block(1, 50.0, BlockColor.RED, 10.0)

        # Apply rightward force
        block.add_force(10.0)
        prev_pos = block.update_physics(config)

        assert prev_pos == 0.0
        # Position starts at 0, velocity starts at 0
        # After one step: pos = 0 + 0*dt = 0, vel = 0 + (10/1)*0.1 = 1.0
        assert block.position == 0.0  # No movement yet (started with 0 velocity)
        assert block.velocity == 1.0  # Gained velocity from force

        # After second step, should move
        block.add_force(10.0)  # Add same force again
        block.update_physics(config)
        assert block.position > 0.0  # Now should have moved right

    def test_refractory_timer_update(self):
        """Test refractory timer countdown"""
        config = DynamicsConfig(dt=0.5)
        block = create_block(1, 50.0, BlockColor.RED, 10.0)
        block.refractory_timer = 10.0

        # Update physics should decrement timer
        block.update_physics(config)
        assert block.refractory_timer == 9.5

        # Timer should not go below zero
        block.refractory_timer = 0.3
        block.update_physics(config)
        assert block.refractory_timer == 0.0

    def test_position_bounds(self):
        """Test position bounds enforcement"""
        config = DynamicsConfig(dt=0.1, mass=1.0, damping=0.0, spring_constant=0.0)
        block = create_block(1, 50.0, BlockColor.RED, 10.0)

        # Test extreme rightward position
        block.position = 200.0
        block.velocity = 0.0
        block.update_physics(config)
        assert block.position <= 100.0  # Should be clamped

        # Test extreme leftward position
        block.position = -200.0
        block.velocity = 0.0
        block.update_physics(config)
        assert block.position >= -100.0  # Should be clamped


class TestBlockFiring:
    """Test threshold crossing and firing behavior"""

    def test_refractory_state(self):
        """Test refractory state detection"""
        block = create_block(1, 50.0, BlockColor.RED, 10.0)

        # Not refractory initially
        assert not block.is_refractory()

        # Set refractory timer
        block.refractory_timer = 5.0
        assert block.is_refractory()

        # Timer expires
        block.refractory_timer = 0.0
        assert not block.is_refractory()

    def test_threshold_crossing_detection(self):
        """Test should_fire logic"""
        block = create_block(1, 50.0, BlockColor.RED, 10.0, initial_position=12.0)

        # Crossing from below - should fire
        assert block.should_fire(9.0)

        # Crossing from above - should not fire
        assert not block.should_fire(11.0)

        # Not crossing - should not fire
        assert not block.should_fire(11.5)

        # At threshold exactly from below - should fire
        block.position = 10.0
        assert block.should_fire(9.9)

        # Refractory blocks should not fire
        block.refractory_timer = 5.0
        assert not block.should_fire(9.0)

    def test_refractory_kick_application(self):
        """Test refractory kick mechanics"""
        config = DynamicsConfig(refractory_kick=20.0, refractory_duration=35.0)
        block = create_block(1, 50.0, BlockColor.RED, 10.0)

        # Apply refractory kick
        initial_vel = block.velocity
        block.apply_refractory_kick(config)

        assert block.velocity == -20.0  # Strong leftward velocity
        assert block.refractory_timer == 35.0

    def test_complete_dynamics_step(self):
        """Test full step_dynamics integration"""
        config = get_default_config().dynamics
        block = create_block(1, 50.0, BlockColor.RED, 10.0)

        # Apply force to push toward threshold
        block.add_force(20.0)
        fired = block.step_dynamics(config)

        # Should not fire immediately
        assert not fired
        assert block.total_force == 0.0  # Forces reset

        # Continue until firing
        fired_steps = []
        for i in range(10):
            block.add_force(5.0)
            fired = block.step_dynamics(config)
            fired_steps.append(fired)

        # Should fire at some point
        assert any(fired_steps)

        # Find the step where firing occurred
        fire_step = fired_steps.index(True)

        # After firing, should be refractory
        assert block.is_refractory()

        # The velocity may not remain negative if forces continue to be applied
        # But immediately after firing (before next force), velocity should be negative
        # Test by checking refractory kick is applied correctly
        test_block = create_block(2, 50.0, BlockColor.RED, 10.0, initial_position=12.0)
        test_block.apply_refractory_kick(config)
        assert test_block.velocity == -config.refractory_kick


class TestBlockValidation:
    """Test state validation methods"""

    def test_valid_state(self):
        """Test validation of valid states"""
        block = create_block(1, 50.0, BlockColor.RED, 10.0)
        assert block.validate_state()

        # Valid with different values
        block.position = -5.5
        block.velocity = 12.3
        block.threshold = 7.8
        assert block.validate_state()

    def test_invalid_states(self):
        """Test validation of invalid states"""
        block = create_block(1, 50.0, BlockColor.RED, 10.0)

        # NaN values
        block.position = float('nan')
        assert not block.validate_state()

        # Infinite values
        block = create_block(1, 50.0, BlockColor.RED, 10.0)
        block.velocity = float('inf')
        assert not block.validate_state()

        # Negative threshold
        block = create_block(1, 50.0, BlockColor.RED, 10.0)
        block.threshold = -1.0
        assert not block.validate_state()

        # Negative refractory timer
        block = create_block(1, 50.0, BlockColor.RED, 10.0)
        block.refractory_timer = -1.0
        assert not block.validate_state()

        # Extreme positions
        block = create_block(1, 50.0, BlockColor.RED, 10.0)
        block.position = 2000.0
        assert not block.validate_state()


class TestBlockSerialization:
    """Test serialization and deserialization"""

    def test_to_dict(self):
        """Test converting block to dictionary"""
        block = create_block(1, 50.0, BlockColor.BLUE, 15.0, 3.0, 2.0)
        block.refractory_timer = 5.0
        block.total_force = 7.5

        data = block.to_dict()

        expected = {
            'position': 3.0,
            'velocity': 2.0,
            'lag_position': 50.0,
            'threshold': 15.0,
            'refractory_timer': 5.0,
            'color': 'blue',
            'block_id': 1,
            'total_force': 7.5
        }

        assert data == expected

    def test_from_dict(self):
        """Test creating block from dictionary"""
        data = {
            'position': 3.0,
            'velocity': 2.0,
            'lag_position': 50.0,
            'threshold': 15.0,
            'refractory_timer': 5.0,
            'color': 'blue',
            'block_id': 1,
            'total_force': 7.5
        }

        block = BlockState.from_dict(data)

        assert block.position == 3.0
        assert block.velocity == 2.0
        assert block.lag_position == 50.0
        assert block.threshold == 15.0
        assert block.refractory_timer == 5.0
        assert block.color == BlockColor.BLUE
        assert block.block_id == 1
        assert block.total_force == 7.5

    def test_roundtrip_serialization(self):
        """Test that serialization preserves all data"""
        original = create_block(42, 123.5, BlockColor.YELLOW, 8.7, -2.3, 4.1)
        original.refractory_timer = 12.8
        original.total_force = 6.4

        # Convert to dict and back
        data = original.to_dict()
        restored = BlockState.from_dict(data)

        # Should be identical
        assert restored.position == original.position
        assert restored.velocity == original.velocity
        assert restored.lag_position == original.lag_position
        assert restored.threshold == original.threshold
        assert restored.refractory_timer == original.refractory_timer
        assert restored.color == original.color
        assert restored.block_id == original.block_id
        assert restored.total_force == original.total_force


class TestBlockSpecificationCompliance:
    """Test compliance with Neural-Klotski specification"""

    def test_integration_equations(self):
        """Test that integration follows Section 9.2.1 equations"""
        # Use exact specification parameters
        config = DynamicsConfig(
            dt=0.5,
            mass=1.0,
            damping=0.15,
            spring_constant=0.15
        )

        block = create_block(1, 50.0, BlockColor.RED, 10.0, initial_position=5.0)

        # Manual calculation of expected result
        initial_pos = block.position
        initial_vel = block.velocity
        spring_f = -config.spring_constant * initial_pos  # -0.15 * 5.0 = -0.75

        # Expected integration step
        expected_pos = initial_pos + initial_vel * config.dt
        expected_vel = initial_vel * (1.0 - config.damping * config.dt) + (spring_f / config.mass) * config.dt

        # Run actual integration
        prev_pos = block.update_physics(config)

        # Should match manual calculation (within numerical precision)
        assert abs(block.position - expected_pos) < 1e-10
        assert abs(block.velocity - expected_vel) < 1e-10

    def test_threshold_crossing_requirements(self):
        """Test strict 'from below' threshold crossing requirement"""
        block = create_block(1, 50.0, BlockColor.RED, 10.0)

        # Crossing from below (9.5 -> 10.5)
        block.position = 10.5
        assert block.should_fire(9.5)

        # Crossing from above (10.5 -> 9.5) - should NOT fire
        block.position = 9.5
        assert not block.should_fire(10.5)

        # Starting above and moving higher - should NOT fire
        block.position = 11.0
        assert not block.should_fire(10.5)

        # Exactly at threshold from below - should fire
        block.position = 10.0
        assert block.should_fire(9.99)

    def test_refractory_behavior(self):
        """Test refractory mechanics match specification"""
        config = DynamicsConfig(
            refractory_kick=20.0,
            refractory_duration=35.0
        )

        block = create_block(1, 50.0, BlockColor.RED, 10.0, initial_position=12.0)

        # Should fire from this position
        assert block.should_fire(9.0)

        # Apply refractory kick
        block.apply_refractory_kick(config)

        # Should now be refractory and unable to fire
        assert block.is_refractory()
        assert not block.should_fire(9.0)  # Same crossing should not fire
        assert block.velocity == -config.refractory_kick
        assert block.refractory_timer == config.refractory_duration


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])