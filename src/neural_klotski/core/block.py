"""
Block dynamics implementation for Neural-Klotski system.

Implements the core BlockState class representing neurons as sliding blocks
with position, velocity, threshold detection, and refractory mechanics
according to Section 9.2 of the specification.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from neural_klotski.math_utils import (
    euler_integration_step_damped, spring_force, clamp,
    threshold_crossing_from_below, is_in_bounds
)
from neural_klotski.config import DynamicsConfig


class BlockColor(Enum):
    """Block color identity determining wire behavior"""
    RED = "red"       # Excitatory blocks (rightward force)
    BLUE = "blue"     # Inhibitory blocks (leftward force)
    YELLOW = "yellow" # Electrical coupling blocks (position-dependent force)


@dataclass
class BlockState:
    """
    Represents the complete state of a single block in the Neural-Klotski system.

    From Section 9.2.1: Each block has position and velocity on the activation axis,
    a firing threshold, and refractory state variables.
    """

    # Spatial properties
    position: float         # Position on activation axis (membrane potential analog)
    velocity: float         # Velocity on activation axis
    lag_position: float     # Fixed position on lag axis (temporal phase)

    # Neural properties
    threshold: float        # Firing threshold value
    refractory_timer: float # Remaining refractory period (0 = not refractory)

    # Identity
    color: BlockColor       # Block color (determines wire behavior)
    block_id: int          # Unique identifier

    # Accumulated forces (reset each timestep)
    total_force: float = 0.0

    def __post_init__(self):
        """Validate initial state parameters"""
        if self.refractory_timer < 0:
            raise ValueError("Refractory timer must be non-negative")
        if self.threshold <= 0:
            raise ValueError("Threshold must be positive")

    def reset_forces(self):
        """Reset accumulated forces for new timestep"""
        self.total_force = 0.0

    def add_force(self, force: float):
        """Add external force to total accumulated force"""
        self.total_force += force

    def calculate_spring_force(self, config: DynamicsConfig) -> float:
        """
        Calculate spring restoring force toward equilibrium.

        From Section 9.2.1: F_spring = -k * x
        Spring pulls block toward position 0 (equilibrium)
        """
        return spring_force(self.position, 0.0, config.spring_constant)

    def is_refractory(self) -> bool:
        """Check if block is in refractory period"""
        return self.refractory_timer > 0

    def should_fire(self, prev_position: float) -> bool:
        """
        Determine if block should fire based on threshold crossing.

        Critical requirement from Section 9.2.2: Block must cross threshold
        from below AND not be in refractory period.
        """
        if self.is_refractory():
            return False

        return threshold_crossing_from_below(prev_position, self.position, self.threshold)

    def apply_refractory_kick(self, config: DynamicsConfig):
        """
        Apply leftward refractory kick when block fires.

        From Section 9.2.2: When block fires, apply strong leftward velocity
        and set refractory timer.
        """
        self.velocity = -config.refractory_kick  # Strong leftward velocity
        self.refractory_timer = config.refractory_duration

    def update_physics(self, config: DynamicsConfig, dt: Optional[float] = None) -> float:
        """
        Update block position and velocity using Euler integration.

        Returns the previous position for firing detection.

        From Section 9.2.1:
        - Total force includes spring force + external forces
        - Integration includes damping
        - Position bounds are enforced as soft constraints
        """
        if dt is None:
            dt = config.dt

        # Store previous position for firing detection
        prev_position = self.position

        # Calculate total force including spring restoring force
        spring_f = self.calculate_spring_force(config)
        total_f = spring_f + self.total_force

        # Update position and velocity with damping
        self.position, self.velocity = euler_integration_step_damped(
            self.position, self.velocity, total_f,
            config.mass, config.damping, dt
        )

        # Apply soft position bounds (clamp to reasonable range)
        # Allow some excursion beyond normal range but prevent runaway
        max_excursion = 100.0  # Allow significant excursion for dynamics
        self.position = clamp(self.position, -max_excursion, max_excursion)

        # Update refractory timer
        if self.refractory_timer > 0:
            self.refractory_timer = max(0.0, self.refractory_timer - dt)

        return prev_position

    def step_dynamics(self, config: DynamicsConfig, dt: Optional[float] = None) -> bool:
        """
        Perform complete dynamics step including firing detection.

        Returns True if block fired this timestep.

        This is the main integration point combining:
        1. Physics update (position/velocity)
        2. Firing detection
        3. Refractory kick application
        """
        # Update physics and get previous position
        prev_position = self.update_physics(config, dt)

        # Check for firing
        fired = self.should_fire(prev_position)

        # Apply refractory kick if fired
        if fired:
            self.apply_refractory_kick(config)

        # Reset forces for next timestep
        self.reset_forces()

        return fired

    def validate_state(self) -> bool:
        """
        Validate current block state for consistency.

        Returns True if state is valid, False otherwise.
        """
        try:
            # Check for NaN or infinite values
            if not all(isinstance(x, (int, float)) and not (x != x or abs(x) == float('inf'))
                      for x in [self.position, self.velocity, self.threshold, self.refractory_timer]):
                return False

            # Check basic constraints
            if self.threshold <= 0:
                return False
            if self.refractory_timer < 0:
                return False

            # Check reasonable bounds
            if not is_in_bounds(self.position, -1000.0, 1000.0):
                return False
            if not is_in_bounds(self.velocity, -1000.0, 1000.0):
                return False

            return True

        except (TypeError, ValueError):
            return False

    def to_dict(self) -> dict:
        """Convert block state to dictionary for serialization"""
        return {
            'position': self.position,
            'velocity': self.velocity,
            'lag_position': self.lag_position,
            'threshold': self.threshold,
            'refractory_timer': self.refractory_timer,
            'color': self.color.value,
            'block_id': self.block_id,
            'total_force': self.total_force
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'BlockState':
        """Create block state from dictionary"""
        return cls(
            position=data['position'],
            velocity=data['velocity'],
            lag_position=data['lag_position'],
            threshold=data['threshold'],
            refractory_timer=data['refractory_timer'],
            color=BlockColor(data['color']),
            block_id=data['block_id'],
            total_force=data.get('total_force', 0.0)
        )

    def __repr__(self) -> str:
        return (f"BlockState(id={self.block_id}, pos={self.position:.3f}, "
                f"vel={self.velocity:.3f}, lag={self.lag_position:.1f}, "
                f"thresh={self.threshold:.3f}, refr={self.refractory_timer:.1f}, "
                f"color={self.color.value})")


def create_block(block_id: int, lag_position: float, color: BlockColor,
                threshold: float, initial_position: float = 0.0,
                initial_velocity: float = 0.0) -> BlockState:
    """
    Factory function to create a properly initialized block.

    Args:
        block_id: Unique identifier for the block
        lag_position: Fixed position on lag axis
        color: Block color identity
        threshold: Firing threshold
        initial_position: Starting position on activation axis
        initial_velocity: Starting velocity

    Returns:
        Initialized BlockState instance
    """
    return BlockState(
        position=initial_position,
        velocity=initial_velocity,
        lag_position=lag_position,
        threshold=threshold,
        refractory_timer=0.0,
        color=color,
        block_id=block_id
    )


if __name__ == "__main__":
    # Test basic block functionality
    from neural_klotski.config import get_default_config

    config = get_default_config()

    # Create test block
    block = create_block(
        block_id=1,
        lag_position=50.0,
        color=BlockColor.RED,
        threshold=10.0,
        initial_position=0.0,
        initial_velocity=0.0
    )

    print(f"Initial state: {block}")
    print(f"Valid state: {block.validate_state()}")

    # Test dynamics step
    block.add_force(15.0)  # External force
    fired = block.step_dynamics(config.dynamics)
    print(f"After step (force=15): {block}")
    print(f"Fired: {fired}")

    # Test multiple steps
    for i in range(5):
        fired = block.step_dynamics(config.dynamics)
        print(f"Step {i+2}: pos={block.position:.3f}, vel={block.velocity:.3f}, "
              f"refr={block.refractory_timer:.1f}, fired={fired}")

    print("Block dynamics test completed successfully!")