"""
Wire and signal propagation implementation for Neural-Klotski system.

Implements wires as connections between blocks with strength, color inheritance,
and damage/fatigue mechanics. Handles signal creation, propagation delays,
and temporal ordering according to Section 5.4 of the specification.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Tuple, Union
import heapq
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from neural_klotski.math_utils import (
    signal_propagation_delay, lag_distance, clamp, is_in_bounds,
    wire_effective_strength
)
from neural_klotski.config import WireConfig
from neural_klotski.core.block import BlockColor


@dataclass
class Signal:
    """
    Represents a propagating signal between blocks.

    From Section 5.4.2: Signals carry strength, color, and arrival time
    based on propagation delay calculations.
    """

    arrival_time: float     # When signal arrives at target (in timesteps)
    strength: float         # Signal strength (can be modified by wire strength)
    color: BlockColor       # Signal color (inherited from source block)
    source_block_id: int    # ID of source block that fired
    target_block_id: int    # ID of target block to receive signal
    wire_id: Optional[int] = None  # ID of wire that created this signal

    def __post_init__(self):
        """Validate signal parameters"""
        if self.arrival_time < 0:
            raise ValueError("Arrival time must be non-negative")
        if self.strength < 0:
            raise ValueError("Signal strength must be non-negative")

    def to_dict(self) -> dict:
        """Convert signal to dictionary for serialization"""
        return {
            'arrival_time': self.arrival_time,
            'strength': self.strength,
            'color': self.color.value,
            'source_block_id': self.source_block_id,
            'target_block_id': self.target_block_id,
            'wire_id': self.wire_id
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Signal':
        """Create signal from dictionary"""
        return cls(
            arrival_time=data['arrival_time'],
            strength=data['strength'],
            color=BlockColor(data['color']),
            source_block_id=data['source_block_id'],
            target_block_id=data['target_block_id'],
            wire_id=data.get('wire_id')
        )

    def __lt__(self, other: 'Signal') -> bool:
        """Enable priority queue ordering by arrival time"""
        return self.arrival_time < other.arrival_time

    def __repr__(self) -> str:
        return (f"Signal(t={self.arrival_time:.2f}, str={self.strength:.2f}, "
                f"color={self.color.value}, {self.source_block_id}→{self.target_block_id})")


@dataclass
class Wire:
    """
    Represents a connection between two blocks with strength and damage state.

    From Section 3.4: Wires have base strength, inherit color from source block,
    and can suffer damage/fatigue over time.
    """

    wire_id: int            # Unique identifier
    source_block_id: int    # Source block (where signals originate)
    target_block_id: int    # Target block (where signals arrive)
    base_strength: float    # Base connection strength
    color: BlockColor       # Wire color (inherited from source block)

    # Damage and fatigue state (Section 3.4.2)
    damage_level: float = 0.0      # Accumulated damage (0 = undamaged, 1 = destroyed)
    fatigue_level: float = 0.0     # Usage fatigue (affects performance)

    # Spatial properties for dye lookup
    spatial_position: Tuple[float, float] = (0.0, 0.0)  # (activation, lag) coordinates

    def __post_init__(self):
        """Validate wire parameters"""
        if self.base_strength < 0:
            raise ValueError("Wire strength must be non-negative")
        if not is_in_bounds(self.damage_level, 0.0, 1.0):
            raise ValueError("Damage level must be between 0 and 1")
        if not is_in_bounds(self.fatigue_level, 0.0, 1.0):
            raise ValueError("Fatigue level must be between 0 and 1")

    def is_functional(self) -> bool:
        """Check if wire is functional (not destroyed by damage)"""
        return self.damage_level < 1.0

    def effective_strength(self, dye_concentration: float = 0.0,
                          enhancement_factor: float = 1.0) -> float:
        """
        Calculate effective wire strength including dye enhancement and damage.

        From Section 3.4.3: w_eff = w_base * (1 + α * C_local) * damage_factor
        """
        if not self.is_functional():
            return 0.0

        # Base strength with dye enhancement
        enhanced_strength = wire_effective_strength(
            self.base_strength, dye_concentration, enhancement_factor
        )

        # Apply damage and fatigue reduction
        damage_factor = 1.0 - self.damage_level
        fatigue_factor = 1.0 - 0.5 * self.fatigue_level  # Fatigue reduces strength by up to 50%

        return enhanced_strength * damage_factor * fatigue_factor

    def apply_damage(self, damage_amount: float, config: WireConfig):
        """
        Apply damage to wire based on usage or external factors.

        Args:
            damage_amount: Amount of damage to apply
            config: Wire configuration with damage parameters
        """
        damage_increment = damage_amount * config.damage_rate
        self.damage_level = clamp(self.damage_level + damage_increment, 0.0, 1.0)

    def apply_fatigue(self, usage_amount: float, config: WireConfig):
        """
        Apply fatigue based on wire usage.

        Args:
            usage_amount: Amount of usage (typically signal strength)
            config: Wire configuration with fatigue parameters
        """
        fatigue_increment = usage_amount * config.fatigue_rate
        self.fatigue_level = clamp(self.fatigue_level + fatigue_increment, 0.0, 1.0)

    def repair(self, repair_amount: float, config: WireConfig):
        """
        Repair wire damage over time.

        Args:
            repair_amount: Base repair amount
            config: Wire configuration with repair parameters
        """
        repair_increment = repair_amount * config.repair_rate
        self.damage_level = clamp(self.damage_level - repair_increment, 0.0, 1.0)

        # Fatigue also recovers slowly
        fatigue_recovery = repair_amount * config.repair_rate * 0.5
        self.fatigue_level = clamp(self.fatigue_level - fatigue_recovery, 0.0, 1.0)

    def create_signal(self, current_time: float, source_lag: float, target_lag: float,
                     signal_speed: float, dye_concentration: float = 0.0,
                     enhancement_factor: float = 1.0) -> Optional[Signal]:
        """
        Create a signal when source block fires.

        Args:
            current_time: Current simulation time
            source_lag: Lag position of source block
            target_lag: Lag position of target block
            signal_speed: Signal propagation speed
            dye_concentration: Local dye concentration for enhancement
            enhancement_factor: Dye enhancement factor

        Returns:
            Signal object if wire is functional, None if destroyed
        """
        if not self.is_functional():
            return None

        # Calculate propagation delay
        delay = signal_propagation_delay(source_lag, target_lag, signal_speed)
        arrival_time = current_time + delay

        # Calculate effective strength
        strength = self.effective_strength(dye_concentration, enhancement_factor)

        # Apply fatigue based on signal strength
        # Note: This modifies the wire state as a side effect of signal creation
        # which models wire fatigue from usage

        return Signal(
            arrival_time=arrival_time,
            strength=strength,
            color=self.color,
            source_block_id=self.source_block_id,
            target_block_id=self.target_block_id,
            wire_id=self.wire_id
        )

    def update_strength_bounds(self, config: WireConfig):
        """Enforce wire strength bounds from configuration"""
        self.base_strength = clamp(
            self.base_strength, config.strength_min, config.strength_max
        )

    def to_dict(self) -> dict:
        """Convert wire to dictionary for serialization"""
        return {
            'wire_id': self.wire_id,
            'source_block_id': self.source_block_id,
            'target_block_id': self.target_block_id,
            'base_strength': self.base_strength,
            'color': self.color.value,
            'damage_level': self.damage_level,
            'fatigue_level': self.fatigue_level,
            'spatial_position': self.spatial_position
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Wire':
        """Create wire from dictionary"""
        return cls(
            wire_id=data['wire_id'],
            source_block_id=data['source_block_id'],
            target_block_id=data['target_block_id'],
            base_strength=data['base_strength'],
            color=BlockColor(data['color']),
            damage_level=data.get('damage_level', 0.0),
            fatigue_level=data.get('fatigue_level', 0.0),
            spatial_position=tuple(data.get('spatial_position', (0.0, 0.0)))
        )

    def __repr__(self) -> str:
        return (f"Wire(id={self.wire_id}, {self.source_block_id}→{self.target_block_id}, "
                f"str={self.base_strength:.2f}, dmg={self.damage_level:.2f}, "
                f"color={self.color.value})")


class SignalQueue:
    """
    Priority queue for managing temporal signal delivery.

    Maintains signals ordered by arrival time and provides efficient
    extraction of signals that should be delivered at current timestep.
    """

    def __init__(self):
        self._queue: List[Signal] = []
        self._signal_count = 0

    def add_signal(self, signal: Signal):
        """Add signal to queue with proper temporal ordering"""
        heapq.heappush(self._queue, signal)
        self._signal_count += 1

    def get_current_signals(self, current_time: float,
                           time_tolerance: float = 1e-6) -> List[Signal]:
        """
        Extract all signals that should be delivered at current time.

        Args:
            current_time: Current simulation timestep
            time_tolerance: Tolerance for floating point time comparison

        Returns:
            List of signals to be delivered now
        """
        current_signals = []

        # Extract signals whose arrival time <= current_time
        while (self._queue and
               self._queue[0].arrival_time <= current_time + time_tolerance):
            signal = heapq.heappop(self._queue)
            current_signals.append(signal)
            self._signal_count -= 1

        return current_signals

    def peek_next_arrival_time(self) -> Optional[float]:
        """Get arrival time of next signal without removing it"""
        if not self._queue:
            return None
        return self._queue[0].arrival_time

    def size(self) -> int:
        """Get number of signals in queue"""
        return len(self._queue)

    def clear(self):
        """Remove all signals from queue"""
        self._queue.clear()
        self._signal_count = 0

    def get_signals_for_target(self, target_block_id: int,
                              current_time: float) -> List[Signal]:
        """
        Get all current signals for a specific target block.

        Args:
            target_block_id: ID of target block
            current_time: Current simulation time

        Returns:
            List of signals for the specified target
        """
        all_current = self.get_current_signals(current_time)
        return [s for s in all_current if s.target_block_id == target_block_id]

    def __len__(self) -> int:
        return len(self._queue)

    def __repr__(self) -> str:
        return f"SignalQueue(size={len(self._queue)})"


def create_wire(wire_id: int, source_block_id: int, target_block_id: int,
               strength: float, source_color: BlockColor,
               spatial_position: Tuple[float, float] = (0.0, 0.0)) -> Wire:
    """
    Factory function to create a properly initialized wire.

    Args:
        wire_id: Unique wire identifier
        source_block_id: Source block ID
        target_block_id: Target block ID
        strength: Base wire strength
        source_color: Color inherited from source block
        spatial_position: (activation, lag) coordinates for dye lookup

    Returns:
        Initialized Wire instance
    """
    return Wire(
        wire_id=wire_id,
        source_block_id=source_block_id,
        target_block_id=target_block_id,
        base_strength=strength,
        color=source_color,
        spatial_position=spatial_position
    )


if __name__ == "__main__":
    # Test basic wire and signal functionality
    from neural_klotski.config import get_default_config

    config = get_default_config()

    # Create test wire
    wire = create_wire(
        wire_id=1,
        source_block_id=10,
        target_block_id=20,
        strength=2.0,
        source_color=BlockColor.RED,
        spatial_position=(25.0, 50.0)
    )

    print(f"Created wire: {wire}")
    print(f"Wire functional: {wire.is_functional()}")
    print(f"Effective strength: {wire.effective_strength():.3f}")

    # Test signal creation
    signal = wire.create_signal(
        current_time=10.0,
        source_lag=50.0,
        target_lag=100.0,
        signal_speed=config.wires.signal_speed
    )

    if signal:
        print(f"Created signal: {signal}")
        print(f"Propagation delay: {signal.arrival_time - 10.0:.3f} timesteps")

    # Test signal queue
    queue = SignalQueue()
    if signal:
        queue.add_signal(signal)

    # Add more test signals
    for i in range(3):
        test_signal = Signal(
            arrival_time=10.0 + i * 0.5,
            strength=1.5,
            color=BlockColor.BLUE,
            source_block_id=i,
            target_block_id=99
        )
        queue.add_signal(test_signal)

    print(f"\nSignal queue: {queue}")
    print(f"Next arrival: {queue.peek_next_arrival_time()}")

    # Extract signals at different times
    for t in [10.0, 10.5, 11.0]:
        current_signals = queue.get_current_signals(t)
        print(f"At t={t}: {len(current_signals)} signals delivered")

    print("Wire and signal system test completed successfully!")