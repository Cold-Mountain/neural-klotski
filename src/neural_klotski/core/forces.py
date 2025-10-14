"""
Force application system for Neural-Klotski wire signals.

Implements the conversion of arriving signals into forces applied to target blocks
according to wire color types (Section 4.2 of the specification):
- Red wires: Excitatory rightward force
- Blue wires: Inhibitory leftward force
- Yellow wires: Electrical coupling (position-dependent force)
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from neural_klotski.core.block import BlockState, BlockColor
from neural_klotski.core.wire import Signal, SignalQueue
from neural_klotski.math_utils import clamp


@dataclass
class ForceApplication:
    """
    Represents a force to be applied to a target block.

    Contains the force magnitude and the block ID that should receive it.
    """
    target_block_id: int
    force_magnitude: float
    source_signal: Signal  # Reference to originating signal for debugging

    def __repr__(self) -> str:
        return f"ForceApplication(target={self.target_block_id}, force={self.force_magnitude:.3f})"


class ForceCalculator:
    """
    Calculates forces from wire signals according to color-specific rules.

    From Section 4.2:
    - Red wires: Apply rightward excitatory force F = +w_eff
    - Blue wires: Apply leftward inhibitory force F = -w_eff
    - Yellow wires: Apply electrical coupling force F = w_eff × (x_source - x_target)
    """

    @staticmethod
    def calculate_red_wire_force(signal: Signal) -> float:
        """
        Calculate excitatory force from red wire signal.

        From Section 4.2.1: Red wires provide rightward excitatory push
        Force = +signal_strength (positive = rightward)

        Args:
            signal: Incoming signal from red wire

        Returns:
            Positive force magnitude (rightward push)
        """
        return signal.strength

    @staticmethod
    def calculate_blue_wire_force(signal: Signal) -> float:
        """
        Calculate inhibitory force from blue wire signal.

        From Section 4.2.2: Blue wires provide leftward inhibitory push
        Force = -signal_strength (negative = leftward)

        Args:
            signal: Incoming signal from blue wire

        Returns:
            Negative force magnitude (leftward push)
        """
        return -signal.strength

    @staticmethod
    def calculate_yellow_wire_force(signal: Signal, source_position: float,
                                   target_position: float) -> float:
        """
        Calculate electrical coupling force from yellow wire signal.

        From Section 4.2.3: Yellow wires provide position-dependent coupling
        Force = signal_strength × (x_source - x_target)
        - If source > target: rightward force on target (toward source)
        - If source < target: leftward force on target (toward source)

        Args:
            signal: Incoming signal from yellow wire
            source_position: Position of source block on activation axis
            target_position: Position of target block on activation axis

        Returns:
            Force magnitude proportional to position difference
        """
        position_difference = source_position - target_position
        return signal.strength * position_difference

    @classmethod
    def calculate_signal_force(cls, signal: Signal, source_position: Optional[float] = None,
                             target_position: Optional[float] = None) -> float:
        """
        Calculate force from signal based on wire color.

        Args:
            signal: Signal to convert to force
            source_position: Source block position (required for yellow wires)
            target_position: Target block position (required for yellow wires)

        Returns:
            Force magnitude to apply to target block

        Raises:
            ValueError: If yellow wire signal lacks position information
        """
        if signal.color == BlockColor.RED:
            return cls.calculate_red_wire_force(signal)
        elif signal.color == BlockColor.BLUE:
            return cls.calculate_blue_wire_force(signal)
        elif signal.color == BlockColor.YELLOW:
            if source_position is None or target_position is None:
                raise ValueError("Yellow wire force calculation requires source and target positions")
            return cls.calculate_yellow_wire_force(signal, source_position, target_position)
        else:
            raise ValueError(f"Unknown wire color: {signal.color}")


class SignalProcessor:
    """
    Processes signals from signal queue and converts them to force applications.

    Manages the complete signal-to-force pipeline including:
    - Extracting current signals from queue
    - Converting signals to forces based on wire color
    - Grouping forces by target block
    - Applying forces to target blocks
    """

    def __init__(self):
        self.force_calculator = ForceCalculator()

    def process_signals_for_timestep(self, signal_queue: SignalQueue,
                                   current_time: float,
                                   blocks: Dict[int, BlockState]) -> List[ForceApplication]:
        """
        Process all signals arriving at current timestep and convert to force applications.

        Args:
            signal_queue: Queue containing pending signals
            current_time: Current simulation timestep
            blocks: Dictionary mapping block IDs to BlockState objects

        Returns:
            List of force applications to be applied to blocks
        """
        # Extract all signals arriving at current time
        current_signals = signal_queue.get_current_signals(current_time)

        # Convert each signal to force application
        force_applications = []

        for signal in current_signals:
            try:
                # Get block positions for yellow wire calculations
                source_pos = None
                target_pos = None

                if signal.color == BlockColor.YELLOW:
                    source_block = blocks.get(signal.source_block_id)
                    target_block = blocks.get(signal.target_block_id)

                    if source_block is None:
                        continue  # Skip if source block not found
                    if target_block is None:
                        continue  # Skip if target block not found

                    source_pos = source_block.position
                    target_pos = target_block.position

                # Calculate force
                force_magnitude = self.force_calculator.calculate_signal_force(
                    signal, source_pos, target_pos
                )

                # Create force application
                force_app = ForceApplication(
                    target_block_id=signal.target_block_id,
                    force_magnitude=force_magnitude,
                    source_signal=signal
                )

                force_applications.append(force_app)

            except (ValueError, KeyError) as e:
                # Log error but continue processing other signals
                print(f"Warning: Failed to process signal {signal}: {e}")
                continue

        return force_applications

    def group_forces_by_target(self, force_applications: List[ForceApplication]) -> Dict[int, float]:
        """
        Group force applications by target block and sum forces.

        Args:
            force_applications: List of individual force applications

        Returns:
            Dictionary mapping target block IDs to total force magnitude
        """
        force_totals = {}

        for force_app in force_applications:
            target_id = force_app.target_block_id
            if target_id not in force_totals:
                force_totals[target_id] = 0.0
            force_totals[target_id] += force_app.force_magnitude

        return force_totals

    def apply_forces_to_blocks(self, force_totals: Dict[int, float],
                             blocks: Dict[int, BlockState],
                             max_force_magnitude: Optional[float] = None) -> int:
        """
        Apply accumulated forces to target blocks.

        Args:
            force_totals: Dictionary mapping block IDs to total force
            blocks: Dictionary mapping block IDs to BlockState objects
            max_force_magnitude: Optional maximum force magnitude for safety

        Returns:
            Number of blocks that received forces
        """
        blocks_affected = 0

        for block_id, total_force in force_totals.items():
            block = blocks.get(block_id)
            if block is None:
                continue  # Skip if block not found

            # Apply force magnitude limits if specified
            if max_force_magnitude is not None:
                total_force = clamp(total_force, -max_force_magnitude, max_force_magnitude)

            # Add force to block's accumulator
            block.add_force(total_force)
            blocks_affected += 1

        return blocks_affected

    def process_timestep(self, signal_queue: SignalQueue, current_time: float,
                        blocks: Dict[int, BlockState],
                        max_force_magnitude: Optional[float] = None) -> Tuple[int, int]:
        """
        Complete signal processing pipeline for one timestep.

        Args:
            signal_queue: Signal queue to process
            current_time: Current simulation time
            blocks: Block dictionary for force application
            max_force_magnitude: Optional force magnitude limit

        Returns:
            Tuple of (signals_processed, blocks_affected)
        """
        # Process signals to force applications
        force_applications = self.process_signals_for_timestep(signal_queue, current_time, blocks)

        # Group and sum forces by target
        force_totals = self.group_forces_by_target(force_applications)

        # Apply forces to blocks
        blocks_affected = self.apply_forces_to_blocks(force_totals, blocks, max_force_magnitude)

        return len(force_applications), blocks_affected


class NetworkSignalManager:
    """
    High-level manager for signal propagation and force application across the network.

    Coordinates signal creation from firing blocks, signal propagation through wires,
    and force application to target blocks for complete network dynamics.
    """

    def __init__(self):
        self.signal_processor = SignalProcessor()
        self.signal_queue = SignalQueue()

    def add_firing_signals(self, fired_block_ids: List[int], current_time: float,
                         blocks: Dict[int, BlockState], wires: List,
                         dye_concentrations: Optional[Dict[int, float]] = None,
                         enhancement_factor: float = 1.0) -> int:
        """
        Create signals from fired blocks and add them to the signal queue.

        Args:
            fired_block_ids: List of block IDs that fired this timestep
            current_time: Current simulation time
            blocks: Dictionary of block states
            wires: List of wires in the network
            dye_concentrations: Optional dye concentrations for enhancement
            enhancement_factor: Dye enhancement factor

        Returns:
            Number of signals created and queued
        """
        signals_created = 0

        if dye_concentrations is None:
            dye_concentrations = {}

        for wire in wires:
            # Check if this wire's source block fired
            if wire.source_block_id not in fired_block_ids:
                continue

            # Get source and target blocks
            source_block = blocks.get(wire.source_block_id)
            target_block = blocks.get(wire.target_block_id)

            if source_block is None or target_block is None:
                continue  # Skip if blocks not found

            # Get dye concentration for this wire
            dye_conc = dye_concentrations.get(wire.wire_id, 0.0)

            # Create signal from wire
            signal = wire.create_signal(
                current_time=current_time,
                source_lag=source_block.lag_position,
                target_lag=target_block.lag_position,
                signal_speed=100.0,  # TODO: Get from config
                dye_concentration=dye_conc,
                enhancement_factor=enhancement_factor
            )

            if signal is not None:
                self.signal_queue.add_signal(signal)
                signals_created += 1

        return signals_created

    def process_network_timestep(self, current_time: float, blocks: Dict[int, BlockState],
                                max_force_magnitude: Optional[float] = None) -> Tuple[int, int]:
        """
        Process all signals for current timestep and apply forces to blocks.

        Args:
            current_time: Current simulation time
            blocks: Dictionary of block states
            max_force_magnitude: Optional force magnitude limit

        Returns:
            Tuple of (signals_processed, blocks_affected)
        """
        return self.signal_processor.process_timestep(
            self.signal_queue, current_time, blocks, max_force_magnitude
        )

    def get_queue_status(self) -> Dict[str, any]:
        """Get status information about signal queue"""
        return {
            'queue_size': len(self.signal_queue),
            'next_arrival': self.signal_queue.peek_next_arrival_time()
        }

    def clear_signals(self):
        """Clear all pending signals"""
        self.signal_queue.clear()


if __name__ == "__main__":
    # Test force calculation system
    from neural_klotski.core.block import create_block
    from neural_klotski.core.wire import Signal, create_wire

    print("Testing Force Application System...")

    # Create test blocks
    blocks = {
        1: create_block(1, 50.0, BlockColor.RED, 10.0, initial_position=5.0),
        2: create_block(2, 100.0, BlockColor.BLUE, 10.0, initial_position=-2.0),
        3: create_block(3, 150.0, BlockColor.YELLOW, 10.0, initial_position=3.0)
    }

    # Test force calculations
    calculator = ForceCalculator()

    # Red wire force (excitatory)
    red_signal = Signal(10.0, 2.5, BlockColor.RED, 1, 2)
    red_force = calculator.calculate_red_wire_force(red_signal)
    print(f"Red wire force: {red_force} (expected: 2.5)")

    # Blue wire force (inhibitory)
    blue_signal = Signal(10.0, 1.8, BlockColor.BLUE, 2, 3)
    blue_force = calculator.calculate_blue_wire_force(blue_signal)
    print(f"Blue wire force: {blue_force} (expected: -1.8)")

    # Yellow wire force (electrical coupling)
    yellow_signal = Signal(10.0, 1.5, BlockColor.YELLOW, 1, 3)
    yellow_force = calculator.calculate_yellow_wire_force(yellow_signal, 5.0, 3.0)
    expected_yellow = 1.5 * (5.0 - 3.0)  # 1.5 * 2.0 = 3.0
    print(f"Yellow wire force: {yellow_force} (expected: {expected_yellow})")

    # Test signal processor
    processor = SignalProcessor()

    # Create signal queue with test signals
    queue = SignalQueue()
    queue.add_signal(red_signal)
    queue.add_signal(blue_signal)
    queue.add_signal(yellow_signal)

    # Process signals
    force_apps = processor.process_signals_for_timestep(queue, 10.0, blocks)
    print(f"\nProcessed {len(force_apps)} signals into force applications:")
    for app in force_apps:
        print(f"  {app}")

    # Group forces
    force_totals = processor.group_forces_by_target(force_apps)
    print(f"\nForce totals by target: {force_totals}")

    # Apply forces to blocks
    blocks_affected = processor.apply_forces_to_blocks(force_totals, blocks)
    print(f"Applied forces to {blocks_affected} blocks")

    # Show block forces
    for block_id, block in blocks.items():
        print(f"Block {block_id} total force: {block.total_force}")

    print("\nForce application system test completed successfully!")