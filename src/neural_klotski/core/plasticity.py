"""
Neural plasticity system for Neural-Klotski learning mechanisms.

Implements Hebbian learning, threshold adaptation, and STDP according to Section 9.4
of the specification. Provides wire strength updates and block threshold adaptation
based on firing patterns and dye concentrations.
"""

from typing import Dict, List, Tuple, Optional, Deque
from dataclasses import dataclass
from collections import deque
from enum import Enum
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from neural_klotski.core.block import BlockState, BlockColor
from neural_klotski.core.wire import Wire
from neural_klotski.core.dye import DyeSystem, DyeColor
from neural_klotski.config import LearningConfig
from neural_klotski.math_utils import clamp


class PlasticityType(Enum):
    """Types of plasticity mechanisms"""
    HEBBIAN = "hebbian"
    STDP = "stdp"
    THRESHOLD_ADAPTATION = "threshold_adaptation"


@dataclass
class FiringEvent:
    """Records a block firing event for plasticity calculations"""
    block_id: int
    firing_time: float
    block_position: float
    block_lag: float

    def __repr__(self) -> str:
        return f"FiringEvent(block={self.block_id}, time={self.firing_time:.1f})"


@dataclass
class PlasticityUpdate:
    """Represents a plasticity-based update to apply"""
    target_type: str  # "wire" or "block"
    target_id: int
    parameter: str  # "strength" or "threshold"
    delta_value: float
    learning_rule: PlasticityType

    def __repr__(self) -> str:
        return f"PlasticityUpdate({self.target_type}_{self.target_id}.{self.parameter} {self.delta_value:+.4f})"


class FiringHistoryTracker:
    """
    Tracks firing history for plasticity calculations.

    Maintains temporal windows of firing events for STDP, threshold adaptation,
    and eligibility trace calculations.
    """

    def __init__(self, window_size: float = 1000.0):
        """
        Initialize firing history tracker.

        Args:
            window_size: Maximum time window to retain events
        """
        self.window_size = window_size
        self.firing_events: Deque[FiringEvent] = deque()
        self.block_firing_counts: Dict[int, int] = {}
        self.last_cleanup_time = 0.0

    def add_firing_event(self, block_id: int, firing_time: float,
                        block_position: float, block_lag: float):
        """Add new firing event to history"""
        event = FiringEvent(block_id, firing_time, block_position, block_lag)
        self.firing_events.append(event)

        # Update firing count
        if block_id not in self.block_firing_counts:
            self.block_firing_counts[block_id] = 0
        self.block_firing_counts[block_id] += 1

    def cleanup_old_events(self, current_time: float):
        """Remove events older than window size"""
        cutoff_time = current_time - self.window_size

        # Remove old events from front
        while self.firing_events and self.firing_events[0].firing_time < cutoff_time:
            old_event = self.firing_events.popleft()
            # Decrement firing count
            if old_event.block_id in self.block_firing_counts:
                self.block_firing_counts[old_event.block_id] -= 1
                if self.block_firing_counts[old_event.block_id] <= 0:
                    del self.block_firing_counts[old_event.block_id]

        self.last_cleanup_time = current_time

    def get_recent_firings(self, block_id: int, current_time: float,
                          time_window: float) -> List[FiringEvent]:
        """Get recent firing events for specific block within time window"""
        cutoff_time = current_time - time_window
        return [event for event in self.firing_events
                if event.block_id == block_id and event.firing_time >= cutoff_time]

    def get_firing_rate(self, block_id: int, current_time: float,
                       measurement_window: float) -> float:
        """Calculate firing rate for block over measurement window"""
        recent_firings = self.get_recent_firings(block_id, current_time, measurement_window)
        return len(recent_firings) / measurement_window if measurement_window > 0 else 0.0

    def get_paired_events(self, source_block_id: int, target_block_id: int,
                         current_time: float, time_window: float) -> List[Tuple[FiringEvent, FiringEvent]]:
        """Get source-target firing pairs within time window for STDP"""
        cutoff_time = current_time - time_window

        source_events = [e for e in self.firing_events
                        if e.block_id == source_block_id and e.firing_time >= cutoff_time]
        target_events = [e for e in self.firing_events
                        if e.block_id == target_block_id and e.firing_time >= cutoff_time]

        # Find all pairs where source fires before target within window
        pairs = []
        for source_event in source_events:
            for target_event in target_events:
                time_diff = target_event.firing_time - source_event.firing_time
                if 0 < time_diff <= time_window:
                    pairs.append((source_event, target_event))

        return pairs

    def clear_history(self):
        """Clear all firing history"""
        self.firing_events.clear()
        self.block_firing_counts.clear()
        self.last_cleanup_time = 0.0


class HebbianLearningRule:
    """
    Implements Hebbian learning for wire strength adaptation.

    From Section 9.4.1: Δw = η_w × β × C_dye × (pre_fire × post_fire)
    where C_dye is local dye concentration enhancing plasticity.
    """

    def __init__(self, config: LearningConfig):
        self.config = config

    def calculate_strength_update(self, wire: Wire, source_fired: bool, target_fired: bool,
                                dye_concentration: float, dt: float) -> float:
        """
        Calculate Hebbian strength update for wire.

        Args:
            wire: Wire to update
            source_fired: Whether source block fired this timestep
            target_fired: Whether target block fired this timestep
            dye_concentration: Local dye concentration at wire
            dt: Timestep size

        Returns:
            Strength change (delta_w)
        """
        if not (source_fired and target_fired):
            return 0.0  # No update if both blocks didn't fire

        # Hebbian update: Δw = η_w × β × C_dye × dt
        delta_w = (self.config.wire_learning_rate *
                  self.config.dye_amplification *
                  dye_concentration * dt)

        return delta_w

    def calculate_anti_hebbian_update(self, wire: Wire, source_fired: bool, target_fired: bool,
                                    dye_concentration: float, dt: float) -> float:
        """
        Calculate anti-Hebbian update (weakening) when only one block fires.

        Args:
            wire: Wire to update
            source_fired: Whether source block fired
            target_fired: Whether target block fired
            dye_concentration: Local dye concentration
            dt: Timestep size

        Returns:
            Negative strength change for weakening
        """
        # Anti-Hebbian if exactly one fired
        if source_fired ^ target_fired:  # XOR - exactly one fired
            # Weaker anti-Hebbian effect
            delta_w = -(self.config.wire_learning_rate *
                       0.3 * self.config.dye_amplification *
                       dye_concentration * dt)
            return delta_w

        return 0.0


class STDPLearningRule:
    """
    Implements Spike-Timing Dependent Plasticity (STDP).

    From Section 9.4.2: Updates based on precise timing between
    pre- and post-synaptic firing events.
    """

    def __init__(self, config: LearningConfig):
        self.config = config

    def calculate_stdp_strength_update(self, wire: Wire, source_event: FiringEvent,
                                     target_event: FiringEvent, dye_concentration: float) -> float:
        """
        Calculate STDP-based strength update.

        Args:
            wire: Wire to update
            source_event: Source block firing event
            target_event: Target block firing event
            dye_concentration: Local dye concentration

        Returns:
            STDP strength change
        """
        # Calculate time difference (Δt = t_post - t_pre)
        delta_t = target_event.firing_time - source_event.firing_time

        if delta_t <= 0:
            return 0.0  # No update for simultaneous or reversed timing

        # STDP exponential decay: exp(-Δt/τ_STDP)
        stdp_kernel = np.exp(-delta_t / self.config.stdp_time_constant)

        # STDP update: Δw = η_w × β × C_dye × exp(-Δt/τ)
        delta_w = (self.config.wire_learning_rate *
                  self.config.dye_amplification *
                  dye_concentration * stdp_kernel)

        return delta_w


class ThresholdAdaptationRule:
    """
    Implements threshold adaptation based on firing rate homeostasis.

    From Section 9.4.3: Adapts block thresholds to maintain target firing rate.
    """

    def __init__(self, config: LearningConfig):
        self.config = config

    def calculate_threshold_update(self, block: BlockState, current_firing_rate: float,
                                 dt: float) -> float:
        """
        Calculate threshold adaptation update.

        Args:
            block: Block to adapt
            current_firing_rate: Current firing rate over measurement window
            dt: Timestep size

        Returns:
            Threshold change (delta_threshold)
        """
        # Error signal: difference from target rate
        rate_error = current_firing_rate - self.config.target_firing_rate

        # Threshold adaptation: Δτ = η_τ × (r_current - r_target) × dt
        # Higher firing rate -> increase threshold (reduce excitability)
        delta_threshold = self.config.threshold_learning_rate * rate_error * dt

        return delta_threshold


class PlasticityManager:
    """
    Central manager for all plasticity mechanisms.

    Coordinates Hebbian learning, STDP, and threshold adaptation across
    the entire network with proper temporal integration.
    """

    def __init__(self, config: LearningConfig):
        """
        Initialize plasticity manager.

        Args:
            config: Learning configuration parameters
        """
        self.config = config

        # Learning rules
        self.hebbian_rule = HebbianLearningRule(config)
        self.stdp_rule = STDPLearningRule(config)
        self.threshold_rule = ThresholdAdaptationRule(config)

        # Firing history tracking
        self.firing_history = FiringHistoryTracker(
            window_size=max(config.temporal_window, config.measurement_window)
        )

        # Statistics
        self.total_updates_applied = 0
        self.updates_by_type = {ptype: 0 for ptype in PlasticityType}

    def process_firing_events(self, fired_block_ids: List[int], current_time: float,
                            blocks: Dict[int, BlockState]):
        """
        Process new firing events and add to history.

        Args:
            fired_block_ids: List of blocks that fired this timestep
            current_time: Current simulation time
            blocks: Dictionary of block states
        """
        for block_id in fired_block_ids:
            block = blocks.get(block_id)
            if block is not None:
                self.firing_history.add_firing_event(
                    block_id, current_time, block.position, block.lag_position
                )

        # Cleanup old events periodically
        if current_time - self.firing_history.last_cleanup_time > 100.0:
            self.firing_history.cleanup_old_events(current_time)

    def calculate_hebbian_updates(self, current_time: float, blocks: Dict[int, BlockState],
                                wires: List[Wire], dye_system: DyeSystem,
                                dt: float) -> List[PlasticityUpdate]:
        """
        Calculate all Hebbian plasticity updates.

        Args:
            current_time: Current simulation time
            blocks: Dictionary of block states
            wires: List of network wires
            dye_system: Dye concentration system
            dt: Timestep size

        Returns:
            List of plasticity updates to apply
        """
        updates = []

        # Get recent firing events
        recent_firings = {}
        for block_id in blocks.keys():
            recent_firings[block_id] = len(self.firing_history.get_recent_firings(
                block_id, current_time, dt
            )) > 0

        for wire in wires:
            source_fired = recent_firings.get(wire.source_block_id, False)
            target_fired = recent_firings.get(wire.target_block_id, False)

            if not (source_fired or target_fired):
                continue  # No activity on this wire

            # Get dye concentration at wire spatial position
            wire_color = DyeColor.from_block_color(wire.color)
            dye_conc = dye_system.get_concentration(
                wire_color, wire.spatial_position[0], wire.spatial_position[1]
            )

            # Calculate Hebbian update
            if source_fired and target_fired:
                # Strengthening update
                delta_w = self.hebbian_rule.calculate_strength_update(
                    wire, source_fired, target_fired, dye_conc, dt
                )
            else:
                # Anti-Hebbian weakening
                delta_w = self.hebbian_rule.calculate_anti_hebbian_update(
                    wire, source_fired, target_fired, dye_conc, dt
                )

            if abs(delta_w) > 1e-8:  # Only create update if significant
                update = PlasticityUpdate(
                    target_type="wire",
                    target_id=wire.wire_id,
                    parameter="strength",
                    delta_value=delta_w,
                    learning_rule=PlasticityType.HEBBIAN
                )
                updates.append(update)

        return updates

    def calculate_stdp_updates(self, current_time: float, wires: List[Wire],
                             dye_system: DyeSystem) -> List[PlasticityUpdate]:
        """
        Calculate STDP-based plasticity updates.

        Args:
            current_time: Current simulation time
            wires: List of network wires
            dye_system: Dye concentration system

        Returns:
            List of STDP plasticity updates
        """
        updates = []

        for wire in wires:
            # Get paired firing events for this wire
            pairs = self.firing_history.get_paired_events(
                wire.source_block_id, wire.target_block_id,
                current_time, self.config.temporal_window
            )

            if not pairs:
                continue  # No paired events

            # Get dye concentration
            wire_color = DyeColor.from_block_color(wire.color)
            dye_conc = dye_system.get_concentration(
                wire_color, wire.spatial_position[0], wire.spatial_position[1]
            )

            # Calculate cumulative STDP update
            total_delta_w = 0.0
            for source_event, target_event in pairs:
                delta_w = self.stdp_rule.calculate_stdp_strength_update(
                    wire, source_event, target_event, dye_conc
                )
                total_delta_w += delta_w

            if abs(total_delta_w) > 1e-8:
                update = PlasticityUpdate(
                    target_type="wire",
                    target_id=wire.wire_id,
                    parameter="strength",
                    delta_value=total_delta_w,
                    learning_rule=PlasticityType.STDP
                )
                updates.append(update)

        return updates

    def calculate_threshold_updates(self, current_time: float, blocks: Dict[int, BlockState],
                                  dt: float) -> List[PlasticityUpdate]:
        """
        Calculate threshold adaptation updates.

        Args:
            current_time: Current simulation time
            blocks: Dictionary of block states
            dt: Timestep size

        Returns:
            List of threshold adaptation updates
        """
        updates = []

        for block_id, block in blocks.items():
            # Calculate current firing rate
            firing_rate = self.firing_history.get_firing_rate(
                block_id, current_time, self.config.measurement_window
            )

            # Calculate threshold adaptation
            delta_threshold = self.threshold_rule.calculate_threshold_update(
                block, firing_rate, dt
            )

            if abs(delta_threshold) > 1e-8:
                update = PlasticityUpdate(
                    target_type="block",
                    target_id=block_id,
                    parameter="threshold",
                    delta_value=delta_threshold,
                    learning_rule=PlasticityType.THRESHOLD_ADAPTATION
                )
                updates.append(update)

        return updates

    def apply_plasticity_updates(self, updates: List[PlasticityUpdate],
                               blocks: Dict[int, BlockState], wires: List[Wire],
                               strength_bounds: Tuple[float, float] = (0.1, 10.0),
                               threshold_bounds: Tuple[float, float] = (30.0, 80.0)) -> int:
        """
        Apply plasticity updates to blocks and wires.

        Args:
            updates: List of updates to apply
            blocks: Dictionary of block states
            wires: List of wires
            strength_bounds: (min, max) wire strength bounds
            threshold_bounds: (min, max) threshold bounds

        Returns:
            Number of updates successfully applied
        """
        updates_applied = 0
        wire_dict = {wire.wire_id: wire for wire in wires}

        for update in updates:
            try:
                if update.target_type == "wire":
                    wire = wire_dict.get(update.target_id)
                    if wire is not None and update.parameter == "strength":
                        # Apply wire strength update with bounds
                        new_strength = wire.base_strength + update.delta_value
                        wire.base_strength = clamp(new_strength, strength_bounds[0], strength_bounds[1])
                        updates_applied += 1

                elif update.target_type == "block":
                    block = blocks.get(update.target_id)
                    if block is not None and update.parameter == "threshold":
                        # Apply threshold update with bounds
                        new_threshold = block.threshold + update.delta_value
                        block.threshold = clamp(new_threshold, threshold_bounds[0], threshold_bounds[1])
                        updates_applied += 1

                # Update statistics
                self.updates_by_type[update.learning_rule] += 1

            except Exception as e:
                print(f"Warning: Failed to apply plasticity update {update}: {e}")
                continue

        self.total_updates_applied += updates_applied
        return updates_applied

    def execute_plasticity_timestep(self, fired_block_ids: List[int], current_time: float,
                                  blocks: Dict[int, BlockState], wires: List[Wire],
                                  dye_system: DyeSystem, dt: float) -> Dict[str, any]:
        """
        Execute complete plasticity processing for one timestep.

        Args:
            fired_block_ids: Blocks that fired this timestep
            current_time: Current simulation time
            blocks: Dictionary of block states
            wires: List of wires
            dye_system: Dye concentration system
            dt: Timestep size

        Returns:
            Dictionary of plasticity statistics
        """
        # Process new firing events
        self.process_firing_events(fired_block_ids, current_time, blocks)

        # Calculate all plasticity updates
        hebbian_updates = self.calculate_hebbian_updates(
            current_time, blocks, wires, dye_system, dt
        )
        stdp_updates = self.calculate_stdp_updates(current_time, wires, dye_system)
        threshold_updates = self.calculate_threshold_updates(current_time, blocks, dt)

        # Combine all updates
        all_updates = hebbian_updates + stdp_updates + threshold_updates

        # Apply updates
        updates_applied = self.apply_plasticity_updates(all_updates, blocks, wires)

        return {
            'hebbian_updates': len(hebbian_updates),
            'stdp_updates': len(stdp_updates),
            'threshold_updates': len(threshold_updates),
            'total_updates': len(all_updates),
            'updates_applied': updates_applied,
            'firing_events_added': len(fired_block_ids),
            'firing_history_size': len(self.firing_history.firing_events)
        }

    def get_plasticity_statistics(self) -> Dict[str, any]:
        """Get comprehensive plasticity statistics"""
        return {
            'total_updates_applied': self.total_updates_applied,
            'updates_by_type': self.updates_by_type.copy(),
            'firing_history_size': len(self.firing_history.firing_events),
            'tracked_blocks': len(self.firing_history.block_firing_counts),
            'learning_config': {
                'wire_learning_rate': self.config.wire_learning_rate,
                'dye_amplification': self.config.dye_amplification,
                'threshold_learning_rate': self.config.threshold_learning_rate,
                'target_firing_rate': self.config.target_firing_rate
            }
        }

    def reset_plasticity(self):
        """Reset all plasticity state"""
        self.firing_history.clear_history()
        self.total_updates_applied = 0
        self.updates_by_type = {ptype: 0 for ptype in PlasticityType}


if __name__ == "__main__":
    # Test plasticity system functionality
    from neural_klotski.config import get_default_config
    from neural_klotski.core.block import create_block
    from neural_klotski.core.wire import create_wire
    from neural_klotski.core.dye import create_dye_system_for_network

    print("Testing Neural-Klotski Plasticity System...")

    # Create test configuration
    config = get_default_config()
    plasticity = PlasticityManager(config.learning)

    # Create test blocks
    blocks = {
        1: create_block(1, 50.0, BlockColor.RED, 45.0),
        2: create_block(2, 100.0, BlockColor.BLUE, 50.0),
        3: create_block(3, 150.0, BlockColor.YELLOW, 55.0)
    }

    # Create test wires
    wires = [
        create_wire(1, 1, 2, 2.0, BlockColor.RED, (10.0, 75.0)),
        create_wire(2, 2, 3, 1.5, BlockColor.BLUE, (-5.0, 125.0))
    ]

    # Create dye system
    dye_system = create_dye_system_for_network((-50, 50), (0, 200), 2.0)

    # Inject some dye for testing
    dye_system.inject_dye(DyeColor.RED, 10.0, 75.0, 0.5)
    dye_system.inject_dye(DyeColor.BLUE, -5.0, 125.0, 0.3)

    print(f"Initial statistics: {plasticity.get_plasticity_statistics()}")

    # Simulate firing events and plasticity over multiple timesteps
    current_time = 0.0
    dt = 0.5

    for step in range(20):
        # Simulate some firing events
        fired_blocks = []
        if step % 3 == 0:
            fired_blocks.append(1)
        if step % 4 == 0:
            fired_blocks.append(2)
        if step % 7 == 0:
            fired_blocks.append(3)

        # Execute plasticity timestep
        stats = plasticity.execute_plasticity_timestep(
            fired_blocks, current_time, blocks, wires, dye_system, dt
        )

        if step % 5 == 0:
            print(f"Step {step}: {stats}")

        current_time += dt

    # Show final statistics
    final_stats = plasticity.get_plasticity_statistics()
    print(f"\nFinal plasticity statistics: {final_stats}")

    # Show updated wire strengths and block thresholds
    print(f"\nUpdated wire strengths:")
    for wire in wires:
        print(f"  Wire {wire.wire_id}: {wire.strength:.4f}")

    print(f"\nUpdated block thresholds:")
    for block_id, block in blocks.items():
        print(f"  Block {block_id}: {block.threshold:.4f}")

    print("\nPlasticity system test completed successfully!")