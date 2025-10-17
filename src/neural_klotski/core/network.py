"""
Complete Neural-Klotski network integration system.

Combines blocks, wires, signals, and forces into a unified simulation framework
that can execute complete network dynamics with proper temporal ordering.
"""

from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from neural_klotski.core.block import BlockState, BlockColor, create_block
from neural_klotski.core.wire import Wire, Signal, SignalQueue, create_wire
from neural_klotski.core.forces import NetworkSignalManager, SignalProcessor
from neural_klotski.core.dye import DyeSystem, create_dye_system_for_network
from neural_klotski.core.plasticity import PlasticityManager
from neural_klotski.core.learning import IntegratedLearningSystem, TrialResult, TrialOutcome
from neural_klotski.config import get_default_config, NeuralKlotskiConfig
from neural_klotski.math_utils import clamp


@dataclass
class NetworkState:
    """
    Complete state of the Neural-Klotski network at a given time.

    Contains all blocks, wires, and pending signals with full temporal information.
    """
    current_time: float
    blocks: Dict[int, BlockState]
    wires: List[Wire]
    signal_queue: SignalQueue
    dye_concentrations: Dict[int, float]  # Wire ID -> dye concentration
    dye_system: Optional[DyeSystem] = None
    plasticity_stats: Optional[Dict[str, any]] = None
    learning_stats: Optional[Dict[str, any]] = None

    def get_block_count(self) -> int:
        """Get total number of blocks"""
        return len(self.blocks)

    def get_wire_count(self) -> int:
        """Get total number of wires"""
        return len(self.wires)

    def get_pending_signal_count(self) -> int:
        """Get number of pending signals"""
        return len(self.signal_queue)

    def get_network_summary(self) -> Dict[str, any]:
        """Get summary statistics of network state"""
        return {
            'time': self.current_time,
            'blocks': self.get_block_count(),
            'wires': self.get_wire_count(),
            'pending_signals': self.get_pending_signal_count(),
            'next_signal_arrival': self.signal_queue.peek_next_arrival_time()
        }


class NeuralKlotskiNetwork:
    """
    Complete Neural-Klotski network simulation system.

    Integrates all components for full network dynamics:
    - Block physics and firing
    - Wire signal propagation
    - Force application from signals
    - Dye diffusion and decay
    - Plasticity and learning
    - Temporal coordination and timestep execution
    """

    def __init__(self, config: Optional[NeuralKlotskiConfig] = None,
                 enable_dye_system: bool = True, enable_plasticity: bool = True,
                 enable_learning: bool = True):
        """
        Initialize network with configuration.

        Args:
            config: Neural-Klotski configuration (uses default if None)
            enable_dye_system: Whether to enable dye diffusion system
            enable_plasticity: Whether to enable plasticity mechanisms
            enable_learning: Whether to enable integrated learning system
        """
        self.config = config if config is not None else get_default_config()

        # Network state
        self.current_time = 0.0
        self.blocks: Dict[int, BlockState] = {}
        self.wires: List[Wire] = []
        self.dye_concentrations: Dict[int, float] = {}

        # Signal management
        self.signal_manager = NetworkSignalManager()

        # Dye system (optional)
        self.dye_system: Optional[DyeSystem] = None
        self.enable_dye_system = enable_dye_system

        # Plasticity system (optional)
        self.plasticity_manager: Optional[PlasticityManager] = None
        self.enable_plasticity = enable_plasticity
        if enable_plasticity:
            self.plasticity_manager = PlasticityManager(self.config.learning)

        # Integrated learning system (optional)
        self.learning_system: Optional[IntegratedLearningSystem] = None
        self.enable_learning = enable_learning
        if enable_learning:
            self.learning_system = IntegratedLearningSystem(self.config.learning, self.config.dyes)

        # Trial management
        self.in_trial = False
        self.trial_start_time = 0.0

        # Statistics
        self.total_timesteps = 0
        self.total_firings = 0
        self.total_signals_created = 0
        self.total_forces_applied = 0
        self.total_plasticity_updates = 0
        self.total_learning_signals = 0

    def add_block(self, block: BlockState):
        """Add block to network"""
        self.blocks[block.block_id] = block

    def add_wire(self, wire: Wire):
        """Add wire to network"""
        self.wires.append(wire)
        # Initialize dye concentration for this wire
        if wire.wire_id not in self.dye_concentrations:
            self.dye_concentrations[wire.wire_id] = 0.0

    def create_block(self, block_id: int, lag_position: float, color: BlockColor,
                    threshold: float, initial_position: float = 0.0,
                    initial_velocity: float = 0.0) -> BlockState:
        """
        Create and add block to network.

        Args:
            block_id: Unique block identifier
            lag_position: Position on lag axis
            color: Block color
            threshold: Firing threshold
            initial_position: Starting position on activation axis
            initial_velocity: Starting velocity

        Returns:
            Created block
        """
        block = create_block(block_id, lag_position, color, threshold,
                           initial_position, initial_velocity)
        self.add_block(block)
        return block

    def create_wire(self, wire_id: int, source_block_id: int, target_block_id: int,
                   strength: float, spatial_position: Tuple[float, float] = (0.0, 0.0)) -> Wire:
        """
        Create and add wire to network.

        Args:
            wire_id: Unique wire identifier
            source_block_id: Source block ID
            target_block_id: Target block ID
            strength: Wire strength
            spatial_position: (activation, lag) coordinates for dye lookup

        Returns:
            Created wire
        """
        # Get source block color for wire inheritance
        source_block = self.blocks.get(source_block_id)
        if source_block is None:
            raise ValueError(f"Source block {source_block_id} not found")

        wire = create_wire(wire_id, source_block_id, target_block_id,
                          strength, source_block.color, spatial_position)
        self.add_wire(wire)
        return wire

    def initialize_dye_system(self, activation_range: Tuple[float, float] = (-50.0, 50.0),
                            lag_range: Tuple[float, float] = (0.0, 250.0),
                            resolution: float = 2.0):
        """
        Initialize dye system for network.

        Args:
            activation_range: (min, max) activation axis bounds
            lag_range: (min, max) lag axis bounds
            resolution: Spatial grid resolution
        """
        if self.enable_dye_system:
            self.dye_system = create_dye_system_for_network(
                activation_range, lag_range, resolution
            )
            # Update dye concentrations for existing wires
            self._update_dye_concentrations()

    def _update_dye_concentrations(self):
        """Update dye concentration cache for all wires"""
        if self.dye_system is None:
            return

        for wire in self.wires:
            from neural_klotski.core.dye import DyeColor
            try:
                dye_color = DyeColor.from_block_color(wire.color)
                concentration = self.dye_system.get_concentration(
                    dye_color, wire.spatial_position[0], wire.spatial_position[1]
                )
                self.dye_concentrations[wire.wire_id] = concentration
            except ValueError:
                self.dye_concentrations[wire.wire_id] = 0.0

    def inject_learning_signal(self, block_color: BlockColor, activation: float,
                             lag: float, success: bool = True):
        """
        Inject learning signal into dye system.

        Args:
            block_color: Color determining dye type
            activation: Spatial activation coordinate
            lag: Spatial lag coordinate
            success: Whether trial was successful
        """
        if self.dye_system is not None:
            self.dye_system.inject_learning_signal(
                block_color, activation, lag, self.config.dyes, success
            )

    def start_learning_trial(self):
        """Start a new learning trial"""
        if self.learning_system is not None:
            self.learning_system.start_trial(self.current_time)
            self.in_trial = True
            self.trial_start_time = self.current_time

    def complete_learning_trial(self, outcome: TrialOutcome, performance_score: float,
                              error_magnitude: float = 0.0,
                              spatial_focus: Optional[Tuple[float, float]] = None) -> Dict[str, any]:
        """
        Complete current learning trial and process outcome.

        Args:
            outcome: Trial outcome type
            performance_score: Performance score (0.0-1.0)
            error_magnitude: Error magnitude for learning signal strength
            spatial_focus: Optional spatial focus for targeted learning

        Returns:
            Trial completion statistics
        """
        if not self.in_trial or self.learning_system is None:
            return {}

        trial_duration = self.current_time - self.trial_start_time
        trial_result = TrialResult(
            outcome=outcome,
            trial_duration=trial_duration,
            performance_score=performance_score,
            error_magnitude=error_magnitude,
            spatial_focus=spatial_focus
        )

        # Process trial outcome
        outcome_stats = self.learning_system.process_trial_outcome(
            trial_result, self.current_time, self.blocks, self.wires, self.dye_system
        )

        self.total_learning_signals += outcome_stats.get('signals_injected', 0)
        self.in_trial = False

        return outcome_stats

    def get_network_state(self) -> NetworkState:
        """Get complete network state snapshot"""
        plasticity_stats = None
        if self.plasticity_manager is not None:
            plasticity_stats = self.plasticity_manager.get_plasticity_statistics()

        learning_stats = None
        if self.learning_system is not None:
            learning_stats = self.learning_system.get_learning_statistics()

        return NetworkState(
            current_time=self.current_time,
            blocks=self.blocks.copy(),
            wires=self.wires.copy(),
            signal_queue=self.signal_manager.signal_queue,
            dye_concentrations=self.dye_concentrations.copy(),
            dye_system=self.dye_system,
            plasticity_stats=plasticity_stats,
            learning_stats=learning_stats
        )

    def step_block_dynamics(self) -> Set[int]:
        """
        Execute one timestep of block dynamics.

        Returns:
            Set of block IDs that fired this timestep
        """
        fired_blocks = set()

        for block in self.blocks.values():
            # Execute block dynamics step
            fired = block.step_dynamics(self.config.dynamics)
            if fired:
                fired_blocks.add(block.block_id)
                self.total_firings += 1

        return fired_blocks

    def process_signals(self) -> Tuple[int, int]:
        """
        Process signals arriving at current timestep.

        Returns:
            Tuple of (signals_processed, blocks_affected)
        """
        signals_processed, blocks_affected = self.signal_manager.process_network_timestep(
            self.current_time, self.blocks, max_force_magnitude=self.config.wires.max_damage
        )

        self.total_forces_applied += blocks_affected
        return signals_processed, blocks_affected

    def create_signals_from_firings(self, fired_block_ids: Set[int]) -> int:
        """
        Create signals from fired blocks.

        Args:
            fired_block_ids: Set of block IDs that fired

        Returns:
            Number of signals created
        """
        signals_created = self.signal_manager.add_firing_signals(
            list(fired_block_ids), self.current_time, self.blocks, self.wires,
            self.dye_concentrations, self.config.dyes.enhancement_factor
        )

        self.total_signals_created += signals_created
        return signals_created

    def execute_timestep(self) -> Dict[str, any]:
        """
        Execute one complete network timestep.

        Performs the full sequence:
        1. Process arriving signals and apply forces
        2. Execute block dynamics (including firing detection)
        3. Process plasticity updates
        4. Step dye dynamics
        5. Create new signals from fired blocks
        6. Advance time

        Returns:
            Dictionary with timestep statistics
        """
        # Step 1: Process signals arriving at current time
        signals_processed, blocks_affected = self.process_signals()

        # Step 2: Execute block dynamics
        fired_blocks = self.step_block_dynamics()

        # Step 3: Process plasticity updates
        plasticity_stats = {}
        if self.plasticity_manager is not None and fired_blocks:
            plasticity_stats = self.plasticity_manager.execute_plasticity_timestep(
                list(fired_blocks), self.current_time, self.blocks, self.wires,
                self.dye_system, self.config.dynamics.dt
            )
            self.total_plasticity_updates += plasticity_stats.get('updates_applied', 0)

        # Step 4: Process learning integration (if in trial)
        learning_stats = {}
        if self.learning_system is not None and self.in_trial:
            # Get activated wire IDs from current signals (simplified approach)
            # For now, just use wire IDs from all wires that have recently created signals
            activated_wire_ids = []
            if fired_blocks:
                # Get wire IDs for wires connected to fired blocks
                for wire in self.wires:
                    if wire.source_block_id in fired_blocks:
                        activated_wire_ids.append(wire.wire_id)

            learning_stats = self.learning_system.process_learning_timestep(
                self.current_time, list(fired_blocks), activated_wire_ids,
                self.blocks, self.wires, self.dye_system, self.plasticity_manager
            )

        # Step 5: Step dye dynamics (diffusion and decay)
        if self.dye_system is not None:
            self.dye_system.step_all_dynamics(self.config.dyes, self.config.dynamics.dt)
            # Update dye concentration cache
            self._update_dye_concentrations()

        # Step 6: Create signals from fired blocks
        signals_created = self.create_signals_from_firings(fired_blocks)

        # Step 7: Advance time
        self.current_time += self.config.dynamics.dt
        self.total_timesteps += 1

        # Return timestep statistics
        stats = {
            'timestep': self.total_timesteps,
            'time': self.current_time,
            'signals_processed': signals_processed,
            'blocks_affected': blocks_affected,
            'blocks_fired': len(fired_blocks),
            'signals_created': signals_created,
            'fired_block_ids': list(fired_blocks),
            'pending_signals': len(self.signal_manager.signal_queue),
            'plasticity_stats': plasticity_stats,
            'learning_stats': learning_stats,
            'in_trial': self.in_trial
        }

        return stats

    def run_simulation(self, num_timesteps: int, verbose: bool = False) -> List[Dict[str, any]]:
        """
        Run network simulation for specified number of timesteps.

        Args:
            num_timesteps: Number of timesteps to execute
            verbose: Whether to print progress information

        Returns:
            List of timestep statistics dictionaries
        """
        timestep_stats = []

        for step in range(num_timesteps):
            stats = self.execute_timestep()
            timestep_stats.append(stats)

            if verbose and (step % 100 == 0 or step < 10):
                print(f"Step {step}: fired={stats['blocks_fired']}, "
                      f"signals={stats['signals_created']}, "
                      f"pending={stats['pending_signals']}")

        if verbose:
            print(f"\nSimulation completed: {num_timesteps} timesteps")
            print(f"Total firings: {self.total_firings}")
            print(f"Total signals: {self.total_signals_created}")
            if self.enable_plasticity:
                print(f"Total plasticity updates: {self.total_plasticity_updates}")
            if self.enable_learning:
                print(f"Total learning signals: {self.total_learning_signals}")

        return timestep_stats

    def get_simulation_statistics(self) -> Dict[str, any]:
        """Get overall simulation statistics"""
        stats = {
            'total_timesteps': self.total_timesteps,
            'current_time': self.current_time,
            'total_firings': self.total_firings,
            'total_signals_created': self.total_signals_created,
            'total_forces_applied': self.total_forces_applied,
            'total_plasticity_updates': self.total_plasticity_updates,
            'total_learning_signals': self.total_learning_signals,
            'network_size': {
                'blocks': len(self.blocks),
                'wires': len(self.wires)
            },
            'signal_queue_status': self.signal_manager.get_queue_status(),
            'dye_system_enabled': self.enable_dye_system,
            'plasticity_enabled': self.enable_plasticity,
            'learning_enabled': self.enable_learning,
            'in_trial': self.in_trial
        }

        # Add plasticity statistics if available
        if self.plasticity_manager is not None:
            stats['plasticity_stats'] = self.plasticity_manager.get_plasticity_statistics()

        # Add dye system statistics if available
        if self.dye_system is not None:
            stats['dye_system_stats'] = self.dye_system.get_system_stats()

        # Add learning system statistics if available
        if self.learning_system is not None:
            stats['learning_stats'] = self.learning_system.get_learning_statistics()

        return stats

    def reset_simulation(self):
        """Reset simulation to initial state"""
        self.current_time = 0.0
        self.total_timesteps = 0
        self.total_firings = 0
        self.total_signals_created = 0
        self.total_forces_applied = 0
        self.total_plasticity_updates = 0
        self.total_learning_signals = 0

        # Reset trial state
        self.in_trial = False
        self.trial_start_time = 0.0

        # Reset all blocks
        for block in self.blocks.values():
            block.position = 0.0
            block.velocity = 0.0
            block.refractory_timer = 0.0
            block.reset_forces()

        # Clear signal queue
        self.signal_manager.clear_signals()

        # Reset plasticity system
        if self.plasticity_manager is not None:
            self.plasticity_manager.reset_plasticity()

        # Clear dye system
        if self.dye_system is not None:
            self.dye_system.clear_all_dye()
            self.dye_concentrations.clear()

        # Reset learning system
        if self.learning_system is not None:
            self.learning_system.reset_learning_state()

    def validate_network(self) -> Tuple[bool, List[str]]:
        """
        Validate network connectivity and configuration.

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check block connectivity
        block_ids = set(self.blocks.keys())

        for wire in self.wires:
            if wire.source_block_id not in block_ids:
                errors.append(f"Wire {wire.wire_id} references non-existent source block {wire.source_block_id}")
            if wire.target_block_id not in block_ids:
                errors.append(f"Wire {wire.wire_id} references non-existent target block {wire.target_block_id}")

        # Check for self-connections
        for wire in self.wires:
            if wire.source_block_id == wire.target_block_id:
                errors.append(f"Wire {wire.wire_id} creates self-connection on block {wire.source_block_id}")

        # Check block state validity
        for block_id, block in self.blocks.items():
            if not block.validate_state():
                errors.append(f"Block {block_id} has invalid state")

        # Check configuration validity
        if not self.config.validate():
            errors.append("Network configuration is invalid")

        return len(errors) == 0, errors


def create_simple_test_network() -> NeuralKlotskiNetwork:
    """
    Create a simple test network for validation.

    Creates a 3-block network with different wire types to test
    all force calculation paths, dye dynamics, plasticity, and learning.
    """
    network = NeuralKlotskiNetwork(enable_dye_system=True, enable_plasticity=True, enable_learning=True)

    # Initialize dye system
    network.initialize_dye_system(activation_range=(-50.0, 50.0), lag_range=(0.0, 200.0))

    # Create blocks with lower thresholds for testing (compatible with existing tests)
    block1 = network.create_block(1, 50.0, BlockColor.RED, 10.0, initial_position=0.0)
    block2 = network.create_block(2, 100.0, BlockColor.BLUE, 10.0, initial_position=0.0)
    block3 = network.create_block(3, 150.0, BlockColor.YELLOW, 10.0, initial_position=0.0)

    # Create wires with spatial positions for dye lookup
    wire1 = network.create_wire(1, 1, 2, 2.0, (10.0, 75.0))  # Red wire: 1 -> 2
    wire2 = network.create_wire(2, 2, 3, 1.5, (-5.0, 125.0))  # Blue wire: 2 -> 3
    wire3 = network.create_wire(3, 3, 1, 1.8, (5.0, 100.0))  # Yellow wire: 3 -> 1

    # Inject some initial dye for testing plasticity
    if network.dye_system is not None:
        network.inject_learning_signal(BlockColor.RED, 10.0, 75.0, success=True)
        network.inject_learning_signal(BlockColor.BLUE, -5.0, 125.0, success=True)

    return network


if __name__ == "__main__":
    # Test complete network integration
    print("Testing Neural-Klotski Network Integration...")

    # Create test network
    network = create_simple_test_network()

    # Validate network
    is_valid, errors = network.validate_network()
    print(f"Network validation: {'PASSED' if is_valid else 'FAILED'}")
    if errors:
        for error in errors:
            print(f"  Error: {error}")

    # Get initial state
    print(f"\nInitial network state: {network.get_network_state().get_network_summary()}")

    # Apply external force to trigger activity
    network.blocks[1].add_force(25.0)  # Strong force to trigger firing

    # Run short simulation
    print("\nRunning 10-timestep simulation...")
    stats = network.run_simulation(10, verbose=True)

    # Show final statistics
    final_stats = network.get_simulation_statistics()
    print(f"\nFinal simulation statistics:")
    for key, value in final_stats.items():
        print(f"  {key}: {value}")

    # Test specific timestep with detailed output
    print(f"\nDetailed output for last timestep:")
    print(f"  Timestep stats: {stats[-1]}")

    print("\nNetwork integration test completed successfully!")