"""
Learning integration system for Neural-Klotski.

Combines dye diffusion with plasticity mechanisms to implement complete learning
capabilities including eligibility traces, temporal credit assignment, and
performance-based learning signal injection.
"""

from typing import Dict, List, Tuple, Optional, Deque, Union
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
from neural_klotski.core.plasticity import PlasticityManager, PlasticityType
from neural_klotski.config import LearningConfig, DyeConfig
from neural_klotski.math_utils import clamp


class TrialOutcome(Enum):
    """Trial outcome types for learning signal injection"""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    TIMEOUT = "timeout"


@dataclass
class LearningEvent:
    """Records a learning event for eligibility trace calculations"""
    event_time: float
    block_id: int
    wire_id: Optional[int]
    event_type: str  # "firing", "signal", "outcome"
    strength: float = 1.0
    spatial_position: Tuple[float, float] = (0.0, 0.0)

    def __repr__(self) -> str:
        return f"LearningEvent({self.event_type}, block={self.block_id}, t={self.event_time:.1f})"


@dataclass
class TrialResult:
    """Complete trial result for learning signal injection"""
    outcome: TrialOutcome
    trial_duration: float
    performance_score: float  # 0.0-1.0, higher is better
    error_magnitude: float = 0.0
    spatial_focus: Optional[Tuple[float, float]] = None  # Where to focus learning signals

    def __repr__(self) -> str:
        return f"TrialResult({self.outcome.value}, score={self.performance_score:.3f})"


class EligibilityTraceManager:
    """
    Manages eligibility traces for temporal credit assignment.

    Implements exponentially decaying traces that mark wires and blocks
    as eligible for learning updates based on recent activity.
    """

    def __init__(self, trace_decay_constant: float = 50.0):
        """
        Initialize eligibility trace manager.

        Args:
            trace_decay_constant: Time constant for eligibility decay (τ_e)
        """
        self.trace_decay_constant = trace_decay_constant

        # Eligibility traces: ID -> current trace value
        self.wire_traces: Dict[int, float] = {}
        self.block_traces: Dict[int, float] = {}

        # Learning events for trace calculation
        self.learning_events: Deque[LearningEvent] = deque()
        self.max_event_history = 1000

        # Statistics
        self.total_trace_updates = 0
        self.last_cleanup_time = 0.0

    def add_learning_event(self, event: LearningEvent):
        """Add learning event to eligibility calculation"""
        self.learning_events.append(event)

        # Maintain history size
        if len(self.learning_events) > self.max_event_history:
            self.learning_events.popleft()

    def update_eligibility_traces(self, current_time: float, fired_block_ids: List[int],
                                activated_wire_ids: List[int]):
        """
        Update eligibility traces based on current activity.

        Args:
            current_time: Current simulation time
            fired_block_ids: Blocks that fired this timestep
            activated_wire_ids: Wires that carried signals this timestep
        """
        # Decay existing traces
        dt = 1.0  # Assume unit timestep for simplicity
        decay_factor = np.exp(-dt / self.trace_decay_constant)

        # Decay wire traces
        for wire_id in list(self.wire_traces.keys()):
            self.wire_traces[wire_id] *= decay_factor
            if self.wire_traces[wire_id] < 0.001:
                del self.wire_traces[wire_id]

        # Decay block traces
        for block_id in list(self.block_traces.keys()):
            self.block_traces[block_id] *= decay_factor
            if self.block_traces[block_id] < 0.001:
                del self.block_traces[block_id]

        # Add new traces for active elements
        for block_id in fired_block_ids:
            self.block_traces[block_id] = 1.0
            event = LearningEvent(current_time, block_id, None, "firing")
            self.add_learning_event(event)

        for wire_id in activated_wire_ids:
            self.wire_traces[wire_id] = 1.0
            # Add learning event (block_id will be set by caller if known)
            event = LearningEvent(current_time, -1, wire_id, "signal")
            self.add_learning_event(event)

        self.total_trace_updates += 1

    def get_eligibility_trace(self, element_type: str, element_id: int) -> float:
        """
        Get current eligibility trace value.

        Args:
            element_type: "wire" or "block"
            element_id: Element identifier

        Returns:
            Current eligibility trace value (0.0-1.0)
        """
        if element_type == "wire":
            return self.wire_traces.get(element_id, 0.0)
        elif element_type == "block":
            return self.block_traces.get(element_id, 0.0)
        else:
            return 0.0

    def get_trace_statistics(self) -> Dict[str, any]:
        """Get eligibility trace statistics"""
        return {
            'active_wire_traces': len(self.wire_traces),
            'active_block_traces': len(self.block_traces),
            'total_trace_updates': self.total_trace_updates,
            'learning_events': len(self.learning_events),
            'avg_wire_trace': np.mean(list(self.wire_traces.values())) if self.wire_traces else 0.0,
            'avg_block_trace': np.mean(list(self.block_traces.values())) if self.block_traces else 0.0
        }

    def clear_traces(self):
        """Clear all eligibility traces"""
        self.wire_traces.clear()
        self.block_traces.clear()
        self.learning_events.clear()
        self.total_trace_updates = 0


class LearningSignalGenerator:
    """
    Generates learning signals based on trial outcomes.

    Converts trial results into spatially and temporally appropriate
    dye injections for plasticity enhancement.
    """

    def __init__(self, dye_config: DyeConfig):
        """
        Initialize learning signal generator.

        Args:
            dye_config: Dye system configuration
        """
        self.dye_config = dye_config
        self.signal_history: List[Tuple[float, TrialResult]] = []
        self.total_signals_generated = 0

    def generate_learning_signals(self, trial_result: TrialResult, current_time: float,
                                eligibility_traces: EligibilityTraceManager,
                                blocks: Dict[int, BlockState], wires: List[Wire]) -> List[Tuple[DyeColor, float, float, float]]:
        """
        Generate learning signals based on trial outcome.

        Args:
            trial_result: Trial outcome and performance metrics
            current_time: Current simulation time
            eligibility_traces: Eligibility trace manager
            blocks: Network blocks
            wires: Network wires

        Returns:
            List of (dye_color, activation, lag, amount) tuples for injection
        """
        signals = []

        # Base injection amount based on outcome
        base_amount = self._calculate_base_injection_amount(trial_result)

        # Generate signals for eligible blocks
        for block_id, trace_strength in eligibility_traces.block_traces.items():
            if trace_strength > 0.1:  # Only for significant traces
                block = blocks.get(block_id)
                if block is not None:
                    # Convert block color to dye color
                    try:
                        dye_color = DyeColor.from_block_color(block.color)

                        # Calculate injection amount with eligibility weighting
                        injection_amount = base_amount * trace_strength

                        # Use block spatial position
                        activation = block.position
                        lag = block.lag_position

                        signals.append((dye_color, activation, lag, injection_amount))

                    except ValueError:
                        continue  # Skip unknown colors

        # Generate signals for eligible wires
        for wire_id, trace_strength in eligibility_traces.wire_traces.items():
            if trace_strength > 0.1:
                # Find the wire
                wire = next((w for w in wires if w.wire_id == wire_id), None)
                if wire is not None:
                    try:
                        dye_color = DyeColor.from_block_color(wire.color)

                        # Calculate injection amount
                        injection_amount = base_amount * trace_strength * 0.8  # Slightly less for wires

                        # Use wire spatial position
                        activation, lag = wire.spatial_position

                        signals.append((dye_color, activation, lag, injection_amount))

                    except ValueError:
                        continue

        # If spatial focus is specified, add targeted signal
        if trial_result.spatial_focus is not None:
            focus_activation, focus_lag = trial_result.spatial_focus

            # Inject all three dye colors at focus point for broad enhancement
            for dye_color in [DyeColor.RED, DyeColor.BLUE, DyeColor.YELLOW]:
                focus_amount = base_amount * 1.5  # Enhanced amount for focused learning
                signals.append((dye_color, focus_activation, focus_lag, focus_amount))

        # Record signal generation
        self.signal_history.append((current_time, trial_result))
        self.total_signals_generated += len(signals)

        return signals

    def _calculate_base_injection_amount(self, trial_result: TrialResult) -> float:
        """Calculate base injection amount based on trial outcome"""
        base = self.dye_config.injection_amount

        if trial_result.outcome == TrialOutcome.SUCCESS:
            # Strong positive signal
            return base * (1.0 + trial_result.performance_score)
        elif trial_result.outcome == TrialOutcome.PARTIAL:
            # Moderate positive signal
            return base * (0.5 + 0.5 * trial_result.performance_score)
        elif trial_result.outcome == TrialOutcome.FAILURE:
            # Weak negative signal (anti-learning)
            return base * 0.2 * (1.0 - trial_result.performance_score)
        elif trial_result.outcome == TrialOutcome.TIMEOUT:
            # Very weak signal
            return base * 0.1
        else:
            return base * 0.5

    def get_signal_statistics(self) -> Dict[str, any]:
        """Get learning signal generation statistics"""
        recent_outcomes = [result.outcome for _, result in self.signal_history[-100:]]
        outcome_counts = {outcome: recent_outcomes.count(outcome) for outcome in TrialOutcome}

        return {
            'total_signals_generated': self.total_signals_generated,
            'signal_history_length': len(self.signal_history),
            'recent_outcome_counts': {k.value: v for k, v in outcome_counts.items()},
            'avg_recent_performance': np.mean([r.performance_score for _, r in self.signal_history[-50:]]) if self.signal_history else 0.0
        }


class AdaptiveLearningController:
    """
    Controls adaptive learning rates and mechanisms.

    Modulates learning parameters based on performance history,
    dye concentrations, and eligibility traces.
    """

    def __init__(self, learning_config: LearningConfig):
        """
        Initialize adaptive learning controller.

        Args:
            learning_config: Learning configuration parameters
        """
        self.learning_config = learning_config
        self.base_wire_learning_rate = learning_config.wire_learning_rate
        self.base_threshold_learning_rate = learning_config.threshold_learning_rate

        # Adaptation state
        self.performance_history: Deque[float] = deque(maxlen=100)
        self.learning_rate_multiplier = 1.0
        self.adaptation_momentum = 0.95

        # Statistics
        self.total_adaptations = 0

    def update_performance_history(self, performance_score: float):
        """Update performance history for adaptation"""
        self.performance_history.append(performance_score)

    def calculate_adaptive_learning_rates(self, dye_concentration: float = 0.0,
                                        eligibility_trace: float = 0.0) -> Dict[str, float]:
        """
        Calculate adaptive learning rates based on current context.

        Args:
            dye_concentration: Local dye concentration
            eligibility_trace: Eligibility trace strength

        Returns:
            Dictionary with adaptive learning rates
        """
        # Base adaptation based on recent performance
        if len(self.performance_history) >= 10:
            recent_performance = np.mean(list(self.performance_history)[-10:])
            overall_performance = np.mean(list(self.performance_history))

            # If recent performance is declining, increase learning rate
            if recent_performance < overall_performance:
                performance_multiplier = 1.2
            else:
                performance_multiplier = 0.9
        else:
            performance_multiplier = 1.0

        # Dye enhancement (from specification)
        dye_multiplier = 1.0 + self.learning_config.dye_amplification * dye_concentration

        # Eligibility trace enhancement
        trace_multiplier = 1.0 + 0.5 * eligibility_trace

        # Combined multiplier with momentum
        target_multiplier = performance_multiplier * dye_multiplier * trace_multiplier
        self.learning_rate_multiplier = (self.adaptation_momentum * self.learning_rate_multiplier +
                                       (1 - self.adaptation_momentum) * target_multiplier)

        # Apply bounds to prevent instability
        self.learning_rate_multiplier = clamp(self.learning_rate_multiplier, 0.1, 5.0)

        self.total_adaptations += 1

        return {
            'wire_learning_rate': self.base_wire_learning_rate * self.learning_rate_multiplier,
            'threshold_learning_rate': self.base_threshold_learning_rate * self.learning_rate_multiplier,
            'multiplier': self.learning_rate_multiplier
        }

    def get_adaptation_statistics(self) -> Dict[str, any]:
        """Get adaptive learning statistics"""
        return {
            'learning_rate_multiplier': self.learning_rate_multiplier,
            'total_adaptations': self.total_adaptations,
            'performance_history_length': len(self.performance_history),
            'recent_avg_performance': np.mean(list(self.performance_history)[-20:]) if self.performance_history else 0.0,
            'overall_avg_performance': np.mean(list(self.performance_history)) if self.performance_history else 0.0
        }


class IntegratedLearningSystem:
    """
    Complete integrated learning system combining all components.

    Orchestrates dye diffusion, plasticity, eligibility traces, and
    adaptive learning for comprehensive learning capabilities.
    """

    def __init__(self, learning_config: LearningConfig, dye_config: DyeConfig):
        """
        Initialize integrated learning system.

        Args:
            learning_config: Learning configuration
            dye_config: Dye system configuration
        """
        self.learning_config = learning_config
        self.dye_config = dye_config

        # Learning components
        self.eligibility_traces = EligibilityTraceManager(dye_config.eligibility_window)
        self.signal_generator = LearningSignalGenerator(dye_config)
        self.adaptive_controller = AdaptiveLearningController(learning_config)

        # Performance tracking
        self.trial_count = 0
        self.successful_trials = 0
        self.total_learning_signals = 0

        # Integration state
        self.last_trial_time = 0.0
        self.current_trial_start = 0.0

    def start_trial(self, current_time: float):
        """Start a new learning trial"""
        self.current_trial_start = current_time
        self.trial_count += 1

    def process_learning_timestep(self, current_time: float, fired_block_ids: List[int],
                                activated_wire_ids: List[int], blocks: Dict[int, BlockState],
                                wires: List[Wire], dye_system: DyeSystem,
                                plasticity_manager: PlasticityManager) -> Dict[str, any]:
        """
        Process complete learning timestep integration.

        Args:
            current_time: Current simulation time
            fired_block_ids: Blocks that fired this timestep
            activated_wire_ids: Wires that carried signals
            blocks: Network blocks
            wires: Network wires
            dye_system: Dye diffusion system
            plasticity_manager: Plasticity system

        Returns:
            Learning timestep statistics
        """
        # Update eligibility traces
        self.eligibility_traces.update_eligibility_traces(
            current_time, fired_block_ids, activated_wire_ids
        )

        # Get current learning context for adaptation
        max_dye_conc = 0.0
        max_eligibility = 0.0

        if dye_system is not None:
            # Sample dye concentrations from active blocks
            for block_id in fired_block_ids:
                block = blocks.get(block_id)
                if block is not None:
                    try:
                        dye_color = DyeColor.from_block_color(block.color)
                        conc = dye_system.get_concentration(dye_color, block.position, block.lag_position)
                        max_dye_conc = max(max_dye_conc, conc)
                    except ValueError:
                        continue

        # Get maximum eligibility trace
        if fired_block_ids:
            max_eligibility = max(self.eligibility_traces.get_eligibility_trace("block", bid)
                                for bid in fired_block_ids)

        # Calculate adaptive learning rates
        adaptive_rates = self.adaptive_controller.calculate_adaptive_learning_rates(
            max_dye_conc, max_eligibility
        )

        # Update plasticity manager with adaptive rates (if possible)
        # Note: This would require extending the plasticity manager to accept dynamic rates

        return {
            'eligibility_stats': self.eligibility_traces.get_trace_statistics(),
            'adaptive_rates': adaptive_rates,
            'max_dye_concentration': max_dye_conc,
            'max_eligibility_trace': max_eligibility,
            'fired_blocks': len(fired_block_ids),
            'activated_wires': len(activated_wire_ids)
        }

    def process_trial_outcome(self, trial_result: TrialResult, current_time: float,
                            blocks: Dict[int, BlockState], wires: List[Wire],
                            dye_system: DyeSystem) -> Dict[str, any]:
        """
        Process trial outcome and generate learning signals.

        Args:
            trial_result: Trial outcome and performance
            current_time: Current simulation time
            blocks: Network blocks
            wires: Network wires
            dye_system: Dye system for signal injection

        Returns:
            Trial processing statistics
        """
        # Update performance history
        self.adaptive_controller.update_performance_history(trial_result.performance_score)

        # Generate learning signals
        learning_signals = self.signal_generator.generate_learning_signals(
            trial_result, current_time, self.eligibility_traces, blocks, wires
        )

        # Inject learning signals into dye system
        signals_injected = 0
        if dye_system is not None:
            for dye_color, activation, lag, amount in learning_signals:
                dye_system.inject_dye(dye_color, activation, lag, amount)
                signals_injected += 1

        self.total_learning_signals += signals_injected

        # Track success rate
        if trial_result.outcome == TrialOutcome.SUCCESS:
            self.successful_trials += 1

        # Record trial completion
        self.last_trial_time = current_time

        return {
            'trial_outcome': trial_result.outcome.value,
            'performance_score': trial_result.performance_score,
            'learning_signals_generated': len(learning_signals),
            'signals_injected': signals_injected,
            'success_rate': self.successful_trials / self.trial_count if self.trial_count > 0 else 0.0,
            'trial_duration': current_time - self.current_trial_start
        }

    def get_learning_statistics(self) -> Dict[str, any]:
        """Get comprehensive learning system statistics"""
        return {
            'trial_count': self.trial_count,
            'successful_trials': self.successful_trials,
            'success_rate': self.successful_trials / self.trial_count if self.trial_count > 0 else 0.0,
            'total_learning_signals': self.total_learning_signals,
            'eligibility_traces': self.eligibility_traces.get_trace_statistics(),
            'signal_generation': self.signal_generator.get_signal_statistics(),
            'adaptive_control': self.adaptive_controller.get_adaptation_statistics()
        }

    def reset_learning_state(self):
        """Reset all learning state"""
        self.eligibility_traces.clear_traces()
        self.adaptive_controller.performance_history.clear()
        self.adaptive_controller.learning_rate_multiplier = 1.0
        self.trial_count = 0
        self.successful_trials = 0
        self.total_learning_signals = 0
        self.last_trial_time = 0.0
        self.current_trial_start = 0.0


if __name__ == "__main__":
    # Test integrated learning system
    from neural_klotski.config import get_default_config
    from neural_klotski.core.block import create_block
    from neural_klotski.core.wire import create_wire
    from neural_klotski.core.dye import create_dye_system_for_network

    print("Testing Neural-Klotski Integrated Learning System...")

    # Create test configuration
    config = get_default_config()
    learning_system = IntegratedLearningSystem(config.learning, config.dyes)

    # Create test network components
    blocks = {
        1: create_block(1, 50.0, BlockColor.RED, 10.0),
        2: create_block(2, 100.0, BlockColor.BLUE, 10.0),
        3: create_block(3, 150.0, BlockColor.YELLOW, 10.0)
    }

    wires = [
        create_wire(1, 1, 2, 2.0, BlockColor.RED, (10.0, 75.0)),
        create_wire(2, 2, 3, 1.5, BlockColor.BLUE, (-5.0, 125.0))
    ]

    dye_system = create_dye_system_for_network((-50, 50), (0, 200), 2.0)

    print(f"Initial learning statistics: {learning_system.get_learning_statistics()}")

    # Simulate learning over multiple trials
    current_time = 0.0
    dt = 0.5

    for trial in range(5):
        learning_system.start_trial(current_time)

        # Simulate trial activity
        for step in range(20):
            # Simulate some firing activity
            fired_blocks = []
            activated_wires = []

            if step % 3 == 0:
                fired_blocks.append(1)
                activated_wires.append(1)
            if step % 4 == 0:
                fired_blocks.append(2)
                activated_wires.append(2)
            if step % 7 == 0:
                fired_blocks.append(3)

            # Process learning timestep
            timestep_stats = learning_system.process_learning_timestep(
                current_time, fired_blocks, activated_wires, blocks, wires,
                dye_system, None  # No plasticity manager in this test
            )

            current_time += dt

        # Generate trial outcome
        performance = 0.6 + 0.3 * np.random.random()  # Random performance 0.6-0.9
        outcome = TrialOutcome.SUCCESS if performance > 0.7 else TrialOutcome.PARTIAL

        trial_result = TrialResult(
            outcome=outcome,
            trial_duration=20 * dt,
            performance_score=performance,
            spatial_focus=(5.0, 100.0) if outcome == TrialOutcome.SUCCESS else None
        )

        # Process trial outcome
        outcome_stats = learning_system.process_trial_outcome(
            trial_result, current_time, blocks, wires, dye_system
        )

        print(f"Trial {trial}: {outcome_stats}")

    # Show final statistics
    final_stats = learning_system.get_learning_statistics()
    print(f"\nFinal learning statistics: {final_stats}")

    print("\nIntegrated learning system test completed successfully!")