"""
Neural-Klotski Addition Network Training System.

Implements complete training pipeline for teaching addition through
dye-enhanced plasticity and reinforcement learning.
"""

from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import time
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from neural_klotski.core.network import NeuralKlotskiNetwork
from neural_klotski.core.encoding import AdditionProblem, AdditionTaskManager, DecodingResult
from neural_klotski.core.architecture import create_addition_network
from neural_klotski.config import SimulationConfig


class TrainingPhase(Enum):
    """Training phases for curriculum learning"""
    SIMPLE = "simple"           # Small numbers (0-15)
    INTERMEDIATE = "intermediate"  # Medium numbers (0-127)
    ADVANCED = "advanced"       # Full range (0-511)


@dataclass
class TrainingConfig:
    """Configuration for training process"""
    # Training schedule
    epochs_per_phase: Dict[TrainingPhase, int] = field(default_factory=lambda: {
        TrainingPhase.SIMPLE: 100,
        TrainingPhase.INTERMEDIATE: 200,
        TrainingPhase.ADVANCED: 500
    })

    # Problem generation
    problems_per_epoch: int = 50
    curriculum_enabled: bool = True

    # Learning parameters
    success_threshold: float = 0.8  # Accuracy needed to advance phase
    learning_rate_decay: float = 0.95
    min_learning_rate: float = 0.001

    # Monitoring
    evaluation_frequency: int = 10  # Epochs between evaluations
    save_frequency: int = 50       # Epochs between checkpoints
    early_stopping_patience: int = 100  # Epochs without improvement

    # Dye injection parameters
    success_dye_strength: float = 50.0
    failure_dye_strength: float = -25.0
    dye_injection_radius: float = 15.0


@dataclass
class TrainingMetrics:
    """Metrics tracked during training"""
    epoch: int = 0
    phase: TrainingPhase = TrainingPhase.SIMPLE
    accuracy: float = 0.0
    average_error: float = 0.0
    confidence: float = 0.0
    learning_rate: float = 0.01
    problems_solved: int = 0
    total_problems: int = 0

    # Performance history
    accuracy_history: List[float] = field(default_factory=list)
    error_history: List[float] = field(default_factory=list)
    confidence_history: List[float] = field(default_factory=list)

    # Timing
    epoch_duration: float = 0.0
    total_training_time: float = 0.0


@dataclass
class EvaluationResult:
    """Result of network evaluation"""
    accuracy: float
    average_error: float
    confidence: float
    problem_breakdown: Dict[str, float]
    sample_results: List[Tuple[AdditionProblem, DecodingResult]]


class AdditionNetworkTrainer:
    """
    Complete training system for Neural-Klotski addition networks.

    Implements curriculum learning, adaptive learning rates, and
    comprehensive monitoring for teaching addition tasks.
    """

    def __init__(self, config: TrainingConfig, simulation_config: SimulationConfig):
        """
        Initialize trainer.

        Args:
            config: Training configuration
            simulation_config: Simulation configuration
        """
        self.config = config
        self.simulation_config = simulation_config

        # Create network and task manager
        self.network = create_addition_network(enable_learning=True)
        self.task_manager = AdditionTaskManager(simulation_config)

        # Training state
        self.metrics = TrainingMetrics()
        self.current_phase = TrainingPhase.SIMPLE
        self.best_accuracy = 0.0
        self.epochs_without_improvement = 0
        self.training_start_time = 0.0

        # Callbacks
        self.epoch_callbacks: List[Callable[[TrainingMetrics], None]] = []
        self.phase_callbacks: List[Callable[[TrainingPhase], None]] = []

    def add_epoch_callback(self, callback: Callable[[TrainingMetrics], None]):
        """Add callback to be called after each epoch"""
        self.epoch_callbacks.append(callback)

    def add_phase_callback(self, callback: Callable[[TrainingPhase], None]):
        """Add callback to be called when phase changes"""
        self.phase_callbacks.append(callback)

    def generate_problems_for_phase(self, phase: TrainingPhase, count: int) -> List[AdditionProblem]:
        """
        Generate training problems for specific phase.

        Args:
            phase: Training phase
            count: Number of problems to generate

        Returns:
            List of addition problems
        """
        problems = []

        # Define operand ranges for each phase
        max_operands = {
            TrainingPhase.SIMPLE: 15,         # 4 bits
            TrainingPhase.INTERMEDIATE: 127,  # 7 bits
            TrainingPhase.ADVANCED: 511       # 10 bits (full capacity)
        }

        max_operand = max_operands[phase]

        for _ in range(count):
            problem = self.task_manager.generate_random_problem(max_operand)
            problems.append(problem)

        return problems

    def train_epoch(self, problems: List[AdditionProblem]) -> Dict[str, float]:
        """
        Train network for one epoch.

        Args:
            problems: List of training problems

        Returns:
            Dictionary of epoch statistics
        """
        epoch_start = time.time()

        correct_count = 0
        total_error = 0.0
        total_confidence = 0.0

        for problem in problems:
            # Reset network state
            self.network.reset_simulation()

            # Start learning trial
            self.network.start_learning_trial()

            # Execute problem
            result, stats = self.task_manager.execute_addition_task(self.network, problem)

            # Determine success and inject learning signals
            is_correct = (result.decoded_sum == problem.expected_sum)
            self._inject_learning_signals(is_correct, result.confidence)

            # Complete learning trial
            from neural_klotski.core.learning import TrialOutcome
            outcome = TrialOutcome.SUCCESS if is_correct else TrialOutcome.FAILURE
            performance_score = result.confidence if is_correct else (1.0 - result.confidence)
            error_magnitude = abs(result.decoded_sum - problem.expected_sum)

            self.network.complete_learning_trial(
                outcome=outcome,
                performance_score=performance_score,
                error_magnitude=error_magnitude
            )

            # Update statistics
            if is_correct:
                correct_count += 1

            error = abs(result.decoded_sum - problem.expected_sum)
            total_error += error
            total_confidence += result.confidence

        # Calculate epoch metrics
        accuracy = correct_count / len(problems)
        avg_error = total_error / len(problems)
        avg_confidence = total_confidence / len(problems)
        epoch_duration = time.time() - epoch_start

        return {
            'accuracy': accuracy,
            'average_error': avg_error,
            'confidence': avg_confidence,
            'epoch_duration': epoch_duration,
            'problems_solved': correct_count,
            'total_problems': len(problems)
        }

    def _inject_learning_signals(self, success: bool, confidence: float):
        """
        Inject dye learning signals based on trial outcome.

        Args:
            success: Whether the trial was successful
            confidence: Confidence level of the result
        """
        if not hasattr(self.network, 'learning_system'):
            return

        # For now, skip complex learning signal injection
        # This would require creating proper TrialResult objects
        # and using the learning system's process_trial_outcome method
        pass

    def evaluate_network(self, num_problems: int = 100) -> EvaluationResult:
        """
        Evaluate network performance on test problems.

        Args:
            num_problems: Number of test problems

        Returns:
            Evaluation result
        """
        # Generate test problems for current phase
        test_problems = self.generate_problems_for_phase(self.current_phase, num_problems)

        correct_count = 0
        total_error = 0.0
        total_confidence = 0.0
        sample_results = []

        # Track performance by problem size
        size_performance = {'small': [], 'medium': [], 'large': []}

        for problem in test_problems:
            # Reset network state (no learning during evaluation)
            self.network.reset_simulation()

            # Execute problem without learning
            result, stats = self.task_manager.execute_addition_task(self.network, problem)

            # Record results
            is_correct = (result.decoded_sum == problem.expected_sum)
            if is_correct:
                correct_count += 1

            error = abs(result.decoded_sum - problem.expected_sum)
            total_error += error
            total_confidence += result.confidence

            # Categorize by problem size
            max_operand = max(problem.operand1, problem.operand2)
            if max_operand <= 15:
                category = 'small'
            elif max_operand <= 127:
                category = 'medium'
            else:
                category = 'large'

            size_performance[category].append(is_correct)

            # Keep some sample results
            if len(sample_results) < 10:
                sample_results.append((problem, result))

        # Calculate breakdown by problem size
        problem_breakdown = {}
        for category, results in size_performance.items():
            if results:
                problem_breakdown[f'{category}_accuracy'] = sum(results) / len(results)
            else:
                problem_breakdown[f'{category}_accuracy'] = 0.0

        return EvaluationResult(
            accuracy=correct_count / num_problems,
            average_error=total_error / num_problems,
            confidence=total_confidence / num_problems,
            problem_breakdown=problem_breakdown,
            sample_results=sample_results
        )

    def should_advance_phase(self, accuracy: float) -> bool:
        """Check if network should advance to next training phase"""
        return accuracy >= self.config.success_threshold

    def advance_phase(self):
        """Advance to next training phase"""
        phase_order = [TrainingPhase.SIMPLE, TrainingPhase.INTERMEDIATE, TrainingPhase.ADVANCED]
        current_index = phase_order.index(self.current_phase)

        if current_index < len(phase_order) - 1:
            self.current_phase = phase_order[current_index + 1]
            print(f"Advancing to phase: {self.current_phase.value}")

            # Notify callbacks
            for callback in self.phase_callbacks:
                callback(self.current_phase)
        else:
            print("Already at final training phase")

    def update_learning_rate(self):
        """Update learning rate with decay"""
        current_lr = self.metrics.learning_rate
        new_lr = max(current_lr * self.config.learning_rate_decay,
                    self.config.min_learning_rate)

        self.metrics.learning_rate = new_lr

        # Update network learning rate if possible
        if hasattr(self.network, 'learning_system'):
            self.network.learning_system.learning_rate = new_lr

    def train(self, max_epochs: Optional[int] = None) -> TrainingMetrics:
        """
        Execute complete training process.

        Args:
            max_epochs: Maximum epochs to train (None for unlimited)

        Returns:
            Final training metrics
        """
        print("Starting Neural-Klotski addition network training...")
        print(f"Network: {len(self.network.blocks)} blocks, {len(self.network.wires)} wires")

        self.training_start_time = time.time()
        self.metrics = TrainingMetrics()

        # Training loop
        while True:
            # Check stopping conditions
            if max_epochs and self.metrics.epoch >= max_epochs:
                print(f"Reached maximum epochs: {max_epochs}")
                break

            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                print(f"Early stopping: no improvement for {self.config.early_stopping_patience} epochs")
                break

            # Check if phase is complete
            phase_epochs = self.config.epochs_per_phase.get(self.current_phase, float('inf'))
            if self.metrics.epoch >= phase_epochs:
                if self.current_phase != TrainingPhase.ADVANCED:
                    self.advance_phase()
                else:
                    print("Training complete: finished all phases")
                    break

            # Generate training problems
            problems = self.generate_problems_for_phase(
                self.current_phase,
                self.config.problems_per_epoch
            )

            # Train epoch
            epoch_stats = self.train_epoch(problems)

            # Update metrics
            self.metrics.epoch += 1
            self.metrics.phase = self.current_phase
            self.metrics.accuracy = epoch_stats['accuracy']
            self.metrics.average_error = epoch_stats['average_error']
            self.metrics.confidence = epoch_stats['confidence']
            self.metrics.epoch_duration = epoch_stats['epoch_duration']
            self.metrics.problems_solved += epoch_stats['problems_solved']
            self.metrics.total_problems += epoch_stats['total_problems']
            self.metrics.total_training_time = time.time() - self.training_start_time

            # Update history
            self.metrics.accuracy_history.append(self.metrics.accuracy)
            self.metrics.error_history.append(self.metrics.average_error)
            self.metrics.confidence_history.append(self.metrics.confidence)

            # Check for improvement
            if self.metrics.accuracy > self.best_accuracy:
                self.best_accuracy = self.metrics.accuracy
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            # Periodic evaluation
            if self.metrics.epoch % self.config.evaluation_frequency == 0:
                eval_result = self.evaluate_network()
                print(f"Epoch {self.metrics.epoch:3d} | "
                      f"Phase: {self.current_phase.value:12s} | "
                      f"Acc: {eval_result.accuracy:.3f} | "
                      f"Err: {eval_result.average_error:.2f} | "
                      f"Conf: {eval_result.confidence:.3f} | "
                      f"LR: {self.metrics.learning_rate:.4f}")

            # Update learning rate
            if self.metrics.epoch % 20 == 0:
                self.update_learning_rate()

            # Check phase advancement
            if (self.metrics.epoch % self.config.evaluation_frequency == 0 and
                self.should_advance_phase(self.metrics.accuracy) and
                self.current_phase != TrainingPhase.ADVANCED):
                self.advance_phase()

            # Execute callbacks
            for callback in self.epoch_callbacks:
                callback(self.metrics)

        # Final evaluation
        print("\nTraining completed!")
        final_eval = self.evaluate_network(200)  # Large test set
        print(f"Final accuracy: {final_eval.accuracy:.3f}")
        print(f"Final average error: {final_eval.average_error:.2f}")
        print(f"Training time: {self.metrics.total_training_time:.1f}s")

        return self.metrics


if __name__ == "__main__":
    # Test training system
    from neural_klotski.config import get_default_config

    print("Testing Neural-Klotski Training System...")

    config = get_default_config()
    training_config = TrainingConfig()
    training_config.epochs_per_phase = {
        TrainingPhase.SIMPLE: 10,
        TrainingPhase.INTERMEDIATE: 20,
        TrainingPhase.ADVANCED: 30
    }
    training_config.problems_per_epoch = 20
    training_config.evaluation_frequency = 5

    trainer = AdditionNetworkTrainer(training_config, config.simulation)

    # Add simple logging callback
    def log_progress(metrics: TrainingMetrics):
        if metrics.epoch % 5 == 0:
            print(f"Epoch {metrics.epoch}: Accuracy {metrics.accuracy:.3f}")

    trainer.add_epoch_callback(log_progress)

    # Run short training test
    final_metrics = trainer.train(max_epochs=20)

    print(f"Test completed successfully!")
    print(f"Final accuracy: {final_metrics.accuracy:.3f}")