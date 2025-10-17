"""
Advanced Learning Rate Scheduling for Neural-Klotski Training.

Implements multiple scheduling strategies including cosine annealing,
warm restarts, and performance-based adaptive scheduling.
"""

from typing import List, Dict, Optional, Callable, Any, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
import math
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from neural_klotski.training.trainer import TrainingMetrics, TrainingPhase
from neural_klotski.training.monitoring import ConvergenceReport, ConvergenceStatus


class ScheduleType(Enum):
    """Types of learning rate schedules"""
    CONSTANT = "constant"
    LINEAR_DECAY = "linear_decay"
    EXPONENTIAL_DECAY = "exponential_decay"
    COSINE_ANNEALING = "cosine_annealing"
    COSINE_WARM_RESTARTS = "cosine_warm_restarts"
    STEP_DECAY = "step_decay"
    POLYNOMIAL_DECAY = "polynomial_decay"
    ADAPTIVE_PERFORMANCE = "adaptive_performance"
    CURRICULUM_BASED = "curriculum_based"


@dataclass
class ScheduleConfig:
    """Configuration for learning rate scheduling"""
    schedule_type: ScheduleType = ScheduleType.COSINE_ANNEALING

    # Basic parameters
    initial_lr: float = 0.01
    final_lr: float = 0.0001

    # Schedule-specific parameters
    decay_rate: float = 0.95           # For exponential decay
    decay_steps: int = 100             # For step decay
    polynomial_power: float = 1.0      # For polynomial decay

    # Cosine annealing parameters
    T_max: int = 200                   # Period for cosine annealing
    eta_min: float = 0.0001           # Minimum LR for cosine

    # Warm restart parameters
    T_0: int = 50                      # Initial restart period
    T_mult: int = 2                    # Period multiplication factor

    # Performance-based parameters
    patience: int = 20                 # Epochs to wait before LR reduction
    threshold: float = 0.01            # Minimum improvement threshold
    reduction_factor: float = 0.5      # Factor to reduce LR by
    cooldown: int = 10                 # Epochs to wait after LR reduction

    # Warm-up parameters
    warmup_epochs: int = 0
    warmup_start_lr: float = 0.0001


class LearningRateScheduler:
    """
    Advanced learning rate scheduler with multiple strategies.

    Supports various scheduling algorithms and can adapt based
    on training progress and performance metrics.
    """

    def __init__(self, config: ScheduleConfig):
        """
        Initialize scheduler.

        Args:
            config: Scheduling configuration
        """
        self.config = config
        self.current_lr = config.initial_lr
        self.epoch = 0

        # Performance tracking for adaptive scheduling
        self.best_performance = 0.0
        self.epochs_since_improvement = 0
        self.in_cooldown = 0

        # Warm restart tracking
        self.restart_cycle = 0
        self.epochs_in_cycle = 0
        self.current_T = config.T_0

        # History
        self.lr_history: List[float] = []
        self.performance_history: List[float] = []

    def step(self, epoch: Optional[int] = None,
             performance_metric: Optional[float] = None) -> float:
        """
        Update learning rate for next epoch.

        Args:
            epoch: Current epoch (if None, uses internal counter)
            performance_metric: Current performance for adaptive scheduling

        Returns:
            New learning rate
        """
        if epoch is not None:
            self.epoch = epoch
        else:
            self.epoch += 1

        # Handle warm-up phase
        if self.epoch <= self.config.warmup_epochs:
            self.current_lr = self._warmup_lr(self.epoch)
        else:
            # Apply main scheduling strategy
            adjusted_epoch = self.epoch - self.config.warmup_epochs
            self.current_lr = self._compute_lr(adjusted_epoch, performance_metric)

        # Record history
        self.lr_history.append(self.current_lr)
        if performance_metric is not None:
            self.performance_history.append(performance_metric)
            self._update_performance_tracking(performance_metric)

        return self.current_lr

    def _warmup_lr(self, epoch: int) -> float:
        """Compute learning rate during warm-up phase"""
        if self.config.warmup_epochs == 0:
            return self.config.initial_lr

        alpha = epoch / self.config.warmup_epochs
        return self.config.warmup_start_lr + alpha * (self.config.initial_lr - self.config.warmup_start_lr)

    def _compute_lr(self, epoch: int, performance_metric: Optional[float] = None) -> float:
        """Compute learning rate based on schedule type"""
        if self.config.schedule_type == ScheduleType.CONSTANT:
            return self.config.initial_lr

        elif self.config.schedule_type == ScheduleType.LINEAR_DECAY:
            return self._linear_decay(epoch)

        elif self.config.schedule_type == ScheduleType.EXPONENTIAL_DECAY:
            return self._exponential_decay(epoch)

        elif self.config.schedule_type == ScheduleType.COSINE_ANNEALING:
            return self._cosine_annealing(epoch)

        elif self.config.schedule_type == ScheduleType.COSINE_WARM_RESTARTS:
            return self._cosine_warm_restarts(epoch)

        elif self.config.schedule_type == ScheduleType.STEP_DECAY:
            return self._step_decay(epoch)

        elif self.config.schedule_type == ScheduleType.POLYNOMIAL_DECAY:
            return self._polynomial_decay(epoch)

        elif self.config.schedule_type == ScheduleType.ADAPTIVE_PERFORMANCE:
            return self._adaptive_performance(performance_metric)

        elif self.config.schedule_type == ScheduleType.CURRICULUM_BASED:
            return self._curriculum_based(epoch)

        else:
            raise ValueError(f"Unknown schedule type: {self.config.schedule_type}")

    def _linear_decay(self, epoch: int) -> float:
        """Linear decay from initial to final LR"""
        if epoch >= self.config.T_max:
            return self.config.final_lr

        alpha = epoch / self.config.T_max
        return self.config.initial_lr * (1 - alpha) + self.config.final_lr * alpha

    def _exponential_decay(self, epoch: int) -> float:
        """Exponential decay"""
        return self.config.initial_lr * (self.config.decay_rate ** epoch)

    def _cosine_annealing(self, epoch: int) -> float:
        """Cosine annealing schedule"""
        eta_min = self.config.eta_min
        eta_max = self.config.initial_lr
        T_max = self.config.T_max

        return eta_min + (eta_max - eta_min) * (1 + math.cos(math.pi * epoch / T_max)) / 2

    def _cosine_warm_restarts(self, epoch: int) -> float:
        """Cosine annealing with warm restarts (SGDR)"""
        # Update cycle tracking
        if self.epochs_in_cycle >= self.current_T:
            self.restart_cycle += 1
            self.epochs_in_cycle = 0
            self.current_T *= self.config.T_mult

        self.epochs_in_cycle += 1

        # Compute cosine annealing within current cycle
        eta_min = self.config.eta_min
        eta_max = self.config.initial_lr

        return eta_min + (eta_max - eta_min) * (
            1 + math.cos(math.pi * self.epochs_in_cycle / self.current_T)
        ) / 2

    def _step_decay(self, epoch: int) -> float:
        """Step-wise decay at fixed intervals"""
        steps = epoch // self.config.decay_steps
        return self.config.initial_lr * (self.config.decay_rate ** steps)

    def _polynomial_decay(self, epoch: int) -> float:
        """Polynomial decay"""
        if epoch >= self.config.T_max:
            return self.config.final_lr

        alpha = epoch / self.config.T_max
        return self.config.final_lr + (self.config.initial_lr - self.config.final_lr) * (
            (1 - alpha) ** self.config.polynomial_power
        )

    def _adaptive_performance(self, performance_metric: Optional[float] = None) -> float:
        """Performance-based adaptive scheduling (ReduceLROnPlateau style)"""
        if performance_metric is None:
            return self.current_lr

        # Check for improvement
        if performance_metric > self.best_performance + self.config.threshold:
            self.best_performance = performance_metric
            self.epochs_since_improvement = 0
            self.in_cooldown = 0
        else:
            self.epochs_since_improvement += 1

        # Reduce LR if no improvement and not in cooldown
        if (self.epochs_since_improvement >= self.config.patience and
            self.in_cooldown == 0):
            self.current_lr = self.current_lr * self.config.reduction_factor
            self.current_lr = max(self.current_lr, self.config.final_lr)  # Don't go below minimum

            self.in_cooldown = self.config.cooldown
            self.epochs_since_improvement = 0

        # Update cooldown
        if self.in_cooldown > 0:
            self.in_cooldown -= 1

        return self.current_lr

    def _curriculum_based(self, epoch: int) -> float:
        """Curriculum-based scheduling (higher LR for easier phases)"""
        # This would need to be integrated with training phase information
        # For now, implement a simple three-phase schedule
        if epoch < 100:  # Simple phase
            return self.config.initial_lr
        elif epoch < 300:  # Intermediate phase
            return self.config.initial_lr * 0.5
        else:  # Advanced phase
            return self.config.initial_lr * 0.1

    def _update_performance_tracking(self, performance_metric: float):
        """Update internal performance tracking"""
        if len(self.performance_history) == 1:  # First metric
            self.best_performance = performance_metric

    def get_lr(self) -> float:
        """Get current learning rate"""
        return self.current_lr

    def state_dict(self) -> Dict[str, Any]:
        """Get scheduler state for checkpointing"""
        return {
            'epoch': self.epoch,
            'current_lr': self.current_lr,
            'best_performance': self.best_performance,
            'epochs_since_improvement': self.epochs_since_improvement,
            'in_cooldown': self.in_cooldown,
            'restart_cycle': self.restart_cycle,
            'epochs_in_cycle': self.epochs_in_cycle,
            'current_T': self.current_T,
            'lr_history': self.lr_history,
            'performance_history': self.performance_history
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load scheduler state from checkpoint"""
        self.epoch = state_dict['epoch']
        self.current_lr = state_dict['current_lr']
        self.best_performance = state_dict['best_performance']
        self.epochs_since_improvement = state_dict['epochs_since_improvement']
        self.in_cooldown = state_dict['in_cooldown']
        self.restart_cycle = state_dict['restart_cycle']
        self.epochs_in_cycle = state_dict['epochs_in_cycle']
        self.current_T = state_dict['current_T']
        self.lr_history = state_dict['lr_history']
        self.performance_history = state_dict['performance_history']


class CombinedScheduler:
    """
    Combines multiple scheduling strategies and chooses the best one
    based on training progress and convergence analysis.
    """

    def __init__(self, schedulers: List[LearningRateScheduler],
                 selection_strategy: str = "performance"):
        """
        Initialize combined scheduler.

        Args:
            schedulers: List of schedulers to combine
            selection_strategy: How to select between schedulers
        """
        self.schedulers = schedulers
        self.selection_strategy = selection_strategy
        self.active_scheduler_idx = 0

        # Performance tracking for each scheduler
        self.scheduler_performance: List[List[float]] = [[] for _ in schedulers]
        self.scheduler_usage: List[int] = [0 for _ in schedulers]

    def step(self, epoch: int, performance_metric: Optional[float] = None,
             convergence_report: Optional[ConvergenceReport] = None) -> float:
        """
        Update learning rate using the best scheduler.

        Args:
            epoch: Current epoch
            performance_metric: Current performance metric
            convergence_report: Convergence analysis report

        Returns:
            New learning rate
        """
        # Select best scheduler
        if self.selection_strategy == "performance" and len(self.scheduler_performance[0]) > 10:
            self._select_best_scheduler()
        elif self.selection_strategy == "convergence" and convergence_report is not None:
            self._select_scheduler_by_convergence(convergence_report)

        # Update all schedulers to keep them in sync
        learning_rates = []
        for i, scheduler in enumerate(self.schedulers):
            lr = scheduler.step(epoch, performance_metric)
            learning_rates.append(lr)

            # Track performance for this scheduler if it was active
            if i == self.active_scheduler_idx and performance_metric is not None:
                self.scheduler_performance[i].append(performance_metric)

        self.scheduler_usage[self.active_scheduler_idx] += 1
        return learning_rates[self.active_scheduler_idx]

    def _select_best_scheduler(self):
        """Select scheduler with best recent performance"""
        if len(self.scheduler_performance[0]) < 10:
            return

        # Calculate recent performance for each scheduler
        recent_performance = []
        for perf_history in self.scheduler_performance:
            if len(perf_history) >= 5:
                recent_avg = np.mean(perf_history[-5:])
                recent_performance.append(recent_avg)
            else:
                recent_performance.append(0.0)

        # Select best performing scheduler
        if recent_performance:
            self.active_scheduler_idx = np.argmax(recent_performance)

    def _select_scheduler_by_convergence(self, convergence_report: ConvergenceReport):
        """Select scheduler based on convergence status"""
        status = convergence_report.status

        # Simple heuristic: different schedulers for different convergence states
        if status == ConvergenceStatus.IMPROVING:
            # Use cosine annealing for smooth improvement
            for i, scheduler in enumerate(self.schedulers):
                if scheduler.config.schedule_type == ScheduleType.COSINE_ANNEALING:
                    self.active_scheduler_idx = i
                    break

        elif status in [ConvergenceStatus.PLATEAU, ConvergenceStatus.CONVERGED]:
            # Use adaptive performance for plateau breaking
            for i, scheduler in enumerate(self.schedulers):
                if scheduler.config.schedule_type == ScheduleType.ADAPTIVE_PERFORMANCE:
                    self.active_scheduler_idx = i
                    break

        elif status == ConvergenceStatus.OSCILLATING:
            # Use exponential decay for stabilization
            for i, scheduler in enumerate(self.schedulers):
                if scheduler.config.schedule_type == ScheduleType.EXPONENTIAL_DECAY:
                    self.active_scheduler_idx = i
                    break

        elif status == ConvergenceStatus.DIVERGING:
            # Use aggressive decay
            for i, scheduler in enumerate(self.schedulers):
                if scheduler.config.schedule_type == ScheduleType.STEP_DECAY:
                    self.active_scheduler_idx = i
                    break

    def get_lr(self) -> float:
        """Get current learning rate from active scheduler"""
        return self.schedulers[self.active_scheduler_idx].get_lr()

    def get_active_scheduler_name(self) -> str:
        """Get name of currently active scheduler"""
        return self.schedulers[self.active_scheduler_idx].config.schedule_type.value

    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get statistics about scheduler usage"""
        total_usage = sum(self.scheduler_usage)
        if total_usage == 0:
            return {}

        stats = {}
        for i, scheduler in enumerate(self.schedulers):
            name = scheduler.config.schedule_type.value
            usage_pct = self.scheduler_usage[i] / total_usage * 100
            avg_performance = (np.mean(self.scheduler_performance[i])
                             if self.scheduler_performance[i] else 0.0)

            stats[name] = {
                'usage_percentage': usage_pct,
                'average_performance': avg_performance,
                'total_epochs': self.scheduler_usage[i]
            }

        return stats


def create_default_scheduler(schedule_type: ScheduleType = ScheduleType.COSINE_ANNEALING,
                           **kwargs) -> LearningRateScheduler:
    """
    Create a scheduler with sensible defaults.

    Args:
        schedule_type: Type of schedule to create
        **kwargs: Override default parameters

    Returns:
        Configured scheduler
    """
    default_params = {
        'schedule_type': schedule_type,
        'initial_lr': 0.01,
        'final_lr': 0.0001,
        'T_max': 200,
        'warmup_epochs': 10
    }

    # Update with provided parameters
    default_params.update(kwargs)

    config = ScheduleConfig(**default_params)
    return LearningRateScheduler(config)


def create_multi_scheduler() -> CombinedScheduler:
    """Create a combined scheduler with multiple strategies"""
    schedulers = [
        create_default_scheduler(ScheduleType.COSINE_ANNEALING),
        create_default_scheduler(ScheduleType.ADAPTIVE_PERFORMANCE),
        create_default_scheduler(ScheduleType.EXPONENTIAL_DECAY),
        create_default_scheduler(ScheduleType.COSINE_WARM_RESTARTS)
    ]

    return CombinedScheduler(schedulers, selection_strategy="convergence")


if __name__ == "__main__":
    # Test scheduling system
    print("Testing Neural-Klotski Learning Rate Scheduling System...")

    # Test individual schedulers
    print("\nTesting individual schedulers:")

    schedules_to_test = [
        ScheduleType.COSINE_ANNEALING,
        ScheduleType.ADAPTIVE_PERFORMANCE,
        ScheduleType.COSINE_WARM_RESTARTS
    ]

    for schedule_type in schedules_to_test:
        print(f"\nTesting {schedule_type.value}:")
        scheduler = create_default_scheduler(schedule_type, T_max=50)

        learning_rates = []
        performance = 0.1

        for epoch in range(60):
            # Simulate improving then plateauing performance
            if epoch < 30:
                performance += 0.01
            else:
                performance += 0.001

            lr = scheduler.step(epoch, performance)
            learning_rates.append(lr)

            if epoch % 10 == 0:
                print(f"  Epoch {epoch:2d}: LR={lr:.6f}, Perf={performance:.3f}")

    # Test combined scheduler
    print(f"\nTesting combined scheduler:")
    combined = create_multi_scheduler()

    for epoch in range(50):
        # Mock convergence report
        if epoch < 20:
            status = ConvergenceStatus.IMPROVING
        elif epoch < 35:
            status = ConvergenceStatus.PLATEAU
        else:
            status = ConvergenceStatus.CONVERGED

        from neural_klotski.training.monitoring import ConvergenceReport
        mock_report = ConvergenceReport(
            status=status,
            confidence=0.8,
            epochs_since_improvement=epoch % 10,
            current_trend=0.01 if epoch < 20 else 0.001,
            smoothed_accuracy=0.5 + epoch * 0.01,
            convergence_score=min(1.0, epoch / 50),
            recommended_action="Continue"
        )

        lr = combined.step(epoch, 0.5 + epoch * 0.01, mock_report)

        if epoch % 10 == 0:
            active_name = combined.get_active_scheduler_name()
            print(f"  Epoch {epoch:2d}: LR={lr:.6f}, Active={active_name}, Status={status.value}")

    # Usage statistics
    usage_stats = combined.get_usage_statistics()
    print(f"\nUsage statistics:")
    for scheduler_name, stats in usage_stats.items():
        print(f"  {scheduler_name}: {stats['usage_percentage']:.1f}% usage")

    print("Scheduling system test completed!")