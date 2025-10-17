"""
Performance Monitoring and Convergence Detection for Neural-Klotski Training.

Provides real-time monitoring, convergence analysis, and adaptive training
control based on learning progress.
"""

from typing import List, Dict, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import time
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from neural_klotski.training.trainer import TrainingMetrics, TrainingPhase


class ConvergenceStatus(Enum):
    """Status of convergence detection"""
    IMPROVING = "improving"           # Still making progress
    PLATEAU = "plateau"              # Performance has plateaued
    CONVERGED = "converged"          # Fully converged
    DIVERGING = "diverging"          # Performance is getting worse
    OSCILLATING = "oscillating"      # Performance is oscillating


@dataclass
class ConvergenceConfig:
    """Configuration for convergence detection"""
    # Convergence thresholds
    improvement_threshold: float = 0.01     # Minimum improvement to consider progress
    plateau_patience: int = 50              # Epochs without improvement for plateau
    convergence_patience: int = 100         # Epochs for full convergence
    divergence_threshold: float = 0.05      # Performance drop threshold for divergence

    # Statistical analysis
    smoothing_window: int = 10              # Window size for smoothing metrics
    trend_window: int = 20                  # Window size for trend analysis
    oscillation_threshold: float = 0.02    # Variance threshold for oscillation detection

    # Early stopping
    early_stopping_enabled: bool = True
    min_epochs_before_stopping: int = 100


@dataclass
class PerformanceSnapshot:
    """Snapshot of performance at a point in time"""
    epoch: int
    timestamp: float
    accuracy: float
    error: float
    confidence: float
    learning_rate: float
    phase: TrainingPhase

    # Derived metrics
    accuracy_trend: Optional[float] = None
    error_trend: Optional[float] = None
    convergence_score: Optional[float] = None


@dataclass
class ConvergenceReport:
    """Report on convergence analysis"""
    status: ConvergenceStatus
    confidence: float                    # 0-1 confidence in the assessment
    epochs_since_improvement: int
    current_trend: float                # Positive = improving, negative = declining
    smoothed_accuracy: float
    convergence_score: float            # 0-1 score indicating convergence

    # Recommendations
    recommended_action: str
    estimated_epochs_to_convergence: Optional[int] = None


class PerformanceMonitor:
    """
    Real-time performance monitoring system.

    Tracks training progress, detects convergence patterns,
    and provides adaptive training recommendations.
    """

    def __init__(self, config: ConvergenceConfig):
        """
        Initialize performance monitor.

        Args:
            config: Convergence detection configuration
        """
        self.config = config

        # Performance history
        self.snapshots: List[PerformanceSnapshot] = []
        self.best_accuracy = 0.0
        self.best_epoch = 0
        self.epochs_since_improvement = 0

        # Convergence tracking
        self.current_status = ConvergenceStatus.IMPROVING
        self.convergence_history: List[ConvergenceReport] = []

        # Alerts and callbacks
        self.alert_callbacks: List[Callable[[ConvergenceReport], None]] = []

    def add_alert_callback(self, callback: Callable[[ConvergenceReport], None]):
        """Add callback for convergence alerts"""
        self.alert_callbacks.append(callback)

    def record_metrics(self, metrics: TrainingMetrics) -> PerformanceSnapshot:
        """
        Record new training metrics.

        Args:
            metrics: Current training metrics

        Returns:
            Performance snapshot
        """
        snapshot = PerformanceSnapshot(
            epoch=metrics.epoch,
            timestamp=time.time(),
            accuracy=metrics.accuracy,
            error=metrics.average_error,
            confidence=metrics.confidence,
            learning_rate=metrics.learning_rate,
            phase=metrics.phase
        )

        # Calculate derived metrics
        if len(self.snapshots) >= self.config.trend_window:
            snapshot.accuracy_trend = self._calculate_trend('accuracy')
            snapshot.error_trend = self._calculate_trend('error')

        snapshot.convergence_score = self._calculate_convergence_score()

        self.snapshots.append(snapshot)

        # Update best performance tracking
        if snapshot.accuracy > self.best_accuracy:
            self.best_accuracy = snapshot.accuracy
            self.best_epoch = snapshot.epoch
            self.epochs_since_improvement = 0
        else:
            self.epochs_since_improvement += 1

        return snapshot

    def analyze_convergence(self) -> ConvergenceReport:
        """
        Analyze current convergence status.

        Returns:
            Convergence report with recommendations
        """
        if len(self.snapshots) < self.config.smoothing_window:
            return ConvergenceReport(
                status=ConvergenceStatus.IMPROVING,
                confidence=0.0,
                epochs_since_improvement=self.epochs_since_improvement,
                current_trend=0.0,
                smoothed_accuracy=self.snapshots[-1].accuracy if self.snapshots else 0.0,
                convergence_score=0.0,
                recommended_action="Continue training (insufficient data)"
            )

        # Calculate smoothed metrics
        recent_snapshots = self.snapshots[-self.config.smoothing_window:]
        smoothed_accuracy = np.mean([s.accuracy for s in recent_snapshots])
        accuracy_variance = np.var([s.accuracy for s in recent_snapshots])

        # Calculate trend
        current_trend = self._calculate_trend('accuracy')
        error_trend = self._calculate_trend('error')

        # Determine convergence status
        status = self._determine_convergence_status(current_trend, accuracy_variance)

        # Calculate confidence in assessment
        confidence = self._calculate_assessment_confidence(status)

        # Generate recommendations
        recommended_action, estimated_epochs = self._generate_recommendations(status, current_trend)

        report = ConvergenceReport(
            status=status,
            confidence=confidence,
            epochs_since_improvement=self.epochs_since_improvement,
            current_trend=current_trend,
            smoothed_accuracy=smoothed_accuracy,
            convergence_score=self._calculate_convergence_score(),
            recommended_action=recommended_action,
            estimated_epochs_to_convergence=estimated_epochs
        )

        self.convergence_history.append(report)
        self.current_status = status

        # Trigger alerts if status changed
        if len(self.convergence_history) > 1:
            previous_status = self.convergence_history[-2].status
            if status != previous_status:
                for callback in self.alert_callbacks:
                    callback(report)

        return report

    def _calculate_trend(self, metric: str) -> float:
        """Calculate trend for a specific metric"""
        if len(self.snapshots) < self.config.trend_window:
            return 0.0

        recent_snapshots = self.snapshots[-self.config.trend_window:]
        values = [getattr(s, metric) for s in recent_snapshots]
        epochs = [s.epoch for s in recent_snapshots]

        # Linear regression to find trend
        n = len(values)
        sum_x = sum(epochs)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(epochs, values))
        sum_x2 = sum(x * x for x in epochs)

        # Slope of linear regression line
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope

    def _calculate_convergence_score(self) -> float:
        """Calculate overall convergence score (0-1)"""
        if len(self.snapshots) < self.config.smoothing_window:
            return 0.0

        # Factors that indicate convergence
        factors = []

        # 1. Stability (low variance in recent performance)
        recent_accuracies = [s.accuracy for s in self.snapshots[-self.config.smoothing_window:]]
        variance = np.var(recent_accuracies)
        stability_score = max(0, 1 - variance / 0.1)  # Normalize to 0-1
        factors.append(stability_score)

        # 2. Trend (minimal improvement trend)
        trend = abs(self._calculate_trend('accuracy'))
        trend_score = max(0, 1 - trend / self.config.improvement_threshold)
        factors.append(trend_score)

        # 3. Time since improvement
        patience_score = min(1.0, self.epochs_since_improvement / self.config.convergence_patience)
        factors.append(patience_score)

        # 4. Absolute performance level
        performance_score = min(1.0, self.best_accuracy)  # Assumes max accuracy is 1.0
        factors.append(performance_score)

        # Weighted average
        weights = [0.3, 0.3, 0.3, 0.1]  # Emphasize stability and trend
        return sum(f * w for f, w in zip(factors, weights))

    def _determine_convergence_status(self, trend: float, variance: float) -> ConvergenceStatus:
        """Determine convergence status based on trends and variance"""
        # Check for divergence first
        if trend < -self.config.divergence_threshold:
            return ConvergenceStatus.DIVERGING

        # Check for oscillation
        if variance > self.config.oscillation_threshold:
            return ConvergenceStatus.OSCILLATING

        # Check for convergence/plateau
        if self.epochs_since_improvement >= self.config.convergence_patience:
            return ConvergenceStatus.CONVERGED
        elif self.epochs_since_improvement >= self.config.plateau_patience:
            return ConvergenceStatus.PLATEAU

        # Check for improvement
        if trend > self.config.improvement_threshold:
            return ConvergenceStatus.IMPROVING

        # Default to plateau if trend is small but positive
        return ConvergenceStatus.PLATEAU

    def _calculate_assessment_confidence(self, status: ConvergenceStatus) -> float:
        """Calculate confidence in convergence assessment"""
        if len(self.snapshots) < self.config.trend_window:
            return 0.0

        # Base confidence on data quantity and consistency
        data_quantity_factor = min(1.0, len(self.snapshots) / (self.config.trend_window * 2))

        # Consistency factor based on recent assessments
        if len(self.convergence_history) >= 5:
            recent_statuses = [r.status for r in self.convergence_history[-5:]]
            consistency = sum(1 for s in recent_statuses if s == status) / len(recent_statuses)
        else:
            consistency = 0.5

        return data_quantity_factor * consistency

    def _generate_recommendations(self, status: ConvergenceStatus,
                                trend: float) -> Tuple[str, Optional[int]]:
        """Generate training recommendations based on status"""
        recommendations = {
            ConvergenceStatus.IMPROVING: (
                "Continue training - performance is improving",
                None
            ),
            ConvergenceStatus.PLATEAU: (
                "Consider reducing learning rate or changing training strategy",
                int(self.config.plateau_patience * 0.5)
            ),
            ConvergenceStatus.CONVERGED: (
                "Training complete - performance has converged",
                0
            ),
            ConvergenceStatus.DIVERGING: (
                "Reduce learning rate immediately - performance is degrading",
                None
            ),
            ConvergenceStatus.OSCILLATING: (
                "Reduce learning rate to stabilize training",
                int(self.config.smoothing_window * 2)
            )
        }

        return recommendations.get(status, ("Continue training", None))

    def should_stop_early(self) -> bool:
        """Determine if training should stop early"""
        if not self.config.early_stopping_enabled:
            return False

        if len(self.snapshots) < self.config.min_epochs_before_stopping:
            return False

        # Stop if converged with high confidence
        if (self.current_status == ConvergenceStatus.CONVERGED and
            len(self.convergence_history) > 0 and
            self.convergence_history[-1].confidence > 0.8):
            return True

        # Stop if diverging severely
        if (self.current_status == ConvergenceStatus.DIVERGING and
            self.epochs_since_improvement > self.config.plateau_patience):
            return True

        return False

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        if not self.snapshots:
            return {"error": "No performance data available"}

        latest = self.snapshots[-1]
        latest_report = self.convergence_history[-1] if self.convergence_history else None

        return {
            'current_performance': {
                'epoch': latest.epoch,
                'accuracy': latest.accuracy,
                'error': latest.error,
                'confidence': latest.confidence,
                'phase': latest.phase.value
            },
            'best_performance': {
                'accuracy': self.best_accuracy,
                'epoch': self.best_epoch,
                'epochs_since': self.epochs_since_improvement
            },
            'convergence_analysis': {
                'status': latest_report.status.value if latest_report else "unknown",
                'confidence': latest_report.confidence if latest_report else 0.0,
                'convergence_score': latest_report.convergence_score if latest_report else 0.0,
                'trend': latest_report.current_trend if latest_report else 0.0
            },
            'recommendations': {
                'action': latest_report.recommended_action if latest_report else "Continue training",
                'should_stop': self.should_stop_early(),
                'estimated_epochs_remaining': latest_report.estimated_epochs_to_convergence if latest_report else None
            },
            'statistics': {
                'total_epochs': len(self.snapshots),
                'training_duration': self.snapshots[-1].timestamp - self.snapshots[0].timestamp if len(self.snapshots) > 1 else 0
            }
        }


class AdaptiveLearningController:
    """
    Adaptive learning rate controller based on performance monitoring.

    Automatically adjusts learning rate based on convergence patterns
    and training progress.
    """

    def __init__(self, initial_learning_rate: float = 0.01,
                 min_learning_rate: float = 0.0001,
                 max_learning_rate: float = 0.1):
        """
        Initialize adaptive controller.

        Args:
            initial_learning_rate: Starting learning rate
            min_learning_rate: Minimum allowed learning rate
            max_learning_rate: Maximum allowed learning rate
        """
        self.initial_lr = initial_learning_rate
        self.min_lr = min_learning_rate
        self.max_lr = max_learning_rate
        self.current_lr = initial_learning_rate

        # Adaptation parameters
        self.decay_factor = 0.8
        self.boost_factor = 1.2
        self.patience_threshold = 20

    def update_learning_rate(self, convergence_report: ConvergenceReport) -> float:
        """
        Update learning rate based on convergence status.

        Args:
            convergence_report: Current convergence analysis

        Returns:
            New learning rate
        """
        status = convergence_report.status

        if status == ConvergenceStatus.IMPROVING:
            # Slightly boost if consistently improving
            if convergence_report.current_trend > 0.02:
                self.current_lr = min(self.current_lr * self.boost_factor, self.max_lr)

        elif status == ConvergenceStatus.PLATEAU:
            # Reduce learning rate to escape plateau
            self.current_lr = max(self.current_lr * self.decay_factor, self.min_lr)

        elif status == ConvergenceStatus.DIVERGING:
            # Aggressive reduction for divergence
            self.current_lr = max(self.current_lr * (self.decay_factor ** 2), self.min_lr)

        elif status == ConvergenceStatus.OSCILLATING:
            # Moderate reduction to stabilize
            self.current_lr = max(self.current_lr * self.decay_factor, self.min_lr)

        # Converged status doesn't change learning rate

        return self.current_lr

    def reset(self):
        """Reset controller to initial state"""
        self.current_lr = self.initial_lr


if __name__ == "__main__":
    # Test monitoring system
    print("Testing Neural-Klotski Performance Monitoring System...")

    config = ConvergenceConfig()
    monitor = PerformanceMonitor(config)
    controller = AdaptiveLearningController()

    # Add alert callback
    def on_convergence_change(report: ConvergenceReport):
        print(f"Convergence status changed to: {report.status.value}")

    monitor.add_alert_callback(on_convergence_change)

    # Simulate training progress
    import random

    base_accuracy = 0.1
    for epoch in range(200):
        # Simulate improving then plateauing performance
        if epoch < 100:
            base_accuracy += random.uniform(0.005, 0.015)  # Improving
        else:
            base_accuracy += random.uniform(-0.005, 0.005)  # Plateauing

        # Add noise
        accuracy = base_accuracy + random.uniform(-0.02, 0.02)
        accuracy = max(0, min(1, accuracy))

        # Create mock metrics
        from neural_klotski.training.trainer import TrainingMetrics, TrainingPhase
        metrics = TrainingMetrics(
            epoch=epoch,
            phase=TrainingPhase.SIMPLE,
            accuracy=accuracy,
            average_error=1.0 - accuracy,
            confidence=accuracy,
            learning_rate=controller.current_lr
        )

        # Record and analyze
        snapshot = monitor.record_metrics(metrics)

        if epoch % 10 == 0:
            report = monitor.analyze_convergence()
            new_lr = controller.update_learning_rate(report)

            print(f"Epoch {epoch:3d}: Acc={accuracy:.3f}, "
                  f"Status={report.status.value}, "
                  f"LR={new_lr:.4f}")

    # Final summary
    summary = monitor.get_performance_summary()
    print(f"\nFinal Summary:")
    print(f"Best accuracy: {summary['best_performance']['accuracy']:.3f}")
    print(f"Convergence status: {summary['convergence_analysis']['status']}")
    print(f"Should stop: {summary['recommendations']['should_stop']}")

    print("Monitoring system test completed!")