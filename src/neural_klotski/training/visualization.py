"""
Training Visualization and Progress Tracking for Neural-Klotski.

Provides real-time visualization of training progress, network state,
and performance metrics.
"""

from typing import List, Dict, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
import sys
import os
import time
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from neural_klotski.training.trainer import TrainingMetrics, TrainingPhase
from neural_klotski.training.monitoring import ConvergenceReport, PerformanceSnapshot
from neural_klotski.core.network import NeuralKlotskiNetwork


@dataclass
class VisualizationConfig:
    """Configuration for training visualization"""
    output_directory: str = "training_logs"
    save_plots: bool = True
    show_live_plots: bool = False
    update_frequency: int = 10  # Updates per epoch

    # Plot configuration
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 100
    style: str = "default"  # matplotlib style

    # Data retention
    max_history_length: int = 10000
    save_raw_data: bool = True


class TrainingVisualizer:
    """
    Real-time training visualization system.

    Creates plots and dashboards for monitoring Neural-Klotski training
    progress including performance metrics, convergence analysis, and
    network state visualization.
    """

    def __init__(self, config: VisualizationConfig):
        """
        Initialize visualizer.

        Args:
            config: Visualization configuration
        """
        self.config = config

        # Create output directory
        self.output_path = Path(config.output_directory)
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Data storage
        self.metrics_history: List[TrainingMetrics] = []
        self.convergence_history: List[ConvergenceReport] = []
        self.learning_rate_history: List[float] = []
        self.epoch_times: List[float] = []

        # Visualization state
        self.session_start_time = time.time()
        self.last_plot_time = 0
        self.plot_counter = 0

        # Try to import matplotlib (optional dependency)
        self.has_matplotlib = self._check_matplotlib()

    def _check_matplotlib(self) -> bool:
        """Check if matplotlib is available"""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            return True
        except ImportError:
            print("Warning: matplotlib not available. Plots will not be generated.")
            return False

    def record_epoch(self, metrics: TrainingMetrics, convergence_report: Optional[ConvergenceReport] = None,
                    learning_rate: Optional[float] = None, epoch_duration: Optional[float] = None):
        """
        Record metrics for an epoch.

        Args:
            metrics: Training metrics for the epoch
            convergence_report: Convergence analysis report
            learning_rate: Current learning rate
            epoch_duration: Time taken for the epoch
        """
        # Store data
        self.metrics_history.append(metrics)

        if convergence_report:
            self.convergence_history.append(convergence_report)

        if learning_rate is not None:
            self.learning_rate_history.append(learning_rate)

        if epoch_duration is not None:
            self.epoch_times.append(epoch_duration)

        # Trim history if too long
        if len(self.metrics_history) > self.config.max_history_length:
            self.metrics_history = self.metrics_history[-self.config.max_history_length:]
            self.convergence_history = self.convergence_history[-self.config.max_history_length:]
            self.learning_rate_history = self.learning_rate_history[-self.config.max_history_length:]
            self.epoch_times = self.epoch_times[-self.config.max_history_length:]

        # Update visualizations
        current_time = time.time()
        time_since_last_plot = current_time - self.last_plot_time

        should_update = (
            time_since_last_plot > (60 / self.config.update_frequency) or  # Time-based
            len(self.metrics_history) % 10 == 0  # Epoch-based
        )

        if should_update:
            self._update_visualizations()
            self.last_plot_time = current_time

    def _update_visualizations(self):
        """Update all visualizations"""
        if not self.has_matplotlib or not self.config.save_plots:
            return

        try:
            self._create_training_progress_plot()
            self._create_convergence_analysis_plot()
            self._create_learning_rate_plot()
            self._create_phase_progression_plot()

            if len(self.metrics_history) > 20:
                self._create_comprehensive_dashboard()

        except Exception as e:
            print(f"Warning: Failed to update visualizations: {e}")

    def _create_training_progress_plot(self):
        """Create training progress plot"""
        if not self.has_matplotlib:
            return

        import matplotlib.pyplot as plt

        if not self.metrics_history:
            return

        epochs = [m.epoch for m in self.metrics_history]
        accuracies = [m.accuracy for m in self.metrics_history]
        errors = [m.average_error for m in self.metrics_history]
        confidences = [m.confidence for m in self.metrics_history]

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=self.config.figure_size, dpi=self.config.dpi)

        # Accuracy plot
        ax1.plot(epochs, accuracies, 'b-', linewidth=2, label='Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Training Progress')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim(0, 1)

        # Error plot
        ax2.plot(epochs, errors, 'r-', linewidth=2, label='Average Error')
        ax2.set_ylabel('Average Error')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Confidence plot
        ax3.plot(epochs, confidences, 'g-', linewidth=2, label='Confidence')
        ax3.set_ylabel('Confidence')
        ax3.set_xlabel('Epoch')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(self.output_path / f'training_progress_{self.plot_counter:04d}.png',
                   bbox_inches='tight', dpi=self.config.dpi)
        plt.close()

    def _create_convergence_analysis_plot(self):
        """Create convergence analysis plot"""
        if not self.has_matplotlib or not self.convergence_history:
            return

        import matplotlib.pyplot as plt

        epochs = [r.epochs_since_improvement for r in self.convergence_history]
        trends = [r.current_trend for r in self.convergence_history]
        convergence_scores = [r.convergence_score for r in self.convergence_history]
        confidences = [r.confidence for r in self.convergence_history]

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=self.config.figure_size, dpi=self.config.dpi)

        # Epochs since improvement
        ax1.plot(range(len(epochs)), epochs, 'r-', linewidth=2)
        ax1.set_ylabel('Epochs Since Improvement')
        ax1.set_title('Convergence Analysis')
        ax1.grid(True, alpha=0.3)

        # Performance trend
        ax2.plot(range(len(trends)), trends, 'b-', linewidth=2)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_ylabel('Performance Trend')
        ax2.grid(True, alpha=0.3)

        # Convergence score and confidence
        ax3.plot(range(len(convergence_scores)), convergence_scores, 'g-', linewidth=2, label='Convergence Score')
        ax3.plot(range(len(confidences)), confidences, 'orange', linewidth=2, label='Confidence')
        ax3.set_ylabel('Score')
        ax3.set_xlabel('Analysis Step')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(self.output_path / f'convergence_analysis_{self.plot_counter:04d}.png',
                   bbox_inches='tight', dpi=self.config.dpi)
        plt.close()

    def _create_learning_rate_plot(self):
        """Create learning rate evolution plot"""
        if not self.has_matplotlib or not self.learning_rate_history:
            return

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.config.dpi)

        epochs = range(len(self.learning_rate_history))
        ax.semilogy(epochs, self.learning_rate_history, 'purple', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate (log scale)')
        ax.set_title('Learning Rate Schedule')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_path / f'learning_rate_{self.plot_counter:04d}.png',
                   bbox_inches='tight', dpi=self.config.dpi)
        plt.close()

    def _create_phase_progression_plot(self):
        """Create training phase progression plot"""
        if not self.has_matplotlib:
            return

        import matplotlib.pyplot as plt

        if not self.metrics_history:
            return

        epochs = [m.epoch for m in self.metrics_history]
        phases = [m.phase.value for m in self.metrics_history]
        accuracies = [m.accuracy for m in self.metrics_history]

        # Create phase color mapping
        phase_colors = {
            TrainingPhase.SIMPLE.value: 'lightblue',
            TrainingPhase.INTERMEDIATE.value: 'lightgreen',
            TrainingPhase.ADVANCED.value: 'lightcoral'
        }

        fig, ax = plt.subplots(figsize=(12, 6), dpi=self.config.dpi)

        # Plot accuracy with phase background colors
        ax.plot(epochs, accuracies, 'b-', linewidth=2, label='Accuracy')

        # Add phase background colors
        current_phase = phases[0] if phases else TrainingPhase.SIMPLE.value
        phase_start = 0

        for i, phase in enumerate(phases):
            if phase != current_phase or i == len(phases) - 1:
                # Phase transition or end
                phase_end = i if i < len(phases) - 1 else len(phases) - 1
                ax.axvspan(epochs[phase_start], epochs[phase_end],
                          alpha=0.3, color=phase_colors.get(current_phase, 'gray'),
                          label=f'Phase: {current_phase}' if current_phase not in [p for p in phases[:i]] else "")

                current_phase = phase
                phase_start = i

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Training Phase Progression')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(self.output_path / f'phase_progression_{self.plot_counter:04d}.png',
                   bbox_inches='tight', dpi=self.config.dpi)
        plt.close()

    def _create_comprehensive_dashboard(self):
        """Create comprehensive training dashboard"""
        if not self.has_matplotlib:
            return

        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(16, 12), dpi=self.config.dpi)

        # Create subplot grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Main accuracy plot
        ax1 = fig.add_subplot(gs[0, :2])
        epochs = [m.epoch for m in self.metrics_history]
        accuracies = [m.accuracy for m in self.metrics_history]
        ax1.plot(epochs, accuracies, 'b-', linewidth=2)
        ax1.set_title('Training Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.grid(True, alpha=0.3)

        # 2. Error evolution
        ax2 = fig.add_subplot(gs[1, 0])
        errors = [m.average_error for m in self.metrics_history]
        ax2.plot(epochs, errors, 'r-', linewidth=2)
        ax2.set_title('Average Error')
        ax2.set_ylabel('Error')
        ax2.grid(True, alpha=0.3)

        # 3. Learning rate
        ax3 = fig.add_subplot(gs[1, 1])
        if self.learning_rate_history:
            ax3.semilogy(range(len(self.learning_rate_history)), self.learning_rate_history, 'purple', linewidth=2)
        ax3.set_title('Learning Rate')
        ax3.grid(True, alpha=0.3)

        # 4. Convergence status
        ax4 = fig.add_subplot(gs[1, 2])
        if self.convergence_history:
            convergence_scores = [r.convergence_score for r in self.convergence_history]
            ax4.plot(range(len(convergence_scores)), convergence_scores, 'g-', linewidth=2)
        ax4.set_title('Convergence Score')
        ax4.set_ylim(0, 1)
        ax4.grid(True, alpha=0.3)

        # 5. Performance statistics
        ax5 = fig.add_subplot(gs[2, :])
        if len(self.metrics_history) > 1:
            recent_metrics = self.metrics_history[-min(100, len(self.metrics_history)):]
            recent_accuracies = [m.accuracy for m in recent_metrics]

            import numpy as np
            ax5.hist(recent_accuracies, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax5.axvline(np.mean(recent_accuracies), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(recent_accuracies):.3f}')
            ax5.set_title('Recent Accuracy Distribution (Last 100 Epochs)')
            ax5.set_xlabel('Accuracy')
            ax5.set_ylabel('Frequency')
            ax5.legend()
            ax5.grid(True, alpha=0.3)

        # 6. Summary statistics text
        ax6 = fig.add_subplot(gs[0, 2])
        ax6.axis('off')

        if self.metrics_history:
            latest = self.metrics_history[-1]
            best_acc = max(m.accuracy for m in self.metrics_history)
            avg_acc = sum(m.accuracy for m in self.metrics_history) / len(self.metrics_history)

            stats_text = f"""Training Summary

Current Epoch: {latest.epoch}
Current Accuracy: {latest.accuracy:.3f}
Best Accuracy: {best_acc:.3f}
Average Accuracy: {avg_acc:.3f}

Current Phase: {latest.phase.value}
Problems Solved: {latest.problems_solved}
Total Problems: {latest.total_problems}

Training Time: {time.time() - self.session_start_time:.1f}s
"""
            ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace')

        plt.suptitle('Neural-Klotski Training Dashboard', fontsize=16, fontweight='bold')
        plt.savefig(self.output_path / f'dashboard_{self.plot_counter:04d}.png',
                   bbox_inches='tight', dpi=self.config.dpi)
        plt.close()

        self.plot_counter += 1

    def save_training_log(self, filename: Optional[str] = None):
        """Save complete training log to JSON"""
        if not self.config.save_raw_data:
            return

        if filename is None:
            timestamp = int(time.time())
            filename = f"training_log_{timestamp}.json"

        log_data = {
            'session_info': {
                'start_time': self.session_start_time,
                'total_epochs': len(self.metrics_history),
                'final_accuracy': self.metrics_history[-1].accuracy if self.metrics_history else 0.0
            },
            'metrics_history': [
                {
                    'epoch': m.epoch,
                    'phase': m.phase.value,
                    'accuracy': m.accuracy,
                    'average_error': m.average_error,
                    'confidence': m.confidence,
                    'problems_solved': m.problems_solved,
                    'total_problems': m.total_problems
                }
                for m in self.metrics_history
            ],
            'convergence_history': [
                {
                    'status': r.status.value,
                    'confidence': r.confidence,
                    'epochs_since_improvement': r.epochs_since_improvement,
                    'current_trend': r.current_trend,
                    'convergence_score': r.convergence_score
                }
                for r in self.convergence_history
            ],
            'learning_rate_history': self.learning_rate_history,
            'epoch_times': self.epoch_times
        }

        log_path = self.output_path / filename
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)

        print(f"Training log saved to: {log_path}")

    def create_final_report(self) -> str:
        """Create final training report"""
        if not self.metrics_history:
            return "No training data available"

        latest = self.metrics_history[-1]
        best_accuracy = max(m.accuracy for m in self.metrics_history)
        best_epoch = next(i for i, m in enumerate(self.metrics_history) if m.accuracy == best_accuracy)

        report = f"""
Neural-Klotski Training Report
============================

Training Summary:
- Total Epochs: {latest.epoch}
- Final Accuracy: {latest.accuracy:.3f}
- Best Accuracy: {best_accuracy:.3f} (at epoch {best_epoch})
- Final Phase: {latest.phase.value}
- Total Training Time: {time.time() - self.session_start_time:.1f} seconds

Performance Statistics:
- Average Accuracy: {sum(m.accuracy for m in self.metrics_history) / len(self.metrics_history):.3f}
- Average Error: {sum(m.average_error for m in self.metrics_history) / len(self.metrics_history):.3f}
- Problems Solved: {latest.problems_solved}
- Total Problems Attempted: {latest.total_problems}

Convergence Analysis:
"""

        if self.convergence_history:
            final_convergence = self.convergence_history[-1]
            report += f"""- Final Status: {final_convergence.status.value}
- Convergence Score: {final_convergence.convergence_score:.3f}
- Assessment Confidence: {final_convergence.confidence:.3f}
"""

        return report

    def __del__(self):
        """Cleanup: save final training log"""
        try:
            if self.config.save_raw_data and self.metrics_history:
                self.save_training_log("final_training_log.json")
        except:
            pass  # Ignore cleanup errors


def create_console_progress_tracker(update_frequency: int = 10) -> Callable:
    """
    Create a simple console-based progress tracker.

    Args:
        update_frequency: How often to print updates

    Returns:
        Callback function for tracking progress
    """
    def track_progress(metrics: TrainingMetrics):
        if metrics.epoch % update_frequency == 0:
            print(f"Epoch {metrics.epoch:4d} | "
                  f"Phase: {metrics.phase.value:12s} | "
                  f"Acc: {metrics.accuracy:.3f} | "
                  f"Err: {metrics.average_error:.3f} | "
                  f"Conf: {metrics.confidence:.3f}")

    return track_progress


if __name__ == "__main__":
    # Test visualization system
    print("Testing Neural-Klotski Visualization System...")

    # Try to import numpy for test
    try:
        import numpy as np
    except ImportError:
        print("Warning: numpy not available for testing")
        np = None

    config = VisualizationConfig(
        output_directory="test_visualization",
        save_plots=True,
        update_frequency=5
    )

    visualizer = TrainingVisualizer(config)

    # Simulate training progress
    from neural_klotski.training.trainer import TrainingMetrics, TrainingPhase
    from neural_klotski.training.monitoring import ConvergenceReport, ConvergenceStatus

    for epoch in range(50):
        # Mock metrics
        metrics = TrainingMetrics(
            epoch=epoch,
            phase=TrainingPhase.SIMPLE if epoch < 20 else TrainingPhase.INTERMEDIATE,
            accuracy=0.1 + epoch * 0.015 + (0.02 if np else 0) * (np.random.random() - 0.5 if np else 0),
            average_error=max(0, 1.0 - (0.1 + epoch * 0.015)),
            confidence=0.5 + epoch * 0.01,
            problems_solved=epoch * 10,
            total_problems=epoch * 15
        )

        # Mock convergence report
        convergence = ConvergenceReport(
            status=ConvergenceStatus.IMPROVING if epoch < 30 else ConvergenceStatus.PLATEAU,
            confidence=0.8,
            epochs_since_improvement=max(0, epoch - 25),
            current_trend=0.01 if epoch < 30 else 0.001,
            smoothed_accuracy=metrics.accuracy,
            convergence_score=min(1.0, epoch / 40),
            recommended_action="Continue training"
        )

        learning_rate = 0.01 * (0.95 ** epoch)

        visualizer.record_epoch(metrics, convergence, learning_rate, 1.5)

    # Generate final report
    report = visualizer.create_final_report()
    print(report)

    # Save training log
    visualizer.save_training_log("test_training_log.json")

    print("Visualization system test completed!")
    print(f"Check {config.output_directory} for generated plots and logs.")