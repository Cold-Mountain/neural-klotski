"""
Performance Logging System for Neural-Klotski Training Visualization

Captures training metrics, convergence data, and performance statistics
in real-time for visualization during training sessions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
import time
import threading
import numpy as np
import json
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from neural_klotski.core.network import NeuralKlotskiNetwork
from neural_klotski.visualization.utils import PerformanceUtils


@dataclass
class TrainingMetrics:
    """Comprehensive training metrics for a single epoch/iteration"""
    timestamp: float
    epoch: int
    iteration: int

    # Error metrics
    total_error: float
    mean_squared_error: float
    classification_accuracy: float
    convergence_ratio: float

    # Learning dynamics
    learning_rate: float
    gradient_magnitude: float
    weight_change_magnitude: float
    plasticity_activity: float

    # Network state metrics
    average_activation: float
    activation_variance: float
    firing_rate: float
    signal_throughput: float

    # Dye system metrics
    total_dye_concentration: float
    dye_diffusion_rate: float
    enhancement_factor: float
    spatial_dye_distribution: Dict[str, float] = field(default_factory=dict)

    # Computational metrics
    iteration_time_ms: float
    memory_usage_mb: float
    cpu_utilization: float

    # Problem-specific metrics
    current_problem: Optional[Tuple[int, int, int]] = None  # (a, b, expected_sum)
    problem_solved: bool = False
    solution_time_ms: float = 0.0
    attempts_count: int = 0


@dataclass
class MetricsSnapshot:
    """Point-in-time snapshot of all training metrics"""
    timestamp: float
    training_session_id: str

    # Current training state
    current_metrics: TrainingMetrics

    # Historical averages (over last N iterations)
    avg_error_10: float = 0.0
    avg_error_100: float = 0.0
    avg_convergence_10: float = 0.0
    avg_convergence_100: float = 0.0

    # Training progress indicators
    total_problems_attempted: int = 0
    total_problems_solved: int = 0
    current_success_rate: float = 0.0

    # Learning curve data
    error_trend: List[float] = field(default_factory=list)
    convergence_trend: List[float] = field(default_factory=list)
    learning_rate_trend: List[float] = field(default_factory=list)

    # Dye dynamics over time
    dye_concentration_history: Dict[str, List[float]] = field(default_factory=dict)
    enhancement_history: List[float] = field(default_factory=list)

    # Performance statistics
    iterations_per_second: float = 0.0
    estimated_completion_time: Optional[float] = None
    training_efficiency_score: float = 0.0


@dataclass
class VisualizationTrainingState:
    """Complete training state optimized for visualization"""
    timestamp: float
    session_id: str

    # Current iteration state
    current_iteration: int
    total_iterations: int
    progress_percentage: float

    # Real-time metrics
    current_snapshot: MetricsSnapshot

    # Learning phase detection
    learning_phase: str  # "exploration", "convergence", "refinement", "completion"
    phase_stability: float  # 0.0 to 1.0
    phase_duration_seconds: float

    # Problem solving state
    current_problem_state: Dict[str, Any] = field(default_factory=dict)
    problem_history: List[Dict[str, Any]] = field(default_factory=list)

    # Visualization hints
    highlight_blocks: List[int] = field(default_factory=list)  # Blocks to highlight
    critical_wires: List[int] = field(default_factory=list)    # Important wire connections
    attention_regions: List[Tuple[float, float, float]] = field(default_factory=list)  # (x, y, radius)

    # System health
    training_health_score: float = 1.0  # 0.0 to 1.0
    anomaly_detected: bool = False
    anomaly_description: str = ""


class PerformanceLogger:
    """
    Real-time performance logging system for Neural-Klotski training.

    Captures comprehensive training metrics and provides real-time analysis
    for visualization components.
    """

    def __init__(self,
                 network: NeuralKlotskiNetwork,
                 session_id: Optional[str] = None,
                 log_to_file: bool = True,
                 log_directory: Optional[Path] = None):
        """
        Initialize performance logger.

        Args:
            network: Neural-Klotski network to monitor
            session_id: Unique identifier for training session
            log_to_file: Whether to save logs to disk
            log_directory: Directory for log files
        """
        self.network = network
        self.session_id = session_id or f"training_{int(time.time())}"
        self.log_to_file = log_to_file
        self.log_directory = log_directory or Path("./logs/training")

        # Create log directory
        if self.log_to_file:
            self.log_directory.mkdir(parents=True, exist_ok=True)
            self.log_file = self.log_directory / f"{self.session_id}.jsonl"

        # State tracking
        self.is_logging = False
        self.iteration_counter = 0
        self.start_time = time.time()

        # Metrics storage
        self.metrics_history: List[TrainingMetrics] = []
        self.snapshots: List[MetricsSnapshot] = []
        self.max_history_size = 10000

        # Performance monitoring
        self.performance_monitor = PerformanceUtils()

        # Threading for async logging
        self.logging_thread: Optional[threading.Thread] = None
        self.logging_lock = threading.Lock()
        self.should_stop_logging = threading.Event()

        # Callbacks for real-time data streaming
        self.metrics_callbacks: List[Callable[[TrainingMetrics], None]] = []
        self.snapshot_callbacks: List[Callable[[MetricsSnapshot], None]] = []
        self.state_callbacks: List[Callable[[VisualizationTrainingState], None]] = []

        # Learning phase detection
        self.current_phase = "exploration"
        self.phase_start_time = time.time()
        self.phase_stability_history: List[float] = []

        # Problem tracking
        self.current_problem: Optional[Tuple[int, int, int]] = None
        self.problem_start_time: float = 0.0
        self.problems_attempted = 0
        self.problems_solved = 0

        # Anomaly detection
        self.baseline_metrics: Dict[str, float] = {}
        self.anomaly_threshold = 3.0  # Standard deviations

    def start_logging(self) -> None:
        """Start performance logging"""
        if self.is_logging:
            return

        self.is_logging = True
        self.start_time = time.time()
        self.iteration_counter = 0
        self.should_stop_logging.clear()

        # Start async logging thread
        self.logging_thread = threading.Thread(target=self._logging_loop, daemon=True)
        self.logging_thread.start()

        print(f"🚀 Performance logging started for session: {self.session_id}")

    def stop_logging(self) -> None:
        """Stop performance logging"""
        if not self.is_logging:
            return

        self.is_logging = False
        self.should_stop_logging.set()

        if self.logging_thread and self.logging_thread.is_alive():
            self.logging_thread.join(timeout=2.0)

        # Final log entry
        if self.log_to_file and self.metrics_history:
            self._save_session_summary()

        print(f"🏁 Performance logging stopped. Session: {self.session_id}")

    def log_iteration(self,
                     epoch: int,
                     iteration: int,
                     error_metrics: Dict[str, float],
                     learning_metrics: Dict[str, float],
                     network_metrics: Dict[str, float]) -> TrainingMetrics:
        """
        Log metrics for a single training iteration.

        Args:
            epoch: Current epoch number
            iteration: Current iteration number
            error_metrics: Error and accuracy metrics
            learning_metrics: Learning rate, gradients, etc.
            network_metrics: Network state metrics

        Returns:
            Complete training metrics object
        """
        if not self.is_logging:
            return None

        timestamp = time.time()

        # Extract dye system metrics
        dye_metrics = self._extract_dye_metrics()

        # Extract computational metrics
        comp_metrics = self._extract_computational_metrics()

        # Create training metrics
        metrics = TrainingMetrics(
            timestamp=timestamp,
            epoch=epoch,
            iteration=iteration,

            # Error metrics
            total_error=error_metrics.get('total_error', 0.0),
            mean_squared_error=error_metrics.get('mse', 0.0),
            classification_accuracy=error_metrics.get('accuracy', 0.0),
            convergence_ratio=error_metrics.get('convergence', 0.0),

            # Learning dynamics
            learning_rate=learning_metrics.get('learning_rate', 0.0),
            gradient_magnitude=learning_metrics.get('gradient_magnitude', 0.0),
            weight_change_magnitude=learning_metrics.get('weight_change', 0.0),
            plasticity_activity=learning_metrics.get('plasticity_activity', 0.0),

            # Network state metrics
            average_activation=network_metrics.get('avg_activation', 0.0),
            activation_variance=network_metrics.get('activation_variance', 0.0),
            firing_rate=network_metrics.get('firing_rate', 0.0),
            signal_throughput=network_metrics.get('signal_throughput', 0.0),

            # Dye system metrics
            total_dye_concentration=dye_metrics['total_concentration'],
            dye_diffusion_rate=dye_metrics['diffusion_rate'],
            enhancement_factor=dye_metrics['enhancement_factor'],
            spatial_dye_distribution=dye_metrics['spatial_distribution'],

            # Computational metrics
            iteration_time_ms=comp_metrics['iteration_time_ms'],
            memory_usage_mb=comp_metrics['memory_usage_mb'],
            cpu_utilization=comp_metrics['cpu_utilization'],

            # Problem-specific metrics
            current_problem=self.current_problem,
            problem_solved=self._check_problem_solved(error_metrics),
            solution_time_ms=(timestamp - self.problem_start_time) * 1000 if self.current_problem else 0.0,
            attempts_count=self.problems_attempted
        )

        # Update iteration counter
        with self.logging_lock:
            self.iteration_counter += 1

            # Store metrics
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > self.max_history_size:
                self.metrics_history = self.metrics_history[-self.max_history_size//2:]

            # Detect anomalies
            self._detect_anomalies(metrics)

            # Update learning phase
            self._update_learning_phase(metrics)

        # Generate snapshot
        snapshot = self._generate_snapshot(metrics)

        # Generate visualization state
        vis_state = self._generate_visualization_state(metrics, snapshot)

        # Notify callbacks
        self._notify_callbacks(metrics, snapshot, vis_state)

        # Log to file
        if self.log_to_file:
            self._append_to_log_file(metrics)

        return metrics

    def set_current_problem(self, a: int, b: int, expected_sum: int) -> None:
        """Set the current problem being solved"""
        self.current_problem = (a, b, expected_sum)
        self.problem_start_time = time.time()
        self.problems_attempted += 1

    def mark_problem_solved(self) -> None:
        """Mark current problem as solved"""
        if self.current_problem:
            self.problems_solved += 1
            self.current_problem = None

    def register_metrics_callback(self, callback: Callable[[TrainingMetrics], None]) -> None:
        """Register callback for training metrics updates"""
        self.metrics_callbacks.append(callback)

    def register_snapshot_callback(self, callback: Callable[[MetricsSnapshot], None]) -> None:
        """Register callback for snapshot updates"""
        self.snapshot_callbacks.append(callback)

    def register_state_callback(self, callback: Callable[[VisualizationTrainingState], None]) -> None:
        """Register callback for visualization state updates"""
        self.state_callbacks.append(callback)

    def get_current_snapshot(self) -> Optional[MetricsSnapshot]:
        """Get most recent metrics snapshot"""
        with self.logging_lock:
            return self.snapshots[-1] if self.snapshots else None

    def get_metrics_history(self, last_n: Optional[int] = None) -> List[TrainingMetrics]:
        """Get training metrics history"""
        with self.logging_lock:
            if last_n:
                return self.metrics_history[-last_n:]
            return self.metrics_history.copy()

    def get_session_summary(self) -> Dict[str, Any]:
        """Get comprehensive session summary"""
        if not self.metrics_history:
            return {}

        duration = time.time() - self.start_time
        latest_metrics = self.metrics_history[-1]

        # Calculate averages
        errors = [m.total_error for m in self.metrics_history]
        convergence = [m.convergence_ratio for m in self.metrics_history]

        return {
            'session_id': self.session_id,
            'duration_seconds': duration,
            'total_iterations': len(self.metrics_history),
            'iterations_per_second': len(self.metrics_history) / duration if duration > 0 else 0,
            'problems_attempted': self.problems_attempted,
            'problems_solved': self.problems_solved,
            'success_rate': self.problems_solved / max(1, self.problems_attempted),
            'final_error': latest_metrics.total_error,
            'final_convergence': latest_metrics.convergence_ratio,
            'average_error': np.mean(errors),
            'error_std': np.std(errors),
            'average_convergence': np.mean(convergence),
            'convergence_std': np.std(convergence),
            'current_phase': self.current_phase,
            'training_efficiency': self._calculate_training_efficiency()
        }

    def _logging_loop(self) -> None:
        """Main logging loop for periodic tasks"""
        while not self.should_stop_logging.wait(timeout=1.0):
            if not self.is_logging:
                break

            try:
                # Periodic maintenance tasks
                self._update_baseline_metrics()
                self._cleanup_old_data()

                # Performance monitoring
                self.performance_monitor.update_performance_stats()

            except Exception as e:
                print(f"Logging loop error: {e}")

    def _extract_dye_metrics(self) -> Dict[str, Any]:
        """Extract dye system metrics from network"""
        # Placeholder - would interface with actual dye system
        return {
            'total_concentration': 0.0,
            'diffusion_rate': 0.0,
            'enhancement_factor': 1.0,
            'spatial_distribution': {'red': 0.0, 'blue': 0.0, 'yellow': 0.0}
        }

    def _extract_computational_metrics(self) -> Dict[str, float]:
        """Extract computational performance metrics"""
        perf_stats = self.performance_monitor.get_performance_stats()

        return {
            'iteration_time_ms': perf_stats.frame_time_ms,
            'memory_usage_mb': perf_stats.memory_usage_mb,
            'cpu_utilization': perf_stats.cpu_usage_percent
        }

    def _check_problem_solved(self, error_metrics: Dict[str, float]) -> bool:
        """Check if current problem is solved based on error metrics"""
        if not self.current_problem:
            return False

        # Simple threshold-based detection
        accuracy = error_metrics.get('accuracy', 0.0)
        convergence = error_metrics.get('convergence', 0.0)

        return accuracy > 0.95 and convergence > 0.9

    def _detect_anomalies(self, metrics: TrainingMetrics) -> None:
        """Detect training anomalies"""
        if len(self.metrics_history) < 10:
            return  # Need baseline

        # Check for sudden spikes in error
        recent_errors = [m.total_error for m in self.metrics_history[-10:]]
        if metrics.total_error > np.mean(recent_errors) + self.anomaly_threshold * np.std(recent_errors):
            print(f"⚠️ Anomaly detected: Error spike at iteration {metrics.iteration}")

    def _update_learning_phase(self, metrics: TrainingMetrics) -> None:
        """Update learning phase based on metrics trends"""
        if len(self.metrics_history) < 20:
            return

        # Analyze recent convergence trend
        recent_convergence = [m.convergence_ratio for m in self.metrics_history[-20:]]
        convergence_trend = np.polyfit(range(len(recent_convergence)), recent_convergence, 1)[0]

        # Phase detection logic
        current_time = time.time()
        phase_duration = current_time - self.phase_start_time

        if self.current_phase == "exploration" and convergence_trend > 0.01:
            self.current_phase = "convergence"
            self.phase_start_time = current_time
        elif self.current_phase == "convergence" and abs(convergence_trend) < 0.001:
            self.current_phase = "refinement"
            self.phase_start_time = current_time
        elif metrics.convergence_ratio > 0.95:
            self.current_phase = "completion"
            self.phase_start_time = current_time

    def _generate_snapshot(self, current_metrics: TrainingMetrics) -> MetricsSnapshot:
        """Generate comprehensive metrics snapshot"""
        timestamp = time.time()

        # Calculate rolling averages
        if len(self.metrics_history) >= 10:
            recent_10 = self.metrics_history[-10:]
            avg_error_10 = np.mean([m.total_error for m in recent_10])
            avg_convergence_10 = np.mean([m.convergence_ratio for m in recent_10])
        else:
            avg_error_10 = current_metrics.total_error
            avg_convergence_10 = current_metrics.convergence_ratio

        if len(self.metrics_history) >= 100:
            recent_100 = self.metrics_history[-100:]
            avg_error_100 = np.mean([m.total_error for m in recent_100])
            avg_convergence_100 = np.mean([m.convergence_ratio for m in recent_100])
        else:
            avg_error_100 = avg_error_10
            avg_convergence_100 = avg_convergence_10

        # Build trends
        error_trend = [m.total_error for m in self.metrics_history[-50:]] if len(self.metrics_history) >= 50 else []
        convergence_trend = [m.convergence_ratio for m in self.metrics_history[-50:]] if len(self.metrics_history) >= 50 else []
        lr_trend = [m.learning_rate for m in self.metrics_history[-50:]] if len(self.metrics_history) >= 50 else []

        # Calculate performance metrics
        duration = timestamp - self.start_time
        iterations_per_second = len(self.metrics_history) / duration if duration > 0 else 0

        snapshot = MetricsSnapshot(
            timestamp=timestamp,
            training_session_id=self.session_id,
            current_metrics=current_metrics,
            avg_error_10=avg_error_10,
            avg_error_100=avg_error_100,
            avg_convergence_10=avg_convergence_10,
            avg_convergence_100=avg_convergence_100,
            total_problems_attempted=self.problems_attempted,
            total_problems_solved=self.problems_solved,
            current_success_rate=self.problems_solved / max(1, self.problems_attempted),
            error_trend=error_trend,
            convergence_trend=convergence_trend,
            learning_rate_trend=lr_trend,
            iterations_per_second=iterations_per_second,
            training_efficiency_score=self._calculate_training_efficiency()
        )

        with self.logging_lock:
            self.snapshots.append(snapshot)
            if len(self.snapshots) > 1000:  # Limit snapshot history
                self.snapshots = self.snapshots[-500:]

        return snapshot

    def _generate_visualization_state(self, metrics: TrainingMetrics, snapshot: MetricsSnapshot) -> VisualizationTrainingState:
        """Generate visualization-optimized training state"""
        current_time = time.time()

        # Calculate progress
        total_iterations = 1000  # Would be configurable
        progress = min(100.0, (metrics.iteration / total_iterations) * 100)

        # Phase stability calculation
        phase_duration = current_time - self.phase_start_time
        stability = min(1.0, phase_duration / 30.0)  # Stabilizes over 30 seconds

        # Highlight important blocks (simplified)
        highlight_blocks = []
        if metrics.firing_rate > 0.1:
            highlight_blocks = list(range(min(5, int(metrics.firing_rate * 79))))

        # Training health assessment
        health_score = min(1.0, (metrics.convergence_ratio + (1.0 - metrics.total_error)) / 2.0)

        vis_state = VisualizationTrainingState(
            timestamp=current_time,
            session_id=self.session_id,
            current_iteration=metrics.iteration,
            total_iterations=total_iterations,
            progress_percentage=progress,
            current_snapshot=snapshot,
            learning_phase=self.current_phase,
            phase_stability=stability,
            phase_duration_seconds=phase_duration,
            current_problem_state={
                'problem': self.current_problem,
                'start_time': self.problem_start_time,
                'duration': current_time - self.problem_start_time if self.current_problem else 0.0
            },
            highlight_blocks=highlight_blocks,
            training_health_score=health_score
        )

        return vis_state

    def _calculate_training_efficiency(self) -> float:
        """Calculate overall training efficiency score"""
        if len(self.metrics_history) < 10:
            return 0.0

        # Factors: convergence rate, error reduction, computational efficiency
        recent_metrics = self.metrics_history[-10:]

        # Convergence improvement
        convergence_improvement = recent_metrics[-1].convergence_ratio - recent_metrics[0].convergence_ratio

        # Error reduction
        error_reduction = recent_metrics[0].total_error - recent_metrics[-1].total_error

        # Computational efficiency (iterations per second)
        duration = time.time() - self.start_time
        iteration_efficiency = len(self.metrics_history) / duration if duration > 0 else 0
        normalized_efficiency = min(1.0, iteration_efficiency / 10.0)  # Normalize to 10 iter/sec

        # Combined efficiency score
        efficiency = (
            0.4 * max(0.0, convergence_improvement) +
            0.4 * max(0.0, error_reduction) +
            0.2 * normalized_efficiency
        )

        return min(1.0, efficiency)

    def _notify_callbacks(self, metrics: TrainingMetrics, snapshot: MetricsSnapshot, vis_state: VisualizationTrainingState) -> None:
        """Notify all registered callbacks"""
        # Metrics callbacks
        for callback in self.metrics_callbacks:
            try:
                callback(metrics)
            except Exception as e:
                print(f"Metrics callback error: {e}")

        # Snapshot callbacks
        for callback in self.snapshot_callbacks:
            try:
                callback(snapshot)
            except Exception as e:
                print(f"Snapshot callback error: {e}")

        # State callbacks
        for callback in self.state_callbacks:
            try:
                callback(vis_state)
            except Exception as e:
                print(f"State callback error: {e}")

    def _append_to_log_file(self, metrics: TrainingMetrics) -> None:
        """Append metrics to log file"""
        if not self.log_to_file:
            return

        try:
            log_entry = {
                'timestamp': metrics.timestamp,
                'epoch': metrics.epoch,
                'iteration': metrics.iteration,
                'total_error': metrics.total_error,
                'convergence_ratio': metrics.convergence_ratio,
                'learning_rate': metrics.learning_rate,
                'firing_rate': metrics.firing_rate,
                'current_problem': metrics.current_problem,
                'problem_solved': metrics.problem_solved
            }

            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')

        except Exception as e:
            print(f"Log file write error: {e}")

    def _save_session_summary(self) -> None:
        """Save complete session summary"""
        summary = self.get_session_summary()
        summary_file = self.log_directory / f"{self.session_id}_summary.json"

        try:
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)

        except Exception as e:
            print(f"Summary save error: {e}")

    def _update_baseline_metrics(self) -> None:
        """Update baseline metrics for anomaly detection"""
        if len(self.metrics_history) < 50:
            return

        recent_metrics = self.metrics_history[-50:]
        self.baseline_metrics = {
            'error_mean': np.mean([m.total_error for m in recent_metrics]),
            'error_std': np.std([m.total_error for m in recent_metrics]),
            'convergence_mean': np.mean([m.convergence_ratio for m in recent_metrics]),
            'convergence_std': np.std([m.convergence_ratio for m in recent_metrics])
        }

    def _cleanup_old_data(self) -> None:
        """Clean up old data to maintain memory limits"""
        # This runs periodically to prevent memory bloat
        current_time = time.time()

        # Remove snapshots older than 1 hour
        cutoff_time = current_time - 3600
        with self.logging_lock:
            self.snapshots = [s for s in self.snapshots if s.timestamp > cutoff_time]