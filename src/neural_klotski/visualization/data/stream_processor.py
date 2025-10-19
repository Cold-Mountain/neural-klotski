"""
Stream Processing System for Neural-Klotski Visualization

Real-time data filtering, aggregation, and processing pipeline for
visualization data streams. Supports complex processing workflows
with configurable filters and aggregators.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic, TypeVar, List, Optional, Callable, Any, Dict, Tuple, Union
import time
import threading
import numpy as np
from enum import Enum
import operator
from collections import defaultdict, deque
import statistics
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from neural_klotski.visualization.data.data_buffer import DataBuffer, TimeSeriesBuffer
from neural_klotski.visualization.data.network_state_capture import VisualizationNetworkState, StateDelta
from neural_klotski.visualization.data.performance_logger import TrainingMetrics, MetricsSnapshot

T = TypeVar('T')
U = TypeVar('U')


class FilterType(Enum):
    """Types of data filters"""
    THRESHOLD = "threshold"
    RANGE = "range"
    PATTERN = "pattern"
    STATISTICAL = "statistical"
    TEMPORAL = "temporal"
    CUSTOM = "custom"


class AggregationType(Enum):
    """Types of data aggregation"""
    MEAN = "mean"
    MEDIAN = "median"
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    STD = "std"
    COUNT = "count"
    RATE = "rate"
    TREND = "trend"
    CUSTOM = "custom"


@dataclass
class ProcessingConfig:
    """Configuration for stream processing"""
    # Filter settings
    enable_filtering: bool = True
    cascade_filters: bool = True
    filter_on_error: str = "skip"  # "skip", "stop", "default"

    # Aggregation settings
    enable_aggregation: bool = True
    aggregation_window_size: int = 10
    aggregation_overlap: float = 0.5  # 0.0 to 1.0

    # Performance settings
    max_processing_time_ms: float = 10.0
    enable_parallel_processing: bool = False
    buffer_size: int = 1000

    # Output settings
    output_format: str = "structured"  # "structured", "flat", "compressed"
    include_metadata: bool = True
    timestamp_precision: int = 6


class DataFilter(Generic[T], ABC):
    """
    Abstract base class for data filters.

    Filters process individual data items and determine whether
    they should pass through the processing pipeline.
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize data filter"""
        self.name = name
        self.config = config or {}
        self.statistics = {
            'items_processed': 0,
            'items_passed': 0,
            'items_filtered': 0,
            'processing_time_ms': 0.0
        }

    @abstractmethod
    def filter(self, item: T) -> bool:
        """
        Filter data item.

        Args:
            item: Data item to filter

        Returns:
            True if item should pass through, False otherwise
        """
        pass

    def process_item(self, item: T) -> bool:
        """Process item with statistics tracking"""
        start_time = time.perf_counter()

        try:
            self.statistics['items_processed'] += 1
            result = self.filter(item)

            if result:
                self.statistics['items_passed'] += 1
            else:
                self.statistics['items_filtered'] += 1

            return result

        except Exception as e:
            print(f"Filter '{self.name}' error: {e}")
            return False

        finally:
            elapsed = (time.perf_counter() - start_time) * 1000
            self.statistics['processing_time_ms'] += elapsed

    def get_statistics(self) -> Dict[str, Any]:
        """Get filter statistics"""
        stats = self.statistics.copy()
        if stats['items_processed'] > 0:
            stats['pass_rate'] = stats['items_passed'] / stats['items_processed']
            stats['filter_rate'] = stats['items_filtered'] / stats['items_processed']
            stats['avg_processing_time_ms'] = stats['processing_time_ms'] / stats['items_processed']
        else:
            stats['pass_rate'] = 0.0
            stats['filter_rate'] = 0.0
            stats['avg_processing_time_ms'] = 0.0

        return stats

    def reset_statistics(self) -> None:
        """Reset filter statistics"""
        self.statistics = {
            'items_processed': 0,
            'items_passed': 0,
            'items_filtered': 0,
            'processing_time_ms': 0.0
        }


class DataAggregator(Generic[T, U], ABC):
    """
    Abstract base class for data aggregators.

    Aggregators process collections of data items and produce
    summary statistics or derived values.
    """

    def __init__(self, name: str, window_size: int = 10, config: Optional[Dict[str, Any]] = None):
        """Initialize data aggregator"""
        self.name = name
        self.window_size = window_size
        self.config = config or {}

        self.statistics = {
            'windows_processed': 0,
            'items_aggregated': 0,
            'processing_time_ms': 0.0
        }

    @abstractmethod
    def aggregate(self, items: List[T]) -> U:
        """
        Aggregate data items.

        Args:
            items: List of data items to aggregate

        Returns:
            Aggregated result
        """
        pass

    def process_window(self, items: List[T]) -> Optional[U]:
        """Process window with statistics tracking"""
        if not items:
            return None

        start_time = time.perf_counter()

        try:
            self.statistics['windows_processed'] += 1
            self.statistics['items_aggregated'] += len(items)

            result = self.aggregate(items)
            return result

        except Exception as e:
            print(f"Aggregator '{self.name}' error: {e}")
            return None

        finally:
            elapsed = (time.perf_counter() - start_time) * 1000
            self.statistics['processing_time_ms'] += elapsed

    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregator statistics"""
        stats = self.statistics.copy()
        if stats['windows_processed'] > 0:
            stats['avg_window_size'] = stats['items_aggregated'] / stats['windows_processed']
            stats['avg_processing_time_ms'] = stats['processing_time_ms'] / stats['windows_processed']
        else:
            stats['avg_window_size'] = 0.0
            stats['avg_processing_time_ms'] = 0.0

        return stats


class ProcessingPipeline(Generic[T]):
    """
    Data processing pipeline with configurable filters and aggregators.

    Coordinates the processing workflow from raw data input to
    aggregated output suitable for visualization.
    """

    def __init__(self, config: Optional[ProcessingConfig] = None):
        """Initialize processing pipeline"""
        self.config = config or ProcessingConfig()

        # Pipeline components
        self.filters: List[DataFilter[T]] = []
        self.aggregators: List[DataAggregator[T, Any]] = []

        # Processing state
        self.is_running = False
        self.input_buffer = TimeSeriesBuffer()
        self.output_buffer = TimeSeriesBuffer()

        # Statistics
        self.pipeline_statistics = {
            'items_input': 0,
            'items_output': 0,
            'processing_cycles': 0,
            'total_processing_time_ms': 0.0,
            'errors': 0
        }

        # Threading
        self.processing_thread: Optional[threading.Thread] = None
        self.should_stop = threading.Event()
        self.processing_lock = threading.RLock()

        # Callbacks
        self.output_callbacks: List[Callable[[Any], None]] = []
        self.error_callbacks: List[Callable[[Exception], None]] = []

    def add_filter(self, filter_instance: DataFilter[T]) -> None:
        """Add filter to pipeline"""
        with self.processing_lock:
            self.filters.append(filter_instance)

    def add_aggregator(self, aggregator: DataAggregator[T, Any]) -> None:
        """Add aggregator to pipeline"""
        with self.processing_lock:
            self.aggregators.append(aggregator)

    def remove_filter(self, filter_name: str) -> bool:
        """Remove filter by name"""
        with self.processing_lock:
            for i, f in enumerate(self.filters):
                if f.name == filter_name:
                    del self.filters[i]
                    return True
            return False

    def remove_aggregator(self, aggregator_name: str) -> bool:
        """Remove aggregator by name"""
        with self.processing_lock:
            for i, a in enumerate(self.aggregators):
                if a.name == aggregator_name:
                    del self.aggregators[i]
                    return True
            return False

    def start_processing(self) -> None:
        """Start pipeline processing"""
        if self.is_running:
            return

        self.is_running = True
        self.should_stop.clear()

        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()

        print(f"🔄 Processing pipeline started with {len(self.filters)} filters and {len(self.aggregators)} aggregators")

    def stop_processing(self) -> None:
        """Stop pipeline processing"""
        if not self.is_running:
            return

        self.is_running = False
        self.should_stop.set()

        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)

        print("⏹️ Processing pipeline stopped")

    def process_item(self, item: T) -> None:
        """Add item to processing pipeline"""
        with self.processing_lock:
            self.input_buffer.append(item)
            self.pipeline_statistics['items_input'] += 1

    def register_output_callback(self, callback: Callable[[Any], None]) -> None:
        """Register callback for processed output"""
        self.output_callbacks.append(callback)

    def register_error_callback(self, callback: Callable[[Exception], None]) -> None:
        """Register callback for processing errors"""
        self.error_callbacks.append(callback)

    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics"""
        with self.processing_lock:
            stats = self.pipeline_statistics.copy()

            # Add component statistics
            stats['filters'] = {f.name: f.get_statistics() for f in self.filters}
            stats['aggregators'] = {a.name: a.get_statistics() for a in self.aggregators}

            # Calculate derived metrics
            if stats['processing_cycles'] > 0:
                stats['avg_processing_time_ms'] = stats['total_processing_time_ms'] / stats['processing_cycles']
                stats['throughput_items_per_sec'] = stats['items_input'] / (stats['total_processing_time_ms'] / 1000)
            else:
                stats['avg_processing_time_ms'] = 0.0
                stats['throughput_items_per_sec'] = 0.0

            if stats['items_input'] > 0:
                stats['output_ratio'] = stats['items_output'] / stats['items_input']
            else:
                stats['output_ratio'] = 0.0

            return stats

    def _processing_loop(self) -> None:
        """Main processing loop"""
        while not self.should_stop.wait(timeout=0.1):
            try:
                self._process_batch()
            except Exception as e:
                self.pipeline_statistics['errors'] += 1
                self._notify_error_callbacks(e)

    def _process_batch(self) -> None:
        """Process a batch of items"""
        cycle_start = time.perf_counter()

        with self.processing_lock:
            # Get batch of items
            batch_size = min(self.config.aggregation_window_size, self.input_buffer.size())
            if batch_size == 0:
                return

            batch = self.input_buffer.get_recent(batch_size)
            if not batch:
                return

            # Apply filters
            if self.config.enable_filtering:
                batch = self._apply_filters(batch)

            # Apply aggregators
            if self.config.enable_aggregation and batch:
                aggregated_results = self._apply_aggregators(batch)

                # Output results
                for result in aggregated_results:
                    self.output_buffer.append(result)
                    self.pipeline_statistics['items_output'] += 1
                    self._notify_output_callbacks(result)

            # Update statistics
            self.pipeline_statistics['processing_cycles'] += 1
            cycle_time = (time.perf_counter() - cycle_start) * 1000
            self.pipeline_statistics['total_processing_time_ms'] += cycle_time

    def _apply_filters(self, items: List[T]) -> List[T]:
        """Apply all filters to items"""
        filtered_items = items

        for filter_instance in self.filters:
            if self.config.cascade_filters:
                # Cascade: each filter operates on previous filter's output
                filtered_items = [item for item in filtered_items if filter_instance.process_item(item)]
            else:
                # Parallel: each filter operates on original items
                filtered_items = [item for item in items if filter_instance.process_item(item)]

        return filtered_items

    def _apply_aggregators(self, items: List[T]) -> List[Any]:
        """Apply all aggregators to items"""
        results = []

        for aggregator in self.aggregators:
            result = aggregator.process_window(items)
            if result is not None:
                results.append({
                    'aggregator': aggregator.name,
                    'result': result,
                    'timestamp': time.time(),
                    'window_size': len(items)
                })

        return results

    def _notify_output_callbacks(self, result: Any) -> None:
        """Notify output callbacks"""
        for callback in self.output_callbacks:
            try:
                callback(result)
            except Exception as e:
                print(f"Output callback error: {e}")

    def _notify_error_callbacks(self, error: Exception) -> None:
        """Notify error callbacks"""
        for callback in self.error_callbacks:
            try:
                callback(error)
            except Exception as e:
                print(f"Error callback error: {e}")


# Concrete Filter Implementations

class ThresholdFilter(DataFilter[VisualizationNetworkState]):
    """Filter network states based on threshold values"""

    def __init__(self, threshold_field: str, threshold_value: float, operator_type: str = "greater"):
        """
        Initialize threshold filter.

        Args:
            threshold_field: Field to check (e.g., 'network_energy', 'firing_rate')
            threshold_value: Threshold value
            operator_type: 'greater', 'less', 'equal', 'not_equal'
        """
        super().__init__(f"threshold_{threshold_field}_{operator_type}")
        self.threshold_field = threshold_field
        self.threshold_value = threshold_value

        self.operator_map = {
            'greater': operator.gt,
            'less': operator.lt,
            'equal': operator.eq,
            'not_equal': operator.ne,
            'greater_equal': operator.ge,
            'less_equal': operator.le
        }
        self.op = self.operator_map.get(operator_type, operator.gt)

    def filter(self, item: VisualizationNetworkState) -> bool:
        """Filter based on threshold"""
        try:
            value = getattr(item, self.threshold_field, 0.0)
            return self.op(value, self.threshold_value)
        except Exception:
            return False


class ActivityFilter(DataFilter[VisualizationNetworkState]):
    """Filter based on network activity levels"""

    def __init__(self, min_activity: float = 0.01, max_activity: float = 1.0):
        super().__init__("activity_filter")
        self.min_activity = min_activity
        self.max_activity = max_activity

    def filter(self, item: VisualizationNetworkState) -> bool:
        """Filter based on activity level"""
        activity = item.get_firing_rate()
        return self.min_activity <= activity <= self.max_activity


class LearningPhaseFilter(DataFilter[TrainingMetrics]):
    """Filter training metrics based on learning phase"""

    def __init__(self, allowed_phases: List[str]):
        super().__init__("learning_phase_filter")
        self.allowed_phases = set(allowed_phases)

    def filter(self, item: TrainingMetrics) -> bool:
        """Filter based on learning phase"""
        # This would need integration with phase detection
        return True  # Placeholder


# Concrete Aggregator Implementations

class NetworkMetricsAggregator(DataAggregator[VisualizationNetworkState, Dict[str, float]]):
    """Aggregate network state metrics"""

    def __init__(self, window_size: int = 10):
        super().__init__("network_metrics", window_size)

    def aggregate(self, items: List[VisualizationNetworkState]) -> Dict[str, float]:
        """Aggregate network metrics"""
        if not items:
            return {}

        # Extract metrics
        firing_rates = [item.get_firing_rate() for item in items]
        energies = [item.network_energy for item in items]
        signal_counts = [item.get_active_signal_count() for item in items]

        return {
            'avg_firing_rate': np.mean(firing_rates),
            'std_firing_rate': np.std(firing_rates),
            'avg_energy': np.mean(energies),
            'std_energy': np.std(energies),
            'avg_signal_count': np.mean(signal_counts),
            'max_signal_count': np.max(signal_counts),
            'window_size': len(items),
            'timestamp': time.time()
        }


class TrainingProgressAggregator(DataAggregator[TrainingMetrics, Dict[str, float]]):
    """Aggregate training progress metrics"""

    def __init__(self, window_size: int = 20):
        super().__init__("training_progress", window_size)

    def aggregate(self, items: List[TrainingMetrics]) -> Dict[str, float]:
        """Aggregate training progress"""
        if not items:
            return {}

        # Extract metrics
        errors = [item.total_error for item in items]
        convergences = [item.convergence_ratio for item in items]
        learning_rates = [item.learning_rate for item in items]

        # Calculate trends
        error_trend = np.polyfit(range(len(errors)), errors, 1)[0] if len(errors) > 1 else 0.0
        convergence_trend = np.polyfit(range(len(convergences)), convergences, 1)[0] if len(convergences) > 1 else 0.0

        return {
            'avg_error': np.mean(errors),
            'error_trend': error_trend,
            'avg_convergence': np.mean(convergences),
            'convergence_trend': convergence_trend,
            'current_learning_rate': learning_rates[-1] if learning_rates else 0.0,
            'learning_stability': 1.0 - np.std(learning_rates) if learning_rates else 0.0,
            'improvement_rate': (errors[0] - errors[-1]) / len(errors) if len(errors) > 1 else 0.0,
            'window_size': len(items),
            'timestamp': time.time()
        }


class StreamProcessor:
    """
    High-level stream processor for Neural-Klotski visualization data.

    Coordinates multiple processing pipelines for different data types
    and provides unified interface for stream processing operations.
    """

    def __init__(self):
        """Initialize stream processor"""
        # Processing pipelines for different data types
        self.network_state_pipeline = ProcessingPipeline[VisualizationNetworkState]()
        self.training_metrics_pipeline = ProcessingPipeline[TrainingMetrics]()
        self.delta_pipeline = ProcessingPipeline[StateDelta]()

        # Processor state
        self.is_running = False

        # Default filters and aggregators
        self._setup_default_processors()

    def _setup_default_processors(self) -> None:
        """Setup default filters and aggregators"""
        # Network state processing
        self.network_state_pipeline.add_filter(ActivityFilter(min_activity=0.001))
        self.network_state_pipeline.add_aggregator(NetworkMetricsAggregator(window_size=10))

        # Training metrics processing
        self.training_metrics_pipeline.add_aggregator(TrainingProgressAggregator(window_size=20))

    def start(self) -> None:
        """Start all processing pipelines"""
        if self.is_running:
            return

        self.network_state_pipeline.start_processing()
        self.training_metrics_pipeline.start_processing()
        self.delta_pipeline.start_processing()

        self.is_running = True
        print("🚀 Stream processor started")

    def stop(self) -> None:
        """Stop all processing pipelines"""
        if not self.is_running:
            return

        self.network_state_pipeline.stop_processing()
        self.training_metrics_pipeline.stop_processing()
        self.delta_pipeline.stop_processing()

        self.is_running = False
        print("⏹️ Stream processor stopped")

    def process_network_state(self, state: VisualizationNetworkState) -> None:
        """Process network state through pipeline"""
        self.network_state_pipeline.process_item(state)

    def process_training_metrics(self, metrics: TrainingMetrics) -> None:
        """Process training metrics through pipeline"""
        self.training_metrics_pipeline.process_item(metrics)

    def process_state_delta(self, delta: StateDelta) -> None:
        """Process state delta through pipeline"""
        self.delta_pipeline.process_item(delta)

    def register_network_callback(self, callback: Callable[[Any], None]) -> None:
        """Register callback for network state processing results"""
        self.network_state_pipeline.register_output_callback(callback)

    def register_training_callback(self, callback: Callable[[Any], None]) -> None:
        """Register callback for training metrics processing results"""
        self.training_metrics_pipeline.register_output_callback(callback)

    def register_delta_callback(self, callback: Callable[[Any], None]) -> None:
        """Register callback for delta processing results"""
        self.delta_pipeline.register_output_callback(callback)

    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get statistics for all pipelines"""
        return {
            'network_state_pipeline': self.network_state_pipeline.get_pipeline_statistics(),
            'training_metrics_pipeline': self.training_metrics_pipeline.get_pipeline_statistics(),
            'delta_pipeline': self.delta_pipeline.get_pipeline_statistics(),
            'is_running': self.is_running
        }


# Factory functions for creating common filter/aggregator combinations

def create_activity_monitoring_pipeline() -> ProcessingPipeline[VisualizationNetworkState]:
    """Create pipeline for monitoring network activity"""
    pipeline = ProcessingPipeline[VisualizationNetworkState]()

    # Add activity filters
    pipeline.add_filter(ActivityFilter(min_activity=0.01))
    pipeline.add_filter(ThresholdFilter('network_energy', 1.0, 'greater'))

    # Add metrics aggregator
    pipeline.add_aggregator(NetworkMetricsAggregator(window_size=15))

    return pipeline


def create_training_monitoring_pipeline() -> ProcessingPipeline[TrainingMetrics]:
    """Create pipeline for monitoring training progress"""
    pipeline = ProcessingPipeline[TrainingMetrics]()

    # Add training progress aggregator
    pipeline.add_aggregator(TrainingProgressAggregator(window_size=25))

    return pipeline