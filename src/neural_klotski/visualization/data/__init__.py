"""
Data Pipeline for Neural-Klotski Visualization

Provides real-time data capture, processing, and management for visualization
components. Handles network state extraction, performance metrics collection,
and data buffering for smooth visualization.
"""

from .network_state_capture import (
    NetworkStateCapture,
    VisualizationNetworkState,
    VisualizationSignal,
    StateDelta,
    CaptureConfig
)

from .performance_logger import (
    PerformanceLogger,
    VisualizationTrainingState,
    MetricsSnapshot,
    TrainingMetrics
)

from .data_buffer import (
    DataBuffer,
    CircularBuffer,
    TimeSeriesBuffer,
    BufferConfig
)

from .stream_processor import (
    StreamProcessor,
    DataFilter,
    DataAggregator,
    ProcessingPipeline
)

__all__ = [
    # Network state capture
    'NetworkStateCapture',
    'VisualizationNetworkState',
    'VisualizationSignal',
    'StateDelta',
    'CaptureConfig',

    # Performance logging
    'PerformanceLogger',
    'VisualizationTrainingState',
    'MetricsSnapshot',
    'TrainingMetrics',

    # Data buffering
    'DataBuffer',
    'CircularBuffer',
    'TimeSeriesBuffer',
    'BufferConfig',

    # Stream processing
    'StreamProcessor',
    'DataFilter',
    'DataAggregator',
    'ProcessingPipeline'
]