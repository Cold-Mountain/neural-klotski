"""
Data Buffer System for Neural-Klotski Visualization

Efficient buffering and management of real-time visualization data with
multiple buffer types optimized for different data patterns and access needs.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar, List, Optional, Iterator, Callable, Any, Dict, Tuple
import time
import threading
import numpy as np
from collections import deque
import weakref

T = TypeVar('T')


@dataclass
class BufferConfig:
    """Configuration for data buffers"""
    # Buffer size limits
    max_size: int = 1000
    max_memory_mb: float = 100.0
    auto_cleanup: bool = True

    # Time-based retention
    max_age_seconds: float = 300.0  # 5 minutes default
    cleanup_interval_seconds: float = 30.0

    # Performance optimization
    enable_compression: bool = False
    compression_threshold: int = 100
    enable_indexing: bool = True

    # Access patterns
    optimize_for_recent: bool = True
    pre_allocate: bool = True


class DataBuffer(Generic[T], ABC):
    """
    Abstract base class for data buffers.

    Provides common interface for different buffer implementations
    optimized for various access patterns and data types.
    """

    def __init__(self, config: Optional[BufferConfig] = None):
        """Initialize data buffer with configuration"""
        self.config = config or BufferConfig()
        self._lock = threading.RLock()
        self._size = 0
        self._memory_usage = 0.0

        # Statistics
        self._total_writes = 0
        self._total_reads = 0
        self._total_evictions = 0

        # Auto-cleanup thread
        self._cleanup_thread: Optional[threading.Thread] = None
        self._should_stop_cleanup = threading.Event()

        if self.config.auto_cleanup:
            self._start_cleanup_thread()

    @abstractmethod
    def append(self, item: T) -> None:
        """Add item to buffer"""
        pass

    @abstractmethod
    def get_recent(self, count: int) -> List[T]:
        """Get most recent N items"""
        pass

    @abstractmethod
    def get_all(self) -> List[T]:
        """Get all items in buffer"""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all items from buffer"""
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        """Check if buffer is empty"""
        pass

    def size(self) -> int:
        """Get current buffer size"""
        with self._lock:
            return self._size

    def memory_usage_mb(self) -> float:
        """Get current memory usage in MB"""
        with self._lock:
            return self._memory_usage

    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        with self._lock:
            return {
                'size': self._size,
                'memory_usage_mb': self._memory_usage,
                'total_writes': self._total_writes,
                'total_reads': self._total_reads,
                'total_evictions': self._total_evictions,
                'max_size': self.config.max_size,
                'max_memory_mb': self.config.max_memory_mb
            }

    def _start_cleanup_thread(self) -> None:
        """Start background cleanup thread"""
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()

    def _cleanup_loop(self) -> None:
        """Background cleanup loop"""
        while not self._should_stop_cleanup.wait(self.config.cleanup_interval_seconds):
            try:
                self._perform_cleanup()
            except Exception as e:
                print(f"Buffer cleanup error: {e}")

    def _perform_cleanup(self) -> None:
        """Perform buffer cleanup operations"""
        with self._lock:
            # Check memory usage
            if self._memory_usage > self.config.max_memory_mb:
                self._evict_by_memory()

            # Age-based cleanup
            self._evict_by_age()

    @abstractmethod
    def _evict_by_memory(self) -> None:
        """Evict items to reduce memory usage"""
        pass

    @abstractmethod
    def _evict_by_age(self) -> None:
        """Evict items based on age"""
        pass

    def _estimate_item_size(self, item: Any) -> float:
        """Estimate memory size of item in MB"""
        # Simple estimation - can be overridden for specific types
        import sys
        return sys.getsizeof(item) / (1024 * 1024)

    def __del__(self):
        """Cleanup on deletion"""
        if self._cleanup_thread:
            self._should_stop_cleanup.set()


class CircularBuffer(DataBuffer[T]):
    """
    Circular buffer with fixed size and FIFO eviction.

    Optimized for high-frequency writes with size-bounded storage.
    Most efficient for real-time data streams.
    """

    def __init__(self, max_size: int, config: Optional[BufferConfig] = None):
        """Initialize circular buffer with fixed size"""
        if config is None:
            config = BufferConfig()
        config.max_size = max_size

        super().__init__(config)

        # Circular buffer storage
        self._buffer: List[Optional[T]] = [None] * max_size
        self._head = 0  # Next write position
        self._count = 0  # Number of items in buffer

        # Index for O(1) recent access
        self._write_order: deque = deque(maxlen=max_size)

    def append(self, item: T) -> None:
        """Add item to circular buffer"""
        with self._lock:
            # Store item
            old_item = self._buffer[self._head]
            self._buffer[self._head] = item

            # Update tracking
            if old_item is not None:
                self._memory_usage -= self._estimate_item_size(old_item)
                self._total_evictions += 1
            else:
                self._count += 1

            self._memory_usage += self._estimate_item_size(item)
            self._write_order.append((self._head, time.time()))

            # Advance head
            self._head = (self._head + 1) % self.config.max_size
            self._total_writes += 1
            self._size = self._count

    def get_recent(self, count: int) -> List[T]:
        """Get most recent N items"""
        with self._lock:
            self._total_reads += 1

            if count <= 0 or self._count == 0:
                return []

            # Get recent indices from write order
            recent_count = min(count, len(self._write_order))
            recent_indices = [idx for idx, _ in list(self._write_order)[-recent_count:]]

            # Retrieve items
            items = []
            for idx in recent_indices:
                if self._buffer[idx] is not None:
                    items.append(self._buffer[idx])

            return items

    def get_all(self) -> List[T]:
        """Get all items in buffer"""
        with self._lock:
            self._total_reads += 1

            items = []
            for idx, _ in self._write_order:
                if self._buffer[idx] is not None:
                    items.append(self._buffer[idx])

            return items

    def get_most_recent(self) -> Optional[T]:
        """Get most recently added item"""
        with self._lock:
            if self._count == 0:
                return None

            # Get most recent index
            recent_idx = self._write_order[-1][0] if self._write_order else None
            return self._buffer[recent_idx] if recent_idx is not None else None

    def clear(self) -> None:
        """Clear all items from buffer"""
        with self._lock:
            self._buffer = [None] * self.config.max_size
            self._head = 0
            self._count = 0
            self._memory_usage = 0.0
            self._write_order.clear()

    def is_empty(self) -> bool:
        """Check if buffer is empty"""
        with self._lock:
            return self._count == 0

    def _evict_by_memory(self) -> None:
        """Memory-based eviction (circular buffer handles this naturally)"""
        # Circular buffer automatically evicts oldest items
        pass

    def _evict_by_age(self) -> None:
        """Age-based eviction"""
        current_time = time.time()
        cutoff_time = current_time - self.config.max_age_seconds

        # Remove old entries from write order
        while self._write_order and self._write_order[0][1] < cutoff_time:
            old_idx, _ = self._write_order.popleft()
            if self._buffer[old_idx] is not None:
                self._memory_usage -= self._estimate_item_size(self._buffer[old_idx])
                self._buffer[old_idx] = None
                self._count -= 1
                self._total_evictions += 1


class TimeSeriesBuffer(DataBuffer[T]):
    """
    Time series buffer optimized for temporal data access.

    Maintains chronological ordering and supports time-based queries.
    Efficient for historical analysis and trend visualization.
    """

    def __init__(self, config: Optional[BufferConfig] = None):
        """Initialize time series buffer"""
        super().__init__(config)

        # Time-ordered storage
        self._data: List[Tuple[float, T]] = []  # (timestamp, item)
        self._timestamps: List[float] = []  # For binary search

        # Index for efficient time-based lookups
        self._time_index: Dict[float, int] = {}

    def append(self, item: T, timestamp: Optional[float] = None) -> None:
        """Add item with timestamp to buffer"""
        if timestamp is None:
            timestamp = time.time()

        with self._lock:
            # Insert in chronological order
            insert_pos = self._find_insert_position(timestamp)

            self._data.insert(insert_pos, (timestamp, item))
            self._timestamps.insert(insert_pos, timestamp)

            # Update index
            if self.config.enable_indexing:
                self._time_index[timestamp] = insert_pos
                self._update_indices_after_insert(insert_pos)

            # Update tracking
            self._size += 1
            self._memory_usage += self._estimate_item_size(item)
            self._total_writes += 1

            # Check size limits
            self._check_size_limits()

    def get_recent(self, count: int) -> List[T]:
        """Get most recent N items"""
        with self._lock:
            self._total_reads += 1

            if count <= 0 or self._size == 0:
                return []

            # Get last N items
            recent_data = self._data[-count:]
            return [item for _, item in recent_data]

    def get_all(self) -> List[T]:
        """Get all items in buffer"""
        with self._lock:
            self._total_reads += 1
            return [item for _, item in self._data]

    def get_time_range(self, start_time: float, end_time: float) -> List[T]:
        """Get items within time range"""
        with self._lock:
            self._total_reads += 1

            # Binary search for time range
            start_idx = self._find_time_index(start_time)
            end_idx = self._find_time_index(end_time, find_upper=True)

            return [item for _, item in self._data[start_idx:end_idx]]

    def get_since(self, since_time: float) -> List[T]:
        """Get all items since specified time"""
        current_time = time.time()
        return self.get_time_range(since_time, current_time)

    def get_last_n_seconds(self, seconds: float) -> List[T]:
        """Get items from last N seconds"""
        current_time = time.time()
        start_time = current_time - seconds
        return self.get_time_range(start_time, current_time)

    def clear(self) -> None:
        """Clear all items from buffer"""
        with self._lock:
            self._data.clear()
            self._timestamps.clear()
            self._time_index.clear()
            self._size = 0
            self._memory_usage = 0.0

    def is_empty(self) -> bool:
        """Check if buffer is empty"""
        with self._lock:
            return self._size == 0

    def _find_insert_position(self, timestamp: float) -> int:
        """Find position to insert timestamp maintaining order"""
        # Binary search for insertion point
        left, right = 0, len(self._timestamps)

        while left < right:
            mid = (left + right) // 2
            if self._timestamps[mid] < timestamp:
                left = mid + 1
            else:
                right = mid

        return left

    def _find_time_index(self, timestamp: float, find_upper: bool = False) -> int:
        """Find index for timestamp using binary search"""
        if not self._timestamps:
            return 0

        # Binary search
        left, right = 0, len(self._timestamps)

        while left < right:
            mid = (left + right) // 2
            if find_upper:
                if self._timestamps[mid] <= timestamp:
                    left = mid + 1
                else:
                    right = mid
            else:
                if self._timestamps[mid] < timestamp:
                    left = mid + 1
                else:
                    right = mid

        return left

    def _update_indices_after_insert(self, insert_pos: int) -> None:
        """Update time index after insertion"""
        # Update all indices after insertion point
        for timestamp, old_index in list(self._time_index.items()):
            if old_index >= insert_pos:
                self._time_index[timestamp] = old_index + 1

    def _check_size_limits(self) -> None:
        """Check and enforce size limits"""
        # Size-based eviction
        while self._size > self.config.max_size:
            self._evict_oldest()

        # Memory-based eviction
        while self._memory_usage > self.config.max_memory_mb:
            self._evict_oldest()

    def _evict_oldest(self) -> None:
        """Evict oldest item"""
        if self._size > 0:
            timestamp, item = self._data.pop(0)
            self._timestamps.pop(0)

            if self.config.enable_indexing and timestamp in self._time_index:
                del self._time_index[timestamp]
                # Update remaining indices
                for ts, idx in self._time_index.items():
                    self._time_index[ts] = idx - 1

            self._size -= 1
            self._memory_usage -= self._estimate_item_size(item)
            self._total_evictions += 1

    def _evict_by_memory(self) -> None:
        """Memory-based eviction"""
        while self._memory_usage > self.config.max_memory_mb and self._size > 0:
            self._evict_oldest()

    def _evict_by_age(self) -> None:
        """Age-based eviction"""
        current_time = time.time()
        cutoff_time = current_time - self.config.max_age_seconds

        # Remove items older than cutoff
        while self._data and self._data[0][0] < cutoff_time:
            self._evict_oldest()


class StreamBuffer(DataBuffer[T]):
    """
    Streaming buffer optimized for continuous data processing.

    Supports windowed operations, real-time filtering, and batch processing.
    Designed for high-throughput data streams with processing callbacks.
    """

    def __init__(self,
                 window_size: int = 100,
                 config: Optional[BufferConfig] = None,
                 processor: Optional[Callable[[List[T]], Any]] = None):
        """
        Initialize streaming buffer.

        Args:
            window_size: Size of processing windows
            config: Buffer configuration
            processor: Optional processing function for windows
        """
        super().__init__(config)

        self.window_size = window_size
        self.processor = processor

        # Stream storage
        self._stream: deque = deque(maxlen=self.config.max_size)
        self._windows: List[List[T]] = []

        # Processing callbacks
        self._window_callbacks: List[Callable[[List[T]], None]] = []
        self._item_callbacks: List[Callable[[T], None]] = []

        # Batch processing
        self._batch_size = window_size
        self._current_batch: List[T] = []

    def append(self, item: T) -> None:
        """Add item to stream buffer"""
        with self._lock:
            # Add to stream
            if len(self._stream) >= self.config.max_size:
                old_item = self._stream[0]
                self._memory_usage -= self._estimate_item_size(old_item)
                self._total_evictions += 1

            self._stream.append(item)
            self._memory_usage += self._estimate_item_size(item)
            self._total_writes += 1
            self._size = len(self._stream)

            # Add to current batch
            self._current_batch.append(item)

            # Process complete windows
            if len(self._current_batch) >= self.window_size:
                self._process_window(self._current_batch.copy())
                self._current_batch.clear()

            # Notify item callbacks
            for callback in self._item_callbacks:
                try:
                    callback(item)
                except Exception as e:
                    print(f"Item callback error: {e}")

    def get_recent(self, count: int) -> List[T]:
        """Get most recent N items"""
        with self._lock:
            self._total_reads += 1
            return list(self._stream)[-count:] if count > 0 else []

    def get_all(self) -> List[T]:
        """Get all items in buffer"""
        with self._lock:
            self._total_reads += 1
            return list(self._stream)

    def get_window(self, window_index: int) -> Optional[List[T]]:
        """Get processed window by index"""
        with self._lock:
            if 0 <= window_index < len(self._windows):
                return self._windows[window_index].copy()
            return None

    def get_latest_window(self) -> Optional[List[T]]:
        """Get most recently processed window"""
        with self._lock:
            return self._windows[-1].copy() if self._windows else None

    def register_window_callback(self, callback: Callable[[List[T]], None]) -> None:
        """Register callback for window processing"""
        self._window_callbacks.append(callback)

    def register_item_callback(self, callback: Callable[[T], None]) -> None:
        """Register callback for individual items"""
        self._item_callbacks.append(callback)

    def force_window_processing(self) -> None:
        """Force processing of current incomplete batch"""
        with self._lock:
            if self._current_batch:
                self._process_window(self._current_batch.copy())
                self._current_batch.clear()

    def clear(self) -> None:
        """Clear all items from buffer"""
        with self._lock:
            self._stream.clear()
            self._windows.clear()
            self._current_batch.clear()
            self._size = 0
            self._memory_usage = 0.0

    def is_empty(self) -> bool:
        """Check if buffer is empty"""
        with self._lock:
            return len(self._stream) == 0

    def _process_window(self, window: List[T]) -> None:
        """Process a complete window"""
        # Store window
        self._windows.append(window)

        # Limit window history
        if len(self._windows) > 100:  # Configurable
            self._windows = self._windows[-50:]

        # Apply processor if available
        if self.processor:
            try:
                self.processor(window)
            except Exception as e:
                print(f"Window processor error: {e}")

        # Notify window callbacks
        for callback in self._window_callbacks:
            try:
                callback(window)
            except Exception as e:
                print(f"Window callback error: {e}")

    def _evict_by_memory(self) -> None:
        """Memory-based eviction"""
        # StreamBuffer uses deque with maxlen, so eviction is automatic
        pass

    def _evict_by_age(self) -> None:
        """Age-based eviction for windows"""
        current_time = time.time()
        cutoff_time = current_time - self.config.max_age_seconds

        # Note: Would need timestamps on windows for proper age-based eviction
        # For now, just limit window count
        if len(self._windows) > 50:
            self._windows = self._windows[-25:]


# Factory function for creating appropriate buffer types
def create_buffer(buffer_type: str,
                  config: Optional[BufferConfig] = None,
                  **kwargs) -> DataBuffer:
    """
    Factory function to create appropriate buffer type.

    Args:
        buffer_type: Type of buffer ('circular', 'timeseries', 'stream')
        config: Buffer configuration
        **kwargs: Additional arguments for specific buffer types

    Returns:
        Configured data buffer instance
    """
    if buffer_type == 'circular':
        max_size = kwargs.get('max_size', 1000)
        return CircularBuffer(max_size, config)

    elif buffer_type == 'timeseries':
        return TimeSeriesBuffer(config)

    elif buffer_type == 'stream':
        window_size = kwargs.get('window_size', 100)
        processor = kwargs.get('processor', None)
        return StreamBuffer(window_size, config, processor)

    else:
        raise ValueError(f"Unknown buffer type: {buffer_type}")


# Buffer pool for managing multiple buffers
class BufferPool:
    """
    Pool manager for multiple data buffers.

    Coordinates multiple buffers and provides unified access
    for complex data management scenarios.
    """

    def __init__(self):
        """Initialize buffer pool"""
        self._buffers: Dict[str, DataBuffer] = {}
        self._lock = threading.RLock()

    def create_buffer(self, name: str, buffer_type: str, config: Optional[BufferConfig] = None, **kwargs) -> DataBuffer:
        """Create and register a new buffer"""
        with self._lock:
            if name in self._buffers:
                raise ValueError(f"Buffer '{name}' already exists")

            buffer = create_buffer(buffer_type, config, **kwargs)
            self._buffers[name] = buffer
            return buffer

    def get_buffer(self, name: str) -> Optional[DataBuffer]:
        """Get buffer by name"""
        with self._lock:
            return self._buffers.get(name)

    def remove_buffer(self, name: str) -> bool:
        """Remove buffer by name"""
        with self._lock:
            if name in self._buffers:
                del self._buffers[name]
                return True
            return False

    def clear_all(self) -> None:
        """Clear all buffers"""
        with self._lock:
            for buffer in self._buffers.values():
                buffer.clear()

    def get_pool_statistics(self) -> Dict[str, Any]:
        """Get statistics for all buffers"""
        with self._lock:
            stats = {}
            for name, buffer in self._buffers.items():
                stats[name] = buffer.get_statistics()
            return stats

    def list_buffers(self) -> List[str]:
        """List all buffer names"""
        with self._lock:
            return list(self._buffers.keys())