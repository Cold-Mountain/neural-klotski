"""
Network State Capture System

Real-time extraction and processing of Neural-Klotski network states for
visualization. Captures block positions, wire activities, signal propagation,
and dye concentrations with efficient delta compression.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Callable, Any
import time
import threading
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from neural_klotski.core.network import NeuralKlotskiNetwork, NetworkState
from neural_klotski.core.block import BlockColor, BlockState
from neural_klotski.core.wire import Signal
from neural_klotski.visualization.utils import PerformanceUtils


@dataclass
class VisualizationSignal:
    """Signal representation optimized for visualization"""
    signal_id: int
    source_block_id: int
    target_block_id: int
    strength: float
    color: BlockColor

    # Spatial information for rendering
    start_position: Tuple[float, float]
    end_position: Tuple[float, float]
    current_position: Tuple[float, float]

    # Temporal information
    creation_time: float
    arrival_time: float
    progress: float  # 0.0 to 1.0

    # Visualization properties
    visual_size: float = 3.0
    visual_intensity: float = 1.0
    trail_positions: List[Tuple[float, float]] = field(default_factory=list)

    def update_progress(self, current_time: float) -> None:
        """Update signal progress based on current time"""
        if self.arrival_time <= self.creation_time:
            self.progress = 1.0
        else:
            elapsed = current_time - self.creation_time
            duration = self.arrival_time - self.creation_time
            self.progress = min(1.0, max(0.0, elapsed / duration))

        # Update current position based on progress
        start_x, start_y = self.start_position
        end_x, end_y = self.end_position
        self.current_position = (
            start_x + self.progress * (end_x - start_x),
            start_y + self.progress * (end_y - start_y)
        )

        # Update trail
        if len(self.trail_positions) > 10:  # Limit trail length
            self.trail_positions = self.trail_positions[-10:]
        self.trail_positions.append(self.current_position)


@dataclass
class VisualizationNetworkState:
    """Complete network state optimized for visualization"""
    timestamp: float

    # Block states (79 blocks)
    block_positions: np.ndarray          # Shape: (79, 2) - (activation, lag)
    block_velocities: np.ndarray         # Shape: (79,) - activation velocities
    block_colors: List[BlockColor]       # Color identity for each block
    block_thresholds: np.ndarray         # Current threshold values
    block_refractory: np.ndarray         # Refractory timer states
    firing_events: Set[int]              # Block IDs that fired this timestep

    # Wire network (1560 connections)
    wire_strengths: np.ndarray           # Current effective strengths
    wire_base_strengths: np.ndarray      # Base strengths (before enhancement)
    dye_enhancements: np.ndarray         # Dye enhancement factors
    active_wires: Set[int]               # Wire IDs with active signals

    # Signal propagation
    active_signals: List[VisualizationSignal]
    signal_queue_size: int

    # Dye concentrations (2D fields)
    red_dye_field: np.ndarray            # Shape: (grid_height, grid_width)
    blue_dye_field: np.ndarray           # Shape: (grid_height, grid_width)
    yellow_dye_field: np.ndarray         # Shape: (grid_height, grid_width)
    dye_grid_bounds: Tuple[float, float, float, float]  # (min_x, min_y, max_x, max_y)

    # Learning and performance metrics
    learning_rate: float = 0.0
    plasticity_activity: float = 0.0
    convergence_score: float = 0.0
    network_energy: float = 0.0

    # Metadata
    capture_duration_ms: float = 0.0
    frame_id: int = 0

    def get_block_count(self) -> int:
        """Get total number of blocks"""
        return len(self.block_positions)

    def get_active_signal_count(self) -> int:
        """Get number of active signals"""
        return len(self.active_signals)

    def get_firing_rate(self) -> float:
        """Get current firing rate (fraction of blocks firing)"""
        if len(self.block_positions) == 0:
            return 0.0
        return len(self.firing_events) / len(self.block_positions)


@dataclass
class StateDelta:
    """Compressed representation of state changes"""
    timestamp: float
    frame_id: int

    # Changed block IDs and their new states
    changed_blocks: Dict[int, Tuple[float, float, float]]  # id -> (position, velocity, threshold)
    firing_changes: Set[int]  # Blocks that started/stopped firing
    refractory_changes: Dict[int, float]  # id -> new refractory time

    # Wire changes
    strength_changes: Dict[int, float]  # wire_id -> new strength
    new_signals: List[VisualizationSignal]
    completed_signals: Set[int]  # Signal IDs that completed

    # Dye field changes (sparse representation)
    dye_changes: Dict[str, List[Tuple[int, int, float]]]  # color -> [(x, y, concentration)]

    # Learning changes
    learning_rate_change: Optional[float] = None
    plasticity_change: Optional[float] = None


@dataclass
class CaptureConfig:
    """Configuration for network state capture"""
    # Capture timing
    capture_rate_hz: float = 60.0
    delta_compression: bool = True
    change_threshold: float = 1e-6

    # Data filtering
    capture_signals: bool = True
    capture_dye_fields: bool = True
    capture_learning_metrics: bool = True
    max_signals_per_frame: int = 1000

    # Dye field sampling
    dye_grid_resolution: int = 100
    dye_sampling_rate: float = 10.0  # Hz (lower than main capture rate)

    # Performance optimization
    async_capture: bool = True
    buffer_size: int = 1000
    compression_level: int = 1  # 0=none, 1=delta, 2=full


class NetworkStateCapture:
    """
    Real-time network state capture system.

    Extracts complete network state from Neural-Klotski networks with
    efficient delta compression and performance optimization.
    """

    def __init__(self,
                 network: NeuralKlotskiNetwork,
                 config: Optional[CaptureConfig] = None):
        """
        Initialize network state capture.

        Args:
            network: Neural-Klotski network to capture
            config: Capture configuration
        """
        self.network = network
        self.config = config or CaptureConfig()

        # State tracking
        self.is_capturing = False
        self.last_state: Optional[VisualizationNetworkState] = None
        self.frame_counter = 0

        # Threading for async capture
        self.capture_thread: Optional[threading.Thread] = None
        self.capture_lock = threading.Lock()

        # Performance monitoring
        self.performance_monitor = PerformanceUtils()

        # Data buffers
        from .data_buffer import CircularBuffer
        self.state_buffer = CircularBuffer(self.config.buffer_size)
        self.delta_buffer = CircularBuffer(self.config.buffer_size)

        # Callbacks for real-time data streaming
        self.state_callbacks: List[Callable[[VisualizationNetworkState], None]] = []
        self.delta_callbacks: List[Callable[[StateDelta], None]] = []

        # Layout manager for position mapping (will be set externally)
        self.layout_manager: Optional[Any] = None

    def start_capture(self) -> None:
        """Start continuous network state capture"""
        if self.is_capturing:
            return

        self.is_capturing = True
        self.frame_counter = 0

        if self.config.async_capture:
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()

    def stop_capture(self) -> None:
        """Stop network state capture"""
        self.is_capturing = False

        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)

        self.capture_thread = None

    def capture_single_state(self) -> VisualizationNetworkState:
        """Capture a single network state snapshot"""
        start_time = time.perf_counter()

        # Get current network state
        network_state = self._extract_network_state()

        # Create visualization state
        vis_state = self._convert_to_visualization_state(network_state, start_time)

        # Store in buffer
        with self.capture_lock:
            self.state_buffer.append(vis_state)

            # Generate delta if we have previous state
            if self.config.delta_compression and self.last_state:
                delta = self._generate_state_delta(self.last_state, vis_state)
                self.delta_buffer.append(delta)

                # Notify delta callbacks
                for callback in self.delta_callbacks:
                    try:
                        callback(delta)
                    except Exception as e:
                        print(f"Delta callback error: {e}")

            self.last_state = vis_state
            self.frame_counter += 1

        # Notify state callbacks
        for callback in self.state_callbacks:
            try:
                callback(vis_state)
            except Exception as e:
                print(f"State callback error: {e}")

        return vis_state

    def get_current_state(self) -> Optional[VisualizationNetworkState]:
        """Get most recent captured state"""
        with self.capture_lock:
            return self.state_buffer.get_most_recent() if not self.state_buffer.is_empty() else None

    def get_state_history(self, duration_seconds: float) -> List[VisualizationNetworkState]:
        """Get state history for specified duration"""
        current_time = time.time()
        cutoff_time = current_time - duration_seconds

        with self.capture_lock:
            states = []
            for state in self.state_buffer.get_all():
                if state.timestamp >= cutoff_time:
                    states.append(state)
            return states

    def register_state_callback(self, callback: Callable[[VisualizationNetworkState], None]) -> None:
        """Register callback for state updates"""
        self.state_callbacks.append(callback)

    def register_delta_callback(self, callback: Callable[[StateDelta], None]) -> None:
        """Register callback for delta updates"""
        self.delta_callbacks.append(callback)

    def set_layout_manager(self, layout_manager: Any) -> None:
        """Set layout manager for position mapping"""
        self.layout_manager = layout_manager

    def _capture_loop(self) -> None:
        """Main capture loop for async operation"""
        target_interval = 1.0 / self.config.capture_rate_hz

        while self.is_capturing:
            loop_start = time.perf_counter()

            try:
                self.capture_single_state()
            except Exception as e:
                print(f"Capture error: {e}")

            # Maintain capture rate
            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0.0, target_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _extract_network_state(self) -> NetworkState:
        """Extract raw network state from Neural-Klotski network"""
        # This will interface with the actual network state
        # For now, we'll create a basic extraction method

        # Get current network state (this would be the actual network interface)
        current_time = self.network.current_time if hasattr(self.network, 'current_time') else time.time()

        # Extract block states
        blocks = {}
        if hasattr(self.network, 'blocks'):
            blocks = self.network.blocks.copy()

        # Extract wire states
        wires = []
        if hasattr(self.network, 'wires'):
            wires = self.network.wires.copy()

        # Extract signal queue
        signal_queue = None
        if hasattr(self.network, 'signal_queue'):
            signal_queue = self.network.signal_queue

        # Create NetworkState (simplified for now)
        return NetworkState(
            current_time=current_time,
            blocks=blocks,
            wires=wires,
            signal_queue=signal_queue,
            dye_concentrations={},
            dye_system=getattr(self.network, 'dye_system', None),
            plasticity_stats=getattr(self.network, 'plasticity_stats', None),
            learning_stats=getattr(self.network, 'learning_stats', None)
        )

    def _convert_to_visualization_state(self, network_state: NetworkState, capture_start: float) -> VisualizationNetworkState:
        """Convert raw network state to visualization-optimized format"""
        capture_time = time.time()

        # Extract block information
        block_count = len(network_state.blocks)
        block_positions = np.zeros((block_count, 2))
        block_velocities = np.zeros(block_count)
        block_colors = []
        block_thresholds = np.zeros(block_count)
        block_refractory = np.zeros(block_count)
        firing_events = set()

        # Process blocks
        for i, (block_id, block_state) in enumerate(network_state.blocks.items()):
            # Get spatial position (use layout manager if available)
            if self.layout_manager and hasattr(self.layout_manager, 'get_block_position'):
                pos = self.layout_manager.get_block_position(block_id)
                block_positions[i] = pos
            else:
                # Use block's activation and lag positions
                block_positions[i] = [block_state.position, block_state.lag_position]

            block_velocities[i] = block_state.velocity
            block_colors.append(block_state.color)
            block_thresholds[i] = block_state.threshold
            block_refractory[i] = block_state.refractory_timer

            # Check for firing (simplified detection)
            if block_state.refractory_timer > 0 or block_state.position > block_state.threshold:
                firing_events.add(block_id)

        # Extract wire information
        wire_count = len(network_state.wires)
        wire_strengths = np.zeros(wire_count)
        wire_base_strengths = np.zeros(wire_count)
        dye_enhancements = np.ones(wire_count)
        active_wires = set()

        for i, wire in enumerate(network_state.wires):
            wire_strengths[i] = getattr(wire, 'effective_strength', wire.strength)
            wire_base_strengths[i] = wire.strength
            # dye_enhancements[i] = calculate from dye system

            # Check if wire has active signals
            if hasattr(wire, 'has_signals') and wire.has_signals():
                active_wires.add(i)

        # Extract active signals
        active_signals = self._extract_signals(network_state, block_positions)

        # Extract dye fields (simplified)
        dye_resolution = self.config.dye_grid_resolution
        red_dye_field = np.zeros((dye_resolution, dye_resolution))
        blue_dye_field = np.zeros((dye_resolution, dye_resolution))
        yellow_dye_field = np.zeros((dye_resolution, dye_resolution))
        dye_grid_bounds = (-100.0, 0.0, 100.0, 100.0)  # Default bounds

        if network_state.dye_system and self.config.capture_dye_fields:
            # Extract dye concentrations (would interface with actual dye system)
            pass

        # Calculate learning metrics
        learning_rate = 0.0
        plasticity_activity = 0.0
        convergence_score = 0.0
        network_energy = float(np.sum(block_velocities**2))

        if self.config.capture_learning_metrics:
            if network_state.learning_stats:
                learning_rate = network_state.learning_stats.get('learning_rate', 0.0)
                plasticity_activity = network_state.learning_stats.get('plasticity_activity', 0.0)

        capture_duration = (time.perf_counter() - capture_start) * 1000  # ms

        return VisualizationNetworkState(
            timestamp=capture_time,
            block_positions=block_positions,
            block_velocities=block_velocities,
            block_colors=block_colors,
            block_thresholds=block_thresholds,
            block_refractory=block_refractory,
            firing_events=firing_events,
            wire_strengths=wire_strengths,
            wire_base_strengths=wire_base_strengths,
            dye_enhancements=dye_enhancements,
            active_wires=active_wires,
            active_signals=active_signals,
            signal_queue_size=len(active_signals),
            red_dye_field=red_dye_field,
            blue_dye_field=blue_dye_field,
            yellow_dye_field=yellow_dye_field,
            dye_grid_bounds=dye_grid_bounds,
            learning_rate=learning_rate,
            plasticity_activity=plasticity_activity,
            convergence_score=convergence_score,
            network_energy=network_energy,
            capture_duration_ms=capture_duration,
            frame_id=self.frame_counter
        )

    def _extract_signals(self, network_state: NetworkState, block_positions: np.ndarray) -> List[VisualizationSignal]:
        """Extract and convert signals for visualization"""
        signals = []
        signal_id = 0

        if network_state.signal_queue and hasattr(network_state.signal_queue, '_queue'):
            for signal in network_state.signal_queue._queue:
                # Create visualization signal
                source_pos = block_positions[signal.source_id] if signal.source_id < len(block_positions) else (0, 0)
                target_pos = block_positions[signal.target_id] if signal.target_id < len(block_positions) else (0, 0)

                vis_signal = VisualizationSignal(
                    signal_id=signal_id,
                    source_block_id=signal.source_id,
                    target_block_id=signal.target_id,
                    strength=signal.strength,
                    color=signal.color,
                    start_position=tuple(source_pos),
                    end_position=tuple(target_pos),
                    current_position=tuple(source_pos),
                    creation_time=signal.creation_time,
                    arrival_time=signal.arrival_time,
                    progress=0.0
                )

                # Update progress
                vis_signal.update_progress(network_state.current_time)
                signals.append(vis_signal)
                signal_id += 1

                # Limit number of signals for performance
                if len(signals) >= self.config.max_signals_per_frame:
                    break

        return signals

    def _generate_state_delta(self, old_state: VisualizationNetworkState, new_state: VisualizationNetworkState) -> StateDelta:
        """Generate compressed delta between two states"""
        delta = StateDelta(
            timestamp=new_state.timestamp,
            frame_id=new_state.frame_id,
            changed_blocks={},
            firing_changes=set(),
            refractory_changes={},
            strength_changes={},
            new_signals=[],
            completed_signals=set(),
            dye_changes={}
        )

        # Find changed blocks
        threshold = self.config.change_threshold
        for i in range(len(new_state.block_positions)):
            old_pos = old_state.block_positions[i] if i < len(old_state.block_positions) else np.array([0, 0])
            new_pos = new_state.block_positions[i]
            old_vel = old_state.block_velocities[i] if i < len(old_state.block_velocities) else 0
            new_vel = new_state.block_velocities[i]
            old_thresh = old_state.block_thresholds[i] if i < len(old_state.block_thresholds) else 0
            new_thresh = new_state.block_thresholds[i]

            # Check for significant changes
            pos_changed = np.linalg.norm(new_pos - old_pos) > threshold
            vel_changed = abs(new_vel - old_vel) > threshold
            thresh_changed = abs(new_thresh - old_thresh) > threshold

            if pos_changed or vel_changed or thresh_changed:
                delta.changed_blocks[i] = (float(new_pos[0]), float(new_vel), float(new_thresh))

        # Find firing changes
        delta.firing_changes = new_state.firing_events.symmetric_difference(old_state.firing_events)

        # Find refractory changes
        for i in range(len(new_state.block_refractory)):
            old_refrac = old_state.block_refractory[i] if i < len(old_state.block_refractory) else 0
            new_refrac = new_state.block_refractory[i]
            if abs(new_refrac - old_refrac) > threshold:
                delta.refractory_changes[i] = float(new_refrac)

        # Find new signals (simplified)
        delta.new_signals = [s for s in new_state.active_signals if s.progress < 0.1]

        # Learning changes
        if abs(new_state.learning_rate - old_state.learning_rate) > threshold:
            delta.learning_rate_change = new_state.learning_rate

        if abs(new_state.plasticity_activity - old_state.plasticity_activity) > threshold:
            delta.plasticity_change = new_state.plasticity_activity

        return delta

    def get_capture_stats(self) -> Dict[str, Any]:
        """Get capture performance statistics"""
        return {
            'is_capturing': self.is_capturing,
            'frames_captured': self.frame_counter,
            'buffer_utilization': self.state_buffer.size() / self.config.buffer_size,
            'capture_rate_hz': self.config.capture_rate_hz,
            'delta_compression': self.config.delta_compression,
            'performance_stats': self.performance_monitor.get_performance_stats().__dict__
        }