# Neural-Klotski Visualization System - API Reference

## 📚 Overview

This document provides comprehensive API documentation for all planned components of the Neural-Klotski Visualization System. This serves as the definitive interface specification for multi-session development.

## 🏗️ Module Structure

```
neural_klotski.visualization/
├── __init__.py                    # Main exports and public API
├── config.py                      # Configuration system
├── utils.py                       # Shared utilities
├── base/                          # Core framework
│   ├── __init__.py
│   ├── visualizer_base.py         # Base visualizer class
│   ├── component_base.py          # Base component class
│   └── renderer_base.py           # Base renderer class
├── data/                          # Data pipeline
│   ├── __init__.py
│   ├── network_state_capture.py   # Real-time state extraction
│   ├── performance_logger.py      # Training metrics capture
│   └── data_buffer.py             # Data buffering system
├── rendering/                     # 2D coordinate and spatial
│   ├── __init__.py
│   ├── coordinate_system.py       # 2D activation×lag space
│   ├── layout_manager.py          # Shelf positioning
│   └── spatial_utils.py           # Coordinate transformations
├── components/                    # Visual components
│   ├── __init__.py
│   ├── block_renderer.py          # Block visualization
│   ├── wire_renderer.py           # Wire connections
│   ├── connectivity_graph.py      # Network topology
│   └── block_states.py            # State-based rendering
├── animation/                     # Animation engine
│   ├── __init__.py
│   ├── animator.py                # Core animation engine
│   ├── interpolation.py           # Movement interpolation
│   ├── frame_manager.py           # Frame rate management
│   └── timeline.py                # Animation sequencing
├── physics/                       # Physics visualization
│   ├── __init__.py
│   ├── motion_renderer.py         # Position/velocity display
│   ├── force_vectors.py           # Force visualization
│   └── spring_dynamics.py         # Spring-damper rendering
├── signals/                       # Signal propagation
│   ├── __init__.py
│   ├── signal_animator.py         # Signal animation
│   ├── propagation_paths.py       # Signal routing
│   ├── temporal_delays.py         # Delay visualization
│   └── signal_queue_display.py    # Queue visualization
├── neural/                        # Neural mechanics
│   ├── __init__.py
│   ├── threshold_display.py       # Threshold indicators
│   ├── firing_animation.py        # Firing events
│   ├── refractory_states.py       # Refractory periods
│   └── activation_patterns.py     # Network activation
├── dye/                          # Dye system
│   ├── __init__.py
│   ├── concentration_maps.py      # 2D concentration fields
│   ├── diffusion_animation.py     # Diffusion animation
│   ├── enhancement_display.py     # Plasticity enhancement
│   └── color_mixing.py            # Multi-color dye
├── training/                      # Training visualization
│   ├── __init__.py
│   ├── progress_dashboard.py      # Training metrics
│   ├── phase_transitions.py       # Curriculum learning
│   ├── convergence_plots.py       # Convergence analysis
│   └── performance_charts.py      # Multi-metric charts
├── plasticity/                    # Learning visualization
│   ├── __init__.py
│   ├── weight_changes.py          # Wire strength evolution
│   ├── learning_heatmaps.py       # Learning activity
│   ├── adaptation_tracking.py     # Threshold adaptation
│   └── hebbian_display.py         # Hebbian learning
├── problems/                      # Problem visualization
│   ├── __init__.py
│   ├── addition_display.py        # Addition problems
│   ├── input_encoding.py          # Binary input display
│   ├── output_decoding.py         # Output visualization
│   └── solution_tracking.py       # Solution progress
├── interface/                     # Interactive controls
│   ├── __init__.py
│   ├── control_panel.py           # Master controls
│   ├── simulation_controls.py     # Play/pause/step
│   ├── parameter_controls.py      # Parameter adjustment
│   ├── view_controls.py           # Zoom/pan/view
│   ├── window_manager.py          # Multi-window layout
│   ├── layout_templates.py        # Dashboard layouts
│   ├── widget_factory.py          # UI components
│   └── responsive_layout.py       # Adaptive layout
├── export/                        # Export and recording
│   ├── __init__.py
│   ├── animation_recorder.py      # Animation recording
│   ├── data_exporter.py           # Data export
│   ├── screenshot_manager.py      # Screenshot system
│   └── report_generator.py        # Report generation
├── performance/                   # Optimization
│   ├── __init__.py
│   ├── frame_optimization.py      # Frame rate optimization
│   ├── memory_management.py       # Memory optimization
│   └── rendering_cache.py         # Rendering cache
├── styling/                       # Visual themes
│   ├── __init__.py
│   ├── block_themes.py            # Block styling
│   ├── wire_themes.py             # Wire styling
│   └── color_schemes.py           # Color schemes
└── examples/                      # Built-in examples
    ├── __init__.py
    ├── basic_visualization.py     # Basic network display
    ├── training_demo.py            # Training visualization
    └── interactive_demo.py         # Interactive controls
```

## 🎯 Core API Classes

### Main Visualizer Interface

```python
class NeuralKlotskiVisualizer:
    """Main visualization interface for Neural-Klotski networks"""

    def __init__(self,
                 network: NeuralKlotskiNetwork,
                 config: Optional[VisualizationConfig] = None):
        """Initialize visualizer with network and configuration"""

    def start_visualization(self) -> None:
        """Launch interactive visualization window"""

    def start_training_visualization(self,
                                   trainer: AdditionNetworkTrainer) -> None:
        """Launch training visualization with live metrics"""

    def capture_screenshot(self, filename: str) -> None:
        """Capture high-quality screenshot"""

    def start_recording(self, filename: str) -> None:
        """Begin animation recording"""

    def stop_recording(self) -> None:
        """Stop animation recording and save"""

    def set_visualization_speed(self, speed_multiplier: float) -> None:
        """Set visualization speed (0.1x to 10x)"""

    def toggle_component_visibility(self, component: str, visible: bool) -> None:
        """Toggle visibility of visualization components"""

    def get_network_state(self) -> VisualizationNetworkState:
        """Get current network state for external analysis"""
```

### Configuration System

```python
@dataclass
class VisualizationConfig:
    """Complete configuration for visualization system"""

    # Display settings
    window_width: int = 1600
    window_height: int = 1200
    target_fps: float = 60.0
    enable_vsync: bool = True

    # Rendering settings
    block_radius: float = 5.0
    wire_thickness_base: float = 1.0
    wire_thickness_scale: float = 3.0
    signal_size: float = 3.0
    signal_trail_length: int = 10

    # Animation settings
    interpolation_mode: str = "cubic"  # "linear", "cubic", "physics"
    animation_smoothing: float = 0.8
    position_update_rate: float = 60.0

    # Dye visualization
    dye_field_resolution: int = 200
    dye_transparency: float = 0.7
    dye_diffusion_animation: bool = True
    dye_concentration_threshold: float = 0.001

    # Training visualization
    metrics_update_rate: float = 10.0
    history_length: int = 10000
    convergence_window_size: int = 50

    # Performance settings
    enable_rendering_cache: bool = True
    max_memory_usage_mb: float = 500.0
    frame_skip_threshold: float = 30.0

    # Export settings
    export_quality: str = "high"  # "low", "medium", "high"
    animation_format: str = "mp4"  # "mp4", "gif", "avi"
    screenshot_dpi: int = 300

class VisualizationTheme:
    """Visual theme configuration"""

    # Block colors
    red_block_color: Tuple[float, float, float] = (1.0, 0.3, 0.3)
    blue_block_color: Tuple[float, float, float] = (0.3, 0.3, 1.0)
    yellow_block_color: Tuple[float, float, float] = (1.0, 0.9, 0.3)

    # Wire colors
    wire_alpha: float = 0.6
    active_wire_highlight: float = 1.5
    signal_glow_intensity: float = 2.0

    # Dye colors
    red_dye_color: Tuple[float, float, float] = (1.0, 0.2, 0.2)
    blue_dye_color: Tuple[float, float, float] = (0.2, 0.2, 1.0)
    yellow_dye_color: Tuple[float, float, float] = (1.0, 0.8, 0.2)

    # Background
    background_color: Tuple[float, float, float] = (0.1, 0.1, 0.1)
    grid_color: Tuple[float, float, float] = (0.3, 0.3, 0.3)
    grid_alpha: float = 0.3
```

### Data Models

```python
@dataclass
class VisualizationNetworkState:
    """Complete network state for visualization"""
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

    # Signal propagation
    active_signals: List[VisualizationSignal]
    signal_queue_size: int

    # Dye concentrations (2D fields)
    red_dye_field: np.ndarray            # Shape: (grid_height, grid_width)
    blue_dye_field: np.ndarray           # Shape: (grid_height, grid_width)
    yellow_dye_field: np.ndarray         # Shape: (grid_height, grid_width)

    # Learning metrics
    learning_rate: float
    plasticity_activity: float
    convergence_score: float

@dataclass
class VisualizationSignal:
    """Signal for visualization with spatial and temporal information"""
    signal_id: int
    source_block_id: int
    target_block_id: int
    strength: float
    color: BlockColor

    # Spatial information
    start_position: Tuple[float, float]  # (activation, lag)
    end_position: Tuple[float, float]    # (activation, lag)
    current_position: Tuple[float, float]  # Interpolated position

    # Temporal information
    creation_time: float
    arrival_time: float
    progress: float  # 0.0 to 1.0

    # Visualization properties
    visual_size: float
    visual_intensity: float
    trail_positions: List[Tuple[float, float]]

@dataclass
class VisualizationTrainingState:
    """Training state for dashboard visualization"""
    epoch: int
    phase: TrainingPhase

    # Performance metrics
    accuracy_history: List[float]
    error_history: List[float]
    confidence_history: List[float]
    learning_rate_history: List[float]

    # Problem-specific metrics
    current_problem: Optional[AdditionProblem]
    expected_output: Optional[int]
    actual_output: Optional[int]
    problem_confidence: float

    # Convergence analysis
    convergence_state: str  # "improving", "plateau", "converged", etc.
    convergence_confidence: float
    epochs_without_improvement: int

    # Phase progression
    phase_accuracy: float
    phase_progress: float  # 0.0 to 1.0
    ready_for_advancement: bool
```

## 🎨 Component APIs

### Base Classes

```python
class VisualizerBase(ABC):
    """Abstract base class for all visualizers"""

    def __init__(self, config: VisualizationConfig):
        """Initialize with configuration"""

    @abstractmethod
    def initialize(self) -> None:
        """Initialize visualization component"""

    @abstractmethod
    def update(self, dt: float) -> None:
        """Update component with delta time"""

    @abstractmethod
    def render(self, renderer: Renderer) -> None:
        """Render component to display"""

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources"""

class ComponentBase(ABC):
    """Base class for visualization components"""

    def __init__(self, config: VisualizationConfig):
        """Initialize component"""

    @property
    def visible(self) -> bool:
        """Component visibility state"""

    @visible.setter
    def visible(self, value: bool) -> None:
        """Set component visibility"""

    @abstractmethod
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Get component bounding box (x_min, y_min, x_max, y_max)"""

class RendererBase(ABC):
    """Base class for rendering backends"""

    @abstractmethod
    def clear(self) -> None:
        """Clear the display"""

    @abstractmethod
    def draw_circle(self, x: float, y: float, radius: float, color: Tuple[float, float, float]) -> None:
        """Draw a circle"""

    @abstractmethod
    def draw_line(self, x1: float, y1: float, x2: float, y2: float,
                  thickness: float, color: Tuple[float, float, float]) -> None:
        """Draw a line"""

    @abstractmethod
    def draw_text(self, x: float, y: float, text: str, size: int,
                  color: Tuple[float, float, float]) -> None:
        """Draw text"""

    @abstractmethod
    def present(self) -> None:
        """Present the rendered frame"""
```

### Data Pipeline

```python
class NetworkStateCapture:
    """Real-time network state capture system"""

    def __init__(self, network: NeuralKlotskiNetwork, capture_rate: float = 60.0):
        """Initialize state capture for network"""

    def start_capture(self) -> None:
        """Begin continuous state capture"""

    def stop_capture(self) -> None:
        """Stop state capture"""

    def get_current_state(self) -> VisualizationNetworkState:
        """Get most recent network state"""

    def get_state_history(self, duration: float) -> List[VisualizationNetworkState]:
        """Get state history for specified duration"""

    def register_state_callback(self, callback: Callable[[VisualizationNetworkState], None]) -> None:
        """Register callback for state updates"""

class PerformanceLogger:
    """Training metrics capture and logging"""

    def __init__(self, trainer: AdditionNetworkTrainer):
        """Initialize logger for trainer"""

    def start_logging(self) -> None:
        """Begin performance logging"""

    def stop_logging(self) -> None:
        """Stop performance logging"""

    def get_training_state(self) -> VisualizationTrainingState:
        """Get current training state"""

    def get_metrics_history(self) -> Dict[str, List[float]]:
        """Get complete metrics history"""

    def export_training_log(self, filename: str) -> None:
        """Export training log to file"""

class DataBuffer:
    """Efficient circular buffer for time-series data"""

    def __init__(self, max_size: int = 10000):
        """Initialize buffer with maximum size"""

    def append(self, data: Any) -> None:
        """Append data to buffer"""

    def get_recent(self, count: int) -> List[Any]:
        """Get most recent N items"""

    def get_range(self, start_time: float, end_time: float) -> List[Any]:
        """Get items within time range"""

    def clear(self) -> None:
        """Clear all buffered data"""
```

### Rendering Components

```python
class CoordinateSystem:
    """2D coordinate system for activation×lag space"""

    def __init__(self, activation_range: Tuple[float, float],
                 lag_range: Tuple[float, float]):
        """Initialize coordinate system with ranges"""

    def world_to_screen(self, world_pos: Tuple[float, float]) -> Tuple[int, int]:
        """Convert world coordinates to screen pixels"""

    def screen_to_world(self, screen_pos: Tuple[int, int]) -> Tuple[float, float]:
        """Convert screen pixels to world coordinates"""

    def set_zoom(self, zoom_level: float) -> None:
        """Set zoom level (1.0 = default)"""

    def set_pan(self, pan_offset: Tuple[float, float]) -> None:
        """Set pan offset in world coordinates"""

    def fit_bounds(self, bounds: Tuple[float, float, float, float]) -> None:
        """Fit coordinate system to specified bounds"""

class LayoutManager:
    """Manages spatial layout of network components"""

    def __init__(self, network_config: NetworkConfig):
        """Initialize layout from network configuration"""

    def get_block_position(self, block_id: int) -> Tuple[float, float]:
        """Get spatial position for block ID"""

    def get_shelf_bounds(self, shelf_type: ShelfType) -> Tuple[float, float, float, float]:
        """Get bounding box for shelf"""

    def update_block_positions(self, network_state: VisualizationNetworkState) -> None:
        """Update block positions from network state"""

    def get_wire_path(self, source_id: int, target_id: int) -> List[Tuple[float, float]]:
        """Get path points for wire between blocks"""

class BlockRenderer(ComponentBase):
    """Renders individual blocks with state visualization"""

    def __init__(self, config: VisualizationConfig, theme: VisualizationTheme):
        """Initialize block renderer"""

    def render_block(self, block_id: int, state: BlockState,
                    position: Tuple[float, float], renderer: RendererBase) -> None:
        """Render single block with current state"""

    def render_firing_effect(self, position: Tuple[float, float],
                           intensity: float, renderer: RendererBase) -> None:
        """Render firing animation effect"""

    def render_threshold_indicator(self, position: Tuple[float, float],
                                 threshold: float, current_activation: float,
                                 renderer: RendererBase) -> None:
        """Render threshold level indicator"""

    def render_refractory_state(self, position: Tuple[float, float],
                              refractory_time: float, renderer: RendererBase) -> None:
        """Render refractory period visualization"""

class WireRenderer(ComponentBase):
    """Renders wire connections and signal propagation"""

    def __init__(self, config: VisualizationConfig, theme: VisualizationTheme):
        """Initialize wire renderer"""

    def render_wire(self, wire: Wire, start_pos: Tuple[float, float],
                   end_pos: Tuple[float, float], renderer: RendererBase) -> None:
        """Render wire connection"""

    def render_signal(self, signal: VisualizationSignal, renderer: RendererBase) -> None:
        """Render signal on wire"""

    def render_signal_trail(self, signal: VisualizationSignal, renderer: RendererBase) -> None:
        """Render signal movement trail"""

    def render_connectivity_graph(self, network_state: VisualizationNetworkState,
                                renderer: RendererBase) -> None:
        """Render complete connectivity visualization"""
```

### Animation System

```python
class Animator:
    """Core animation engine for smooth real-time updates"""

    def __init__(self, target_fps: float = 60.0):
        """Initialize animator with target frame rate"""

    def start(self) -> None:
        """Start animation loop"""

    def stop(self) -> None:
        """Stop animation loop"""

    def add_component(self, component: ComponentBase) -> None:
        """Add component to animation loop"""

    def remove_component(self, component: ComponentBase) -> None:
        """Remove component from animation loop"""

    def set_speed_multiplier(self, multiplier: float) -> None:
        """Set animation speed multiplier"""

class Interpolator:
    """Smooth interpolation for position and property changes"""

    @staticmethod
    def linear(start: float, end: float, t: float) -> float:
        """Linear interpolation"""

    @staticmethod
    def cubic_spline(points: List[Tuple[float, float]], t: float) -> float:
        """Cubic spline interpolation"""

    @staticmethod
    def physics_based(start: float, end: float, velocity: float,
                     damping: float, dt: float) -> Tuple[float, float]:
        """Physics-based interpolation with velocity"""

class FrameManager:
    """Manages frame timing and rendering optimization"""

    def __init__(self, target_fps: float):
        """Initialize frame manager"""

    def begin_frame(self) -> float:
        """Begin frame timing, return delta time"""

    def end_frame(self) -> None:
        """End frame timing and handle synchronization"""

    def get_current_fps(self) -> float:
        """Get current measured frame rate"""

    def should_skip_frame(self) -> bool:
        """Determine if frame should be skipped for performance"""
```

### Training Visualization

```python
class ProgressDashboard(ComponentBase):
    """Real-time training progress visualization"""

    def __init__(self, config: VisualizationConfig):
        """Initialize progress dashboard"""

    def update_metrics(self, training_state: VisualizationTrainingState) -> None:
        """Update dashboard with latest training metrics"""

    def render_accuracy_plot(self, renderer: RendererBase) -> None:
        """Render accuracy over time plot"""

    def render_phase_progress(self, renderer: RendererBase) -> None:
        """Render current phase progress"""

    def render_convergence_analysis(self, renderer: RendererBase) -> None:
        """Render convergence state analysis"""

class PhaseTransitions(ComponentBase):
    """Visualizes curriculum learning phase transitions"""

    def __init__(self, config: VisualizationConfig):
        """Initialize phase transition visualizer"""

    def render_phase_timeline(self, training_state: VisualizationTrainingState,
                            renderer: RendererBase) -> None:
        """Render training phase timeline"""

    def render_phase_metrics(self, training_state: VisualizationTrainingState,
                           renderer: RendererBase) -> None:
        """Render per-phase performance metrics"""
```

### Interactive Interface

```python
class ControlPanel:
    """Master control interface for visualization"""

    def __init__(self, visualizer: NeuralKlotskiVisualizer):
        """Initialize control panel"""

    def create_simulation_controls(self) -> tk.Frame:
        """Create play/pause/step control widgets"""

    def create_parameter_controls(self) -> tk.Frame:
        """Create parameter adjustment widgets"""

    def create_view_controls(self) -> tk.Frame:
        """Create zoom/pan/view control widgets"""

    def create_export_controls(self) -> tk.Frame:
        """Create export and recording controls"""

class SimulationControls:
    """Play/pause/step simulation controls"""

    def __init__(self, visualizer: NeuralKlotskiVisualizer):
        """Initialize simulation controls"""

    def play(self) -> None:
        """Start simulation playback"""

    def pause(self) -> None:
        """Pause simulation"""

    def step(self) -> None:
        """Execute single simulation step"""

    def reset(self) -> None:
        """Reset simulation to initial state"""

    def set_speed(self, speed_multiplier: float) -> None:
        """Set simulation speed multiplier"""
```

### Export System

```python
class AnimationRecorder:
    """Records visualization as video animations"""

    def __init__(self, output_format: str = "mp4", quality: str = "high"):
        """Initialize animation recorder"""

    def start_recording(self, filename: str, fps: float = 60.0) -> None:
        """Begin recording animation"""

    def capture_frame(self, frame_data: np.ndarray) -> None:
        """Capture single frame for animation"""

    def stop_recording(self) -> None:
        """Stop recording and save animation file"""

class ScreenshotManager:
    """High-quality screenshot capture system"""

    def __init__(self, dpi: int = 300):
        """Initialize screenshot manager"""

    def capture_screenshot(self, filename: str, format: str = "png") -> None:
        """Capture high-quality screenshot"""

    def capture_component(self, component: ComponentBase, filename: str) -> None:
        """Capture screenshot of specific component"""

class DataExporter:
    """Export visualization data and metrics"""

    def __init__(self):
        """Initialize data exporter"""

    def export_network_state(self, state: VisualizationNetworkState,
                           filename: str, format: str = "json") -> None:
        """Export network state data"""

    def export_training_metrics(self, metrics: VisualizationTrainingState,
                               filename: str, format: str = "csv") -> None:
        """Export training metrics data"""

    def export_animation_data(self, state_history: List[VisualizationNetworkState],
                            filename: str) -> None:
        """Export complete animation state data"""
```

## 🚀 Usage Examples

### Basic Network Visualization

```python
from neural_klotski.visualization import NeuralKlotskiVisualizer
from neural_klotski.core.architecture import create_addition_network

# Create network
network = create_addition_network(enable_learning=True)

# Create visualizer with default configuration
visualizer = NeuralKlotskiVisualizer(network)

# Launch visualization
visualizer.start_visualization()
```

### Training Visualization

```python
from neural_klotski.visualization import NeuralKlotskiVisualizer
from neural_klotski.training.trainer import AdditionNetworkTrainer, TrainingConfig

# Create trainer
training_config = TrainingConfig()
trainer = AdditionNetworkTrainer(training_config, sim_config)

# Create visualizer
visualizer = NeuralKlotskiVisualizer(trainer.network)

# Launch training visualization
visualizer.start_training_visualization(trainer)
```

### Custom Configuration

```python
from neural_klotski.visualization import VisualizationConfig, VisualizationTheme

# Custom configuration
config = VisualizationConfig(
    window_width=1920,
    window_height=1080,
    target_fps=60.0,
    block_radius=8.0,
    enable_rendering_cache=True
)

# Custom theme
theme = VisualizationTheme(
    red_block_color=(1.0, 0.2, 0.2),
    blue_block_color=(0.2, 0.2, 1.0),
    background_color=(0.05, 0.05, 0.05)
)

# Create visualizer with custom settings
visualizer = NeuralKlotskiVisualizer(network, config)
visualizer.set_theme(theme)
```

---

This API reference provides the complete interface specifications for implementing the Neural-Klotski Visualization System across multiple development sessions. Each component is designed to be independently implementable while maintaining clear integration points.