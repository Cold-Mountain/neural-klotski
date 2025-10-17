# Neural-Klotski Visualization System - Technical Specifications

## 📋 System Requirements

### Performance Targets
- **Frame Rate**: 60 FPS for real-time visualization
- **Memory Usage**: <500MB for complete visualization suite
- **Startup Time**: <5 seconds for full interface initialization
- **Response Time**: <100ms for interactive controls
- **CPU Usage**: Single-core sufficient, multi-core advantageous

### Platform Compatibility
- **Primary Platforms**: macOS, Linux, Windows
- **Python Versions**: 3.8, 3.9, 3.10, 3.11, 3.12
- **Display Resolution**: Minimum 1280x720, Optimal 1920x1080+
- **Memory**: Minimum 4GB RAM, Recommended 8GB+

## 🏗️ Architecture Overview

### Core Framework Design

```
Visualization System Architecture:

┌─────────────────────────────────────────────────────────────┐
│                    User Interface Layer                     │
├─────────────────────────────────────────────────────────────┤
│  Control Panel  │  Dashboard  │  Export Tools  │  Settings  │
├─────────────────────────────────────────────────────────────┤
│                   Rendering Engine                          │
├─────────────────────────────────────────────────────────────┤
│ Block Renderer │ Wire Renderer │ Signal Animator │ Dye Maps │
├─────────────────────────────────────────────────────────────┤
│                  Animation Framework                        │
├─────────────────────────────────────────────────────────────┤
│ Interpolation │ Frame Manager │ Timeline │ Performance Cache │
├─────────────────────────────────────────────────────────────┤
│                    Data Pipeline                            │
├─────────────────────────────────────────────────────────────┤
│  State Capture  │  Data Buffer  │  Metrics Logger  │  I/O   │
├─────────────────────────────────────────────────────────────┤
│                Neural-Klotski Core System                   │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack

#### GUI Framework: tkinter
- **Rationale**: Built-in Python library, cross-platform, lightweight
- **Components**: Main window, control panels, dialogs
- **Layout**: Grid-based responsive design with multiple frames
- **Threading**: Separate UI and simulation threads for responsiveness

#### Plotting Engine: matplotlib
- **Backend**: TkAgg for tkinter integration
- **Optimization**: Blitting for efficient real-time updates
- **Features**: 2D plotting, annotations, custom artists
- **Memory Management**: Figure recycling and selective redraw

#### Animation System: Custom Engine
- **Frame Rate**: Locked 60 FPS with adaptive frame skipping
- **Interpolation**: Linear, cubic spline, and physics-based
- **Timeline**: Event scheduling and synchronization
- **Buffering**: Double-buffered rendering with selective updates

## 📊 Data Structures and Models

### Network State Representation

```python
@dataclass
class VisualizationNetworkState:
    """Complete network state for visualization"""
    timestamp: float

    # Block states (79 blocks)
    block_positions: np.ndarray      # Shape: (79, 2) - (activation, lag)
    block_velocities: np.ndarray     # Shape: (79,) - activation axis velocities
    block_colors: List[BlockColor]   # Color identity for each block
    block_thresholds: np.ndarray     # Current threshold values
    block_refractory: np.ndarray     # Refractory timer states
    firing_events: Set[int]          # Block IDs that fired this timestep

    # Wire network (1560 connections)
    wire_strengths: np.ndarray       # Current effective strengths
    wire_base_strengths: np.ndarray  # Base strengths (before dye enhancement)
    dye_enhancements: np.ndarray     # Dye enhancement factors

    # Signal propagation
    active_signals: List[VisualizationSignal]
    signal_queue_size: int

    # Dye concentrations (2D fields)
    red_dye_field: np.ndarray        # Shape: (grid_height, grid_width)
    blue_dye_field: np.ndarray       # Shape: (grid_height, grid_width)
    yellow_dye_field: np.ndarray     # Shape: (grid_height, grid_width)

    # Learning metrics
    learning_rate: float
    plasticity_activity: float
    convergence_score: float
```

### Signal Visualization Model

```python
@dataclass
class VisualizationSignal:
    """Signal for visualization with path information"""
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
    trail_length: int
```

### Training Metrics Model

```python
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

## 🎨 Rendering Specifications

### 2D Coordinate System

#### Physical Space Mapping
- **Activation Axis**: Horizontal (X), range approximately [-100, +100]
- **Lag Axis**: Vertical (Y), range [0, 100] for 4 shelves
- **Screen Mapping**: Linear transformation with zoom/pan support
- **Aspect Ratio**: Preserved 1:1 for spatial accuracy

#### Block Representation
- **Shape**: Circular markers (radius proportional to zoom level)
- **Color**: RGB mapping based on BlockColor enum
  - Red blocks: #FF4444 (excitatory)
  - Blue blocks: #4444FF (inhibitory)
  - Yellow blocks: #FFDD44 (coupling)
- **State Indicators**:
  - Firing: Bright white flash with expanding ring
  - Refractory: Dimmed color with countdown timer
  - Threshold: Adjustable radius indicator

#### Wire Representation
- **Rendering**: Straight lines between block centers
- **Thickness**: Proportional to effective wire strength (1-5 pixels)
- **Color**: Matches source block color with transparency
- **Style Differentiation**:
  - K-nearest neighbors: Solid lines
  - Long-range connections: Dashed lines
  - Active signals: Highlighted with glow effect

### Animation Framework

#### Frame Management
```python
class FrameManager:
    """Manages frame rate and rendering optimization"""
    target_fps: float = 60.0
    adaptive_quality: bool = True
    frame_skip_threshold: float = 50.0  # Skip frames if FPS drops below

    def update_frame(self, dt: float) -> bool:
        """Update frame with delta time, return if should render"""

    def get_interpolation_factor(self) -> float:
        """Get smooth interpolation factor for animations"""
```

#### Interpolation System
- **Position Interpolation**: Cubic spline for smooth block movement
- **Signal Propagation**: Linear interpolation along wire paths
- **Color Transitions**: HSV color space interpolation
- **Dye Diffusion**: Gaussian blur-based smooth field updates

### Dye Visualization

#### 2D Field Representation
- **Grid Resolution**: 200x200 for smooth visualization
- **Color Mapping**:
  - Red dye: Red channel intensity
  - Blue dye: Blue channel intensity
  - Yellow dye: Yellow channel intensity
  - Color mixing: Additive blending with saturation limits
- **Rendering**: Contour plots with alpha blending
- **Update Rate**: 10Hz for smooth diffusion animation

#### Concentration Visualization
- **Method**: Heatmap overlay with adjustable transparency
- **Scale**: Logarithmic scaling for wide dynamic range
- **Threshold**: Minimum concentration for visibility (reduces noise)
- **Animation**: Smooth temporal interpolation of concentration changes

## 🔧 Performance Optimization

### Rendering Optimization

#### Selective Redraw System
```python
class SelectiveRenderer:
    """Optimized rendering with change detection"""

    def mark_dirty(self, component: str, region: Optional[Rect] = None):
        """Mark component/region as needing redraw"""

    def render_frame(self) -> None:
        """Render only changed components"""

    def force_full_redraw(self) -> None:
        """Force complete frame redraw"""
```

#### Rendering Cache
- **Block Positions**: Cache static block appearance, update only positions
- **Wire Network**: Pre-compute wire paths, update only active signals
- **UI Elements**: Cache control panel rendering
- **Dye Fields**: Progressive update with dirty region tracking

### Memory Management

#### Data Buffering Strategy
- **Ring Buffers**: Fixed-size buffers for time-series data
- **Lazy Loading**: Load visualization components on demand
- **Memory Pools**: Reuse signal and animation objects
- **Garbage Collection**: Explicit cleanup of expired data

#### Memory Limits
```python
class MemoryManager:
    """Memory usage control for visualization"""
    max_history_length: int = 10000      # Maximum data points to keep
    max_active_signals: int = 1000       # Maximum simultaneous signals
    dye_field_compression: bool = True    # Compress dye field history

    def check_memory_usage(self) -> float:
        """Return current memory usage percentage"""

    def cleanup_expired_data(self) -> None:
        """Remove old data beyond limits"""
```

## 🎮 Interactive Features

### Control Interface Specifications

#### Main Control Panel
- **Simulation Controls**: Play, Pause, Step, Reset buttons
- **Speed Control**: Slider for 0.1x to 10x speed multiplier
- **View Controls**: Zoom, Pan, Center, Fit-to-window
- **Display Options**: Toggle wire visibility, signal trails, dye overlay

#### Parameter Adjustment Panel
- **Learning Parameters**: Real-time adjustment of learning rates
- **Network Parameters**: Threshold adjustments, force scaling
- **Visualization Parameters**: Color schemes, transparency, animation speed
- **Problem Input**: Custom addition problem specification

#### Information Display Panel
- **Network Statistics**: Block count, wire count, signal count
- **Performance Metrics**: FPS, memory usage, update rate
- **Current State**: Selected block/wire details, problem status
- **Training Progress**: Current epoch, phase, accuracy metrics

### User Interaction Handling

#### Mouse/Keyboard Controls
```python
class InteractionHandler:
    """Handle user input and camera controls"""

    def handle_mouse_click(self, x: int, y: int, button: str) -> None:
        """Handle mouse clicks for selection and interaction"""

    def handle_mouse_drag(self, dx: int, dy: int) -> None:
        """Handle camera panning with mouse drag"""

    def handle_scroll(self, delta: int) -> None:
        """Handle zoom with mouse wheel"""

    def handle_key_press(self, key: str) -> None:
        """Handle keyboard shortcuts"""
```

#### Selection System
- **Block Selection**: Click to select individual blocks for detailed view
- **Wire Selection**: Click near wire to highlight and show properties
- **Region Selection**: Drag to select multiple components
- **Context Menus**: Right-click for context-sensitive options

## 📡 Data Pipeline Architecture

### Real-Time State Capture

#### Network State Extraction
```python
class NetworkStateCapture:
    """Capture network state for visualization"""

    def capture_current_state(self, network: NeuralKlotskiNetwork) -> VisualizationNetworkState:
        """Extract complete network state"""

    def capture_delta_state(self, network: NeuralKlotskiNetwork) -> VisualizationStateDelta:
        """Extract only changed state components"""

    def setup_continuous_capture(self, update_rate: float) -> None:
        """Setup automatic state capture at specified rate"""
```

#### Data Streaming
- **Update Rate**: 60Hz for real-time visualization
- **Compression**: Delta compression for reduced data transfer
- **Buffering**: Producer-consumer pattern with thread-safe queues
- **Filtering**: Configurable filtering for noise reduction

### Training Integration

#### Trainer Instrumentation
```python
class TrainingInstrumentation:
    """Instrument trainer for visualization data collection"""

    def attach_to_trainer(self, trainer: AdditionNetworkTrainer) -> None:
        """Attach data collection hooks to trainer"""

    def collect_epoch_metrics(self, metrics: TrainingMetrics) -> None:
        """Collect end-of-epoch training metrics"""

    def collect_problem_metrics(self, problem: AdditionProblem, result: DecodingResult) -> None:
        """Collect per-problem performance data"""
```

## 🧪 Testing and Validation

### Unit Testing Requirements

#### Component Testing
- **Rendering Components**: Visual regression testing with reference images
- **Animation System**: Frame timing and interpolation accuracy
- **Data Pipeline**: State capture accuracy and performance
- **User Interface**: Control functionality and responsiveness

#### Integration Testing
- **End-to-End**: Complete visualization workflow testing
- **Performance**: Frame rate and memory usage under load
- **Cross-Platform**: Consistent behavior across operating systems
- **Stress Testing**: Large network visualization and long training runs

### Performance Benchmarking

#### Benchmark Scenarios
1. **Static Network**: 79 blocks, no animation (baseline performance)
2. **Active Simulation**: Full dynamics with signal propagation
3. **Training Visualization**: Real-time training with all overlays
4. **Stress Test**: Maximum signals and dye diffusion activity

#### Metrics Collection
```python
class PerformanceBenchmark:
    """Performance monitoring and benchmarking"""

    def start_benchmark(self, scenario: str) -> None:
        """Begin performance measurement"""

    def record_frame_metrics(self, render_time: float, frame_size: int) -> None:
        """Record per-frame performance data"""

    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance analysis"""
```

## 🔐 Security and Safety

### Input Validation
- **Parameter Bounds**: Validate all user-provided parameters
- **File I/O**: Secure handling of export files and configurations
- **Memory Safety**: Bounds checking for all array operations
- **Error Handling**: Graceful degradation on invalid inputs

### Resource Limits
- **Memory Caps**: Hard limits on memory usage with cleanup
- **CPU Throttling**: Prevent excessive CPU usage in background
- **File Size Limits**: Maximum sizes for exported animations
- **Network Timeouts**: Timeouts for any network operations

## 🚀 Deployment and Installation

### Distribution Strategy
- **Self-Contained**: No additional installation beyond Neural-Klotski
- **Dependencies**: Use only standard library and existing requirements
- **Configuration**: Portable configuration files
- **Documentation**: Comprehensive user and developer guides

### Platform-Specific Considerations

#### macOS
- **Retina Display**: High-DPI scaling support
- **App Bundle**: Optional .app packaging for distribution
- **Metal Backend**: Hardware acceleration where available

#### Linux
- **Display Managers**: X11 and Wayland compatibility
- **Package Managers**: Integration with pip and conda
- **Dependencies**: Handle tkinter availability variations

#### Windows
- **DPI Awareness**: Windows scaling factor support
- **Executable**: Optional .exe packaging with PyInstaller
- **Path Handling**: Windows path separator compatibility

---

This technical specification provides the detailed requirements and architecture for implementing the Neural-Klotski Visualization System across multiple development sessions.