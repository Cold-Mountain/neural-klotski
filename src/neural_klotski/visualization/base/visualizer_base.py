"""
Base Visualizer Class

Abstract base class for all Neural-Klotski visualizers. Defines the core
interface and shared functionality that all visualizer implementations
must follow.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Callable, List
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from neural_klotski.visualization.config import VisualizationConfig, VisualizationTheme
from neural_klotski.visualization.utils import PerformanceUtils
from neural_klotski.core.network import NeuralKlotskiNetwork


class VisualizerBase(ABC):
    """
    Abstract base class for all Neural-Klotski visualizers.

    Provides the core interface and shared functionality that all visualizer
    implementations must implement, including initialization, update loop,
    rendering, and cleanup.
    """

    def __init__(self,
                 network: NeuralKlotskiNetwork,
                 config: Optional[VisualizationConfig] = None,
                 theme: Optional[VisualizationTheme] = None):
        """
        Initialize base visualizer.

        Args:
            network: Neural-Klotski network to visualize
            config: Visualization configuration (uses default if None)
            theme: Visual theme (uses default if None)
        """
        self.network = network
        self.config = config or VisualizationConfig()
        self.theme = theme or VisualizationTheme()

        # Performance monitoring
        self.performance_monitor = PerformanceUtils()

        # State management
        self.is_initialized = False
        self.is_running = False
        self.is_paused = False

        # Component registry
        self.components: Dict[str, 'ComponentBase'] = {}
        self.update_callbacks: List[Callable[[float], None]] = []
        self.render_callbacks: List[Callable[[], None]] = []

        # Timing
        self.current_time = 0.0
        self.last_update_time = 0.0
        self.target_frame_time = 1.0 / self.config.target_fps

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the visualizer.

        This method must be implemented by subclasses to handle platform-specific
        initialization, create rendering contexts, and set up the visualization
        environment.
        """
        pass

    @abstractmethod
    def start_visualization(self) -> None:
        """
        Start the visualization main loop.

        This method must be implemented by subclasses to begin the main
        visualization loop, handle events, and manage the update/render cycle.
        """
        pass

    @abstractmethod
    def stop_visualization(self) -> None:
        """
        Stop the visualization and clean up resources.

        This method must be implemented by subclasses to properly shut down
        the visualization, clean up resources, and exit gracefully.
        """
        pass

    @abstractmethod
    def update(self, dt: float) -> None:
        """
        Update visualization state.

        Args:
            dt: Delta time since last update in seconds
        """
        pass

    @abstractmethod
    def render(self) -> None:
        """
        Render the current visualization frame.

        This method must be implemented by subclasses to handle the actual
        rendering of all visualization components.
        """
        pass

    def cleanup(self) -> None:
        """
        Clean up resources used by the visualizer.

        Base implementation handles common cleanup tasks. Subclasses should
        call super().cleanup() and add their own cleanup logic.
        """
        # Clean up all components
        for component in self.components.values():
            if hasattr(component, 'cleanup'):
                component.cleanup()

        self.components.clear()
        self.update_callbacks.clear()
        self.render_callbacks.clear()

        self.is_initialized = False
        self.is_running = False

    # Component management
    def add_component(self, name: str, component: 'ComponentBase') -> None:
        """Add a visualization component"""
        self.components[name] = component
        if hasattr(component, 'initialize'):
            component.initialize()

    def remove_component(self, name: str) -> Optional['ComponentBase']:
        """Remove a visualization component"""
        if name in self.components:
            component = self.components.pop(name)
            if hasattr(component, 'cleanup'):
                component.cleanup()
            return component
        return None

    def get_component(self, name: str) -> Optional['ComponentBase']:
        """Get a visualization component by name"""
        return self.components.get(name)

    def has_component(self, name: str) -> bool:
        """Check if component exists"""
        return name in self.components

    # Callback management
    def add_update_callback(self, callback: Callable[[float], None]) -> None:
        """Add callback to be called during update phase"""
        self.update_callbacks.append(callback)

    def add_render_callback(self, callback: Callable[[], None]) -> None:
        """Add callback to be called during render phase"""
        self.render_callbacks.append(callback)

    def remove_update_callback(self, callback: Callable[[float], None]) -> None:
        """Remove update callback"""
        if callback in self.update_callbacks:
            self.update_callbacks.remove(callback)

    def remove_render_callback(self, callback: Callable[[], None]) -> None:
        """Remove render callback"""
        if callback in self.render_callbacks:
            self.render_callbacks.remove(callback)

    # Control methods
    def pause(self) -> None:
        """Pause the visualization"""
        self.is_paused = True

    def resume(self) -> None:
        """Resume the visualization"""
        self.is_paused = False

    def toggle_pause(self) -> None:
        """Toggle pause state"""
        self.is_paused = not self.is_paused

    def reset(self) -> None:
        """Reset the visualization to initial state"""
        self.current_time = 0.0
        self.last_update_time = 0.0

        # Reset all components
        for component in self.components.values():
            if hasattr(component, 'reset'):
                component.reset()

    # Configuration management
    def update_config(self, new_config: VisualizationConfig) -> None:
        """Update visualization configuration"""
        if new_config.validate():
            self.config = new_config
            self.target_frame_time = 1.0 / self.config.target_fps

            # Notify components of config change
            for component in self.components.values():
                if hasattr(component, 'on_config_changed'):
                    component.on_config_changed(new_config)

    def update_theme(self, new_theme: VisualizationTheme) -> None:
        """Update visual theme"""
        self.theme = new_theme

        # Notify components of theme change
        for component in self.components.values():
            if hasattr(component, 'on_theme_changed'):
                component.on_theme_changed(new_theme)

    # Performance monitoring
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        stats = self.performance_monitor.get_performance_stats()
        return {
            'fps': stats.fps,
            'frame_time_ms': stats.frame_time_ms,
            'memory_usage_mb': stats.memory_usage_mb,
            'cpu_percent': stats.cpu_percent,
            'component_count': len(self.components),
            'is_running': self.is_running,
            'is_paused': self.is_paused
        }

    # Utility methods
    def should_skip_frame(self) -> bool:
        """Determine if current frame should be skipped for performance"""
        if not self.config.adaptive_quality:
            return False

        return self.performance_monitor.should_skip_frame(
            self.config.target_fps,
            self.config.frame_skip_threshold / 100.0
        )

    def get_adaptive_quality_factor(self) -> float:
        """Get quality factor for adaptive rendering"""
        if not self.config.adaptive_quality:
            return 1.0

        return self.performance_monitor.adaptive_quality_factor(self.config.target_fps)

    # Base update implementation
    def _base_update(self, dt: float) -> None:
        """Base update logic called before subclass update"""
        self.current_time += dt

        # Update all components
        for component in self.components.values():
            if hasattr(component, 'update') and (hasattr(component, 'visible') and component.visible):
                component.update(dt)

        # Call update callbacks
        for callback in self.update_callbacks:
            try:
                callback(dt)
            except Exception as e:
                if self.config.debug_mode:
                    print(f"Update callback error: {e}")

    # Base render implementation
    def _base_render(self) -> None:
        """Base render logic called before subclass render"""
        # Render all visible components
        for component in self.components.values():
            if hasattr(component, 'render') and (hasattr(component, 'visible') and component.visible):
                component.render()

        # Call render callbacks
        for callback in self.render_callbacks:
            try:
                callback()
            except Exception as e:
                if self.config.debug_mode:
                    print(f"Render callback error: {e}")

    # Export functionality
    @abstractmethod
    def capture_screenshot(self, filename: str) -> None:
        """Capture screenshot of current visualization"""
        pass

    @abstractmethod
    def start_recording(self, filename: str) -> None:
        """Start recording animation"""
        pass

    @abstractmethod
    def stop_recording(self) -> None:
        """Stop recording animation"""
        pass

    # Event handling (to be extended by subclasses)
    def on_mouse_click(self, x: int, y: int, button: str) -> None:
        """Handle mouse click events"""
        pass

    def on_mouse_move(self, x: int, y: int) -> None:
        """Handle mouse move events"""
        pass

    def on_key_press(self, key: str) -> None:
        """Handle key press events"""
        # Default keyboard shortcuts
        if self.config.keyboard_shortcuts:
            if key == 'space':
                self.toggle_pause()
            elif key == 'r':
                self.reset()
            elif key == 'q' or key == 'escape':
                self.stop_visualization()

    def on_window_resize(self, width: int, height: int) -> None:
        """Handle window resize events"""
        self.config.window_width = width
        self.config.window_height = height

        # Notify components of resize
        for component in self.components.values():
            if hasattr(component, 'on_window_resize'):
                component.on_window_resize(width, height)

    # Validation
    def validate_state(self) -> List[str]:
        """Validate current visualizer state and return issues"""
        issues = []

        if not self.is_initialized:
            issues.append("Visualizer not initialized")

        if not self.config.validate():
            issues.append("Invalid configuration")

        if self.network is None:
            issues.append("No network to visualize")

        # Validate components
        for name, component in self.components.items():
            if hasattr(component, 'validate'):
                component_issues = component.validate()
                issues.extend([f"{name}: {issue}" for issue in component_issues])

        return issues