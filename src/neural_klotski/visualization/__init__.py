"""
Neural-Klotski Visualization System

A comprehensive, real-time visualization suite for the Neural-Klotski bio-inspired
neural network system. Provides complete insight into network dynamics, learning
processes, and system behavior through interactive, animated displays.

Key Components:
- Real-time network visualization with 79 blocks in 2D space
- Signal propagation animation with temporal delays
- Dye system visualization with diffusion effects
- Training progress dashboard with live metrics
- Interactive controls for simulation management
- Export capabilities for animations and data

Example Usage:
    >>> from neural_klotski.visualization import NeuralKlotskiVisualizer
    >>> from neural_klotski.core.architecture import create_addition_network
    >>>
    >>> network = create_addition_network(enable_learning=True)
    >>> visualizer = NeuralKlotskiVisualizer(network)
    >>> visualizer.start_visualization()
"""

from .config import (
    VisualizationConfig,
    VisualizationTheme,
    get_default_visualization_config,
    get_default_theme
)

from .utils import (
    ColorUtils,
    CoordinateUtils,
    AnimationUtils,
    PerformanceUtils
)

# Main visualizer class will be imported from base when implemented
# from .base import NeuralKlotskiVisualizer

__version__ = "1.0.0"
__author__ = "Neural-Klotski Development Team"

# Public API exports
__all__ = [
    # Configuration
    'VisualizationConfig',
    'VisualizationTheme',
    'get_default_visualization_config',
    'get_default_theme',

    # Utilities
    'ColorUtils',
    'CoordinateUtils',
    'AnimationUtils',
    'PerformanceUtils',

    # Main visualizer (when implemented)
    # 'NeuralKlotskiVisualizer',
]

# Module metadata for introspection
VISUALIZATION_MODULES = {
    'base': 'Core visualization framework',
    'data': 'Real-time data capture pipeline',
    'rendering': '2D coordinate system and spatial rendering',
    'components': 'Block, wire, and UI component renderers',
    'animation': 'Animation engine and interpolation',
    'physics': 'Physics simulation visualization',
    'signals': 'Signal propagation animation',
    'neural': 'Neural firing and threshold visualization',
    'dye': 'Dye system and diffusion visualization',
    'training': 'Training progress and metrics',
    'plasticity': 'Learning and adaptation visualization',
    'problems': 'Addition problem and I/O visualization',
    'interface': 'Interactive controls and dashboard',
    'export': 'Animation recording and data export',
    'performance': 'Optimization and caching',
    'styling': 'Visual themes and color schemes',
    'examples': 'Built-in examples and demos'
}

def get_module_info() -> dict:
    """Get information about visualization module components"""
    return {
        'version': __version__,
        'modules': VISUALIZATION_MODULES,
        'total_modules': len(VISUALIZATION_MODULES),
        'implementation_status': 'Phase 1B - Module Structure'
    }

def list_available_components() -> list:
    """List all planned visualization components"""
    return list(VISUALIZATION_MODULES.keys())

# Development phase tracking
DEVELOPMENT_PHASE = "1B"
DEVELOPMENT_STATUS = "Module Structure - In Progress"

# Performance and compatibility information
PERFORMANCE_TARGETS = {
    'target_fps': 60.0,
    'max_memory_mb': 500.0,
    'startup_time_s': 5.0,
    'response_time_ms': 100.0
}

COMPATIBILITY_INFO = {
    'python_versions': ['3.8', '3.9', '3.10', '3.11', '3.12'],
    'platforms': ['macOS', 'Linux', 'Windows'],
    'gui_framework': 'tkinter',
    'plotting_engine': 'matplotlib'
}