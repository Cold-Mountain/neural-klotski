"""
Visualization Configuration System

Provides comprehensive configuration management for all visualization components
including display settings, performance parameters, visual themes, and
export options.
"""

from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Optional, List
from enum import Enum
import json
import os


class InterpolationMode(Enum):
    """Animation interpolation modes"""
    LINEAR = "linear"
    CUBIC = "cubic"
    PHYSICS = "physics"
    SMOOTH = "smooth"


class ExportFormat(Enum):
    """Export format options"""
    MP4 = "mp4"
    GIF = "gif"
    AVI = "avi"
    PNG = "png"
    JPG = "jpg"
    JSON = "json"
    CSV = "csv"


class ExportQuality(Enum):
    """Export quality settings"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"


@dataclass
class VisualizationConfig:
    """Complete configuration for Neural-Klotski visualization system"""

    # Display settings
    window_width: int = 1600
    window_height: int = 1200
    window_title: str = "Neural-Klotski Visualization"
    target_fps: float = 60.0
    enable_vsync: bool = True
    fullscreen: bool = False

    # Rendering settings
    block_radius: float = 5.0
    wire_thickness_base: float = 1.0
    wire_thickness_scale: float = 3.0
    signal_size: float = 3.0
    signal_trail_length: int = 10
    grid_spacing: float = 10.0
    grid_visible: bool = True

    # Animation settings
    interpolation_mode: InterpolationMode = InterpolationMode.CUBIC
    animation_smoothing: float = 0.8
    position_update_rate: float = 60.0
    enable_motion_blur: bool = False
    trail_fade_rate: float = 0.9

    # Dye visualization
    dye_field_resolution: int = 200
    dye_transparency: float = 0.7
    dye_diffusion_animation: bool = True
    dye_concentration_threshold: float = 0.001
    dye_color_intensity: float = 1.0

    # Training visualization
    metrics_update_rate: float = 10.0
    history_length: int = 10000
    convergence_window_size: int = 50
    show_live_metrics: bool = True
    metrics_smoothing: float = 0.95

    # Performance settings
    enable_rendering_cache: bool = True
    max_memory_usage_mb: float = 500.0
    frame_skip_threshold: float = 30.0
    adaptive_quality: bool = True
    background_processing: bool = True

    # Interactive settings
    mouse_sensitivity: float = 1.0
    zoom_sensitivity: float = 1.2
    pan_sensitivity: float = 1.0
    keyboard_shortcuts: bool = True
    context_menus: bool = True

    # Export settings
    export_quality: ExportQuality = ExportQuality.HIGH
    animation_format: ExportFormat = ExportFormat.MP4
    screenshot_format: ExportFormat = ExportFormat.PNG
    screenshot_dpi: int = 300
    animation_fps: float = 30.0

    # Developer settings
    debug_mode: bool = False
    show_fps: bool = False
    show_memory_usage: bool = False
    profiling_enabled: bool = False
    verbose_logging: bool = False

    def validate(self) -> bool:
        """Validate configuration parameters"""
        checks = [
            self.window_width > 0,
            self.window_height > 0,
            0.1 <= self.target_fps <= 120.0,
            self.block_radius > 0,
            self.wire_thickness_base > 0,
            self.wire_thickness_scale > 0,
            self.signal_size > 0,
            0 <= self.dye_transparency <= 1.0,
            self.dye_field_resolution > 0,
            self.max_memory_usage_mb > 0,
            0 <= self.animation_smoothing <= 1.0,
            self.metrics_update_rate > 0,
            self.history_length > 0
        ]
        return all(checks)

    def save_to_file(self, filename: str) -> None:
        """Save configuration to JSON file"""
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                config_dict[key] = value.value
            else:
                config_dict[key] = value

        with open(filename, 'w') as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load_from_file(cls, filename: str) -> 'VisualizationConfig':
        """Load configuration from JSON file"""
        with open(filename, 'r') as f:
            config_dict = json.load(f)

        # Convert enum values back to enums
        for key, value in config_dict.items():
            if key == 'interpolation_mode':
                config_dict[key] = InterpolationMode(value)
            elif key in ['export_quality']:
                config_dict[key] = ExportQuality(value)
            elif key in ['animation_format', 'screenshot_format']:
                config_dict[key] = ExportFormat(value)

        return cls(**config_dict)


@dataclass
class VisualizationTheme:
    """Visual theme configuration for consistent styling"""

    # Theme metadata
    name: str = "default"
    description: str = "Default Neural-Klotski theme"

    # Block colors (RGB tuples, 0.0-1.0)
    red_block_color: Tuple[float, float, float] = (1.0, 0.3, 0.3)
    blue_block_color: Tuple[float, float, float] = (0.3, 0.3, 1.0)
    yellow_block_color: Tuple[float, float, float] = (1.0, 0.9, 0.3)

    # Block state colors
    firing_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    refractory_color_modifier: float = 0.5  # Dimming factor for refractory blocks
    threshold_indicator_color: Tuple[float, float, float] = (0.8, 0.8, 0.8)

    # Wire colors and properties
    wire_alpha: float = 0.6
    active_wire_highlight: float = 1.5
    signal_glow_intensity: float = 2.0
    wire_color_inherit: bool = True  # Inherit color from source block

    # Dye colors
    red_dye_color: Tuple[float, float, float] = (1.0, 0.2, 0.2)
    blue_dye_color: Tuple[float, float, float] = (0.2, 0.2, 1.0)
    yellow_dye_color: Tuple[float, float, float] = (1.0, 0.8, 0.2)

    # Background and interface
    background_color: Tuple[float, float, float] = (0.1, 0.1, 0.1)
    grid_color: Tuple[float, float, float] = (0.3, 0.3, 0.3)
    grid_alpha: float = 0.3
    text_color: Tuple[float, float, float] = (0.9, 0.9, 0.9)
    ui_accent_color: Tuple[float, float, float] = (0.2, 0.6, 1.0)

    # Training visualization colors
    accuracy_color: Tuple[float, float, float] = (0.2, 0.8, 0.2)
    error_color: Tuple[float, float, float] = (0.8, 0.2, 0.2)
    learning_rate_color: Tuple[float, float, float] = (0.8, 0.6, 0.2)
    convergence_color: Tuple[float, float, float] = (0.6, 0.2, 0.8)

    # Signal trail properties
    signal_trail_alpha_start: float = 1.0
    signal_trail_alpha_end: float = 0.1
    signal_pulse_frequency: float = 2.0  # Hz

    # Animation effects
    firing_flash_duration: float = 0.2  # seconds
    firing_ring_expansion: float = 2.0  # radius multiplier
    hover_highlight_intensity: float = 0.3


def get_default_visualization_config() -> VisualizationConfig:
    """Get default visualization configuration"""
    return VisualizationConfig()


def get_default_theme() -> VisualizationTheme:
    """Get default visualization theme"""
    return VisualizationTheme()


def get_dark_theme() -> VisualizationTheme:
    """Get dark theme for visualization"""
    theme = VisualizationTheme(
        name="dark",
        description="Dark theme with high contrast",
        background_color=(0.05, 0.05, 0.05),
        grid_color=(0.2, 0.2, 0.2),
        text_color=(0.95, 0.95, 0.95)
    )
    return theme


def get_light_theme() -> VisualizationTheme:
    """Get light theme for visualization"""
    theme = VisualizationTheme(
        name="light",
        description="Light theme for presentations",
        background_color=(0.95, 0.95, 0.95),
        grid_color=(0.7, 0.7, 0.7),
        text_color=(0.1, 0.1, 0.1),
        red_block_color=(0.8, 0.2, 0.2),
        blue_block_color=(0.2, 0.2, 0.8),
        yellow_block_color=(0.8, 0.7, 0.1)
    )
    return theme


def get_high_contrast_theme() -> VisualizationTheme:
    """Get high contrast theme for accessibility"""
    theme = VisualizationTheme(
        name="high_contrast",
        description="High contrast theme for accessibility",
        background_color=(0.0, 0.0, 0.0),
        grid_color=(0.5, 0.5, 0.5),
        text_color=(1.0, 1.0, 1.0),
        red_block_color=(1.0, 0.0, 0.0),
        blue_block_color=(0.0, 0.0, 1.0),
        yellow_block_color=(1.0, 1.0, 0.0),
        wire_alpha=0.8,
        signal_glow_intensity=3.0
    )
    return theme


# Theme registry for easy access
AVAILABLE_THEMES = {
    'default': get_default_theme,
    'dark': get_dark_theme,
    'light': get_light_theme,
    'high_contrast': get_high_contrast_theme
}


def get_theme_by_name(name: str) -> VisualizationTheme:
    """Get theme by name from registry"""
    if name not in AVAILABLE_THEMES:
        raise ValueError(f"Unknown theme: {name}. Available themes: {list(AVAILABLE_THEMES.keys())}")
    return AVAILABLE_THEMES[name]()


def list_available_themes() -> List[str]:
    """List all available theme names"""
    return list(AVAILABLE_THEMES.keys())


# Configuration file handling
DEFAULT_CONFIG_PATH = os.path.expanduser("~/.neural_klotski_viz_config.json")


def save_default_config(config: VisualizationConfig, path: str = DEFAULT_CONFIG_PATH) -> None:
    """Save configuration as user default"""
    config.save_to_file(path)


def load_default_config(path: str = DEFAULT_CONFIG_PATH) -> VisualizationConfig:
    """Load user default configuration, fallback to system default"""
    if os.path.exists(path):
        try:
            return VisualizationConfig.load_from_file(path)
        except Exception:
            # Fallback to default if loading fails
            return get_default_visualization_config()
    else:
        return get_default_visualization_config()


# Performance preset configurations
def get_performance_config() -> VisualizationConfig:
    """Get performance-optimized configuration"""
    config = get_default_visualization_config()
    config.target_fps = 30.0
    config.dye_field_resolution = 100
    config.enable_motion_blur = False
    config.adaptive_quality = True
    config.signal_trail_length = 5
    config.metrics_update_rate = 5.0
    config.enable_rendering_cache = True
    return config


def get_quality_config() -> VisualizationConfig:
    """Get quality-optimized configuration"""
    config = get_default_visualization_config()
    config.target_fps = 60.0
    config.dye_field_resolution = 300
    config.enable_motion_blur = True
    config.signal_trail_length = 20
    config.metrics_update_rate = 30.0
    config.screenshot_dpi = 600
    return config


# Configuration validation and migration
def validate_config(config: VisualizationConfig) -> List[str]:
    """Validate configuration and return list of issues"""
    issues = []

    if not config.validate():
        issues.append("Basic validation failed")

    # Check for reasonable values
    if config.target_fps > 120:
        issues.append("Target FPS too high (>120)")

    if config.max_memory_usage_mb > 2000:
        issues.append("Memory limit very high (>2GB)")

    if config.dye_field_resolution > 500:
        issues.append("Dye field resolution very high (may impact performance)")

    return issues